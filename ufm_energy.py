"""UFM (Unsupervised FixMatch) training for Energy with test-time adaptation.

Based on energy_train_reverse.py for data/model loading and learn_ufm.py for UFM loss.
Results are saved to checkpoints_tta directory.
"""

import os
import time
import json
import logging
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import torch
import torchvision
from typing import Optional

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import (
    ALL_DATASETS,
    get_energy_finetuned_path,
    get_finetuned_path,
    get_zeroshot_path,
)
from src.datasets import get_dataloader, maybe_dictionarize, get_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.sigma_param import SigmaParametrization
from src.models.task_vectors import NonLinearTaskVector
from torch.nn.utils.stateless import functional_call
from src.utils.utils import cosine_lr, load_checkpoint_safe
from src.datasets.remote_sensing import sample_k_shot_indices
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from torch.cuda.amp import GradScaler


def setup_simple_logger(name: str = __name__) -> logging.Logger:
    """Setup a clean logger with minimal formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


# Dataset-specific epochs for UFM-Energy training (general datasets)
UFM_SIGMA_EPOCHS_PER_DATASET = {
    # "Cars": 20,
    "DTD": 20,
    # "EuroSAT": 20,
    "GTSRB": 20,
    "MNIST": 20,
    # "RESISC45": 20,
    # "SUN397": 20,
    "SVHN": 20,
    "CIFAR10": 20,
    "CIFAR100": 20,
    "STL10": 20,
    "Food101": 20,
    "Flowers102": 20,
    # "FER2013": 20,
    "PCAM": 20,
    "OxfordIIITPet": 20,
    "RenderedSST2": 20,
    "EMNIST": 20,
    "FashionMNIST": 20,
    # "KMNIST": 20,
    "FGVCAircraft": 20,
    "CUB200": 20,
    "Country211": 20,
}


def ssl_loss_trusted(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    targets: torch.FloatTensor = None,
    trusted: torch.BoolTensor = None,
    thresh: float = 0.99
) -> torch.Tensor:
    """
    UFM (Unsupervised FixMatch) loss with trusted samples.
    
    Args:
        logits1: logits from weak augmentation
        logits2: logits from strong augmentation
        targets: ground-truth pseudo-labels (for trusted samples)
        trusted: boolean mask for trusted samples
        thresh: confidence threshold for pseudo-labeling
    """
    one_hot = logits1.softmax(1).detach()
    
    guessed_targets = one_hot * 0.5  # temperature sharpening
    guessed_targets = guessed_targets / guessed_targets.sum(dim=1, keepdim=True)
    
    if trusted is not None:
        guessed_targets[trusted] = targets[trusted].to(guessed_targets)
    
    one_hot = guessed_targets.detach()
    one_hot = torch.nn.functional.one_hot(
        torch.argmax(guessed_targets, dim=1), 
        num_classes=one_hot.shape[1]
    ).float()
    
    w, _ = guessed_targets.max(1)
    w = (w > thresh).to(logits2)
    
    if trusted is not None:
        trusted = trusted.to(logits1)
    
    loss = (torch.nn.functional.cross_entropy(
        logits2, guessed_targets, reduction='none') * w).sum()
    
    return loss / w.sum() if w.sum() > 0 else loss


class IndexWrapper(torch.utils.data.Dataset):
    """Wrapper that returns index along with data."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if isinstance(data, dict):
            data["index"] = idx
            return data
        else:
            # Assume (image, label) tuple
            return {"images": data[0], "labels": data[1], "index": idx}
    
    def __len__(self):
        return len(self.dataset)


class TwoStreamBatchSampler(torch.utils.data.Sampler):
    """Sample unlabeled and labeled (trusted) data in each batch."""
    def __init__(self, unlabeled_indices, labeled_indices, batch_size):
        self.unlabeled_indices = unlabeled_indices
        self.labeled_indices = labeled_indices
        self.batch_size = batch_size
        
        # Compute batch counts
        self.n_unlabeled = len(unlabeled_indices)
        self.n_labeled = len(labeled_indices)
        self.n_batches = max(self.n_unlabeled, self.n_labeled) // batch_size
    
    def __iter__(self):
        unlabeled_iter = iter(torch.randperm(self.n_unlabeled))
        labeled_iter = iter(torch.randperm(self.n_labeled))
        
        for _ in range(self.n_batches):
            batch = []
            
            # Sample from unlabeled
            for _ in range(self.batch_size // 2):
                try:
                    idx = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(torch.randperm(self.n_unlabeled))
                    idx = next(unlabeled_iter)
                batch.append(self.unlabeled_indices[idx])
            
            # Sample from labeled
            for _ in range(self.batch_size // 2):
                try:
                    idx = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(torch.randperm(self.n_labeled))
                    idx = next(labeled_iter)
                batch.append(self.labeled_indices[idx])
            
            yield batch
    
    def __len__(self):
        return self.n_batches


class TwoAsymetricTransform:
    """Apply two different transforms (weak and strong augmentation)."""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __call__(self, x):
        weak = self.weak_transform(x)
        strong = self.strong_transform(x)
        return {"images": weak, "images_": strong}


def get_preds(dataset, model, classification_head, device, base_params, base_buffers, sigma_modules, sigma_key_map):
    """Get predictions for trusted sample selection."""
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4
    )
    
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            
            # Build params with sigma deltas
            params_map = {}
            for name, p in base_params.items():
                params_map[name] = p.clone()
            
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in params_map and module.sigma.numel() > 0:
                    delta = module().to(params_map[orig_key].device)
                    if params_map[orig_key].shape == delta.shape:
                        params_map[orig_key] = params_map[orig_key] + delta
            
            # Functional forward
            def encoder_forward(mod, x):
                merged = {}
                merged.update(base_buffers)
                merged.update(params_map)
                return functional_call(mod, merged, (x,))
            
            features = encoder_forward(model, inputs)
            logits = classification_head(features)
            all_preds.append(logits.softmax(dim=1).cpu())
    
    return torch.cat(all_preds, dim=0)


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_ufm_energy_config_tag(cfg) -> str:
    num_tasks_minus_one = _sanitize_value(len(cfg.DATASETS_ALL) - 1)
    lr_part = _sanitize_value(cfg.sigma_lr)
    svd_part = _sanitize_value(getattr(cfg, "svd_keep_topk", 2))
    init_mode_part = _sanitize_value(getattr(cfg, "initialize_sigma", "average"))
    warmup_ratio_part = _sanitize_value(getattr(cfg, "warmup_ratio", 0.1))
    wd_part = _sanitize_value(getattr(cfg, "sigma_wd", 0.0))
    return f"ufm_energy_{num_tasks_minus_one}_{lr_part}_{svd_part}_{init_mode_part}_{warmup_ratio_part}_{wd_part}"


def save_k_shot_indices(indices, save_dir, dataset_name, k, seed):
    """Save k-shot indices to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    with open(indices_path, 'w') as f:
        json.dump({"indices": indices, "dataset": dataset_name, "k": k, "seed": seed}, f)
    return indices_path


def load_k_shot_indices(save_dir, k, seed):
    """Load k-shot indices from a JSON file if it exists."""
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    if os.path.exists(indices_path):
        with open(indices_path, 'r') as f:
            data = json.load(f)
            return data["indices"]
    return None


def compute_and_sum_svd_mem_reduction(task_vectors, config, sigma_reduce: str = "mean"):
    """Compute SVD reduction (from energy_train_reverse.py)."""
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 3)))
    sigma_reduce = str(sigma_reduce).lower()
    
    with torch.no_grad():
        new_vector = {}
        
        def is_matrix_key(tv0, key):
            return (
                tv0.vector[key].ndim == 2 and
                all(t not in key for t in ("text_projection", "positional", "token_embedding"))
            )
        
        tv0 = task_vectors[0]
        for key in tv0.vector:
            if not is_matrix_key(tv0, key):
                avg = None
                for i, tv in enumerate(task_vectors):
                    vec = tv.vector[key].to(device)
                    avg = vec.clone() if i == 0 else avg + (vec - avg) / (i + 1)
                new_vector[key] = avg
                continue
            
            vec0 = task_vectors[0].vector[key].to(device)
            u0, s0, vh0 = torch.linalg.svd(vec0, full_matrices=False)
            m = int(u0.shape[0])
            r = int(s0.shape[0])
            n = int(vh0.shape[1])
            
            if r == 0:
                new_vector[key] = torch.zeros_like(vec0)
                continue
            
            num_used = min(num_tasks, r)
            max_per_task = max(1, r // num_used)
            k = min(desired_k, max_per_task)
            chunks = int(k * num_used)
            
            sum_u = torch.zeros((m, chunks), device=device, dtype=u0.dtype)
            sum_v = torch.zeros((chunks, n), device=device, dtype=vh0.dtype)
            
            for i, tv in enumerate(task_vectors[:num_used]):
                vec = tv.vector[key].to(device)
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                r_i = int(s.shape[0])
                k_i = min(k, r_i)
                start = i * k
                end = start + k_i
                sum_u[:, start:end] = u[:, :k_i]
                sum_v[start:end, :] = vh[:k_i, :]
            
            u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_orth = u_u @ vh_u
            V_orth = u_v @ vh_v
            
            all_sigma_diags = []
            U_orth_T = U_orth.T
            V_orth_T = V_orth.T
            
            for tv in task_vectors:
                M_i = tv.vector[key].to(device)
                Sigma_i_prime = (U_orth_T @ M_i) @ V_orth_T
                sigma_task_diag = torch.diag(Sigma_i_prime)
                all_sigma_diags.append(sigma_task_diag)
            
            if not all_sigma_diags:
                Sigma = torch.zeros((chunks, chunks), device=device, dtype=u0.dtype)
            else:
                stacked_sigmas = torch.stack(all_sigma_diags, dim=0)
                if sigma_reduce in ("mean", "average"):
                    agg_sigma_diag = torch.mean(stacked_sigmas, dim=0)
                elif sigma_reduce == "max":
                    agg_sigma_diag = torch.max(stacked_sigmas, dim=0).values
                elif sigma_reduce == "sum":
                    agg_sigma_diag = torch.sum(stacked_sigmas, dim=0)
                else:
                    agg_sigma_diag = torch.mean(stacked_sigmas, dim=0)
                Sigma = torch.diag(agg_sigma_diag)
            
            new_vector[key] = [U_orth, Sigma, V_orth]
    
    return new_vector


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def run_ufm_energy(cfg: DictConfig) -> None:
    """Main UFM training loop for Energy."""
    logger = setup_simple_logger(__name__)
    
    logger.info("=" * 100)
    logger.info("UFM (Unsupervised FixMatch) Training for Energy")
    logger.info("=" * 100)
    
    with open_dict(cfg):
        # Auto-set sigma_epochs if not provided
        test_ds = cfg.test_dataset
        if test_ds and test_ds in UFM_SIGMA_EPOCHS_PER_DATASET and cfg.sigma_epochs is None:
            cfg.sigma_epochs = UFM_SIGMA_EPOCHS_PER_DATASET[test_ds]
            logger.info(f"✓ Auto-set sigma_epochs={cfg.sigma_epochs} for {test_ds} (dataset-specific)")
        elif cfg.sigma_epochs is None:
            cfg.sigma_epochs = 10
            logger.info(f"Using default sigma_epochs={cfg.sigma_epochs}")
        
        if not cfg.config_tag:
            cfg.config_tag = build_ufm_energy_config_tag(cfg)
    
    # Setup datasets
    test_ds = cfg.test_dataset
    
    if hasattr(cfg, 'DATASETS_ALL') and cfg.DATASETS_ALL:
        base_list = list(cfg.DATASETS_ALL)
    else:
        base_list = ALL_DATASETS[:cfg.num_tasks]
    
    if test_ds in base_list:
        base_list = [d for d in base_list if d != test_ds]
    
    cfg.DATASETS = base_list
    cfg.num_tasks = len(base_list)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in base_list]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)
    
    logger.info(f"Using config tag: {cfg.config_tag}")
    logger.info(f"Basis datasets: {cfg.DATASETS}")
    
    # Load fine-tuned checkpoints
    logger.info("Loading fine-tuned checkpoints for basis tasks...")
    ft_checks = []
    for dataset in cfg.DATASETS_VAL:
        path = get_finetuned_path(cfg.model_location, dataset, model=cfg.model)
        if os.path.exists(path):
            logger.info(f"✓ {path} exists")
            ft_checks.append(load_checkpoint_safe(path, map_location="cpu"))
        else:
            logger.error(f"✗ {path} does not exist")
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
    
    # Load zeroshot checkpoint
    first_dataset = cfg.DATASETS_VAL[0] if cfg.DATASETS_VAL else "dummy"
    zeroshot_path = get_zeroshot_path(cfg.model_location, first_dataset, model=cfg.model)
    logger.info(f"Loading zeroshot model from: {zeroshot_path}")
    ptm_check = load_checkpoint_safe(zeroshot_path, map_location="cpu")
    
    overall_start = time.time()
    
    # Create task vectors
    task_vectors = [
        NonLinearTaskVector(cfg.model, ptm_check, check) for check in ft_checks
    ]
    
    # SVD initialization
    svd_start = time.time()
    init_mode = getattr(cfg, "initialize_sigma", "average").lower()
    if init_mode in ("average", "mean"):
        logger.info("Using SVD initialization with sigma_reduce=mean")
        svd_dict = compute_and_sum_svd_mem_reduction(task_vectors, cfg, sigma_reduce="mean")
    elif init_mode == "sum":
        logger.info("Using SVD initialization with sigma_reduce=sum")
        svd_dict = compute_and_sum_svd_mem_reduction(task_vectors, cfg, sigma_reduce="sum")
    svd_time = time.time() - svd_start
    logger.info(f"Computed SVD bases in {svd_time:.2f}s")
    
    # Export SVD bases
    basis = {}
    for key, value in svd_dict.items():
        if isinstance(value, list) and len(value) == 3:
            U_orth, diag_s, V_orth = value
            sigma_vec = torch.diagonal(diag_s).clone().detach().cpu()
            basis[key] = {
                "U": U_orth.clone().detach().cpu(),
                "V": V_orth.clone().detach().cpu(),
                "sigma": sigma_vec,
            }
    
    logger.info(f"UFM training on held-out dataset: {test_ds}")
    
    # Build sigma modules
    sigma_modules = torch.nn.ModuleDict()
    sigma_key_map = {}
    
    for key, fv in basis.items():
        if all(k in fv for k in ("U", "V", "sigma")):
            U, V, sigma = fv["U"], fv["V"], fv["sigma"]
            if U.ndim == 2 and V.ndim == 2 and sigma.ndim == 1:
                safe_key = key.replace(".", "_")
                if safe_key in sigma_key_map:
                    suffix = 1
                    candidate = f"{safe_key}_{suffix}"
                    while candidate in sigma_key_map:
                        suffix += 1
                        candidate = f"{safe_key}_{suffix}"
                    safe_key = candidate
                sigma_key_map[safe_key] = key
                sigma_modules[safe_key] = SigmaParametrization(U, V, sigma)
    
    sigma_modules = sigma_modules.cuda()
    
    trainable_params = sum(p.numel() for p in sigma_modules.parameters() if p.requires_grad)
    logger.info(f"=" * 80)
    logger.info(f"Number of trainable sigma parameters: {trainable_params:,}")
    logger.info(f"Number of sigma modules: {len(sigma_modules)}")
    logger.info(f"=" * 80)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    val_dataset_name = test_ds + "Val"
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    
    with open_dict(cfg):
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
    
    # Load encoder and head
    image_encoder = ImageEncoder(cfg.model).cuda()
    classification_head = get_classification_head(cfg, test_ds).cuda()
    
    # Prepare validation transform
    val_preprocess = image_encoder.val_preprocess
    
    # Load test dataset for trusted sample selection
    logger.info(f"Loading test dataset: {val_dataset_name}")
    test_dataset = get_dataset(
        test_ds,
        val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    test_data = test_dataset.test_dataset if hasattr(test_dataset, 'test_dataset') else test_dataset
    
    # Capture base params and buffers
    base_params = {
        name: p.detach().clone()
        for name, p in image_encoder.named_parameters()
    }
    base_buffers = {
        name: b.detach().clone()
        for name, b in image_encoder.named_buffers()
    }
    
    # Get predictions for trusted sample selection
    logger.info("Computing predictions for trusted sample selection...")
    preds = get_preds(test_data, image_encoder, classification_head, cfg.device,
                     base_params, base_buffers, sigma_modules, sigma_key_map)
    
    # Select trusted samples
    num_classes = classification_head.out_features
    k_per_class = min(int((len(test_data) / num_classes) / 10), 100)
    logger.info(f"Selecting {k_per_class} trusted samples per class")
    
    confs, amax = preds.max(dim=-1)
    trusted = torch.tensor([], dtype=torch.long)
    for c in range(num_classes):
        ids_c = torch.argsort(preds[:, c])
        trusted = torch.cat((trusted, ids_c[-k_per_class:]))
    
    trusted, _ = torch.sort(trusted)
    unlabeled = torch.tensor([i for i in range(len(test_data)) if i not in trusted])
    
    preds_onehot = torch.nn.functional.one_hot(amax, num_classes=num_classes).float()
    trusted_bool = torch.zeros(len(preds), dtype=torch.bool)
    trusted_bool[trusted] = True
    
    logger.info(f"Trusted samples: {len(trusted)}, Unlabeled: {len(unlabeled)}")
    
    # Prepare strong augmentation
    strong_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + val_preprocess.transforms[-3:])
    
    # Create asymmetric transform
    asym_transform = TwoAsymetricTransform(val_preprocess, strong_preprocess)
    
    # Reload dataset with asymmetric transform
    dataset_ufm = get_dataset(
        test_ds,
        asym_transform,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    test_data_ufm = dataset_ufm.test_dataset if hasattr(dataset_ufm, 'test_dataset') else dataset_ufm
    
    # Wrap with index
    index_dataset = IndexWrapper(test_data_ufm)
    
    # Create two-stream batch sampler
    sampler = TwoStreamBatchSampler(unlabeled, trusted, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        index_dataset,
        batch_sampler=sampler,
        num_workers=4
    )
    
    # Prepare validation loader
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val samples: {len(val_loader.dataset)}")
    
    # Setup optimizer and scheduler
    params = [p for p in sigma_modules.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)
    
    num_batches = len(train_loader)
    total_steps = int(cfg.sigma_epochs) * num_batches
    scheduler = cosine_lr(optimizer, cfg.sigma_lr, int(cfg.warmup_ratio * total_steps), total_steps)
    
    loss_fn = ssl_loss_trusted
    
    # Training loop
    logger.info(f"Starting UFM training for {cfg.sigma_epochs} epochs...")
    best_acc = 0.0
    epoch_times = []
    
    for epoch in range(int(cfg.sigma_epochs)):
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader):
            step = epoch * num_batches + i
            
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            inputs_aug = batch["images_"].cuda()
            idx = batch["index"]
            
            # Build delta map with autograd connectivity
            delta_map = {}
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in base_params and module.sigma.numel() > 0:
                    delta = module()
                    if delta.shape == base_params[orig_key].shape:
                        delta_map[orig_key] = delta
            
            # Combine base params with delta
            params_map = {}
            for name, p in base_params.items():
                if name in delta_map:
                    params_map[name] = p.detach() + delta_map[name]
                else:
                    params_map[name] = p.detach()
            
            # Forward using functional_call
            def encoder_forward(mod, x):
                merged = {}
                merged.update(base_buffers)
                merged.update(params_map)
                return functional_call(mod, merged, (x,))
            
            # Weak augmentation forward
            features = encoder_forward(image_encoder, inputs)
            logits = classification_head(features)
            
            # Strong augmentation forward
            features_aug = encoder_forward(image_encoder, inputs_aug)
            logits_aug = classification_head(features_aug)
            
            # UFM loss
            loss = loss_fn(logits, logits_aug, preds_onehot[idx].cuda(), trusted_bool[idx].cuda())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler(step)
            
            if i % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{i}/{num_batches}] Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
        
        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)
        logger.info(f"Epoch {epoch} training time: {epoch_train_time:.2f}s")
        
        # Evaluate after each epoch
        with torch.no_grad():
            eval_params = {}
            for name, p in base_params.items():
                eval_params[name] = p.clone()
            
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in eval_params and module.sigma.numel() > 0:
                    delta = module().to(eval_params[orig_key].device)
                    if eval_params[orig_key].shape == delta.shape:
                        eval_params[orig_key] = eval_params[orig_key] + delta
            
            image_encoder.load_state_dict(eval_params, strict=False)
            
            val_metrics = evaluate_encoder_with_dataloader(
                image_encoder, classification_head, val_loader, cfg.device
            )
            val_acc = val_metrics['top1']
        
        logger.info(f"Epoch {epoch}: Accuracy = {val_acc * 100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info(f"✓ New best accuracy: {best_acc * 100:.2f}%")
    
    # Final evaluation
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation...")
    logger.info("=" * 100 + "\n")
    
    with torch.no_grad():
        materialized = {}
        for name, p in base_params.items():
            materialized[name] = p.clone()
        
        for safe_key, module in sigma_modules.items():
            orig_key = sigma_key_map.get(safe_key, safe_key)
            if orig_key in materialized and module.sigma.numel() > 0:
                delta = module().to(materialized[orig_key].device)
                if materialized[orig_key].shape == delta.shape:
                    materialized[orig_key] = materialized[orig_key] + delta
        
        image_encoder.load_state_dict(materialized, strict=False)
        
        final_metrics = evaluate_encoder_with_dataloader(
            image_encoder, classification_head, val_loader, cfg.device
        )
        final_acc = final_metrics['top1']
    
    logger.info(f"Final accuracy: {final_acc * 100:.2f}%")
    
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory: {gpu_peak_mem_mb:.2f} MB")
    
    # Save results
    save_dir = os.path.join(
        "./models/checkpoints_tta",
        cfg.model,
        val_dataset_name,
        cfg.config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save sigma modules
    energy_path = os.path.join(save_dir, "ufm_energy.pt")
    sigma_state = {safe_key: module.state_dict() for safe_key, module in sigma_modules.items()}
    torch.save(sigma_state, energy_path)
    logger.info(f"✓ Saved UFM-Energy sigma to {energy_path}")
    
    # Save results JSON
    results_path = os.path.join(save_dir, "ufm_energy_results_none.json")
    results = {
        "target_dataset": test_ds,
        "final_accuracy": float(final_acc),
        "best_accuracy": float(best_acc),
        "k_shot": k,
        "model": cfg.model,
        "sigma_epochs": cfg.sigma_epochs,
        "sigma_lr": cfg.sigma_lr,
        "sigma_wd": cfg.sigma_wd,
        "svd_keep_topk": getattr(cfg, "svd_keep_topk", 2),
        "initialize_sigma": getattr(cfg, "initialize_sigma", None),
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "batch_size": cfg.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "config_tag": cfg.config_tag,
        "method": "ufm_energy",
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")


if __name__ == "__main__":
    from src.datasets.registry import registry as DATASET_REGISTRY
    
    parser = argparse.ArgumentParser(
        description="UFM-Energy training for general datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    allowed_test_datasets = sorted(
        [name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")]
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Held-out dataset for UFM adaptation (sigma epochs auto-set by dataset size)"
    )
    
    # Config and model
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_reverse.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Vision backbone (e.g., ViT-B-32, ViT-B-16)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--sigma_epochs",
        type=int,
        help="Number of sigma training epochs (auto-set per dataset if not provided)"
    )
    parser.add_argument(
        "--sigma_lr",
        type=float,
        help="Learning rate for sigma optimization"
    )
    parser.add_argument(
        "--sigma_wd",
        type=float,
        help="Weight decay for sigma optimization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--k",
        type=int,
        dest="train_k",
        help="K-shot samples per class (0=fullshot)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        help="Warmup ratio for sigma learning rate"
    )
    
    # SVD and initialization
    parser.add_argument(
        "--svd_keep_topk",
        type=int,
        help="Number of singular vectors to keep per task"
    )
    parser.add_argument(
        "--initialize_sigma",
        type=str,
        choices=["average", "sum", "tsvm"],
        help="Initialization strategy for sigma basis"
    )
    
    # Other
    parser.add_argument(
        "--config_tag",
        type=str,
        help="Custom tag for output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for k-shot sampling"
    )
    parser.add_argument(
        "--data_location",
        type=str,
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory"
    )
    parser.add_argument(
        "--openclip_cachedir",
        type=str,
        help="Directory for caching models from OpenCLIP"
    )
    parser.add_argument(
        "--num_grad_accumulation",
        type=int,
        help="Gradient accumulation steps"
    )
    
    args = parser.parse_args()
    
    # Load config file
    cfg = load_config(args.config_file)
    
    # Merge CLI arguments (only non-None values override config)
    cli_overrides = {k: v for k, v in vars(args).items() 
                     if v is not None and k != "config_file"}
    
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    
    # Validate required fields
    if not cfg.get("test_dataset"):
        parser.error("--test_dataset is required")
    
    # Model-specific adjustments
    if cfg.model == "ViT-L-14":
        if not hasattr(cfg, 'batch_size_override') or not cfg.batch_size_override:
            original_batch_size = cfg.get('batch_size', 128)
            if original_batch_size >= 128:
                cfg.batch_size = 32
                print(f"Adjusted batch_size to {cfg.batch_size} for ViT-L-14")
    
    OmegaConf.set_struct(cfg, True)
    run_ufm_energy(cfg)

