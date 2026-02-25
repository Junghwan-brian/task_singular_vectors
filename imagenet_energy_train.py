import os
import time
import json
import logging
import argparse
from typing import Optional, cast, Sized, Dict, List

import torch
import torchvision
from torch.nn.utils.stateless import functional_call

from omegaconf import OmegaConf, open_dict

from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.eval.eval import eval_single_dataset
from src.models.task_vectors import NonLinearTaskVector
from src.utils.sigma_param import SigmaParametrization
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.utils.variables_and_paths import (
    ALL_DATASETS,
    get_finetuned_path,
    get_zeroshot_path,
)
from src.datasets.remote_sensing import sample_k_shot_indices
from src.datasets.imagenet_ood import ImageNetILSVRCVal
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from src.utils.utils import cosine_lr


def grid_search_sigma_alpha(
    sigma_modules: torch.nn.ModuleDict,
    sigma_key_map,
    base_params,
    base_buffers,
    model,
    train_loader,
    device,
    alphas=None,
    max_batches: int = None,
    apply_best: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """
    모든 Sigma(diag)에 동일 스케일(alpha)을 곱해본 뒤,
    주어진 데이터에서 정확도가 가장 높은 alpha를 선택.
    Returns:
        (best_alpha, best_acc)
    """
    if alphas is None:
        alphas = [1, 5, 10, 15, 20]
    if logger is None:
        logger = logging.getLogger(__name__)
    model.eval()
    best_alpha = None
    best_acc = -1.0
    with torch.no_grad():
        for alpha in alphas:
            correct = 0.0
            total = 0.0
            for b_idx, batch in enumerate(train_loader):
                if max_batches is not None and max_batches > 0 and b_idx >= max_batches:
                    break
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(device)
                labels = batch["labels"].to(device)
                # build delta map (no grad)
                delta_map = {}
                for safe_key, module in sigma_modules.items():
                    orig_key = sigma_key_map.get(safe_key, safe_key)
                    if orig_key in base_params and module.sigma.numel() > 0:
                        # scaled delta: U @ diag(relu(sigma) * alpha) @ V
                        sigma_vec = torch.relu(module.sigma) * float(alpha)
                        delta = module.U @ torch.diag(sigma_vec) @ module.V
                        if delta.shape == base_params[orig_key].shape:
                            delta_map[orig_key] = delta
                # merge params
                params_map = {}
                for name, p in base_params.items():
                    if name in delta_map:
                        params_map[name] = p + delta_map[name]
                    else:
                        params_map[name] = p
                # functional forward
                def encoder_forward(mod, x):
                    merged = {}
                    merged.update(base_buffers)
                    merged.update(params_map)
                    return functional_call(mod, merged, (x,))
                features = encoder_forward(model.image_encoder, inputs)
                logits = model.classification_head(features)
                preds = logits.argmax(dim=1, keepdim=False)
                correct += float(preds.eq(labels).sum().item())
                total += float(labels.size(0))
            acc = (correct / total) if total > 0 else 0.0
            logger.info(f"[alpha-grid] alpha={alpha} -> train accuracy={acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
    if best_alpha is None:
        best_alpha = 1.0
    # 적용
    if apply_best:
        for _, module in sigma_modules.items():
            if module.sigma.numel() > 0:
                module.sigma.data.mul_(float(best_alpha))
        logger.info(f"[alpha-grid] Selected alpha={best_alpha} (acc={best_acc*100:.2f}%), applied to sigma.")
    else:
        logger.info(f"[alpha-grid] Selected alpha={best_alpha} (acc={best_acc*100:.2f}%), not applied (apply_best=False).")
    return best_alpha, best_acc


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def get_ood_dataloader(dataset_name: str, preprocess, cfg):
    """Create OOD dataloader for evaluation"""
    dataset = get_dataset(
        dataset_name,
        preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    return get_dataloader(dataset, is_train=False, args=cfg, image_encoder=None)


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_config_tag(cfg) -> str:
    lr_part = _sanitize_value(cfg.sigma_lr)
    svd_part = _sanitize_value(getattr(cfg, "svd_keep_topk", 2))
    init_mode_part = _sanitize_value(
        getattr(cfg, "initialize_sigma", "average"))
    warmup_part = _sanitize_value(getattr(cfg, "warmup_ratio", 0.1))
    wd_part = _sanitize_value(getattr(cfg, "sigma_wd", 0.0))
    k_part = _sanitize_value(getattr(cfg, "train_k", 0))
    return f"imagenet_energy_{lr_part}_{svd_part}_{init_mode_part}_{warmup_part}_{wd_part}_{k_part}shot"


def compute_and_sum_svd_mem_reduction_average(task_vectors, config):
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 3)))
    with torch.no_grad():
        new_vector = {}

        def is_matrix_key(tv0, key):
            return (
                tv0.vector[key].ndim == 2 and
                all(t not in key for t in ("text_projection",
                    "positional", "token_embedding"))
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
            U_orth_T = U_orth.T
            V_orth_T = V_orth.T
            all_sigma_diags = []
            for tv in task_vectors:
                M_i = tv.vector[key].to(device)
                Sigma_i_prime = (U_orth_T @ M_i) @ V_orth_T
                sigma_task_diag = torch.diag(Sigma_i_prime)
                all_sigma_diags.append(sigma_task_diag)
            if not all_sigma_diags:
                Sigma = torch.zeros(
                    (chunks, chunks), device=device, dtype=u0.dtype)
            else:
                stacked_sigmas = torch.stack(all_sigma_diags, dim=0)
                mean_sigma_diag = torch.mean(stacked_sigmas, dim=0)
                Sigma = torch.diag(mean_sigma_diag)
            new_vector[key] = [U_orth, Sigma, V_orth]
    return new_vector


def run_imagenet_energy(cfg) -> None:
    logger = setup_logger(__name__)
    device = cfg.device

    with open_dict(cfg):
        if cfg.sigma_epochs is None:
            cfg.sigma_epochs = 20
        if not cfg.get("config_tag"):
            cfg.config_tag = build_config_tag(cfg)
        if cfg.get("adapter_lr") is None:
            cfg.adapter_lr = cfg.sigma_lr
        if cfg.get("adapter_wd") is None:
            cfg.adapter_wd = cfg.sigma_wd
        if cfg.get("warmup_ratio") is None:
            cfg.warmup_ratio = 0.1

    test_ds = cfg.test_dataset  # e.g., "ImageNetILSVRC"

    # Basis datasets (exclude ImageNet)
    if hasattr(cfg, 'DATASETS_ALL') and cfg.DATASETS_ALL:
        base_list = list(cfg.DATASETS_ALL)
    else:
        base_list = ALL_DATASETS[:cfg.num_tasks]

    cfg.DATASETS = base_list
    cfg.num_tasks = len(base_list)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in base_list]
    cfg.data_location = os.path.expanduser(cfg.data_location)

    logger.info(f"Using config tag: {cfg.config_tag}")
    logger.info("Loading fine-tuned checkpoints for basis tasks:")
    logger.info(f"datasets: {cfg.DATASETS_VAL}")
    logger.info(f"model: {cfg.model}")

    ft_checks = []
    for dataset in cfg.DATASETS_VAL:
        path = get_finetuned_path(cfg.model_location, dataset, model=cfg.model)
        if os.path.exists(path):
            logger.info(f"✓ {path} exists")
            ft_checks.append(torch.load(path, map_location="cpu", weights_only=False))
        else:
            logger.error(f"✗ {path} does not exist")
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")

    # Zeroshot checkpoint
    first_dataset = cfg.DATASETS_VAL[0] if cfg.DATASETS_VAL else "dummy"
    zeroshot_path = get_zeroshot_path(
        cfg.model_location, first_dataset, model=cfg.model)
    logger.info(f"Loading zeroshot model from: {zeroshot_path}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu", weights_only=False)

    overall_start = time.time()
    task_vectors = [NonLinearTaskVector(
        cfg.model, ptm_check, check) for check in ft_checks]

    # Build SVD bases
    svd_start = time.time()
    svd_dict = compute_and_sum_svd_mem_reduction_average(task_vectors, cfg)
    svd_time = time.time() - svd_start
    logger.info(f"Computed SVD bases in {svd_time:.2f}s")

    # Export basis
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

    # Sigma modules
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
    sigma_modules = sigma_modules.to(device)

    # Count params
    trainable_params = sum(p.numel()
                           for p in sigma_modules.parameters() if p.requires_grad)
    logger.info("=" * 80)
    logger.info(f"Number of trainable sigma parameters: {trainable_params:,}")
    logger.info(f"Number of sigma modules: {len(sigma_modules)}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Dataset objects
    val_dataset_name = test_ds + "Val"
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    with open_dict(cfg):
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)

    pretrained_encoder = ImageEncoder(cfg.model).to(device)
    image_encoder = ImageEncoder(cfg.model).to(device)

    # Train/Val datasets from ImageNet-1k validation folder (flat)
    logger.info(f"Loading ImageNet-1k validation dataset for train/val split: {val_dataset_name}")
    train_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=224, scale=(
            0.5, 1.0), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])

    # Build two dataset objects pointing to the same files but with different transforms
    _train_val_obj = ImageNetILSVRCVal(train_preprocess, location=cfg.data_location, batch_size=cfg.batch_size,
                                       num_workers=int(getattr(cfg, "num_workers", 2)))
    base_train_dataset = _train_val_obj.test_dataset  # use training augmentations

    _eval_val_obj = ImageNetILSVRCVal(ImageEncoder(cfg.model).val_preprocess, location=cfg.data_location,
                                      batch_size=cfg.batch_size, num_workers=int(getattr(cfg, "num_workers", 2)))
    base_val_dataset = _eval_val_obj.test_dataset  # use eval transforms

    classification_head = get_classification_head(cfg, test_ds)
    model = ImageClassifier(image_encoder, classification_head).to(device)
    model.freeze_head()

    # Build initial loaders; will replace with k-shot split below
    def _build_loader(dataset, is_train: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=is_train,
            num_workers=int(getattr(cfg, "num_workers", 8)),
            pin_memory=True,
        )
    train_loader = _build_loader(base_train_dataset, is_train=True)
    val_full_loader = _build_loader(base_val_dataset, is_train=False)

    # k-shot sampling
    if k is not None and k > 0:
        logger.info(f"Applying k-shot sampling: {k} samples per class (seed={int(getattr(cfg,'seed',1))})")

        def _save_k_shot_indices(indices, save_dir, dataset_name, kshot, seed):
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"k_shot_indices_k{kshot}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump({"indices": indices, "dataset": dataset_name, "k": int(kshot), "seed": int(seed)}, f)
            return path

        def _load_k_shot_indices(save_dir, kshot, seed):
            path = os.path.join(save_dir, f"k_shot_indices_k{kshot}_seed{seed}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                return data.get("indices", None)
            return None

        def _extract_labels_for_indices(dataset, indices):
            import numpy as np
            labels: List[int] = []
            # try fast paths
            if hasattr(dataset, "targets") and dataset.targets is not None:
                tgt = dataset.targets
                if torch.is_tensor(tgt):
                    tgt = tgt.cpu().numpy()
                elif not isinstance(tgt, np.ndarray):
                    tgt = np.array(tgt)
                for idx in indices:
                    labels.append(int(tgt[idx]))
                return labels
            # fallback: index pairs
            for idx in indices:
                _, label = dataset[idx]
                if torch.is_tensor(label):
                    label = int(label.item())
                labels.append(int(label))
            return labels

        def _subsample_from_larger_k(larger_indices, dataset, target_k):
            import numpy as np
            labels = _extract_labels_for_indices(dataset, larger_indices)
            class_to_indices: Dict[int, List[int]] = {}
            for idx, lab in zip(larger_indices, labels):
                class_to_indices.setdefault(int(lab), []).append(idx)
            selected: List[int] = []
            for lab in sorted(class_to_indices.keys()):
                selected.extend(class_to_indices[lab][:target_k])
            return selected

        seed = int(getattr(cfg, "seed", 1))
        indices_dir = os.path.join(cfg.model_location, cfg.model, val_dataset_name)
        selected_indices = _load_k_shot_indices(indices_dir, k, seed)

        if selected_indices is None:
            larger_k = 16
            if k < larger_k:
                larger = _load_k_shot_indices(indices_dir, larger_k, seed)
                if larger is not None:
                    logger.info(f"✓ Subsampling from existing {larger_k}-shot to {k}-shot (seed={seed})")
                    selected_indices = _subsample_from_larger_k(larger, base_train_dataset, k)
                    saved_path = _save_k_shot_indices(selected_indices, indices_dir, val_dataset_name, k, seed)
                    logger.info(f"✓ Saved {k}-shot indices to {saved_path}")
        if selected_indices is None:
            logger.info(f"Sampling new {k}-shot indices for {cfg.test_dataset} (seed={seed})")
            selected_indices = sample_k_shot_indices(
                base_train_dataset, k, seed=seed, verbose=True, progress_desc=f"{cfg.test_dataset} {k}-shot"
            )
            saved_path = _save_k_shot_indices(selected_indices, indices_dir, val_dataset_name, k, seed)
            logger.info(f"✓ Saved k-shot indices to {saved_path}")

        # Build train loader from selected indices (train transforms)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_train_dataset, selected_indices),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg, "num_workers", 8)),
            pin_memory=True,
        )
        logger.info(f"✓ Created {k}-shot train loader with {len(selected_indices)} samples")

        # Build validation loader from the complement (eval transforms)
        all_indices = set(range(len(base_val_dataset)))
        comp_indices = sorted(all_indices.difference(set(selected_indices)))
        if len(comp_indices) == 0:
            raise RuntimeError("Validation set is empty after k-shot split; check data and k value.")
        val_full_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_val_dataset, comp_indices),
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg, "num_workers", 8)),
            pin_memory=True,
        )

    # Save dirs
    config_dir = os.path.join(
        cfg.model_location, cfg.model, val_dataset_name, cfg.config_tag)
    os.makedirs(config_dir, exist_ok=True)
    energy_save_dir = os.path.join(config_dir, shot_folder)
    os.makedirs(energy_save_dir, exist_ok=True)
    
    energy_path = os.path.join(energy_save_dir, "energy.pt")
    
    # Define evaluation helper function
    def _eval_on_loader(image_encoder_mod, loader) -> float:
        image_encoder_mod.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                feats = image_encoder_mod(x)
                logits = model.classification_head(feats)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.size(0))
        if total == 0:
            raise RuntimeError("Validation loader is empty.")
        return float(correct / total)
    
    # Check if eval_only mode
    if getattr(cfg, 'eval_only', False):
        logger.info("=" * 80)
        logger.info("EVAL ONLY MODE: Loading existing model for evaluation")
        logger.info("=" * 80)
        
        if not os.path.exists(energy_path):
            logger.error(f"Energy model not found: {energy_path}")
            logger.error("Cannot run eval_only mode without trained model!")
            return
        
        logger.info(f"Loading energy model from: {energy_path}")
        model.image_encoder = ImageEncoder.load(cfg.model, energy_path).to(device)
        logger.info("✓ Model loaded successfully")
        
        # Skip to evaluation
        model.eval()
        with torch.no_grad():
            final_acc = _eval_on_loader(model.image_encoder, val_full_loader)
        logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")
        
        # OOD evaluations
        ood_results = {}
        ood_list = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2MFVal"]
        logger.info("\n" + "=" * 100)
        logger.info("Evaluating on OOD datasets...")
        logger.info("=" * 100 + "\n")
        model.eval()
        with torch.no_grad():
            for ood_name in ood_list:
                ood_loader = get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)
                ood_metrics = evaluate_encoder_with_dataloader(
                    model.image_encoder, model.classification_head, ood_loader, cfg.device
                )
                ood_results[ood_name] = float(ood_metrics['top1'])
                logger.info(f"OOD {ood_name}: {100.0 * ood_results[ood_name]:.2f}%")
        
        # Compose unified evaluation results
        all_eval_accuracies = {val_dataset_name: float(final_acc)}
        for k_name, v_acc in ood_results.items():
            all_eval_accuracies[k_name] = float(v_acc)
        
        # Save updated results
        results_path = os.path.join(energy_save_dir, f"energy_results_imagenet.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}
        
        results.update({
            "final_accuracy": float(final_acc),
            "ood_accuracies": ood_results,
            "all_eval_accuracies": all_eval_accuracies,
        })
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"✓ Updated results saved to {results_path}")
        logger.info("\n" + "=" * 100)
        logger.info("EVAL ONLY MODE COMPLETE")
        logger.info("=" * 100)
        return

    # Optimizer
    params = [p for p in sigma_modules.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)

    # Base params/buffers snapshot
    base_params = {name: p.detach().clone()
                   for name, p in model.image_encoder.named_parameters()}
    base_buffers = {name: b.detach().clone()
                    for name, b in model.image_encoder.named_buffers()}
    base_state_dict = {name: t.detach().clone()
                       for name, t in model.image_encoder.state_dict().items()}

    # History
    loss_history = []
    val_history = []
    eval_epochs = set(range(int(cfg.sigma_epochs))) if int(cfg.sigma_epochs) <= 5 else {
        min(int(cfg.sigma_epochs) - 1, int(round(i * (int(cfg.sigma_epochs) - 1) / 4))) for i in range(5)
    }
    eval_counter = 0

    def record_validation(stage: str, epoch_value, accuracy_value):
        nonlocal eval_counter
        record = {
            "stage": stage,
            "epoch": int(epoch_value),
            "accuracy": float(accuracy_value),
            "elapsed_seconds": float(time.time() - overall_start),
            "evaluation_index": int(eval_counter),
        }
        val_history.append(record)
        eval_counter += 1
        return record

    # Pretrained/zeroshot evals
    model.eval()
    with torch.no_grad():
        # Evaluate on our validation split instead of registry dataset to reflect k-shot split
        # pretrained_acc = _eval_on_loader(pretrained_encoder, val_full_loader)
        # logger.info(
        #     f"Pretrained encoder validation accuracy: {pretrained_acc * 100:.2f}%")
        # record_validation("pretrained", -2, pretrained_acc)

        eval_params = {name: p.clone() for name, p in base_params.items()}
        for safe_key, module in sigma_modules.items():
            orig_key = sigma_key_map.get(safe_key, safe_key)
            if orig_key in eval_params and module.sigma.numel() > 0:
                sigma_mod = cast(SigmaParametrization, module)
                delta = sigma_mod.forward().to(eval_params[orig_key].device)
                if eval_params[orig_key].shape == delta.shape:
                    eval_params[orig_key] = eval_params[orig_key] + delta
        model.image_encoder.load_state_dict(eval_params, strict=False)
        # zeroshot_acc = _eval_on_loader(model.image_encoder, val_full_loader)
        # logger.info(
        #     f"Zeroshot encoder validation accuracy: {zeroshot_acc * 100:.2f}%")
        # record_validation("zeroshot", -1, zeroshot_acc)
        model.image_encoder.load_state_dict(base_state_dict, strict=False)

    # Alpha grid search for Sigma scaling (pre-train)
    try:
        alphas = getattr(cfg, "alpha_grid_alphas", [1, 3, 5, 7, 10])
        max_batches = getattr(cfg, "alpha_grid_max_batches", 50)
        if isinstance(max_batches, (int, float)):
            max_batches = int(max(0, max_batches))
        else:
            max_batches = 50
        best_alpha, best_acc = grid_search_sigma_alpha(
            sigma_modules=sigma_modules,
            sigma_key_map=sigma_key_map,
            base_params=base_params,
            base_buffers=base_buffers,
            model=model,
            train_loader=train_loader,
            device=device,
            alphas=alphas,
            max_batches=max_batches,
            apply_best=True,
            logger=logger,
        )
        logger.info(f"[alpha-grid] Using alpha={best_alpha} before training (acc={best_acc*100:.2f}%)")
    except Exception as e:
        logger.warning(f"[alpha-grid] Skipped due to error: {e}")

    # Train
    model.train()
    epoch_times = []
    logger.info(f"Starting sigma fine-tuning for {cfg.sigma_epochs} epochs...")
    steps_per_epoch = len(cast(Sized, train_loader))
    logger.info(
        f"Train dataset size: {len(cast(Sized, train_loader.dataset))}, Batch size: {cfg.batch_size}, Steps per epoch: {steps_per_epoch}")

    # Cosine LR scheduler with warmup (step-wise)
    num_batches = steps_per_epoch
    total_steps = int(cfg.sigma_epochs) * num_batches
    warmup_steps = int(float(getattr(cfg, "warmup_ratio", 0.1)) * total_steps)
    scheduler = cosine_lr(optimizer, cfg.sigma_lr, warmup_steps, total_steps)

    for epoch in range(int(cfg.sigma_epochs)):
        epoch_start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            delta_map = {}
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in base_params and module.sigma.numel() > 0:
                    sigma_mod = cast(SigmaParametrization, module)
                    delta = sigma_mod.forward()
                    if delta.shape == base_params[orig_key].shape:
                        delta_map[orig_key] = delta

            params_map = {}
            for name, p in base_params.items():
                if name in delta_map:
                    params_map[name] = p.detach() + delta_map[name]
                else:
                    params_map[name] = p.detach()

            def encoder_forward(mod, x):
                merged = {}
                merged.update(base_buffers)
                merged.update(params_map)
                return functional_call(mod, merged, (x,))

            features = encoder_forward(model.image_encoder, inputs)
            logits = model.classification_head(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            # step-wise LR scheduling
            step = epoch * steps_per_epoch + i
            scheduler(step)

            loss_history.append(
                {"epoch": int(epoch), "iteration": int(i), "loss": float(loss.item())})

            if i == 0:
                try:
                    grad_sum = 0.0
                    for p in params:
                        if p.grad is not None:
                            grad_sum += float(p.grad.detach().abs().sum().item())
                    logger.info(
                        f"[sigma] epoch {epoch} {i + 1}/{len(train_loader)} loss {loss.item():.4f} grad_sum {grad_sum:.4e} lr {optimizer.param_groups[0]['lr']:.6f}")
                except Exception:
                    logger.info(
                        f"[sigma] epoch {epoch} {i + 1}/{len(train_loader)} loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.6f}")

        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)

        # if epoch in eval_epochs:
        #     model.eval()
        #     with torch.no_grad():
        #         eval_params = {name: p.clone()
        #                        for name, p in base_params.items()}
        #         for safe_key, module in sigma_modules.items():
        #             orig_key = sigma_key_map.get(safe_key, safe_key)
        #             if orig_key in eval_params and module.sigma.numel() > 0:
        #                 sigma_mod = cast(SigmaParametrization, module)
        #                 delta = sigma_mod.forward().to(
        #                     eval_params[orig_key].device)
        #                 if eval_params[orig_key].shape == delta.shape:
        #                     eval_params[orig_key] = eval_params[orig_key] + delta
        #         model.image_encoder.load_state_dict(eval_params, strict=False)
        #         val_acc = _eval_on_loader(model.image_encoder, val_full_loader)
        #         logger.info(
        #             f"[sigma] epoch {epoch} validation accuracy: {val_acc * 100:.2f}%")
        #         record_validation("epoch", epoch, val_acc)
        #         model.image_encoder.load_state_dict(
        #             base_state_dict, strict=False)
        #     model.train()

    # Materialize deltas and save
    with torch.no_grad():
        materialized = {name: p.clone() for name, p in base_params.items()}
        for safe_key, module in sigma_modules.items():
            orig_key = sigma_key_map.get(safe_key, safe_key)
            if orig_key in materialized and module.sigma.numel() > 0:
                sigma_mod = cast(SigmaParametrization, module)
                delta = sigma_mod.forward().to(materialized[orig_key].device)
                if materialized[orig_key].shape == delta.shape:
                    materialized[orig_key] = materialized[orig_key] + delta
        model.image_encoder.load_state_dict(materialized, strict=False)

    os.makedirs(energy_save_dir, exist_ok=True)
    model.image_encoder.save(energy_path)
    logger.info(f"Saved energy-trained encoder to {energy_path}")

    # Final eval on ImageNet val
    model.eval()
    with torch.no_grad():
        final_acc = _eval_on_loader(model.image_encoder, val_full_loader)
    logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")
    record_validation("final", int(cfg.sigma_epochs), final_acc)

    # OOD evals + include ImageNet val in a single combined dict - use fine-tuned head
    ood_results = {}
    ood_list = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2MFVal"]
    logger.info("\n" + "=" * 100)
    logger.info("Evaluating on OOD datasets...")
    logger.info("=" * 100 + "\n")
    model.eval()
    with torch.no_grad():
        for ood_name in ood_list:
            ood_loader = get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)
            ood_metrics = evaluate_encoder_with_dataloader(
                model.image_encoder, model.classification_head, ood_loader, cfg.device
            )
            ood_results[ood_name] = float(ood_metrics['top1'])
            logger.info(
                f"OOD {ood_name}: {100.0 * ood_results[ood_name]:.2f}%")
            record_validation(f"ood:{ood_name}", int(
                cfg.sigma_epochs), ood_results[ood_name])

    # Compose unified evaluation results
    all_eval_accuracies = {val_dataset_name: float(final_acc)}
    for k_name, v_acc in ood_results.items():
        all_eval_accuracies[k_name] = float(v_acc)

    # Timings
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / \
        len(epoch_times) if epoch_times else 0.0
    logger.info(
        f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

    # Save results JSON
    results_path = os.path.join(
        energy_save_dir, f"energy_results_imagenet.json")
    results = {
        "target_dataset": test_ds,
        "final_accuracy": float(final_acc),
        "k_shot": k,
        "model": cfg.model,
        "sigma_epochs": cfg.sigma_epochs,
        "sigma_lr": cfg.sigma_lr,
        "svd_keep_topk": getattr(cfg, "svd_keep_topk", 2),
        "initialize_sigma": getattr(cfg, "initialize_sigma", None),
        "adapter_choice": getattr(cfg, "adapter", "none"),
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "batch_size": cfg.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "validation_history": val_history,
        "evaluation_schedule": [int(ep) for ep in sorted(eval_epochs)],
        # "pretrained_accuracy": float(pretrained_acc),
        # "zeroshot_accuracy": float(zeroshot_acc),
        "config_tag": cfg.config_tag,
        "ood_accuracies": ood_results,
        "all_eval_accuracies": all_eval_accuracies,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet Energy-based few-shot training with OOD evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--test_dataset", type=str, default="ImageNetILSVRC",
                        help="Held-out dataset to train (ImageNetILSVRC)")

    # Config/model
    parser.add_argument("--config_file", type=str,
                        default="config/config_reverse.yaml", help="Path to configuration YAML file")
    parser.add_argument("--model", type=str,
                        help="Vision backbone (e.g., ViT-B-32, ViT-B-16)")

    # Training hyperparameters
    parser.add_argument("--sigma_epochs", type=int,
                        help="Number of sigma training epochs")
    parser.add_argument("--sigma_lr", type=float,
                        help="Learning rate for sigma optimization")
    parser.add_argument("--sigma_wd", type=float,
                        help="Weight decay for sigma optimization")
    parser.add_argument("--warmup_ratio", type=float,
                        help="Warmup ratio for sigma learning rate")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--k", type=int, dest="train_k",
                        help="K-shot samples per class (0=fullshot)")

    # SVD
    parser.add_argument("--svd_keep_topk", type=int,
                        help="Number of singular vectors to keep per task")
    parser.add_argument("--initialize_sigma", type=str, default="average",
                        choices=["average"], help="Initialization strategy for sigma basis")

    # Other
    parser.add_argument("--config_tag", type=str,
                        help="Custom tag for output directory")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for k-shot sampling")
    parser.add_argument("--data_location", type=str, default="./datasets",
                        help="Root dir containing 'imagenet' and OOD datasets")
    parser.add_argument("--model_location", type=str,
                        default="./models/checkpoints", help="Where checkpoints/heads are stored")
    parser.add_argument("--num_tasks", type=int, default=17,
                        help="How many basis tasks to use from ALL_DATASETS")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gpu", type=int, help="GPU id (overrides --device as cuda:{id})")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing model without training")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    OmegaConf.set_struct(cfg, False)
    cli_overrides = {k: v for k, v in vars(
        args).items() if v is not None and k != "config_file"}
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    # Override device with specific GPU if provided
    if getattr(args, "gpu", None) is not None:
        try:
            gpu_id = int(args.gpu)
            cfg.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.device = "cpu"
    OmegaConf.set_struct(cfg, True)

    run_imagenet_energy(cfg)


if __name__ == "__main__":
    main()
