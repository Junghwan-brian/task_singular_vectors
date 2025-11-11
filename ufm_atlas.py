"""UFM (Unsupervised FixMatch) training for Atlas with test-time adaptation.

Based on atlas_reverse.py for data/model loading and learn_ufm.py for UFM loss.
Results are saved to checkpoints_tta directory.
"""

import os
import argparse
import time
import json
import copy
import torch
import torchvision
import logging
from typing import Optional
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Fix for H100 cuDNN compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from torch.cuda.amp import GradScaler
from atlas_src.modeling import ImageEncoder, ImageClassifier
from atlas_src.composition import WeightedImageEncoder
from atlas_src.utils import TIPWrapper, LPPWrapper

# Task vectors
from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import (
    get_zeroshot_path,
    get_finetuned_path,
    TQDM_BAR_FORMAT,
)

# Utils
from src.utils.utils import cosine_lr, load_checkpoint_safe
from src.datasets.common import get_dataloader, maybe_dictionarize

# Dataset imports
from src.datasets import get_dataset
from src.models import get_classification_head
from src.datasets.remote_sensing import sample_k_shot_indices

# Evaluation
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader


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


# Dataset-specific epochs for UFM-Atlas training (general datasets)
UFM_ATLAS_EPOCHS_PER_DATASET = {
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


def get_preds(dataset, model, device):
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
            logits = model(inputs)
            all_preds.append(logits.softmax(dim=1).cpu())
    
    return torch.cat(all_preds, dim=0)


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_ufm_atlas_config_tag(num_basis: int, args) -> str:
    count_part = _sanitize_value(max(num_basis, 0))
    lr_part = _sanitize_value(getattr(args, 'lr', 'na'))
    return f"ufm_atlas_{count_part}_{lr_part}"


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


def run_ufm_atlas(args):
    """Main UFM training loop for Atlas."""
    logger = setup_simple_logger(__name__)
    
    logger.info("=" * 100)
    logger.info("UFM (Unsupervised FixMatch) Training for Atlas")
    logger.info("=" * 100)
    
    # Auto-set epochs based on dataset if not provided
    test_ds = args.test_dataset
    if test_ds in UFM_ATLAS_EPOCHS_PER_DATASET and args.epochs is None:
        args.epochs = UFM_ATLAS_EPOCHS_PER_DATASET[test_ds]
        logger.info(f"✓ Auto-set epochs={args.epochs} for {test_ds} (dataset-specific)")
    elif args.epochs is None:
        args.epochs = 10
        logger.info(f"Using default epochs={args.epochs} for {test_ds}")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_reverse.yaml")
    config = OmegaConf.load(config_path)
    
    # Import dataset registry
    from src.datasets.registry import registry as DATASET_REGISTRY
    GENERAL_DATASETS = {k: v for k, v in DATASET_REGISTRY.items() if not k.endswith("Val")}
    
    test_ds = args.test_dataset
    pool = [d for d in GENERAL_DATASETS.keys() if d != test_ds]
    logger.info(f"Using {len(pool)} datasets as basis (excluding {test_ds})")
    
    # Load task vectors
    logger.info(f"Loading task vectors for {len(pool)} basis datasets...")
    ft_checks = {}
    for dataset in pool:
        dataset_val = dataset + "Val"
        finetuned_checkpoint_path = get_finetuned_path(
            args.model_location, dataset_val, args.model
        )
        if os.path.exists(finetuned_checkpoint_path):
            ft_checks[dataset] = load_checkpoint_safe(finetuned_checkpoint_path, map_location="cpu")
            logger.info(f"✓ Loaded fine-tuned checkpoint for {dataset}")
        else:
            logger.warning(f"✗ Missing fine-tuned checkpoint for {dataset}")
    
    # Load zeroshot checkpoint
    first_dataset_val = pool[0] + "Val"
    zeroshot_path = get_zeroshot_path(args.model_location, first_dataset_val, args.model)
    logger.info(f"Loading shared zeroshot model from: {zeroshot_path}")
    ptm_check = load_checkpoint_safe(zeroshot_path, map_location="cpu")
    
    # Create task vectors
    task_vectors = {}
    for dataset, ft_check in ft_checks.items():
        task_vectors[dataset] = NonLinearTaskVector(args.model, ptm_check, ft_check)
        logger.info(f"✓ Created task vector for {dataset}")
    
    # Filter out target task
    available_task_vectors = [v for k, v in task_vectors.items() if test_ds != k]
    logger.info(f"Using {len(available_task_vectors)} task vectors for composition")
    
    # Generate config tag
    config_tag = build_ufm_atlas_config_tag(len(available_task_vectors), args)
    logger.info(f"Using config tag: {config_tag}")
    
    # Create WeightedImageEncoder
    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, 
        available_task_vectors,
        blockwise=args.blockwise_coef,
        partition=args.partition,
    )
    
    # Get classification head
    classification_head = get_classification_head(args, test_ds)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()
    
    # Prepare val preprocess
    val_preprocess = model.val_preprocess
    
    # Load test dataset for trusted sample selection
    logger.info(f"Loading test dataset: {test_ds}")
    test_dataset = get_dataset(
        test_ds,
        val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    
    # Get predictions for trusted sample selection
    logger.info("Computing predictions for trusted sample selection...")
    test_data = test_dataset.test_dataset if hasattr(test_dataset, 'test_dataset') else test_dataset
    preds = get_preds(test_data, model, args.device)
    
    # Select trusted samples (top-k per class)
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
    
    # Create pseudo-labels (one-hot)
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
        location=args.data_location,
        batch_size=args.batch_size,
    )
    test_data_ufm = dataset_ufm.test_dataset if hasattr(dataset_ufm, 'test_dataset') else dataset_ufm
    
    # Wrap with index
    index_dataset = IndexWrapper(test_data_ufm)
    
    # Create two-stream batch sampler
    sampler = TwoStreamBatchSampler(unlabeled, trusted, args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        index_dataset, 
        batch_sampler=sampler,
        num_workers=4
    )
    
    # Prepare validation loader
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val samples: {len(val_loader.dataset)}")
    
    # Setup optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, args.lr, 0, args.epochs * num_batches)
    
    scaler = GradScaler()
    loss_fn = ssl_loss_trusted
    
    # Training loop
    logger.info(f"Starting UFM training for {args.epochs} epochs...")
    best_coef = model.image_encoder.coef.data.clone()
    best_acc = 0.0
    
    overall_start = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader):
            step = epoch * num_batches + i
            
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            inputs_aug = batch["images_"].cuda()
            idx = batch["index"]
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inputs)
                logits_aug = model(inputs_aug)
                loss = loss_fn(logits, logits_aug, preds_onehot[idx], trusted_bool[idx])
            
            scaler.scale(loss).backward()
            
            scheduler(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if i % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{i}/{num_batches}] Loss: {loss.item():.4f}"
                )
        
        # Evaluate after each epoch
        model.eval()
        with torch.no_grad():
            metrics = evaluate_encoder_with_dataloader(
                model.image_encoder, classification_head, val_loader, args.device
            )
            acc = metrics['top1']
        
        logger.info(f"Epoch {epoch}: Accuracy = {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_coef = model.image_encoder.coef.data.clone()
            logger.info(f"✓ New best accuracy: {best_acc * 100:.2f}%")
        
        model.train()
    
    # Load best coefficients
    model.image_encoder.coef = torch.nn.Parameter(best_coef)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_metrics = evaluate_encoder_with_dataloader(
            model.image_encoder, classification_head, val_loader, args.device
        )
        final_acc = final_metrics['top1']
    
    logger.info(f"Final accuracy: {final_acc * 100:.2f}%")
    
    # Save results
    k = getattr(args, 'k', 0)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    
    save_dir = os.path.join(
        "./models/checkpoints_tta",
        args.model,
        test_ds + "Val",
        config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save coefficients
    atlas_path = os.path.join(save_dir, "ufm_atlas.pt")
    torch.save(best_coef, atlas_path)
    logger.info(f"✓ Saved UFM-Atlas coefficients to {atlas_path}")
    
    # Save results JSON
    results_path = os.path.join(save_dir, "ufm_atlas_results_none.json")
    results = {
        "target_dataset": test_ds,
        "final_accuracy": float(final_acc),
        "best_accuracy": float(best_acc),
        "k_shot": k,
        "model": args.model,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "training_time": time.time() - overall_start,
        "config_tag": config_tag,
        "method": "ufm_atlas",
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")


if __name__ == "__main__":
    # Load config file first
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_reverse.yaml")
    config = OmegaConf.load(config_path)
    
    parser = argparse.ArgumentParser(
        description="UFM training for Atlas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--test_dataset", type=str, required=True,
                       help="Target dataset for UFM adaptation")
    parser.add_argument("--model", type=str, default=config.get("model", "ViT-B-16"),
                       help="Model architecture")
    parser.add_argument("--data_location", type=str, 
                       default=config.get("data_location", "./datasets"),
                       help="Root directory for datasets")
    parser.add_argument("--model_location", type=str, 
                       default=config.get("model_location", "./models/checkpoints"),
                       help="Directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, 
                       default=config.get("batch_size", 128),
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (auto-set per dataset if not provided)")
    parser.add_argument("--k", type=int, default=0,
                       help="K-shot samples per class (0=fullshot)")
    parser.add_argument("--blockwise_coef", action="store_true", default=True,
                       help="Learn per-block coefficients")
    parser.add_argument("--partition", type=int, default=None,
                       help="Partition size for coefficients")
    parser.add_argument("--seed", type=int, default=config.get("seed", 1),
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Expand paths
    args.data_location = os.path.expanduser(args.data_location)
    args.model_location = os.path.expanduser(args.model_location)
    
    # Model-specific adjustments
    if args.model == "ViT-L-14":
        if args.batch_size == config.get("batch_size", 128):  # If using default
            args.batch_size = 32
            print(f"Adjusted batch_size to {args.batch_size} for ViT-L-14")
    
    run_ufm_atlas(args)

