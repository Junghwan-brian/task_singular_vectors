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

torch.backends.cuda.enable_cudnn_sdp(False)
# Additional cuDNN settings for H100 compatibility
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
# Set cuDNN benchmark to False for stability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    # "Cars": 1,
    "DTD": 1,
    # "EuroSAT": 1,
    "GTSRB": 1,
    "MNIST": 1,
    # "RESISC45": 1,
    # "SUN397": 1,
    "SVHN": 1,
    "CIFAR10": 1,
    "CIFAR100": 1,
    "STL10": 1,
    "Food101": 1,
    "Flowers102": 1,
    # "FER2013": 1,
    "PCAM": 1,
    "OxfordIIITPet": 1,
    "RenderedSST2": 1,
    "EMNIST": 1,
    "FashionMNIST": 1,
    # "KMNIST": 1,
    "FGVCAircraft": 1,
    "CUB200": 1,
    "Country211": 1,
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
            # Handle dict returned by TwoAsymetricTransform
            result = data.copy()
            result["index"] = idx
            return result
        elif isinstance(data, tuple) and len(data) == 2:
            # Handle (data, label) tuple
            if isinstance(data[0], dict):
                # Transform returned a dict (e.g., TwoAsymetricTransform)
                result = data[0].copy()
                result["labels"] = data[1]
                result["index"] = idx
                return result
            else:
                # Normal (image, label) tuple
                return {"images": data[0], "labels": data[1], "index": idx}
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
    
    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    """Custom collate function to handle nested dicts from TwoAsymetricTransform."""
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == "index":
            result[key] = torch.tensor([item[key] for item in batch])
        elif key in ["images", "images_"]:
            # Stack image tensors
            result[key] = torch.stack([item[key] for item in batch])
        elif key == "labels":
            # Handle labels if present
            labels = [item[key] for item in batch]
            if isinstance(labels[0], torch.Tensor):
                result[key] = torch.stack(labels)
            else:
                result[key] = torch.tensor(labels)
        else:
            # Default: try to stack or keep as list
            try:
                result[key] = torch.stack([item[key] for item in batch])
            except:
                result[key] = [item[key] for item in batch]
    
    return result


class TwoStreamBatchSampler(torch.utils.data.Sampler):
    """Sample unlabeled and labeled (trusted) data in each batch."""
    def __init__(self, unlabeled_indices, labeled_indices, batch_size):
        # Convert to Python list to avoid torch.Tensor indexing issues
        self.unlabeled_indices = unlabeled_indices.tolist() if torch.is_tensor(unlabeled_indices) else list(unlabeled_indices)
        self.labeled_indices = labeled_indices.tolist() if torch.is_tensor(labeled_indices) else list(labeled_indices)
        self.batch_size = batch_size
        
        # Compute batch counts
        self.n_unlabeled = len(self.unlabeled_indices)
        self.n_labeled = len(self.labeled_indices)
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
                # Convert idx to Python int to ensure compatibility with all datasets
                batch.append(self.unlabeled_indices[int(idx)])
            
            # Sample from labeled
            for _ in range(self.batch_size // 2):
                try:
                    idx = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(torch.randperm(self.n_labeled))
                    idx = next(labeled_iter)
                # Convert idx to Python int to ensure compatibility with all datasets
                batch.append(self.labeled_indices[int(idx)])
            
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
        dataset, batch_size=128, shuffle=False, num_workers=2
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
    
    # Use basis_datasets from args if available, otherwise fall back to config.DATASETS_ALL
    if hasattr(args, 'basis_datasets') and args.basis_datasets:
        pool = args.basis_datasets
        logger.info(f"Using {len(pool)} basis datasets from args (excluding {test_ds})")
    elif hasattr(config, 'DATASETS_ALL') and config.DATASETS_ALL:
        test_ds = args.test_dataset
        pool = [d for d in config.DATASETS_ALL if d != test_ds]
        logger.info(f"Using {len(pool)} datasets from config.DATASETS_ALL (excluding {test_ds})")
    else:
        # Fallback to registry
        from src.datasets.registry import registry as DATASET_REGISTRY
        GENERAL_DATASETS = {k: v for k, v in DATASET_REGISTRY.items() if not k.endswith("Val")}
        test_ds = args.test_dataset
        pool = [d for d in GENERAL_DATASETS.keys() if d != test_ds]
        logger.info(f"Using {len(pool)} datasets from registry (excluding {test_ds})")
    
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
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Prepare validation loader
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
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


def create_ufm_atlas_parser(config: OmegaConf) -> argparse.ArgumentParser:
    """
    Create argument parser with defaults from config file.
    Based on atlas_reverse.py's create_atlas_parser.
    """
    parser = argparse.ArgumentParser(
        description="UFM-Atlas task vector composition for general datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Target dataset for UFM leave-one-out training"
    )
    
    # Config file
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_reverse.yaml",
        help="Path to configuration YAML file"
    )
    
    # Model and paths
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("model", "ViT-B-32"),
        help="Model architecture"
    )
    parser.add_argument(
        "--data_location",
        type=str,
        default=config.get("data_location", "./datasets"),
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        default=config.get("model_location", "./models/checkpoints"),
        help="Directory for model checkpoints"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,  # UFM default
        help="Learning rate"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,  # Atlas default
        help="Weight decay"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (auto-set per dataset if not provided)"
    )
    parser.add_argument(
        "--num_grad_accumulation",
        type=int,
        default=config.get("num_grad_accumulation", 1),
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.get("train_k", 0),
        help="K-shot samples per class (0=fullshot)"
    )
    
    # Atlas-specific options
    parser.add_argument(
        "--blockwise_coef",
        action="store_true",
        default=config.get("blockwise_coef", True),
        help="Learn per-block coefficients"
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Partition size for fine-grained coefficient learning"
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="ufm",
        help="Loss function (UFM for this script)"
    )
    parser.add_argument(
        "--lp_reg",
        type=int,
        default=None,
        choices=[1, 2],
        help="Regularization for learned coefficients"
    )
    
    # Other options
    parser.add_argument(
        "--epochs_per_task",
        type=int,
        default=config.get("epochs_per_task", 10),
        help="Default epochs per task"
    )
    parser.add_argument(
        "--config_tag",
        type=str,
        default=None,
        help="Custom tag for output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("seed", 1),
        help="Random seed"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print frequency during training"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/ufm_atlas",
        help="Directory for logs"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=config.get("cache_dir", None),
        help="Cache directory"
    )
    parser.add_argument(
        "--openclip_cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=config.get("world_size", 1),
        help="Number of processes for distributed training"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.get("port", 12355),
        help="Port for distributed training"
    )
    
    return parser


if __name__ == "__main__":
    # Load config file first
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_reverse.yaml")
    config = OmegaConf.load(config_path)
    
    # Create parser with config defaults
    parser = create_ufm_atlas_parser(config)
    args = parser.parse_args()
    if args.model == "ViT-B-16":
        args.batch_size = 64
    elif args.model == "ViT-L-14":
        args.batch_size = 32
    elif args.model == "ViT-B-32":
        args.batch_size = 16
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    # Set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model-specific adjustments
    if args.model == "ViT-L-14":
        if args.batch_size == config.get("batch_size", 128):  # If using default
            args.batch_size = 32
        if args.num_grad_accumulation == config.get("num_grad_accumulation", 1):
            args.num_grad_accumulation = 2
    
    # Expand paths
    args.data_location = os.path.expanduser(args.data_location)
    args.model_location = os.path.expanduser(args.model_location)
    
    # Setup save directory
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)
    
    # Initialize zeroshot accuracy dictionary
    if not hasattr(args, 'zs_acc'):
        args.zs_acc = {}
    
    # Leave-one-out dataset setup: use config.DATASETS_ALL and exclude test_dataset
    args.basis_datasets = [d for d in config.DATASETS_ALL if d != args.test_dataset]
    args.target_datasets = {args.test_dataset: args.epochs_per_task}
    
    # Setup logging directory
    args.logdir = os.path.join(args.logdir, args.model)
    if args.k > 0:
        args.logdir += f"/{args.k}shots"
    else:
        args.logdir += "/fullshot"
    if args.seed is not None:
        args.logdir += f"/{args.seed}"
    
    os.makedirs(args.logdir, exist_ok=True)
    
    # Legacy paths (kept for compatibility)
    args.head_path = os.path.join(args.logdir, "ufm_atlas_composition.pt")
    args.log_path = os.path.join(args.logdir, "ufm_atlas_composition.json")
    
    # Setup file logging (with timestamps for the log file)
    log_file_path = os.path.join(args.logdir, "ufm_atlas.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Get root logger and add file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Setup clean console logger
    logger = logging.getLogger(__name__ + "_main")
    logger.info(f"Leave-one-out mode: Test dataset = {args.test_dataset}")
    logger.info(f"Basis datasets: {len(args.basis_datasets)} datasets")
    
    print(args.batch_size)
    run_ufm_atlas(args)

