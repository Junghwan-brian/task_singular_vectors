"""UFM Baseline: Zeroshot and Linear Norm training for test-time adaptation.

This script evaluates:
1. Zeroshot performance (no training)
2. Linear Norm (LN) training (only layer normalization parameters)

Results are saved to checkpoints_tta directory.
"""

import os
import time
import json
import logging
import argparse
from omegaconf import OmegaConf, open_dict
import torch
import torchvision
from typing import Optional

from src.datasets import get_dataloader, maybe_dictionarize, get_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.utils import cosine_lr, load_checkpoint_safe
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from torch.cuda.amp import GradScaler

# Fix for H100 cuDNN compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_cudnn_sdp(False)
# torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


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


# Dataset-specific epochs for Linear Norm training
UFM_LN_EPOCHS_PER_DATASET = {
    "DTD": 1,
    "GTSRB": 1,
    "MNIST": 1,
    "SVHN": 1,
    "CIFAR10": 1,
    "CIFAR100": 1,
    "STL10": 1,
    "Food101": 1,
    "Flowers102": 1,
    "PCAM": 1,
    "OxfordIIITPet": 1,
    "RenderedSST2": 1,
    "EMNIST": 1,
    "FashionMNIST": 1,
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
    """UFM (Unsupervised FixMatch) loss with trusted samples."""
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
            result = data.copy()
            result["index"] = idx
            return result
        elif isinstance(data, tuple) and len(data) == 2:
            if isinstance(data[0], dict):
                result = data[0].copy()
                result["labels"] = data[1]
                result["index"] = idx
                return result
            else:
                return {"images": data[0], "labels": data[1], "index": idx}
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
    
    def __len__(self):
        return len(self.dataset)


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


def collate_fn(batch):
    """Custom collate function to handle nested dicts from TwoAsymetricTransform."""
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == "index":
            result[key] = torch.tensor([item[key] for item in batch])
        elif key in ["images", "images_"]:
            result[key] = torch.stack([item[key] for item in batch])
        elif key == "labels":
            labels = [item[key] for item in batch]
            if isinstance(labels[0], torch.Tensor):
                result[key] = torch.stack(labels)
            else:
                result[key] = torch.tensor(labels)
        else:
            try:
                result[key] = torch.stack([item[key] for item in batch])
            except:
                result[key] = [item[key] for item in batch]
    
    return result


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


def run_ufm_zeroshot(cfg) -> dict:
    """Evaluate zeroshot performance without any training."""
    logger = setup_simple_logger(__name__)
    
    logger.info("=" * 100)
    logger.info("UFM Baseline: Zeroshot Evaluation")
    logger.info("=" * 100)
    
    test_ds = cfg.test_dataset
    
    # Setup save_dir if not present (required by get_classification_head)
    with open_dict(cfg):
        if not hasattr(cfg, 'save_dir') or cfg.save_dir is None:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
    
    # Load encoder and head
    image_encoder = ImageEncoder(cfg.model).cuda()
    classification_head = get_classification_head(cfg, test_ds).cuda()
    
    # Prepare validation transform
    val_preprocess = image_encoder.val_preprocess
    
    # Load test dataset
    logger.info(f"Loading test dataset: {test_ds}")
    test_dataset = get_dataset(
        test_ds,
        val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    test_data = test_dataset.test_dataset if hasattr(test_dataset, 'test_dataset') else test_dataset
    
    # Prepare validation loader
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Evaluating zeroshot performance on {len(val_loader.dataset)} samples...")
    
    # Evaluate
    image_encoder.eval()
    classification_head.eval()
    
    overall_start = time.time()
    with torch.no_grad():
        metrics = evaluate_encoder_with_dataloader(
            image_encoder, classification_head, val_loader, cfg.device
        )
        zeroshot_acc = metrics['top1']
    
    eval_time = time.time() - overall_start
    
    logger.info(f"Zeroshot accuracy: {zeroshot_acc * 100:.2f}%")
    logger.info(f"Evaluation time: {eval_time:.2f}s")
    
    # Save results
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    config_tag = "ufm_zeroshot_0"
    
    save_dir = os.path.join(
        "./models/checkpoints_tta",
        cfg.model,
        test_ds + "Val",
        config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results JSON
    results_path = os.path.join(save_dir, "ufm_zeroshot_results_none.json")
    results = {
        "target_dataset": test_ds,
        "final_accuracy": float(zeroshot_acc),
        "k_shot": k,
        "model": cfg.model,
        "evaluation_time": eval_time,
        "config_tag": config_tag,
        "method": "ufm_zeroshot",
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved zeroshot results to {results_path}")
    
    return results


def run_ufm_ln(cfg) -> dict:
    """Train only Layer Normalization parameters using UFM."""
    logger = setup_simple_logger(__name__)
    
    logger.info("=" * 100)
    logger.info("UFM Baseline: Linear Norm (LN) Training")
    logger.info("=" * 100)
    
    # Auto-set epochs if not provided
    test_ds = cfg.test_dataset
    if test_ds in UFM_LN_EPOCHS_PER_DATASET and cfg.ln_epochs is None:
        cfg.ln_epochs = UFM_LN_EPOCHS_PER_DATASET[test_ds]
        logger.info(f"✓ Auto-set ln_epochs={cfg.ln_epochs} for {test_ds} (dataset-specific)")
    elif cfg.ln_epochs is None:
        cfg.ln_epochs = 1
        logger.info(f"Using default ln_epochs={cfg.ln_epochs}")
    
    # Setup save_dir if not present (required by get_classification_head)
    with open_dict(cfg):
        if not hasattr(cfg, 'save_dir') or cfg.save_dir is None:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
    
    # Load encoder and head
    image_encoder = ImageEncoder(cfg.model).cuda()
    classification_head = get_classification_head(cfg, test_ds).cuda()
    
    # Freeze all parameters except LayerNorm
    for name, param in image_encoder.named_parameters():
        if 'ln' in name.lower() or 'norm' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters (LayerNorm only): {trainable_params:,}")
    
    # Prepare validation transform
    val_preprocess = image_encoder.val_preprocess
    
    # Load test dataset
    logger.info(f"Loading test dataset: {test_ds}")
    test_dataset = get_dataset(
        test_ds,
        val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    test_data = test_dataset.test_dataset if hasattr(test_dataset, 'test_dataset') else test_dataset
    
    # Get predictions for trusted sample selection
    logger.info("Computing predictions for trusted sample selection...")
    model = ImageClassifier(image_encoder, classification_head)
    preds = get_preds(test_data, model, cfg.device)
    
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
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Prepare validation loader
    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val samples: {len(val_loader.dataset)}")
    
    # Setup optimizer and scheduler
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.ln_lr, weight_decay=cfg.ln_wd)
    
    num_batches = len(train_loader)
    total_steps = int(cfg.ln_epochs) * num_batches
    scheduler = cosine_lr(optimizer, cfg.ln_lr, 0, total_steps)
    
    loss_fn = ssl_loss_trusted
    
    # Training loop
    logger.info(f"Starting Linear Norm training for {cfg.ln_epochs} epochs...")
    overall_start = time.time()
    
    for epoch in range(int(cfg.ln_epochs)):
        image_encoder.train()
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader):
            step = epoch * num_batches + i
            
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            inputs_aug = batch["images_"].cuda()
            idx = batch["index"]
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features = image_encoder(inputs)
                logits = classification_head(features)
                
                features_aug = image_encoder(inputs_aug)
                logits_aug = classification_head(features_aug)
                
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
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} training time: {epoch_time:.2f}s")
    
    training_time = time.time() - overall_start
    
    # Final evaluation
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation...")
    logger.info("=" * 100 + "\n")
    
    image_encoder.eval()
    classification_head.eval()
    
    with torch.no_grad():
        final_metrics = evaluate_encoder_with_dataloader(
            image_encoder, classification_head, val_loader, cfg.device
        )
        final_acc = final_metrics['top1']
    
    logger.info(f"Final accuracy: {final_acc * 100:.2f}%")
    
    # Save results
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    config_tag = f"ufm_ln_{_sanitize_value(cfg.ln_lr)}_{cfg.ln_epochs}"
    
    save_dir = os.path.join(
        "./models/checkpoints_tta",
        cfg.model,
        test_ds + "Val",
        config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results JSON
    results_path = os.path.join(save_dir, "ufm_ln_results_none.json")
    results = {
        "target_dataset": test_ds,
        "final_accuracy": float(final_acc),
        "k_shot": k,
        "model": cfg.model,
        "ln_epochs": cfg.ln_epochs,
        "ln_lr": cfg.ln_lr,
        "ln_wd": cfg.ln_wd,
        "training_time": training_time,
        "trainable_params": trainable_params,
        "config_tag": config_tag,
        "method": "ufm_ln",
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved Linear Norm results to {results_path}")
    
    return results


def load_config(path: str):
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UFM Baseline: Zeroshot and Linear Norm training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Target dataset for evaluation"
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
    
    # Training hyperparameters for Linear Norm
    parser.add_argument(
        "--ln_epochs",
        type=int,
        help="Number of Linear Norm training epochs (auto-set per dataset if not provided)"
    )
    parser.add_argument(
        "--ln_lr",
        type=float,
        default=1e-3,
        help="Learning rate for Linear Norm optimization"
    )
    parser.add_argument(
        "--ln_wd",
        type=float,
        default=0.0,
        help="Weight decay for Linear Norm optimization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--k",
        type=int,
        dest="train_k",
        help="K-shot samples per class (0=fullshot)"
    )
    
    # Mode selection
    parser.add_argument(
        "--skip_zeroshot",
        action="store_true",
        help="Skip zeroshot evaluation"
    )
    parser.add_argument(
        "--skip_ln",
        action="store_true",
        help="Skip Linear Norm training"
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed"
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
    
    args = parser.parse_args()
    
    # Load config file
    cfg = load_config(args.config_file)
    
    # Merge CLI arguments
    cli_overrides = {k: v for k, v in vars(args).items() 
                     if v is not None and k not in ["config_file", "skip_zeroshot", "skip_ln"]}
    
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    
    # Validate required fields
    if not cfg.get("test_dataset"):
        parser.error("--test_dataset is required")
    
    # Model-specific adjustments
    if cfg.model == "ViT-B-16":
        cfg.batch_size = 64
    elif cfg.model == "ViT-L-14":
        cfg.batch_size = 32
    elif cfg.model == "ViT-B-32":
        cfg.batch_size = 16
    else:
        raise ValueError(f"Invalid model: {cfg.model}")
    
    # Set device
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Expand paths and setup save_dir
    cfg.data_location = os.path.expanduser(cfg.data_location)
    cfg.model_location = os.path.expanduser(cfg.model_location)
    
    # Setup save_dir (required by get_classification_head)
    if not hasattr(cfg, 'save_dir') or cfg.save_dir is None:
        cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
    
    OmegaConf.set_struct(cfg, True)
    
    # Run zeroshot evaluation
    if not args.skip_zeroshot:
        zeroshot_results = run_ufm_zeroshot(cfg)
        print(f"\n{'='*80}")
        print(f"Zeroshot Accuracy: {zeroshot_results['final_accuracy']*100:.2f}%")
        print(f"{'='*80}\n")
    
    # Run Linear Norm training
    if not args.skip_ln:
        ln_results = run_ufm_ln(cfg)
        print(f"\n{'='*80}")
        print(f"Linear Norm Accuracy: {ln_results['final_accuracy']*100:.2f}%")
        print(f"{'='*80}\n")
    
    print("✓ UFM Baseline complete!")

