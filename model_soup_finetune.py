import os
import sys
import time
import json
import math
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torchvision

from src.utils.variables_and_paths import ALL_DATASETS, get_finetuned_path
from src.datasets import get_dataset, get_dataloader, maybe_dictionarize
from src.models import ImageEncoder, ImageClassifier, get_classification_head
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from src.datasets.remote_sensing import sample_k_shot_indices
from src.utils.utils import load_checkpoint_safe, cosine_lr


def setup_simple_logger(name: str = __name__) -> logging.Logger:
    """Setup a clean logger with minimal formatting (no INFO/DEBUG prefixes)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return "".join(
        ch if (str.isalnum(ch) or ch in {"-", "_"}) else "_"
        for ch in str(val).replace(".", "p")
    )


def build_soupft_config_tag(cfg) -> str:
    num_models = _sanitize_value(getattr(cfg, "num_soup_models", 0))
    lr = _sanitize_value(getattr(cfg, "ft_lr", 1e-5))
    epochs = _sanitize_value(getattr(cfg, "ft_epochs", 20))
    wd = _sanitize_value(getattr(cfg, "ft_wd", 0.0))
    warmup = _sanitize_value(getattr(cfg, "warmup_ratio", 0.1))
    # Example: baseline_soupft_n16_1ep5_20_0_0p1
    return f"baseline_soupft_n{num_models}_{lr}_{epochs}_{wd}_{warmup}"


def save_k_shot_indices(indices, save_dir, dataset_name, k, seed):
    """Save k-shot indices to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    with open(indices_path, "w") as f:
        json.dump({"indices": indices, "dataset": dataset_name, "k": k, "seed": seed}, f)
    return indices_path


def load_k_shot_indices(save_dir, k, seed):
    """Load k-shot indices from a JSON file if it exists."""
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    if os.path.exists(indices_path):
        with open(indices_path, "r") as f:
            data = json.load(f)
            return data["indices"]
    return None


def subsample_from_larger_k(larger_indices, dataset, target_k: int, seed: int):
    """
    Subsample target_k indices per class from a larger k-shot set.
    Uses deterministic selection (first target_k samples per class).
    """
    import numpy as np
    from torch.utils.data import Subset

    _ = seed  # deterministic strategy doesn't need RNG; keep signature for compatibility
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    labels = []
    if hasattr(base_dataset, "targets"):
        all_targets = base_dataset.targets
        if torch.is_tensor(all_targets):
            all_targets = all_targets.cpu().numpy()
        elif not isinstance(all_targets, np.ndarray):
            all_targets = np.array(all_targets)
        labels = [int(all_targets[idx]) for idx in larger_indices]
    elif hasattr(base_dataset, "data") and hasattr(base_dataset.data, "iloc"):
        try:
            all_targets = base_dataset.data["target"].values
            labels = [int(all_targets[idx]) - 1 for idx in larger_indices]
        except Exception:
            labels = []
    elif hasattr(base_dataset, "samples") and base_dataset.samples is not None:
        try:
            all_samples = base_dataset.samples
            labels = [int(all_samples[idx][1]) for idx in larger_indices]
        except Exception:
            labels = []
    elif hasattr(base_dataset, "_labels"):
        all_labels = base_dataset._labels
        if torch.is_tensor(all_labels):
            all_labels = all_labels.cpu().numpy()
        elif not isinstance(all_labels, np.ndarray):
            all_labels = np.array(all_labels)
        labels = [int(all_labels[idx]) for idx in larger_indices]

    if not labels:
        for idx in larger_indices:
            _, label = base_dataset[idx]
            if torch.is_tensor(label):
                label = label.item()
            labels.append(int(label))

    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in zip(larger_indices, labels):
        class_to_indices.setdefault(int(label), []).append(int(idx))

    selected_indices = []
    for label in sorted(class_to_indices.keys()):
        selected_indices.extend(class_to_indices[label][: int(target_k)])
    return selected_indices


def _set_seed(seed: int):
    try:
        import random
        import numpy as np

        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _state_dict_from_checkpoint(checkpoint: dict) -> dict:
    """
    Unwrap common checkpoint formats into a plain state_dict.
    - Already a state_dict: returns as-is
    - Dict with 'state_dict' key: returns that
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


def average_state_dicts(
    state_dicts: List[dict],
    reference_keys: Optional[set] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Element-wise average of state_dict tensors.

    - Averages only keys common to all state_dicts (and optionally in reference_keys).
    - Averages only floating tensors. Non-floating tensors are copied from the first dict.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")

    key_sets = [set(sd.keys()) for sd in state_dicts]
    common_keys = set.intersection(*key_sets)
    if reference_keys is not None:
        common_keys = common_keys.intersection(set(reference_keys))

    # Deterministic ordering for reproducibility
    common_keys = sorted(common_keys)

    out = OrderedDict()
    skipped_shape = 0
    for key in common_keys:
        tensors = [sd[key] for sd in state_dicts]
        first = tensors[0]
        if not torch.is_tensor(first):
            out[key] = first
            continue
        # Require same shape across all
        shapes_ok = all(torch.is_tensor(t) and tuple(t.shape) == tuple(first.shape) for t in tensors)
        if not shapes_ok:
            skipped_shape += 1
            continue

        if first.is_floating_point():
            acc = torch.zeros_like(first, dtype=torch.float32)
            for t in tensors:
                acc += t.detach().to(dtype=torch.float32)
            acc /= float(len(tensors))
            out[key] = acc.to(dtype=first.dtype)
        else:
            # e.g., num_batches_tracked (int64) -> copy
            out[key] = first

    if skipped_shape > 0:
        logger.info(f"[soup] Skipped {skipped_shape} keys due to shape mismatch.")
    logger.info(f"[soup] Averaged {len(out)} keys (common across {len(state_dicts)} checkpoints).")
    return out


def run_soup_finetune(cfg: DictConfig) -> None:
    logger = setup_simple_logger(__name__)

    # Resolve device
    with open_dict(cfg):
        if not getattr(cfg, "device", None):
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(cfg.device) == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"

        if not getattr(cfg, "data_location", None):
            cfg.data_location = "./datasets"
        cfg.data_location = os.path.expanduser(cfg.data_location)

        if not getattr(cfg, "model_location", None):
            cfg.model_location = "./models/checkpoints"
        cfg.model_location = os.path.expanduser(cfg.model_location)

        if not getattr(cfg, "ft_epochs", None):
            cfg.ft_epochs = 20
        if not getattr(cfg, "ft_lr", None):
            cfg.ft_lr = 1e-5
        if not getattr(cfg, "ft_wd", None):
            cfg.ft_wd = 0.0
        if not getattr(cfg, "warmup_ratio", None):
            cfg.warmup_ratio = 0.1

        # Ensure few-shot param name matches energy_train_reverse.py convention
        if getattr(cfg, "train_k", None) is None:
            cfg.train_k = 0

        # Default save_dir convention (same root as other scripts)
        if not getattr(cfg, "save_dir", None):
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)

    OmegaConf.set_struct(cfg, True)
    _set_seed(int(getattr(cfg, "seed", 1)))

    target_ds = str(cfg.target_dataset)
    val_dataset_name = target_ds + "Val"
    k = int(getattr(cfg, "train_k", 0))
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    logger.info("=" * 100)
    logger.info("Model Soup + Finetune (baseline)")
    logger.info("=" * 100)
    logger.info(f"Target dataset: {target_ds}")
    logger.info(f"Model: {cfg.model}")
    logger.info(f"K-shot: {k}")
    logger.info(f"Seed: {getattr(cfg, 'seed', 1)}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Data location: {cfg.data_location}")
    logger.info(f"Model location: {cfg.model_location}")
    logger.info("=" * 100)

    # Determine soup source datasets: use cfg.DATASETS_ALL if present; otherwise fallback
    if hasattr(cfg, "DATASETS_ALL") and cfg.DATASETS_ALL:
        datasets_all = list(cfg.DATASETS_ALL)
    else:
        datasets_all = list(ALL_DATASETS)

    soup_datasets = [d for d in datasets_all if d != target_ds]
    soup_datasets_val = [d + "Val" for d in soup_datasets]

    # Load fine-tuned checkpoints for soup
    logger.info("[soup] Loading fine-tuned checkpoints:")
    ft_state_dicts: List[dict] = []
    ft_paths: List[str] = []
    for ds_val in soup_datasets_val:
        path = get_finetuned_path(cfg.model_location, ds_val, model=cfg.model)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
        ft_paths.append(path)
        ckpt = load_checkpoint_safe(path, map_location="cpu")
        ft_state_dicts.append(_state_dict_from_checkpoint(ckpt))
        logger.info(f"✓ {path}")

    # Build averaged encoder weights (only keys that match the encoder)
    ref_encoder = ImageEncoder(cfg.model)
    ref_keys = set(ref_encoder.state_dict().keys())
    avg_sd = average_state_dicts(ft_state_dicts, reference_keys=ref_keys, logger=logger)

    with open_dict(cfg):
        cfg.num_soup_models = int(len(ft_state_dicts))
        if not getattr(cfg, "config_tag", None):
            cfg.config_tag = build_soupft_config_tag(cfg)
    OmegaConf.set_struct(cfg, True)

    # Prepare output dirs
    config_dir = os.path.join(cfg.save_dir, val_dataset_name, cfg.config_tag)
    result_dir = os.path.join(config_dir, shot_folder)
    os.makedirs(result_dir, exist_ok=True)

    soup_ckpt_path = os.path.join(result_dir, "soup_encoder.pt")
    torch.save(avg_sd, soup_ckpt_path)
    logger.info(f"[soup] Saved averaged encoder to {soup_ckpt_path}")

    # Create model initialized with soup weights
    device = torch.device(str(cfg.device))
    image_encoder = ImageEncoder(cfg.model)
    missing, unexpected = image_encoder.load_state_dict(avg_sd, strict=False)
    if missing:
        logger.info(f"[soup] load_state_dict missing keys: {len(missing)}")
    if unexpected:
        logger.info(f"[soup] load_state_dict unexpected keys: {len(unexpected)}")

    image_encoder = image_encoder.to(device)
    classification_head = get_classification_head(cfg, target_ds).to(device)
    model = ImageClassifier(image_encoder, classification_head).to(device)

    # Data loading (match energy/baselines)
    train_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        + image_encoder.train_preprocess.transforms[-3:]
    )

    logger.info(f"Loading dataset for training: {val_dataset_name}")
    dataset_train = get_dataset(
        target_ds,
        train_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    train_loader = get_dataloader(dataset_train, is_train=True, args=cfg, image_encoder=None)

    # Apply k-shot sampling if specified (same file naming convention as energy)
    if k > 0:
        logger.info(f"Applying k-shot sampling: {k} samples per class")
        try:
            seed = int(getattr(cfg, "seed", 1))
            indices_save_dir = os.path.join(cfg.model_location, cfg.model, val_dataset_name)

            selected_indices = load_k_shot_indices(indices_save_dir, k, seed)
            if selected_indices is not None:
                logger.info(f"✓ Loaded existing {k}-shot indices (seed={seed})")
            else:
                larger_k = 16
                if k < larger_k:
                    larger_indices = load_k_shot_indices(indices_save_dir, larger_k, seed)
                    if larger_indices is not None:
                        logger.info(
                            f"✓ Subsampling from existing {larger_k}-shot indices to {k}-shot"
                        )
                        base_ds = getattr(dataset_train, "train_dataset", dataset_train)
                        selected_indices = subsample_from_larger_k(larger_indices, base_ds, k, seed)
                        indices_path = save_k_shot_indices(
                            selected_indices, indices_save_dir, val_dataset_name, k, seed
                        )
                        logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
                    else:
                        logger.info(
                            f"Sampling new {k}-shot indices from full dataset (seed={seed})"
                        )
                        selected_indices = sample_k_shot_indices(
                            dataset_train,
                            k,
                            seed=seed,
                            verbose=True,
                            progress_desc=f"{target_ds} {k}-shot",
                        )
                        indices_path = save_k_shot_indices(
                            selected_indices, indices_save_dir, val_dataset_name, k, seed
                        )
                        logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
                else:
                    logger.info(f"Sampling new {k}-shot indices from full dataset (seed={seed})")
                    selected_indices = sample_k_shot_indices(
                        dataset_train,
                        k,
                        seed=seed,
                        verbose=True,
                        progress_desc=f"{target_ds} {k}-shot",
                    )
                    indices_path = save_k_shot_indices(
                        selected_indices, indices_save_dir, val_dataset_name, k, seed
                    )
                    logger.info(f"✓ Saved {k}-shot indices to {indices_path}")

            base_dataset = getattr(dataset_train, "train_dataset", None)
            if base_dataset is None:
                base_dataset = getattr(train_loader, "dataset", None)

            if base_dataset is not None:
                num_workers = 2
                collate_fn = getattr(train_loader, "collate_fn", None)
                train_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(base_dataset, selected_indices),
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                )
                logger.info(f"✓ Created {k}-shot dataloader with {len(selected_indices)} samples")
            else:
                logger.warning(
                    "Could not locate base train_dataset for k-shot subsetting; using full loader instead."
                )
        except Exception as e:
            logger.error(f"Failed to apply k-shot sampling: {e}")
            logger.warning("Falling back to full training set")

    logger.info(f"Loading validation dataset: {val_dataset_name}")
    val_dataset = get_dataset(
        target_ds,
        image_encoder.val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    val_loader = get_dataloader(val_dataset, is_train=False, args=cfg, image_encoder=None)
    logger.info(f"✓ Validation dataset loaded ({len(val_loader.dataset)} samples)")

    # Finetune (train all params)
    for p in model.parameters():
        p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(params, lr=float(cfg.ft_lr), weight_decay=float(cfg.ft_wd))
    num_batches = len(train_loader)
    total_steps = max(1, int(cfg.ft_epochs) * max(1, num_batches))
    warmup_steps = int(float(cfg.warmup_ratio) * total_steps)
    scheduler = cosine_lr(optimizer, float(cfg.ft_lr), warmup_steps, total_steps)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    loss_history = []
    epoch_times = []
    overall_start = time.time()

    logger.info(f"Starting finetuning for {int(cfg.ft_epochs)} epochs...")
    logger.info(
        f"Train dataset size: {len(train_loader.dataset)}, Batch size: {cfg.batch_size}, Steps/epoch: {num_batches}"
    )

    model.train()
    step = 0
    for epoch in range(int(cfg.ft_epochs)):
        epoch_start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler(step)
            step += 1

            loss_history.append({"epoch": int(epoch), "iteration": int(i), "loss": float(loss.item())})
            if i == 0:
                logger.info(
                    f"[soupft] epoch {epoch} {i + 1}/{max(1, num_batches)} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}"
                )

        epoch_times.append(time.time() - epoch_start)

    # Final evaluation
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 100 + "\n")

    model.eval()
    with torch.no_grad():
        final_metrics = evaluate_encoder_with_dataloader(
            model.image_encoder, model.classification_head, val_loader, str(cfg.device)
        )
        final_acc = float(final_metrics.get("top1", 0.0))

    logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")

    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    gpu_peak_mem_mb = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    finetuned_encoder_path = os.path.join(result_dir, "soupft_encoder.pt")
    torch.save(model.image_encoder.state_dict(), finetuned_encoder_path)
    logger.info(f"Saved finetuned encoder to {finetuned_encoder_path}")

    results_path = os.path.join(result_dir, "baseline_results_none.json")
    results = {
        "method": "model_soup_finetune",
        "target_dataset": target_ds,
        "final_accuracy": float(final_acc),
        "k_shot": int(k),
        "model": str(cfg.model),
        "ft_epochs": int(cfg.ft_epochs),
        "ft_lr": float(cfg.ft_lr),
        "ft_wd": float(cfg.ft_wd),
        "warmup_ratio": float(cfg.warmup_ratio),
        "training_time": float(min_epoch_time),
        "avg_epoch_time": float(avg_epoch_time),
        "all_epoch_times": epoch_times,
        "trainable_params": int(trainable_params),
        "batch_size": int(cfg.batch_size),
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "elapsed_seconds": float(time.time() - overall_start),
        "config_tag": str(cfg.config_tag),
        "num_soup_models": int(getattr(cfg, "num_soup_models", len(ft_state_dicts))),
        "soup_datasets": soup_datasets,
        "soup_checkpoint_paths": ft_paths,
        "soup_checkpoint_saved": soup_ckpt_path,
        "finetuned_encoder_saved": finetuned_encoder_path,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


if __name__ == "__main__":
    from src.datasets.registry import registry as DATASET_REGISTRY

    allowed_datasets = sorted([name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")])

    parser = argparse.ArgumentParser(
        description="Model soup (average task-finetuned weights) then finetune on target dataset (few-shot supported).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target_dataset", type=str, required=True, choices=allowed_datasets)
    parser.add_argument("--config_file", type=str, default="config/config_reverse.yaml")
    parser.add_argument("--model", type=str, help="Vision backbone (e.g., ViT-B-32)")

    # Few-shot (match energy_train_reverse.py arg name)
    parser.add_argument("--k", type=int, dest="train_k", default=16, help="K-shot samples per class (0=fullshot)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    # Finetune hyperparams
    parser.add_argument("--ft_epochs", type=int, default=20)
    parser.add_argument("--ft_lr", type=float, default=1e-5)
    parser.add_argument("--ft_wd", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)

    # Output tag override
    parser.add_argument("--config_tag", type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config_file)
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config_file"}
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))

    if not cfg.get("target_dataset"):
        parser.error("--target_dataset is required")
    if not cfg.get("model"):
        parser.error("--model is required (or set it in config)")

    OmegaConf.set_struct(cfg, True)
    run_soup_finetune(cfg)

