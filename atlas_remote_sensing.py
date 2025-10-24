"""Learn the coefficients on task vectors for remote sensing datasets
under the few-shot setting and find the optimal combination.

Adapted from atlas.py for remote sensing datasets.
Uses get_remote_sensing_dataset() instead of get_dataset().
"""

import os
import argparse
import sys
import time
import json
import copy
import torch
import torchvision
import logging
import subprocess
from omegaconf import OmegaConf

from torch.cuda.amp import GradScaler
from atlas_src.modeling import ImageEncoder, ImageClassifier
from atlas_src.composition import WeightedImageEncoder
from atlas_src.utils import TIPWrapper, LPPWrapper

# Task vectors from energy environment
from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import (
    get_zeroshot_path,
    get_finetuned_path,
)

# Utils
from src.utils.utils import cosine_lr
from src.datasets.common import get_dataloader, maybe_dictionarize

# Remote sensing specific imports
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
    REMOTE_SENSING_DATASETS,
    sample_k_shot_indices,
)

# Evaluation function
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_atlas_config_tag(num_basis: int, args) -> str:
    count_part = _sanitize_value(max(num_basis, 0))
    k_part = _sanitize_value(getattr(args, 'k', 'na'))
    lr_part = _sanitize_value(getattr(args, 'lr', 'na'))
    return f"atlas_{count_part}_{k_part}_{lr_part}"

# Dataset-specific epochs for Atlas training (matching fine-tuning epochs)
ATLAS_EPOCHS_PER_DATASET = {
    "AID": 10,              # ~10,000 train samples, 600x600
    "CLRS": 10,             # ~30,000 train samples, 256x256
    "EuroSAT_RGB": 15,      # ~21,600 train samples, 64x64
    "MLRSNet": 15,          # ~17,000 train samples, 256x256
    "NWPU-RESISC45": 15,    # ~25,200 train samples, 256x256
    "Optimal-31": 50,       # ~6,200 train samples, 256x256
    "PatternNet": 20,       # ~10,000 train samples, 256x256
    "RS_C11": 60,           # ~5,000 train samples, 512x512
    "RSD46-WHU": 20,        # ~10,000 train samples, 256x256
    "RSI-CB128": 15,        # ~18,000 train samples, 128x128
    "RSSCN7": 80,           # ~2,800 train samples, 400x400
    "SAT-4": 5,             # ~60,000 train samples, 28x28
    "SAT-6": 10,            # ~40,000 train samples, 28x28
    "SIRI-WHU": 100,        # ~2,400 train samples, 200x200
    "UC_Merced": 100,       # ~2,100 train samples, 256x256
    "WHU-RS19": 150,        # ~1,000 train samples, 600x600
}
# ATLAS_EPOCHS_PER_DATASET = {
#     "AID": 5,              # ~10,000 train samples, 600x600
#     "CLRS": 5,             # ~30,000 train samples, 256x256
#     "EuroSAT_RGB": 5,      # ~21,600 train samples, 64x64
#     "MLRSNet": 5,          # ~17,000 train samples, 256x256
#     "NWPU-RESISC45": 5,    # ~25,200 train samples, 256x256
#     "Optimal-31": 5,       # ~6,200 train samples, 256x256
#     "PatternNet": 5,       # ~10,000 train samples, 256x256
#     "RS_C11": 5,           # ~5,000 train samples, 512x512
#     "RSD46-WHU": 5,        # ~10,000 train samples, 256x256
#     "RSI-CB128": 5,        # ~18,000 train samples, 128x128
#     "RSSCN7": 5,           # ~2,800 train samples, 400x400
#     "SAT-4": 5,             # ~60,000 train samples, 28x28
#     "SIRI-WHU": 5,        # ~2,400 train samples, 200x200
#     "UC_Merced": 5,       # ~2,100 train samples, 256x256
#     "WHU-RS19": 5,        # ~1,000 train samples, 600x600
# }


def compute_eval_epochs(total_epochs: int, max_evals: int = 5) -> set:
    total_epochs = max(int(total_epochs), 1)
    if total_epochs <= max_evals:
        return set(range(total_epochs))
    eval_epochs = {
        min(total_epochs - 1, int(round(i * (total_epochs - 1) / (max_evals - 1))))
        for i in range(max_evals)
    }
    return eval_epochs


class ValidationRecorder:
    """Utility to collect validation metrics without duplicating closure logic."""

    def __init__(self, start_time: float, val_history: list):
        self.start_time = float(start_time)
        self.val_history = val_history
        self.eval_counter = 0

    def __call__(self, stage: str, epoch_value, accuracy_value):
        record = {
            "stage": stage,
            "epoch": int(epoch_value),
            "accuracy": float(accuracy_value),
            "elapsed_seconds": float(time.time() - self.start_time),
            "evaluation_index": int(self.eval_counter),
        }
        self.val_history.append(record)
        self.eval_counter += 1
        return record


def evaluate_adapter_model(adapter_model, dataloader, device: str) -> float:
    """Compute top-1 accuracy for the adapter-enhanced classifier."""
    adapter_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            logits = adapter_model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def train_adapter_remote(
    model: ImageClassifier,
    base_train_loader,
    val_loader,
    args,
    logger,
    save_dir: str,
):
    """Optional TIP / LP++ adapter fine-tuning for remote sensing Atlas."""

    adapter_choice = getattr(args, "adapter", None)
    if not adapter_choice:
        return None

    adapter_choice = adapter_choice.lower()
    if adapter_choice in ("", "none"):
        return None

    internal_choice = adapter_choice
    if adapter_choice == "lp++":
        internal_choice = "lpp"
    display_choice = "lp++" if internal_choice == "lpp" else internal_choice

    if internal_choice not in {"tip", "lpp"}:
        logger.warning(
            f"[adapter:{adapter_choice}] Unsupported adapter requested; skipping."
        )
        return None

    device = next(model.parameters()).device

    base_dataset = base_train_loader.dataset
    batch_size = getattr(base_train_loader, "batch_size", args.batch_size)
    num_workers = getattr(base_train_loader, "num_workers", 0)
    pin_memory = getattr(base_train_loader, "pin_memory", True)

    cache_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    adapter_train_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"[adapter:{display_choice}] Building feature cache for adapter warm-up...")
    features_cache = []
    labels_cache = []
    model.eval()
    with torch.no_grad():
        for batch in cache_loader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            logits, feats = model(inputs, return_features=True)
            features_cache.append(feats.detach().cpu())
            labels = batch["labels"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels_cache.append(labels.detach().cpu().long())

    if not features_cache:
        logger.warning("[adapter] Training dataset is empty; skipping adapter training.")
        return None

    features_cache = torch.cat(features_cache, dim=0)
    labels_cache = torch.cat(labels_cache, dim=0).long()
    model.train()

    shots_value = None
    if internal_choice == "lpp":
        shots = getattr(args, "k", None)
        if shots is None or shots <= 0:
            shots = 100
        shots_value = int(shots)
        logger.info(f"[adapter:lp++] Initializing LP++ with shots={shots_value}")
        adapter_model = LPPWrapper(model, features_cache, labels_cache, shots_value)
        adapter_lr = float(getattr(adapter_model, "lr_temp", args.lr))
        adapter_epochs = 5
    else:
        adapter_model = TIPWrapper(model, features_cache, labels_cache)
        adapter_lr = 1e-3
        adapter_epochs = 5

    adapter_model = adapter_model.to(device)

    params = [p for p in adapter_model.parameters() if p.requires_grad]
    if not params:
        logger.warning("[adapter] No trainable parameters found; skipping adapter training.")
        return None

    num_batches = len(adapter_train_loader)
    if num_batches == 0:
        logger.warning("[adapter] No batches available; skipping adapter training.")
        return None

    grad_accum = max(1, getattr(args, "num_grad_accumulation", 1))
    total_scheduler_steps = max(1, adapter_epochs * num_batches // grad_accum)
    optimizer = torch.optim.AdamW(params, lr=adapter_lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, adapter_lr, 0, total_scheduler_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    eval_epochs = compute_eval_epochs(adapter_epochs)
    loss_history = []
    val_history = []
    epoch_loss_history = []

    train_start = time.time()
    record_validation = ValidationRecorder(train_start, val_history)

    initial_acc = evaluate_adapter_model(adapter_model, val_loader, device)
    record_validation("initial", -1, initial_acc)
    logger.info(
        f"[adapter:{display_choice}] Initial accuracy {initial_acc * 100:.2f}%"
    )
    best_acc = initial_acc
    best_state = copy.deepcopy(adapter_model.state_dict())

    step = 0
    for epoch in range(adapter_epochs):
        adapter_model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for i, batch in enumerate(adapter_train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"]
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            logits = adapter_model(inputs)
            loss = loss_fn(logits, labels)
            epoch_loss_sum += float(loss.item())
            epoch_loss_count += 1

            loss_history.append(
                {
                    "epoch": int(epoch),
                    "iteration": int(i),
                    "global_step": int(epoch * num_batches + i),
                    "loss": float(loss.item()),
                }
            )

            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        epoch_loss_history.append(avg_epoch_loss)
        logger.info(
            f"[adapter:{display_choice}] epoch {epoch}: train loss {avg_epoch_loss:.6f}"
        )

        if epoch in eval_epochs:
            acc = evaluate_adapter_model(adapter_model, val_loader, device)
            record_validation("epoch", epoch, acc)
            logger.info(
                f"[adapter:{display_choice}] epoch {epoch}: accuracy {acc * 100:.2f}%"
            )
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(adapter_model.state_dict())

    adapter_model.load_state_dict(best_state)
    final_acc = evaluate_adapter_model(adapter_model, val_loader, device)
    record_validation("final", adapter_epochs, final_acc)
    training_time = time.time() - train_start
    logger.info(
        f"[adapter:{display_choice}] Final accuracy {final_acc * 100:.2f}% (best {best_acc * 100:.2f}%)"
    )

    summary = {
        "adapter_type": display_choice,
        "adapter_internal_type": internal_choice,
        "epochs": int(adapter_epochs),
        "learning_rate": float(adapter_lr),
        "training_time": training_time,
        "final_accuracy": float(final_acc),
        "best_val_accuracy": float(best_acc),
        "loss_history": loss_history,
        "epoch_loss_history": epoch_loss_history,
        "validation_history": val_history,
        "evaluation_schedule": [int(ep) for ep in sorted(eval_epochs)],
    }
    if shots_value is not None:
        summary["shots"] = shots_value

    model.train()

    return summary


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_single(args):
    """Entry point for single-GPU atlas training."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    logger.info(f"Adapter option: {getattr(args, 'adapter_display', 'none')}")

    if not hasattr(args, 'model_location') or args.model_location is None:
        args.model_location = os.path.expanduser("./models/checkpoints_remote_sensing")
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)

    test_ds = getattr(args, 'test_dataset', None)
    if hasattr(args, 'basis_datasets') and args.basis_datasets:
        pool = list(args.basis_datasets)
        logger.info(f"Using {len(pool)} explicitly specified basis datasets")
    elif test_ds and test_ds in REMOTE_SENSING_DATASETS:
        pool = [d for d in REMOTE_SENSING_DATASETS.keys() if d != test_ds]
        logger.info(f"Leave-one-out mode: using {len(pool)} datasets as basis (excluding {test_ds})")
    else:
        pool = list(REMOTE_SENSING_DATASETS.keys())
        logger.info(f"Using all {len(pool)} remote sensing datasets as basis")

    args.basis_datasets_list = pool
    comp_acc = {}

    if hasattr(args, 'target_datasets') and args.target_datasets:
        target_list = list(args.target_datasets.items())
    elif test_ds:
        default_epochs = getattr(args, 'epochs_per_task', 10)
        target_list = [(test_ds, default_epochs)]
    else:
        logger.error("No target dataset specified. Use --test_dataset or --datasets")
        return

    logger.info(f"Target datasets: {[d for d, _ in target_list]}")

    for dataset, epochs in target_list:
        args.target_dataset = dataset + "Val"
        if dataset in ATLAS_EPOCHS_PER_DATASET:
            args.epochs = ATLAS_EPOCHS_PER_DATASET[dataset]
            logger.info(f"✓ Auto-set epochs={args.epochs} for {dataset} (dataset-specific)")
        else:
            args.epochs = epochs
            logger.info(f"Using default epochs={args.epochs} for {dataset}")

        zs_json_path = os.path.join(args.save_dir, f"{dataset}Val", "zeroshot_accuracies.json")
        if os.path.isfile(zs_json_path):
            with open(zs_json_path, 'r') as f:
                args.zs_acc = json.load(f)
            comp_acc[f"{dataset}Val_zeroshot"] = args.zs_acc.get(f"{dataset}Val", 0.0)
        else:
            if not hasattr(args, 'zs_acc'):
                args.zs_acc = {}

        k = getattr(args, 'k', 0)
        data_amount = f"{k} shots per class" if k > 0 else "full dataset"
        logger.info("=" * 100)
        logger.info(f"Learning task vector coefficients on {dataset} with {args.model} - {data_amount}")
        logger.info("=" * 100)

        comp_acc = train_single_task(args, comp_acc, logger)


def train_single_task(args, comp_acc=None, logger=None):
    """Train atlas coefficients on remote sensing dataset"""

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

    if comp_acc is None:
        comp_acc = {}
    
    target_dataset = args.target_dataset

    logger.info(f"Loading task vectors for {len(args.basis_datasets_list)} basis datasets...")

    task_vectors = {}
    for dataset in args.basis_datasets_list:
        # Use Val suffix for consistency with fine-tuning
        dataset_val = dataset + "Val"
        pretrained_checkpoint_path = get_zeroshot_path(
            args.model_location, dataset_val, args.model)
        finetuned_checkpoint_path = get_finetuned_path(
            args.model_location, dataset_val, args.model)
        
        if os.path.exists(pretrained_checkpoint_path) and os.path.exists(finetuned_checkpoint_path):
            pretrained_state = torch.load(pretrained_checkpoint_path, map_location="cpu")
            finetuned_state = torch.load(finetuned_checkpoint_path, map_location="cpu")

            task_vectors[dataset] = NonLinearTaskVector(
                args.model, pretrained_state, finetuned_state)

            logger.info(f"✓ Loaded task vector for {dataset}")
        else:
            logger.warning(f"✗ Missing checkpoints for {dataset}")

    if not task_vectors:
        logger.error("No task vectors loaded; aborting.")
        return comp_acc or {}

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task (leave-one-out)
    available_task_vectors = [
        v for k, v in task_vectors.items() if orig_dataset != k]
    expected_basis_count = len(getattr(args, "basis_datasets_list", []))
    actual_basis_count = len(available_task_vectors)
    if expected_basis_count and actual_basis_count != expected_basis_count:
        logger.warning(
            f"Loaded {actual_basis_count} of {expected_basis_count} requested task vectors "
            "because some checkpoints were missing."
        )
    
    logger.info(f"Using {len(available_task_vectors)} task vectors for composition")
    logger.info(f"Target dataset: {orig_dataset} (held out)")

    config_tag = getattr(args, 'config_tag', None)
    if not config_tag:
        basis_count_for_tag = expected_basis_count or actual_basis_count
        config_tag = build_atlas_config_tag(basis_count_for_tag, args)
        args.config_tag = config_tag
    logger.info(f"Using config tag: {config_tag}")

    # Create WeightedImageEncoder with task vectors
    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, available_task_vectors, 
        blockwise=args.blockwise_coef, 
        partition=args.partition,
    )

    # TIP's more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), 
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])

    # Load remote sensing dataset
    train_dataset = get_remote_sensing_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Get classification head for remote sensing dataset
    classification_head = get_remote_sensing_classification_head(args, target_dataset, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    eval_dataset = get_remote_sensing_dataset(
        target_dataset,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Few-shot sampling using unified k-shot function
    k = getattr(args, 'k', 0)
    logger.info("Using full training dataset (no k-shot sampling)" if k <= 0 else f"Applying k-shot sampling: {k} samples per class")
    if k > 0:
        selected_indices = sample_k_shot_indices(train_dataset, k, seed=0, verbose=True)
        base_dataset = getattr(train_dataset, "train_dataset", train_dataset)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_dataset, selected_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
        )
    else:
        train_loader = train_dataset.train_loader

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 80)
    logger.info(f"Trainable parameters (atlas): {trainable_params:,}")
    logger.info(f"Number of task vectors: {len(available_task_vectors)}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_batches = len(train_loader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()
    overall_start = time.time()

    eval_epochs = compute_eval_epochs(args.epochs)
    loss_history = []
    val_history = []
    record_validation = ValidationRecorder(overall_start, val_history)

    # Load validation dataset using get_dataloader (unified evaluation approach)
    val_loader = get_dataloader(eval_dataset, is_train=False, args=args, image_encoder=None)
    
    # Evaluate zeroshot accuracy using unified evaluation function
    image_encoder.eval()
    classification_head.eval()
    pretrained_metrics = evaluate_encoder_with_dataloader(
        image_encoder, classification_head, val_loader, 'cuda')
    pretrained_acc = pretrained_metrics['top1']
    comp_acc[f"{target_dataset}_zeroshot"] = pretrained_acc
    args.zs_acc[f"{target_dataset}"] = pretrained_acc
    logger.info(
        f"=> Zero-shot accuracy on {target_dataset}:\t{100*pretrained_acc:.2f}%.")

    record_validation("pretrained", -2, pretrained_acc)
    record_validation("zeroshot", -1, pretrained_acc)

    image_encoder.train()
    classification_head.train()

    best_coef = model.image_encoder.coef.data.clone()
    best_acc = pretrained_acc
    train_start = time.time()

    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step = epoch * num_batches + i
                loss_history.append(
                    {
                        "epoch": int(epoch),
                        "iteration": int(i),
                        "global_step": int(global_step),
                        "loss": float(loss.item()),
                    }
                )

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
            ):
                percent_complete = 100 * (i + 1) / len(train_loader)
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

        # Evaluate after selected epochs using unified evaluation function
        if epoch in eval_epochs:
            image_encoder = model.image_encoder
            coef = model.image_encoder.coef
            
            image_encoder.eval()
            classification_head.eval()
            
            # Use unified evaluation function
            metrics = evaluate_encoder_with_dataloader(
                image_encoder, classification_head, val_loader, 'cuda')
            acc = metrics['top1']
            
            # Set back to train mode
            image_encoder.train()
            classification_head.train()
            
            logger.info(f"Epoch {epoch}: Accuracy = {100*acc:.2f}%")
            record_validation("epoch", epoch, acc)
            
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()
                logger.info(f"✓ New best accuracy: {100*best_acc:.2f}%")

    training_time = time.time() - train_start
    comp_acc[target_dataset] = best_acc
    target_dataset_clean = target_dataset.replace("Val", "")
    image_encoder = model.image_encoder
    image_encoder.coef = torch.nn.Parameter(best_coef)

    # Final evaluation using unified evaluation function
    image_encoder.eval()
    classification_head.eval()

    final_metrics = evaluate_encoder_with_dataloader(
        image_encoder, classification_head, val_loader, 'cuda')
    final_acc = final_metrics['top1']

    comp_acc[target_dataset_clean] = final_acc

    logger.info(f"Final test accuracy: {100*final_acc:.2f}%")
    record_validation("final", args.epochs, final_acc)

    # Save coefficients to dataset-specific directory
    k = getattr(args, 'k', 0)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    save_dir = os.path.join(
        args.model_location,
        args.model,
        target_dataset,
        config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)

    atlas_path = os.path.join(save_dir, "atlas.pt")
    torch.save(best_coef, atlas_path)
    logger.info(f"✓ Saved learned atlas coefficients to {atlas_path}")

    adapter_summary = train_adapter_remote(model, train_loader, val_loader, args, logger, save_dir)
    if adapter_summary:
        adapter_results_path = os.path.join(
            save_dir,
            f"atlas_adapter_{adapter_summary['adapter_type']}.json",
        )
        adapter_summary_with_path = dict(adapter_summary)
        adapter_summary_with_path["results_path"] = adapter_results_path
        with open(adapter_results_path, "w") as f:
            json.dump(adapter_summary_with_path, f, indent=4)
        adapter_summary = adapter_summary_with_path
        comp_acc[f"{target_dataset_clean}_{adapter_summary['adapter_type']}"] = adapter_summary["final_accuracy"]
        logger.info(
            f"[adapter:{adapter_summary['adapter_type']}] Saved adapter results to {adapter_results_path}"
        )

    log_path = os.path.join(save_dir, "atlas_results.json")
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")
    result_log = {
        "target_dataset": target_dataset_clean,
        "final_accuracy": final_acc,
        "best_val_accuracy": best_acc,
        "k_shot": k,
        "model": args.model,
        "epochs": args.epochs,
        "training_time": training_time,
        "trainable_params": trainable_params,
        "batch_size": args.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "validation_history": val_history,
        "evaluation_schedule": [int(ep) for ep in sorted(eval_epochs)],
        "pretrained_accuracy": float(pretrained_acc),
        "zeroshot_accuracy": float(pretrained_acc),
        "config_tag": config_tag,
        "adapter_choice": ("none" if not getattr(args, "adapter", None) else (
            "lp++" if args.adapter == "lpp" else args.adapter
        )),
        "adapter_results": adapter_summary,
    }
    with open(log_path, 'w') as f:
        json.dump(result_log, f, indent=4)
    logger.info(f"✓ Saved results to {log_path}")

    return comp_acc


if __name__ == "__main__":
    
    # Load config first to get default values
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_remote_sensing.yaml")
    config = OmegaConf.load(config_path)
    
    # Get default model from config
    default_model = config.get("model", "ViT-B-32")

    # Lightweight argparse wrapper to override key hyperparameters via CLI
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--model", type=str, default=default_model,
                         help=f"Model architecture (default: {default_model} from config)")
    wrapper.add_argument("--batch_size", type=int, default=None)
    wrapper.add_argument("--lr", type=float, default=None)
    wrapper.add_argument("--wd", type=float, default=None)
    wrapper.add_argument("--epochs", type=int, default=None)
    wrapper.add_argument("--k", type=int, default=0,
                         help="k-shot per class (e.g., 16 = 16 samples per class). "
                              "Set to 0 to use full dataset (default: 16)")
    wrapper.add_argument("--config_tag", type=str, default=None,
                         help="Optional tag to group outputs for this configuration")
    wrapper.add_argument("--data_location", type=str, default="./datasets")
    wrapper.add_argument("--model_location", type=str, default="./models/checkpoints_remote_sensing")
    wrapper.add_argument("--seed", type=int, default=1)
    wrapper.add_argument("--print_every", type=int, default=10)
    
    # Dataset controls (similar to energy_train_remote_sensing.py)
    wrapper.add_argument("--test_dataset", type=str, required=True,
                         help="Dataset to treat as target (leave-one-out with others as basis)")
    wrapper.add_argument("--epochs_per_task", type=int, default=10)
    
    # Atlas-specific options
    wrapper.add_argument("--blockwise_coef", action="store_true", default=True,
                         help="Learn coefficient per parameter block")
    wrapper.add_argument("--partition", type=int, default=None,
                         help="Partition size for fine-grained coefficient learning")
    wrapper.add_argument("--adapter", type=str, default="none",
                         choices=["none", "tip", "lp++"],
                         help="Optionally train an adapter (TIP or LP++) after learning atlas coefficients")

    cli_args, unknown = wrapper.parse_known_args()
    unknown = list(unknown)
    adapter_option = (cli_args.adapter or "none").lower()
    if adapter_option != "none":
        passthrough = "lpp" if adapter_option == "lp++" else adapter_option
        unknown.extend(["--adapter", passthrough])

    # Import atlas args parser for remaining arguments
    from atlas_src.args import parse_arguments
    sys.argv = [sys.argv[0]] + unknown
    args = parse_arguments()

    # Overlay CLI overrides
    if cli_args.model is not None:
        args.model = cli_args.model
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.lr is not None:
        args.lr = cli_args.lr
    if cli_args.wd is not None:
        args.wd = cli_args.wd
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.data_location is not None:
        args.data_location = cli_args.data_location
    if cli_args.model_location is not None:
        args.model_location = cli_args.model_location
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.print_every is not None:
        args.print_every = cli_args.print_every
    if cli_args.blockwise_coef is not None:
        args.blockwise_coef = cli_args.blockwise_coef
    if cli_args.partition is not None:
        args.partition = cli_args.partition
    if cli_args.config_tag is not None:
        args.config_tag = cli_args.config_tag
    if getattr(args, "adapter", None):
        args.adapter_display = "lp++" if args.adapter == "lpp" else args.adapter
    else:
        args.adapter_display = "none"

    # Set k-shot parameter
    if cli_args.k is not None:
        args.k = cli_args.k
    else:
        args.k = 16  # default: 16-shot

    # Default hyperparameters (following atlas.py)
    args.lr = 1e-1
    args.batch_size = 32 if args.model == "ViT-L-14" else 512
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.epochs_per_task = cli_args.epochs_per_task
    

    logger = logging.getLogger(__name__)
    args.test_dataset = cli_args.test_dataset
    args.basis_datasets = [d for d in config.DATASETS_ALL if d != cli_args.test_dataset]
    args.target_datasets = {cli_args.test_dataset: args.epochs_per_task}
    logger.info(f"Leave-one-out mode: Test dataset = {args.test_dataset}")
    logger.info(f"Basis datasets: {len(args.basis_datasets)} datasets")

    # Setup logging directory (legacy - results now saved per-dataset)
    if not hasattr(args, 'logdir'):
        args.logdir = os.path.join("logs", "atlas_remote_sensing", args.model)
    else:
        args.logdir += f"/{args.model}"
    
    if args.k > 0:
        args.logdir += f"/{args.k}shots"
    else:
        args.logdir += "/fullshot"

    if args.seed is not None:
        args.logdir += f"/{args.seed}"

    # Legacy paths (kept for compatibility, actual results saved per-dataset)
    args.head_path = os.path.join(args.logdir, "learned_composition_remote_sensing.pt")
    args.log_path = os.path.join(args.logdir, "learned_composition_remote_sensing.json")

    os.makedirs(args.logdir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.logdir, "atlas_remote_sensing.log")),
        ]
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger(__name__).addHandler(console_handler)

    run_single(args)
