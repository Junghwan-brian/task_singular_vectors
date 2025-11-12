"""
ImageNet Atlas training with OOD evaluation.
Task vector composition on ImageNet-1k validation set with OOD evaluation.
"""

import os
import time
import json
import logging
import argparse
import copy
from typing import Optional

import torch
import torchvision
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf, open_dict

from atlas_src.modeling import ImageEncoder, ImageClassifier
from atlas_src.composition import WeightedImageEncoder
from atlas_src.utils import TIPWrapper, LPPWrapper

from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import (
    get_zeroshot_path,
    get_finetuned_path,
    ALL_DATASETS,
)
from src.utils.utils import cosine_lr, load_checkpoint_safe
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models import get_classification_head
from src.datasets.remote_sensing import sample_k_shot_indices
from src.datasets.imagenet_ood import ImageNetILSVRCVal
from src.eval.eval import eval_single_dataset
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_atlas_config_tag(num_basis: int, args) -> str:
    count_part = _sanitize_value(max(num_basis, 0))
    lr_part = _sanitize_value(getattr(args, 'lr', 'na'))
    k_part = _sanitize_value(getattr(args, 'k', 0))
    return f"imagenet_atlas_{count_part}_{lr_part}_{k_part}shot"


class ValidationRecorder:
    """Utility to collect validation metrics."""
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


def train_adapter(
    model: ImageClassifier,
    base_train_loader,
    val_loader,
    args,
    logger,
    save_dir: str,
):
    """Optional TIP / LP++ adapter fine-tuning for ImageNet Atlas."""
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
        logger.warning(f"[adapter:{adapter_choice}] Unsupported adapter; skipping.")
        return None

    device = next(model.parameters()).device

    base_dataset = base_train_loader.dataset
    batch_size = getattr(base_train_loader, "batch_size", args.batch_size)
    num_workers = getattr(base_train_loader, "num_workers", 2)
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

    logger.info(f"[adapter:{display_choice}] Building feature cache...")
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
        logger.warning("[adapter] Training dataset is empty; skipping.")
        return None

    features_cache = torch.cat(features_cache, dim=0)
    labels_cache = torch.cat(labels_cache, dim=0).long()
    model.train()

    shots_value = None
    if internal_choice == "lpp":
        shots = getattr(args, "k", None)
        shots_value = int(shots)
        logger.info(f"[adapter:lp++] Initializing LP++ with shots={shots_value}")
        adapter_model = LPPWrapper(model, features_cache, labels_cache, shots_value)
        adapter_lr = float(getattr(adapter_model, "lr_temp", args.lr))
        adapter_epochs = 20
    else:
        adapter_model = TIPWrapper(model, features_cache, labels_cache)
        adapter_lr = 1e-3
        adapter_epochs = 20

    adapter_model = adapter_model.to(device)

    params = [p for p in adapter_model.parameters() if p.requires_grad]
    if not params:
        logger.warning("[adapter] No trainable parameters; skipping.")
        return None

    trainable_param_count = sum(p.numel() for p in params)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    num_batches = len(adapter_train_loader)
    if num_batches == 0:
        logger.warning("[adapter] No batches available; skipping.")
        return None

    grad_accum = max(1, getattr(args, "num_grad_accumulation", 1))
    total_scheduler_steps = max(1, adapter_epochs * num_batches // grad_accum)
    optimizer = torch.optim.AdamW(params, lr=adapter_lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, adapter_lr, 0, total_scheduler_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_history = []
    val_history = []
    epoch_loss_history = []

    train_start = time.time()
    record_validation = ValidationRecorder(train_start, val_history)

    initial_acc = evaluate_adapter_model(adapter_model, val_loader, device)
    record_validation("initial", -1, initial_acc)
    logger.info(f"[adapter:{display_choice}] Initial accuracy {initial_acc * 100:.2f}%")
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

            loss_history.append({
                "epoch": int(epoch),
                "iteration": int(i),
                "global_step": int(epoch * num_batches + i),
                "loss": float(loss.item()),
            })

            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        epoch_loss_history.append(avg_epoch_loss)
        logger.info(f"[adapter:{display_choice}] epoch {epoch}: train loss {avg_epoch_loss:.6f}")

    adapter_model.load_state_dict(best_state)
    final_acc = evaluate_adapter_model(adapter_model, val_loader, device)
    record_validation("final", adapter_epochs, final_acc)
    training_time = time.time() - train_start
    logger.info(f"[adapter:{display_choice}] Final accuracy {final_acc * 100:.2f}%")

    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        try:
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

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
        "trainable_params": int(trainable_param_count),
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
    }
    if shots_value is not None:
        summary["shots"] = shots_value

    model.train()

    return summary


def run_imagenet_atlas(args):
    """Entry point for ImageNet atlas training with OOD evaluation."""
    logger = setup_logger(__name__)
    
    if not hasattr(args, 'model_location') or args.model_location is None:
        args.model_location = os.path.expanduser("./models/checkpoints")
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)

    # Load basis task vectors (leave-one-out from general datasets)
    if hasattr(args, 'basis_datasets') and args.basis_datasets:
        pool = list(args.basis_datasets)
        logger.info(f"Using {len(pool)} explicitly specified basis datasets")
    else:
        pool = ALL_DATASETS[:args.num_tasks]
        logger.info(f"Using {len(pool)} general datasets as basis")

    args.basis_datasets_list = pool

    test_ds = "ImageNetILSVRC"
    val_dataset_name = test_ds + "Val"
    k = getattr(args, 'k', 0)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    logger.info("=" * 100)
    logger.info(f"Learning task vector coefficients on ImageNet with {args.model}")
    logger.info("=" * 100)

    logger.info(f"Loading task vectors for {len(args.basis_datasets_list)} basis datasets...")

    # Load fine-tuned checkpoints for all basis datasets
    ft_checks = {}
    for dataset in args.basis_datasets_list:
        dataset_val = dataset + "Val"
        finetuned_checkpoint_path = get_finetuned_path(
            args.model_location, dataset_val, args.model)
        
        if os.path.exists(finetuned_checkpoint_path):
            ft_checks[dataset] = load_checkpoint_safe(finetuned_checkpoint_path, map_location="cpu")
            logger.info(f"✓ Loaded fine-tuned checkpoint for {dataset}")
        else:
            logger.warning(f"✗ Missing fine-tuned checkpoint for {dataset}")

    # Load zeroshot checkpoint
    first_dataset_val = args.basis_datasets_list[0] + "Val" if args.basis_datasets_list else "dummy"
    zeroshot_path = get_zeroshot_path(args.model_location, first_dataset_val, args.model)
    
    logger.info(f"Loading shared zeroshot model from: {zeroshot_path}")
    ptm_check = load_checkpoint_safe(zeroshot_path, map_location="cpu")

    # Create task vectors
    task_vectors = {}
    for dataset, ft_check in ft_checks.items():
        task_vectors[dataset] = NonLinearTaskVector(args.model, ptm_check, ft_check)
        logger.info(f"✓ Created task vector for {dataset}")

    if not task_vectors:
        logger.error("No task vectors loaded; aborting.")
        return

    available_task_vectors = list(task_vectors.values())
    logger.info(f"Using {len(available_task_vectors)} task vectors for composition")

    config_tag = getattr(args, 'config_tag', None)
    if not config_tag:
        config_tag = build_atlas_config_tag(len(available_task_vectors), args)
        args.config_tag = config_tag
    logger.info(f"Using config tag: {config_tag}")

    # Create WeightedImageEncoder with task vectors
    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, available_task_vectors,
        blockwise=args.blockwise_coef,
        partition=args.partition,
    )

    # TIP's aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])

    # Load ImageNet datasets
    logger.info(f"Loading ImageNet-1k validation dataset: {val_dataset_name}")
    _train_val_obj = ImageNetILSVRCVal(
        preprocess_fn, location=args.data_location, batch_size=args.batch_size,
        num_workers=int(getattr(args, "num_workers", 2)))
    base_train_dataset = _train_val_obj.test_dataset

    _eval_val_obj = ImageNetILSVRCVal(
        image_encoder.val_preprocess, location=args.data_location,
        batch_size=args.batch_size, num_workers=int(getattr(args, "num_workers", 2)))
    base_val_dataset = _eval_val_obj.test_dataset

    classification_head = get_classification_head(args, test_ds)
    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    def _build_loader(dataset, is_train: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=is_train,
            num_workers=int(getattr(args, "num_workers", 2)),
            pin_memory=True,
        )

    train_loader = _build_loader(base_train_dataset, is_train=True)
    val_loader = _build_loader(base_val_dataset, is_train=False)

    # k-shot sampling
    if k > 0:
        logger.info(f"Applying k-shot sampling: {k} samples per class")
        try:
            seed = int(getattr(args, 'seed', 1))
            
            def _save_k_shot_indices(indices, save_dir, dataset_name, kshot, seed):
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"k_shot_indices_k{kshot}_seed{seed}.json")
                with open(path, "w") as f:
                    json.dump({"indices": indices, "dataset": dataset_name, 
                              "k": int(kshot), "seed": int(seed)}, f)
                return path

            def _load_k_shot_indices(save_dir, kshot, seed):
                path = os.path.join(save_dir, f"k_shot_indices_k{kshot}_seed{seed}.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                    return data.get("indices", None)
                return None

            def _subsample_from_larger_k(larger_indices, dataset, target_k):
                import numpy as np
                labels = []
                if hasattr(dataset, "targets"):
                    tgt = dataset.targets
                    if torch.is_tensor(tgt):
                        tgt = tgt.cpu().numpy()
                    elif not isinstance(tgt, np.ndarray):
                        tgt = np.array(tgt)
                    for idx in larger_indices:
                        labels.append(int(tgt[idx]))
                else:
                    for idx in larger_indices:
                        _, label = dataset[idx]
                        if torch.is_tensor(label):
                            label = int(label.item())
                        labels.append(int(label))
                
                class_to_indices = {}
                for idx, lab in zip(larger_indices, labels):
                    class_to_indices.setdefault(int(lab), []).append(idx)
                selected = []
                for lab in sorted(class_to_indices.keys()):
                    selected.extend(class_to_indices[lab][:target_k])
                return selected

            indices_dir = os.path.join(args.model_location, args.model, val_dataset_name)
            selected_indices = _load_k_shot_indices(indices_dir, k, seed)

            if selected_indices is None:
                larger_k = 16
                if k < larger_k:
                    larger = _load_k_shot_indices(indices_dir, larger_k, seed)
                    if larger is not None:
                        logger.info(f"✓ Subsampling from {larger_k}-shot to {k}-shot")
                        selected_indices = _subsample_from_larger_k(larger, base_train_dataset, k)
                        _save_k_shot_indices(selected_indices, indices_dir, val_dataset_name, k, seed)
            if selected_indices is None:
                logger.info(f"Sampling new {k}-shot indices (seed={seed})")
                selected_indices = sample_k_shot_indices(
                    base_train_dataset, k, seed=seed, verbose=True, 
                    progress_desc=f"ImageNet {k}-shot"
                )
                _save_k_shot_indices(selected_indices, indices_dir, val_dataset_name, k, seed)

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(base_train_dataset, selected_indices),
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(getattr(args, "num_workers", 2)),
                pin_memory=True,
            )
            logger.info(f"✓ Created {k}-shot train loader with {len(selected_indices)} samples")

            all_indices = set(range(len(base_val_dataset)))
            comp_indices = sorted(all_indices.difference(set(selected_indices)))
            if len(comp_indices) == 0:
                raise RuntimeError("Validation set is empty after k-shot split")
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(base_val_dataset, comp_indices),
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(getattr(args, "num_workers", 2)),
                pin_memory=True,
            )
        except Exception as e:
            logger.error(f"Failed to apply k-shot sampling: {e}")
            logger.warning("Falling back to full training set")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 80)
    logger.info(f"Trainable parameters (atlas): {trainable_params:,}")
    logger.info(f"Number of task vectors: {len(available_task_vectors)}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_batches = len(train_loader)
    print_every = max(int(num_batches / 10), 1)

    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()
    overall_start = time.time()

    loss_history = []
    val_history = []
    record_validation = ValidationRecorder(overall_start, val_history)

    image_encoder.train()
    classification_head = model.classification_head
    classification_head.train()

    epoch_times = []
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
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
                loss_history.append({
                    "epoch": int(epoch),
                    "iteration": int(i),
                    "global_step": int(global_step),
                    "loss": float(loss.item()),
                })

            batch_time = time.time() - start_time

            if (step % print_every == 0 and ((i + 1) % args.num_grad_accumulation == 0)):
                percent_complete = 100 * (i + 1) / len(train_loader)
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )

        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)
        logger.info(f"Epoch {epoch} training time: {epoch_train_time:.2f}s")

    # Final evaluation on ImageNet validation set
    image_encoder.eval()
    classification_head.eval()

    final_metrics = evaluate_encoder_with_dataloader(
        image_encoder, classification_head, val_loader, args.device
    )
    final_acc = final_metrics['top1']

    logger.info(f"Final validation accuracy: {100*final_acc:.2f}%")
    record_validation("final", args.epochs, final_acc)

    # OOD evaluations
    ood_results = {}
    ood_list = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2MFVal"]
    logger.info("\n" + "=" * 100)
    logger.info("Evaluating on OOD datasets...")
    logger.info("=" * 100 + "\n")
    
    image_encoder.eval()
    with torch.no_grad():
        for ood_name in ood_list:
            m = eval_single_dataset(image_encoder, ood_name, args)
            ood_results[ood_name] = float(m.get("top1", 0.0))
            logger.info(f"OOD {ood_name}: {100.0 * ood_results[ood_name]:.2f}%")
            record_validation(f"ood:{ood_name}", int(args.epochs), ood_results[ood_name])

    # Compose unified evaluation results
    all_eval_accuracies = {"ImageNetILSVRC": float(final_acc)}
    for k_name, v_acc in ood_results.items():
        all_eval_accuracies[k_name] = float(v_acc)

    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")

    # Save coefficients
    save_dir = os.path.join(
        args.model_location,
        args.model,
        val_dataset_name,
        config_tag,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)

    atlas_path = os.path.join(save_dir, "atlas.pt")
    final_coef = model.image_encoder.coef.data.clone()
    torch.save(final_coef, atlas_path)
    logger.info(f"✓ Saved learned atlas coefficients to {atlas_path}")

    adapter_result_tag = "none"
    adapter_choice_value = "none"
    adapter_summary = train_adapter(model, train_loader, val_loader, args, logger, save_dir)
    if adapter_summary:
        adapter_type = adapter_summary.get("adapter_type", "none")
        adapter_result_tag = adapter_type.lower().replace("++", "pp")
    else:
        adapter_summary = None

    log_path = os.path.join(save_dir, f"atlas_results_imagenet.json")
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")
    
    result_log = {
        "target_dataset": test_ds,
        "final_accuracy": final_acc,
        "ood_accuracies": ood_results,
        "all_eval_accuracies": all_eval_accuracies,
        "k_shot": k,
        "model": args.model,
        "epochs": args.epochs,
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "batch_size": args.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "validation_history": val_history,
        "config_tag": config_tag,
        "adapter_choice": adapter_choice_value,
        "adapter_results": adapter_summary,
    }
    with open(log_path, 'w') as f:
        json.dump(result_log, f, indent=4)
    logger.info(f"✓ Saved results to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet Atlas training with OOD evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument("--config_file", type=str,
                        default="config/config_reverse.yaml", help="Path to configuration YAML file")

    # Model and paths
    parser.add_argument("--model", type=str, default="ViT-B-32", help="Model architecture")
    parser.add_argument("--data_location", type=str, default="./datasets",
                        help="Root directory for datasets")
    parser.add_argument("--model_location", type=str, default="./models/checkpoints",
                        help="Directory for model checkpoints")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_grad_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--k", type=int, default=0, help="K-shot samples per class (0=fullshot)")

    # Atlas-specific options
    parser.add_argument("--blockwise_coef", action="store_true", default=True,
                        help="Learn per-block coefficients")
    parser.add_argument("--partition", type=int, default=None,
                        help="Partition size for fine-grained coefficient learning")

    # Adapter options
    parser.add_argument("--adapter", type=str, default="none",
                        choices=["none", "tip", "lp++"], help="Optional adapter after atlas training")

    # Other
    parser.add_argument("--num_tasks", type=int, default=17,
                        help="How many basis tasks to use from ALL_DATASETS")
    parser.add_argument("--config_tag", type=str, default=None, help="Custom tag for output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gpu", type=int, help="GPU id (overrides --device as cuda:{id})")

    args = parser.parse_args()

    # Load config if exists
    if os.path.exists(args.config_file):
        config = OmegaConf.load(args.config_file)
        # Merge CLI args with config (CLI takes precedence)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None}
        config.update(cli_dict)
        args = argparse.Namespace(**OmegaConf.to_container(config))

    # Set device
    if getattr(args, "gpu", None) is not None:
        try:
            gpu_id = int(args.gpu)
            args.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize adapter choice
    adapter_choice = args.adapter.lower()
    if adapter_choice == "lp++":
        args.adapter = "lpp"
    args.adapter_display = "lp++" if args.adapter == "lpp" else args.adapter

    # Model-specific adjustments
    if args.model == "ViT-L-14":
        if args.batch_size == 128:
            args.batch_size = 32
        if args.num_grad_accumulation == 1:
            args.num_grad_accumulation = 2

    # Expand paths
    args.data_location = os.path.expanduser(args.data_location)
    args.model_location = os.path.expanduser(args.model_location)

    # Setup save directory
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)

    # Setup logging
    logger = setup_logger(__name__)
    logger.info(f"ImageNet Atlas training with {args.model}")
    logger.info(f"Using {args.num_tasks} basis task vectors")

    run_imagenet_atlas(args)


if __name__ == "__main__":
    main()

