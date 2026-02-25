"""
ImageNet baseline training with OOD evaluation.
Supports Linear Probe, TIP Adapter, LP++, and LoRA on ImageNet-1k validation set.
"""

import os
import time
import json
import logging
import argparse
import math
from typing import Optional, Dict, List

import torch
import torchvision
from omegaconf import OmegaConf, open_dict

from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.eval.eval import eval_single_dataset
from src.datasets.imagenet_ood import ImageNetILSVRCVal
from src.datasets.common import maybe_dictionarize
from src.datasets.remote_sensing import sample_k_shot_indices
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from atlas_src.utils import TIPWrapper, LPPWrapper


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


def build_baseline_config_tag(cfg) -> str:
    """Build config tag for baseline method."""
    method = cfg.baseline_method
    k_part = _sanitize_value(getattr(cfg, "train_k", 0))
    
    if method == 'linear_probe':
        lr = _sanitize_value(cfg.lp_lr)
        epochs = _sanitize_value(cfg.lp_epochs)
        wd = _sanitize_value(cfg.lp_wd)
        return f"imagenet_baseline_lp_{lr}_{epochs}_{wd}_{k_part}shot"
    
    elif method == 'tip_adapter':
        wd = _sanitize_value(cfg.adapter_wd)
        return f"imagenet_baseline_tip_{wd}_{k_part}shot"
    
    elif method == 'lp++':
        wd = _sanitize_value(cfg.adapter_wd)
        return f"imagenet_baseline_lpp_{wd}_{k_part}shot"
    
    elif method == 'lora':
        r = _sanitize_value(cfg.lora_r)
        alpha = _sanitize_value(cfg.lora_alpha)
        lr = _sanitize_value(cfg.lora_lr)
        epochs = _sanitize_value(cfg.lora_epochs)
        wd = _sanitize_value(cfg.lora_wd)
        return f"imagenet_baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}_{k_part}shot"
    
    else:
        return f"imagenet_baseline_{method}_{k_part}shot"


class LoRALinear(torch.nn.Module):
    def __init__(self, base_linear: torch.nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.r = int(max(1, r))
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_A = torch.nn.Parameter(torch.zeros(in_f, self.r))
        self.lora_B = torch.nn.Parameter(torch.zeros(self.r, out_f))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.dropout = torch.nn.Dropout(
            p=float(dropout)) if dropout and dropout > 0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.linear(x, self.base.weight, self.base.bias)
        lora_delta = self.dropout(x) @ self.lora_A @ self.lora_B
        return y + self.scaling * lora_delta

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def apply_lora_to_module(module: torch.nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
        else:
            apply_lora_to_module(child, r=r, alpha=alpha, dropout=dropout)


def cache_features(ddp_model, ddp_loader, device):
    all_features, all_labels, all_indexes, all_logits = [], [], [], []
    ddp_model = ddp_model.to(device)
    ddp_model.eval()
    with torch.no_grad():
        for batch in ddp_loader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            logits, features = ddp_model(inputs, return_features=True)
            labels = batch["labels"]
            all_features.append(features.detach().cpu())
            all_labels.append(labels)
            if "index" in batch:
                all_indexes.append(batch["index"])
            else:
                if len(all_indexes) == 0:
                    all_indexes.append(torch.arange(len(inputs)))
                else:
                    start = int(torch.cat(all_indexes).numel())
                    all_indexes.append(torch.arange(start, start + len(inputs)))
            all_logits.append(logits.detach().cpu())

    logits_cache = torch.cat(all_logits)
    features_cache = torch.cat(all_features)
    labels = torch.cat(all_labels)
    indexes = torch.cat(all_indexes)
    indexes_to_i = {indexes[i].item(): i for i in range(len(indexes))}
    return logits_cache, features_cache, labels, indexes_to_i


class ReturnFeaturesClassifier(torch.nn.Module):
    def __init__(self, base: ImageClassifier):
        super().__init__()
        self.image_encoder = base.image_encoder
        self.classification_head = base.classification_head
        self.train_preprocess = getattr(base, "train_preprocess", None)
        self.val_preprocess = getattr(base, "val_preprocess", None)

    def forward(self, inputs, return_features: bool = False):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        if return_features:
            return outputs, features / (features.norm(dim=-1, keepdim=True) + 1e-12)
        return outputs

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)


def get_ood_dataloader(dataset_name: str, preprocess, cfg):
    """Create OOD dataloader for evaluation"""
    from src.datasets.registry import get_dataset
    from src.datasets.common import get_dataloader
    
    dataset = get_dataset(
        dataset_name,
        preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    return get_dataloader(dataset, is_train=False, args=cfg, image_encoder=None)


def evaluate_adapter_model(adapter_model: torch.nn.Module, dataloader, device) -> dict:
    """Evaluate adapter model on a dataloader"""
    adapter_model.eval()
    top1, correct, n = 0.0, 0.0, 0.0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            logits = adapter_model(x)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
    top1 = correct / max(1, n)
    return {"top1": top1}


def train_linear_probe(model, train_loader, val_loader, cfg, logger):
    """Train linear probe with OOD evaluation"""
    image_encoder = model.image_encoder
    for p in image_encoder.parameters():
        p.requires_grad = False
    model.classification_head.weight.requires_grad_(True)
    model.classification_head.bias.requires_grad_(True)

    params = [p for p in model.classification_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.lp_lr), weight_decay=float(cfg.lp_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.lp_lr_step_size), gamma=float(cfg.lp_lr_gamma))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = cfg.device
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_batches = len(train_loader)
    loss_history = []
    epoch_times = []
    
    logger.info(f"Starting linear probe training for {cfg.lp_epochs} epochs...")
    
    for epoch in range(int(cfg.lp_epochs)):
        epoch_start = time.time()
        model.train()
        
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            loss_history.append({
                "epoch": int(epoch),
                "iteration": int(i),
                "loss": float(loss.item()),
            })

            if i == 0:
                logger.info(
                    f"[linear_probe] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)

    # Final evaluation on ImageNet validation set
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 100 + "\n")
    
    model.eval()
    with torch.no_grad():
        final_metrics = evaluate_encoder_with_dataloader(
            model.image_encoder, model.classification_head, val_loader, cfg.device
        )
        final_acc = final_metrics['top1']
    
    logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")
    
    # OOD evaluations - use fine-tuned classification head
    ood_results = {}
    ood_list = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2MFVal"]
    logger.info("\n" + "=" * 100)
    logger.info("Evaluating on OOD datasets...")
    logger.info("=" * 100 + "\n")
    
    model.eval()
    with torch.no_grad():
        for ood_name in ood_list:
            ood_loader = get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)
            
            # Use evaluate_encoder_with_dataloader to ensure trained head is used
            ood_metrics = evaluate_encoder_with_dataloader(
                model.image_encoder, model.classification_head, ood_loader, cfg.device
            )
            ood_acc = ood_metrics['top1']
            ood_results[ood_name] = float(ood_acc)
            logger.info(f"OOD {ood_name}: {100.0 * ood_results[ood_name]:.2f}%")
    
    # Compose unified evaluation results
    all_eval_accuracies = {"ImageNetILSVRC": float(final_acc)}
    for k_name, v_acc in ood_results.items():
        all_eval_accuracies[k_name] = float(v_acc)
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")
    
    trainable_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable parameters: {trainable_params:,}")

    return {
        "method": "linear_probe",
        "final_accuracy": float(final_acc),
        "ood_accuracies": ood_results,
        "all_eval_accuracies": all_eval_accuracies,
        "epochs": cfg.lp_epochs,
        "lr": cfg.lp_lr,
        "wd": cfg.lp_wd,
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
    }


def train_tip_or_lpp(model, train_loader, val_loader, cfg, logger, adapter: str):
    """Train TIP or LP++ adapter with OOD evaluation"""
    ddp_model = ReturnFeaturesClassifier(model).to(cfg.device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Building feature cache...")
    cache_start = time.time()
    logits_cache, features_cache, labels, indexes_to_i = cache_features(
        ddp_model, train_loader, cfg.device)
    cache_time = time.time() - cache_start
    logger.info(f"Feature cache built in {cache_time:.2f}s")

    adapter_model = ddp_model
    if adapter == 'lpp':
        shots = int(cfg.train_k) if int(cfg.train_k) > 0 else 0
        logger.info(f"[adapter:lp++] Initializing LP++ with shots={shots}")
        adapter_model = LPPWrapper(adapter_model, features_cache, labels, shots)
        # Reduce epochs to prevent overfitting on few-shot data
        epochs = 50 if shots <= 4 else 100 if shots <= 8 else 200
        adapter_model = adapter_model.to(cfg.device)
        
        # Use much smaller learning rates to prevent instability
        lr_temp = float(getattr(adapter_model, 'lr_temp', 1e-1)) * 0.01  # 100x smaller
        lr_alpha = float(getattr(adapter_model, 'lr_alpha', 1e-3)) * 0.1  # 10x smaller
        
        logger.info(f"[adapter:lp++] Using stabilized learning rates: lr_temp={lr_temp:.6f}, lr_alpha={lr_alpha:.6f}, epochs={epochs}")
        logger.warning(f"[adapter:lp++] Original LP++ may not work well on ImageNet few-shot. Consider using TIP or Linear Probe instead.")
        
        param_groups = [
            {'params': adapter_model.adapter.parameters(), 'lr': lr_temp},
            {'params': [adapter_model.alpha_vec], 'lr': lr_alpha}
        ]
    elif adapter == 'tip':
        adapter_model = TIPWrapper(adapter_model, features_cache, labels)
        epochs = 20
        adapter_model = adapter_model.to(cfg.device)
        param_groups = [{'params': [adapter_model.beta_alpha], 'lr': 1e-3}]
    else:
        raise NotImplementedError(f"Adapter {adapter} unknown")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.adapter_wd)
    num_batches = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * num_batches)
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in adapter_model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable adapter parameters: {trainable_params:,}")

    loss_history = []
    epoch_times = []
    
    logger.info(f"Starting {adapter} adapter training for {epochs} epochs...")
    
    adapter_model.train()
    total_cache = int(logits_cache.size(0))
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader):
            step = epoch * max(1, num_batches) + i
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(cfg.device)

            if "index" in batch:
                ids = [indexes_to_i[j.item()] for j in batch['index']]
            else:
                start_id = int(step % max(1, total_cache))
                ids = [int((start_id + t) % max(1, total_cache))
                       for t in range(len(inputs))]

            l_cache, f_cache = logits_cache[ids].to(inputs), features_cache[ids].to(inputs)
            if adapter == 'lpp':
                logits = adapter_model(inputs)
            else:
                logits = adapter_model(inputs, l_cache, f_cache)
            labels_b = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            loss_history.append({
                "epoch": int(epoch),
                "iteration": int(i),
                "loss": float(loss.item()),
            })

            if i == 0:
                lrs = [f"{pg['lr']:.6f}" for pg in optimizer.param_groups]
                logger.info(
                    f"[adapter:{adapter}] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lrs {lrs}")

            scheduler.step()
        
        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)

    # Final evaluation on ImageNet validation set
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 100 + "\n")
    
    adapter_model.eval()
    with torch.no_grad():
        metrics = evaluate_adapter_model(adapter_model, val_loader, cfg.device)
        final_acc = metrics['top1']
    
    logger.info(f"Adapter '{adapter}' Acc: {final_acc*100:.2f}%")
    
    # OOD evaluations - use the adapted model (encoder + adapter)
    ood_results = {}
    ood_list = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2MFVal"]
    logger.info("\n" + "=" * 100)
    logger.info("Evaluating on OOD datasets...")
    logger.info("=" * 100 + "\n")
    
    adapter_model.eval()
    with torch.no_grad():
        for ood_name in ood_list:
            ood_loader = get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)
            
            # Evaluate using the trained adapter model
            ood_metrics = evaluate_adapter_model(adapter_model, ood_loader, cfg.device)
            ood_acc = ood_metrics['top1']
            ood_results[ood_name] = float(ood_acc)
            logger.info(f"OOD {ood_name}: {100.0 * ood_results[ood_name]:.2f}%")
    
    # Compose unified evaluation results
    all_eval_accuracies = {"ImageNetILSVRC": float(final_acc)}
    for k_name, v_acc in ood_results.items():
        all_eval_accuracies[k_name] = float(v_acc)
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

    return {
        "method": adapter,
        "final_accuracy": float(final_acc),
        "ood_accuracies": ood_results,
        "all_eval_accuracies": all_eval_accuracies,
        "epochs": epochs,
        "adapter_wd": cfg.adapter_wd,
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "cache_build_time": cache_time,
    }


def train_lora(model, train_loader, val_loader, cfg, logger):
    """Train LoRA with OOD evaluation"""
    apply_lora_to_module(model.image_encoder, r=int(cfg.lora_r), 
                        alpha=float(cfg.lora_alpha), dropout=float(cfg.lora_dropout))

    for p in model.image_encoder.parameters():
        if not isinstance(p, torch.nn.Parameter):
            continue
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            if hasattr(module, 'base'):
                module.base.weight.requires_grad_(False)
                if module.base.bias is not None:
                    module.base.bias.requires_grad_(False)
        elif isinstance(module, torch.nn.Linear):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

    if bool(getattr(cfg, 'lora_tune_head', True)):
        model.classification_head.weight.requires_grad_(True)
        model.classification_head.bias.requires_grad_(True)
    else:
        model.classification_head.weight.requires_grad_(False)
        model.classification_head.bias.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable LoRA parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(params, lr=float(cfg.lora_lr), weight_decay=float(cfg.lora_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.lora_lr_step_size), gamma=float(cfg.lora_lr_gamma))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = cfg.device
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    num_batches = len(train_loader)
    loss_history = []
    epoch_times = []
    
    logger.info(f"Starting LoRA training for {cfg.lora_epochs} epochs...")
    
    for epoch in range(int(cfg.lora_epochs)):
        epoch_start = time.time()
        model.train()
        
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            loss_history.append({
                "epoch": int(epoch),
                "iteration": int(i),
                "loss": float(loss.item()),
            })

            if i == 0:
                logger.info(
                    f"[lora] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        epoch_train_time = time.time() - epoch_start
        epoch_times.append(epoch_train_time)

    # Final evaluation on ImageNet validation set
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 100 + "\n")
    
    model.eval()
    with torch.no_grad():
        final_metrics = evaluate_encoder_with_dataloader(
            model.image_encoder, model.classification_head, val_loader, cfg.device
        )
        final_acc = final_metrics['top1']
    
    logger.info(f"LoRA validation accuracy: {final_acc*100:.2f}%")
    
    # OOD evaluations - use LoRA-adapted encoder
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
    all_eval_accuracies = {"ImageNetILSVRC": float(final_acc)}
    for k_name, v_acc in ood_results.items():
        all_eval_accuracies[k_name] = float(v_acc)
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

    return {
        "method": "lora",
        "final_accuracy": float(final_acc),
        "ood_accuracies": ood_results,
        "all_eval_accuracies": all_eval_accuracies,
        "epochs": cfg.lora_epochs,
        "lr": cfg.lora_lr,
        "wd": cfg.lora_wd,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_tune_head": getattr(cfg, 'lora_tune_head', True),
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
    }


def run_imagenet_baseline(cfg) -> None:
    logger = setup_logger(__name__)
    device = cfg.device

    with open_dict(cfg):
        if not cfg.get("config_tag"):
            cfg.config_tag = build_baseline_config_tag(cfg)

    test_ds = "ImageNetILSVRC"
    val_dataset_name = test_ds + "Val"
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    logger.info(f"Using config tag: {cfg.config_tag}")
    logger.info(f"Baseline method: {cfg.baseline_method}")
    logger.info(f"Model: {cfg.model}")
    logger.info(f"K-shot: {k}")

    with open_dict(cfg):
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)

    image_encoder = ImageEncoder(cfg.model).to(device)

    # Train/Val datasets from ImageNet-1k validation folder
    logger.info(f"Loading ImageNet-1k validation dataset for train/val split: {val_dataset_name}")
    train_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1.0), 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])

    _train_val_obj = ImageNetILSVRCVal(
        train_preprocess, location=cfg.data_location, batch_size=cfg.batch_size,
        num_workers=int(getattr(cfg, "num_workers", 2)))
    base_train_dataset = _train_val_obj.test_dataset

    _eval_val_obj = ImageNetILSVRCVal(
        ImageEncoder(cfg.model).val_preprocess, location=cfg.data_location,
        batch_size=cfg.batch_size, num_workers=int(getattr(cfg, "num_workers", 2)))
    base_val_dataset = _eval_val_obj.test_dataset

    classification_head = get_classification_head(cfg, test_ds)
    model = ImageClassifier(image_encoder, classification_head).to(device)
    model.freeze_head()

    def _build_loader(dataset, is_train: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=is_train,
            num_workers=int(getattr(cfg, "num_workers", 2)),
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
            if hasattr(dataset, "targets") and dataset.targets is not None:
                tgt = dataset.targets
                if torch.is_tensor(tgt):
                    tgt = tgt.cpu().numpy()
                elif not isinstance(tgt, np.ndarray):
                    tgt = np.array(tgt)
                for idx in indices:
                    labels.append(int(tgt[idx]))
                return labels
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
            logger.info(f"Sampling new {k}-shot indices (seed={seed})")
            selected_indices = sample_k_shot_indices(
                base_train_dataset, k, seed=seed, verbose=True, progress_desc=f"ImageNet {k}-shot"
            )
            saved_path = _save_k_shot_indices(selected_indices, indices_dir, val_dataset_name, k, seed)
            logger.info(f"✓ Saved k-shot indices to {saved_path}")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_train_dataset, selected_indices),
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=int(getattr(cfg, "num_workers", 2)),
            pin_memory=True,
        )
        logger.info(f"✓ Created {k}-shot train loader with {len(selected_indices)} samples")

        all_indices = set(range(len(base_val_dataset)))
        comp_indices = sorted(all_indices.difference(set(selected_indices)))
        if len(comp_indices) == 0:
            raise RuntimeError("Validation set is empty after k-shot split")
        val_full_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_val_dataset, comp_indices),
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(getattr(cfg, "num_workers", 2)),
            pin_memory=True,
        )

    # Save dirs
    config_dir = os.path.join(cfg.model_location, cfg.model, val_dataset_name, cfg.config_tag)
    os.makedirs(config_dir, exist_ok=True)
    save_dir = os.path.join(config_dir, shot_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, f"baseline_results_imagenet.json")
    
    # Check if eval_only mode
    if getattr(cfg, 'eval_only', False):
        logger.info("=" * 80)
        logger.info("EVAL ONLY MODE")
        logger.info("=" * 80)
        logger.warning("Baseline methods don't save model checkpoints.")
        logger.warning("Skipping re-evaluation. Results JSON already exists or needs retraining.")
        logger.info(f"Results path: {results_path}")
        if os.path.exists(results_path):
            logger.info("✓ Results file exists. No action needed.")
        else:
            logger.warning("✗ Results file not found. Please retrain without --eval_only flag.")
        return

    # Train baseline
    logger.info(f"\nTraining baseline '{cfg.baseline_method}' on ImageNet")
    if cfg.baseline_method == 'linear_probe':
        results = train_linear_probe(model, train_loader, val_full_loader, cfg, logger)
    elif cfg.baseline_method == 'tip_adapter':
        results = train_tip_or_lpp(model, train_loader, val_full_loader, cfg, logger, adapter='tip')
    elif cfg.baseline_method == 'lp++':
        results = train_tip_or_lpp(model, train_loader, val_full_loader, cfg, logger, adapter='lpp')
    elif cfg.baseline_method == 'lora':
        results = train_lora(model, train_loader, val_full_loader, cfg, logger)
    else:
        raise NotImplementedError(f"Unknown baseline method: {cfg.baseline_method}")

    # Save results JSON
    results.update({
        "target_dataset": test_ds,
        "k_shot": k,
        "model": cfg.model,
        "batch_size": cfg.batch_size,
        "config_tag": cfg.config_tag,
    })
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")

    logger.info("\n" + "=" * 100)
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {results['final_accuracy']*100:.2f}%")
    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet Baseline training with OOD evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config/model
    parser.add_argument("--config_file", type=str,
                        default="config/config_reverse.yaml", help="Path to configuration YAML file")
    parser.add_argument("--model", type=str, help="Vision backbone (e.g., ViT-B-32, ViT-B-16)")

    # Baseline method
    parser.add_argument("--baseline_method", type=str,
                        choices=["linear_probe", "tip_adapter", "lp++", "lora"],
                        default="linear_probe", help="Baseline to train")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--k", type=int, dest="train_k", help="K-shot samples per class (0=fullshot)")

    # Linear probe hparams
    parser.add_argument("--lp_epochs", type=int, help="Linear probe epochs")
    parser.add_argument("--lp_lr", type=float, help="Linear probe learning rate")
    parser.add_argument("--lp_wd", type=float, help="Linear probe weight decay")
    parser.add_argument("--lp_lr_step_size", type=int, help="Linear probe LR step size")
    parser.add_argument("--lp_lr_gamma", type=float, help="Linear probe LR gamma")

    # TIP/LPP hparams
    parser.add_argument("--adapter_wd", type=float, help="Adapter weight decay")

    # LoRA hparams
    parser.add_argument("--lora_epochs", type=int, help="LoRA epochs")
    parser.add_argument("--lora_lr", type=float, help="LoRA learning rate")
    parser.add_argument("--lora_wd", type=float, help="LoRA weight decay")
    parser.add_argument("--lora_lr_step_size", type=int, help="LoRA LR step size")
    parser.add_argument("--lora_lr_gamma", type=float, help="LoRA LR gamma")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--lora_tune_head", type=int, choices=[0, 1], help="Tune classification head")

    # Other
    parser.add_argument("--config_tag", type=str, help="Custom tag for output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for k-shot sampling")
    parser.add_argument("--data_location", type=str, default="./datasets",
                        help="Root dir containing imagenet")
    parser.add_argument("--model_location", type=str,
                        default="./models/checkpoints", help="Where checkpoints/heads are stored")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--gpu", type=int, help="GPU id (overrides --device as cuda:{id})")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing results without training")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    OmegaConf.set_struct(cfg, False)
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config_file"}
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    
    # Set defaults for optional fields
    if not cfg.get("train_k"):
        cfg.train_k = 0
    if not cfg.get("lp_epochs"):
        cfg.lp_epochs = 20
    if not cfg.get("lp_lr"):
        cfg.lp_lr = 1e-3
    if not cfg.get("lp_wd"):
        cfg.lp_wd = 0.0
    if not cfg.get("lp_lr_step_size"):
        cfg.lp_lr_step_size = 1
    if not cfg.get("lp_lr_gamma"):
        cfg.lp_lr_gamma = 0.5
    if not cfg.get("adapter_wd"):
        cfg.adapter_wd = 0.0
    if not cfg.get("lora_epochs"):
        cfg.lora_epochs = 20
    if not cfg.get("lora_lr"):
        cfg.lora_lr = 1e-4
    if not cfg.get("lora_wd"):
        cfg.lora_wd = 0.0
    if not cfg.get("lora_lr_step_size"):
        cfg.lora_lr_step_size = 1
    if not cfg.get("lora_lr_gamma"):
        cfg.lora_lr_gamma = 0.5
    if not cfg.get("lora_r"):
        cfg.lora_r = 8
    if not cfg.get("lora_alpha"):
        cfg.lora_alpha = 16.0
    if not cfg.get("lora_dropout"):
        cfg.lora_dropout = 0.0
    if not cfg.get("lora_tune_head"):
        cfg.lora_tune_head = True
    
    # Override device with specific GPU if provided
    if getattr(args, "gpu", None) is not None:
        try:
            gpu_id = int(args.gpu)
            cfg.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.device = "cpu"
    
    OmegaConf.set_struct(cfg, True)

    run_imagenet_baseline(cfg)


if __name__ == "__main__":
    main()

