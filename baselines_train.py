import os
import sys
import time
import logging
import argparse
import json

from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torchvision
import math

from src.utils.variables_and_paths import get_finetuned_path
from src.datasets import maybe_dictionarize, get_dataset, get_dataloader
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.models.modeling import ClassificationHead
from atlas_src.utils import TIPWrapper, LPPWrapper
from src.datasets.remote_sensing import sample_k_shot_indices


def setup_simple_logger(name: str = __name__) -> logging.Logger:
    """Setup a clean logger with minimal formatting (no INFO/DEBUG prefixes)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with simple format (just the message)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def _sanitize_value(val):
    """Sanitize value for use in config tag."""
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_baseline_config_tag(cfg) -> str:
    """Build config tag for baseline method."""
    method = cfg.baseline_method
    
    if method == 'linear_probe':
        lr = _sanitize_value(cfg.lp_lr)
        epochs = _sanitize_value(cfg.lp_epochs)
        wd = _sanitize_value(cfg.lp_wd)
        return f"baseline_lp_{lr}_{epochs}_{wd}"
    
    elif method == 'tip_adapter':
        wd = _sanitize_value(cfg.adapter_wd)
        return f"baseline_tip_{wd}"
    
    elif method == 'lp++':
        wd = _sanitize_value(cfg.adapter_wd)
        return f"baseline_lpp_{wd}"
    
    elif method == 'lora':
        r = _sanitize_value(cfg.lora_r)
        alpha = _sanitize_value(cfg.lora_alpha)
        lr = _sanitize_value(cfg.lora_lr)
        epochs = _sanitize_value(cfg.lora_epochs)
        wd = _sanitize_value(cfg.lora_wd)
        return f"baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}"
    
    else:
        return f"baseline_{method}"


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
            # Use model's normalized feature path for TIP/LPP stability
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
                    all_indexes.append(torch.arange(
                        start, start + len(inputs)))
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


def eval_adapter_dataset(adapter_model: torch.nn.Module, dataloader, device) -> dict:
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


def train_linear_probe(model, train_loader, val_loader, cfg, train_dataset_name, logger):
    """Train linear probe with time tracking and JSON result saving"""
    image_encoder = model.image_encoder
    # Freeze encoder, train head
    for p in image_encoder.parameters():
        p.requires_grad = False
    model.classification_head.weight.requires_grad_(True)
    model.classification_head.bias.requires_grad_(True)

    params = [p for p in model.classification_head.parameters()
              if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(
        cfg.lp_lr), weight_decay=float(cfg.lp_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(
        cfg.lp_lr_step_size), gamma=float(cfg.lp_lr_gamma))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = cfg.device
    model = model.to(device)
    
    # Track peak GPU memory
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

    # Final evaluation on validation set
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
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in params)
    logger.info(f"Number of trainable parameters: {trainable_params:,}")

    # Save results
    save_dir = cfg.save_dir
    k = int(cfg.k_shot)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    config_tag = getattr(cfg, 'config_tag', 'baseline_lp_default')
    
    result_dir = os.path.join(save_dir, train_dataset_name, config_tag, shot_folder)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results to JSON (use baseline naming convention)
    results_path = os.path.join(result_dir, f"baseline_results_none.json")
    results = {
        "method": "linear_probe",
        "target_dataset": train_dataset_name.replace("Val", ""),
        "final_accuracy": float(final_acc),
        "k_shot": k,
        "model": cfg.model,
        "epochs": cfg.lp_epochs,
        "lr": cfg.lp_lr,
        "wd": cfg.lp_wd,
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "batch_size": cfg.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "config_tag": config_tag,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")
    
    return results


def train_tip_or_lpp(model, train_loader, val_loader, cfg, train_dataset_name, logger, adapter: str):
    """Train TIP or LP++ adapter with time tracking and JSON result saving"""
    # Ensure classifier supports return_features for adapter wrappers
    ddp_model = ReturnFeaturesClassifier(model).to(cfg.device)
    
    # Track peak GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Build caches (logits/features) over the (possibly k-shot) training loader
    logger.info("Building feature cache...")
    cache_start = time.time()
    logits_cache, features_cache, labels, indexes_to_i = cache_features(
        ddp_model, train_loader, cfg.device)
    cache_time = time.time() - cache_start
    logger.info(f"Feature cache built in {cache_time:.2f}s")

    adapter_model = ddp_model
    if adapter == 'lpp':
        shots = int(cfg.k_shot) if int(cfg.k_shot) > 0 else 0
        adapter_model = LPPWrapper(
            adapter_model, features_cache, labels, shots)
        epochs = 20
        adapter_model = adapter_model.to(cfg.device)
        try:
            if hasattr(adapter_model, 'alpha_vec') and getattr(adapter_model.alpha_vec, 'requires_grad', None) is not False:
                adapter_model.alpha_vec.requires_grad = False
        except Exception:
            pass
        param_groups = [
            {'params': adapter_model.adapter.parameters(), 'lr': 1e-1}
        ]
    elif adapter == 'tip':
        adapter_model = TIPWrapper(adapter_model, features_cache, labels)
        epochs = 20
        adapter_model = adapter_model.to(cfg.device)
        param_groups = [
            {'params': [adapter_model.beta_alpha], 'lr': 1e-3}
        ]
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

            l_cache, f_cache = logits_cache[ids].to(
                inputs), features_cache[ids].to(inputs)
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

    # Final evaluation on validation set
    logger.info("\n" + "=" * 100)
    logger.info("Final evaluation on validation set...")
    logger.info("=" * 100 + "\n")
    
    adapter_model.eval()
    with torch.no_grad():
        metrics = eval_adapter_dataset(adapter_model, val_loader, cfg.device)
        final_acc = metrics['top1']
    
    logger.info(f"Adapter '{adapter}' Acc: {final_acc*100:.2f}%")
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

    # Save results
    save_dir = cfg.save_dir
    k = int(cfg.k_shot)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    config_tag = getattr(cfg, 'config_tag', f'baseline_{adapter}_default')
    
    result_dir = os.path.join(save_dir, train_dataset_name, config_tag, shot_folder)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results to JSON (use baseline naming convention)
    results_path = os.path.join(result_dir, f"baseline_results_none.json")
    results = {
        "method": adapter,
        "target_dataset": train_dataset_name.replace("Val", ""),
        "final_accuracy": float(final_acc),
        "k_shot": k,
        "model": cfg.model,
        "epochs": epochs,
        "adapter_wd": cfg.adapter_wd,
        "training_time": min_epoch_time,
        "avg_epoch_time": avg_epoch_time,
        "all_epoch_times": epoch_times,
        "trainable_params": trainable_params,
        "batch_size": cfg.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "cache_build_time": cache_time,
        "config_tag": config_tag,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")
    
    return results


def train_lora(model, train_loader, val_loader, cfg, train_dataset_name, logger):
    """Train LoRA with time tracking and JSON result saving"""
    apply_lora_to_module(model.image_encoder, r=int(cfg.lora_r), alpha=float(
        cfg.lora_alpha), dropout=float(cfg.lora_dropout))

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
    
    optimizer = torch.optim.AdamW(params, lr=float(
        cfg.lora_lr), weight_decay=float(cfg.lora_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(
        cfg.lora_lr_step_size), gamma=float(cfg.lora_lr_gamma))
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

    # Final evaluation
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
    
    # Time and memory stats
    min_epoch_time = min(epoch_times) if epoch_times else 0.0
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")
    
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

    # Save results
    save_dir = cfg.save_dir
    k = int(cfg.k_shot)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"
    config_tag = getattr(cfg, 'config_tag', 'baseline_lora_default')
    
    result_dir = os.path.join(save_dir, train_dataset_name, config_tag, shot_folder)
    os.makedirs(result_dir, exist_ok=True)

    # Save results to JSON (use baseline naming convention)
    results_path = os.path.join(result_dir, f"baseline_results_none.json")
    results = {
        "method": "lora",
        "target_dataset": train_dataset_name.replace("Val", ""),
        "final_accuracy": float(final_acc),
        "k_shot": k,
        "model": cfg.model,
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
        "batch_size": cfg.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
        "config_tag": config_tag,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"✓ Saved results to {results_path}")
    
    return results


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def run_baseline_training(cfg: DictConfig) -> None:
    """Run baseline training on general dataset"""
    logger = setup_simple_logger(__name__)
    
    # Set device
    if not cfg.device:
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set random seed
    try:
        import random
        import numpy as np
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    
    # Setup paths (using open_dict for OmegaConf struct mode compatibility)
    with open_dict(cfg):
        if not hasattr(cfg, "model_location") or cfg.model_location in (None, ""):
            cfg.model_location = os.path.expanduser("./models/checkpoints_remote_sensing")
        if not hasattr(cfg, "save_dir") or cfg.save_dir in (None, ""):
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        if not hasattr(cfg, "data_location") or cfg.data_location in (None, ""):
            cfg.data_location = os.path.expanduser("datasets")
        
        cfg.data_location = os.path.expanduser(cfg.data_location)
        
        # Build config tag for this baseline configuration
        config_tag = build_baseline_config_tag(cfg)
        cfg.config_tag = config_tag
    
    OmegaConf.set_struct(cfg, True)
    
    logger.info("=" * 100)
    logger.info(f"Baseline Training: {cfg.baseline_method}")
    logger.info("=" * 100)
    logger.info(f"Target dataset: {cfg.target_dataset}")
    logger.info(f"Model: {cfg.model}")
    logger.info(f"K-shot: {cfg.k_shot}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Config tag: {config_tag}")
    logger.info("=" * 100)
    
    # Load dataset (add Val suffix for consistency with atlas/energy)
    train_dataset_name = cfg.target_dataset + "Val"
    
    logger.info(f"Loading dataset: {train_dataset_name}")
    image_encoder = ImageEncoder(cfg.model).to(cfg.device)
    
    # Load dataset (same as energy_train_reverse.py)
    train_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224,
            scale=(0.5, 1.0),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])
    
    train_dataset = get_dataset(
        cfg.target_dataset,
        train_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    
    # Get classification head (same as energy_train_reverse.py)
    classification_head = get_classification_head(cfg, cfg.target_dataset)
    
    # Create model
    model = ImageClassifier(image_encoder, classification_head).to(cfg.device)
    model.freeze_head()
    
    # Get train loader from dataset
    train_loader = get_dataloader(
        train_dataset, is_train=True, args=cfg, image_encoder=None
    )
    
    # Apply k-shot sampling if specified
    k = int(cfg.k_shot)
    if k > 0:
        logger.info(f"Applying k-shot sampling: {k} samples per class")
        try:
            seed = int(cfg.seed)
            
            # Create directory for saving indices
            indices_save_dir = os.path.join(cfg.model_location, cfg.model, train_dataset_name)
            
            # Try to load existing k-shot indices
            selected_indices = load_k_shot_indices(indices_save_dir, k, seed)
            
            if selected_indices is not None:
                logger.info(f"✓ Loaded existing {k}-shot indices (seed={seed})")
            else:
                # Sample new indices from scratch
                logger.info(f"Sampling new {k}-shot indices from full dataset (seed={seed})")
                selected_indices = sample_k_shot_indices(
                    train_dataset,
                    k,
                    seed=seed,
                    verbose=True,
                    progress_desc=f"{cfg.target_dataset} {k}-shot",
                )
                # Save the indices for future use
                indices_path = save_k_shot_indices(selected_indices, indices_save_dir, train_dataset_name, k, seed)
                logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
            
            # Get base dataset
            base_dataset = getattr(train_dataset, "train_dataset", None)
            if base_dataset is None:
                base_dataset = getattr(train_loader, "dataset", None)
            
            if base_dataset is not None:
                num_workers = 2  # Fixed num_workers
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
                logger.warning("Could not locate base train_dataset for k-shot subsetting; using full loader instead.")
        except Exception as e:
            logger.error(f"Failed to apply k-shot sampling: {e}")
            logger.warning("Falling back to full training set")
    
    # Load validation dataset (same as energy_train_reverse.py)
    logger.info(f"Loading validation dataset: {train_dataset_name}")
    val_dataset = get_dataset(
        cfg.target_dataset,
        image_encoder.val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    val_loader = get_dataloader(
        val_dataset, is_train=False, args=cfg, image_encoder=None
    )
    logger.info(f"✓ Validation loader ready ({len(val_loader.dataset)} samples)")
    
    # Train baseline
    logger.info(f"\nTraining baseline '{cfg.baseline_method}' on {cfg.target_dataset}")
    if cfg.baseline_method == 'linear_probe':
        results = train_linear_probe(model, train_loader, val_loader, cfg,
                           train_dataset_name, logger)
    elif cfg.baseline_method == 'tip_adapter':
        results = train_tip_or_lpp(model, train_loader, val_loader, cfg,
                         train_dataset_name, logger, adapter='tip')
    elif cfg.baseline_method == 'lp++':
        results = train_tip_or_lpp(model, train_loader, val_loader, cfg,
                         train_dataset_name, logger, adapter='lpp')
    elif cfg.baseline_method == 'lora':
        results = train_lora(model, train_loader, val_loader, cfg, train_dataset_name, logger)
    else:
        raise NotImplementedError(
            f"Unknown baseline method: {cfg.baseline_method}")
    
    logger.info("\n" + "=" * 100)
    logger.info("Training complete!")
    logger.info(f"Final accuracy: {results['final_accuracy']*100:.2f}%")
    logger.info("=" * 100)


if __name__ == "__main__":
    from src.datasets.registry import registry as DATASET_REGISTRY
    
    # Get allowed test datasets (exclude Val versions)
    allowed_test_datasets = sorted(
        [name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")]
    )
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Baseline training for general datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
        choices=allowed_test_datasets,
        help="Target dataset to train on",
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
    
    # Baseline method
    parser.add_argument(
        "--baseline_method",
        type=str,
        choices=["linear_probe", "tip_adapter", "lp++", "lora"],
        default="linear_probe",
        help="Baseline to train",
    )
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--k",
        type=int,
        dest="k_shot",
        help="K-shot samples per class (0=fullshot)"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, help="Device to use (e.g., cuda:0, cpu)")
    
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
    parser.add_argument("--lora_tune_head", type=int, choices=[0, 1], help="Tune classification head with LoRA")
    
    args = parser.parse_args()
    
    # Load config file
    cfg = load_config(args.config_file)
    
    # Merge CLI arguments (only non-None values override config)
    cli_overrides = {k: v for k, v in vars(args).items() 
                     if v is not None and k != "config_file"}
    
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    
    # Validate required fields
    if not cfg.get("target_dataset"):
        parser.error("--target_dataset is required")
    
    # Set defaults for optional fields
    if not cfg.get("k_shot"):
        cfg.k_shot = 0
    if not cfg.get("device"):
        cfg.device = ""
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
    
    OmegaConf.set_struct(cfg, True)
    run_baseline_training(cfg)
