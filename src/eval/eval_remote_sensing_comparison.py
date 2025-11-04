"""
Evaluation and comparison module for Energy Training on Remote Sensing datasets.

This module provides functions to:
1. Evaluate multiple merging methods (sum, average, TSVM, etc.)
2. Compare Energy-trained model with fine-tuned baseline and merging methods
3. Generate comparison tables and save results
"""

import os
import json
import pandas as pd
from datetime import datetime
import torch
import logging
from typing import Dict, Any, Optional, List, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.models.task_vectors import ImageEncoder, NonLinearTaskVector


def _select_sigma_layers(
    sigma_modules: torch.nn.ModuleDict,
    sigma_key_map: Dict[str, str],
    num_layers: int = 3,
) -> List[Tuple[str, str, str]]:
    if len(sigma_modules) == 0:
        return []
    keys = list(sigma_modules.keys())
    total = len(keys)
    if total <= num_layers:
        return [
            (keys[idx], sigma_key_map.get(keys[idx], keys[idx]), f"Layer {idx}")
            for idx in range(total)
        ]
    positions = [0, max(0, min(total - 1, total // 2)), total - 1]
    labels = ["Early", "Middle", "Late"]
    selected = []
    for pos, label in zip(positions, labels):
        safe_key = keys[pos]
        orig_key = sigma_key_map.get(safe_key, safe_key)
        selected.append((safe_key, orig_key, label))
    return selected


def visualize_sigma_matrices(
    sigma_modules: torch.nn.ModuleDict,
    sigma_key_map: Dict[str, str],
    epoch,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    json_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    logger = logging.getLogger(__name__)
    try:
        layers = _select_sigma_layers(sigma_modules, sigma_key_map)
        if not layers:
            return []

        # Compute global min/max across all sigma diagonals for consistent legends
        global_min = float("inf")
        global_max = float("-inf")
        all_layer_payload: List[Dict[str, object]] = []
        for safe_key, module in sigma_modules.items():
            sigma_vals = module.sigma.detach().cpu().numpy()
            global_min = min(global_min, float(sigma_vals.min()))
            global_max = max(global_max, float(sigma_vals.max()))
            orig_key = sigma_key_map.get(safe_key, safe_key)
            all_layer_payload.append(
                {
                    "safe_key": safe_key,
                    "orig_key": orig_key,
                    "sigma_values": sigma_vals.tolist(),
                    "min": float(sigma_vals.min()),
                    "max": float(sigma_vals.max()),
                    "mean": float(sigma_vals.mean()),
                    "std": float(sigma_vals.std()),
                    "numel": int(sigma_vals.size),
                }
            )
        if not all_layer_payload:
            return []

        fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]

        results = []
        for ax, (safe_key, orig_key, label) in zip(axes, layers):
            module = sigma_modules[safe_key]
            sigma_vals = module.sigma.detach().cpu().numpy()
            diag_matrix = np.diag(sigma_vals)

            im = ax.imshow(
                diag_matrix,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=global_min,
                vmax=global_max,
            )
            ax.set_xlabel("Column Index", fontsize=12)
            ax.set_ylabel("Row Index", fontsize=12)
            ax.set_title(f"{label} Layer\n{orig_key}", fontsize=14)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Sigma Value", rotation=90, labelpad=10, fontsize=12)

            stats = (
                f"Shape: {diag_matrix.shape[0]}×{diag_matrix.shape[1]}\n"
                f"Mean: {sigma_vals.mean():.4f}\n"
                f"Std: {sigma_vals.std():.4f}\n"
                f"Max: {sigma_vals.max():.4f}\n"
                f"Min: {sigma_vals.min():.4f}"
            )
            ax.text(
                1.40,
                0.5,
                stats,
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                verticalalignment="center",
            )

            results.append(
                {
                    "safe_key": safe_key,
                    "orig_key": orig_key,
                    "label": label,
                    "sigma_values": sigma_vals.copy(),
                    "epoch": epoch,
                }
            )

        if title is None:
            title = f"Sigma Diagonal Matrices - Epoch {epoch}"
        else:
            title = f"{title}\nEpoch {epoch}"
        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.debug(f"Saved sigma visualization: {save_path}")

        if json_path:
            payload = {
                "epoch": epoch,
                "title": title,
                "global_min": global_min,
                "global_max": global_max,
                "layers": all_layer_payload,
            }
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as fp:
                json.dump(payload, fp, indent=4)
            logger.debug(f"Saved sigma values to {json_path}")

        return results
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not visualize sigma matrices: {exc}")
        return []


def evaluate_encoder_with_dataloader(
    image_encoder: ImageEncoder,
    classification_head,
    dataloader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate an image encoder using a pre-loaded dataloader.
    
    Args:
        image_encoder: Image encoder to evaluate
        classification_head: Classification head
        dataloader: Pre-loaded dataloader
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics (top1 accuracy)
    """
    import torch
    from src.models.modeling import ImageClassifier
    from src.datasets.common import maybe_dictionarize
    from src.utils import utils
    
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()
    
    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            
            logits = utils.get_logits(x, model)
            
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        
        top1 = correct / n
    
    return {"top1": top1}
