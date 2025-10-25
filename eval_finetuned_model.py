#!/usr/bin/env python3
"""
Evaluate all remote sensing datasets and generate a comprehensive results table.

This script:
1. Discovers all datasets in the checkpoints directory
2. Evaluates zeroshot and finetuned models for each dataset
3. Generates a single comprehensive table image with all results
"""

import os
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional
from pathlib import Path

# Set environment variables for cache directories BEFORE importing torch/open_clip
os.environ['HF_HOME'] = '/home/hjh/openclip-cachedir/open_clip'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/hjh/openclip-cachedir/open_clip'
os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode to prevent downloads

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.models import ImageEncoder
from src.datasets import get_dataloader
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
)
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_encoder(model_name: str, checkpoint_path: str, device: str) -> Optional[ImageEncoder]:
    """Load an image encoder from checkpoint."""
    if not os.path.exists(checkpoint_path):
        LOGGER.warning(f"Checkpoint missing: {checkpoint_path}")
        return None
    
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            encoder = ImageEncoder(model_name)
            encoder.load_state_dict(state, strict=False)
            encoder = encoder.to(device)
            encoder.eval()
            return encoder
        else:
            LOGGER.warning(f"Unexpected checkpoint format: {checkpoint_path}")
            return None
    except Exception as e:
        LOGGER.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def evaluate_encoder_accuracy(
    encoder: ImageEncoder,
    classification_head,
    val_loader,
    device: str,
) -> float:
    """Evaluate encoder and return accuracy percentage."""
    try:
        metrics = evaluate_encoder_with_dataloader(
            encoder, classification_head, val_loader, device
        )
        return metrics.get("top1", 0.0) * 100.0
    except Exception as e:
        LOGGER.error(f"Error during evaluation: {e}")
        return 0.0


def discover_datasets(model_root: str) -> List[str]:
    """Discover all dataset directories."""
    datasets = []
    if not os.path.exists(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")
    
    for entry in sorted(os.listdir(model_root)):
        path = os.path.join(model_root, entry)
        if os.path.isdir(path) and entry.endswith("Val"):
            dataset_name = entry.replace("Val", "")
            datasets.append(dataset_name)
    
    return datasets


def evaluate_single_dataset(
    dataset: str,
    model: str,
    model_root: str,
    data_location: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, any]:
    """Evaluate a single dataset and return results."""
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Evaluating: {dataset}")
    LOGGER.info(f"{'='*80}")
    
    dataset_val = f"{dataset}Val"
    dataset_dir = os.path.join(model_root, dataset_val)
    
    # Prepare result dictionary
    result = {
        'Dataset': dataset,
        'Zeroshot': None,
        'Finetuned': None,
        'Improvement': None,
    }
    
    try:
        # Load dataset
        template_encoder = ImageEncoder(model).to(device)
        val_dataset = get_remote_sensing_dataset(
            dataset_val,
            template_encoder.val_preprocess,
            location=data_location,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        # Get classification head
        args_ns = SimpleNamespace(
            save_dir=model_root,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        classification_head = get_remote_sensing_classification_head(
            args_ns, dataset_val, val_dataset
        )
        classification_head = classification_head.to(device)
        classification_head.eval()
        
        val_loader = get_dataloader(
            val_dataset, is_train=False, args=args_ns, image_encoder=None
        )
        
        # Evaluate zeroshot
        zeroshot_path = os.path.join(dataset_dir, "nonlinear_zeroshot.pt")
        if os.path.exists(zeroshot_path):
            LOGGER.info(f"  Evaluating zeroshot: {zeroshot_path}")
            encoder = load_encoder(model, zeroshot_path, device)
            if encoder:
                zs_acc = evaluate_encoder_accuracy(
                    encoder, classification_head, val_loader, device
                )
                result['Zeroshot'] = zs_acc
                LOGGER.info(f"  ✓ Zeroshot accuracy: {zs_acc:.2f}%")
        
        # Evaluate finetuned
        finetuned_path = os.path.join(dataset_dir, "nonlinear_finetuned.pt")
        if os.path.exists(finetuned_path):
            LOGGER.info(f"  Evaluating finetuned: {finetuned_path}")
            encoder = load_encoder(model, finetuned_path, device)
            if encoder:
                ft_acc = evaluate_encoder_accuracy(
                    encoder, classification_head, val_loader, device
                )
                result['Finetuned'] = ft_acc
                LOGGER.info(f"  ✓ Finetuned accuracy: {ft_acc:.2f}%")
        
        # Calculate improvement
        if result['Zeroshot'] is not None and result['Finetuned'] is not None:
            result['Improvement'] = result['Finetuned'] - result['Zeroshot']
            LOGGER.info(f"  ✓ Improvement: {result['Improvement']:.2f}%")
        
    except Exception as e:
        LOGGER.error(f"Error evaluating {dataset}: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def create_results_table(
    results: List[Dict],
    model: str,
    save_dir: str,
) -> str:
    """Create a comprehensive table image from all results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'Dataset': 'AVERAGE',
        'Zeroshot': df['Zeroshot'].mean() if df['Zeroshot'].notna().any() else None,
        'Finetuned': df['Finetuned'].mean() if df['Finetuned'].notna().any() else None,
        'Improvement': df['Improvement'].mean() if df['Improvement'].notna().any() else None,
    }
    
    # Add stats row
    df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    
    # Format for display
    display_df = df.copy()
    for col in ['Zeroshot', 'Finetuned', 'Improvement']:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    
    # Create figure
    fig_height = max(8, len(df) * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table_data = display_df.values.tolist()
    col_labels = ['Dataset', 'Zeroshot (%)', 'Finetuned (%)', 'Improvement (%)']
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.20, 0.20, 0.20],
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    # Header styling
    for idx in range(len(col_labels)):
        cell = table[(0, idx)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Row styling
    num_rows = len(table_data)
    for row_idx in range(1, num_rows + 1):
        for col_idx in range(len(col_labels)):
            cell = table[(row_idx, col_idx)]
            
            # Last row (average) styling
            if row_idx == num_rows:
                cell.set_facecolor('#FFC000')
                cell.set_text_props(weight='bold', fontsize=10)
            # Alternating row colors
            elif row_idx % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            
            # Highlight positive improvements in green
            if col_idx == 3 and row_idx < num_rows:  # Improvement column
                try:
                    val = float(table_data[row_idx - 1][col_idx])
                    if val > 0:
                        cell.set_facecolor('#C6E0B4')
                        cell.set_text_props(weight='bold')
                except (ValueError, TypeError):
                    pass
    
    # Title
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"Remote Sensing Datasets Evaluation Results\n"
    title += f"Model: {model} | Generated: {timestamp}\n"
    title += f"Total Datasets: {len(results)}"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(save_dir, f"all_datasets_results_{model}_{timestamp_file}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"all_datasets_results_{model}_{timestamp_file}.csv")
    df.to_csv(csv_path, index=False)
    
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"✓ Saved results table: {png_path}")
    LOGGER.info(f"✓ Saved results CSV: {csv_path}")
    LOGGER.info(f"{'='*80}")
    
    return png_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate all remote sensing datasets and generate comprehensive results table"
    )
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        help="Model architecture (default: ViT-B-16)"
    )
    parser.add_argument(
        "--model_location",
        default="./models/checkpoints_remote_sensing",
        help="Root directory containing checkpoints"
    )
    parser.add_argument(
        "--data_location",
        default="./datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        default="./results/evaluation",
        help="Directory to save results"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Starting evaluation for model: {args.model}")
    LOGGER.info(f"Device: {args.device}")
    LOGGER.info(f"{'='*80}\n")
    
    # Discover datasets
    model_root = os.path.join(args.model_location, args.model)
    all_datasets = discover_datasets(model_root)
    
    if args.datasets:
        datasets_to_eval = [d for d in all_datasets if d in args.datasets]
    else:
        datasets_to_eval = all_datasets
    
    LOGGER.info(f"Found {len(datasets_to_eval)} datasets to evaluate:")
    for ds in datasets_to_eval:
        LOGGER.info(f"  - {ds}")
    LOGGER.info("")
    
    # Evaluate all datasets
    results = []
    for dataset in datasets_to_eval:
        result = evaluate_single_dataset(
            dataset=dataset,
            model=args.model,
            model_root=model_root,
            data_location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        results.append(result)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create comprehensive results table
    if results:
        create_results_table(
            results=results,
            model=args.model,
            save_dir=args.output_dir,
        )
        
        # Print summary
        LOGGER.info("\n" + "="*80)
        LOGGER.info("EVALUATION SUMMARY")
        LOGGER.info("="*80)
        for result in results:
            zs = f"{result['Zeroshot']:.2f}%" if result['Zeroshot'] is not None else "N/A"
            ft = f"{result['Finetuned']:.2f}%" if result['Finetuned'] is not None else "N/A"
            imp = f"{result['Improvement']:.2f}%" if result['Improvement'] is not None else "N/A"
            LOGGER.info(f"{result['Dataset']:20s} | ZS: {zs:8s} | FT: {ft:8s} | Δ: {imp:8s}")
        LOGGER.info("="*80 + "\n")
    else:
        LOGGER.warning("No results to display")


if __name__ == "__main__":
    main()

