"""
Evaluation Script for Fine-tuned Remote Sensing Models

This script evaluates all fine-tuned models on their validation/test datasets
and generates comprehensive results in CSV and image format.

Usage:
    python eval_finetuned_remote_sensing.py [--models MODEL1 MODEL2 ...] [--datasets DATASET1 DATASET2 ...]
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import from existing codebase
from src.models import ImageEncoder
from src.datasets.common import get_dataloader
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
    REMOTE_SENSING_DATASETS,
)
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader
from src.utils.variables_and_paths import get_finetuned_path, TQDM_BAR_FORMAT


def evaluate_single_model(
    model_name: str,
    dataset_name: str,
    model_location: str,
    data_location: str,
    batch_size: int = 32,
    device: str = "cuda",
) -> Optional[Dict]:
    """
    Evaluate a single fine-tuned model on a dataset.
    
    Args:
        model_name: Model architecture (e.g., "ViT-B-32")
        dataset_name: Dataset name (e.g., "AID")
        model_location: Path to model checkpoints
        data_location: Path to datasets
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation results or None if model doesn't exist
    """
    # Add Val suffix for consistency
    dataset_val = dataset_name + "Val"
    
    # Get fine-tuned model path
    ft_path = get_finetuned_path(model_location, dataset_val, model_name)
    
    if not os.path.exists(ft_path):
        print(f"  ✗ Model not found: {ft_path}")
        return None
    
    try:
        # Load fine-tuned model
        print(f"  Loading model from: {ft_path}")
        image_encoder = ImageEncoder(model_name).to(device)
        checkpoint = torch.load(ft_path, map_location=device)
        image_encoder.load_state_dict(checkpoint, strict=False)
        image_encoder.eval()
        
        # Load validation dataset
        print(f"  Loading validation dataset: {dataset_val}")
        dataset = get_remote_sensing_dataset(
            dataset_val,
            image_encoder.val_preprocess,
            location=data_location,
            batch_size=batch_size,
            num_workers=2,
        )
        
        # Get classification head
        from types import SimpleNamespace
        args = SimpleNamespace(
            model=model_name,
            save_dir=os.path.join(model_location, model_name),
            device=device,
        )
        classification_head = get_remote_sensing_classification_head(args, dataset_val, dataset)
        classification_head = classification_head.to(device)
        
        # Get validation dataloader
        val_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
        
        # Evaluate
        print(f"  Evaluating...")
        with torch.no_grad():
            metrics = evaluate_encoder_with_dataloader(
                image_encoder, classification_head, val_loader, device
            )
        
        accuracy = metrics['top1']
        print(f"  ✓ Accuracy: {accuracy * 100:.2f}%")
        
        # Load training info if available
        info_path = ft_path.replace('.pt', '_training_info.json')
        training_info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                training_info = json.load(f)
        
        result = {
            'model': model_name,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'num_classes': len(dataset.classnames),
            'checkpoint_path': ft_path,
            **training_info
        }
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error evaluating {model_name} on {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_results_table(results: List[Dict]) -> Dict:
    """
    Create a comprehensive results table from evaluation results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with table data organized by dataset and model
    """
    if not results:
        return {}
    
    # Organize results by dataset
    dataset_results = defaultdict(dict)
    for result in results:
        dataset = result['dataset']
        model = result['model']
        dataset_results[dataset][model] = result['accuracy']
        dataset_results[dataset]['num_classes'] = result['num_classes']
    
    # Sort datasets
    sorted_datasets = sorted(dataset_results.keys())
    
    # Create table structure
    model_order = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    
    table = {
        'datasets': sorted_datasets,
        'models': model_order,
        'data': {}
    }
    
    for dataset in sorted_datasets:
        table['data'][dataset] = {
            'num_classes': dataset_results[dataset].get('num_classes', 0),
            'accuracies': {
                model: dataset_results[dataset].get(model, None)
                for model in model_order
            }
        }
    
    return table


def save_results_table_as_csv(table: Dict, output_path: str):
    """
    Save results table as CSV file.
    
    Args:
        table: Table dictionary with results
        output_path: Path to save CSV
    """
    if not table or not table.get('datasets'):
        print("Empty table, skipping CSV generation")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['Dataset', 'Classes'] + table['models']
        writer.writerow(header)
        
        # Write data rows
        for dataset in table['datasets']:
            data = table['data'][dataset]
            row = [dataset, data['num_classes']]
            for model in table['models']:
                acc = data['accuracies'].get(model)
                if acc is not None:
                    row.append(f"{acc:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"✓ Saved results CSV: {output_path}")


def save_results_table_as_image(
    table: Dict,
    output_path: str,
    title: str = "Fine-tuned Model Accuracies on Remote Sensing Datasets"
):
    """
    Save results table as a publication-quality image.
    
    Args:
        table: Table dictionary with results
        output_path: Path to save image
        title: Title for the table
    """
    if not table or not table.get('datasets'):
        print("Empty table, skipping image generation")
        return
    
    datasets = table['datasets']
    models = table['models']
    
    fig, ax = plt.subplots(figsize=(12, len(datasets) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Format table data
    table_data = []
    for dataset in datasets:
        data = table['data'][dataset]
        row = [dataset, str(data['num_classes'])]
        for model in models:
            acc = data['accuracies'].get(model)
            if acc is not None:
                row.append(f"{acc:.4f}")
            else:
                row.append("—")
        table_data.append(row)
    
    # Create table
    mpl_table = ax.table(
        cellText=table_data,
        colLabels=['Dataset', 'Classes'] + models,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(10)
    mpl_table.scale(1, 2)
    
    # Header styling
    num_cols = len(models) + 2
    for i in range(num_cols):
        cell = mpl_table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(num_cols):
            cell = mpl_table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved results image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned remote sensing models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate (default: all remote sensing datasets)"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        default="./models/checkpoints_remote_sensing",
        help="Path to model checkpoints"
    )
    parser.add_argument(
        "--data_location",
        type=str,
        default="./datasets",
        help="Path to datasets"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evaluation",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    SIGMA_EPOCHS_PER_DATASET = {
    "AID": 20,              # ~10,000 train samples, 600x600
    "CLRS": 20,             # ~30,000 train samples, 256x256
    "EuroSAT_RGB": 20,      # ~21,600 train samples, 64x64
    "MLRSNet": 20,          # ~17,000 train samples, 256x256
    "NWPU-RESISC45": 20,    # ~25,200 train samples, 256x256
    "Optimal-31": 20,       # ~6,200 train samples, 256x256
    "PatternNet": 20,       # ~10,000 train samples, 256x256
    "RS_C11": 20,           # ~5,000 train samples, 512x512
    "RSD46-WHU": 20,        # ~10,000 train samples, 256x256
    "RSI-CB128": 20,        # ~18,000 train samples, 128x128
    "RSSCN7": 20,           # ~2,800 train samples, 400x400
    "SAT-4": 20,             # ~60,000 train samples, 28x28
    "SIRI-WHU": 20,        # ~2,400 train samples, 200x200
    "UC_Merced": 20,       # ~2,100 train samples, 256x256
    "WHU-RS19": 20,        # ~1,000 train samples, 600x600
}
    # Get list of datasets to evaluate
    if args.datasets is None:
        datasets = sorted(list(SIGMA_EPOCHS_PER_DATASET.keys()))
    else:
        datasets = args.datasets
    
    print("\n" + "=" * 100)
    print("EVALUATING FINE-TUNED REMOTE SENSING MODELS")
    print("=" * 100)
    print(f"Models: {args.models}")
    print(f"Datasets: {len(datasets)} total")
    print(f"Model location: {args.model_location}")
    print(f"Data location: {args.data_location}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 100 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate all models on all datasets
    all_results = []
    
    for dataset in tqdm(datasets, desc="Evaluating datasets", bar_format=TQDM_BAR_FORMAT):
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset}")
        print(f"{'='*100}")
        
        for model in args.models:
            print(f"\n  Model: {model}")
            result = evaluate_single_model(
                model_name=model,
                dataset_name=dataset,
                model_location=args.model_location,
                data_location=args.data_location,
                batch_size=args.batch_size,
                device=args.device,
            )
            
            if result is not None:
                all_results.append(result)
    
    # Save detailed results as JSON
    json_path = os.path.join(args.output_dir, "detailed_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✓ Saved detailed results: {json_path}")
    
    # Create and save results table
    if all_results:
        results_table = create_results_table(all_results)
        
        # Save as CSV
        csv_path = os.path.join(args.output_dir, "finetuned_accuracies.csv")
        save_results_table_as_csv(results_table, csv_path)
        
        # Save as image
        image_path = os.path.join(args.output_dir, "finetuned_accuracies.png")
        save_results_table_as_image(results_table, image_path)
        
        # Print summary
        print("\n" + "=" * 100)
        print("EVALUATION SUMMARY")
        print("=" * 100)
        print(f"{'Dataset':<20} {'Classes':<10} {' '.join([f'{m:<10}' for m in results_table['models']])}")
        print("-" * 100)
        for dataset in results_table['datasets']:
            data = results_table['data'][dataset]
            acc_strs = []
            for model in results_table['models']:
                acc = data['accuracies'].get(model)
                if acc is not None:
                    acc_strs.append(f"{acc:.4f}    ")
                else:
                    acc_strs.append("—         ")
            print(f"{dataset:<20} {data['num_classes']:<10} {''.join(acc_strs)}")
        print("=" * 100)
        
        # Compute statistics
        print("\nModel Statistics:")
        for model in args.models:
            accuracies = []
            for dataset in results_table['datasets']:
                acc = results_table['data'][dataset]['accuracies'].get(model)
                if acc is not None:
                    accuracies.append(acc)
            
            if len(accuracies) > 0:
                print(f"  {model}:")
                print(f"    Datasets evaluated: {len(accuracies)}")
                print(f"    Mean accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
                print(f"    Std accuracy: {np.std(accuracies):.4f}")
                print(f"    Min accuracy: {np.min(accuracies):.4f} ({np.min(accuracies)*100:.2f}%)")
                print(f"    Max accuracy: {np.max(accuracies):.4f} ({np.max(accuracies)*100:.2f}%)")
    else:
        print("\n⚠ No results to save")
    
    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

