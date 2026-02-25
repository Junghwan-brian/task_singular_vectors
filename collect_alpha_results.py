#!/usr/bin/env python3
"""
Alpha ablation results collection and analysis script.
Collects results from all datasets and creates a summary.
"""

import os
import json
import pandas as pd
from pathlib import Path

def collect_alpha_results(base_dir="./models/checkpoints/ViT-B-32"):
    """Collect alpha ablation results from all datasets."""
    
    datasets = [
        "DTD", "GTSRB", "MNIST", "SVHN", "STL10",
        "OxfordIIITPet", "Flowers102", "CIFAR100", "PCAM", "CIFAR10",
        "Food101", "FashionMNIST", "RenderedSST2", "EMNIST",
        "FGVCAircraft", "CUB200", "Country211"
    ]
    
    results = []
    
    for dataset in datasets:
        # Path pattern: base_dir/[DATASET]Val/ablation_alpha/fullshots/energy_results_none.json
        result_path = os.path.join(
            base_dir,
            f"{dataset}Val",
            "ablation_alpha",
            "fullshots",
            "energy_results_none.json"
        )
        
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    data = json.load(f)
                
                # Extract alpha value from validation history if available
                # The alpha grid search should have logged the selected alpha
                selected_alpha = None
                if "validation_history" in data and len(data["validation_history"]) > 0:
                    # Look for alpha information in the data
                    # (may need to add this to the save logic)
                    pass
                
                result_entry = {
                    "dataset": dataset,
                    "final_accuracy": data.get("final_accuracy", 0.0) * 100,
                    "training_time": data.get("training_time", 0.0),
                    "trainable_params": data.get("trainable_params", 0),
                    "gpu_peak_mem_mb": data.get("gpu_peak_mem_mb", 0.0),
                    "sigma_epochs": data.get("sigma_epochs", 0),
                    "sigma_lr": data.get("sigma_lr", 0.0),
                    "svd_keep_topk": data.get("svd_keep_topk", 0),
                }
                results.append(result_entry)
                print(f"✓ Loaded results for {dataset}: {result_entry['final_accuracy']:.2f}%")
            except Exception as e:
                print(f"✗ Error loading {dataset}: {e}")
        else:
            print(f"✗ Results not found for {dataset}: {result_path}")
    
    if not results:
        print("\n❌ No results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by accuracy
    df = df.sort_values("final_accuracy", ascending=False)
    
    # Create output directory
    output_dir = Path("ablation_alpha")
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "alpha_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to: {csv_path}")
    
    # Save summary statistics
    summary = {
        "mean_accuracy": float(df["final_accuracy"].mean()),
        "std_accuracy": float(df["final_accuracy"].std()),
        "min_accuracy": float(df["final_accuracy"].min()),
        "max_accuracy": float(df["final_accuracy"].max()),
        "num_datasets": len(df),
        "mean_training_time": float(df["training_time"].mean()),
        "total_trainable_params": int(df["trainable_params"].iloc[0]) if len(df) > 0 else 0,
    }
    
    summary_path = output_dir / "alpha_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Saved summary to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ALPHA ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"Number of datasets: {summary['num_datasets']}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
    print(f"Min accuracy: {summary['min_accuracy']:.2f}%")
    print(f"Max accuracy: {summary['max_accuracy']:.2f}%")
    print(f"Mean training time: {summary['mean_training_time']:.2f}s per epoch")
    print(f"Trainable parameters: {summary['total_trainable_params']:,}")
    print("=" * 80)
    
    # Print top 5 and bottom 5
    print("\n📊 Top 5 Datasets by Accuracy:")
    print(df[["dataset", "final_accuracy"]].head(5).to_string(index=False))
    
    print("\n📊 Bottom 5 Datasets by Accuracy:")
    print(df[["dataset", "final_accuracy"]].tail(5).to_string(index=False))
    
    return df


if __name__ == "__main__":
    print("Collecting alpha ablation results...")
    print("=" * 80)
    df = collect_alpha_results()
    
    if df is not None:
        print("\n✅ Results collection completed successfully!")
        print(f"📁 Results saved in: ablation_alpha/")
    else:
        print("\n❌ No results were collected.")
