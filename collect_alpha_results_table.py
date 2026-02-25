#!/usr/bin/env python3
"""
Alpha ablation results collection and table generation.
Creates a table showing accuracy for each alpha value across all datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def collect_alpha_results_table(base_dir="./models/checkpoints/ViT-B-32"):
    """Collect alpha ablation results and create a table."""
    
    datasets = [
        "DTD", "GTSRB", "MNIST", "SVHN", "STL10",
        "OxfordIIITPet", "Flowers102", "CIFAR100", "PCAM", "CIFAR10",
        "Food101", "FashionMNIST", "RenderedSST2", "EMNIST",
        "FGVCAircraft", "CUB200", "Country211"
    ]
    
    alphas = [1, 5, 10, 15]
    
    # Dictionary to store results: {dataset: {alpha: accuracy}}
    results_dict = {}
    
    print("=" * 80)
    print("Collecting results from experiments...")
    print("=" * 80)
    
    for dataset in datasets:
        results_dict[dataset] = {}
        
        for alpha in alphas:
            # Path pattern: base_dir/[DATASET]Val/ablation_alpha_[ALPHA]/fullshots/energy_results_none.json
            result_path = os.path.join(
                base_dir,
                f"{dataset}Val",
                f"ablation_alpha_{alpha}",
                "4shots",
                "energy_results_none.json"
            )
            
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        data = json.load(f)
                    
                    accuracy = data.get("final_accuracy", 0.0) * 100
                    results_dict[dataset][alpha] = accuracy
                    print(f"✓ {dataset:20s} alpha={alpha:2d}: {accuracy:6.2f}%")
                except Exception as e:
                    print(f"✗ Error loading {dataset} alpha={alpha}: {e}")
                    results_dict[dataset][alpha] = None
            else:
                print(f"✗ Not found: {dataset} alpha={alpha}")
                results_dict[dataset][alpha] = None
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.columns = [f"Alpha={a}" for a in alphas]
    
    # Add mean row
    mean_row = df.mean(axis=0, skipna=True)
    df.loc['Mean'] = mean_row
    
    # Add std row
    std_row = df.iloc[:-1].std(axis=0, skipna=True)  # Exclude mean row
    df.loc['Std'] = std_row
    
    # Create output directory
    output_dir = Path("ablation_alpha")
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "alpha_ablation_table.csv"
    df.to_csv(csv_path)
    print(f"\n✓ Saved table to: {csv_path}")
    
    # Print table
    print("\n" + "=" * 80)
    print("ALPHA ABLATION RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))
    print("=" * 80)
    
    # Create summary statistics
    summary = {}
    for alpha in alphas:
        col_name = f"Alpha={alpha}"
        values = df[col_name].iloc[:-2]  # Exclude mean and std rows
        valid_values = values.dropna()
        
        summary[f"alpha_{alpha}"] = {
            "mean": float(valid_values.mean()) if len(valid_values) > 0 else None,
            "std": float(valid_values.std()) if len(valid_values) > 0 else None,
            "min": float(valid_values.min()) if len(valid_values) > 0 else None,
            "max": float(valid_values.max()) if len(valid_values) > 0 else None,
            "num_datasets": int(len(valid_values)),
        }
    
    # Save summary
    summary_path = output_dir / "alpha_ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Saved summary to: {summary_path}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("ALPHA COMPARISON SUMMARY")
    print("=" * 80)
    for alpha in alphas:
        stats = summary[f"alpha_{alpha}"]
        if stats["mean"] is not None:
            print(f"Alpha={alpha:2d}: {stats['mean']:6.2f}% ± {stats['std']:5.2f}% "
                  f"(min={stats['min']:6.2f}%, max={stats['max']:6.2f}%, n={stats['num_datasets']})")
    print("=" * 80)
    
    # Find best alpha
    mean_accs = [(alpha, summary[f"alpha_{alpha}"]["mean"]) 
                 for alpha in alphas if summary[f"alpha_{alpha}"]["mean"] is not None]
    if mean_accs:
        best_alpha, best_acc = max(mean_accs, key=lambda x: x[1])
        print(f"\n🏆 Best alpha: {best_alpha} with mean accuracy {best_acc:.2f}%")
    
    return df, summary


if __name__ == "__main__":
    print("Collecting alpha ablation results...")
    df, summary = collect_alpha_results_table()
    print("\n✅ Results collection and table generation completed!")
    print(f"📁 Results saved in: ablation_alpha/")
