#!/usr/bin/env python3
"""
Plot ablation study results for paper figures.

This script reads JSON results from ablation/results/ and generates:
1. num_basis_tasks vs accuracy plots (grouped by model and k-shot)
2. svd_keep_topk vs accuracy plots (grouped by model and k-shot)
"""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def _setup_publication_style() -> None:
    """
    Configure matplotlib for publication-quality figures (similar to plot_energy_vs_maml.py).
    """
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.figsize': (10, 6),
        'lines.linewidth': 4,
        'lines.markersize': 10,
        'grid.alpha': 0.3,
    })


def _format_model_name(model: str) -> str:
    """
    Format model name for display in legends (replace last '-' with '/').
    
    Args:
        model: Model name (e.g., 'ViT-B-32')
    
    Returns:
        Formatted model name (e.g., 'ViT-B/32')
    """
    # Replace only the last '-' with '/'
    if '-' in model:
        parts = model.rsplit('-', 1)
        return '/'.join(parts)
    return model


def load_all_results(results_dir: str = "ablation/results") -> List[Dict]:
    """Load all JSON result files from the ablation results directory."""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    results = []
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    
    print(f"Loaded {len(results)} result files")
    return results


def group_results_for_basis_plot(results: List[Dict]) -> Dict[Tuple[str, int, int], Dict[int, List[float]]]:
    """
    Group results for num_basis_tasks vs accuracy plot.
    
    Returns:
        Dict mapping (model, k_shot, svd_keep_topk) -> {num_basis_tasks: [accuracies from different datasets]}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        model = result.get("model", "unknown")
        k_shot = int(result.get("k_shot", 0))
        svd_keep_topk = int(result.get("svd_keep_topk", 0))
        num_basis_tasks = int(result.get("num_basis_tasks", 0))
        accuracy = float(result.get("final_accuracy", 0.0))
        
        key = (model, k_shot, svd_keep_topk)
        grouped[key][num_basis_tasks].append(accuracy)
    
    return grouped


def group_results_for_topk_plot(results: List[Dict]) -> Dict[Tuple[str, int, int], Dict[int, List[float]]]:
    """
    Group results for svd_keep_topk vs accuracy plot.
    
    Returns:
        Dict mapping (model, k_shot, num_basis_tasks) -> {svd_keep_topk: [accuracies from different datasets]}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        model = result.get("model", "unknown")
        k_shot = int(result.get("k_shot", 0))
        svd_keep_topk = int(result.get("svd_keep_topk", 0))
        num_basis_tasks = int(result.get("num_basis_tasks", 0))
        accuracy = float(result.get("final_accuracy", 0.0))
        
        key = (model, k_shot, num_basis_tasks)
        grouped[key][svd_keep_topk].append(accuracy)
    
    return grouped


def plot_basis_ablation(grouped_data: Dict[Tuple[str, int, int], Dict[int, List[float]]], 
                        output_dir: str = "ablation/plots"):
    """
    Plot num_basis_tasks vs accuracy (only for svd_keep_topk=12).
    
    Creates mean plot across datasets (without std) and individual dataset plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    _setup_publication_style()
    
    # Filter only topk=12
    filtered_data = {k: v for k, v in grouped_data.items() if k[2] == 12}
    
    # Group by k_shot -> model -> data_dict
    by_kshot: Dict[int, Dict[str, Dict[int, List[float]]]] = defaultdict(dict)
    for (model, k_shot, _topk), data in filtered_data.items():
        by_kshot[int(k_shot)][str(model)] = data
    
    for k_shot, model_to_data in by_kshot.items():
        # Mean plot across datasets
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_mean_accs = []  # Collect all plotted values for y-axis limits
        # Build union of basis values (exclude 0)
        union_basis = set()
        for data_dict in model_to_data.values():
            basis_vals = [b for b in data_dict.keys() if int(b) > 0]
            union_basis.update(basis_vals)
        union_basis_values = sorted(union_basis)
        x_index_by_basis = {b: i for i, b in enumerate(union_basis_values)}
        
        # Color palette and model order
        palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBafe4"]
        preferred = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
        present_models = list(model_to_data.keys())
        ordered_models = [m for m in preferred if m in present_models] + [m for m in present_models if m not in preferred]
        model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}
        
        for model in ordered_models:
            data_dict = model_to_data[model]  # {num_basis_tasks: [accuracies]}
            basis_values = sorted([b for b in data_dict.keys() if int(b) > 0])
            if not basis_values:
                continue
            mean_accs = []
            std_accs = []
            x_positions = []
            for basis in basis_values:
                accs = np.array(data_dict[basis]) * 100  # Convert to percentage
                mean_accs.append(np.mean(accs))
                std_accs.append(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
                x_positions.append(x_index_by_basis[basis])
            mean_accs = np.array(mean_accs, dtype=float)
            std_accs = np.array(std_accs, dtype=float)
            all_mean_accs.extend(mean_accs.tolist())
            
            # Plot mean line
            ax.plot(x_positions, mean_accs, marker='o', linewidth=4, 
                   color=model_to_color[model], label=_format_model_name(model), zorder=3)
            # Plot std band (scaled down to 0.3x for visibility)
            band_scale = 0.05
            ax.fill_between(x_positions, 
                           mean_accs - band_scale * std_accs, 
                           mean_accs + band_scale * std_accs,
                           color=model_to_color[model], 
                           alpha=0.2,
                           zorder=2)
        
        # Set xticks to union
        ax.set_xticks(np.arange(len(union_basis_values)))
        ax.set_xticklabels(union_basis_values)
        
        # Calculate y-axis limits (nearest 10) based on plotted values
        if all_mean_accs:
            min_acc = min(all_mean_accs)
            max_acc = max(all_mean_accs)
            y_min = int(np.floor(min_acc / 2) * 2)
            y_max = int(np.ceil(max_acc / 2) * 2)
            ax.set_ylim(y_min, y_max+1)
            
            # Set integer ticks with consistent spacing
            y_range = y_max - y_min
            if y_range <= 10:
                tick_step = 3
            elif y_range <= 20:
                tick_step = 3
            elif y_range <= 40:
                tick_step = 5
            else:
                tick_step = 10
            ax.set_yticks(np.arange(y_min, y_max + 1, tick_step))
        
        ax.set_xlabel("Number of Task Vector", fontsize=22, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save mean plot
        filename_base = f"basis_ablation_k{k_shot}_mean"
        base_path = os.path.join(output_dir, filename_base)
        # plt.savefig(base_path + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(base_path + ".png", dpi=1200, bbox_inches='tight')
        # print(f"Saved: {base_path}.pdf")
        print(f"Saved: {base_path}.png")
        plt.close()


def plot_topk_ablation(grouped_data: Dict[Tuple[str, int, int], Dict[int, List[float]]], 
                       output_dir: str = "ablation/plots"):
    """
    Plot svd_keep_topk vs accuracy (only for num_basis_tasks=0, i.e., basis=all).
    
    Creates mean plot across datasets (without std) and individual dataset plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    _setup_publication_style()
    
    # Filter only basis=0 (all)
    filtered_data = {k: v for k, v in grouped_data.items() if k[2] == 0}
    
    # Group by k_shot -> model -> data_dict
    by_kshot: Dict[int, Dict[str, Dict[int, List[float]]]] = defaultdict(dict)
    for (model, k_shot, _num_basis), data in filtered_data.items():
        by_kshot[int(k_shot)][str(model)] = data
    
    for k_shot, model_to_data in by_kshot.items():
        # Mean plot across datasets
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_mean_accs = []  # Collect all plotted values for y-axis limits
        # Build union of topk values
        union_topk = set()
        for data_dict in model_to_data.values():
            union_topk.update(list(data_dict.keys()))
        union_topk_values = sorted(union_topk)
        x_index_by_topk = {t: i for i, t in enumerate(union_topk_values)}
        
        # Color palette and model order
        palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBafe4"]
        preferred = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
        present_models = list(model_to_data.keys())
        ordered_models = [m for m in preferred if m in present_models] + [m for m in present_models if m not in preferred]
        model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}
        
        for model in ordered_models:
            data_dict = model_to_data[model]  # {svd_keep_topk: [accuracies]}
            topk_values = sorted(list(data_dict.keys()))
            mean_accs = []
            std_accs = []
            x_positions = []
            for topk in topk_values:
                accs = np.array(data_dict[topk]) * 100  # Convert to percentage
                mean_accs.append(np.mean(accs))
                std_accs.append(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)
                x_positions.append(x_index_by_topk[topk])
            mean_accs = np.array(mean_accs, dtype=float)
            std_accs = np.array(std_accs, dtype=float)
            all_mean_accs.extend(mean_accs.tolist())
            
            # Plot mean line
            ax.plot(x_positions, mean_accs, marker='s', linewidth=4, 
                   color=model_to_color[model], label=_format_model_name(model), zorder=3)
            # Plot std band (scaled down to 0.3x for visibility)
            band_scale = 0.05
            ax.fill_between(x_positions, 
                           mean_accs - band_scale * std_accs, 
                           mean_accs + band_scale * std_accs,
                           color=model_to_color[model], 
                           alpha=0.2,
                           zorder=2)
        
        # Set xticks to union
        ax.set_xticks(np.arange(len(union_topk_values)))
        ax.set_xticklabels(union_topk_values)
        
        # Calculate y-axis limits (nearest 10) based on plotted values
        if all_mean_accs:
            min_acc = min(all_mean_accs)
            max_acc = max(all_mean_accs)
            y_min = int(np.floor(min_acc / 2) * 2)
            y_max = int(np.ceil(max_acc / 2) * 2)
            ax.set_ylim(y_min, y_max)
            
            # Set integer ticks with consistent spacing
            y_range = y_max - y_min
            if y_range <= 10:
                tick_step = 3
            elif y_range <= 20:
                tick_step = 3
            elif y_range <= 40:
                tick_step = 5
            else:
                tick_step = 10
            ax.set_yticks(np.arange(y_min, y_max + 1, tick_step))
        
        ax.set_xlabel(r"Number of rank r", fontsize=22, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save mean plot
        filename_base = f"topk_ablation_k{k_shot}_mean"
        base_path = os.path.join(output_dir, filename_base)
        # plt.savefig(base_path + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(base_path + ".png", dpi=1200, bbox_inches='tight')
        # print(f"Saved: {base_path}.pdf")
        print(f"Saved: {base_path}.png")
        plt.close()


def plot_basis_ablation_per_dataset(results: List[Dict], output_dir: str = "ablation/plots/per_dataset"):
    """
    Plot num_basis_tasks vs accuracy for each dataset individually (only for svd_keep_topk=12).
    Each plot shows all models together with a legend.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    _setup_publication_style()
    
    # Filter only topk=12
    filtered_results = [r for r in results if r.get("svd_keep_topk", 0) == 12]
    
    # Group by (dataset, k_shot) -> {model: [results]}
    by_dataset = defaultdict(lambda: defaultdict(list))
    for r in filtered_results:
        key = (r.get("test_dataset", "unknown"), r.get("k_shot", 0))
        model = r.get("model", "unknown")
        by_dataset[key][model].append(r)
    
    # Color palette and model order
    palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBafe4"]
    preferred = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    
    for (dataset, k_shot), model_to_data in by_dataset.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_accs = []  # Collect all plotted values for y-axis limits
        
        # Build union of basis values (exclude 0)
        union_basis = set()
        for data in model_to_data.values():
            basis_vals = [int(r.get("num_basis_tasks", 0)) for r in data if int(r.get("num_basis_tasks", 0)) > 0]
            union_basis.update(basis_vals)
        union_basis_values = sorted(union_basis)
        x_index_by_basis = {b: i for i, b in enumerate(union_basis_values)}
        
        # Order models
        present_models = list(model_to_data.keys())
        ordered_models = [m for m in preferred if m in present_models] + [m for m in present_models if m not in preferred]
        model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}
        
        for model in ordered_models:
            data = model_to_data[model]
            # Sort by num_basis_tasks, exclude basis=0
            data_sorted = sorted([r for r in data if int(r.get("num_basis_tasks", 0)) > 0],
                                 key=lambda x: x.get("num_basis_tasks", 0))
            
            if not data_sorted:
                continue
            
            basis_values = [r.get("num_basis_tasks", 0) for r in data_sorted]
            accs = [r.get("final_accuracy", 0.0) * 100 for r in data_sorted]
            
            # Plot with union-based x-axis
            x_positions = [x_index_by_basis[b] for b in basis_values]
            ax.plot(x_positions, accs, marker='o', linewidth=4, 
                   color=model_to_color[model], label=_format_model_name(model), zorder=3)
            all_accs.extend(accs)
        
        # Set xticks to union
        ax.set_xticks(np.arange(len(union_basis_values)))
        ax.set_xticklabels(union_basis_values)
        
        # Calculate y-axis limits (nearest 5) based on plotted values
        if all_accs:
            min_acc = min(all_accs)
            max_acc = max(all_accs)
            y_min = int(np.floor(min_acc / 2) * 2)
            y_max = int(np.ceil(max_acc / 2) * 2)
            ax.set_ylim(y_min, y_max)
            
            # Set integer ticks with consistent spacing
            y_range = y_max - y_min
            if y_range <= 10:
                tick_step = 2
            elif y_range <= 20:
                tick_step = 2
            elif y_range <= 40:
                tick_step = 5
            else:
                tick_step = 10
            ax.set_yticks(np.arange(y_min, y_max + 1, tick_step))
        
        ax.set_xlabel("Number of Task Vector", fontsize=22, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save plot
        filename_base = f"basis_ablation_{dataset}_k{k_shot}"
        base_path = os.path.join(output_dir, filename_base)
        # plt.savefig(base_path + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(base_path + ".png", dpi=1200, bbox_inches='tight')
        # print(f"Saved: {base_path}.pdf")
        print(f"Saved: {base_path}.png")
        plt.close()


def plot_topk_ablation_per_dataset(results: List[Dict], output_dir: str = "ablation/plots/per_dataset"):
    """
    Plot svd_keep_topk vs accuracy for each dataset individually (only for num_basis_tasks=0).
    Each plot shows all models together with a legend.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    _setup_publication_style()
    
    # Filter only basis=0
    filtered_results = [r for r in results if r.get("num_basis_tasks", 0) == 0]
    
    # Group by (dataset, k_shot) -> {model: [results]}
    by_dataset = defaultdict(lambda: defaultdict(list))
    for r in filtered_results:
        key = (r.get("test_dataset", "unknown"), r.get("k_shot", 0))
        model = r.get("model", "unknown")
        by_dataset[key][model].append(r)
    
    # Color palette and model order
    palette = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBafe4"]
    preferred = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
    
    for (dataset, k_shot), model_to_data in by_dataset.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_accs = []  # Collect all plotted values for y-axis limits
        
        # Build union of topk values
        union_topk = set()
        for data in model_to_data.values():
            topk_vals = [int(r.get("svd_keep_topk", 0)) for r in data]
            union_topk.update(topk_vals)
        union_topk_values = sorted(union_topk)
        x_index_by_topk = {t: i for i, t in enumerate(union_topk_values)}
        
        # Order models
        present_models = list(model_to_data.keys())
        ordered_models = [m for m in preferred if m in present_models] + [m for m in present_models if m not in preferred]
        model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(ordered_models)}
        
        for model in ordered_models:
            data = model_to_data[model]
            # Sort by svd_keep_topk
            data_sorted = sorted(data, key=lambda x: x.get("svd_keep_topk", 0))
            
            if not data_sorted:
                continue
            
            topk_values = [r.get("svd_keep_topk", 0) for r in data_sorted]
            accs = [r.get("final_accuracy", 0.0) * 100 for r in data_sorted]
            
            # Plot with union-based x-axis
            x_positions = [x_index_by_topk[t] for t in topk_values]
            ax.plot(x_positions, accs, marker='s', linewidth=4, 
                   color=model_to_color[model], label=_format_model_name(model), zorder=3)
            all_accs.extend(accs)
        
        # Set xticks to union
        ax.set_xticks(np.arange(len(union_topk_values)))
        ax.set_xticklabels(union_topk_values)
        
        # Calculate y-axis limits (nearest 5) based on plotted values
        if all_accs:
            min_acc = min(all_accs)
            max_acc = max(all_accs)
            y_min = int(np.floor(min_acc / 2) * 2)
            y_max = int(np.ceil(max_acc / 2) * 2)
            ax.set_ylim(y_min, y_max)
            
            # Set integer ticks with consistent spacing
            y_range = y_max - y_min
            if y_range <= 10:
                tick_step = 2
            elif y_range <= 20:
                tick_step = 2
            elif y_range <= 40:
                tick_step = 5
            else:
                tick_step = 10
            ax.set_yticks(np.arange(y_min, y_max + 1, tick_step))
        
        ax.set_xlabel("SVD Keep Top-K", fontsize=22, fontweight='bold')
        ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=2)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        # Save plot
        filename_base = f"topk_ablation_{dataset}_k{k_shot}"
        base_path = os.path.join(output_dir, filename_base)
        # plt.savefig(base_path + ".pdf", dpi=300, bbox_inches='tight')
        plt.savefig(base_path + ".png", dpi=1200, bbox_inches='tight')
        # print(f"Saved: {base_path}.pdf")
        print(f"Saved: {base_path}.png")
        plt.close()


def print_summary_table(results: List[Dict]):
    """Print a summary table of all results."""
    if not results:
        print("No results to summarize.")
        return
    
    print("\n" + "="*120)
    print("ABLATION STUDY SUMMARY (Individual Results)")
    print("="*120)
    print(f"{'Dataset':<15} {'Model':<12} {'K':<5} {'TopK':<6} {'Basis':<7} {'Accuracy':<12} {'Time(s)':<10}")
    print("-"*120)
    
    # Sort by dataset, model, k, topk, basis
    sorted_results = sorted(results, key=lambda x: (
        x.get("test_dataset", ""),
        x.get("model", ""),
        x.get("k_shot", 0),
        x.get("svd_keep_topk", 0),
        x.get("num_basis_tasks", 0),
    ))
    
    for r in sorted_results:
        dataset = r.get("test_dataset", "N/A")
        model = r.get("model", "N/A")
        k = r.get("k_shot", 0)
        topk = r.get("svd_keep_topk", 0)
        basis = r.get("num_basis_tasks", 0)
        acc = r.get("final_accuracy", 0.0) * 100
        time_s = r.get("training_time_min_epoch", 0.0)
        
        basis_str = "all" if basis == 0 else str(basis)
        print(f"{dataset:<15} {model:<12} {k:<5} {topk:<6} {basis_str:<7} {acc:>10.2f}% {time_s:>9.2f}")
    
    print("="*120 + "\n")
    
    # Aggregated summary
    print("="*120)
    print("AGGREGATED SUMMARY (Mean ± Std across datasets)")
    print("="*120)
    print(f"{'Model':<12} {'K':<5} {'TopK':<6} {'Basis':<7} {'#Datasets':<10} {'Accuracy (mean±std)':<25} {'Time(s)':<10}")
    print("-"*120)
    
    # Group by (model, k_shot, svd_keep_topk, num_basis_tasks)
    aggregated = defaultdict(list)
    for r in results:
        key = (
            r.get("model", "N/A"),
            r.get("k_shot", 0),
            r.get("svd_keep_topk", 0),
            r.get("num_basis_tasks", 0),
        )
        aggregated[key].append({
            "accuracy": r.get("final_accuracy", 0.0) * 100,
            "time": r.get("training_time_min_epoch", 0.0),
        })
    
    # Sort and display aggregated results
    for key in sorted(aggregated.keys()):
        model, k, topk, basis = key
        data = aggregated[key]
        
        accs = [d["accuracy"] for d in data]
        times = [d["time"] for d in data]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        n_datasets = len(data)
        
        basis_str = "all" if basis == 0 else str(basis)
        acc_str = f"{mean_acc:.2f}% ± {std_acc:.2f}%"
        print(f"{model:<12} {k:<5} {topk:<6} {basis_str:<7} {n_datasets:<10} {acc_str:<25} {mean_time:>9.2f}")
    
    print("="*120 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--results_dir", type=str, default="ablation/results",
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="ablation/plots",
                       help="Directory to save plot images")
    parser.add_argument("--no_plots", action="store_true",
                       help="Only print summary table, don't generate plots")
    args = parser.parse_args()
    
    # Load all results
    results = load_all_results(args.results_dir)
    
    if not results:
        print("No results found. Please run ablation experiments first.")
        return
    
    # Print summary
    print_summary_table(results)
    
    if not args.no_plots:
        # Generate mean plots (across datasets)
        print("\nGenerating mean plots across datasets...")
        basis_grouped = group_results_for_basis_plot(results)
        plot_basis_ablation(basis_grouped, args.output_dir)
        
        topk_grouped = group_results_for_topk_plot(results)
        plot_topk_ablation(topk_grouped, args.output_dir)
        
        # Generate per-dataset plots
        print("\nGenerating per-dataset plots...")
        per_dataset_dir = os.path.join(args.output_dir, "per_dataset")
        plot_basis_ablation_per_dataset(results, per_dataset_dir)
        plot_topk_ablation_per_dataset(results, per_dataset_dir)
        
        print(f"\nAll plots saved to: {args.output_dir}")
        print(f"Per-dataset plots saved to: {per_dataset_dir}")


if __name__ == "__main__":
    main()

