"""
Aggregate and visualize UFM (test-time adaptation) experiment results.
Groups results by model, dataset, and shot setting.

This script loads UFM-Atlas and UFM-Energy results from models/checkpoints_tta
and aggregates them similar to eval_single_task.py.
"""

import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_ufm_energy_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Parse UFM-Energy config tag to extract hyperparameters.
    Format: ufm_energy_{num_tasks}_{lr}_{topk}_{init_mode}_{warmup_ratio}_{sigma_wd}
    Example: ufm_energy_16_0p001_12_average_0p1_0
    """
    parts = config_tag.split('_')
    if len(parts) < 7 or not config_tag.startswith('ufm_energy_'):
        return {}
    
    result = {
        'num_tasks': parts[2],
        'lr': parts[3].replace('p', '.'),
        'svd_keep_topk': parts[4],
        'initialize_sigma': parts[5],
        'warmup_ratio': parts[6].replace('p', '.'),
    }
    
    # Add sigma_wd if present
    if len(parts) >= 8:
        result['sigma_wd'] = parts[7].replace('p', '.')
    else:
        result['sigma_wd'] = '0.0'
    
    return result


def parse_ufm_atlas_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Parse UFM-Atlas config tag to extract hyperparameters.
    Format: ufm_atlas_{num_basis}_{lr}
    Example: ufm_atlas_16_0p1
    """
    parts = config_tag.split('_')
    if len(parts) < 4 or not config_tag.startswith('ufm_atlas_'):
        return {}
    
    result = {
        'num_basis': parts[2],
        'lr': parts[3].replace('p', '.'),
    }
    
    return result


def load_results_from_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Load results from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None


def get_accuracy_from_ufm_data(data: Dict[str, Any]) -> Optional[float]:
    """Extract final accuracy from UFM result data."""
    final_acc = data.get('final_accuracy')
    if final_acc is not None:
        # Convert to percentage if needed
        return final_acc * 100 if final_acc <= 1.0 else final_acc
    
    return None


def discover_ufm_results(tta_model_location: str) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Discover all UFM results organized by model -> dataset -> shot -> list of baselines.
    
    Returns:
        {
            'ViT-B-32': {
                'CIFAR10': {
                    'fullshots': [
                        {
                            'method': 'UFM-Energy',
                            'lr': '0.001',
                            'svd_keep_topk': '12',
                            'initialize_sigma': 'average',
                            'warmup_ratio': '0.1',
                            'sigma_wd': '0.0',
                            'accuracy': 83.54,
                            'config_tag': 'ufm_energy_16_0p001_12_average_0p1_0',
                        },
                        {
                            'method': 'UFM-Atlas',
                            'lr': '0.1',
                            'accuracy': 82.30,
                            'config_tag': 'ufm_atlas_16_0p1',
                        },
                        ...
                    ],
                    ...
                },
                ...
            },
            ...
        }
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    if not os.path.exists(tta_model_location):
        logger.error(f"TTA model location not found: {tta_model_location}")
        return {}
    
    # Iterate through models
    for model_name in sorted(os.listdir(tta_model_location)):
        model_path = os.path.join(tta_model_location, model_name)
        if not os.path.isdir(model_path):
            continue
        
        logger.info(f"Processing model: {model_name}")
        
        # Iterate through datasets
        for dataset_entry in sorted(os.listdir(model_path)):
            dataset_path = os.path.join(model_path, dataset_entry)
            if not os.path.isdir(dataset_path):
                continue
            
            # Remove 'Val' suffix if present
            dataset_name = dataset_entry.replace('Val', '') if dataset_entry.endswith('Val') else dataset_entry
            
            # Look for config directories
            for config_entry in sorted(os.listdir(dataset_path)):
                config_path = os.path.join(dataset_path, config_entry)
                if not os.path.isdir(config_path):
                    continue
                
                config_tag = config_entry
                
                # Look for shot directories inside config
                for shot_entry in sorted(os.listdir(config_path)):
                    shot_path = os.path.join(config_path, shot_entry)
                    if not os.path.isdir(shot_path):
                        continue
                    
                    if shot_entry.endswith('shots') or shot_entry.endswith('shot'):
                        shot_name = shot_entry if shot_entry.endswith('shots') else f"{shot_entry}s"
                        process_ufm_shot_directory(
                            results, model_name, dataset_name, shot_name, shot_path, config_tag
                        )
    
    return dict(results)


def process_ufm_shot_directory(
    results: Dict,
    model_name: str,
    dataset_name: str,
    shot_name: str,
    shot_path: str,
    config_tag: str
) -> None:
    """Process a single shot directory and extract UFM baseline results."""
    for filename in sorted(os.listdir(shot_path)):
        if not filename.endswith('.json'):
            continue
        
        # Process UFM energy and atlas result files
        is_ufm_energy = filename == 'ufm_energy_results_none.json'
        is_ufm_atlas = filename == 'ufm_atlas_results_none.json'
        
        if not (is_ufm_energy or is_ufm_atlas):
            continue
        
        json_path = os.path.join(shot_path, filename)
        data = load_results_from_json(json_path)
        if not data:
            continue
        
        # Extract accuracy
        accuracy = get_accuracy_from_ufm_data(data)
        if accuracy is None:
            logger.warning(f"No accuracy found in {json_path}")
            continue
        
        # Determine method
        if is_ufm_energy:
            method = 'UFM-Energy'
            hyperparams = parse_ufm_energy_config_tag(config_tag)
            
            baseline = {
                'method': method,
                'lr': hyperparams.get('lr', 'unknown'),
                'svd_keep_topk': hyperparams.get('svd_keep_topk', 'unknown'),
                'initialize_sigma': hyperparams.get('initialize_sigma', 'unknown'),
                'warmup_ratio': hyperparams.get('warmup_ratio', 'unknown'),
                'sigma_wd': hyperparams.get('sigma_wd', '0.0'),
                'accuracy': accuracy,
                'config_tag': config_tag,
                'json_file': filename,
                'sigma_alpha': data.get('sigma_alpha', None),
                'trainable_params': data.get('trainable_params', None),
            }
        elif is_ufm_atlas:
            method = 'UFM-Atlas'
            hyperparams = parse_ufm_atlas_config_tag(config_tag)
            
            baseline = {
                'method': method,
                'lr': hyperparams.get('lr', 'unknown'),
                'num_basis': hyperparams.get('num_basis', 'unknown'),
                'accuracy': accuracy,
                'config_tag': config_tag,
                'json_file': filename,
            }
        else:
            return
        
        results[model_name][dataset_name][shot_name].append(baseline)
        logger.debug(f"Added baseline: {model_name}/{dataset_name}/{shot_name} - {method} - {accuracy:.2f}%")


def format_baseline_label(baseline: Dict[str, Any]) -> str:
    """Create a readable label for a baseline."""
    method = baseline['method']
    
    if method == 'UFM-Energy':
        lr = baseline.get('lr', '?')
        topk = baseline.get('svd_keep_topk', '?')
        init = baseline.get('initialize_sigma', '?')
        warmup = baseline.get('warmup_ratio', '?')
        wd = baseline.get('sigma_wd', '0.0')
        label = f"UFM-Energy(lr={lr}, k={topk}, init={init}, w={warmup}, wd={wd})"
    elif method == 'UFM-Atlas':
        lr = baseline.get('lr', '?')
        label = f"UFM-Atlas(lr={lr})"
    else:
        label = method
    
    return label


def visualize_shot_results(
    model_name: str,
    dataset_name: str,
    shot_name: str,
    baselines: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """Create a visualization for a specific model/dataset/shot combination."""
    if not baselines:
        logger.warning(f"No baselines to visualize for {model_name}/{dataset_name}/{shot_name}")
        return
    
    # Sort baselines by accuracy (descending)
    baselines_sorted = sorted(baselines, key=lambda x: x['accuracy'], reverse=True)
    
    # Prepare data for visualization
    labels = [format_baseline_label(b) for b in baselines_sorted]
    accuracies = [b['accuracy'] for b in baselines_sorted]
    
    # Assign colors based on method
    colors = []
    for b in baselines_sorted:
        if b['method'] == 'UFM-Energy':
            colors.append('steelblue')
        elif b['method'] == 'UFM-Atlas':
            colors.append('coral')
        else:
            colors.append('gray')
    
    # Create figure
    fig_height = max(6, len(baselines) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Create horizontal bar chart
    y_positions = range(len(labels))
    bars = ax.barh(y_positions, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Highlight best result
    if bars:
        bars[0].set_edgecolor('darkgreen')
        bars[0].set_linewidth=2.5
    
    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        ax.text(
            width + 1, bar.get_y() + bar.get_height() / 2,
            f'{acc:.2f}%',
            ha='left', va='center',
            fontsize=9,
            fontweight='bold' if i == 0 else 'normal',
            color='darkgreen' if i == 0 else 'black'
        )
    
    # Title
    shot_display = shot_name.replace('shots', '-shot').replace('fullshots', 'Full-shot')
    title = f'UFM TTA Results | {model_name} | {dataset_name} | {shot_display}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    energy_patch = mpatches.Patch(color='steelblue', label='UFM-Energy')
    atlas_patch = mpatches.Patch(color='coral', label='UFM-Atlas')
    ax.legend(handles=[energy_patch, atlas_patch], loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"ufm_{model_name}_{dataset_name}_{shot_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved: {output_path}")


def aggregate_across_datasets(all_results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate results across all datasets to get average accuracy.
    
    Returns:
        {
            'ViT-B-32': {
                'fullshots': {
                    'UFM-Energy_lr=0.001_k=12_init=average_w=0.1_wd=0.0': 85.5,
                    'UFM-Atlas_lr=0.1': 82.3,
                    ...
                },
                ...
            },
            ...
        }
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for model_name, datasets in all_results.items():
        for dataset_name, shots in datasets.items():
            for shot_name, baselines in shots.items():
                for baseline in baselines:
                    method = baseline['method']
                    accuracy = baseline['accuracy']
                    
                    # Create unique key based on method type
                    if method == 'UFM-Energy':
                        lr = baseline.get('lr', 'unknown')
                        topk = baseline.get('svd_keep_topk', 'unknown')
                        init = baseline.get('initialize_sigma', 'unknown')
                        warmup = baseline.get('warmup_ratio', 'unknown')
                        wd = baseline.get('sigma_wd', '0.0')
                        key = f"UFM-Energy_lr={lr}_k={topk}_init={init}_w={warmup}_wd={wd}"
                    elif method == 'UFM-Atlas':
                        lr = baseline.get('lr', 'unknown')
                        key = f"UFM-Atlas_lr={lr}"
                    else:
                        key = method
                    
                    aggregated[model_name][shot_name][key].append(accuracy)
    
    # Average the accuracies
    averaged = {}
    for model_name, shots in aggregated.items():
        averaged[model_name] = {}
        for shot_name, methods in shots.items():
            averaged[model_name][shot_name] = {}
            for method_key, accuracies in methods.items():
                averaged[model_name][shot_name][method_key] = sum(accuracies) / len(accuracies)
    
    return averaged


def select_best_method_config_per_dataset(
    all_results: Dict,
    method_name: str
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    For each dataset, model, and shot, select the best hyperparameter configuration for a given method.
    
    Args:
        all_results: Results dictionary
        method_name: Name of the method (e.g., 'UFM-Energy', 'UFM-Atlas')
    
    Returns:
        {
            'CIFAR10': {
                'ViT-B-32': {
                    'fullshots': (config_key, accuracy, config_tag),
                    ...
                },
                ...
            },
            ...
        }
    """
    best_config_per_dataset = defaultdict(lambda: defaultdict(dict))
    
    for model_name, datasets in all_results.items():
        for dataset_name, shots in datasets.items():
            for shot_name, baselines in shots.items():
                # Filter baselines for this method
                method_baselines = [
                    b for b in baselines if b['method'] == method_name
                ]
                
                if method_baselines:
                    # Select best performing configuration
                    best_baseline = max(method_baselines, key=lambda b: b['accuracy'])
                    
                    # Build config key based on method type
                    if method_name == 'UFM-Energy':
                        lr = best_baseline.get('lr', 'unknown')
                        topk = best_baseline.get('svd_keep_topk', 'unknown')
                        init = best_baseline.get('initialize_sigma', 'unknown')
                        warmup = best_baseline.get('warmup_ratio', 'unknown')
                        wd = best_baseline.get('sigma_wd', '0.0')
                        config_key = f"UFM-Energy_lr={lr}_k={topk}_init={init}_w={warmup}_wd={wd}"
                    elif method_name == 'UFM-Atlas':
                        lr = best_baseline.get('lr', 'unknown')
                        config_key = f"UFM-Atlas_lr={lr}"
                    else:
                        config_key = method_name
                    
                    best_config_per_dataset[dataset_name][model_name][shot_name] = (
                        config_key,
                        best_baseline['accuracy'],
                        best_baseline.get('config_tag', '')
                    )
    
    return dict(best_config_per_dataset)


def compute_averaged_best_configs_from_aggregated(
    averaged_results: Dict,
    method_prefix: str
) -> Dict[str, Dict[str, tuple]]:
    """
    Select best configs from already-averaged results across all datasets.
    
    Args:
        averaged_results: Already averaged results from aggregate_across_datasets()
        method_prefix: Method prefix to filter (e.g., 'UFM-Energy', 'UFM-Atlas')
    
    Returns:
        {
            'ViT-B-32': {
                'fullshots': (config_key, avg_accuracy),
                ...
            },
            ...
        }
    """
    best_configs = {}
    
    for model_name, shots in averaged_results.items():
        best_configs[model_name] = {}
        for shot_name, methods in shots.items():
            # Filter methods matching the prefix
            matching_configs = {
                config_key: accuracy
                for config_key, accuracy in methods.items()
                if config_key.startswith(method_prefix)
            }
            
            # Select best config based on average accuracy
            if matching_configs:
                best_key = max(matching_configs, key=matching_configs.get)
                best_acc = matching_configs[best_key]
                best_configs[model_name][shot_name] = (best_key, best_acc)
    
    return best_configs


def visualize_aggregated_table(
    averaged_results: Dict,
    best_ufm_energy_configs: Dict,
    best_ufm_atlas_configs: Dict,
    output_dir: str
) -> None:
    """
    Create a table visualization showing averaged accuracy across all datasets.
    
    Args:
        averaged_results: Averaged accuracy for all methods/configs
        best_ufm_energy_configs: Best UFM-Energy configs per model/shot
        best_ufm_atlas_configs: Best UFM-Atlas configs per model/shot
        output_dir: Output directory
    """
    # Define shot order
    shot_order = ['fullshots', '0shots']  # TTA typically uses fullshot (0-shot means zero labeled samples)
    
    # Get models
    models = sorted(averaged_results.keys())
    
    # Create data structure for table
    # Structure: method -> model -> shot -> accuracy
    table_data = defaultdict(lambda: defaultdict(dict))
    best_config_info = defaultdict(lambda: defaultdict(dict))
    
    for model_name in models:
        for shot_name in shot_order:
            if shot_name not in averaged_results[model_name]:
                continue
            
            methods = averaged_results[model_name][shot_name]
            
            # Handle UFM-Energy - use best config
            if model_name in best_ufm_energy_configs and shot_name in best_ufm_energy_configs[model_name]:
                best_config_key, best_acc = best_ufm_energy_configs[model_name][shot_name]
                table_data['UFM-Energy (best config)'][model_name][shot_name] = best_acc
                best_config_info['UFM-Energy'][model_name][shot_name] = best_config_key
            
            # Handle UFM-Atlas - use best config
            if model_name in best_ufm_atlas_configs and shot_name in best_ufm_atlas_configs[model_name]:
                best_config_key, best_acc = best_ufm_atlas_configs[model_name][shot_name]
                table_data['UFM-Atlas (best config)'][model_name][shot_name] = best_acc
                best_config_info['UFM-Atlas'][model_name][shot_name] = best_config_key
    
    # Method list
    method_list = ['UFM-Energy (best config)', 'UFM-Atlas (best config)']
    
    # Create table visualization
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare column headers
    col_labels = ['Method']
    for model in models:
        for shot in shot_order:
            if shot in averaged_results.get(model, {}):
                shot_display = shot.replace('shots', '').replace('full', 'Full')
                col_labels.append(f"{model}\n{shot_display}")
    
    # Prepare table data
    table_content = []
    for method in method_list:
        row = [method.replace(' (best config)', ' (best)')]
        for model in models:
            for shot in shot_order:
                if shot not in averaged_results.get(model, {}):
                    continue
                acc = table_data[method].get(model, {}).get(shot, None)
                if acc is not None:
                    row.append(f"{acc:.2f}")
                else:
                    row.append("-")
        table_content.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_content,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Color method column
    for i in range(len(table_content)):
        cell = table[(i+1, 0)]
        cell.set_facecolor('#E8F5E9')
        cell.set_text_props(weight='bold', ha='left')
    
    # Find best values per column and highlight
    for col_idx in range(1, len(col_labels)):
        values = []
        for row_idx in range(len(table_content)):
            cell_text = table_content[row_idx][col_idx]
            if cell_text != "-":
                try:
                    values.append((float(cell_text), row_idx))
                except:
                    pass
        
        if values:
            max_val, max_row_idx = max(values, key=lambda x: x[0])
            cell = table[(max_row_idx + 1, col_idx)]
            cell.set_facecolor('#FFEB3B')
            cell.set_text_props(weight='bold')
    
    plt.title('UFM Test-Time Adaptation Results - Dataset-Averaged Accuracy (%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ufm_aggregated_results_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved aggregated table: {output_path}")
    
    # Save as CSV
    csv_output_path = os.path.join(output_dir, 'ufm_aggregated_results_table.csv')
    with open(csv_output_path, 'w') as f:
        f.write(','.join(col_labels) + '\n')
        for row in table_content:
            f.write(','.join(row) + '\n')
    
    logger.info(f"Saved CSV table: {csv_output_path}")
    
    # Save best configurations to text file
    config_output_path = os.path.join(output_dir, 'ufm_best_configs.txt')
    with open(config_output_path, 'w') as f:
        f.write("Best Hyperparameter Configurations for UFM Methods\n")
        f.write("=" * 80 + "\n\n")
        
        for method_name in ['UFM-Energy', 'UFM-Atlas']:
            if method_name in best_config_info and best_config_info[method_name]:
                f.write(f"Method: {method_name}\n")
                f.write("=" * 80 + "\n")
                for model in models:
                    if model in best_config_info[method_name]:
                        f.write(f"  Model: {model}\n")
                        f.write("  " + "-" * 78 + "\n")
                        for shot in shot_order:
                            if shot in best_config_info[method_name].get(model, {}):
                                config_key = best_config_info[method_name][model][shot]
                                accuracy = table_data[f'{method_name} (best config)'][model][shot]
                                f.write(f"    {shot}: {config_key} (Acc: {accuracy:.2f}%)\n")
                        f.write("\n")
                f.write("\n")
    
    logger.info(f"Saved best configurations: {config_output_path}")


def create_comprehensive_table(
    all_results: Dict,
    best_ufm_energy_per_dataset: Dict,
    best_ufm_atlas_per_dataset: Dict,
    output_dir: str
) -> None:
    """
    Create a comprehensive table with datasets as columns and average.
    
    Args:
        all_results: All results
        best_ufm_energy_per_dataset: Best UFM-Energy configs per dataset
        best_ufm_atlas_per_dataset: Best UFM-Atlas configs per dataset
        output_dir: Output directory
    """
    # Define shot order
    shot_order = ['fullshots', '0shots']
    
    # Collect all datasets
    all_datasets = set()
    for model_data in all_results.values():
        all_datasets.update(model_data.keys())
    all_datasets = sorted(all_datasets)
    
    # Collect all models
    models = sorted(all_results.keys())
    
    # Build data structure: [model][shot][method][dataset] = accuracy
    data_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name in models:
        for dataset_name in all_results[model_name]:
            for shot_name in shot_order:
                if shot_name not in all_results[model_name][dataset_name]:
                    continue
                
                # For UFM-Energy, use dataset-specific best config
                if dataset_name in best_ufm_energy_per_dataset:
                    if model_name in best_ufm_energy_per_dataset[dataset_name]:
                        if shot_name in best_ufm_energy_per_dataset[dataset_name][model_name]:
                            _, best_acc, _ = best_ufm_energy_per_dataset[dataset_name][model_name][shot_name]
                            data_structure[model_name][shot_name]['UFM-Energy (best)'][dataset_name] = best_acc
                
                # For UFM-Atlas, use dataset-specific best config
                if dataset_name in best_ufm_atlas_per_dataset:
                    if model_name in best_ufm_atlas_per_dataset[dataset_name]:
                        if shot_name in best_ufm_atlas_per_dataset[dataset_name][model_name]:
                            _, best_acc, _ = best_ufm_atlas_per_dataset[dataset_name][model_name][shot_name]
                            data_structure[model_name][shot_name]['UFM-Atlas (best)'][dataset_name] = best_acc
    
    # Create table
    col_labels = ['Model', 'Shot', 'Method'] + all_datasets + ['Average']
    
    # Prepare table content
    table_content = []
    method_list = ['UFM-Energy (best)', 'UFM-Atlas (best)']
    
    for model_idx, model_name in enumerate(models):
        for shot_idx, shot_name in enumerate(shot_order):
            # Check if this shot exists for this model
            shot_exists = False
            for dataset_name in all_datasets:
                if shot_name in all_results.get(model_name, {}).get(dataset_name, {}):
                    shot_exists = True
                    break
            
            if not shot_exists:
                continue
            
            shot_display = shot_name.replace('shots', '').replace('full', 'Full')
            
            for method_idx, method_key in enumerate(method_list):
                # First method in shot shows model name (for first shot only) and shot number
                if method_idx == 0:
                    if shot_idx == 0:
                        row = [model_name, shot_display, method_key]
                    else:
                        row = ['', shot_display, method_key]
                else:
                    row = ['', '', method_key]
                
                # Get accuracy for each dataset
                accuracies = []
                for dataset_name in all_datasets:
                    acc = data_structure[model_name][shot_name][method_key].get(dataset_name, None)
                    if acc is not None:
                        row.append(f"{acc:.2f}")
                        accuracies.append(acc)
                    else:
                        row.append("-")
                
                # Calculate average
                if accuracies:
                    avg = sum(accuracies) / len(accuracies)
                    row.append(f"{avg:.2f}")
                else:
                    row.append("-")
                
                table_content.append(row)
            
            # Add separator row between shots
            if shot_idx < len(shot_order) - 1:
                table_content.append([''] * len(col_labels))
        
        # Add separator row between models
        if model_idx < len(models) - 1:
            table_content.append([''] * len(col_labels))
    
    # Create figure
    n_rows = len(table_content)
    n_cols = len(col_labels)
    fig_height = max(8, n_rows * 0.3 + 2)
    fig_width = max(16, n_cols * 1.0)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_content,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Color header row
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=9)
    
    # Color first three columns
    for i in range(len(table_content)):
        for j in range(3):
            cell = table[(i+1, j)]
            if table_content[i][j]:
                cell.set_facecolor('#E8F5E9')
                if j == 0:
                    cell.set_text_props(weight='bold', ha='center', fontsize=9)
                elif j == 1:
                    cell.set_text_props(weight='bold', ha='center')
                else:
                    cell.set_text_props(weight='bold', ha='left')
    
    plt.title('UFM Test-Time Adaptation Results - All Datasets and Average', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ufm_comprehensive_results_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved comprehensive table: {output_path}")
    
    # Save as CSV
    csv_output_path = os.path.join(output_dir, 'ufm_comprehensive_results_table.csv')
    with open(csv_output_path, 'w') as f:
        f.write(','.join(col_labels) + '\n')
        for row in table_content:
            f.write(','.join(row) + '\n')
    
    logger.info(f"Saved CSV: {csv_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and visualize UFM (test-time adaptation) experiment results'
    )
    parser.add_argument(
        '--model_location',
        type=str,
        default='./models/checkpoints_tta',
        help='Root directory containing TTA model checkpoints'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results_tta',
        help='Directory to save visualization results'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='Specific models to process (e.g., ViT-B-32 ViT-B-16)'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help='Specific datasets to process (e.g., CIFAR10 CIFAR100)'
    )
    parser.add_argument(
        '--shots',
        nargs='*',
        default=None,
        help='Specific shots to process (e.g., fullshots)'
    )
    parser.add_argument(
        '--aggregate_only',
        action='store_true',
        help='Only generate aggregated visualization (skip individual charts)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("UFM Test-Time Adaptation Results Aggregation")
    logger.info("=" * 80)
    logger.info(f"TTA model location: {args.tta_model_location}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Discover all UFM results
    logger.info("\nDiscovering UFM results...")
    all_results = discover_ufm_results(args.tta_model_location)
    
    if not all_results:
        logger.error("No UFM results found!")
        return
    
    # Apply filters
    if args.models:
        all_results = {k: v for k, v in all_results.items() if k in args.models}
    if args.datasets:
        for model in all_results:
            all_results[model] = {k: v for k, v in all_results[model].items() if k in args.datasets}
    if args.shots:
        shots_normalized = [s if s.endswith('shots') else f"{s}shots" for s in args.shots]
        for model in all_results:
            for dataset in all_results[model]:
                all_results[model][dataset] = {
                    k: v for k, v in all_results[model][dataset].items() if k in shots_normalized
                }
    
    logger.info(f"\nFound {len(all_results)} models")
    
    # Select best configurations per dataset
    logger.info("\nSelecting best UFM-Energy configurations per dataset...")
    best_ufm_energy_per_dataset = select_best_method_config_per_dataset(all_results, 'UFM-Energy')
    
    logger.info("Selecting best UFM-Atlas configurations per dataset...")
    best_ufm_atlas_per_dataset = select_best_method_config_per_dataset(all_results, 'UFM-Atlas')
    
    # Save best configurations to JSON
    best_config_path = os.path.join(args.output_dir, 'ufm_best_configs_per_dataset.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(best_config_path, 'w') as f:
        serializable_config = {}
        
        # Add UFM-Energy configs
        serializable_config['UFM-Energy'] = {}
        for dataset, models in best_ufm_energy_per_dataset.items():
            serializable_config['UFM-Energy'][dataset] = {}
            for model, shots in models.items():
                serializable_config['UFM-Energy'][dataset][model] = {}
                for shot, (config_key, accuracy, config_tag) in shots.items():
                    serializable_config['UFM-Energy'][dataset][model][shot] = {
                        'config_key': config_key,
                        'accuracy': accuracy,
                        'config_tag': config_tag
                    }
        
        # Add UFM-Atlas configs
        serializable_config['UFM-Atlas'] = {}
        for dataset, models in best_ufm_atlas_per_dataset.items():
            serializable_config['UFM-Atlas'][dataset] = {}
            for model, shots in models.items():
                serializable_config['UFM-Atlas'][dataset][model] = {}
                for shot, (config_key, accuracy, config_tag) in shots.items():
                    serializable_config['UFM-Atlas'][dataset][model][shot] = {
                        'config_key': config_key,
                        'accuracy': accuracy,
                        'config_tag': config_tag
                    }
        
        json.dump(serializable_config, f, indent=2)
    logger.info(f"Saved best configs per dataset to: {best_config_path}")
    
    # Generate aggregated visualization
    logger.info("\nAggregating results across datasets...")
    averaged_results = aggregate_across_datasets(all_results)
    
    # Compute best configs from averaged results
    logger.info("Computing best UFM-Energy configuration from averaged results...")
    best_ufm_energy_configs_avg = compute_averaged_best_configs_from_aggregated(averaged_results, 'UFM-Energy')
    
    logger.info("Computing best UFM-Atlas configuration from averaged results...")
    best_ufm_atlas_configs_avg = compute_averaged_best_configs_from_aggregated(averaged_results, 'UFM-Atlas')
    
    logger.info("Generating aggregated visualization...")
    visualize_aggregated_table(
        averaged_results,
        best_ufm_energy_configs_avg,
        best_ufm_atlas_configs_avg,
        args.output_dir
    )
    
    # Generate comprehensive table
    logger.info("\nGenerating comprehensive table (all datasets + average)...")
    create_comprehensive_table(
        all_results,
        best_ufm_energy_per_dataset,
        best_ufm_atlas_per_dataset,
        args.output_dir
    )
    
    # Generate individual visualizations if not aggregate_only
    if not args.aggregate_only:
        total_combinations = sum(
            len(shots_dict)
            for dataset_dict in all_results.values()
            for shots_dict in dataset_dict.values()
        )
        
        logger.info(f"\nTotal combinations to visualize: {total_combinations}")
        logger.info("\nGenerating individual visualizations...")
        count = 0
        for model_name, datasets in sorted(all_results.items()):
            for dataset_name, shots in sorted(datasets.items()):
                for shot_name, baselines in sorted(shots.items()):
                    count += 1
                    logger.info(f"[{count}/{total_combinations}] Processing {model_name}/{dataset_name}/{shot_name} ({len(baselines)} baselines)")
                    visualize_shot_results(
                        model_name, dataset_name, shot_name, baselines, args.output_dir
                    )
    
    logger.info("=" * 80)
    logger.info(f"✓ Complete! Results saved to {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

