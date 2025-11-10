"""
Aggregate and visualize remote sensing experiment results.
Groups results by model, dataset, and shot setting.
"""

import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_energy_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Parse energy config tag to extract hyperparameters.
    Format: energy_{num_tasks}_{lr}_{topk}_{init_mode}_{warmup_ratio}_{sigma_wd}
    Example: energy_14_0p001_4_average_0p1_0p0
    
    Note: Old format (without sigma_wd) is also supported for backward compatibility.
    """
    parts = config_tag.split('_')
    if len(parts) < 6 or parts[0] != 'energy':
        return {}
    
    result = {
        'num_tasks': parts[1],
        'lr': parts[2].replace('p', '.'),
        'svd_keep_topk': parts[3],
        'initialize_sigma': parts[4],
        'warmup_ratio': parts[5].replace('p', '.'),
    }
    
    # Add sigma_wd if present (new format)
    if len(parts) >= 7:
        result['sigma_wd'] = parts[6].replace('p', '.')
    else:
        result['sigma_wd'] = '0.0'  # Default for backward compatibility
    
    return result


def parse_adapter_from_filename(filename: str) -> str:
    """
    Extract adapter type from filename.
    Example: energy_results_lp++.json -> lp++
    """
    if filename.endswith('.json'):
        filename = filename[:-5]
    
    if filename.startswith('energy_results_'):
        adapter = filename[len('energy_results_'):]
        return adapter if adapter else 'none'
    elif filename.startswith('atlas_results_'):
        adapter = filename[len('atlas_results_'):]
        return adapter if adapter else 'none'
    
    return 'none'


def load_results_from_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Load results from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None


def get_accuracy_from_data(data: Dict[str, Any], adapter: str) -> Optional[float]:
    """Extract final accuracy from result data."""
    # If adapter is used, get accuracy from adapter_results
    if adapter != 'none':
        adapter_results = data.get('adapter_results', {})
        if adapter_results:
            # Get best accuracy from validation history
            val_history = adapter_results.get('validation_history', [])
            if val_history:
                best_acc = max(
                    (record.get('accuracy', 0) for record in val_history if 'accuracy' in record),
                    default=None
                )
                if best_acc is not None:
                    return best_acc * 100 if best_acc <= 1.0 else best_acc
    
    # Otherwise use final_accuracy from main results
    final_acc = data.get('final_accuracy')
    if final_acc is not None:
        return final_acc * 100 if final_acc <= 1.0 else final_acc
    
    return None


def discover_results(model_location: str) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """
    Discover all results organized by model -> dataset -> shot -> list of baselines.
    
    Returns:
        {
            'ViT-B-32': {
                'CLRS': {
                    '16shots': [
                        {
                            'method': 'Energy',
                            'lr': '0.001',
                            'svd_keep_topk': '4',
                            'initialize_sigma': 'average',
                            'warmup_ratio': '0.1',
                            'adapter': 'tip',
                            'accuracy': 85.5,
                            'config_tag': 'energy_14_0p001_4_average_0p1',
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
    
    if not os.path.exists(model_location):
        logger.error(f"Model location not found: {model_location}")
        return {}
    
    # Iterate through models
    for model_name in sorted(os.listdir(model_location)):
        model_path = os.path.join(model_location, model_name)
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
            
            # Look for config directories or shot directories
            for config_or_shot in sorted(os.listdir(dataset_path)):
                config_path = os.path.join(dataset_path, config_or_shot)
                if not os.path.isdir(config_path):
                    continue
                
                # Check if this is a shot directory (e.g., 16shots, fullshots)
                if config_or_shot.endswith('shots') or config_or_shot.endswith('shot'):
                    # Direct shot directory (legacy structure)
                    shot_name = config_or_shot if config_or_shot.endswith('shots') else f"{config_or_shot}s"
                    process_shot_directory(
                        results, model_name, dataset_name, shot_name, config_path, config_tag=None
                    )
                else:
                    # Config directory (new structure: energy_14_0p001_4_average_0p1/16shots/)
                    config_tag = config_or_shot
                    for shot_entry in sorted(os.listdir(config_path)):
                        shot_path = os.path.join(config_path, shot_entry)
                        if not os.path.isdir(shot_path):
                            continue
                        if shot_entry.endswith('shots') or shot_entry.endswith('shot'):
                            shot_name = shot_entry if shot_entry.endswith('shots') else f"{shot_entry}s"
                            process_shot_directory(
                                results, model_name, dataset_name, shot_name, shot_path, config_tag
                            )
    
    return dict(results)


def process_shot_directory(
    results: Dict,
    model_name: str,
    dataset_name: str,
    shot_name: str,
    shot_path: str,
    config_tag: Optional[str]
) -> None:
    """Process a single shot directory and extract baseline results."""
    for filename in sorted(os.listdir(shot_path)):
        if not filename.endswith('.json'):
            continue
        
        # Only process energy_results_*.json and atlas_results_*.json
        if not (filename.startswith('energy_results') or filename.startswith('atlas_results')):
            continue
        
        json_path = os.path.join(shot_path, filename)
        data = load_results_from_json(json_path)
        if not data:
            continue
        
        # Extract adapter
        adapter = parse_adapter_from_filename(filename)
        
        # Extract accuracy
        accuracy = get_accuracy_from_data(data, adapter)
        if accuracy is None:
            logger.warning(f"No accuracy found in {json_path}")
            continue
        
        # Determine method
        if filename.startswith('energy_results'):
            method = 'Energy'
            # Parse config tag
            tag_to_parse = config_tag or data.get('config_tag', '')
            hyperparams = parse_energy_config_tag(tag_to_parse) if tag_to_parse else {}
            
            baseline = {
                'method': method,
                'lr': hyperparams.get('lr', 'unknown'),
                'svd_keep_topk': hyperparams.get('svd_keep_topk', 'unknown'),
                'initialize_sigma': hyperparams.get('initialize_sigma', 'unknown'),
                'warmup_ratio': hyperparams.get('warmup_ratio', 'unknown'),
                'sigma_wd': hyperparams.get('sigma_wd', '0.0'),
                'adapter': adapter,
                'accuracy': accuracy,
                'config_tag': config_tag or '',
                'json_file': filename,
            }
        elif filename.startswith('atlas_results'):
            method = 'Atlas'
            baseline = {
                'method': method,
                'lr': data.get('lr', 'unknown'),
                'adapter': adapter,
                'accuracy': accuracy,
                'config_tag': config_tag or '',
                'json_file': filename,
            }
        else:
            continue
        
        results[model_name][dataset_name][shot_name].append(baseline)
        logger.debug(f"Added baseline: {model_name}/{dataset_name}/{shot_name} - {method} - {accuracy:.2f}%")


def format_baseline_label(baseline: Dict[str, Any]) -> str:
    """Create a readable label for a baseline."""
    method = baseline['method']
    adapter = baseline['adapter']
    
    if method == 'Energy':
        lr = baseline.get('lr', '?')
        topk = baseline.get('svd_keep_topk', '?')
        init = baseline.get('initialize_sigma', '?')
        warmup = baseline.get('warmup_ratio', '?')
        wd = baseline.get('sigma_wd', '0.0')
        
        label = f"Energy(lr={lr}, k={topk}, init={init}, w={warmup}, wd={wd})"
        if adapter != 'none':
            label += f" + {adapter.upper()}"
    elif method == 'Atlas':
        lr = baseline.get('lr', '?')
        label = f"Atlas(lr={lr})"
        if adapter != 'none':
            label += f" + {adapter.upper()}"
    else:
        label = method
        if adapter != 'none':
            label += f" + {adapter.upper()}"
    
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
    
    # Create figure
    fig_height = max(6, len(baselines) * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Create horizontal bar chart
    y_positions = range(len(labels))
    bars = ax.barh(y_positions, accuracies, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Highlight best result
    if bars:
        bars[0].set_color('darkgreen')
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(1.5)
    
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
    title = f'{model_name} | {dataset_name} | {shot_display}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    best_patch = mpatches.Patch(color='darkgreen', label='Best Result')
    other_patch = mpatches.Patch(color='steelblue', label='Other Results')
    ax.legend(handles=[best_patch, other_patch], loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{model_name}_{dataset_name}_{shot_name}.png"
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
                '16shots': {
                    'Energy_lr=0.001_k=4_init=average_w=0.1_wd=0.0': 85.5,
                    'Atlas_none': 82.3,
                    'Atlas_tip': 87.1,
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
                    adapter = baseline.get('adapter', 'none')
                    accuracy = baseline['accuracy']
                    
                    # Create unique key based on method type
                    if method == 'Energy':
                        # For Energy, use hyperparameters as key
                        lr = baseline.get('lr', 'unknown')
                        topk = baseline.get('svd_keep_topk', 'unknown')
                        init = baseline.get('initialize_sigma', 'unknown')
                        warmup = baseline.get('warmup_ratio', 'unknown')
                        wd = baseline.get('sigma_wd', '0.0')
                        key = f"Energy_lr={lr}_k={topk}_init={init}_w={warmup}_wd={wd}"
                    elif method == 'Atlas':
                        # For Atlas, distinguish by adapter
                        key = f"Atlas_{adapter}"
                    else:
                        key = f"{method}_{adapter}"
                    
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


def select_best_energy_config(averaged_results: Dict) -> Dict[str, Dict[str, tuple]]:
    """
    For each model and shot, select the best Energy hyperparameter configuration.
    
    Returns:
        {
            'ViT-B-32': {
                '16shots': ('Energy_lr=0.001_k=4_init=average_w=0.1_wd=0.0', 85.5),
                ...
            },
            ...
        }
    """
    best_energy = {}
    
    for model_name, shots in averaged_results.items():
        best_energy[model_name] = {}
        for shot_name, methods in shots.items():
            # Filter Energy methods
            energy_methods = {k: v for k, v in methods.items() if k.startswith('Energy_')}
            
            if energy_methods:
                # Select best performing Energy configuration
                best_key = max(energy_methods, key=energy_methods.get)
                best_acc = energy_methods[best_key]
                best_energy[model_name][shot_name] = (best_key, best_acc)
    
    return best_energy


def visualize_aggregated_table(
    averaged_results: Dict,
    best_energy_configs: Dict,
    output_dir: str
) -> None:
    """
    Create a table visualization similar to Table 2 in the paper.
    Shows averaged accuracy across all datasets.
    """
    # Define shot order
    shot_order = ['1shots', '2shots', '4shots', '8shots', '16shots']
    
    # Collect all unique methods
    all_methods = set()
    for model_name, shots in averaged_results.items():
        for shot_name, methods in shots.items():
            all_methods.update(methods.keys())
    
    # Separate Energy and non-Energy methods
    energy_methods = {m for m in all_methods if m.startswith('Energy_')}
    atlas_methods = {m for m in all_methods if m.startswith('Atlas_')}
    other_methods = all_methods - energy_methods - atlas_methods
    
    # Prepare method list (Energy as single entry, then Atlas variants, then others)
    method_list = ['Energy (best config)'] + sorted(atlas_methods) + sorted(other_methods)
    
    # Get models
    models = sorted(averaged_results.keys())
    
    # Create data structure for plotting
    # Structure: method -> model -> shot -> accuracy
    table_data = defaultdict(lambda: defaultdict(dict))
    energy_config_info = defaultdict(dict)
    
    for model_name in models:
        for shot_name in shot_order:
            if shot_name not in averaged_results[model_name]:
                continue
            
            methods = averaged_results[model_name][shot_name]
            
            # Handle Energy - use best config
            if model_name in best_energy_configs and shot_name in best_energy_configs[model_name]:
                best_config_key, best_acc = best_energy_configs[model_name][shot_name]
                table_data['Energy (best config)'][model_name][shot_name] = best_acc
                energy_config_info[model_name][shot_name] = best_config_key
            
            # Handle Atlas and others
            for method_key, accuracy in methods.items():
                if method_key.startswith('Atlas_'):
                    table_data[method_key][model_name][shot_name] = accuracy
                elif not method_key.startswith('Energy_'):
                    table_data[method_key][model_name][shot_name] = accuracy
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, max(8, len(method_list) * 0.6)))
    
    # Prepare data for grouped bar chart
    n_models = len(models)
    n_shots = len(shot_order)
    n_methods = len(method_list)
    
    x_positions = np.arange(n_models * n_shots)
    bar_width = 0.8 / n_methods
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    
    for method_idx, method in enumerate(method_list):
        accuracies = []
        for model in models:
            for shot in shot_order:
                acc = table_data[method].get(model, {}).get(shot, 0)
                accuracies.append(acc)
        
        offset = (method_idx - n_methods / 2) * bar_width + bar_width / 2
        ax.bar(x_positions + offset, accuracies, bar_width, 
               label=format_method_label(method), color=colors[method_idx],
               edgecolor='black', linewidth=0.5)
    
    # Customize x-axis
    x_labels = []
    for model in models:
        for shot in shot_order:
            shot_num = shot.replace('shots', '')
            x_labels.append(f"{model}\n{shot_num}-shot")
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%) - Averaged Across Datasets', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model and Shot Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Few-Shot Remote Sensing Results - Dataset-Averaged Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aggregated_results_all_datasets.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved aggregated visualization: {output_path}")
    
    # Save best Energy configurations to text file
    config_output_path = os.path.join(output_dir, 'best_energy_configs.txt')
    with open(config_output_path, 'w') as f:
        f.write("Best Energy Hyperparameter Configurations\n")
        f.write("=" * 80 + "\n\n")
        for model in models:
            f.write(f"Model: {model}\n")
            f.write("-" * 80 + "\n")
            for shot in shot_order:
                if shot in energy_config_info.get(model, {}):
                    config_key = energy_config_info[model][shot]
                    accuracy = table_data['Energy (best config)'][model][shot]
                    f.write(f"  {shot}: {config_key} (Acc: {accuracy:.2f}%)\n")
            f.write("\n")
    
    logger.info(f"Saved best Energy configurations: {config_output_path}")


def format_method_label(method_key: str) -> str:
    """Format method key for display."""
    if method_key == 'Energy (best config)':
        return 'Energy (best)'
    elif method_key.startswith('Atlas_'):
        adapter = method_key.replace('Atlas_', '')
        if adapter == 'none':
            return 'Atlas'
        else:
            return f'Atlas+{adapter.upper()}'
    else:
        return method_key


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and visualize remote sensing experiment results'
    )
    parser.add_argument(
        '--model_location',
        type=str,
        default='./models/checkpoints_remote_sensing',
        help='Root directory containing model checkpoints'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
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
        help='Specific datasets to process (e.g., CLRS MLRSNet)'
    )
    parser.add_argument(
        '--shots',
        nargs='*',
        default=None,
        help='Specific shots to process (e.g., 16shots 8shots)'
    )
    parser.add_argument(
        '--aggregate_only',
        action='store_true',
        help='Only generate aggregated visualization (skip individual charts)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Remote Sensing Results Aggregation")
    logger.info("=" * 80)
    logger.info(f"Model location: {args.model_location}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Discover all results
    logger.info("\nDiscovering results...")
    all_results = discover_results(args.model_location)
    
    if not all_results:
        logger.error("No results found!")
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
    
    # Generate aggregated visualization
    logger.info("\nAggregating results across datasets...")
    averaged_results = aggregate_across_datasets(all_results)
    
    logger.info("Selecting best Energy configurations...")
    best_energy_configs = select_best_energy_config(averaged_results)
    
    logger.info("Generating aggregated visualization...")
    visualize_aggregated_table(averaged_results, best_energy_configs, args.output_dir)
    
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
