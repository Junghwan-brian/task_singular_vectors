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
    Format: energy_{num_tasks}_{lr}_{topk}_{init_mode}_{warmup_ratio}
    Example: energy_14_0p001_4_average_0p1
    """
    parts = config_tag.split('_')
    if len(parts) < 6 or parts[0] != 'energy':
        return {}
    
    return {
        'num_tasks': parts[1],
        'lr': parts[2].replace('p', '.'),
        'svd_keep_topk': parts[3],
        'initialize_sigma': parts[4],
        'warmup_ratio': parts[5].replace('p', '.'),
    }


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
        
        label = f"Energy(lr={lr}, k={topk}, init={init}, w={warmup})"
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
    
    # Count total combinations
    total_combinations = sum(
        len(shots_dict)
        for dataset_dict in all_results.values()
        for shots_dict in dataset_dict.values()
    )
    
    logger.info(f"\nFound {len(all_results)} models")
    logger.info(f"Total combinations to visualize: {total_combinations}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
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
