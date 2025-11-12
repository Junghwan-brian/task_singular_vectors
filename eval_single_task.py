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


def parse_baseline_config_tag(config_tag: str, method: str) -> Dict[str, str]:
    """
    Parse baseline config tag to extract hyperparameters.
    
    LinearProbe format: baseline_lp_{lr}_{epochs}
    Example: baseline_lp_0p1_100
    
    LoRA format: baseline_lora_{r}_{alpha}_{lr}_{epochs}
    Example: baseline_lora_8_16_0p001_50
    
    TIP format: baseline_tip_{alpha}_{beta}_{epochs}
    Example: baseline_tip_1p0_5p5_20
    
    LP++ format: baseline_lpp_{ratio}_{lambda}
    Example: baseline_lpp_0p2_10
    """
    if not config_tag:
        return {}
    
    parts = config_tag.split('_')
    
    if config_tag.startswith('baseline_lp_') and method == 'LinearProbe':
        # baseline_lp_{lr}_{epochs}
        if len(parts) >= 4:
            return {
                'lr': parts[2].replace('p', '.'),
                'epochs': parts[3],
            }
    elif config_tag.startswith('baseline_lora_') and method == 'LoRA':
        # baseline_lora_{r}_{alpha}_{lr}_{epochs}
        if len(parts) >= 6:
            return {
                'r': parts[2],
                'alpha': parts[3].replace('p', '.'),
                'lr': parts[4].replace('p', '.'),
                'epochs': parts[5] if len(parts) > 5 else 'unknown',
            }
    elif config_tag.startswith('baseline_tip_') and method == 'TIP':
        # baseline_tip_{alpha}_{beta}_{epochs}
        if len(parts) >= 5:
            return {
                'alpha': parts[2].replace('p', '.'),
                'beta': parts[3].replace('p', '.'),
                'epochs': parts[4] if len(parts) > 4 else 'unknown',
            }
    elif config_tag.startswith('baseline_lpp_') and method == 'LP++':
        # baseline_lpp_{ratio}_{lambda}
        if len(parts) >= 4:
            return {
                'ratio': parts[2].replace('p', '.'),
                'lambda': parts[3].replace('p', '.'),
            }
    
    return {}


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
    # For non-adapter runs, prefer best from root-level validation_history if available
    root_val_history = data.get('validation_history', [])
    if root_val_history:
        best_root_acc = max(
            (record.get('accuracy', 0) for record in root_val_history if 'accuracy' in record),
            default=None
        )
        if best_root_acc is not None:
            return best_root_acc * 100 if best_root_acc <= 1.0 else best_root_acc
            
    # Otherwise use final_accuracy from main results
    final_acc = data.get('final_accuracy')
    if final_acc is not None:
        return final_acc * 100 if final_acc <= 1.0 else final_acc
    
    return None


def get_params_and_memory_from_data(data: Dict[str, Any], adapter: str) -> tuple[Optional[int], Optional[float]]:
    """Extract trainable params and GPU memory from result data."""
    # If adapter is used, get from adapter_results
    if adapter != 'none':
        adapter_results = data.get('adapter_results', {})
        if adapter_results:
            params = adapter_results.get('trainable_params')
            memory = adapter_results.get('gpu_peak_mem_mb')
            if params is not None or memory is not None:
                return (params, memory)
    
    # Otherwise get from main results
    params = data.get('trainable_params')
    memory = data.get('gpu_peak_mem_mb')
    return (params, memory)


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
        
        # Process energy, atlas, and baseline result files
        is_energy = filename.startswith('energy_results')
        is_atlas = filename.startswith('atlas_results')
        is_baseline = filename.startswith('baseline_results') or (filename.endswith('_results.json') and not (is_energy or is_atlas))
        
        if not (is_energy or is_atlas or is_baseline):
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
        
        # Extract params and memory
        trainable_params, gpu_memory = get_params_and_memory_from_data(data, adapter)
        
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
                'trainable_params': trainable_params,
                'gpu_memory': gpu_memory,
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
                'trainable_params': trainable_params,
                'gpu_memory': gpu_memory,
            }
        elif is_baseline:
            # Handle baseline methods: linear_probe, tip, lora, lp++, zeroshot
            method_name = data.get('method', 'Unknown')
            
            # Map method names to display names (remove "baseline" prefix)
            method_display_map = {
                'linear_probe': 'LinearProbe',
                'tip': 'TIP',
                'tip_adapter': 'TIP',
                'lpp': 'LP++',
                'lp++': 'LP++',
                'lp_plus_plus': 'LP++',
                'lora': 'LoRA',
                'zeroshot': 'ZeroShot',
            }
            
            method = method_display_map.get(method_name, method_name.title())
            
            # Parse baseline hyperparameters from config_tag
            tag_to_parse = config_tag or data.get('config_tag', '')
            hyperparams = parse_baseline_config_tag(tag_to_parse, method) if tag_to_parse else {}
            
            baseline = {
                'method': method,
                'adapter': 'none',
                'accuracy': accuracy,
                'config_tag': config_tag or '',
                'json_file': filename,
                'hyperparams': hyperparams,  # Store parsed hyperparameters
                'trainable_params': trainable_params,
                'gpu_memory': gpu_memory,
            }
        else:
            continue
        
        results[model_name][dataset_name][shot_name].append(baseline)
        logger.debug(f"Added baseline: {model_name}/{dataset_name}/{shot_name} - {method} - {accuracy:.2f}%")


def format_baseline_label(baseline: Dict[str, Any]) -> str:
    """Create a readable label for a baseline."""
    method = baseline['method']
    adapter = baseline.get('adapter', 'none')
    
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
    elif method == 'LinearProbe':
        # Show LinearProbe hyperparameters
        hyperparams = baseline.get('hyperparams', {})
        if hyperparams:
            lr = hyperparams.get('lr', '?')
            epochs = hyperparams.get('epochs', '?')
            label = f"LinearProbe(lr={lr}, ep={epochs})"
        else:
            label = method
    elif method == 'TIP':
        # Show TIP hyperparameters
        hyperparams = baseline.get('hyperparams', {})
        if hyperparams:
            alpha = hyperparams.get('alpha', '?')
            beta = hyperparams.get('beta', '?')
            label = f"TIP(α={alpha}, β={beta})"
        else:
            label = method
    elif method == 'LP++':
        # Show LP++ hyperparameters
        hyperparams = baseline.get('hyperparams', {})
        if hyperparams:
            ratio = hyperparams.get('ratio', '?')
            lambda_val = hyperparams.get('lambda', '?')
            label = f"LP++(ratio={ratio}, λ={lambda_val})"
        else:
            label = method
    elif method == 'LoRA':
        # Show LoRA hyperparameters
        hyperparams = baseline.get('hyperparams', {})
        if hyperparams:
            r = hyperparams.get('r', '?')
            alpha = hyperparams.get('alpha', '?')
            lr = hyperparams.get('lr', '?')
            label = f"LoRA(r={r}, α={alpha}, lr={lr})"
        else:
            label = method
    elif method == 'ZeroShot':
        # ZeroShot has no hyperparameters
        label = 'ZeroShot'
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


def aggregate_across_datasets(all_results: Dict) -> tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, tuple]]]]:
    """
    Aggregate results across all datasets to get average accuracy.
    Also collect params and memory info (which should be same across datasets).
    
    Returns:
        (averaged_results, params_memory_info)
        
        averaged_results:
        {
            'ViT-B-32': {
                '16shots': {
                    'Energy_lr=0.001_k=4_init=average_w=0.1_wd=0.0': 85.5,
                    'Atlas_none': 82.3,
                    'Atlas_tip': 87.1,
                    'LinearProbe': 80.5,
                    'LoRA': 83.2,
                    ...
                },
                ...
            },
            ...
        }
        
        params_memory_info:
        {
            'ViT-B-32': {
                '16shots': {
                    'Energy_lr=0.001_k=4_init=average_w=0.1_wd=0.0': (trainable_params, gpu_memory),
                    ...
                },
                ...
            },
            ...
        }
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    params_memory = defaultdict(lambda: defaultdict(dict))
    
    for model_name, datasets in all_results.items():
        for dataset_name, shots in datasets.items():
            for shot_name, baselines in shots.items():
                for baseline in baselines:
                    method = baseline['method']
                    adapter = baseline.get('adapter', 'none')
                    accuracy = baseline['accuracy']
                    trainable_params = baseline.get('trainable_params')
                    gpu_memory = baseline.get('gpu_memory')
                    
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
                    elif method == 'LinearProbe':
                        # For LinearProbe, use hyperparameters as key
                        hyperparams = baseline.get('hyperparams', {})
                        if hyperparams:
                            lr = hyperparams.get('lr', 'unknown')
                            epochs = hyperparams.get('epochs', 'unknown')
                            key = f"LinearProbe_lr={lr}_ep={epochs}"
                        else:
                            key = method
                    elif method == 'LoRA':
                        # For LoRA, use hyperparameters as key
                        hyperparams = baseline.get('hyperparams', {})
                        if hyperparams:
                            r = hyperparams.get('r', 'unknown')
                            alpha = hyperparams.get('alpha', 'unknown')
                            lr = hyperparams.get('lr', 'unknown')
                            key = f"LoRA_r={r}_alpha={alpha}_lr={lr}"
                        else:
                            key = method
                    elif method == 'TIP':
                        # For TIP, use hyperparameters as key
                        hyperparams = baseline.get('hyperparams', {})
                        if hyperparams:
                            alpha = hyperparams.get('alpha', 'unknown')
                            beta = hyperparams.get('beta', 'unknown')
                            key = f"TIP_alpha={alpha}_beta={beta}"
                        else:
                            key = method
                    elif method == 'LP++':
                        # For LP++, use hyperparameters as key
                        hyperparams = baseline.get('hyperparams', {})
                        if hyperparams:
                            ratio = hyperparams.get('ratio', 'unknown')
                            lambda_val = hyperparams.get('lambda', 'unknown')
                            key = f"LP++_ratio={ratio}_lambda={lambda_val}"
                        else:
                            key = method
                    elif method == 'ZeroShot':
                        # ZeroShot has no hyperparameters
                        key = 'ZeroShot'
                    else:
                        key = f"{method}_{adapter}"
                    
                    aggregated[model_name][shot_name][key].append(accuracy)
                    
                    # Store params and memory (should be same across datasets, so we just keep the first one)
                    if key not in params_memory[model_name][shot_name]:
                        params_memory[model_name][shot_name][key] = (trainable_params, gpu_memory)
    
    # Average the accuracies
    averaged = {}
    for model_name, shots in aggregated.items():
        averaged[model_name] = {}
        for shot_name, methods in shots.items():
            averaged[model_name][shot_name] = {}
            for method_key, accuracies in methods.items():
                averaged[model_name][shot_name][method_key] = sum(accuracies) / len(accuracies)
    
    return averaged, dict(params_memory)


def select_best_method_config_per_dataset(
    all_results: Dict,
    method_name: str
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    For each dataset, model, and shot, select the best hyperparameter configuration for a given method.
    
    Args:
        all_results: Results dictionary
        method_name: Name of the method (e.g., 'Energy', 'LoRA', 'LinearProbe', 'TIP', 'LP++')
    
    Returns:
        {
            'AID': {
                'ViT-B-32': {
                    '16shots': (config_key, accuracy, config_tag),
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
                    if method_name == 'Energy':
                        lr = best_baseline.get('lr', 'unknown')
                        topk = best_baseline.get('svd_keep_topk', 'unknown')
                        init = best_baseline.get('initialize_sigma', 'unknown')
                        warmup = best_baseline.get('warmup_ratio', 'unknown')
                        wd = best_baseline.get('sigma_wd', '0.0')
                        config_key = f"Energy_lr={lr}_k={topk}_init={init}_w={warmup}_wd={wd}"
                    elif method_name == 'LinearProbe':
                        hyperparams = best_baseline.get('hyperparams', {})
                        if hyperparams:
                            lr = hyperparams.get('lr', 'unknown')
                            epochs = hyperparams.get('epochs', 'unknown')
                            config_key = f"LinearProbe_lr={lr}_ep={epochs}"
                        else:
                            config_key = method_name
                    elif method_name == 'LoRA':
                        hyperparams = best_baseline.get('hyperparams', {})
                        if hyperparams:
                            r = hyperparams.get('r', 'unknown')
                            alpha = hyperparams.get('alpha', 'unknown')
                            lr = hyperparams.get('lr', 'unknown')
                            config_key = f"LoRA_r={r}_alpha={alpha}_lr={lr}"
                        else:
                            config_key = method_name
                    elif method_name == 'TIP':
                        hyperparams = best_baseline.get('hyperparams', {})
                        if hyperparams:
                            alpha = hyperparams.get('alpha', 'unknown')
                            beta = hyperparams.get('beta', 'unknown')
                            config_key = f"TIP_alpha={alpha}_beta={beta}"
                        else:
                            config_key = method_name
                    elif method_name == 'LP++':
                        hyperparams = best_baseline.get('hyperparams', {})
                        if hyperparams:
                            ratio = hyperparams.get('ratio', 'unknown')
                            lambda_val = hyperparams.get('lambda', 'unknown')
                            config_key = f"LP++_ratio={ratio}_lambda={lambda_val}"
                        else:
                            config_key = method_name
                    else:
                        config_key = method_name
                    
                    best_config_per_dataset[dataset_name][model_name][shot_name] = (
                        config_key,
                        best_baseline['accuracy'],
                        best_baseline.get('config_tag', '')
                    )
    
    return dict(best_config_per_dataset)


def select_best_energy_config_per_dataset(
    all_results: Dict
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    For each dataset, model, and shot, select the best Energy hyperparameter configuration.
    (Wrapper for backward compatibility)
    """
    return select_best_method_config_per_dataset(all_results, 'Energy')


def compute_averaged_best_configs_from_aggregated(
    averaged_results: Dict,
    method_prefix: str
) -> Dict[str, Dict[str, tuple]]:
    """
    Select best configs from already-averaged results across all datasets.
    This ensures we select based on true average performance across ALL datasets,
    not just the best-per-dataset configs.
    
    Args:
        averaged_results: Already averaged results from aggregate_across_datasets()
        method_prefix: Method prefix to filter (e.g., 'Energy', 'LoRA', 'LinearProbe')
    
    Returns:
        {
            'ViT-B-32': {
                '16shots': (config_key, avg_accuracy),
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


def compute_averaged_best_configs(
    best_configs_per_dataset: Dict,
    method_prefix: str = None
) -> Dict[str, Dict[str, tuple]]:
    """
    DEPRECATED: Use compute_averaged_best_configs_from_aggregated instead.
    
    This function only averages the best-per-dataset configs, which may not
    represent the true best average config across all datasets.
    
    Compute averaged best configs across datasets for each model/shot.
    Used for the aggregated table visualization.
    
    Args:
        best_configs_per_dataset: Best configs per dataset
        method_prefix: Optional prefix to filter config keys (e.g., 'Energy', 'LoRA')
    
    Returns:
        {
            'ViT-B-32': {
                '16shots': (config_key, avg_accuracy),
                ...
            },
            ...
        }
    """
    # Group by model and shot, average accuracy across datasets
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for dataset, models in best_configs_per_dataset.items():
        for model, shots in models.items():
            for shot, (config_key, accuracy, config_tag) in shots.items():
                # Filter by method prefix if specified
                if method_prefix is None or config_key.startswith(method_prefix):
                    grouped[model][shot][config_key].append(accuracy)
    
    # Select best average config for each model/shot
    best_avg = {}
    for model, shots in grouped.items():
        best_avg[model] = {}
        for shot, configs in shots.items():
            # Average accuracy for each config
            config_averages = {
                config_key: sum(accuracies) / len(accuracies)
                for config_key, accuracies in configs.items()
            }
            # Select best
            if config_averages:
                best_key = max(config_averages, key=config_averages.get)
                best_acc = config_averages[best_key]
                best_avg[model][shot] = (best_key, best_acc)
    
    return best_avg


def compute_averaged_best_energy_configs(
    best_energy_per_dataset: Dict
) -> Dict[str, Dict[str, tuple]]:
    """
    Compute averaged best Energy configs across datasets for each model/shot.
    (Wrapper for backward compatibility)
    """
    return compute_averaged_best_configs(best_energy_per_dataset, 'Energy')


def visualize_aggregated_table(
    averaged_results: Dict,
    best_energy_configs: Dict,
    best_baseline_configs: Dict,
    params_memory_info: Dict,
    output_dir: str
) -> None:
    """
    Create a table visualization similar to Table 2 in the paper.
    Shows averaged accuracy across all datasets.
    
    Args:
        averaged_results: Averaged accuracy for all methods/configs
        best_energy_configs: Best Energy configs per model/shot
        best_baseline_configs: Dict of best configs for baseline methods (LoRA, LinearProbe, TIP, LP++)
        params_memory_info: Params and memory info for each method/config
        output_dir: Output directory
    """
    # Define shot order
    shot_order = ['1shots', '2shots', '4shots', '8shots', '16shots']
    
    # Collect all unique methods
    all_methods = set()
    for model_name, shots in averaged_results.items():
        for shot_name, methods in shots.items():
            all_methods.update(methods.keys())
    
    # Separate Energy, Atlas, baseline, and other methods
    energy_methods = {m for m in all_methods if m.startswith('Energy_')}
    atlas_methods = {m for m in all_methods if m.startswith('Atlas_')}
    lora_methods = {m for m in all_methods if m.startswith('LoRA')}
    linearprobe_methods = {m for m in all_methods if m.startswith('LinearProbe')}
    tip_methods = {m for m in all_methods if m.startswith('TIP')}
    lpp_methods = {m for m in all_methods if m.startswith('LP++')}
    zeroshot_methods = {m for m in all_methods if m == 'ZeroShot'}
    other_methods = all_methods - energy_methods - atlas_methods - lora_methods - linearprobe_methods - tip_methods - lpp_methods - zeroshot_methods
    
    # Prepare method list (best configs for each method type)
    method_list = (
        ['Energy (best config)'] +
        ['LoRA (best config)'] +
        ['LinearProbe (best config)'] +
        ['TIP (best config)'] +
        ['LP++ (best config)'] +
        sorted(zeroshot_methods) +
        sorted(atlas_methods) +
        sorted(other_methods)
    )
    
    # Get models
    models = sorted(averaged_results.keys())
    
    # Create data structure for table
    # Structure: method -> model -> shot -> accuracy
    table_data = defaultdict(lambda: defaultdict(dict))
    best_config_info = defaultdict(lambda: defaultdict(dict))  # Store all best configs
    
    for model_name in models:
        for shot_name in shot_order:
            if shot_name not in averaged_results[model_name]:
                continue
            
            methods = averaged_results[model_name][shot_name]
            
            # Handle Energy - use best config
            if model_name in best_energy_configs and shot_name in best_energy_configs[model_name]:
                best_config_key, best_acc = best_energy_configs[model_name][shot_name]
                table_data['Energy (best config)'][model_name][shot_name] = best_acc
                best_config_info['Energy'][model_name][shot_name] = best_config_key
            
            # Handle baseline methods - use best configs
            for method_name in ['LoRA', 'LinearProbe', 'TIP', 'LP++']:
                if method_name in best_baseline_configs:
                    method_configs = best_baseline_configs[method_name]
                    if model_name in method_configs and shot_name in method_configs[model_name]:
                        best_config_key, best_acc = method_configs[model_name][shot_name]
                        table_data[f'{method_name} (best config)'][model_name][shot_name] = best_acc
                        best_config_info[method_name][model_name][shot_name] = best_config_key
            
            # Handle Atlas, ZeroShot, and others
            for method_key, accuracy in methods.items():
                if method_key.startswith('Atlas_'):
                    table_data[method_key][model_name][shot_name] = accuracy
                elif method_key == 'ZeroShot':
                    table_data[method_key][model_name][shot_name] = accuracy
                elif not any(method_key.startswith(prefix) for prefix in ['Energy_', 'LoRA_', 'LinearProbe_', 'TIP_', 'LP++']):
                    table_data[method_key][model_name][shot_name] = accuracy
    
    # Create table visualization
    fig = plt.figure(figsize=(18, max(6, len(method_list) * 0.5 + 2)))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare column headers
    col_labels = ['Method']
    for model in models:
        for shot in shot_order:
            shot_num = shot.replace('shots', '')
            col_labels.append(f"{model}\n{shot_num}-shot")
    
    # Prepare table data
    table_content = []
    for method in method_list:
        row = [format_method_label(method)]
        for model in models:
            for shot in shot_order:
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
    table.set_fontsize(9)
    table.scale(1, 2)
    
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
    
    plt.title('Few-Shot Remote Sensing Results - Dataset-Averaged Accuracy (%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'aggregated_results_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved aggregated table: {output_path}")
    
    # Save as CSV
    csv_output_path = os.path.join(output_dir, 'aggregated_results_table.csv')
    with open(csv_output_path, 'w') as f:
        f.write(','.join(col_labels) + '\n')
        for row in table_content:
            f.write(','.join(row) + '\n')
    
    logger.info(f"Saved CSV table: {csv_output_path}")
    
    # Save best configurations to text file
    config_output_path = os.path.join(output_dir, 'best_method_configs.txt')
    with open(config_output_path, 'w') as f:
        f.write("Best Hyperparameter Configurations for All Methods\n")
        f.write("=" * 80 + "\n\n")
        
        for method_name in ['Energy', 'LoRA', 'LinearProbe', 'TIP', 'LP++']:
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

    # ------------------------------------------------------------------
    # Additionally, create a k-shot averaged table across models
    # Columns: Method, 1-shot, 2-shot, 4-shot, 8-shot, 16-shot, Average, Params, GPU Memory
    # ------------------------------------------------------------------
    shot_avg_col_labels = ['Method'] + [s.replace('shots', '-shot') for s in shot_order] + ['Average', 'Params (M)', 'GPU Mem (MB)']
    shot_avg_table_content = []
    
    for method in method_list:
        row = [format_method_label(method)]
        all_shot_vals = []  # For computing overall average
        
        for shot in shot_order:
            # Gather accuracies for this method/shot across models
            shot_vals = []
            for model in models:
                acc = table_data[method].get(model, {}).get(shot, None)
                if acc is not None:
                    shot_vals.append(acc)
            if shot_vals:
                avg_acc = sum(shot_vals) / len(shot_vals)
                row.append(f"{avg_acc:.2f}")
                all_shot_vals.extend(shot_vals)
            else:
                row.append("-")
        
        # Add overall average across all shots and models
        if all_shot_vals:
            row.append(f"{(sum(all_shot_vals) / len(all_shot_vals)):.2f}")
        else:
            row.append("-")
        
        # Add params and memory info
        # Get from the best config for this method (use first model and shot where available)
        params_found = None
        memory_found = None
        for model in models:
            for shot in shot_order:
                # Get the config key for this method
                config_key = None
                if method == 'Energy (best config)' and model in best_energy_configs and shot in best_energy_configs[model]:
                    config_key, _ = best_energy_configs[model][shot]
                elif method.endswith(' (best config)'):
                    method_name = method.replace(' (best config)', '')
                    if method_name in best_baseline_configs:
                        method_configs = best_baseline_configs[method_name]
                        if model in method_configs and shot in method_configs[model]:
                            config_key, _ = method_configs[model][shot]
                else:
                    config_key = method
                
                # Get params and memory from params_memory_info
                if config_key and model in params_memory_info and shot in params_memory_info[model]:
                    if config_key in params_memory_info[model][shot]:
                        params, memory = params_memory_info[model][shot][config_key]
                        if params is not None:
                            params_found = params
                        if memory is not None:
                            memory_found = memory
                        break
            if params_found is not None or memory_found is not None:
                break
        
        # Format params in millions
        if params_found is not None:
            row.append(f"{params_found / 1e6:.2f}")
        else:
            row.append("-")
        
        # Format memory
        if memory_found is not None:
            row.append(f"{memory_found:.1f}")
        else:
            row.append("-")
        
        shot_avg_table_content.append(row)
    
    # Create figure for shot-averaged table (wider to accommodate new columns)
    fig2 = plt.figure(figsize=(16, max(6, len(method_list) * 0.5 + 2)))
    ax2 = fig2.add_subplot(111)
    ax2.axis('tight')
    ax2.axis('off')
    
    table2 = ax2.table(
        cellText=shot_avg_table_content,
        colLabels=shot_avg_col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 2)
    
    # Header styling
    for i in range(len(shot_avg_col_labels)):
        cell = table2[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Method column styling
    for i in range(len(shot_avg_table_content)):
        cell = table2[(i+1, 0)]
        cell.set_facecolor('#E8F5E9')
        cell.set_text_props(weight='bold', ha='left')
    
    plt.title('Few-Shot Remote Sensing Results - Model-Averaged per k-shot (%)',
              fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    shot_avg_png = os.path.join(output_dir, 'aggregated_results_by_shot_table.png')
    plt.savefig(shot_avg_png, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved shot-averaged table: {shot_avg_png}")
    
    # Save CSV for shot-averaged table
    shot_avg_csv = os.path.join(output_dir, 'aggregated_results_by_shot_table.csv')
    with open(shot_avg_csv, 'w') as f:
        f.write(','.join(shot_avg_col_labels) + '\n')
        for row in shot_avg_table_content:
            f.write(','.join(row) + '\n')
    logger.info(f"Saved shot-averaged CSV: {shot_avg_csv}")


def create_comprehensive_table(
    all_results: Dict,
    best_energy_per_dataset: Dict,
    best_baseline_per_dataset: Dict,
    params_memory_info: Dict,
    output_dir: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Create a single comprehensive table with datasets as columns.
    Rows are organized by (Model, Shot, Method).
    Last column shows the average across datasets, followed by params and memory.
    Uses dataset-specific best configurations for all methods.
    
    Args:
        all_results: All results
        best_energy_per_dataset: Best Energy configs per dataset
        best_baseline_per_dataset: Dict of best configs per dataset for baseline methods
        params_memory_info: Params and memory info for each method/config
        output_dir: Output directory
        
    Returns:
        averaged_per_shot: {model -> shot -> method -> average_accuracy}
    """
    # Define shot order
    shot_order = ['1shots', '2shots', '4shots', '8shots', '16shots']
    
    # Collect all datasets
    all_datasets = set()
    for model_data in all_results.values():
        all_datasets.update(model_data.keys())
    all_datasets = sorted(all_datasets)
    
    # Collect all models
    models = sorted(all_results.keys())
    
    # Collect all unique methods across all datasets
    all_methods = set()
    has_energy = False
    has_lora = False
    has_linearprobe = False
    has_tip = False
    has_lpp = False
    has_zeroshot = False
    atlas_methods = set()
    other_methods = set()
    
    for model_name in models:
        for dataset_name in all_results[model_name]:
            for shot_name in all_results[model_name][dataset_name]:
                for baseline in all_results[model_name][dataset_name][shot_name]:
                    method = baseline['method']
                    adapter = baseline.get('adapter', 'none')
                    if method == 'Energy':
                        has_energy = True
                    elif method == 'Atlas':
                        atlas_methods.add(f'Atlas_{adapter}')
                    elif method == 'LoRA':
                        has_lora = True
                    elif method == 'LinearProbe':
                        has_linearprobe = True
                    elif method == 'TIP':
                        has_tip = True
                    elif method == 'LP++':
                        has_lpp = True
                    elif method == 'ZeroShot':
                        has_zeroshot = True
                    else:
                        other_methods.add(f'{method}_{adapter}')
    
    # Prepare method list with best configs
    method_list = []
    if has_energy:
        method_list.append('Energy (best config)')
    if has_lora:
        method_list.append('LoRA (best config)')
    if has_linearprobe:
        method_list.append('LinearProbe (best config)')
    if has_tip:
        method_list.append('TIP (best config)')
    if has_lpp:
        method_list.append('LP++ (best config)')
    if has_zeroshot:
        method_list.append('ZeroShot')
    method_list.extend(sorted(atlas_methods))
    method_list.extend(sorted(other_methods))
    
    # Build data structure: [model][shot][method][dataset] = accuracy
    data_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name in models:
        for dataset_name in all_results[model_name]:
            for shot_name in shot_order:
                if shot_name not in all_results[model_name][dataset_name]:
                    continue
                
                baselines = all_results[model_name][dataset_name][shot_name]
                
                # Group baselines by method
                energy_baselines = []
                atlas_baselines = defaultdict(list)
                baseline_method_results = defaultdict(list)
                other_baselines = defaultdict(list)
                
                for baseline in baselines:
                    method = baseline['method']
                    adapter = baseline.get('adapter', 'none')
                    accuracy = baseline['accuracy']
                    
                    if method == 'Energy':
                        energy_baselines.append((baseline, accuracy))
                    elif method == 'Atlas':
                        atlas_baselines[f'Atlas_{adapter}'].append(accuracy)
                    elif method in ['LinearProbe', 'TIP', 'LP++', 'LoRA']:
                        baseline_method_results[method].append(accuracy)
                    elif method == 'ZeroShot':
                        baseline_method_results['ZeroShot'].append(accuracy)
                    else:
                        other_baselines[f'{method}_{adapter}'].append(accuracy)
                
                # For Energy, use dataset-specific best config from best_energy_per_dataset
                if dataset_name in best_energy_per_dataset:
                    if model_name in best_energy_per_dataset[dataset_name]:
                        if shot_name in best_energy_per_dataset[dataset_name][model_name]:
                            _, best_acc, _ = best_energy_per_dataset[dataset_name][model_name][shot_name]
                            data_structure[model_name][shot_name]['Energy (best config)'][dataset_name] = best_acc
                
                # For baseline methods, use dataset-specific best config
                for method_name in ['LoRA', 'LinearProbe', 'TIP', 'LP++']:
                    if method_name in best_baseline_per_dataset:
                        if dataset_name in best_baseline_per_dataset[method_name]:
                            if model_name in best_baseline_per_dataset[method_name][dataset_name]:
                                if shot_name in best_baseline_per_dataset[method_name][dataset_name][model_name]:
                                    _, best_acc, _ = best_baseline_per_dataset[method_name][dataset_name][model_name][shot_name]
                                    data_structure[model_name][shot_name][f'{method_name} (best config)'][dataset_name] = best_acc
                
                # For ZeroShot (no hyperparameters, just take first result)
                if 'ZeroShot' in baseline_method_results and baseline_method_results['ZeroShot']:
                    data_structure[model_name][shot_name]['ZeroShot'][dataset_name] = baseline_method_results['ZeroShot'][0]
                
                # For Atlas
                for atlas_key, accuracies in atlas_baselines.items():
                    if accuracies:
                        data_structure[model_name][shot_name][atlas_key][dataset_name] = accuracies[0]
                
                # For other methods
                for other_key, accuracies in other_baselines.items():
                    if accuracies:
                        data_structure[model_name][shot_name][other_key][dataset_name] = accuracies[0]
    
    # Create table
    # Columns: Model | Shot | Method | Dataset1 | Dataset2 | ... | Average | Params (M) | GPU Mem (MB)
    col_labels = ['Model', 'Shot', 'Method'] + all_datasets + ['Average', 'Params (M)', 'GPU Mem (MB)']
    
    # Prepare table content with row metadata
    table_content = []
    row_metadata = []  # Store (model, shot, method) for each row
    
    # Store averaged results to return: {model -> shot -> method -> average}
    averaged_per_shot = defaultdict(lambda: defaultdict(dict))
    
    for model_name in models:
        for shot_idx, shot_name in enumerate(shot_order):
            shot_display = shot_name.replace('shots', '')
            
            for method_idx, method_key in enumerate(method_list):
                # First method in shot shows model name (for first shot only) and shot number
                if method_idx == 0:
                    if shot_idx == 0:
                        row = [model_name, shot_display, format_method_label(method_key)]
                    else:
                        row = ['', shot_display, format_method_label(method_key)]
                else:
                    row = ['', '', format_method_label(method_key)]
                
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
                    # Store averaged result
                    averaged_per_shot[model_name][shot_name][method_key] = avg
                else:
                    row.append("-")
                
                # Add params and memory info
                # Get the config key for this method
                config_key = None
                if method_key == 'Energy (best config)':
                    # Use the dataset-specific best config (just take first dataset for params/memory)
                    for dataset_name in all_datasets:
                        if dataset_name in best_energy_per_dataset:
                            if model_name in best_energy_per_dataset[dataset_name]:
                                if shot_name in best_energy_per_dataset[dataset_name][model_name]:
                                    config_key, _, _ = best_energy_per_dataset[dataset_name][model_name][shot_name]
                                    break
                        if config_key:
                            break
                elif method_key.endswith(' (best config)'):
                    method_name = method_key.replace(' (best config)', '')
                    if method_name in best_baseline_per_dataset:
                        for dataset_name in all_datasets:
                            if dataset_name in best_baseline_per_dataset[method_name]:
                                if model_name in best_baseline_per_dataset[method_name][dataset_name]:
                                    if shot_name in best_baseline_per_dataset[method_name][dataset_name][model_name]:
                                        config_key, _, _ = best_baseline_per_dataset[method_name][dataset_name][model_name][shot_name]
                                        break
                            if config_key:
                                break
                else:
                    config_key = method_key
                
                # Get params and memory from params_memory_info
                params_found = None
                memory_found = None
                if config_key and model_name in params_memory_info and shot_name in params_memory_info[model_name]:
                    if config_key in params_memory_info[model_name][shot_name]:
                        params, memory = params_memory_info[model_name][shot_name][config_key]
                        params_found = params
                        memory_found = memory
                
                # Format params in millions
                if params_found is not None:
                    row.append(f"{params_found / 1e6:.2f}")
                else:
                    row.append("-")
                
                # Format memory
                if memory_found is not None:
                    row.append(f"{memory_found:.1f}")
                else:
                    row.append("-")
                
                table_content.append(row)
                row_metadata.append((model_name, shot_name, method_key))
            
            # Add separator row between shots (not after last shot of a model)
            if shot_idx < len(shot_order) - 1:
                table_content.append([''] * len(col_labels))
                row_metadata.append((None, None, None))
        
        # Add separator row between models (visual separation)
        if model_name != models[-1]:
            table_content.append([''] * len(col_labels))
            row_metadata.append((None, None, None))
    
    # Create figure
    n_rows = len(table_content)
    n_cols = len(col_labels)
    fig_height = max(12, n_rows * 0.25 + 2)
    fig_width = max(20, n_cols * 1.2)
    
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
    
    # Color first three columns (Model, Shot, Method)
    for i in range(len(table_content)):
        for j in range(3):
            cell = table[(i+1, j)]
            if table_content[i][j]:  # Not empty separator row
                cell.set_facecolor('#E8F5E9')
                if j == 0:  # Model column
                    cell.set_text_props(weight='bold', ha='center', fontsize=9)
                elif j == 1:  # Shot column
                    cell.set_text_props(weight='bold', ha='center')
                else:  # Method column
                    cell.set_text_props(weight='bold', ha='left')
    
    # Highlight best values per model/shot combination with bold text (no color)
    # Group by model and shot
    current_group_start = 0
    for i in range(len(table_content) + 1):
        # Check if we reached end of a shot group (empty row or end of list)
        if i == len(table_content) or row_metadata[i][0] is None:
            # Process the current group
            if current_group_start < i:
                # For each dataset column (and average), find best value in this group
                for col_idx in range(3, len(col_labels)):
                    values = []
                    for row_idx in range(current_group_start, i):
                        cell_text = table_content[row_idx][col_idx]
                        if cell_text != "-" and cell_text != "":
                            try:
                                values.append((float(cell_text), row_idx))
                            except:
                                pass
                    
                    if values:
                        max_val, max_row_idx = max(values, key=lambda x: x[0])
                        cell = table[(max_row_idx + 1, col_idx)]
                        cell.set_text_props(weight='bold')
            
            # Move to next group
            current_group_start = i + 1
    
    plt.title('Few-Shot Remote Sensing Results - All Datasets and Average', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comprehensive_results_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved comprehensive table: {output_path}")
    
    # Save as CSV
    csv_output_path = os.path.join(output_dir, 'comprehensive_results_table.csv')
    with open(csv_output_path, 'w') as f:
        f.write(','.join(col_labels) + '\n')
        for row in table_content:
            f.write(','.join(row) + '\n')
    
    logger.info(f"Saved CSV: {csv_output_path}")
    
    # Return averaged results for use in aggregated_results_by_shot_table
    return dict(averaged_per_shot)


def create_aggregated_by_shot_table_from_comprehensive(
    comprehensive_averaged_results: Dict[str, Dict[str, Dict[str, float]]],
    params_memory_info: Dict,
    best_energy_configs: Dict,
    best_baseline_configs: Dict,
    output_dir: str
) -> None:
    """
    Create aggregated by shot table using results from comprehensive table.
    This ensures consistency - uses the same dataset-specific best configs.
    
    Args:
        comprehensive_averaged_results: {model -> shot -> method -> average_accuracy}
            (returned from create_comprehensive_table)
        params_memory_info: Params and memory info for each method/config
        best_energy_configs: Best Energy configs per dataset
        best_baseline_configs: Best baseline configs per dataset
        output_dir: Output directory
    """
    shot_order = ['1shots', '2shots', '4shots', '8shots', '16shots']
    
    # Collect all unique methods
    all_methods = set()
    for model_data in comprehensive_averaged_results.values():
        for shot_data in model_data.values():
            all_methods.update(shot_data.keys())
    
    # Sort methods (same order as comprehensive table)
    method_list = []
    if 'Energy (best config)' in all_methods:
        method_list.append('Energy (best config)')
    if 'LoRA (best config)' in all_methods:
        method_list.append('LoRA (best config)')
    if 'LinearProbe (best config)' in all_methods:
        method_list.append('LinearProbe (best config)')
    if 'TIP (best config)' in all_methods:
        method_list.append('TIP (best config)')
    if 'LP++ (best config)' in all_methods:
        method_list.append('LP++ (best config)')
    
    # Add Atlas and other methods
    atlas_methods = sorted([m for m in all_methods if m.startswith('Atlas_')])
    other_methods = sorted([m for m in all_methods if m not in method_list and not m.startswith('Atlas_')])
    method_list.extend(atlas_methods)
    method_list.extend(other_methods)
    
    # Create table: columns are shots, rows are methods
    # For each method/shot, average across all models
    shot_avg_col_labels = ['Method'] + [s.replace('shots', '-shot') for s in shot_order]
    shot_avg_table_content = []
    
    models = list(comprehensive_averaged_results.keys())
    
    for method in method_list:
        row = [format_method_label(method)]
        for shot in shot_order:
            # Gather accuracies for this method/shot across models
            shot_vals = []
            for model in models:
                if shot in comprehensive_averaged_results[model]:
                    acc = comprehensive_averaged_results[model][shot].get(method, None)
                    if acc is not None:
                        shot_vals.append(acc)
            if shot_vals:
                row.append(f"{(sum(shot_vals) / len(shot_vals)):.2f}")
            else:
                row.append("-")
        shot_avg_table_content.append(row)
    
    # Create figure for shot-averaged table
    fig = plt.figure(figsize=(12, max(6, len(method_list) * 0.5 + 2)))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=shot_avg_table_content,
        colLabels=shot_avg_col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(shot_avg_col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Method column styling
    for i in range(len(shot_avg_table_content)):
        cell = table[(i+1, 0)]
        cell.set_facecolor('#E8F5E9')
        cell.set_text_props(weight='bold', ha='left')
    
    plt.title('Few-Shot Remote Sensing Results - Model-Averaged per k-shot (%)',
              fontsize=14, fontweight='bold', pad=20)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    shot_avg_png = os.path.join(output_dir, 'aggregated_results_by_shot_table.png')
    plt.savefig(shot_avg_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved shot-averaged table: {shot_avg_png}")
    
    # Save CSV for shot-averaged table
    shot_avg_csv = os.path.join(output_dir, 'aggregated_results_by_shot_table.csv')
    with open(shot_avg_csv, 'w') as f:
        f.write(','.join(shot_avg_col_labels) + '\n')
        for row in shot_avg_table_content:
            f.write(','.join(row) + '\n')
    logger.info(f"Saved shot-averaged CSV: {shot_avg_csv}")


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
    elif method_key in ['LinearProbe', 'TIP', 'LP++', 'LoRA', 'ZeroShot']:
        # Baseline methods - return as-is
        return method_key
    elif method_key.endswith('(best config)'):
        # Handle "LoRA (best config)" -> "LoRA (best)"
        base_name = method_key.replace(' (best config)', '')
        return f'{base_name} (best)'
    else:
        return method_key


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and visualize remote sensing experiment results'
    )
    parser.add_argument(
        '--model_location',
        type=str,
        default='./models/checkpoints',
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
    
    # Select best configurations per dataset for all methods
    logger.info("\nSelecting best configurations per dataset for all methods...")
    best_energy_per_dataset = select_best_energy_config_per_dataset(all_results)
    
    # Select best configs for baseline methods
    best_baseline_per_dataset = {}
    for method_name in ['LoRA', 'LinearProbe', 'TIP', 'LP++']:
        logger.info(f"Selecting best {method_name} configurations per dataset...")
        best_baseline_per_dataset[method_name] = select_best_method_config_per_dataset(all_results, method_name)
    
    # Save best configurations to JSON for future use
    best_config_path = os.path.join(args.output_dir, 'best_configs_per_dataset.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(best_config_path, 'w') as f:
        # Convert to serializable format
        serializable_config = {'Energy': {}}
        
        # Add Energy configs
        for dataset, models in best_energy_per_dataset.items():
            serializable_config['Energy'][dataset] = {}
            for model, shots in models.items():
                serializable_config['Energy'][dataset][model] = {}
                for shot, (config_key, accuracy, config_tag) in shots.items():
                    serializable_config['Energy'][dataset][model][shot] = {
                        'config_key': config_key,
                        'accuracy': accuracy,
                        'config_tag': config_tag
                    }
        
        # Add baseline method configs
        for method_name, method_data in best_baseline_per_dataset.items():
            serializable_config[method_name] = {}
            for dataset, models in method_data.items():
                serializable_config[method_name][dataset] = {}
                for model, shots in models.items():
                    serializable_config[method_name][dataset][model] = {}
                    for shot, (config_key, accuracy, config_tag) in shots.items():
                        serializable_config[method_name][dataset][model][shot] = {
                            'config_key': config_key,
                            'accuracy': accuracy,
                            'config_tag': config_tag
                        }
        
        json.dump(serializable_config, f, indent=2)
    logger.info(f"Saved best configs per dataset to: {best_config_path}")
    
    # Aggregate results across datasets to get params_memory_info
    logger.info("\nAggregating results across datasets...")
    averaged_results, params_memory_info = aggregate_across_datasets(all_results)
    
    # Generate comprehensive table with all datasets as columns
    logger.info("\nGenerating comprehensive table (all datasets + average)...")
    comprehensive_averaged_results = create_comprehensive_table(
        all_results, best_energy_per_dataset, best_baseline_per_dataset, params_memory_info, args.output_dir
    )
    
    # Generate aggregated by shot table using results from comprehensive table
    # This ensures consistency - both tables use the same dataset-specific best configs
    logger.info("\nGenerating aggregated by shot table (using comprehensive table results)...")
    create_aggregated_by_shot_table_from_comprehensive(comprehensive_averaged_results, args.output_dir)
    
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
