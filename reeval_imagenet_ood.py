#!/usr/bin/env python3
"""
Re-evaluate OOD results for already trained ImageNet models.

This script loads existing trained models (Energy, Atlas, Baselines) and re-evaluates
them on OOD datasets (ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2)
with corrected evaluation using fine-tuned classification heads.

Supports parallel execution across multiple GPUs for faster evaluation.

Supported Methods:
    - energy: Energy-based method
    - atlas: Atlas method
    - baseline: All baseline methods (Linear Probe, TIP, LP++, LoRA)
    - linear_probe: Linear probe baseline
    - tip_adapter: TIP adapter baseline
    - lp++: LP++ baseline
    - lora: LoRA baseline

Usage:
    python reeval_imagenet_ood.py
        Re-evaluate all existing models (all methods) found in model_location.

    python reeval_imagenet_ood.py --models ViT-B-16 ViT-L-14
        Re-evaluate only specific models.

    python reeval_imagenet_ood.py --methods energy atlas
        Re-evaluate only specific methods.

    python reeval_imagenet_ood.py --methods baseline
        Re-evaluate all baseline methods (Linear Probe, TIP, LP++, LoRA).

    python reeval_imagenet_ood.py --methods lora tip_adapter
        Re-evaluate only LoRA and TIP adapter baselines.

    python reeval_imagenet_ood.py --shots 4 16
        Re-evaluate only specific shot configurations.

    python reeval_imagenet_ood.py --num_gpus 4
        Use 4 GPUs for parallel execution (GPUs 0-3).

    python reeval_imagenet_ood.py --gpu_ids 0 2 4 6
        Use specific GPU IDs (0, 2, 4, 6) for parallel execution.
"""

import argparse
import os
import sys
import subprocess
from typing import List
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def find_existing_models(model_location: str, models: List[str] = None) -> dict:
    """
    Find all existing trained models in the model_location directory.
    
    Returns:
        {
            'energy': [(model, config_tag, k), ...],
            'atlas': [(model, config_tag, k), ...],
            'baseline': [(model, config_tag, k, baseline_type), ...],
        }
    """
    if not os.path.exists(model_location):
        print(f"Error: Model location not found: {model_location}")
        return {}
    
    found = {'energy': [], 'atlas': [], 'baseline': []}
    
    for model_name in os.listdir(model_location):
        if models and model_name not in models:
            continue
        
        model_path = os.path.join(model_location, model_name)
        if not os.path.isdir(model_path):
            continue
        
        # Look for ImageNet results
        for dataset_entry in os.listdir(model_path):
            if 'imagenet' not in dataset_entry.lower():
                continue
            
            dataset_path = os.path.join(model_path, dataset_entry)
            if not os.path.isdir(dataset_path):
                continue
            
            # Scan config directories
            for config_entry in os.listdir(dataset_path):
                config_path = os.path.join(dataset_path, config_entry)
                if not os.path.isdir(config_path):
                    continue
                
                # Scan shot directories
                for shot_entry in os.listdir(config_path):
                    shot_path = os.path.join(config_path, shot_entry)
                    if not os.path.isdir(shot_path):
                        continue
                    
                    # Extract k value
                    if shot_entry.endswith('shots'):
                        k = int(shot_entry.replace('shots', ''))
                    elif shot_entry.endswith('shot'):
                        k = int(shot_entry.replace('shot', ''))
                    else:
                        continue
                    
                    # Check for energy model
                    energy_path = os.path.join(shot_path, 'energy.pt')
                    if os.path.exists(energy_path):
                        found['energy'].append((model_name, config_entry, k))
                    
                    # Check for atlas model
                    atlas_path = os.path.join(shot_path, 'atlas.pt')
                    if os.path.exists(atlas_path):
                        found['atlas'].append((model_name, config_entry, k))
                    
                    # Check for baseline model (by results file)
                    baseline_results_path = os.path.join(shot_path, 'baseline_results_imagenet.json')
                    if os.path.exists(baseline_results_path) and config_entry.startswith('imagenet_baseline_'):
                        # Determine baseline type from config
                        if '_lp_' in config_entry and '_lpp_' not in config_entry:
                            baseline_type = 'linear_probe'
                        elif '_tip_' in config_entry:
                            baseline_type = 'tip_adapter'
                        elif '_lpp_' in config_entry:
                            baseline_type = 'lp++'
                        elif '_lora_' in config_entry:
                            baseline_type = 'lora'
                        else:
                            baseline_type = 'unknown'
                        
                        if baseline_type != 'unknown':
                            found['baseline'].append((model_name, config_entry, k, baseline_type))
    
    return found


def parse_energy_config(config_tag: str) -> dict:
    """Parse energy config tag to extract hyperparameters."""
    # imagenet_energy_{sigma_lr}_{topk}_{init_mode}_{warmup_ratio}_{sigma_wd}_{k}shot
    if not config_tag.startswith('imagenet_energy_'):
        return {}
    
    parts = config_tag.split('_')
    if len(parts) < 8:
        return {}
    
    return {
        'sigma_lr': parts[2].replace('p', '.'),
        'topk': parts[3],
        'init_mode': parts[4],
        'warmup_ratio': parts[5].replace('p', '.'),
        'sigma_wd': parts[6].replace('p', '.'),
    }


def parse_atlas_config(config_tag: str) -> dict:
    """Parse atlas config tag to extract hyperparameters."""
    # imagenet_atlas_{num_basis}_{lr}_{k}_shot
    if not config_tag.startswith('imagenet_atlas_'):
        return {}
    
    parts = config_tag.split('_')
    if len(parts) < 6:
        return {}
    
    return {
        'num_basis': parts[2],
        'lr': parts[3].replace('p', '.'),
    }


def parse_baseline_config(config_tag: str, baseline_type: str) -> dict:
    """Parse baseline config tag to extract hyperparameters."""
    if not config_tag.startswith('imagenet_baseline_'):
        return {}
    
    parts = config_tag.split('_')
    
    if baseline_type == 'linear_probe':
        # imagenet_baseline_lp_{lr}_{epochs}_{wd}_{k}shot
        if len(parts) < 7:
            return {}
        return {
            'method': 'linear_probe',
            'lr': parts[3].replace('p', '.'),
            'epochs': parts[4],
            'wd': parts[5].replace('p', '.'),
        }
    
    elif baseline_type == 'tip_adapter':
        # imagenet_baseline_tip_{wd}_{k}shot
        if len(parts) < 5:
            return {}
        return {
            'method': 'tip_adapter',
            'wd': parts[3].replace('p', '.'),
        }
    
    elif baseline_type == 'lp++':
        # imagenet_baseline_lpp_{wd}_{k}shot
        if len(parts) < 5:
            return {}
        return {
            'method': 'lp++',
            'wd': parts[3].replace('p', '.'),
        }
    
    elif baseline_type == 'lora':
        # imagenet_baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}_{k}shot
        if len(parts) < 9:
            return {}
        return {
            'method': 'lora',
            'r': parts[3],
            'alpha': parts[4],
            'lr': parts[5].replace('p', '.'),
            'epochs': parts[6],
            'wd': parts[7].replace('p', '.'),
        }
    
    return {}


def run_command_with_gpu(args_tuple):
    """Execute a command with a specific GPU assignment."""
    method, model, k, cmd, gpu_id, job_id, total_jobs = args_tuple
    
    # Set GPU environment
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[{job_id}/{total_jobs}] Starting {method} {model} {k}-shot on GPU {gpu_id}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"[{job_id}/{total_jobs}] ✓ {method} {model} {k}-shot completed on GPU {gpu_id} ({elapsed:.1f}s)")
        return True, method, model, k, gpu_id, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[{job_id}/{total_jobs}] ✗ {method} {model} {k}-shot failed on GPU {gpu_id} ({elapsed:.1f}s)")
        print(f"  Error: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        return False, method, model, k, gpu_id, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{job_id}/{total_jobs}] ✗ {method} {model} {k}-shot error on GPU {gpu_id}: {str(e)}")
        return False, method, model, k, gpu_id, elapsed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model_location',
        type=str,
        default='./models/checkpoints',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='Specific models to re-evaluate (e.g., ViT-B-16 ViT-L-14)'
    )
    parser.add_argument(
        '--methods',
        nargs='*',
        choices=['energy', 'atlas', 'baseline', 'linear_probe', 'tip_adapter', 'lp++', 'lora', 'all'],
        default=['all'],
        help='Methods to re-evaluate (baseline includes all baseline methods)'
    )
    parser.add_argument(
        '--shots',
        nargs='*',
        type=int,
        default=[4,16],
        help='Specific shot configurations to re-evaluate (e.g., 4 16)'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=8,
        help='Number of GPUs to use for parallel execution (default: 1)'
    )
    parser.add_argument(
        '--gpu_ids',
        nargs='*',
        type=int,
        default=None,
        help='Specific GPU IDs to use (e.g., 0 1 2 3). If not specified, uses 0 to num_gpus-1'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    
    args = parser.parse_args()
    
    # Expand 'all' and 'baseline' methods
    methods = args.methods
    if 'all' in methods:
        methods = ['energy', 'atlas', 'baseline']
    if 'baseline' in methods:
        # Replace 'baseline' with all specific baseline methods
        methods = [m for m in methods if m != 'baseline']
        methods.extend(['linear_probe', 'tip_adapter', 'lp++', 'lora'])
    
    # Remove duplicates while preserving order
    seen = set()
    methods = [m for m in methods if not (m in seen or seen.add(m))]
    
    # Setup GPU IDs
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
        num_gpus = len(gpu_ids)
    else:
        num_gpus = args.num_gpus
        gpu_ids = list(range(num_gpus))
    
    print("=" * 80)
    print("Re-evaluating ImageNet OOD results for trained models")
    print("=" * 80)
    print(f"Model location: {args.model_location}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Parallel execution: {num_gpus} GPU(s) - {gpu_ids}")
    if args.models:
        print(f"Models: {', '.join(args.models)}")
    if args.shots:
        print(f"Shots: {', '.join(map(str, args.shots))}")
    print()
    
    # Find existing models
    print("Scanning for existing models...")
    found = find_existing_models(args.model_location, args.models)
    
    total_found = sum(len(v) for v in found.values())
    if total_found == 0:
        print("No trained models found!")
        return
    
    # Count models by method
    for method in ['energy', 'atlas']:
        if method in found and found[method]:
            print(f"  {method}: {len(found[method])} models")
    
    # Count baseline models by type
    if found['baseline']:
        baseline_counts = {}
        for model, config, k, btype in found['baseline']:
            baseline_counts[btype] = baseline_counts.get(btype, 0) + 1
        for btype, count in baseline_counts.items():
            print(f"  {btype}: {count} models")
    print()
    
    # Build commands
    commands = []
    
    if 'energy' in methods and found['energy']:
        print(f"Building Energy re-evaluation commands...")
        for model, config_tag, k in found['energy']:
            if args.shots and k not in args.shots:
                continue
            
            hparams = parse_energy_config(config_tag)
            if not hparams:
                print(f"  Warning: Could not parse config {config_tag}, skipping")
                continue
            
            cmd = [
                sys.executable,
                "imagenet_energy_train.py",
                "--model", model,
                "--k", str(k),
                "--sigma_lr", hparams['sigma_lr'],
                "--svd_keep_topk", hparams['topk'],
                "--initialize_sigma", hparams['init_mode'],
                "--warmup_ratio", hparams['warmup_ratio'],
                "--sigma_wd", hparams['sigma_wd'],
                "--eval_only",
            ]
            commands.append(('energy', model, k, cmd))
    
    if 'atlas' in methods and found['atlas']:
        print(f"Building Atlas re-evaluation commands...")
        for model, config_tag, k in found['atlas']:
            if args.shots and k not in args.shots:
                continue
            
            hparams = parse_atlas_config(config_tag)
            if not hparams:
                print(f"  Warning: Could not parse config {config_tag}, skipping")
                continue
            
            cmd = [
                sys.executable,
                "imagenet_atlas_train.py",
                "--model", model,
                "--k", str(k),
                "--lr", hparams['lr'],
                "--num_tasks", hparams['num_basis'],
                "--eval_only",
                "--config_file", "config/config_reverse.yaml",  # Ensure config is loaded
            ]
            commands.append(('atlas', model, k, cmd))
    
    # Build baseline commands
    baseline_methods = ['linear_probe', 'tip_adapter', 'lp++', 'lora']
    for baseline_method in baseline_methods:
        if baseline_method not in methods:
            continue
        
        # Filter baseline models by type
        matching_baselines = [(m, c, k, bt) for m, c, k, bt in found['baseline'] 
                             if bt == baseline_method]
        
        if not matching_baselines:
            continue
        
        print(f"Building {baseline_method} re-evaluation commands...")
        for model, config_tag, k, btype in matching_baselines:
            if args.shots and k not in args.shots:
                continue
            
            hparams = parse_baseline_config(config_tag, btype)
            if not hparams:
                print(f"  Warning: Could not parse config {config_tag}, skipping")
                continue
            
            cmd = [
                sys.executable,
                "imagenet_baselines_train.py",
                "--model", model,
                "--k", str(k),
                "--baseline_method", hparams['method'],
                "--eval_only",
                "--config_file", "config/config_reverse.yaml",
            ]
            
            # Add method-specific arguments
            if baseline_method == 'linear_probe':
                cmd.extend([
                    "--lp_lr", hparams['lr'],
                    "--lp_epochs", hparams['epochs'],
                    "--lp_wd", hparams['wd'],
                ])
            elif baseline_method == 'tip_adapter':
                cmd.extend([
                    "--adapter_wd", hparams['wd'],
                ])
            elif baseline_method == 'lp++':
                cmd.extend([
                    "--adapter_wd", hparams['wd'],
                ])
            elif baseline_method == 'lora':
                cmd.extend([
                    "--lora_r", hparams['r'],
                    "--lora_alpha", hparams['alpha'],
                    "--lora_lr", hparams['lr'],
                    "--lora_epochs", hparams['epochs'],
                    "--lora_wd", hparams['wd'],
                ])
            
            commands.append((baseline_method, model, k, cmd))
    
    print(f"\nTotal commands to execute: {len(commands)}")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        for i, (method, model, k, cmd) in enumerate(commands, 1):
            gpu_id = gpu_ids[(i - 1) % num_gpus]
            print(f"\n[{i}/{len(commands)}] [{method}] {model} {k}-shot (GPU {gpu_id}):")
            print("  " + " ".join(cmd))
        return
    
    # Execute commands in parallel
    print("\n" + "=" * 80)
    print(f"Executing re-evaluation commands with {num_gpus} parallel workers...")
    print("=" * 80 + "\n")
    
    # Prepare job arguments with GPU assignments (round-robin)
    job_args = []
    for i, (method, model, k, cmd) in enumerate(commands):
        gpu_id = gpu_ids[i % num_gpus]
        job_args.append((method, model, k, cmd, gpu_id, i + 1, len(commands)))
    
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    try:
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            # Submit all jobs
            futures = {executor.submit(run_command_with_gpu, args): args for args in job_args}
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    success, method, model, k, gpu_id, elapsed = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"  ✗ Job failed with exception: {str(e)}")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        executor.shutdown(wait=False, cancel_futures=True)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Re-evaluation complete in {total_time:.1f}s")
    print(f"Results: {success_count} succeeded, {fail_count} failed")
    print(f"Speedup: ~{len(commands) / (total_time / 60):.1f}x compared to sequential")
    print("=" * 80)


if __name__ == "__main__":
    main()

