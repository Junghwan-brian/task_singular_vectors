"""
GPU Memory Calculator for Different Methods
============================================

Measures GPU memory usage for different methods WITHOUT loading datasets.
Only loads models and task vectors to GPU to measure static memory footprint.

Usage:
    # Single method
    python gpu_calculation.py --mode remote_sensing --method atlas --test_dataset CLRS
    python gpu_calculation.py --mode remote_sensing --method energy --test_dataset CLRS
    python gpu_calculation.py --mode reverse --method baseline --baseline_method lora
    
    # All methods at once
    python gpu_calculation.py --mode remote_sensing --test_dataset CLRS --all
    python gpu_calculation.py --mode reverse --test_dataset DTD --all
"""

import os
import argparse
import torch
import json
import sys
from typing import Dict, Optional, List
from omegaconf import OmegaConf

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory stats"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_config_and_datasets(mode: str) -> tuple:
    """Load config file and get dataset list based on mode"""
    if mode == "remote_sensing":
        config_path = "config/config_remote_sensing.yaml"
    elif mode == "reverse":
        config_path = "config/config_reverse.yaml"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    cfg = OmegaConf.load(config_path)
    all_datasets = list(cfg.DATASETS_ALL)
    
    return cfg, all_datasets


def measure_atlas_remote_sensing(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for Atlas method on remote sensing datasets"""
    from atlas_src.modeling import ImageEncoder
    from atlas_src.composition import WeightedImageEncoder
    from src.models.task_vectors import NonLinearTaskVector
    from src.utils.variables_and_paths import get_zeroshot_path, get_finetuned_path
    
    print("\n" + "="*80)
    print("📊 Atlas (Remote Sensing) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    # Get basis datasets (leave-one-out)
    test_ds = args.test_dataset
    if test_ds:
        basis_datasets = [d for d in all_datasets if d != test_ds]
        print(f"Test dataset: {test_ds}")
        print(f"Basis datasets: {len(basis_datasets)} tasks (leave-one-out)")
    else:
        # Use all datasets except the first one (convention)
        basis_datasets = all_datasets[1:]
        print(f"Using {len(basis_datasets)} datasets as basis (excluding first: {all_datasets[0]})")
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    # Load task vectors
    print(f"\n2️⃣  Loading {len(basis_datasets)} task vectors...")
    ft_checks = {}
    for i, dataset in enumerate(basis_datasets[:args.max_tasks], 1):
        dataset_val = dataset + "Val"
        finetuned_path = get_finetuned_path(
            args.model_location, dataset_val, args.model)
        
        if os.path.exists(finetuned_path):
            ft_checks[dataset] = torch.load(finetuned_path, map_location="cpu")
            print(f"   [{i}/{len(basis_datasets)}] Loaded {dataset}")
        else:
            print(f"   [{i}/{len(basis_datasets)}] ⚠️  Missing {dataset}")
    
    # Load zeroshot checkpoint
    first_dataset_val = basis_datasets[0] + "Val"
    zeroshot_path = get_zeroshot_path(
        args.model_location, first_dataset_val, args.model)
    print(f"\n   Loading zeroshot model: {os.path.basename(zeroshot_path)}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu")
    
    # Create task vectors (in CPU memory initially)
    task_vectors = []
    for dataset, ft_check in ft_checks.items():
        tv = NonLinearTaskVector(args.model, ptm_check, ft_check)
        task_vectors.append(tv)
    
    print(f"   ✓ Created {len(task_vectors)} task vectors (in CPU memory)")
    
    # Create WeightedImageEncoder (this moves task vectors to GPU)
    print("\n3️⃣  Creating WeightedImageEncoder (moving task vectors to GPU)...")
    weighted_encoder = WeightedImageEncoder(
        image_encoder, 
        task_vectors,
        blockwise=getattr(args, 'blockwise_coef', True),
        partition=getattr(args, 'partition', None),
    )
    weighted_encoder = weighted_encoder.cuda()
    
    mem_after_weighted = get_gpu_memory_mb()
    task_vectors_mem = mem_after_weighted - mem_after_encoder
    print(f"   Task vectors: {task_vectors_mem:.2f} MB")
    print(f"   Total: {mem_after_weighted:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in weighted_encoder.parameters())
    trainable_params = sum(p.numel() for p in weighted_encoder.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    print(f"Task vectors:        {task_vectors_mem:>10.2f} MB  ({len(task_vectors)} vectors)")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {mem_after_weighted:>10.2f} MB")
    print(f"\nTotal parameters:    {total_params:>10,}")
    print(f"Trainable params:    {trainable_params:>10,}  (coefficients only)")
    print("="*80)
    
    # Cleanup
    del weighted_encoder, image_encoder, task_vectors
    reset_gpu_memory()
    
    return {
        "method": "atlas",
        "mode": "remote_sensing",
        "model": args.model,
        "test_dataset": test_ds,
        "num_task_vectors": len(ft_checks),
        "base_encoder_mb": encoder_mem,
        "task_vectors_mb": task_vectors_mem,
        "total_gpu_mb": mem_after_weighted,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def measure_energy_remote_sensing(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for Energy method on remote sensing datasets"""
    from src.models import ImageEncoder
    from src.models.task_vectors import NonLinearTaskVector
    from src.utils.variables_and_paths import get_zeroshot_path, get_finetuned_path
    from src.utils.sigma_param import SigmaParametrization
    
    print("\n" + "="*80)
    print("📊 Energy (Remote Sensing) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    # Get basis datasets
    test_ds = args.test_dataset
    if test_ds:
        basis_datasets = [d for d in all_datasets if d != test_ds]
        print(f"Test dataset: {test_ds}")
        print(f"Basis datasets: {len(basis_datasets)} tasks (leave-one-out)")
    else:
        basis_datasets = all_datasets[1:]
        print(f"Using {len(basis_datasets)} datasets as basis (excluding first: {all_datasets[0]})")
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args.model).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    # Load task vectors for SVD
    print(f"\n2️⃣  Loading {len(basis_datasets)} task vectors for SVD compression...")
    ft_checks = []
    for i, dataset in enumerate(basis_datasets[:args.max_tasks], 1):
        dataset_val = dataset + "Val"
        finetuned_path = get_finetuned_path(
            args.model_location, dataset_val, args.model)
        
        if os.path.exists(finetuned_path):
            ft_checks.append(torch.load(finetuned_path, map_location="cpu"))
            print(f"   [{i}/{len(basis_datasets)}] Loaded {dataset}")
    
    first_dataset_val = basis_datasets[0] + "Val"
    zeroshot_path = get_zeroshot_path(
        args.model_location, first_dataset_val, args.model)
    print(f"\n   Loading zeroshot model: {os.path.basename(zeroshot_path)}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu")
    
    # Create task vectors
    task_vectors = [
        NonLinearTaskVector(args.model, ptm_check, check) 
        for check in ft_checks
    ]
    print(f"   ✓ Created {len(task_vectors)} task vectors")
    
    # Compute SVD compression
    print(f"\n3️⃣  Computing SVD compression (k={args.svd_keep_topk} per task)...")
    from energy_train_remote_sensing import compute_and_sum_svd_mem_reduction
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.device = 'cuda'
            self.DATASETS = basis_datasets
            self.svd_keep_topk = args.svd_keep_topk
    
    config = SimpleConfig()
    svd_dict = compute_and_sum_svd_mem_reduction(
        task_vectors, config, sigma_reduce="mean")
    
    print(f"   ✓ SVD compression complete")
    
    # Create sigma modules
    print("\n4️⃣  Creating Sigma modules...")
    sigma_modules = torch.nn.ModuleDict()
    sigma_key_map = {}
    
    for key, fv in svd_dict.items():
        if isinstance(fv, list) and len(fv) == 3:
            U_orth, diag_s, V_orth = fv
            sigma_vec = torch.diagonal(diag_s).clone().detach()
            
            safe_key = key.replace(".", "_")
            if safe_key in sigma_key_map:
                suffix = 1
                candidate = f"{safe_key}_{suffix}"
                while candidate in sigma_key_map:
                    suffix += 1
                    candidate = f"{safe_key}_{suffix}"
                safe_key = candidate
            
            sigma_key_map[safe_key] = key
            sigma_modules[safe_key] = SigmaParametrization(
                U_orth.cpu(), V_orth.cpu(), sigma_vec.cpu())
    
    sigma_modules = sigma_modules.cuda()
    
    mem_after_sigma = get_gpu_memory_mb()
    sigma_mem = mem_after_sigma - mem_after_encoder
    print(f"   Sigma modules: {sigma_mem:.2f} MB ({len(sigma_modules)} modules)")
    print(f"   Total: {mem_after_sigma:.2f} MB")
    
    # Count parameters
    total_encoder_params = sum(p.numel() for p in image_encoder.parameters())
    sigma_params = sum(p.numel() for p in sigma_modules.parameters())
    trainable_sigma_params = sum(
        p.numel() for p in sigma_modules.parameters() if p.requires_grad)
    
    # Calculate compression ratio
    original_tv_size = len(task_vectors) * 432  # MB per task vector
    compression_ratio = (original_tv_size / sigma_mem) if sigma_mem > 0 else 0
    
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    print(f"Sigma modules:       {sigma_mem:>10.2f} MB  ({len(sigma_modules)} modules)")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {mem_after_sigma:>10.2f} MB")
    print(f"\nOriginal TV size:    {original_tv_size:>10.2f} MB  ({len(task_vectors)} × 432 MB)")
    print(f"Compression ratio:   {compression_ratio:>10.1f}x")
    print(f"\nEncoder parameters:  {total_encoder_params:>10,}")
    print(f"Sigma parameters:    {sigma_params:>10,}")
    print(f"Trainable (sigma):   {trainable_sigma_params:>10,}")
    print("="*80)
    
    # Cleanup
    del sigma_modules, image_encoder, task_vectors
    reset_gpu_memory()
    
    return {
        "method": "energy",
        "mode": "remote_sensing",
        "model": args.model,
        "test_dataset": test_ds,
        "num_task_vectors": len(task_vectors),
        "svd_keep_topk": args.svd_keep_topk,
        "base_encoder_mb": encoder_mem,
        "sigma_modules_mb": sigma_mem,
        "total_gpu_mb": mem_after_sigma,
        "original_tv_size_mb": original_tv_size,
        "compression_ratio": compression_ratio,
        "encoder_params": total_encoder_params,
        "sigma_params": sigma_params,
        "trainable_params": trainable_sigma_params,
    }


def measure_baseline_remote_sensing(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for baseline methods on remote sensing datasets"""
    from src.models import ImageEncoder
    
    print("\n" + "="*80)
    print(f"📊 Baseline: {args.baseline_method} (Remote Sensing) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args.model).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    additional_mem = 0
    additional_params = 0
    trainable_params = 0
    cache_info = None
    
    if args.baseline_method == 'linear_probe':
        print("\n2️⃣  Linear probe: Only classification head (trainable)")
        # Classification head is small, typically ~1-5MB depending on num_classes
        # We'll estimate based on typical remote sensing datasets (10-50 classes)
        avg_num_classes = 30  # Average for remote sensing
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        head_params = feature_dim * avg_num_classes + avg_num_classes  # weight + bias
        trainable_params = head_params
        additional_mem = head_params * 4 / (1024 ** 2)  # Convert to MB
        
    elif args.baseline_method == 'tip':
        print(f"\n2️⃣  TIP Adapter: Feature cache + beta parameter (k={args.k_shot} shot)")
        # Feature cache size: k × num_classes × feature_dim × 4 bytes
        k = args.k_shot
        avg_num_classes = 30
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        
        cache_samples = k * avg_num_classes
        cache_size_mb = cache_samples * feature_dim * 4 / (1024 ** 2)
        
        # TIP has beta_alpha (scalar), cache_keys and cache_values
        tip_params = cache_samples * feature_dim * 2 + 1  # keys + values + beta
        trainable_params = 1  # Only beta_alpha is trainable
        additional_params = tip_params
        additional_mem = cache_size_mb
        
        cache_info = {
            "k_shot": k,
            "cache_samples": cache_samples,
            "feature_dim": feature_dim,
            "cache_size_mb": cache_size_mb
        }
        
        print(f"   Feature cache: {cache_size_mb:.2f} MB ({cache_samples} samples × {feature_dim} dims)")
        print(f"   Beta parameter: ~0.00 MB (1 scalar)")
        
    elif args.baseline_method == 'lp++':
        print(f"\n2️⃣  LP++ Adapter: Feature cache + alpha_vec + adapter (k={args.k_shot} shot)")
        # Similar to TIP but with additional alpha_vec and adapter weights
        k = args.k_shot
        avg_num_classes = 30
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        
        cache_samples = k * avg_num_classes
        cache_size_mb = cache_samples * feature_dim * 4 / (1024 ** 2)
        
        # LP++ has alpha_vec (num_classes) and adapter (feature_dim × num_classes)
        lpp_params = cache_samples * feature_dim * 2 + avg_num_classes + feature_dim * avg_num_classes
        trainable_params = avg_num_classes + feature_dim * avg_num_classes  # alpha_vec + adapter
        additional_params = lpp_params
        additional_mem = cache_size_mb + trainable_params * 4 / (1024 ** 2)
        
        cache_info = {
            "k_shot": k,
            "cache_samples": cache_samples,
            "feature_dim": feature_dim,
            "cache_size_mb": cache_size_mb,
            "alpha_vec_params": avg_num_classes,
            "adapter_params": feature_dim * avg_num_classes
        }
        
        print(f"   Feature cache: {cache_size_mb:.2f} MB ({cache_samples} samples × {feature_dim} dims)")
        print(f"   Alpha vec: {avg_num_classes} params")
        print(f"   Adapter: {feature_dim * avg_num_classes:,} params")
    
    else:
        raise ValueError(f"Unknown baseline method: {args.baseline_method}")
    
    total_mem = mem_after_encoder + additional_mem
    total_params = sum(p.numel() for p in image_encoder.parameters())
    
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    if additional_mem > 0:
        print(f"Additional modules:  {additional_mem:>10.2f} MB")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {total_mem:>10.2f} MB")
    print(f"\nTotal parameters:    {total_params:>10,}")
    if trainable_params > 0:
        print(f"Trainable params:    {trainable_params:>10,}")
    print("="*80)
    
    # Cleanup
    del image_encoder
    reset_gpu_memory()
    
    result = {
        "method": "baseline",
        "baseline_method": args.baseline_method,
        "mode": "remote_sensing",
        "model": args.model,
        "k_shot": args.k_shot,
        "base_encoder_mb": encoder_mem,
        "additional_mb": additional_mem,
        "total_gpu_mb": total_mem,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    
    if cache_info:
        result["cache_info"] = cache_info
    
    return result


def measure_atlas_reverse(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for Atlas method on general (reverse) datasets"""
    from atlas_src.modeling import ImageEncoder
    from atlas_src.composition import WeightedImageEncoder
    from src.models.task_vectors import NonLinearTaskVector
    from src.utils.variables_and_paths import get_zeroshot_path, get_finetuned_path
    
    print("\n" + "="*80)
    print("📊 Atlas (Reverse/General Datasets) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    # Get basis datasets (leave-one-out)
    test_ds = args.test_dataset
    if test_ds:
        basis_datasets = [d for d in all_datasets if d != test_ds]
        print(f"Test dataset: {test_ds}")
        print(f"Basis datasets: {len(basis_datasets)} tasks (leave-one-out)")
    else:
        basis_datasets = all_datasets[1:]
        print(f"Using {len(basis_datasets)} datasets as basis (excluding first: {all_datasets[0]})")
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    # Load task vectors
    print(f"\n2️⃣  Loading {len(basis_datasets)} task vectors...")
    ft_checks = {}
    for i, dataset in enumerate(basis_datasets[:args.max_tasks], 1):
        finetuned_path = get_finetuned_path(
            args.model_location, dataset, args.model)
        
        if os.path.exists(finetuned_path):
            ft_checks[dataset] = torch.load(finetuned_path, map_location="cpu")
            print(f"   [{i}/{len(basis_datasets)}] Loaded {dataset}")
        else:
            print(f"   [{i}/{len(basis_datasets)}] ⚠️  Missing {dataset}")
    
    # Load zeroshot checkpoint
    first_dataset = basis_datasets[0]
    zeroshot_path = get_zeroshot_path(
        args.model_location, first_dataset, args.model)
    print(f"\n   Loading zeroshot model: {os.path.basename(zeroshot_path)}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu")
    
    # Create task vectors
    task_vectors = []
    for dataset, ft_check in ft_checks.items():
        tv = NonLinearTaskVector(args.model, ptm_check, ft_check)
        task_vectors.append(tv)
    
    print(f"   ✓ Created {len(task_vectors)} task vectors (in CPU memory)")
    
    # Create WeightedImageEncoder
    print("\n3️⃣  Creating WeightedImageEncoder (moving task vectors to GPU)...")
    weighted_encoder = WeightedImageEncoder(
        image_encoder, 
        task_vectors,
        blockwise=getattr(args, 'blockwise_coef', True),
        partition=getattr(args, 'partition', None),
    )
    weighted_encoder = weighted_encoder.cuda()
    
    mem_after_weighted = get_gpu_memory_mb()
    task_vectors_mem = mem_after_weighted - mem_after_encoder
    print(f"   Task vectors: {task_vectors_mem:.2f} MB")
    print(f"   Total: {mem_after_weighted:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in weighted_encoder.parameters())
    trainable_params = sum(p.numel() for p in weighted_encoder.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    print(f"Task vectors:        {task_vectors_mem:>10.2f} MB  ({len(task_vectors)} vectors)")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {mem_after_weighted:>10.2f} MB")
    print(f"\nTotal parameters:    {total_params:>10,}")
    print(f"Trainable params:    {trainable_params:>10,}  (coefficients only)")
    print("="*80)
    
    # Cleanup
    del weighted_encoder, image_encoder, task_vectors
    reset_gpu_memory()
    
    return {
        "method": "atlas",
        "mode": "reverse",
        "model": args.model,
        "test_dataset": test_ds,
        "num_task_vectors": len(ft_checks),
        "base_encoder_mb": encoder_mem,
        "task_vectors_mb": task_vectors_mem,
        "total_gpu_mb": mem_after_weighted,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def measure_energy_reverse(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for Energy method on general (reverse) datasets"""
    from src.models import ImageEncoder
    from src.models.task_vectors import NonLinearTaskVector
    from src.utils.variables_and_paths import get_zeroshot_path, get_finetuned_path
    from src.utils.sigma_param import SigmaParametrization
    
    print("\n" + "="*80)
    print("📊 Energy (Reverse/General Datasets) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    # Get basis datasets
    test_ds = args.test_dataset
    if test_ds:
        basis_datasets = [d for d in all_datasets if d != test_ds]
        print(f"Test dataset: {test_ds}")
        print(f"Basis datasets: {len(basis_datasets)} tasks (leave-one-out)")
    else:
        basis_datasets = all_datasets[1:]
        print(f"Using {len(basis_datasets)} datasets as basis (excluding first: {all_datasets[0]})")
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args.model).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    # Load task vectors for SVD
    print(f"\n2️⃣  Loading {len(basis_datasets)} task vectors for SVD compression...")
    ft_checks = []
    for i, dataset in enumerate(basis_datasets[:args.max_tasks], 1):
        finetuned_path = get_finetuned_path(
            args.model_location, dataset, args.model)
        
        if os.path.exists(finetuned_path):
            ft_checks.append(torch.load(finetuned_path, map_location="cpu"))
            print(f"   [{i}/{len(basis_datasets)}] Loaded {dataset}")
    
    first_dataset = basis_datasets[0]
    zeroshot_path = get_zeroshot_path(
        args.model_location, first_dataset, args.model)
    print(f"\n   Loading zeroshot model: {os.path.basename(zeroshot_path)}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu")
    
    # Create task vectors
    task_vectors = [
        NonLinearTaskVector(args.model, ptm_check, check) 
        for check in ft_checks
    ]
    print(f"   ✓ Created {len(task_vectors)} task vectors")
    
    # Compute SVD compression
    print(f"\n3️⃣  Computing SVD compression (k={args.svd_keep_topk} per task)...")
    from energy_train_remote_sensing import compute_and_sum_svd_mem_reduction
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.device = 'cuda'
            self.DATASETS = basis_datasets
            self.svd_keep_topk = args.svd_keep_topk
    
    config = SimpleConfig()
    svd_dict = compute_and_sum_svd_mem_reduction(
        task_vectors, config, sigma_reduce="mean")
    
    print(f"   ✓ SVD compression complete")
    
    # Create sigma modules
    print("\n4️⃣  Creating Sigma modules...")
    sigma_modules = torch.nn.ModuleDict()
    sigma_key_map = {}
    
    for key, fv in svd_dict.items():
        if isinstance(fv, list) and len(fv) == 3:
            U_orth, diag_s, V_orth = fv
            sigma_vec = torch.diagonal(diag_s).clone().detach()
            
            safe_key = key.replace(".", "_")
            if safe_key in sigma_key_map:
                suffix = 1
                candidate = f"{safe_key}_{suffix}"
                while candidate in sigma_key_map:
                    suffix += 1
                    candidate = f"{safe_key}_{suffix}"
                safe_key = candidate
            
            sigma_key_map[safe_key] = key
            sigma_modules[safe_key] = SigmaParametrization(
                U_orth.cpu(), V_orth.cpu(), sigma_vec.cpu())
    
    sigma_modules = sigma_modules.cuda()
    
    mem_after_sigma = get_gpu_memory_mb()
    sigma_mem = mem_after_sigma - mem_after_encoder
    print(f"   Sigma modules: {sigma_mem:.2f} MB ({len(sigma_modules)} modules)")
    print(f"   Total: {mem_after_sigma:.2f} MB")
    
    # Count parameters
    total_encoder_params = sum(p.numel() for p in image_encoder.parameters())
    sigma_params = sum(p.numel() for p in sigma_modules.parameters())
    trainable_sigma_params = sum(
        p.numel() for p in sigma_modules.parameters() if p.requires_grad)
    
    # Calculate compression ratio
    original_tv_size = len(task_vectors) * 432  # MB per task vector
    compression_ratio = (original_tv_size / sigma_mem) if sigma_mem > 0 else 0
    num_tasks = len(task_vectors)
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    print(f"Sigma modules:       {sigma_mem:>10.2f} MB  ({len(sigma_modules)} modules)")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {mem_after_sigma:>10.2f} MB")
    print(f"\nOriginal TV size:    {original_tv_size:>10.2f} MB  ({len(task_vectors)} × 432 MB)")
    print(f"Compression ratio:   {compression_ratio:>10.1f}x")
    print(f"\nEncoder parameters:  {total_encoder_params:>10,}")
    print(f"Sigma parameters:    {sigma_params:>10,}")
    print(f"Trainable (sigma):   {trainable_sigma_params:>10,}")
    print("="*80)
    
    # Cleanup
    del sigma_modules, image_encoder, task_vectors
    reset_gpu_memory()
    
    return {
        "method": "energy",
        "mode": "reverse",
        "model": args.model,
        "test_dataset": test_ds,
        "num_task_vectors": num_tasks,
        "svd_keep_topk": args.svd_keep_topk,
        "base_encoder_mb": encoder_mem,
        "sigma_modules_mb": sigma_mem,
        "total_gpu_mb": mem_after_sigma,
        "original_tv_size_mb": original_tv_size,
        "compression_ratio": compression_ratio,
        "encoder_params": total_encoder_params,
        "sigma_params": sigma_params,
        "trainable_params": trainable_sigma_params,
    }


def measure_baseline_reverse(args, cfg, all_datasets) -> Dict:
    """Measure GPU memory for baseline methods on general (reverse) datasets"""
    from src.models import ImageEncoder
    
    print("\n" + "="*80)
    print(f"📊 Baseline: {args.baseline_method} (Reverse) GPU Memory Calculation")
    print("="*80)
    
    reset_gpu_memory()
    
    mem_before = get_gpu_memory_mb()
    print(f"\n📍 Initial GPU memory: {mem_before:.2f} MB")
    
    # Load base encoder
    print("\n1️⃣  Loading base encoder...")
    image_encoder = ImageEncoder(args.model).cuda()
    mem_after_encoder = get_gpu_memory_mb()
    encoder_mem = mem_after_encoder - mem_before
    print(f"   Base encoder: {encoder_mem:.2f} MB")
    print(f"   Total: {mem_after_encoder:.2f} MB")
    
    additional_mem = 0
    additional_params = 0
    trainable_params = 0
    cache_info = None
    
    if args.baseline_method == 'linear_probe':
        print("\n2️⃣  Linear probe: Only classification head (trainable)")
        # Classification head size estimation
        avg_num_classes = 50  # Average for general datasets
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        head_params = feature_dim * avg_num_classes + avg_num_classes
        trainable_params = head_params
        additional_mem = head_params * 4 / (1024 ** 2)
        
    elif args.baseline_method == 'tip':
        print(f"\n2️⃣  TIP Adapter: Feature cache + beta parameter (k={args.k_shot} shot)")
        k = args.k_shot
        avg_num_classes = 50  # Average for general datasets
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        
        cache_samples = k * avg_num_classes
        cache_size_mb = cache_samples * feature_dim * 4 / (1024 ** 2)
        
        tip_params = cache_samples * feature_dim * 2 + 1
        trainable_params = 1
        additional_params = tip_params
        additional_mem = cache_size_mb
        
        cache_info = {
            "k_shot": k,
            "cache_samples": cache_samples,
            "feature_dim": feature_dim,
            "cache_size_mb": cache_size_mb
        }
        
        print(f"   Feature cache: {cache_size_mb:.2f} MB ({cache_samples} samples × {feature_dim} dims)")
        print(f"   Beta parameter: ~0.00 MB (1 scalar)")
        
    elif args.baseline_method == 'lp++':
        print(f"\n2️⃣  LP++ Adapter: Feature cache + alpha_vec + adapter (k={args.k_shot} shot)")
        k = args.k_shot
        avg_num_classes = 50
        feature_dim = 512 if 'B-32' in args.model else 768 if 'B-16' in args.model else 768
        
        cache_samples = k * avg_num_classes
        cache_size_mb = cache_samples * feature_dim * 4 / (1024 ** 2)
        
        lpp_params = cache_samples * feature_dim * 2 + avg_num_classes + feature_dim * avg_num_classes
        trainable_params = avg_num_classes + feature_dim * avg_num_classes
        additional_params = lpp_params
        additional_mem = cache_size_mb + trainable_params * 4 / (1024 ** 2)
        
        cache_info = {
            "k_shot": k,
            "cache_samples": cache_samples,
            "feature_dim": feature_dim,
            "cache_size_mb": cache_size_mb,
            "alpha_vec_params": avg_num_classes,
            "adapter_params": feature_dim * avg_num_classes
        }
        
        print(f"   Feature cache: {cache_size_mb:.2f} MB ({cache_samples} samples × {feature_dim} dims)")
        print(f"   Alpha vec: {avg_num_classes} params")
        print(f"   Adapter: {feature_dim * avg_num_classes:,} params")
    
    else:
        raise ValueError(f"Unknown baseline method for reverse: {args.baseline_method}")
    
    total_mem = mem_after_encoder + additional_mem
    total_params = sum(p.numel() for p in image_encoder.parameters())
    
    print("\n" + "="*80)
    print("📈 Final Results:")
    print("="*80)
    print(f"Base encoder:        {encoder_mem:>10.2f} MB")
    if additional_mem > 0:
        print(f"Additional modules:  {additional_mem:>10.2f} MB")
    print(f"{'─'*80}")
    print(f"Total GPU memory:    {total_mem:>10.2f} MB")
    print(f"\nTotal parameters:    {total_params:>10,}")
    if trainable_params > 0:
        print(f"Trainable params:    {trainable_params:>10,}")
    print("="*80)
    
    # Cleanup
    del image_encoder
    reset_gpu_memory()
    
    result = {
        "method": "baseline",
        "baseline_method": args.baseline_method,
        "mode": "reverse",
        "model": args.model,
        "k_shot": args.k_shot,
        "base_encoder_mb": encoder_mem,
        "additional_mb": additional_mem,
        "total_gpu_mb": total_mem,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    
    if cache_info:
        result["cache_info"] = cache_info
    
    return result


def generate_save_path(args, method: str, baseline_method: Optional[str] = None) -> str:
    """Generate adaptive save path for results"""
    os.makedirs("results/gpu_memory", exist_ok=True)
    
    # Format: results/gpu_memory/{mode}_{model}_{method}_{baseline_method}_k{k}.json
    model_safe = args.model.replace("/", "-").replace(":", "-")
    
    if method == "baseline" and baseline_method:
        filename = f"{args.mode}_{model_safe}_{method}_{baseline_method}_k{args.k_shot}.json"
    else:
        filename = f"{args.mode}_{model_safe}_{method}_k{args.k_shot}.json"
    
    if args.test_dataset:
        filename = f"{args.test_dataset}_{filename}"
    
    return os.path.join("results/gpu_memory", filename)


def run_all_methods(args, cfg, all_datasets) -> List[Dict]:
    """Run all methods sequentially and collect results"""
    results = []
    
    print("\n" + "="*100)
    print(f"🚀 Running ALL methods for mode={args.mode}, model={args.model}, k={args.k_shot}")
    if args.test_dataset:
        print(f"   Test dataset: {args.test_dataset}")
    print("="*100)
    
    # 1. Atlas
    try:
        print("\n" + "🔹"*50)
        print("Running Atlas...")
        print("🔹"*50)
        if args.mode == "remote_sensing":
            result = measure_atlas_remote_sensing(args, cfg, all_datasets)
        else:
            result = measure_atlas_reverse(args, cfg, all_datasets)
        results.append(result)
        
        # Save individual result
        save_path = generate_save_path(args, "atlas")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"✅ Saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Atlas failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Energy
    try:
        print("\n" + "🔹"*50)
        print("Running Energy...")
        print("🔹"*50)
        if args.mode == "remote_sensing":
            result = measure_energy_remote_sensing(args, cfg, all_datasets)
        else:
            result = measure_energy_reverse(args, cfg, all_datasets)
        results.append(result)
        
        # Save individual result
        save_path = generate_save_path(args, "energy")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"✅ Saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Energy failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Baselines - same for both modes now
    baseline_methods = ["linear_probe", "tip", "lp++"]
    
    for baseline_method in baseline_methods:
        try:
            print("\n" + "🔹"*50)
            print(f"Running Baseline: {baseline_method}...")
            print("🔹"*50)
            
            # Temporarily set baseline_method
            original_method = args.baseline_method
            args.baseline_method = baseline_method
            
            if args.mode == "remote_sensing":
                result = measure_baseline_remote_sensing(args, cfg, all_datasets)
            else:
                result = measure_baseline_reverse(args, cfg, all_datasets)
            results.append(result)
            
            # Save individual result
            save_path = generate_save_path(args, "baseline", baseline_method)
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"✅ Saved to: {save_path}")
            
            # Restore original
            args.baseline_method = original_method
            
        except Exception as e:
            print(f"❌ Baseline {baseline_method} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*100)
    print("📊 SUMMARY: GPU Memory Usage Comparison")
    print("="*100)
    print(f"{'Method':<25} {'GPU Memory (MB)':<20} {'Trainable Params':<20}")
    print("-"*100)
    
    for result in results:
        method_name = result.get('method', 'unknown')
        if method_name == 'baseline':
            method_name = f"baseline_{result.get('baseline_method', 'unknown')}"
        
        gpu_mem = result.get('total_gpu_mb', 0)
        trainable = result.get('trainable_params', 0)
        
        print(f"{method_name:<25} {gpu_mem:<20.2f} {trainable:<20,}")
    
    print("="*100)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPU Memory Calculator for Different Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode and method
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["remote_sensing", "reverse"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["atlas", "energy", "baseline"],
        help="Method to measure (not needed if --all is specified)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all methods sequentially"
    )
    
    # Dataset and model
    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset (for leave-one-out mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model architecture (default from config)"
    )
    parser.add_argument(
        "--model_location",
        type=str,
        help="Model checkpoint directory (default from config)"
    )
    
    # Baseline-specific
    parser.add_argument(
        "--baseline_method",
        type=str,
        choices=["linear_probe", "tip", "lp++"],
        help="Baseline method (optional, will run all if not specified with --all or method=baseline)"
    )
    
    # K-shot
    parser.add_argument(
        "--k_shot",
        type=int,
        default=16,
        help="K-shot samples per class (default: 16 for TIP/LP++ comparison)"
    )
    
    # Energy-specific
    parser.add_argument(
        "--svd_keep_topk",
        type=int,
        default=12,
        help="Number of singular vectors to keep per task (for Energy)"
    )
    
    # Other
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=999,
        help="Maximum number of tasks to load (for testing)"
    )
    parser.add_argument(
        "--save_json",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--blockwise_coef",
        action="store_true",
        default=True,
        help="Use blockwise coefficients for Atlas"
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Partition size for Atlas"
    )
    parser.add_argument(
        "--openclip_cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.method:
        parser.error("Either --method or --all must be specified")
    
    # Set device
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires GPU.")
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    print("\n" + "="*80)
    print("🖥️  GPU Device Information")
    print("="*80)
    print(f"Device: {device_name}")
    print(f"Total Memory: {device_memory:.2f} GB")
    print("="*80)
    
    # Load config and datasets
    cfg, all_datasets = load_config_and_datasets(args.mode)
    print(f"\n📁 Loaded config from: config/config_{args.mode}.yaml")
    print(f"   All datasets: {all_datasets}")
    
    # Override config with command line arguments
    if args.model:
        cfg.model = args.model
    if args.model_location:
        cfg.model_location = args.model_location
    
    # Set defaults from config
    args.model = cfg.model
    
    # Set model_location based on mode if not provided
    if not args.model_location:
        if args.mode == "remote_sensing":
            args.model_location = "./models/checkpoints_remote_sensing"
        else:  # reverse
            args.model_location = "./models/checkpoints"
    
    args.model_location = os.path.expanduser(args.model_location)
    
    print(f"\n📦 Configuration:")
    print(f"   Mode: {args.mode}")
    print(f"   Model: {args.model}")
    print(f"   Model location: {args.model_location}")
    print(f"   K-shot: {args.k_shot}")
    if args.test_dataset:
        print(f"   Test dataset: {args.test_dataset}")
    
    # Measure memory based on mode and method
    try:
        if args.all:
            # Run all methods automatically
            results_list = run_all_methods(args, cfg, all_datasets)
            
            # Summary is already printed in run_all_methods
            print(f"\n✅ All results saved to: results/gpu_memory/")
        
        elif args.method == "baseline" and not args.baseline_method:
            # Run all baseline methods if baseline_method not specified
            print("\n" + "="*100)
            print(f"🚀 Running ALL Baseline methods for mode={args.mode}, model={args.model}, k={args.k_shot}")
            if args.test_dataset:
                print(f"   Test dataset: {args.test_dataset}")
            print("="*100)
            
            results_list = []
            baseline_methods = ["linear_probe", "tip", "lp++"]
            
            for baseline_method in baseline_methods:
                try:
                    print("\n" + "🔹"*50)
                    print(f"Running Baseline: {baseline_method}...")
                    print("🔹"*50)
                    
                    args.baseline_method = baseline_method
                    
                    if args.mode == "remote_sensing":
                        result = measure_baseline_remote_sensing(args, cfg, all_datasets)
                    else:
                        result = measure_baseline_reverse(args, cfg, all_datasets)
                    results_list.append(result)
                    
                    # Save individual result
                    save_path = generate_save_path(args, "baseline", baseline_method)
                    with open(save_path, 'w') as f:
                        json.dump(result, f, indent=4)
                    print(f"✅ Saved to: {save_path}")
                    
                except Exception as e:
                    print(f"❌ Baseline {baseline_method} failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Print summary
            print("\n" + "="*100)
            print("📊 SUMMARY: Baseline Methods GPU Memory Usage")
            print("="*100)
            print(f"{'Method':<25} {'GPU Memory (MB)':<20} {'Trainable Params':<20}")
            print("-"*100)
            
            for result in results_list:
                method_name = f"baseline_{result.get('baseline_method', 'unknown')}"
                gpu_mem = result.get('total_gpu_mb', 0)
                trainable = result.get('trainable_params', 0)
                print(f"{method_name:<25} {gpu_mem:<20.2f} {trainable:<20,}")
            
            print("="*100)
            print(f"\n✅ All baseline results saved to: results/gpu_memory/")
        
        else:
            # Single method
            if args.mode == "remote_sensing":
                if args.method == "atlas":
                    results = measure_atlas_remote_sensing(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "atlas")
                elif args.method == "energy":
                    results = measure_energy_remote_sensing(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "energy")
                elif args.method == "baseline":
                    results = measure_baseline_remote_sensing(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "baseline", args.baseline_method)
            
            elif args.mode == "reverse":
                if args.method == "atlas":
                    results = measure_atlas_reverse(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "atlas")
                elif args.method == "energy":
                    results = measure_energy_reverse(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "energy")
                elif args.method == "baseline":
                    results = measure_baseline_reverse(args, cfg, all_datasets)
                    save_path = generate_save_path(args, "baseline", args.baseline_method)
            
            # Save single result with adaptive path
            if args.save_json:
                # Use custom path if provided
                save_path = os.path.expanduser(args.save_json)
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n✅ Results saved to: {save_path}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
