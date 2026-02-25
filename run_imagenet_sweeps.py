#!/usr/bin/env python3
"""
Utility to launch ImageNet training sweeps across energy, atlas, and baseline methods.

Schedules commands over multiple GPUs in parallel for ImageNet-1k validation split training
with OOD evaluation (ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2).

Usage:
    python run_imagenet_sweeps.py
        Launch the full sweep.

    python run_imagenet_sweeps.py --dry-run
        Print commands without executing them.

    python run_imagenet_sweeps.py --skip-energy
        Only run atlas and baseline sweeps.

    python run_imagenet_sweeps.py --limit 10
        Run only the first 10 commands.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import threading
import time
from typing import List, Sequence


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_reverse.yaml")
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


CFG = _load_config()


def _resolve_model_root() -> str:
    default_root = os.path.join(".", "models", "checkpoints")
    raw_root = CFG.get("model_location", default_root) if isinstance(CFG, dict) else default_root
    return os.path.expanduser(raw_root)


MODEL_ROOT = _resolve_model_root()


def _sanitize_value(val) -> str:
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return "".join(
        ch if (str.isalnum(ch) or ch in {"-", "_"}) else "_"
        for ch in str(val).replace(".", "p")
    )


def _adapter_tag(value: str) -> str:
    display = (value or "none").strip().lower()
    if display in {"", "none"}:
        return "none"
    if display in {"lp++", "lpp"}:
        return "lp++"
    return display.replace(" ", "_")


def _path_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(os.path.abspath(path))


def _shot_folder(k: int) -> str:
    return f"{k}shots" if k > 0 else "fullshots"


def _energy_config_tag(sigma_lr: float, topk: int, init_mode: str, warmup_ratio: float, sigma_wd: float, k: int) -> str:
    init_value = (init_mode or "average").strip().lower()
    return "imagenet_energy_{}_{}_{}_{}_{}_{}shot".format(
        _sanitize_value(sigma_lr),
        _sanitize_value(topk),
        _sanitize_value(init_value),
        _sanitize_value(warmup_ratio),
        _sanitize_value(sigma_wd),
        _sanitize_value(k),
    )


def _atlas_config_tag(num_basis: int, lr: float, k: int) -> str:
    return "imagenet_atlas_{}_{}_{}_shot".format(
        _sanitize_value(max(int(num_basis), 0)),
        _sanitize_value(lr),
        _sanitize_value(k),
    )


def _baseline_config_tag(method: str, k: int, **kwargs) -> str:
    """Build config tag for baseline method."""
    if method == 'linear_probe':
        lr = _sanitize_value(kwargs.get('lp_lr', 1e-3))
        epochs = _sanitize_value(kwargs.get('lp_epochs', 20))
        wd = _sanitize_value(kwargs.get('lp_wd', 0.0))
        return f"imagenet_baseline_lp_{lr}_{epochs}_{wd}_{k}shot"
    elif method == 'tip_adapter':
        wd = _sanitize_value(kwargs.get('adapter_wd', 0.0))
        return f"imagenet_baseline_tip_{wd}_{k}shot"
    elif method == 'lp++':
        wd = _sanitize_value(kwargs.get('adapter_wd', 0.0))
        return f"imagenet_baseline_lpp_{wd}_{k}shot"
    elif method == 'lora':
        r = _sanitize_value(kwargs.get('lora_r', 8))
        alpha = _sanitize_value(kwargs.get('lora_alpha', 16.0))
        lr = _sanitize_value(kwargs.get('lora_lr', 1e-4))
        epochs = _sanitize_value(kwargs.get('lora_epochs', 20))
        wd = _sanitize_value(kwargs.get('lora_wd', 0.0))
        return f"imagenet_baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}_{k}shot"
    else:
        return f"imagenet_baseline_{method}_{k}shot"


def _expected_energy_paths(
    model: str,
    init_mode: str,
    sigma_lr: float,
    topk: int,
    sigma_wd: float,
    k: int,
    warmup_ratio: float,
) -> tuple[str, str]:
    config_tag = _energy_config_tag(sigma_lr, topk, init_mode, warmup_ratio, sigma_wd, k)
    dataset_dir = "ImageNetILSVRCVal"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    energy_pt = os.path.join(base_dir, "energy.pt")
    results_json = os.path.join(base_dir, "energy_results_imagenet.json")
    return energy_pt, results_json


def _expected_atlas_paths(
    model: str,
    lr: float,
    k: int,
    num_basis: int = 17,
) -> tuple[str, str]:
    config_tag = _atlas_config_tag(num_basis, lr, k)
    dataset_dir = "ImageNetILSVRCVal"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    atlas_pt = os.path.join(base_dir, "atlas.pt")
    results_json = os.path.join(base_dir, "atlas_results_imagenet.json")
    return atlas_pt, results_json


def _expected_baseline_paths(
    model: str,
    method: str,
    k: int,
    **kwargs
) -> str:
    """Return expected baseline results path."""
    config_tag = _baseline_config_tag(method, k, **kwargs)
    dataset_dir = "ImageNetILSVRCVal"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    results_json = os.path.join(base_dir, "baseline_results_imagenet.json")
    return results_json


# GPU and hyperparameter configurations
GPU_IDS = [0,1,2,3,4,5,6,7]  # Default GPU IDs

# Energy configurations for ImageNet
ENERGY_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
ENERGY_INITIALIZE_SIGMA = ["average"]
ENERGY_K = [4]
ENERGY_SVD_KEEP_TOPK = [12]
ENERGY_SIGMA_LR = [1e-3]
ENERGY_SIGMA_WD = [0.0]
ENERGY_WARMUP_RATIO = [0.1]

# Atlas configurations for ImageNet
ATLAS_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
ATLAS_K = [4, 16]
ATLAS_LR = [0.1]
ATLAS_NUM_BASIS = [17]  # Number of basis task vectors

# Baseline configurations for ImageNet
BASELINE_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
BASELINE_METHODS = ["linear_probe", "lp++", "tip_adapter", "lora"]
BASELINE_K = [4, 16]

# Linear Probe hyperparameters
BASELINE_LP_LR = [1e-3]
BASELINE_LP_EPOCHS = [20]
BASELINE_LP_WD = [0.0]

# Adapter (TIP/LP++) hyperparameters
BASELINE_ADAPTER_WD = [0.0]

# LoRA hyperparameters
BASELINE_LORA_R = [8]
BASELINE_LORA_ALPHA = [32.0]
BASELINE_LORA_LR = [1e-4]
BASELINE_LORA_EPOCHS = [20]
BASELINE_LORA_WD = [0.0]


def build_energy_commands() -> List[List[str]]:
    """Build Energy grid search commands for ImageNet."""
    commands: List[List[str]] = []
    
    for model, init_mode, k, topk, sigma_lr, sigma_wd, warmup_ratio in itertools.product(
        ENERGY_MODELS,
        ENERGY_INITIALIZE_SIGMA,
        ENERGY_K,
        ENERGY_SVD_KEEP_TOPK,
        ENERGY_SIGMA_LR,
        ENERGY_SIGMA_WD,
        ENERGY_WARMUP_RATIO,
    ):
        _, results_json = _expected_energy_paths(
            model=model,
            init_mode=init_mode,
            sigma_lr=sigma_lr,
            topk=topk,
            sigma_wd=sigma_wd,
            k=int(k),
            warmup_ratio=warmup_ratio,
        )
        if _path_exists(results_json):
            print(
                f"[skip] energy {model} ImageNet (init={init_mode}, k={k}, topk={topk}, "
                f"lr={sigma_lr}, wd={sigma_wd}, warmup={warmup_ratio}) -> {results_json}",
                flush=True,
            )
            continue
        
        cmd = [
            sys.executable,
            "imagenet_energy_train.py",
            "--model", model,
            "--initialize_sigma", init_mode,
            "--k", str(k),
            "--svd_keep_topk", str(topk),
            "--sigma_lr", f"{sigma_lr:.6g}",
            "--sigma_wd", f"{sigma_wd:.6g}",
            "--warmup_ratio", f"{warmup_ratio:.6g}",
        ]
        commands.append(cmd)
    
    return commands


def build_atlas_commands() -> List[List[str]]:
    """Build Atlas commands for ImageNet."""
    commands: List[List[str]] = []
    
    for model, k, lr, num_basis in itertools.product(
        ATLAS_MODELS, ATLAS_K, ATLAS_LR, ATLAS_NUM_BASIS
    ):
        _, results_json = _expected_atlas_paths(
            model=model,
            lr=float(lr),
            k=int(k),
            num_basis=int(num_basis),
        )
        # if _path_exists(results_json):
        #     print(
        #         f"[skip] atlas {model} ImageNet (k={k}, lr={lr}, num_basis={num_basis}) -> {results_json}",
        #         flush=True,
        #     )
        #     continue
        
        cmd = [
            sys.executable,
            "imagenet_atlas_train.py",
            "--model", model,
            "--k", str(k),
            "--lr", f"{lr:.6g}",
            "--num_tasks", str(num_basis),
            "--config_file", "config/config_reverse.yaml",
        ]
        commands.append(cmd)
    
    return commands


def build_baseline_commands() -> List[List[str]]:
    """Build baseline training commands for ImageNet."""
    commands: List[List[str]] = []
    
    for model, method, k in itertools.product(
        BASELINE_MODELS, BASELINE_METHODS, BASELINE_K
    ):
        # Build hyperparameters based on method
        if method == 'linear_probe':
            for lp_lr, lp_epochs, lp_wd in itertools.product(
                BASELINE_LP_LR, BASELINE_LP_EPOCHS, BASELINE_LP_WD
            ):
                hparams = {
                    'lp_lr': lp_lr,
                    'lp_epochs': lp_epochs,
                    'lp_wd': lp_wd,
                }
                results_json = _expected_baseline_paths(
                    model=model, method=method, k=k, **hparams
                )
                # if _path_exists(results_json):
                #     print(
                #         f"[skip] baseline {model} ImageNet (method={method}, k={k}) -> {results_json}",
                #         flush=True,
                #     )
                #     continue
                
                cmd = [
                    sys.executable,
                    "imagenet_baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--lp_lr", f"{lp_lr:.6g}",
                    "--lp_epochs", str(lp_epochs),
                    "--lp_wd", f"{lp_wd:.6g}",
                ]
                commands.append(cmd)
        
        elif method in ['tip_adapter', 'lp++']:
            for adapter_wd in BASELINE_ADAPTER_WD:
                hparams = {'adapter_wd': adapter_wd}
                results_json = _expected_baseline_paths(
                    model=model, method=method, k=k, **hparams
                )
                # if _path_exists(results_json):
                #     print(
                #         f"[skip] baseline {model} ImageNet (method={method}, k={k}, wd={adapter_wd}) -> {results_json}",
                #         flush=True,
                #     )
                #     continue
                
                cmd = [
                    sys.executable,
                    "imagenet_baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--adapter_wd", f"{adapter_wd:.6g}",
                ]
                commands.append(cmd)
        
        elif method == 'lora':
            for lora_r, lora_alpha, lora_lr, lora_epochs, lora_wd in itertools.product(
                BASELINE_LORA_R, BASELINE_LORA_ALPHA, BASELINE_LORA_LR,
                BASELINE_LORA_EPOCHS, BASELINE_LORA_WD
            ):
                hparams = {
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_lr': lora_lr,
                    'lora_epochs': lora_epochs,
                    'lora_wd': lora_wd,
                }
                results_json = _expected_baseline_paths(
                    model=model, method=method, k=k, **hparams
                )
                # if _path_exists(results_json):
                #     print(
                #         f"[skip] baseline {model} ImageNet (method={method}, k={k}) -> {results_json}",
                #         flush=True,
                #     )
                #     continue
                
                cmd = [
                    sys.executable,
                    "imagenet_baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--lora_r", str(lora_r),
                    "--lora_alpha", f"{lora_alpha:.6g}",
                    "--lora_lr", f"{lora_lr:.6g}",
                    "--lora_epochs", str(lora_epochs),
                    "--lora_wd", f"{lora_wd:.6g}",
                ]
                commands.append(cmd)
    
    return commands


def run_commands_in_parallel(
    commands: Sequence[Sequence[str]],
    gpu_ids: Sequence[int],
    per_gpu: int = 1,
    dry_run: bool = False,
) -> None:
    """Run commands in parallel across multiple GPUs."""
    if dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        print(f"Total commands: {len(commands)}")
        return

    queue_lock = threading.Lock()
    command_iter = iter(commands)
    failures: List[Sequence[str]] = []

    def worker(gpu_id: int, slot: int) -> None:
        nonlocal command_iter
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        while True:
            with queue_lock:
                try:
                    cmd = next(command_iter)
                except StopIteration:
                    return
                display = " ".join(cmd)
                print(f"[GPU {gpu_id} slot {slot}] starting: {display}", flush=True)

            start = time.time()
            proc = subprocess.Popen(cmd, env=env)
            ret = proc.wait()
            elapsed = time.time() - start

            status = "OK" if ret == 0 else f"FAIL ({ret})"
            print(
                f"[GPU {gpu_id} slot {slot}] finished ({status}) in {elapsed / 60:.2f} min: {' '.join(cmd)}",
                flush=True,
            )

            if ret != 0:
                with queue_lock:
                    failures.append(cmd)

    per_gpu = max(1, int(per_gpu))
    threads = []
    for gpu in gpu_ids:
        for slot in range(per_gpu):
            thread = threading.Thread(target=worker, args=(gpu, slot), daemon=True)
            threads.append(thread)
    if not threads:
        raise ValueError("No GPU workers configured. Check --gpu-ids and --per-gpu arguments.")
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        print("\nSome commands failed:", flush=True)
        for cmd in failures:
            print("  " + " ".join(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-energy", action="store_true", help="Skip energy sweeps")
    parser.add_argument("--skip-atlas", action="store_true", help="Skip atlas sweeps")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline sweeps")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of commands (after shuffling)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle command order before scheduling",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--per-gpu",
        type=int,
        default=4,
        help="Number of commands to run concurrently on each GPU",
    )
    args = parser.parse_args()

    commands: List[List[str]] = []
    
    if not args.skip_energy:
        energy_cmds = build_energy_commands()
        commands.extend(energy_cmds)
        print(f"Added {len(energy_cmds)} Energy commands for ImageNet")
    
    if not args.skip_atlas:
        atlas_cmds = build_atlas_commands()
        commands.extend(atlas_cmds)
        print(f"Added {len(atlas_cmds)} Atlas commands for ImageNet")
    
    if not args.skip_baselines:
        baseline_cmds = build_baseline_commands()
        commands.extend(baseline_cmds)
        print(f"Added {len(baseline_cmds)} Baseline commands for ImageNet")

    if not commands:
        print("Nothing to run (all sweeps skipped).", file=sys.stderr)
        return

    if args.shuffle:
        random.shuffle(commands)

    if args.limit is not None:
        commands = commands[: args.limit]

    gpu_ids = GPU_IDS
    if args.gpu_ids is not None:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip() != ""]
        except ValueError as exc:
            raise SystemExit(f"Invalid --gpu-ids value: {args.gpu_ids}") from exc
        if not gpu_ids:
            raise SystemExit("--gpu-ids produced an empty list")

    commands_per_gpu = max(1, args.per_gpu)

    total_workers = len(gpu_ids) * commands_per_gpu
    print(
        f"Prepared {len(commands)} ImageNet commands across {len(gpu_ids)} GPUs with "
        f"{commands_per_gpu} slots each ({total_workers} total workers)."
    )

    run_commands_in_parallel(commands, gpu_ids, per_gpu=commands_per_gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

