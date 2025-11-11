#!/usr/bin/env python3
"""
Utility to launch all requested remote sensing sweeps across the energy and atlas scripts.

By default this enumerates the full Cartesian product of the hyper-parameter grids supplied
in the specification and schedules the resulting commands over eight GPUs
(`CUDA_VISIBLE_DEVICES=0-7`) in parallel.

⚠️ The combined grid is very large (tens of thousands of runs). Adjust the lists below
before launching to match the experiments you actually want to execute.

Usage:
    python run_remote_sensing_sweeps.py
        Launch the full sweep (be careful – this is huge).

    python run_remote_sensing_sweeps.py --dry-run
        Print the commands that would run, without executing them.

    python run_remote_sensing_sweeps.py --skip-energy
        Only run atlas sweeps (honours other flags as well).

    python run_remote_sensing_sweeps.py --limit 10
        Run only the first 10 commands (after optional shuffling).
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
from typing import Iterable, List, Sequence


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


CFG = _load_config()


def _resolve_model_root() -> str:
    default_root = os.path.join(".", "models", "checkpoints_remote_sensing")
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


def _datasets_all_from_config() -> Sequence[str]:
    candidates = CFG.get("DATASETS_ALL") if isinstance(CFG, dict) else None
    if isinstance(candidates, (list, tuple)):
        return list(candidates)
    return list(DATASETS_ALL.keys())


def _atlas_default_lr() -> float:
    if isinstance(CFG, dict):
        value = CFG.get("lr")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 1e-1


def _energy_config_tag(init_mode: str, sigma_lr: float, topk: int, sigma_wd: float, warmup_ratio: float) -> str:
    datasets_all = _datasets_all_from_config()
    candidate_int = len(datasets_all)
    num_tasks_minus_one = max(candidate_int - 1, 0)
    init_value = (init_mode or "average").strip().lower()
    return "energy_{}_{}_{}_{}_{}_{}".format(
        _sanitize_value(num_tasks_minus_one),
        _sanitize_value(sigma_lr),
        _sanitize_value(topk),
        _sanitize_value(init_value),
        _sanitize_value(warmup_ratio),
        _sanitize_value(sigma_wd),
    )


def _atlas_config_tag(lr: float) -> str:
    datasets_all = _datasets_all_from_config()
    num_basis = max(len(datasets_all) - 1, 0)
    return "atlas_{}_{}".format(
        _sanitize_value(max(int(num_basis), 0)),
        _sanitize_value(lr),
    )


def _shot_folder(k: int) -> str:
    return f"{k}shots" if k > 0 else "fullshots"


def _expected_energy_paths(
    model: str,
    dataset: str,
    init_mode: str,
    adapter: str,
    sigma_lr: float,
    topk: int,
    sigma_wd: float,
    k: int,
    warmup_ratio: float,
) -> tuple[str, str]:
    sigma_lr = float(sigma_lr)
    topk = int(topk)
    sigma_wd = float(sigma_wd)
    k = int(k)
    warmup_ratio = float(warmup_ratio)
    config_tag = _energy_config_tag(init_mode, sigma_lr, topk, sigma_wd, warmup_ratio)
    adapter_tag = _adapter_tag(adapter)
    dataset_dir = f"{dataset}Val"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    energy_pt = os.path.join(base_dir, "energy.pt")
    results_json = os.path.join(base_dir, f"energy_results_{adapter_tag}.json")
    return energy_pt, results_json


def _expected_atlas_paths(
    model: str,
    dataset: str,
    adapter: str,
    lr: float,
    k: int,
) -> tuple[str, str]:
    lr = float(lr)
    k = int(k)
    config_tag = _atlas_config_tag(lr)
    adapter_tag = _adapter_tag(adapter)
    dataset_dir = f"{dataset}Val"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    atlas_pt = os.path.join(base_dir, "atlas.pt")
    results_json = os.path.join(base_dir, f"atlas_results_{adapter_tag}.json")
    return atlas_pt, results_json

DATASETS_ALL = {
    # "Cars": 20,
    "DTD": 20,
    # "EuroSAT": 20,
    "GTSRB": 20,
    "MNIST": 20,
    # "RESISC45": 20,
    # "SUN397": 20,
    "SVHN": 20,
    "CIFAR10": 20,
    "CIFAR100": 20,
    "STL10": 20,
    "Food101":20,
    "Flowers102": 20,
    # "FER2013": 20,
    "PCAM":20,
    "OxfordIIITPet": 20,
    "RenderedSST2": 20,
    "EMNIST":20,
    "FashionMNIST":20,
    # "KMNIST":20,
    "FGVCAircraft": 20,
    "CUB200": 20,
    "Country211": 20,
}


GPU_IDS = [0,1]  # Default GPU IDs, can be overridden via CLI
ENERGY_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
ENERGY_INITIALIZE_SIGMA = ["average"]
ENERGY_ADAPTERS = ["none"]
ENERGY_K = [1,2,4,8,16]
ENERGY_SVD_KEEP_TOPK = [14, 16]
ENERGY_SIGMA_LR = [1e-3, 5e-3]
ENERGY_SIGMA_WD = [0.0]
ENERGY_WARMUP_RATIO = [0.1]

ATLAS_MODELS = ["ViT-B-16"]
ATLAS_ADAPTERS = ["none", "lp++", "tip"]
ATLAS_K = [16]

# Baseline configurations
BASELINE_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
BASELINE_METHODS = ["lp++"]
BASELINE_K = [1,2,4,8,16]
BASELINE_LP_LR = [1e-3]
BASELINE_LP_EPOCHS = [20]
BASELINE_LP_WD = [0.0]
BASELINE_ADAPTER_WD = [0.0, 0.1, 0.01]
BASELINE_LORA_R = [8]
BASELINE_LORA_ALPHA = [16.0]
BASELINE_LORA_LR = [1e-4]
BASELINE_LORA_EPOCHS = [20]
BASELINE_LORA_WD = [0.0]

def build_energy_commands(datasets: Sequence[str]) -> List[List[str]]:
    """Build Energy grid search commands."""
    commands: List[List[str]] = []
    
    for model, init_mode, adapter, dataset, k, topk, sigma_lr, sigma_wd, warmup_ratio in itertools.product(
        ENERGY_MODELS,
        ENERGY_INITIALIZE_SIGMA,
        ENERGY_ADAPTERS,
        datasets,
        ENERGY_K,
        ENERGY_SVD_KEEP_TOPK,
        ENERGY_SIGMA_LR,
        ENERGY_SIGMA_WD,
        ENERGY_WARMUP_RATIO,
    ):
        _, results_json = _expected_energy_paths(
            model=model,
            dataset=dataset,
            init_mode=init_mode,
            adapter=adapter,
            sigma_lr=sigma_lr,
            topk=topk,
            sigma_wd=sigma_wd,
            k=int(k),
            warmup_ratio=warmup_ratio,
        )
        if _path_exists(results_json):
            print(
                f"[skip] energy {model} {dataset} (init={init_mode}, adapter={adapter}, k={k}, wd={sigma_wd}, warmup={warmup_ratio}) -> {results_json}",
                flush=True,
            )
            continue
        cmd = [
            sys.executable,
            "energy_train_reverse.py",
            "--model",
            model,
            "--initialize_sigma",
            init_mode,
            "--k",
            str(k),
            "--test_dataset",
            dataset,
            "--svd_keep_topk",
            str(topk),
            "--sigma_lr",
            f"{sigma_lr:.6g}",
            "--sigma_wd",
            f"{sigma_wd:.6g}",
            "--warmup_ratio",
            f"{warmup_ratio:.6g}",
            "--adapter",
            adapter,
        ]
        commands.append(cmd)
    
    return commands


def build_atlas_commands(datasets: Sequence[str]) -> List[List[str]]:
    commands: List[List[str]] = []
    lr_default = _atlas_default_lr()
    for model, adapter, dataset, k in itertools.product(
        ATLAS_MODELS, ATLAS_ADAPTERS, datasets, ATLAS_K
    ):
        _, results_json = _expected_atlas_paths(
            model=model,
            dataset=dataset,
            adapter=adapter,
            lr=float(lr_default),
            k=int(k),
        )
        if _path_exists(results_json):
            print(
                f"[skip] atlas {model} {dataset} (adapter={adapter}, k={k}) -> {results_json}",
                flush=True,
            )
            continue
        cmd = [
            sys.executable,
            "atlas_reverse.py",
            "--model",
            model,
            "--k",
            str(k),
            "--test_dataset",
            dataset,
            "--adapter",
            adapter,
        ]
        commands.append(cmd)
    return commands


def _baseline_config_tag(method: str, **kwargs) -> str:
    """Build config tag for baseline method."""
    if method == 'linear_probe':
        lr = _sanitize_value(kwargs.get('lp_lr', 1e-3))
        epochs = _sanitize_value(kwargs.get('lp_epochs', 20))
        wd = _sanitize_value(kwargs.get('lp_wd', 0.0))
        return f"baseline_lp_{lr}_{epochs}_{wd}"
    elif method == 'tip_adapter':
        wd = _sanitize_value(kwargs.get('adapter_wd', 0.0))
        return f"baseline_tip_{wd}"
    elif method == 'lp++':
        wd = _sanitize_value(kwargs.get('adapter_wd', 0.0))
        return f"baseline_lpp_{wd}"
    elif method == 'lora':
        r = _sanitize_value(kwargs.get('lora_r', 8))
        alpha = _sanitize_value(kwargs.get('lora_alpha', 16.0))
        lr = _sanitize_value(kwargs.get('lora_lr', 1e-4))
        epochs = _sanitize_value(kwargs.get('lora_epochs', 20))
        wd = _sanitize_value(kwargs.get('lora_wd', 0.0))
        return f"baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}"
    else:
        return f"baseline_{method}"


def _expected_baseline_paths(
    model: str,
    dataset: str,
    method: str,
    k: int,
    **kwargs
) -> str:
    """Return expected baseline results path."""
    config_tag = _baseline_config_tag(method, **kwargs)
    dataset_dir = f"{dataset}Val"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    results_json = os.path.join(base_dir, "baseline_results_none.json")
    return results_json


def build_baseline_commands(datasets: Sequence[str]) -> List[List[str]]:
    """Build baseline training commands for general datasets."""
    commands: List[List[str]] = []
    
    for model, method, dataset, k in itertools.product(
        BASELINE_MODELS, BASELINE_METHODS, datasets, BASELINE_K
    ):
        # Build hyperparameters based on method
        hparams = {}
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
                    model=model, dataset=dataset, method=method, k=k, **hparams
                )
                if _path_exists(results_json):
                    print(
                        f"[skip] baseline {model} {dataset} (method={method}, k={k}) -> {results_json}",
                        flush=True,
                    )
                    continue
                
                cmd = [
                    sys.executable,
                    "baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--target_dataset", dataset,
                    "--lp_lr", f"{lp_lr:.6g}",
                    "--lp_epochs", str(lp_epochs),
                    "--lp_wd", f"{lp_wd:.6g}",
                ]
                commands.append(cmd)
        
        elif method in ['tip_adapter', 'lp++']:
            for adapter_wd in BASELINE_ADAPTER_WD:
                hparams = {'adapter_wd': adapter_wd}
                results_json = _expected_baseline_paths(
                    model=model, dataset=dataset, method=method, k=k, **hparams
                )
                # if _path_exists(results_json):
                #     print(
                #         f"[skip] baseline {model} {dataset} (method={method}, k={k}) -> {results_json}",
                #         flush=True,
                #     )
                #     continue
                
                cmd = [
                    sys.executable,
                    "baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--target_dataset", dataset,
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
                    model=model, dataset=dataset, method=method, k=k, **hparams
                )
                if _path_exists(results_json):
                    print(
                        f"[skip] baseline {model} {dataset} (method={method}, k={k}) -> {results_json}",
                        flush=True,
                    )
                    continue
                
                cmd = [
                    sys.executable,
                    "baselines_train.py",
                    "--model", model,
                    "--baseline_method", method,
                    "--k", str(k),
                    "--target_dataset", dataset,
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


def build_energy_adapter_commands(datasets: Sequence[str], best_config_file: str) -> List[List[str]]:
    """Build Energy + Adapter (TIP/LP++) commands using best configs per dataset."""
    commands: List[List[str]] = []
    
    if not os.path.exists(best_config_file):
        print(f"[warning] Best config file not found: {best_config_file}", flush=True)
        return commands
    
    try:
        with open(best_config_file, 'r') as f:
            best_configs = json.load(f)
    except Exception as e:
        print(f"[warning] Failed to load best config file: {e}", flush=True)
        return commands
    
    adapters = ['tip', 'lp++']
    
    for dataset in datasets:
        if dataset not in best_configs:
            continue
        
        for model in best_configs[dataset]:
            for shot in best_configs[dataset][model]:
                config_data = best_configs[dataset][model][shot]
                config_tag = config_data.get('config_tag', '')
                
                if not config_tag:
                    continue
                
                # Parse config tag to extract hyperparameters
                # Format: energy_{num_tasks}_{lr}_{topk}_{init}_{warmup}_{wd}
                parts = config_tag.split('_')
                if len(parts) < 7 or parts[0] != 'energy':
                    continue
                
                sigma_lr = parts[2].replace('p', '.')
                topk = parts[3]
                init_mode = parts[4]
                warmup_ratio = parts[5].replace('p', '.')
                sigma_wd = parts[6].replace('p', '.')
                
                k = int(shot.replace('shots', ''))
                
                for adapter in adapters:
                    adapter_tag = _adapter_tag(adapter)
                    dataset_dir = f"{dataset}Val"
                    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
                    results_json = os.path.join(base_dir, f"energy_results_{adapter_tag}.json")
                    
                    if _path_exists(results_json):
                        print(
                            f"[skip] energy+{adapter} {model} {dataset} (k={k}) -> {results_json}",
                            flush=True,
                        )
                        continue
                    
                    cmd = [
                        sys.executable,
                        "energy_train_reverse.py",
                        "--model", model,
                        "--initialize_sigma", init_mode,
                        "--k", str(k),
                        "--test_dataset", dataset,
                        "--svd_keep_topk", topk,
                        "--sigma_lr", sigma_lr,
                        "--sigma_wd", sigma_wd,
                        "--warmup_ratio", warmup_ratio,
                        "--adapter", adapter,
                    ]
                    commands.append(cmd)
    
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-energy", action="store_true", help="Skip energy sweeps")
    parser.add_argument("--skip-atlas", action="store_true", help="Skip atlas sweeps")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline sweeps")
    parser.add_argument("--skip-energy-adapters", action="store_true", help="Skip Energy + Adapter sweeps")
    parser.add_argument(
        "--best_config_file",
        type=str,
        default="./results/best_energy_configs_per_dataset.json",
        help="Path to best Energy configs JSON file for adapter training",
    )
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
        help="Comma-separated list of GPU IDs to use (default: current GPU_IDS setting)",
    )
    parser.add_argument(
        "--per-gpu",
        type=int,
        default=8,
        help="Number of commands to run concurrently on each GPU",
    )
    args = parser.parse_args()

    # datasets = sorted(REMOTE_SENSING_DATASETS.keys())
    datasets = sorted(DATASETS_ALL.keys())

    commands: List[List[str]] = []
    if not args.skip_energy:
        commands.extend(build_energy_commands(datasets))
    if not args.skip_atlas:
        commands.extend(build_atlas_commands(datasets))
    if not args.skip_baselines:
        commands.extend(build_baseline_commands(datasets))
    if not args.skip_energy_adapters:
        energy_adapter_cmds = build_energy_adapter_commands(datasets, args.best_config_file)
        if energy_adapter_cmds:
            commands.extend(energy_adapter_cmds)
            print(f"Added {len(energy_adapter_cmds)} Energy+Adapter commands")

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
        f"Prepared {len(commands)} commands across {len(gpu_ids)} GPUs with {commands_per_gpu} slots each "
        f"({total_workers} total workers)."
    )

    run_commands_in_parallel(commands, gpu_ids, per_gpu=commands_per_gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
