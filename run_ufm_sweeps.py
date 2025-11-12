#!/usr/bin/env python3
"""
Utility to launch UFM (Unsupervised FixMatch) experiments for test-time adaptation.

This script runs ufm_atlas.py and ufm_energy.py across multiple datasets in parallel.

Usage:
    python run_ufm_sweeps.py
        Launch the full sweep.

    python run_ufm_sweeps.py --dry-run
        Print the commands that would run, without executing them.

    python run_ufm_sweeps.py --skip-ufm-energy
        Only run UFM-Atlas sweeps.

    python run_ufm_sweeps.py --limit 10
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
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_reverse.yaml")
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
    default_root = os.path.join(".", "models", "checkpoints_tta")
    raw_root = CFG.get("tta_model_location", default_root) if isinstance(CFG, dict) else default_root
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


def _path_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(os.path.abspath(path))


def _datasets_all_from_config() -> Sequence[str]:
    candidates = CFG.get("DATASETS_ALL") if isinstance(CFG, dict) else None
    if isinstance(candidates, (list, tuple)):
        return list(candidates)
    return list(DATASETS_ALL.keys())


def _ufm_atlas_config_tag(num_basis: int, lr: float) -> str:
    return "ufm_atlas_{}_{}".format(
        _sanitize_value(max(int(num_basis), 0)),
        _sanitize_value(lr),
    )


def _ufm_energy_config_tag(init_mode: str, sigma_lr: float, topk: int, sigma_wd: float, warmup_ratio: float) -> str:
    datasets_all = _datasets_all_from_config()
    candidate_int = len(datasets_all)
    num_tasks_minus_one = max(candidate_int - 1, 0)
    init_value = (init_mode or "average").strip().lower()
    return "ufm_energy_{}_{}_{}_{}_{}_{}".format(
        _sanitize_value(num_tasks_minus_one),
        _sanitize_value(sigma_lr),
        _sanitize_value(topk),
        _sanitize_value(init_value),
        _sanitize_value(warmup_ratio),
        _sanitize_value(sigma_wd),
    )


def _shot_folder(k: int) -> str:
    return f"{k}shots" if k > 0 else "fullshots"


def _expected_ufm_atlas_paths(
    model: str,
    dataset: str,
    lr: float,
    k: int,
) -> tuple[str, str]:
    lr = float(lr)
    k = int(k)
    datasets_all = _datasets_all_from_config()
    num_basis = max(len(datasets_all) - 1, 0)
    config_tag = _ufm_atlas_config_tag(num_basis, lr)
    dataset_dir = f"{dataset}Val"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    atlas_pt = os.path.join(base_dir, "ufm_atlas.pt")
    results_json = os.path.join(base_dir, "ufm_atlas_results_none.json")
    return atlas_pt, results_json


def _expected_ufm_energy_paths(
    model: str,
    dataset: str,
    init_mode: str,
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
    config_tag = _ufm_energy_config_tag(init_mode, sigma_lr, topk, sigma_wd, warmup_ratio)
    dataset_dir = f"{dataset}Val"
    base_dir = os.path.join(MODEL_ROOT, model, dataset_dir, config_tag, _shot_folder(k))
    energy_pt = os.path.join(base_dir, "ufm_energy.pt")
    results_json = os.path.join(base_dir, "ufm_energy_results_none.json")
    return energy_pt, results_json


# Dataset configuration
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
    "Food101": 20,
    "Flowers102": 20,
    # "FER2013": 20,
    "PCAM": 20,
    "OxfordIIITPet": 20,
    "RenderedSST2": 20,
    "EMNIST": 20,
    "FashionMNIST": 20,
    # "KMNIST": 20,
    "FGVCAircraft": 20,
    "CUB200": 20,
    "Country211": 20,
}


GPU_IDS = [0, 1]  # Default GPU IDs, can be overridden via CLI

# UFM-Atlas configurations
UFM_ATLAS_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
UFM_ATLAS_K = [0]  # fullshot only for test-time adaptation
UFM_ATLAS_LR = [1e-3, 5e-3, 1e-2, 1e-4]
UFM_ATLAS_WD = [0.1]
UFM_ATLAS_EPOCHS = [None]  # Will auto-set per dataset

# UFM-Energy configurations
UFM_ENERGY_MODELS = ["ViT-B-16", "ViT-L-14", "ViT-B-32"]
UFM_ENERGY_K = [0]  # fullshot only for test-time adaptation
UFM_ENERGY_INITIALIZE_SIGMA = ["average"]
UFM_ENERGY_SVD_KEEP_TOPK = [10, 12, 14, 16]
UFM_ENERGY_SIGMA_LR = [1e-3, 5e-3, 1e-4, 1e-2]
UFM_ENERGY_SIGMA_WD = [0.0]
UFM_ENERGY_WARMUP_RATIO = [0.1]


def build_ufm_atlas_commands(datasets: Sequence[str]) -> List[List[str]]:
    """Build UFM-Atlas grid search commands."""
    commands: List[List[str]] = []
    
    for model, dataset, k, lr, wd in itertools.product(
        UFM_ATLAS_MODELS,
        datasets,
        UFM_ATLAS_K,
        UFM_ATLAS_LR,
        UFM_ATLAS_WD,
    ):
        _, results_json = _expected_ufm_atlas_paths(
            model=model,
            dataset=dataset,
            lr=lr,
            k=int(k),
        )
        if _path_exists(results_json):
            print(
                f"[skip] ufm-atlas {model} {dataset} (k={k}, lr={lr}, wd={wd}) -> {results_json}",
                flush=True,
            )
            continue
        
        cmd = [
            sys.executable,
            "ufm_atlas.py",
            "--model",
            model,
            "--test_dataset",
            dataset,
            "--k",
            str(k),
            "--lr",
            f"{lr:.6g}",
            "--wd",
            f"{wd:.6g}",
        ]
        commands.append(cmd)
    
    return commands


def build_ufm_energy_commands(datasets: Sequence[str]) -> List[List[str]]:
    """Build UFM-Energy grid search commands."""
    commands: List[List[str]] = []
    
    for model, init_mode, dataset, k, topk, sigma_lr, sigma_wd, warmup_ratio in itertools.product(
        UFM_ENERGY_MODELS,
        UFM_ENERGY_INITIALIZE_SIGMA,
        datasets,
        UFM_ENERGY_K,
        UFM_ENERGY_SVD_KEEP_TOPK,
        UFM_ENERGY_SIGMA_LR,
        UFM_ENERGY_SIGMA_WD,
        UFM_ENERGY_WARMUP_RATIO,
    ):
        _, results_json = _expected_ufm_energy_paths(
            model=model,
            dataset=dataset,
            init_mode=init_mode,
            sigma_lr=sigma_lr,
            topk=topk,
            sigma_wd=sigma_wd,
            k=int(k),
            warmup_ratio=warmup_ratio,
        )
        if _path_exists(results_json):
            print(
                f"[skip] ufm-energy {model} {dataset} (init={init_mode}, k={k}, topk={topk}, lr={sigma_lr}, wd={sigma_wd}, warmup={warmup_ratio}) -> {results_json}",
                flush=True,
            )
            continue
        
        cmd = [
            sys.executable,
            "ufm_energy.py",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ufm-energy", action="store_true", help="Skip UFM-Energy sweeps")
    parser.add_argument("--skip-ufm-atlas", action="store_true", help="Skip UFM-Atlas sweeps")
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

    datasets = sorted(DATASETS_ALL.keys())

    commands: List[List[str]] = []
    if not args.skip_ufm_atlas:
        ufm_atlas_cmds = build_ufm_atlas_commands(datasets)
        commands.extend(ufm_atlas_cmds)
        print(f"Added {len(ufm_atlas_cmds)} UFM-Atlas commands")
    
    if not args.skip_ufm_energy:
        ufm_energy_cmds = build_ufm_energy_commands(datasets)
        commands.extend(ufm_energy_cmds)
        print(f"Added {len(ufm_energy_cmds)} UFM-Energy commands")

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

