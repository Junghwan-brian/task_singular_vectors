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
import os
import random
import subprocess
import sys
import threading
import time
from typing import Iterable, List, Sequence

from src.datasets.remote_sensing import REMOTE_SENSING_DATASETS

GPU_IDS = list(range(8))  # Adjust if you have a different number of GPUs

ENERGY_MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-16"]
ENERGY_INITIALIZE_SIGMA = ["tsvm", "average"]
ENERGY_ADAPTERS = ["none", "tip", "lp++"]
ENERGY_K = [0, 1, 2, 4, 8, 16]
ENERGY_SVD_KEEP_TOPK = [3, 4, 5, 6]
ENERGY_SIGMA_LR = [1e-2, 1e-3, 1e-4, 1e-1]

ATLAS_MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-16"]
ATLAS_ADAPTERS = ["none", "tip", "lp++"]
ATLAS_K = [0, 1, 2, 4, 8, 16]


def build_energy_commands(datasets: Sequence[str]) -> List[List[str]]:
    commands: List[List[str]] = []
    for model, init_mode, adapter, dataset, k, topk, sigma_lr in itertools.product(
        ENERGY_MODELS,
        ENERGY_INITIALIZE_SIGMA,
        ENERGY_ADAPTERS,
        datasets,
        ENERGY_K,
        ENERGY_SVD_KEEP_TOPK,
        ENERGY_SIGMA_LR,
    ):
        cmd = [
            sys.executable,
            "energy_train_remote_sensing.py",
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
            "--adapter",
            adapter,
        ]
        commands.append(cmd)
    return commands


def build_atlas_commands(datasets: Sequence[str]) -> List[List[str]]:
    commands: List[List[str]] = []
    for model, adapter, dataset, k in itertools.product(
        ATLAS_MODELS, ATLAS_ADAPTERS, datasets, ATLAS_K
    ):
        cmd = [
            sys.executable,
            "atlas_remote_sensing.py",
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


def run_commands_in_parallel(
    commands: Sequence[Sequence[str]],
    gpu_ids: Sequence[int],
    dry_run: bool = False,
) -> None:
    if dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return

    queue_lock = threading.Lock()
    command_iter = iter(commands)
    failures: List[Sequence[str]] = []

    def worker(gpu_id: int) -> None:
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
                print(f"[GPU {gpu_id}] starting: {display}", flush=True)

            start = time.time()
            proc = subprocess.Popen(cmd, env=env)
            ret = proc.wait()
            elapsed = time.time() - start

            status = "OK" if ret == 0 else f"FAIL ({ret})"
            print(
                f"[GPU {gpu_id}] finished ({status}) in {elapsed / 60:.2f} min: {' '.join(cmd)}",
                flush=True,
            )

            if ret != 0:
                with queue_lock:
                    failures.append(cmd)

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpu_ids
    ]
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
    args = parser.parse_args()

    datasets = sorted(REMOTE_SENSING_DATASETS.keys())

    commands: List[List[str]] = []
    if not args.skip_energy:
        commands.extend(build_energy_commands(datasets))
    if not args.skip_atlas:
        commands.extend(build_atlas_commands(datasets))

    if not commands:
        print("Nothing to run (both sweeps skipped).", file=sys.stderr)
        return

    if args.shuffle:
        random.shuffle(commands)

    if args.limit is not None:
        commands = commands[: args.limit]

    print(f"Prepared {len(commands)} commands across {len(GPU_IDS)} GPUs.")

    run_commands_in_parallel(commands, GPU_IDS, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
