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


REMOTE_SENSING_DATASETS = {
    "AID": 10,
    "CLRS": 10,
    "EuroSAT_RGB": 12,
    "MLRSNet": 15,
    "NWPU-RESISC45": 15,
    "Optimal-31": 50,
    "PatternNet": 20,
    "RS_C11": 60,
    "RSD46-WHU": 20,
    "RSI-CB128": 15,
    "RSSCN7": 80,
    "SAT-4": 5,
    "SIRI-WHU": 100,
    "UC_Merced": 100,
    "WHU-RS19": 150,
}


GPU_IDS = list(range(4))  # Default GPU IDs, can be overridden via CLI

ENERGY_MODELS = ["ViT-B-32"]
ENERGY_INITIALIZE_SIGMA = ["tsvm", "average"]
ENERGY_ADAPTERS = ["none", "lp++", "tip"]
ENERGY_K = [16]
ENERGY_SVD_KEEP_TOPK = [5]
ENERGY_SIGMA_LR = [1e-3]

ATLAS_MODELS = ["ViT-B-32"]
ATLAS_ADAPTERS = ["none", "lp++", "tip"]
ATLAS_K = [16]


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
    per_gpu: int = 1,
    dry_run: bool = False,
) -> None:
    if dry_run:
        for cmd in commands:
            print(" ".join(cmd))
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
        default=4,
        help="Number of commands to run concurrently on each GPU",
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
