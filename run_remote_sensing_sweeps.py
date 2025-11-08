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


def _load_remote_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_remote_sensing.yaml")
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


REMOTE_CFG = _load_remote_config()


def _resolve_model_root() -> str:
    default_root = os.path.join(".", "models", "checkpoints_remote_sensing")
    raw_root = REMOTE_CFG.get("model_location", default_root) if isinstance(REMOTE_CFG, dict) else default_root
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
    candidates = REMOTE_CFG.get("DATASETS_ALL") if isinstance(REMOTE_CFG, dict) else None
    if isinstance(candidates, (list, tuple)):
        return list(candidates)
    return list(REMOTE_SENSING_DATASETS.keys())


def _atlas_default_lr() -> float:
    if isinstance(REMOTE_CFG, dict):
        value = REMOTE_CFG.get("lr")
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
    "SAT-6": 10,
    "SIRI-WHU": 100,
    "UC_Merced": 100,
    "WHU-RS19": 150,
}

GPU_IDS = [0,1,2,3,4,5,6,7]  # Default GPU IDs, can be overridden via CLI
ENERGY_MODELS = ["ViT-B-16"]
ENERGY_INITIALIZE_SIGMA = ["average", "sum"]
ENERGY_ADAPTERS = ["none"]
ENERGY_K = [1,2,4,8,16]
ENERGY_SVD_KEEP_TOPK = [10, 12]
ENERGY_SIGMA_LR = [1e-3]
ENERGY_SIGMA_WD = [0.05, 0.1]
ENERGY_WARMUP_RATIO = [0.1]

ATLAS_MODELS = ["ViT-B-16"]
ATLAS_ADAPTERS = ["none", "lp++", "tip"]
ATLAS_K = [1,2,4,8,16]

def build_energy_commands(datasets: Sequence[str]) -> List[List[str]]:
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
