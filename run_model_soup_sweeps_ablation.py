from __future__ import annotations

import argparse
import itertools
import os
import random
import subprocess
import sys
import threading
import time
from typing import List, Sequence


def _load_config(best_config_file: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), best_config_file)
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


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


def _datasets_all_from_config(cfg: dict, fallback: Sequence[str]) -> Sequence[str]:
    candidates = cfg.get("DATASETS_ALL") if isinstance(cfg, dict) else None
    if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
        return list(candidates)
    return list(fallback)


def _shot_folder(k: int) -> str:
    return f"{k}shots" if int(k) > 0 else "fullshots"


def _soupft_config_tag(num_soup_models: int, ft_lr: float, ft_epochs: int, ft_wd: float, warmup_ratio: float) -> str:
    # Must match build_soupft_config_tag() in model_soup_finetune.py
    return "baseline_soupft_n{}_{}_{}_{}_{}".format(
        _sanitize_value(int(num_soup_models)),
        _sanitize_value(float(ft_lr)),
        _sanitize_value(int(ft_epochs)),
        _sanitize_value(float(ft_wd)),
        _sanitize_value(float(warmup_ratio)),
    )


def _expected_soupft_results_path(
    model_root: str,
    model: str,
    target_dataset: str,
    config_tag: str,
    k: int,
) -> str:
    # Must match model_soup_finetune.py output layout:
    #   save_dir = model_location/model
    #   config_dir = save_dir/{target}Val/{config_tag}
    #   result_dir = config_dir/{k}shots
    return os.path.join(
        model_root,
        model,
        f"{target_dataset}Val",
        config_tag,
        _shot_folder(int(k)),
        "baseline_results_none.json",
    )


# Default dataset list (used only as fallback if config does not provide DATASETS_ALL)
DATASETS_ALL = [
    "DTD",
    "GTSRB",
    "MNIST",
    "SVHN",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "CIFAR100",
    "PCAM",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "EMNIST",
    "CUB200",
    "FGVCAircraft",
    "Country211"
]


# Defaults for sweep grid
GPU_IDS = [0, 1, 2, 3]  # can be overridden via CLI
SOUPFT_MODELS = ["ViT-B-32"]
SOUPFT_K = [2, 4]
SOUPFT_FT_LR = [3e-5]
SOUPFT_FT_EPOCHS = [20]
SOUPFT_FT_WD = [0.0]
SOUPFT_WARMUP_RATIO = [0.1]
SOUPFT_SEEDS = [1]


def build_soupft_commands(
    datasets: Sequence[str],
    best_config_file: str,
    model_root: str,
    cfg_dict: dict,
) -> List[List[str]]:
    commands: List[List[str]] = []

    datasets_all = _datasets_all_from_config(cfg_dict, DATASETS_ALL)
    total_candidate = len(datasets_all)

    for model, dataset, k, ft_lr, ft_epochs, ft_wd, warmup_ratio, seed in itertools.product(
        SOUPFT_MODELS,
        datasets,
        SOUPFT_K,
        SOUPFT_FT_LR,
        SOUPFT_FT_EPOCHS,
        SOUPFT_FT_WD,
        SOUPFT_WARMUP_RATIO,
        SOUPFT_SEEDS,
    ):
        # Soup uses all tasks except the target => N = len(DATASETS_ALL) - 1
        num_soup_models = max(int(total_candidate) - 1, 0)
        config_tag = _soupft_config_tag(
            num_soup_models=num_soup_models,
            ft_lr=float(ft_lr),
            ft_epochs=int(ft_epochs),
            ft_wd=float(ft_wd),
            warmup_ratio=float(warmup_ratio),
        )
        results_json = _expected_soupft_results_path(
            model_root=model_root,
            model=model,
            target_dataset=dataset,
            config_tag=config_tag,
            k=int(k),
        )
        if _path_exists(results_json):
            print(
                f"[skip] soupft {model} {dataset} (k={k}, lr={ft_lr}, epochs={ft_epochs}, wd={ft_wd}, warmup={warmup_ratio}, seed={seed}) -> {results_json}",
                flush=True,
            )
            continue

        cmd = [
            sys.executable,
            "model_soup_finetune.py",
            "--config_file",
            best_config_file,
            "--model",
            model,
            "--target_dataset",
            dataset,
            "--k",
            str(int(k)),
            "--ft_epochs",
            str(int(ft_epochs)),
            "--ft_lr",
            f"{float(ft_lr):.6g}",
            "--ft_wd",
            f"{float(ft_wd):.6g}",
            "--warmup_ratio",
            f"{float(warmup_ratio):.6g}",
            "--seed",
            str(int(seed)),
            "--config_tag",
            config_tag,
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
    parser = argparse.ArgumentParser(
        description="Run model_soup_finetune.py sweeps in parallel across GPUs (skip if results already exist)."
    )
    parser.add_argument("--skip-soupft", action="store_true", help="Skip model soup finetune sweeps")
    parser.add_argument(
        "--best_config_file",
        type=str,
        default="./config/config_reverse.yaml",
        help="Path to config file (relative to repo root).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of commands (after shuffling)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle command order before scheduling")
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

    cfg_dict = _load_config(args.best_config_file)
    model_root = os.path.expanduser(
        (cfg_dict.get("model_location", os.path.join(".", "models", "checkpoints")) if isinstance(cfg_dict, dict) else os.path.join(".", "models", "checkpoints"))
    )

    datasets_all = _datasets_all_from_config(cfg_dict, DATASETS_ALL)
    datasets = sorted(list(datasets_all))

    commands: List[List[str]] = []
    if not args.skip_soupft:
        commands.extend(
            build_soupft_commands(
                datasets=datasets,
                best_config_file=args.best_config_file,
                model_root=model_root,
                cfg_dict=cfg_dict,
            )
        )

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

    commands_per_gpu = max(1, int(args.per_gpu))
    total_workers = len(gpu_ids) * commands_per_gpu
    print(
        f"Prepared {len(commands)} commands across {len(gpu_ids)} GPUs with {commands_per_gpu} slots each "
        f"({total_workers} total workers)."
    )

    run_commands_in_parallel(commands, gpu_ids, per_gpu=commands_per_gpu, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

