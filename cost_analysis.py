import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

import energy_train_reverse as etr


def _tensor_nbytes(t: torch.Tensor) -> int:
    # torch.Tensor.element_size() is bytes per element
    return int(t.numel()) * int(t.element_size())


def _bytes_to_mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0**2)


def _task_vector_nbytes(tv: Any) -> int:
    # NonLinearTaskVector exposes .vector: Dict[str, Tensor]
    total = 0
    for v in tv.vector.values():
        if torch.is_tensor(v):
            total += _tensor_nbytes(v)
    return int(total)


def _svd_basis_nbytes(svd_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    compute_and_sum_svd_mem_reduction returns:
      - for non-2D keys: Tensor (average)
      - for 2D keys: [U_orth, Sigma(diag-matrix), V_orth]

    We estimate a compact storage size consistent with downstream usage:
      - store U (m x chunks), V (chunks x n), and sigma_vec (chunks,)
    """
    u_bytes = 0
    v_bytes = 0
    sigma_vec_bytes = 0
    for value in svd_dict.values():
        if isinstance(value, list) and len(value) == 3:
            U, Sigma, V = value
            if torch.is_tensor(U):
                u_bytes += _tensor_nbytes(U)
            if torch.is_tensor(V):
                v_bytes += _tensor_nbytes(V)
            # store sigma as vector (diagonal), which is what run_energy exports
            if torch.is_tensor(Sigma) and Sigma.ndim == 2:
                sigma_vec = torch.diagonal(Sigma)
                sigma_vec_bytes += _tensor_nbytes(sigma_vec)
    return {
        "U_bytes": int(u_bytes),
        "V_bytes": int(v_bytes),
        "sigma_vec_bytes": int(sigma_vec_bytes),
        "basis_total_bytes": int(u_bytes + v_bytes + sigma_vec_bytes),
    }


def _cuda_sync_if_needed(device: str) -> None:
    dev = str(device or "").lower()
    if "cuda" in dev and torch.cuda.is_available():
        torch.cuda.synchronize()


def _select_basis_datasets(test_dataset: Optional[str], max_task_vectors: int) -> List[str]:
    all_ds = list(etr.ALL_DATASETS)
    if test_dataset:
        all_ds = [d for d in all_ds if d != test_dataset]
    return all_ds[: max(0, int(max_task_vectors))]


def _load_task_vectors(cfg: Any, basis_datasets: List[str]) -> List[Any]:
    # Load fine-tuned checkpoints for basis tasks
    ft_checks = []
    for dataset in basis_datasets:
        dataset_val = dataset + "Val"
        path = etr.get_finetuned_path(cfg.model_location, dataset_val, model=cfg.model)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
        ft_checks.append(etr.load_checkpoint_safe(path, map_location="cpu"))

    # Load pretrained / zeroshot checkpoint (convention: first dataset)
    first_dataset_val = (basis_datasets[0] + "Val") if basis_datasets else "dummy"
    zeroshot_path = etr.get_zeroshot_path(cfg.model_location, first_dataset_val, model=cfg.model)
    ptm_check = etr.load_checkpoint_safe(zeroshot_path, map_location="cpu")

    return [etr.NonLinearTaskVector(cfg.model, ptm_check, check) for check in ft_checks]


def _summarize_task_vector_memory(task_vectors: List[Any]) -> Dict[str, Any]:
    per_task_bytes = [_task_vector_nbytes(tv) for tv in task_vectors]
    mean_bytes = int(sum(per_task_bytes) / max(1, len(per_task_bytes)))
    sum_bytes = int(sum(per_task_bytes))
    return {
        "per_task_vector_bytes": per_task_bytes,
        "per_task_vector_mb": [_bytes_to_mb(b) for b in per_task_bytes],
        "task_vector_bytes_mean": mean_bytes,
        "task_vector_bytes_sum": sum_bytes,
        "task_vector_mb_mean": _bytes_to_mb(mean_bytes),
        "task_vector_mb_sum": _bytes_to_mb(sum_bytes),
    }


def run_cost_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    # Load config and merge CLI overrides
    cfg = etr.load_config(args.config_file)
    OmegaConf.set_struct(cfg, False)

    if args.model is not None:
        cfg.model = args.model
    if args.model_location is not None:
        cfg.model_location = args.model_location
    if args.device is not None:
        cfg.device = args.device
    if args.svd_keep_topk is not None:
        cfg.svd_keep_topk = int(args.svd_keep_topk)

    sigma_reduce = str(args.sigma_reduce).lower()

    max_task_vectors = int(args.max_task_vectors)
    test_dataset = args.test_dataset

    basis_datasets = _select_basis_datasets(test_dataset, max_task_vectors)
    if not basis_datasets:
        raise ValueError("No basis datasets selected. Increase --max_task_vectors or adjust --test_dataset.")

    # Keep cfg fields used by compute_and_sum_svd_mem_reduction consistent
    cfg.DATASETS = list(basis_datasets)
    cfg.num_tasks = len(basis_datasets)
    cfg.DATASETS_VAL = [d + "Val" for d in basis_datasets]

    end_to_end_start = time.perf_counter()
    task_vectors_all = _load_task_vectors(cfg, basis_datasets)
    end_to_end_load_done = time.perf_counter()

    # Sweep counts: 1..max_task_vectors
    runs: List[Dict[str, Any]] = []
    for count in range(1, len(task_vectors_all) + 1):
        task_vectors = task_vectors_all[:count]

        mem_summary = _summarize_task_vector_memory(task_vectors)

        _cuda_sync_if_needed(cfg.device)
        t0 = time.perf_counter()
        svd_dict = etr.compute_and_sum_svd_mem_reduction(task_vectors, cfg, sigma_reduce=sigma_reduce)
        _cuda_sync_if_needed(cfg.device)
        elapsed = time.perf_counter() - t0

        basis_mem = _svd_basis_nbytes(svd_dict)

        # "per-task vector time (average)" as total / count
        per_task_time_avg = float(elapsed) / float(count)

        # If you wanted to store per-task sigmas, you'd store one sigma vector per task.
        # Use the aggregated sigma vector size as a proxy for per-task sigma storage.
        sigma_bytes_per_task_proxy = int(basis_mem["sigma_vec_bytes"])
        reduced_total_bytes_proxy = int(basis_mem["U_bytes"] + basis_mem["V_bytes"] + (count * sigma_bytes_per_task_proxy))

        runs.append(
            {
                "task_vector_count": int(count),
                "sigma_reduce": sigma_reduce,
                "svd_keep_topk": int(getattr(cfg, "svd_keep_topk", 0)),
                "svd_compute_time_seconds": float(elapsed),
                "svd_compute_time_seconds_per_task_avg": float(per_task_time_avg),
                "task_vector_memory": mem_summary,
                "svd_basis_memory_bytes": basis_mem,
                "svd_basis_memory_mb": {k.replace("_bytes", "_mb"): _bytes_to_mb(v) for k, v in basis_mem.items()},
                "svd_reduced_total_bytes_proxy": reduced_total_bytes_proxy,
                "svd_reduced_total_mb_proxy": _bytes_to_mb(reduced_total_bytes_proxy),
            }
        )

    end_to_end_end = time.perf_counter()

    result: Dict[str, Any] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device": str(getattr(cfg, "device", "unknown")),
        },
        "config": {
            "config_file": args.config_file,
            "model": str(getattr(cfg, "model", "")),
            "model_location": str(getattr(cfg, "model_location", "")),
            "test_dataset": test_dataset,
            "basis_datasets": basis_datasets,
            "max_task_vectors": int(max_task_vectors),
        },
        "timing": {
            "load_task_vectors_seconds": float(end_to_end_load_done - end_to_end_start),
            "end_to_end_seconds": float(end_to_end_end - end_to_end_start),
        },
        "runs": runs,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost analysis for compute_and_sum_svd_mem_reduction vs number of task vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, default="config/config_reverse.yaml", help="Path to YAML config")
    parser.add_argument("--test_dataset", type=str, default=None, help="Exclude this dataset from basis list (optional)")
    parser.add_argument("--max_task_vectors", type=int, required=True, help="Maximum number of task vectors to sweep (1..N)")
    parser.add_argument("--sigma_reduce", type=str, default="mean", choices=["mean", "average", "max", "sum"])
    parser.add_argument("--svd_keep_topk", type=int, default=None, help="Override svd_keep_topk in config")
    parser.add_argument("--model", type=str, default=None, help="Override model in config (e.g., ViT-B-32)")
    parser.add_argument("--model_location", type=str, default=None, help="Override model_location in config")
    parser.add_argument("--device", type=str, default=None, help="Override device in config (e.g., cpu, cuda)")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output json path (default: cost_analysis/<auto_name>.json)",
    )

    args = parser.parse_args()

    result = run_cost_analysis(args)

    os.makedirs("cost_analysis", exist_ok=True)
    if args.output_json:
        out_path = args.output_json
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model = str(result["config"].get("model", "model")).replace("/", "_")
        max_tv = int(result["config"]["max_task_vectors"])
        sigma_reduce = str(result["runs"][0]["sigma_reduce"]) if result.get("runs") else "mean"
        out_path = os.path.join("cost_analysis", f"svd_mem_reduction_cost_{model}_{sigma_reduce}_max{max_tv}_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✓ Saved cost analysis to {out_path}")


if __name__ == "__main__":
    main()

