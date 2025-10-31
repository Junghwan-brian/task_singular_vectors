import os
import json
import math
import re
import argparse
import logging
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import pandas as pd

from src.models import ImageEncoder
from src.datasets import get_dataloader
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
)
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("eval_single_task_remote_sensing")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_encoder(model_name: str, checkpoint_path: str, device: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        LOGGER.debug(f"Checkpoint missing, skipping: {checkpoint_path}")
        return {"status": "missing"}

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        encoder = ImageEncoder(model_name)
        encoder.load_state_dict(state, strict=False)
        encoder = encoder.to(device)
        encoder.eval()
        return {"status": "ok", "encoder": encoder}
    return {"status": "raw", "object": state}


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        LOGGER.debug(f"JSON missing, skipping: {path}")
        return None
    try:
        with open(path, "r") as fp:
            return json.load(fp)
    except Exception as exc:  # pragma: no cover - best effort parsing
        LOGGER.warning(f"Failed to load JSON {path}: {exc}")
        return None


def to_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    if abs(val) <= 1.0:
        return val * 100.0
    return val


def best_accuracy_from_history(history: Optional[Sequence[Dict[str, Any]]]) -> Optional[float]:
    if not history:
        return None
    best: Optional[float] = None
    for record in history:
        accuracy = record.get("accuracy")
        if isinstance(accuracy, (int, float)):
            percent = to_percent(accuracy)
            if percent is not None and (best is None or percent > best):
                best = percent
    return best


def find_stage_accuracy(history: Optional[Sequence[Dict[str, Any]]], stage: str) -> Optional[float]:
    if not history:
        return None
    for record in history:
        if record.get("stage") == stage:
            accuracy = record.get("accuracy")
            if isinstance(accuracy, (int, float)):
                return to_percent(accuracy)
    return None


def format_numeric(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.2f}"


def format_train_params(value: Optional[Any]) -> str:
    if value is None:
        return "N/A"
    try:
        intval = int(value)
        return f"{intval:,}"
    except (TypeError, ValueError):
        return str(value)


def format_training_time_value(total: Optional[float], base: Optional[float], adapter: Optional[float]) -> str:
    if total is None and base is None and adapter is None:
        return "N/A"
    base_str = format_numeric(base if base is not None else total)
    if adapter is None or adapter == 0:
        return base_str if base_str != "N/A" else format_numeric(total)
    adapter_str = format_numeric(adapter)
    if total is None:
        return f"{base_str} (+{adapter_str})"
    total_str = format_numeric(total)
    return f"{total_str} (+{adapter_str})"



def parse_adapter_tag_from_filename(filename: str, prefix: str) -> str:
    base = filename
    if base.endswith(".json"):
        base = base[:-5]
    suffix = base[len(prefix) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    return canonical_adapter_tag(suffix if suffix else "none")


def compute_metrics_from_json(method_label: str, adapter_tag: str, data: Dict[str, Any]) -> Dict[str, Any]:
    validation_history = data.get("validation_history", [])
    accuracy = best_accuracy_from_history(validation_history)
    if accuracy is None:
        accuracy = to_percent(data.get("final_accuracy"))

    base_training_time = data.get("training_time")
    training_time = base_training_time
    gpu_mem = data.get("gpu_peak_mem_mb")

    initial_accuracy = None
    if method_label.startswith("Energy") and adapter_tag == "none":
        initial_accuracy = find_stage_accuracy(validation_history, "zeroshot")

    adapter_time = None
    if adapter_tag != "none":
        adapter_results = data.get("adapter_results") or {}
        adapter_history = adapter_results.get("validation_history")
        adapter_best = best_accuracy_from_history(adapter_history)
        if adapter_best is not None:
            accuracy = adapter_best
        adapter_time = adapter_results.get("training_time")
        if training_time is not None and adapter_time is not None:
            training_time = training_time + adapter_time
        else:
            training_time = training_time or adapter_time
        gpu_mem = None
        training_params = None
    else:
        adapter_time = None
        training_params = data.get("trainable_params")

    return {
        "accuracy": accuracy,
        "initial_accuracy": initial_accuracy,
        "training_time": training_time,
        "base_training_time": base_training_time,
        "adapter_training_time": adapter_time,
        "gpu": gpu_mem,
        "training_params": training_params,
    }


class DatasetCollector:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.shots: "OrderedDict[str, OrderedDict[str, OrderedDict[str, Dict[str, Any]]]]" = OrderedDict()
        self.baseline_accuracy: Optional[float] = None

    def add_entry(
        self,
        shot_key: str,
        method_label: str,
        adapter_label: str,
        metrics: Dict[str, Any],
    ) -> None:
        shot_bucket = self.shots.setdefault(shot_key, OrderedDict())
        method_bucket = shot_bucket.setdefault(method_label, OrderedDict())
        method_bucket[adapter_label] = metrics


METHOD_DISPLAY_ORDER = [
    "Full_finetuned",
    "Atlas",
    "Energy(tsvm)",
    "Energy(average)",
    "Zeroshot(tsvm)",
    "Zeroshot(average)",
    "Pretrained",
]

METHOD_DISPLAY_NAMES = {
    "Full_finetuned": "Full finetuned",
    "Atlas": "Atlas",
    "Pretrained": "Pretrained",
}

METHOD_TEXT_COLORS = {
    "Full_finetuned": "#008000",
    "Pretrained": "#C00000",
}

SHOT_TEXT_COLORS = {
    "Full-shot": "#008000",
    "Zero-shot": "#C00000",
}

SHOT_ACC_COLORS = {
    "fullshot": "#008000",
    "fullshots": "#008000",
    "zeroshot": "#C00000",
    "zero-shot": "#C00000",
}

ADAPTER_DISPLAY_ORDER = ["None", "LP++", "TIP"]


def normalize_method_for_order(method: str) -> str:
    if method.startswith("Energy_"):
        parts = method.split("_")
        if len(parts) >= 2:
            return f"Energy({parts[1]})"
    if method.startswith("Zeroshot_"):
        parts = method.split("_")
        if len(parts) >= 2:
            return f"Zeroshot({parts[1]})"
    return method


def method_sort_key(method: str) -> int:
    normalized = normalize_method_for_order(method)
    try:
        return METHOD_DISPLAY_ORDER.index(normalized)
    except ValueError:
        return len(METHOD_DISPLAY_ORDER) + 1


def adapter_display_name(tag: str) -> str:
    tag = (tag or "none").strip().lower()
    if tag in {"", "none"}:
        return "None"
    if tag == "lp++":
        return "LP++"
    if tag == "tip":
        return "TIP"
    return tag.upper()


def canonical_adapter_tag(tag: str) -> str:
    tag = (tag or "none").strip().lower()
    if tag in {"", "none"}:
        return "none"
    if tag in {"lp++", "lpp"}:
        return "lp++"
    if tag == "tip":
        return "tip"
    return tag


def method_display_label(method: str) -> str:
    if method in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method]
    if method.startswith("Energy_"):
        parts = method.split("_")
        if len(parts) >= 3:
            return f"Energy({parts[1]}) top-{parts[2]}"
        if len(parts) >= 2:
            return f"Energy({parts[1]})"
    if method.startswith("Zeroshot_"):
        parts = method.split("_")
        if len(parts) >= 3:
            return f"Zeroshot({parts[1]}) top-{parts[2]}"
        if len(parts) >= 2:
            return f"Zeroshot({parts[1]})"
    return method.replace("_", " ")


def adapter_sort_key(adapter_label: str) -> Tuple[int, str]:
    try:
        return (ADAPTER_DISPLAY_ORDER.index(adapter_label), adapter_label)
    except ValueError:
        return (len(ADAPTER_DISPLAY_ORDER), adapter_label)


def determine_energy_variant(config_tag: Optional[str], data: Dict[str, Any], filename: str) -> str:
    candidate = data.get("initialize_sigma")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip().lower()
    if config_tag:
        parts = config_tag.split("_")
        if len(parts) >= 5:
            return parts[4].lower()
    base_name = filename[:-5] if filename.endswith(".json") else filename
    parts = base_name.split("_")
    if parts:
        tail = parts[-1].lower()
        if tail:
            return tail
    return "unknown"


def determine_energy_topk(config_tag: Optional[str], filename: str) -> str:
    if config_tag:
        parts = config_tag.split("_")
        if len(parts) >= 4 and parts[0].lower() == "energy":
            return parts[3]
    base_name = filename[:-5] if filename.endswith(".json") else filename
    parts = base_name.split("_")
    for token in parts:
        if token.isdigit():
            return token
    return "unknown"


def determine_method_label(config_tag: Optional[str], filename: str, data: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    tag_source = (config_tag or filename or "").lower()
    if tag_source.startswith("atlas"):
        return "Atlas", None, None
    if tag_source.startswith("energy"):
        variant = determine_energy_variant(config_tag, data, filename)
        topk = determine_energy_topk(config_tag, filename)
        method_label = f"Energy_{variant}_{topk}" if variant else f"Energy_{topk}"
        return method_label, variant, topk
    return "Other", None, None


def shot_sort_value(shot: str) -> int:
    if not shot:
        return 10**6
    lower = shot.strip().lower()
    if lower in {"full dataset", "full", "fullshot", "fullshots"}:
        return -1
    if lower in {"-", "n/a"}:
        return 10**6
    match = re.match(r"(\d+)", lower)
    if match:
        return int(match.group(1))
    return 10**5


def format_shot_display(shot_key: str) -> str:
    if not shot_key:
        return "-"
    lower = shot_key.strip().lower()
    if lower in {"fullshot", "fullshots"}:
        return "Full-shot"
    if lower in {"zeroshot", "zero-shot"}:
        return "Zero-shot"
    if lower.endswith("shots"):
        value = lower[:-5]
        return f"{value}-shot"
    if lower.endswith("shot"):
        value = lower[:-4]
        return f"{value}-shot"
    return shot_key


def sorted_shots(collector: DatasetCollector) -> List[str]:
    keys = list(collector.shots.keys())

    def priority(shot_key: str) -> Tuple[int, int, int]:
        lower = shot_key.strip().lower()
        if lower in {"fullshot", "fullshots"}:
            return (0, 0, 0)
        if lower in {"zeroshot", "zero-shot"}:
            return (1, 0, 0)
        return (2, shot_sort_value(shot_key), 0)

    decorated = []
    for idx, key in enumerate(keys):
        pri = priority(key)
        decorated.append((pri, idx, key))
    decorated.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    return [item[2] for item in decorated]


# ---------------------------------------------------------------------------
# Directory discovery helpers
# ---------------------------------------------------------------------------

def discover_datasets(model_root: str, datasets_filter: Optional[Iterable[str]]) -> List[Tuple[str, str]]:
    datasets_filter_set = {d if d.endswith("Val") else f"{d}Val" for d in datasets_filter} if datasets_filter else None

    discovered: List[Tuple[str, str]] = []
    if not os.path.exists(model_root):
        raise FileNotFoundError(f"Model directory not found: {model_root}")

    for entry in sorted(os.listdir(model_root)):
        path = os.path.join(model_root, entry)
        if not os.path.isdir(path):
            continue
        if not entry.endswith("Val"):
            continue
        if datasets_filter_set and entry not in datasets_filter_set:
            continue
        dataset_name = entry[:-3] if entry.endswith("Val") else entry
        discovered.append((dataset_name, path))

    return discovered


def canonicalize_shot_name(value: str) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    lower = raw.lower()
    if lower in {"full", "fullshot", "fullshots"}:
        return "fullshots"
    if lower.endswith("shots"):
        return lower
    if lower.endswith("shot"):
        return f"{lower}s"
    if lower.isdigit():
        return f"{lower}shots"
    return ""


def normalize_shot_filter(shots: Optional[Iterable[str]]) -> Optional[set]:
    if not shots:
        return None
    normalized = set()
    for shot in shots:
        canonical = canonicalize_shot_name(shot)
        if canonical:
            normalized.add(canonical)
    return normalized or None


def discover_shot_dirs(dataset_dir: str, shots_filter: Optional[set]) -> List[str]:
    shot_dirs: List[str] = []
    for entry in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(path):
            continue
        canonical = canonicalize_shot_name(entry)
        if not canonical:
            continue
        if shots_filter and canonical not in shots_filter:
            continue
        shot_dirs.append(entry)
    return shot_dirs


def discover_config_shot_dirs(dataset_dir: str, shots_filter: Optional[set]) -> List[Tuple[Optional[str], List[str]]]:
    configs: List[Tuple[Optional[str], List[str]]] = []
    entries = sorted(os.listdir(dataset_dir))
    legacy_shots: List[str] = []
    for entry in entries:
        path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(path):
            continue
        canonical = canonicalize_shot_name(entry)
        if canonical:
            if not shots_filter or canonical in shots_filter:
                legacy_shots.append(entry)
        else:
            shots = discover_shot_dirs(path, shots_filter)
            if shots:
                configs.append((entry, shots))
    if configs:
        return configs
    if legacy_shots:
        return [(None, legacy_shots)]
    return []


# ---------------------------------------------------------------------------
# Baseline evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_encoder_accuracy(
    encoder: ImageEncoder,
    classification_head,
    val_loader,
    device: str,
) -> float:
    metrics = evaluate_encoder_with_dataloader(
        encoder, classification_head, val_loader, device
    )
    return metrics.get("top1", 0.0) * 100.0


def populate_baseline_entries(
    collector: DatasetCollector,
    dataset_dir: str,
    classification_head,
    val_loader,
    model: str,
    device: str,
) -> None:
    dataset = collector.dataset_name
    # Full finetuned baseline
    finetuned_ckpt = os.path.join(dataset_dir, "nonlinear_finetuned.pt")
    finetuned_info = load_json(os.path.join(dataset_dir, "nonlinear_finetuned_training_info.json"))
    finetuned_time = None
    finetuned_gpu = None
    if finetuned_info:
        finetuned_time = finetuned_info.get("total_time_seconds")
        finetuned_gpu = finetuned_info.get("gpu_peak_mem_mb")

    finetuned_encoder = load_encoder(model, finetuned_ckpt, device)
    if finetuned_encoder.get("status") == "ok":
        finetuned_acc = evaluate_encoder_accuracy(finetuned_encoder["encoder"], classification_head, val_loader, device)
        collector.baseline_accuracy = finetuned_acc
        finetuned_train_params = None
        if finetuned_info:
            finetuned_train_params = finetuned_info.get("trainable_params")
        collector.add_entry(
            "fullshot",
            "Full_finetuned",
            "None",
            {
                "accuracy": finetuned_acc,
                "initial_accuracy": None,
                "training_time": finetuned_time,
                "base_training_time": finetuned_time,
                "adapter_training_time": None,
                "gpu": finetuned_gpu,
                "training_params": finetuned_train_params,
            },
        )
    else:
        LOGGER.warning(f"{dataset}: missing or invalid full finetuned checkpoint ({finetuned_ckpt})")

    # Pretrained baseline
    pretrained_ckpt = os.path.join(dataset_dir, "nonlinear_zeroshot.pt")
    pretrained_encoder = load_encoder(model, pretrained_ckpt, device)
    if pretrained_encoder.get("status") == "ok":
        pretrained_acc = evaluate_encoder_accuracy(pretrained_encoder["encoder"], classification_head, val_loader, device)
        collector.add_entry(
            "zeroshot",
            "Pretrained",
            "None",
            {
                "accuracy": pretrained_acc,
                "initial_accuracy": None,
                "training_time": None,
                "base_training_time": None,
                "adapter_training_time": None,
                "gpu": None,
                "training_params": None,
            },
        )
    else:
        LOGGER.warning(f"{dataset}: missing or invalid pretrained checkpoint ({pretrained_ckpt})")


# ---------------------------------------------------------------------------
# JSON result aggregation
# ---------------------------------------------------------------------------

def build_entry_from_json(
    config_tag: Optional[str],
    shot_folder: str,
    filename: str,
    data: Dict[str, Any],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    file_lower = filename.lower()
    if file_lower.startswith("atlas_results"):
        adapter_tag = parse_adapter_tag_from_filename(filename, "atlas_results")
    elif file_lower.startswith("energy_results"):
        adapter_tag = parse_adapter_tag_from_filename(filename, "energy_results")
    else:
        return None
    method_label, _, _ = determine_method_label(config_tag, filename, data)
    adapter_display = adapter_display_name(adapter_tag)
    metrics = compute_metrics_from_json(method_label, adapter_tag, data)
    return method_label, adapter_display, metrics


def collect_config_results(
    collector: DatasetCollector,
    dataset_dir: str,
    config_tag: Optional[str],
    shot_folder: str,
) -> None:
    if config_tag:
        shot_path = os.path.join(dataset_dir, config_tag, shot_folder)
    else:
        shot_path = os.path.join(dataset_dir, shot_folder)

    if not os.path.isdir(shot_path):
        LOGGER.debug(f"Shot directory missing, skipping: {shot_path}")
        return

    for filename in sorted(os.listdir(shot_path)):
        if not filename.endswith(".json"):
            continue
        lowered = filename.lower()
        if not (lowered.startswith("atlas") or lowered.startswith("energy")):
            continue
        data = load_json(os.path.join(shot_path, filename))
        if not isinstance(data, dict):
            continue
        entry = build_entry_from_json(
            config_tag=config_tag,
            shot_folder=shot_folder,
            filename=filename,
            data=data,
        )
        if not entry:
            continue
        method_label, adapter_display, metrics = entry
        collector.add_entry(shot_folder, method_label, adapter_display, metrics)


def prepare_rows_for_render(
    collectors: List[DatasetCollector],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    rows: List[Dict[str, Any]] = []
    dataset_ranges: List[Tuple[int, int]] = []
    shot_ranges: List[Tuple[int, int]] = []
    method_ranges: List[Tuple[int, int]] = []

    row_index = 0

    dataset_best_map: Dict[str, Optional[float]] = {}
    for collector in collectors:
        best_val: Optional[float] = None
        for shot_methods in collector.shots.values():
            for method_label, adapter_map in shot_methods.items():
                if method_label == "Full_finetuned":
                    continue
                for metrics in adapter_map.values():
                    acc = metrics.get("accuracy")
                    if acc is None:
                        continue
                    if best_val is None or acc > best_val:
                        best_val = acc
        dataset_best_map[collector.dataset_name] = best_val

    for dataset_idx, collector in enumerate(collectors):
        dataset_start = row_index
        dataset_has_rows = False

        shot_order = sorted_shots(collector)
        dataset_printed = False

        for shot_key in shot_order:
            methods = collector.shots.get(shot_key)
            if not methods:
                continue
            method_keys = sorted(methods.keys(), key=method_sort_key)
            if not method_keys:
                continue
            shot_start = row_index
            shot_has_rows = False
            shot_display = format_shot_display(shot_key)
            shot_printed = False

            for method_label in method_keys:
                adapters = methods.get(method_label, {})
                if not adapters:
                    continue
                adapter_items = sorted(adapters.items(), key=lambda item: adapter_sort_key(item[0]))
                method_start = row_index
                method_has_rows = False
                method_display = method_display_label(method_label)
                method_printed = False
                dataset_best = dataset_best_map.get(collector.dataset_name)

                for adapter_label, metrics in adapter_items:
                    accuracy_value = metrics.get("accuracy")
                    highlight_accuracy = (
                        method_label != "Full_finetuned"
                        and method_label != "Pretrained"
                        and dataset_best is not None
                        and accuracy_value is not None
                        and abs(accuracy_value - dataset_best) < 1e-6
                    )
                    rows.append(
                        {
                            "dataset": collector.dataset_name if not dataset_printed else "",
                            "shot": shot_display if not shot_printed else "",
                            "method": method_display if not method_printed else "",
                            "adapter": adapter_label,
                            "accuracy": accuracy_value,
                            "initial": metrics.get("initial_accuracy"),
                            "time": metrics.get("training_time"),
                            "base_time": metrics.get("base_training_time"),
                            "adapter_time": metrics.get("adapter_training_time"),
                            "gpu": metrics.get("gpu"),
                            "training_params": metrics.get("training_params"),
                            "shot_key": shot_key,
                            "is_spacer": False,
                            "method_key": method_label,
                            "highlight_accuracy": highlight_accuracy,
                        }
                    )
                    dataset_printed = True
                    shot_printed = True
                    method_printed = True
                    method_has_rows = True
                    shot_has_rows = True
                    dataset_has_rows = True
                    row_index += 1

                if method_has_rows:
                    method_ranges.append((method_start, row_index - 1))

            if shot_has_rows:
                shot_ranges.append((shot_start, row_index - 1))

        if dataset_has_rows:
            dataset_ranges.append((dataset_start, row_index - 1))

        # Spacer between datasets
        if dataset_idx < len(collectors) - 1:
            rows.append({"is_spacer": True})
            row_index += 1

    return rows, dataset_ranges, shot_ranges, method_ranges


def write_summary_csv(collectors: List[DatasetCollector], csv_path: str) -> None:
    csv_rows: List[Dict[str, Any]] = []
    for collector in collectors:
        for shot_key, methods in collector.shots.items():
            shot_display = format_shot_display(shot_key)
            for method_label, adapters in methods.items():
                method_display = method_display_label(method_label)
                for adapter_label, metrics in adapters.items():
                    csv_rows.append(
                        {
                            "Dataset": collector.dataset_name,
                            "Shot": shot_display,
                            "Method": method_display,
                            "Adapter": adapter_label,
                            "Accuracy (%)": format_numeric(metrics.get("accuracy")),
                            "Initial Accuracy (%)": format_numeric(metrics.get("initial_accuracy")),
                            "Training Time (s)": format_training_time_value(
                                metrics.get("training_time"),
                                metrics.get("base_training_time"),
                                metrics.get("adapter_training_time"),
                            ),
                            "GPU Peak Mem (MB)": format_numeric(metrics.get("gpu")),
                            "Training Params": format_train_params(metrics.get("training_params")),
                        }
                    )

    if not csv_rows:
        LOGGER.warning("No rows collected; skipping summary CSV export.")
        return

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)


def draw_bracket(ax, x: float, y_top: float, y_bottom: float, width: float = 0.015, linewidth: float = 1.2) -> None:
    ax.plot([x, x + width], [y_top, y_top], color="black", lw=linewidth)
    ax.plot([x, x], [y_bottom, y_top], color="black", lw=linewidth)
    ax.plot([x, x + width], [y_bottom, y_bottom], color="black", lw=linewidth)


def render_summary_tree(collectors: List[DatasetCollector], model: str, model_root: str) -> None:
    rows, dataset_ranges, shot_ranges, method_ranges = prepare_rows_for_render(collectors)

    if not rows or all(entry.get("is_spacer") for entry in rows):
        LOGGER.warning("No rows collected; skipping summary image export.")
        return

    total_rows = len(rows)
    fig_height = max(5.0, total_rows * 0.44 + 2.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    ax.set_xlim(0, 1.28)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_step = 1.0 / (total_rows + 6)
    header_y = 1 - y_step
    start_y = header_y - y_step
    y_positions = [start_y - idx * y_step for idx in range(total_rows)]

    header_columns = [
        {"key": "dataset", "label": "Datasets", "x": 0.05, "color": "#C00000"},
        {"key": "shot", "label": "Shots", "x": 0.20, "color": "#C00000"},
        {"key": "method", "label": "Methods", "x": 0.35, "color": "#C00000"},
        {"key": "adapter", "label": "Adapter", "x": 0.60, "color": "#C00000"},
        {"key": "accuracy", "label": "Accuracy", "x": 0.82, "color": "#1F4E79"},
        {"key": "initial", "label": "Initial Accuracy", "x": 1.00, "color": "#1F4E79"},
        {"key": "time", "label": "Training Time", "x": 1.20, "color": "#1F4E79"},
        {"key": "gpu", "label": "GPU", "x": 1.40, "color": "#1F4E79"},
        {"key": "training_params", "label": "Training Params", "x": 1.50, "color": "#1F4E79"},
    ]

    for column in header_columns:
        ax.text(
            column["x"],
            header_y,
            column["label"],
            color=column["color"],
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

    metrics_color = "#1F4E79"
    highlight_color = "#FF8C00"

    for idx, entry in enumerate(rows):
        if entry.get("is_spacer"):
            continue
        y = y_positions[idx]
        dataset_text = entry.get("dataset", "")
        ax.text(0.05, y, dataset_text, fontsize=11, color="black", ha="left", va="center")

        shot_text = entry.get("shot", "")
        ax.text(0.20, y, shot_text, fontsize=11, color="black", ha="left", va="center")

        method_text = entry.get("method", "")
        ax.text(0.35, y, method_text, fontsize=11, color="black", ha="left", va="center")

        ax.text(0.60, y, entry.get("adapter", ""), fontsize=11, color="black", ha="left", va="center")

        shot_key = (entry.get("shot_key") or "").lower()
        acc_color = metrics_color
        if entry.get("highlight_accuracy"):
            acc_color = highlight_color
        elif shot_key in SHOT_ACC_COLORS:
            acc_color = SHOT_ACC_COLORS[shot_key]
        ax.text(0.82, y, format_numeric(entry.get("accuracy")), fontsize=11, color=acc_color, ha="left", va="center")
        ax.text(1.00, y, format_numeric(entry.get("initial")), fontsize=11, color=metrics_color, ha="left", va="center")

        time_text = format_training_time_value(
            entry.get("time"), entry.get("base_time"), entry.get("adapter_time")
        )
        ax.text(1.20, y, time_text, fontsize=11, color=metrics_color, ha="left", va="center")
        ax.text(1.40, y, format_numeric(entry.get("gpu")), fontsize=11, color=metrics_color, ha="left", va="center")
        ax.text(1.50, y, format_train_params(entry.get("training_params")), fontsize=11, color=metrics_color, ha="left", va="center")

    # Draw brackets for datasets, shots, methods
    for start, end in dataset_ranges:
        if end >= start:
            draw_bracket(ax, 0.02, y_positions[start], y_positions[end], width=0.02, linewidth=1.4)

    for start, end in shot_ranges:
        if end >= start:
            draw_bracket(ax, 0.30, y_positions[start], y_positions[end], width=0.02, linewidth=1.2)

    for start, end in method_ranges:
        if end >= start:
            draw_bracket(ax, 0.55, y_positions[start], y_positions[end], width=0.02, linewidth=1.0)

    fig.tight_layout()
    image_path = os.path.join(model_root, f"{model}_summary.png")
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    LOGGER.info(f"Saved summary image to {image_path}")


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    dataset_dir: str,
    shots_filter: Optional[set],
    model: str,
    data_location: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> DatasetCollector:
    LOGGER.info(f"Processing dataset: {dataset}")
    val_name = f"{dataset}Val"
    template_encoder = ImageEncoder(model).to(device)
    val_dataset = get_remote_sensing_dataset(
        val_name,
        template_encoder.val_preprocess,
        location=data_location,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    args_ns = SimpleNamespace(
        save_dir=os.path.join(dataset_dir, ".."),
        model=model,
        device=device,
        batch_size=batch_size,
    )
    classification_head = get_remote_sensing_classification_head(args_ns, val_name, val_dataset)
    classification_head = classification_head.to(device)
    classification_head.eval()

    val_loader = get_dataloader(val_dataset, is_train=False, args=args_ns, image_encoder=None)

    collector = DatasetCollector(dataset)
    try:
        populate_baseline_entries(
            collector=collector,
            dataset_dir=dataset_dir,
            classification_head=classification_head,
            val_loader=val_loader,
            model=model,
            device=device,
        )
    finally:
        del classification_head
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config_shots = discover_config_shot_dirs(dataset_dir, shots_filter)
    for config_tag, shot_folders in config_shots:
        inferred_method, _, _ = determine_method_label(config_tag, config_tag or "", {})
        LOGGER.info(f"{dataset}: {config_tag or 'legacy'} -> {inferred_method}, shots={shot_folders}")
        for shot_folder in shot_folders:
            before_count = sum(
                len(methods)
                for methods in collector.shots.get(shot_folder, {}).values()
            )
            collect_config_results(collector, dataset_dir, config_tag, shot_folder)
            after_bucket = collector.shots.get(shot_folder, OrderedDict())
            after_count = sum(len(adapters) for adapters in after_bucket.values())
            if after_count == before_count:
                LOGGER.warning(f"{dataset} | {shot_folder}: no JSON results found under {config_tag or 'legacy'}")
    return collector


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------

def save_summary_table(rows: List[Dict[str, Any]], model: str, model_root: str) -> None:
    if not rows:
        LOGGER.warning("No rows collected; skipping summary export.")
        return

    df = pd.DataFrame(rows)
    df["MethodOrder"] = df["Method"].apply(method_sort_key)
    df["ShotOrder"] = df["Shot"].apply(shot_sort_value)
    df["_FinalNumeric"] = pd.to_numeric(df["Final Accuracy (%)"], errors="coerce")

    highlight_map: Dict[Any, str] = {}
    for dataset_name, group in df.groupby("Dataset"):
        valid = group["_FinalNumeric"].dropna().sort_values(ascending=False)
        if not valid.empty:
            highlight_map[valid.index[0]] = "best"
            if len(valid) > 1:
                highlight_map[valid.index[1]] = "second"

    df.sort_values(
        ["Dataset", "ShotOrder", "MethodOrder", "Adapter"],
        inplace=True,
    )

    columns = [
        "Dataset",
        "Shot",
        "Method",
        "Adapter",
        "Best Accuracy (%)",
        "Final Accuracy (%)",
        "Training Time (s)",
        "GPU Peak Mem (MB)",
    ]

    csv_df = df[columns].copy()
    csv_path = os.path.join(model_root, f"{model}_summary.csv")
    csv_df.to_csv(csv_path, index=False)

    display_df = csv_df.copy()
    numeric_cols = ["Best Accuracy (%)", "Final Accuracy (%)", "Training Time (s)", "GPU Peak Mem (MB)"]
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(format_numeric)

    display_df.fillna("N/A", inplace=True)

    table_rows: List[List[str]] = []
    row_styles: List[Dict[str, Any]] = []
    for dataset_name, group in display_df.groupby("Dataset"):
        header_row = [dataset_name] + [""] * (len(columns) - 1)
        table_rows.append(header_row)
        row_styles.append({"type": "section"})
        for idx, row in group.iterrows():
            row_values = [
                "",
                row["Shot"],
                row["Method"],
                row["Adapter"],
                row["Best Accuracy (%)"],
                row["Final Accuracy (%)"],
                row["Training Time (s)"],
                row["GPU Peak Mem (MB)"],
            ]
            table_rows.append(row_values)
            row_styles.append({"type": "data", "highlight": highlight_map.get(idx)})

    table_height = max(3, 0.35 * len(table_rows) + 1)
    table_width = max(12, 1.2 * len(columns))

    fig, ax = plt.subplots(figsize=(table_width, table_height))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    for idx in range(len(columns)):
        table[(0, idx)].set_facecolor("#4472C4")
        table[(0, idx)].set_text_props(weight="bold", color="white")

    section_color = "#DDEBF7"
    best_color = "#D9EAD3"
    second_color = "#FFF2CC"

    for row_idx, style in enumerate(row_styles):
        table_row = row_idx + 1  # account for header row at index 0
        if style["type"] == "section":
            for col_idx in range(len(columns)):
                cell = table[(table_row, col_idx)]
                cell.set_facecolor(section_color)
                weight = "bold" if col_idx == 0 else "normal"
                cell.set_text_props(weight=weight)
                cell.set_linewidth(1.0)
        else:
            highlight = style.get("highlight")
            if highlight == "best":
                facecolor = best_color
            elif highlight == "second":
                facecolor = second_color
            else:
                facecolor = "white"
            for col_idx in range(len(columns)):
                cell = table[(table_row, col_idx)]
                cell.set_facecolor(facecolor)
                cell.set_linewidth(0.5)
            if highlight == "best":
                table[(table_row, columns.index("Method"))].set_text_props(weight="bold")

    fig.tight_layout()
    image_path = os.path.join(model_root, f"{model}_summary.png")
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    LOGGER.info(f"Saved summary CSV to {csv_path}")
    LOGGER.info(f"Saved summary image to {image_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate remote sensing evaluation results (atlas / energy / baseline)."
    )
    parser.add_argument("--model", default="ViT-B-32", help="Model architecture name (default: ViT-B-32)")
    parser.add_argument(
        "--model_location",
        default="./models/checkpoints_remote_sensing",
        help="Root directory containing checkpoints",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset names to include (without 'Val'). Processes all if omitted.",
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        default=None,
        help="Optional list of shot folders (e.g., 16shot, 16shots, fullshots). Processes all if omitted.",
    )
    parser.add_argument("--data_location", default="./datasets", help="Dataset root for baseline evaluation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    model_root = os.path.join(args.model_location, args.model)
    dataset_dirs = discover_datasets(model_root, args.datasets)
    if not dataset_dirs:
        LOGGER.warning(f"No datasets found under {model_root}")
        return

    shots_filter = normalize_shot_filter(args.shots)

    collectors: List[DatasetCollector] = []
    for dataset, dataset_dir in dataset_dirs:
        collector = evaluate_dataset(
            dataset=dataset,
            dataset_dir=dataset_dir,
            shots_filter=shots_filter,
            model=args.model,
            data_location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        collectors.append(collector)

    if not collectors:
        LOGGER.warning("No datasets processed; nothing to export.")
        return

    csv_path = os.path.join(model_root, f"{args.model}_summary.csv")
    write_summary_csv(collectors, csv_path)
    LOGGER.info(f"Saved summary CSV to {csv_path}")

    render_summary_tree(collectors, args.model, model_root)


if __name__ == "__main__":
    main()
