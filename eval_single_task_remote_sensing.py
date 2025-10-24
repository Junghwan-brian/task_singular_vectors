import os
import argparse
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import pandas as pd

from src.models import ImageEncoder
from src.datasets import get_dataloader
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
)
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

LOGGER = logging.getLogger("eval_single_task_remote_sensing")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_encoder(model_name: str, checkpoint_path: str, device: str) -> Dict[str, object]:
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
    else:
        return {"status": "raw", "object": state}


def fetch_training_time(json_path: str, candidate_keys: Sequence[str]) -> str:
    if not os.path.exists(json_path):
        return "N/A"
    try:
        import json

        with open(json_path, "r") as fp:
            data = json.load(fp)
        for key in candidate_keys:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return f"{value:.2f}"
                return str(value)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning(f"Could not read training time from {json_path}: {exc}")
    return "N/A"


def fetch_accuracy_from_json(json_path: str, candidate_keys: Sequence[str]) -> Optional[float]:
    if not os.path.exists(json_path):
        return None
    try:
        import json

        with open(json_path, "r") as fp:
            data = json.load(fp)
        for key in candidate_keys:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return float(value) * (100.0 if value <= 1.0 else 1.0)
        return None
    except Exception as exc:  # pragma: no cover
        LOGGER.warning(f"Could not read accuracy from {json_path}: {exc}")
        return None


def build_results_table(
    results: List[Dict[str, str]],
    save_dir: str,
    dataset: str,
    model: str,
    shot_folder: str,
) -> Tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"comparison_{dataset}_{model}_{shot_folder}"

    csv_path = os.path.join(save_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.axis("tight")

    cols = ["Config", "Variant", "Training Data", "Training Time (s)", "Accuracy (%)"]
    table = ax.table(
        cellText=[[row[c] for c in cols] for _, row in df.iterrows()],
        colLabels=cols,
        cellLoc="left",
        loc="center",
        colWidths=[0.18, 0.26, 0.20, 0.18, 0.18],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for idx in range(len(cols)):
        table[(0, idx)].set_facecolor("#4472C4")
        table[(0, idx)].set_text_props(weight="bold", color="white")

    for row_idx in range(1, len(df) + 1):
        variant = table[(row_idx, 1)].get_text().get_text().lower()
        for col_idx in range(len(cols)):
            if "energy" in variant:
                table[(row_idx, col_idx)].set_facecolor("#C6E0B4")
                table[(row_idx, col_idx)].set_text_props(weight="bold")
            elif row_idx % 2 == 0:
                table[(row_idx, col_idx)].set_facecolor("#F2F2F2")

    title = f"{dataset} ({shot_folder}) • {model}\nGenerated {timestamp}"
    plt.title(title, fontsize=14, fontweight="bold", pad=18)

    png_path = os.path.join(save_dir, f"{base_name}_{timestamp}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    LOGGER.info(f"✓ Saved comparison CSV: {csv_path}")
    LOGGER.info(f"✓ Saved comparison PNG: {png_path}")
    return csv_path, png_path


# ---------------------------------------------------------------------------
# Discovery utilities
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


def normalize_shot_filter(shots: Optional[Iterable[str]]) -> Optional[set]:
    if not shots:
        return None
    normalized = set()
    for shot in shots:
        shot = str(shot)
        normalized.add(shot if shot.endswith("shot") else f"{shot}shot")
    return normalized


def discover_shot_dirs(dataset_dir: str, shots_filter: Optional[set]) -> List[str]:
    shot_dirs: List[str] = []
    for entry in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(path):
            continue
        if not entry.endswith("shot"):
            continue
        if shots_filter and entry not in shots_filter:
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
        if entry.endswith("shot") or entry.endswith("shots"):
            if not shots_filter or entry in shots_filter:
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


def format_shot_name(shot_folder: str) -> str:
    name = shot_folder.lower()
    if name in {"fullshot", "fullshots"}:
        return "Full dataset"
    if name.endswith("shots"):
        core = shot_folder[:-5]
        return f"{core}-shot"
    if name.endswith("shot"):
        return shot_folder.replace("shot", "-shot")
    return shot_folder


# ---------------------------------------------------------------------------
# Evaluation routines
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


def evaluate_base_variants(
    dataset: str,
    dataset_dir: str,
    classification_head,
    val_loader,
    model: str,
    device: str,
    config_label: str,
) -> List[Dict[str, str]]:
    variants = [
        (
            "Full Finetuned",
            os.path.join(dataset_dir, "nonlinear_finetuned.pt"),
            "Full dataset",
        ),
        (
            "Zeroshot",
            os.path.join(dataset_dir, "nonlinear_zeroshot.pt"),
            "0-shot",
        ),
    ]

    results: List[Dict[str, str]] = []
    for variant_name, path, training_data in variants:
        load_info = load_encoder(model, path, device)
        if load_info["status"] == "missing":
            LOGGER.warning(f"Skipping {variant_name} (checkpoint missing)")
            continue
        if load_info["status"] != "ok":
            LOGGER.warning(f"Skipping {variant_name} (unexpected checkpoint format: {type(load_info['object'])})")
            continue
        encoder = load_info["encoder"]
        accuracy = evaluate_encoder_accuracy(encoder, classification_head, val_loader, device)
        LOGGER.info(f"{dataset} | {variant_name}: {accuracy:.2f}%  ({path})")
        results.append(
            {
                "Variant": variant_name,
                "Config": config_label,
                "Training Data": training_data,
                "Training Time (s)": "N/A",
                "Accuracy (%)": f"{accuracy:.2f}",
            }
        )
    return results


def evaluate_shot_variants(
    dataset: str,
    dataset_dir: str,
    shot_folder: str,
    base_results: List[Dict[str, str]],
    classification_head,
    val_loader,
    model: str,
    device: str,
    config_tag: Optional[str],
) -> Optional[List[Dict[str, str]]]:
    if config_tag:
        shot_dir = os.path.join(dataset_dir, config_tag, shot_folder)
    else:
        shot_dir = os.path.join(dataset_dir, shot_folder)
    if not os.path.isdir(shot_dir):
        LOGGER.debug(f"Shot directory missing, skipping: {shot_dir}")
        return None

    results = [dict(row) for row in base_results]

    variants = [
        (
            "Atlas",
            os.path.join(shot_dir, "atlas.pt"),
            fetch_training_time(os.path.join(shot_dir, "atlas_results.json"), ("training_time", "total_time")),
            ("final_accuracy", "best_val_accuracy"),
        ),
        (
            "Energy",
            os.path.join(shot_dir, "energy.pt"),
            fetch_training_time(
                os.path.join(shot_dir, "energy_results.json"),
                ("total_time", "sigma_train_time"),
            ),
            ("final_accuracy",),
        ),
    ]

    added = False
    for variant_name, path, training_time, accuracy_keys in variants:
        load_info = load_encoder(model, path, device)
        if load_info["status"] == "missing":
            LOGGER.warning(f"{dataset} | {shot_folder}: missing {variant_name} checkpoint ({path})")
            continue
        if load_info["status"] == "ok":
            encoder = load_info["encoder"]
            accuracy = evaluate_encoder_accuracy(encoder, classification_head, val_loader, device)
            LOGGER.info(f"{dataset} | {shot_folder} | {variant_name}: {accuracy:.2f}%  ({path})")
        else:
            json_path = os.path.join(shot_dir, f"{variant_name.lower()}_results.json")
            accuracy = fetch_accuracy_from_json(json_path, accuracy_keys)
            if accuracy is None:
                LOGGER.warning(
                    f"{dataset} | {shot_folder}: {variant_name} checkpoint is non-state_dict and no accuracy in {json_path}"
                )
                continue
            LOGGER.info(
                f"{dataset} | {shot_folder} | {variant_name}: {accuracy:.2f}% (loaded from {json_path})"
            )
        results.append(
            {
                "Variant": variant_name,
                "Config": config_tag or "-",
                "Training Data": format_shot_name(shot_folder),
                "Training Time (s)": training_time if training_time != "N/A" else "N/A",
                "Accuracy (%)": f"{accuracy:.2f}",
            }
        )
        added = True

    if not added:
        LOGGER.warning(f"{dataset} | {shot_folder}: no shot-specific checkpoints found; skipping summary")
        return None

    return results


def evaluate_dataset(
    dataset: str,
    dataset_dir: str,
    shots_filter: Optional[set],
    model: str,
    data_location: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info(f"Evaluating dataset: {dataset} ({dataset_dir})")
    LOGGER.info("=" * 80)

    template_encoder = ImageEncoder(model).to(device)
    val_dataset = get_remote_sensing_dataset(
        f"{dataset}Val",
        template_encoder.val_preprocess,
        location=data_location,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    args_ns = SimpleNamespace(
        save_dir=os.path.join(dataset_dir, ".."),  # for head caching
        model=model,
        device=device,
        batch_size=batch_size,
    )
    classification_head = get_remote_sensing_classification_head(args_ns, f"{dataset}Val", val_dataset)
    classification_head = classification_head.to(device)
    classification_head.eval()

    val_loader = get_dataloader(val_dataset, is_train=False, args=args_ns, image_encoder=None)

    config_shots = discover_config_shot_dirs(dataset_dir, shots_filter)
    if not config_shots:
        LOGGER.warning(f"{dataset}: no shot/config folders found; skipping.")
        return

    for config_tag, shot_folders in config_shots:
        label = config_tag or "-"
        base_results = evaluate_base_variants(
            dataset, dataset_dir, classification_head, val_loader, model, device, label
        )
        if not base_results:
            LOGGER.warning(f"{dataset}: unable to compute base results for config {label}; skipping.")
            continue
        for shot_folder in shot_folders:
            shot_results = evaluate_shot_variants(
                dataset,
                dataset_dir,
                shot_folder,
                base_results,
                classification_head,
                val_loader,
                model,
                device,
                config_tag,
            )
            if shot_results is None:
                continue
            if config_tag:
                shot_dir = os.path.join(dataset_dir, config_tag, shot_folder)
            else:
                shot_dir = os.path.join(dataset_dir, shot_folder)
            build_results_table(shot_results, shot_dir, dataset, model, shot_folder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate (zeroshot, finetuned, atlas, energy) encoders for remote sensing datasets."
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
        help="Optional list of dataset names to evaluate (without 'Val'). Evaluates all if omitted.",
    )
    parser.add_argument(
        "--shots",
        nargs="*",
        default=None,
        help="Optional list of shot folders (e.g., 16shot fullshot). Evaluates all if omitted.",
    )
    parser.add_argument("--data_location", default="./datasets")
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

    for dataset, dataset_dir in dataset_dirs:
        evaluate_dataset(
            dataset=dataset,
            dataset_dir=dataset_dir,
            shots_filter=shots_filter,
            model=args.model,
            data_location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )


if __name__ == "__main__":
    main()
