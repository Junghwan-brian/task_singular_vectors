"""
Test script to verify remote sensing datasets load correctly
Supports both single-label and multi-label datasets
"""
import os
import sys
from pathlib import Path

import torch
import math
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.datasets.remote_sensing import get_remote_sensing_dataset, REMOTE_SENSING_DATASETS
from src.datasets.remote_sensing_templates import get_remote_sensing_template


# ============================================================================
# Dataset Classification
# ============================================================================
# Keep this in sync with finetune_remote_sensing_datasets.py
# Verified by analyzing all 19 datasets' label structures

# Multi-label datasets (images can have multiple labels)
# Each image has MULTIPLE class labels
MULTILABEL_DATASETS = [
    "MultiScene",       # Parquet: 'label' column with arrays [4, 5, 6, 15, 16, 21, 22]
    "Million-AID",      # Parquet: label_1, label_2, label_3 columns (3 labels per image)
    "RSI-CB256",        # Parquet: label_1, label_2 columns (2 labels per image)
]

# Single-label datasets (images have ONE class label)
# All others are single-label by default
SINGLELABEL_DATASETS = [
    ds for ds in REMOTE_SENSING_DATASETS.keys() 
    if ds not in MULTILABEL_DATASETS
]


def get_simple_preprocess():
    """Simple preprocessing for testing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


def test_single_dataset(dataset_name, location="./datasets"):
    """Test loading a single dataset"""
    is_multilabel = dataset_name in MULTILABEL_DATASETS
    label_type = "MULTI-LABEL" if is_multilabel else "SINGLE-LABEL"
    
    print("\n" + "=" * 100)
    print(f"Testing: {dataset_name} [{label_type}]")
    print("=" * 100)
    
    try:
        # Check if dataset exists
        dataset_path = Path(location) / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset not found at {dataset_path}")
            return False
        
        # Load dataset
        preprocess = get_simple_preprocess()
        dataset = get_remote_sensing_dataset(
            dataset_name,
            preprocess=preprocess,
            location=location,
            batch_size=32,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            is_multilabel=is_multilabel,  # Pass multi-label flag
        )
        
        # Check attributes
        assert hasattr(dataset, 'train_dataset'), "Missing train_dataset"
        assert hasattr(dataset, 'test_dataset'), "Missing test_dataset"
        assert hasattr(dataset, 'train_loader'), "Missing train_loader"
        assert hasattr(dataset, 'test_loader'), "Missing test_loader"
        assert hasattr(dataset, 'classnames'), "Missing classnames"
        
        print(f"✓ Dataset structure: OK")
        
        # Detect format
        readme_path = dataset_path / "README.md"
        has_readme = readme_path.exists()
        
        # Print info
        print(f"\nDataset Information:")
        print(f"  Format: {'Parquet' if has_readme and any((dataset_path / 'data').glob('*.parquet')) else 'ImageFolder'}")
        print(f"  Train samples: {len(dataset.train_dataset)}")
        print(f"  Test samples: {len(dataset.test_dataset)}")
        print(f"  Number of classes: {len(dataset.classnames)}")
        
        # Print class names with better formatting
        print(f"\n  Class Names ({len(dataset.classnames)} total):")
        if len(dataset.classnames) <= 15:
            for i, name in enumerate(dataset.classnames):
                print(f"    {i:2d}: {name}")
        else:
            for i, name in enumerate(dataset.classnames[:10]):
                print(f"    {i:2d}: {name}")
            print(f"    ... and {len(dataset.classnames) - 10} more ...")
            for i, name in enumerate(dataset.classnames[-3:], start=len(dataset.classnames)-3):
                print(f"    {i:2d}: {name}")
        
        # Check if class names are meaningful (not just numbers)
        numeric_names = sum(1 for name in dataset.classnames if name.isdigit())
        if numeric_names > len(dataset.classnames) * 0.5:
            print(f"\n  ⚠️  WARNING: {numeric_names}/{len(dataset.classnames)} class names are numeric!")
            print(f"      This might indicate class names were not parsed correctly.")
        else:
            print(f"\n  ✓ Class names are meaningful (not just numbers)")
        
        # Test loading a batch
        print(f"\nTesting data loading...")
        train_batch = next(iter(dataset.train_loader))
        
        if isinstance(train_batch, dict):
            images = train_batch["images"]
            labels = train_batch["labels"]
        elif isinstance(train_batch, (list, tuple)):
            images, labels = train_batch
        else:
            raise ValueError(f"Unknown batch format: {type(train_batch)}")
        
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Label dtype: {labels.dtype}")
        
        assert images.shape[0] == labels.shape[0], "Batch size mismatch"
        assert len(images.shape) == 4, f"Expected 4D images, got {len(images.shape)}D"
        assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
        
        # Check label format based on type
        if is_multilabel:
            # Multi-label: expect [batch_size, num_classes]
            print(f"  Label format: Multi-label (multi-hot encoding)")
            if len(labels.shape) == 2:
                print(f"  ✓ Correct shape for multi-label: [batch_size, num_classes]")
                assert labels.shape[1] == len(dataset.classnames), \
                    f"Label dimension mismatch: {labels.shape[1]} != {len(dataset.classnames)}"
            else:
                print(f"  ⚠️  WARNING: Expected 2D labels for multi-label, got {len(labels.shape)}D")
                print(f"      This dataset might not be truly multi-label!")
        else:
            # Single-label: expect [batch_size]
            print(f"  Label format: Single-label (class indices)")
            print(f"  Labels min/max: {labels.min()}/{labels.max()}")
            if len(labels.shape) == 1:
                print(f"  ✓ Correct shape for single-label: [batch_size]")
                assert labels.max() < len(dataset.classnames), "Label out of range"
            else:
                print(f"  ⚠️  WARNING: Expected 1D labels for single-label, got {len(labels.shape)}D")
                print(f"      This dataset might be multi-label!")
        
        print(f"✓ Data loading: OK")
        
        # Show actual class names for first few samples
        print(f"\n  Sample labels in this batch:")
        num_samples_to_show = min(5, len(labels))
        
        if is_multilabel and len(labels.shape) == 2:
            # Multi-label: show all active classes
            for i in range(num_samples_to_show):
                active_classes = torch.where(labels[i] > 0.5)[0].tolist()
                class_names = [dataset.classnames[idx] for idx in active_classes]
                print(f"    Sample {i}: {len(active_classes)} classes → {class_names}")
        else:
            # Single-label: show one class
            for i in range(num_samples_to_show):
                if len(labels.shape) == 1:
                    label_idx = labels[i].item()
                    class_name = dataset.classnames[label_idx]
                    print(f"    Sample {i}: label={label_idx} → class='{class_name}'")
                else:
                    # Fallback for unexpected shape
                    print(f"    Sample {i}: labels={labels[i]}")
        
        # Test templates
        template = get_remote_sensing_template(dataset_name)
        print(f"\nText Templates ({len(template)} total):")
        example_class = dataset.classnames[0]
        for i, t in enumerate(template[:3]):
            print(f"  {i+1}. {t(example_class)}")
        if len(template) > 3:
            print(f"  ... and {len(template) - 3} more")
        
        print(f"\n✅ {dataset_name} passed all tests!")
        return True
        
    except Exception as e:
        print(f"\n❌ {dataset_name} failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_datasets(location="./datasets"):
    """Test all available datasets"""
    print("\n" + "=" * 100)
    print("Testing All Remote Sensing Datasets")
    print("=" * 100)
    
    results = {}
    available_datasets = []
    available_single = []
    available_multi = []
    
    # Check which datasets are available
    for dataset_name in sorted(REMOTE_SENSING_DATASETS.keys()):
        dataset_path = Path(location) / dataset_name
        if dataset_path.exists():
            available_datasets.append(dataset_name)
            if dataset_name in MULTILABEL_DATASETS:
                available_multi.append(dataset_name)
            else:
                available_single.append(dataset_name)
    
    print(f"\nFound {len(available_datasets)} datasets in {location}:")
    print(f"  - Single-label: {len(available_single)} datasets")
    for ds in available_single:
        print(f"    • {ds}")
    if available_multi:
        print(f"  - Multi-label: {len(available_multi)} datasets")
        for ds in available_multi:
            print(f"    • {ds}")
    
    print(f"\n{'='*100}\n")
    
    # Test each dataset
    for dataset_name in available_datasets:
        success = test_single_dataset(dataset_name, location)
        results[dataset_name] = success
    
    # Summary
    print("\n" + "=" * 100)
    print("Test Summary")
    print("=" * 100)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nResults: {passed}/{total} datasets passed")
    print(f"\n✅ Passed:")
    for ds, success in results.items():
        if success:
            print(f"  - {ds}")
    
    if passed < total:
        print(f"\n❌ Failed:")
        for ds, success in results.items():
            if not success:
                print(f"  - {ds}")
    
    print("\n" + "=" * 100)
    
    return results



def _extract_image_and_label(sample):
    if isinstance(sample, dict):
        img = sample.get("images") or sample.get("image")
        label = sample.get("labels") or sample.get("label")
        return img, label
    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        return sample[0], sample[1]
    raise ValueError(f"Unsupported sample format: {type(sample)}")


def _tensor_to_uint8_image(img):
    if torch.is_tensor(img):
        img_np = img.detach().cpu()
        if img_np.ndim == 3 and img_np.shape[0] in {1, 3}:
            img_np = img_np.permute(1, 2, 0)
        img_np = img_np.numpy()
    elif isinstance(img, Image.Image):
        img_np = np.array(img)
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        return None
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0.0, 1.0)
        img_np = (img_np * 255).astype(np.uint8)
    return img_np


def _visualize_single_label_dataset(dataset_name, rs_dataset):
    splits = []
    for split_name in ["train_dataset", "test_dataset", "val_dataset"]:
        split = getattr(rs_dataset, split_name, None)
        if split is not None:
            splits.append((split_name.replace("_dataset", ""), split))
    if not splits:
        splits.append(("dataset", rs_dataset))

    class_to_image = {}
    num_classes = len(rs_dataset.classnames)
    print(f"Collecting one sample per class for {dataset_name} ({num_classes} classes)...")

    for split_name, split in splits:
        print(f"  Traversing {split_name} split (size={len(split)})")
        for idx in range(len(split)):
            try:
                sample = split[idx]
                img, label = _extract_image_and_label(sample)
            except Exception:
                continue

            if img is None or label is None:
                continue

            if torch.is_tensor(label):
                if label.ndim != 0:
                    continue
                label_idx = int(label.item())
            elif isinstance(label, (int, np.integer)):
                label_idx = int(label)
            else:
                continue

            if label_idx not in class_to_image:
                img_np = _tensor_to_uint8_image(img)
                if img_np is not None:
                    class_to_image[label_idx] = img_np

            if len(class_to_image) == num_classes:
                break
        if len(class_to_image) == num_classes:
            break

    output_dir = Path("./test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_all_classes.png"

    cols = min(6, max(3, int(math.ceil(math.sqrt(num_classes)))))
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    flat_axes = axes.flatten()

    for class_idx in range(num_classes):
        ax = flat_axes[class_idx]
        img_np = class_to_image.get(class_idx)
        if img_np is None:
            ax.axis("off")
            continue
        ax.imshow(img_np)
        title = f"[{class_idx}] {rs_dataset.classnames[class_idx]}"
        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.axis("off")

    for pad_idx in range(num_classes, len(flat_axes)):
        flat_axes[pad_idx].axis("off")

    missing = sorted(set(range(num_classes)) - set(class_to_image.keys()))
    if missing:
        missing_names = [rs_dataset.classnames[i] for i in missing]
        print(f"⚠️  Missing samples for classes: {missing_names}")
    print(f"Collected samples for {len(class_to_image)}/{num_classes} classes")

    fig.suptitle(f"{dataset_name} [SINGLE-LABEL] - One Sample per Class", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved visualization to {output_file}")


def visualize_samples(dataset_name, location="./datasets", num_samples=9):
    """Visualize sample images from a dataset"""
    is_multilabel = dataset_name in MULTILABEL_DATASETS
    label_type = "MULTI-LABEL" if is_multilabel else "SINGLE-LABEL"

    print("\n" + "=" * 100)
    print(f"Visualizing samples from {dataset_name} [{label_type}]")
    print("=" * 100 + "\n")

    try:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = get_remote_sensing_dataset(
            dataset_name,
            preprocess=preprocess,
            location=location,
            batch_size=num_samples,
            num_workers=0,
            is_multilabel=is_multilabel,
        )

        print(f"Dataset loaded with {len(dataset.classnames)} classes")
        print(f"Class names preview: {dataset.classnames[:5]}...")
        print(f"Label type: {label_type}")

        numeric_names = sum(1 for name in dataset.classnames if name.isdigit())
        if numeric_names > len(dataset.classnames) * 0.5:
            print("⚠️  WARNING: Most class names are numeric. This might be an issue!")
        else:
            print("✓ Class names are meaningful")

        if not is_multilabel:
            _visualize_single_label_dataset(dataset_name, dataset)
            return

        batch = next(iter(dataset.train_loader))
        images, labels = _extract_image_and_label(batch)
        if images is None or labels is None:
            raise ValueError("Could not obtain batch for visualization")

        print(f"Label shape: {labels.shape}")
        if len(labels.shape) != 2:
            print(f"⚠️  WARNING: Expected 2D labels for multi-label, got {len(labels.shape)}D")

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        subtitle = f"{dataset_name} [{label_type}] - Sample Images"
        fig.suptitle(subtitle, fontsize=16, fontweight='bold')

        for idx in range(min(num_samples, len(images))):
            ax = axes[idx // 3, idx % 3]
            img_np = _tensor_to_uint8_image(images[idx])
            if img_np is None:
                ax.axis('off')
                continue
            ax.imshow(img_np)

            active_classes = torch.where(labels[idx] > 0.5)[0].tolist()
            if len(active_classes) == 0:
                title = "No labels"
            elif len(active_classes) <= 3:
                class_names = [dataset.classnames[i] for i in active_classes]
                title = "\n".join(class_names)
            else:
                class_names = [dataset.classnames[i] for i in active_classes[:2]]
                title = f"{class_names[0]}\n{class_names[1]}\n+{len(active_classes)-2} more"

            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(2)

        plt.tight_layout()
        output_dir = Path("./test_outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{dataset_name}_samples.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_file}")

        print(f"\nClasses shown in visualization:")
        for idx in range(min(num_samples, len(images))):
            active_classes = torch.where(labels[idx] > 0.5)[0].tolist()
            class_names = [dataset.classnames[i] for i in active_classes]
            indices_str = ", ".join([str(i) for i in active_classes])
            print(f"  Sample {idx}: [{indices_str}] {class_names}")

        plt.close()

    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test remote sensing datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Test a specific dataset (default: test all)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="./datasets",
        help="Location of datasets directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize sample images",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Remote Sensing Datasets:")
        print("=" * 100)
        
        # Count datasets
        single_count = 0
        multi_count = 0
        found_count = 0
        
        print("\n[SINGLE-LABEL DATASETS]")
        for ds in sorted(SINGLELABEL_DATASETS):
            dataset_path = Path(args.location) / ds
            exists = dataset_path.exists()
            status = "✓ Found" if exists else "✗ Not found"
            print(f"  {ds:30s} {status:15s} {dataset_path}")
            if exists:
                single_count += 1
                found_count += 1
        
        if MULTILABEL_DATASETS:
            print("\n[MULTI-LABEL DATASETS]")
            for ds in sorted(MULTILABEL_DATASETS):
                dataset_path = Path(args.location) / ds
                exists = dataset_path.exists()
                status = "✓ Found" if exists else "✗ Not found"
                print(f"  {ds:30s} {status:15s} {dataset_path}")
                if exists:
                    multi_count += 1
                    found_count += 1
        
        print("\n" + "=" * 100)
        print(f"Summary: {found_count} datasets found")
        print(f"  - Single-label: {single_count}")
        if MULTILABEL_DATASETS:
            print(f"  - Multi-label: {multi_count}")
        print("=" * 100)
        sys.exit(0)
    
    if args.dataset:
        # Test single dataset
        visualize_samples(args.dataset, args.location)
        success = test_single_dataset(args.dataset, args.location)
        
        
        sys.exit(0 if success else 1)
    else:
        # Test all datasets
        results = test_all_datasets(args.location)
        
        if args.visualize:
            print("\n" + "=" * 100)
            print("Generating Visualizations")
            print("=" * 100)
            for dataset_name, success in results.items():
                if success:
                    visualize_samples(dataset_name, args.location)
        
        # Exit with error if any tests failed
        sys.exit(0 if all(results.values()) else 1)