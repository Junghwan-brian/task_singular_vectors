"""
Remote Sensing Dataset Loaders
Supports both ImageFolder format and Parquet format datasets
"""
import os
import re
from pathlib import Path
from typing import Optional, Callable, Dict
from PIL import Image
import io

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


def clean_dataset_logs(log_filename, dataset_name):
    """
    Remove previous logs for a specific dataset from the log file.
    This allows re-running a dataset without deleting logs from other datasets.
    
    Args:
        log_filename: Path to the log file
        dataset_name: Name of the dataset (e.g., 'AIDVal')
    """
    if not os.path.exists(log_filename):
        return
    
    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
        
        # Filter out lines containing this dataset
        filtered_lines = [line for line in lines if f"'dataset': '{dataset_name}'" not in line]
        
        with open(log_filename, 'w') as f:
            f.writelines(filtered_lines)
        
        print(f"Cleaned previous logs for {dataset_name} from {log_filename}")
    except Exception as e:
        print(f"Warning: Could not clean logs for {dataset_name}: {e}")


def sample_k_shot_indices(dataset, k, seed=0, verbose=True):
    """
    Unified k-shot sampling function for remote sensing datasets.
    
    Samples exactly k examples per class from the dataset.
    Uses numpy random seed for reproducibility and consistent results
    across different runs and scripts.
    
    Args:
        dataset: Dataset object (should have train_dataset attribute or be a dataset itself)
        k: Number of samples per class
        seed: Random seed for reproducibility (default: 0)
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        List of indices for the k-shot subset
    """
    # Get the base dataset
    if hasattr(dataset, 'train_dataset'):
        base_dataset = dataset.train_dataset
    else:
        base_dataset = dataset
    
    # Extract labels from dataset
    if hasattr(base_dataset, 'targets'):
        # torchvision.datasets.ImageFolder style
        labels = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        # Custom dataset with labels attribute
        labels = np.array(base_dataset.labels)
    elif hasattr(base_dataset, 'data'):
        # Some datasets store (data, labels) tuples
        try:
            labels = np.array([item[1] for item in base_dataset.data])
        except:
            # Fallback: iterate through dataset
            labels = []
            for i in range(len(base_dataset)):
                try:
                    _, label = base_dataset[i]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    labels.append(label)
                except:
                    pass
            labels = np.array(labels)
    else:
        # Fallback: iterate through dataset
        if verbose:
            print("Extracting labels from dataset by iteration...")
        labels = []
        for i in range(len(base_dataset)):
            try:
                _, label = base_dataset[i]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to get label for index {i}: {e}")
                continue
        labels = np.array(labels)
    
    if len(labels) == 0:
        raise ValueError("Could not extract labels from dataset")
    
    # Get unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if verbose:
        print(f"Found {num_classes} classes in dataset")
        print(f"Sampling {k} examples per class (seed={seed})...")
    
    # Set numpy random seed for reproducibility
    np.random.seed(seed)
    selected_indices = []
    class_sample_counts = {}
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        
        if len(cls_indices) < k:
            if verbose:
                print(f"  Class {cls}: has only {len(cls_indices)} samples (requested {k}), using all")
            selected_indices.extend(cls_indices.tolist())
            class_sample_counts[int(cls)] = len(cls_indices)
        else:
            # Randomly sample k indices from this class using np.random.choice
            sampled = np.random.choice(cls_indices, size=k, replace=False)
            selected_indices.extend(sampled.tolist())
            class_sample_counts[int(cls)] = k
    
    # Print summary of samples per class
    if verbose:
        print(f"\nK-shot sampling summary:")
        print(f"  Total classes: {num_classes}")
        print(f"  Requested samples per class: {k}")
        print(f"  Total selected samples: {len(selected_indices)}")
        print(f"\n  Samples per class:")
        for cls in sorted(class_sample_counts.keys()):
            count = class_sample_counts[cls]
            status = "✓" if count == k else f"⚠ (only {count})"
            print(f"    Class {cls:3d}: {count:3d} samples {status}")
    
    return selected_indices


def eval_single_dataset_remote_sensing(image_encoder, dataset_name, args):
    """
    Evaluate model on a single remote sensing dataset
    
    Same as eval_single_dataset() but uses get_remote_sensing_dataset()
    
    Args:
        image_encoder: Image encoder to evaluate
        dataset_name: Name of the dataset (should end with "Val")
        args: Arguments with data_location, batch_size, device
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    import time
    import torch
    from src.datasets.common import get_dataloader, maybe_dictionarize
    from src.models.modeling import ImageClassifier
    from src.utils import utils
    
    start_time = time.time()
    
    # Get classification head
    classification_head = get_remote_sensing_classification_head(args, dataset_name, None)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()
    
    # Load dataset
    dataset = get_remote_sensing_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    
    # Get test loader (is_train=False → dataset.test_loader)
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device
    
    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            
            logits = utils.get_logits(x, model)
            
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        
        top1 = correct / n
    
    metrics = {"top1": top1}
    dt = time.time() - start_time
    print(
        f"Done evaluating on {dataset_name}.\t Accuracy: {100*top1:.2f}%.\t Total time: {dt:.2f}s"
    )
    
    return metrics


def get_remote_sensing_classification_head(args, dataset_name, dataset_obj):
    """
    Build or load classification head for remote sensing dataset
    
    Args:
        args: Arguments object with model, save_dir, device attributes
        dataset_name: Name of the dataset (with 'Val' suffix)
        dataset_obj: Dataset object with classnames attribute
    
    Returns:
        ClassificationHead module
    """
    from src.models import ImageEncoder
    from src.models.modeling import ClassificationHead
    from src.datasets.remote_sensing_templates import get_remote_sensing_template
    import open_clip
    from tqdm import tqdm
    from src.utils.variables_and_paths import TQDM_BAR_FORMAT
    
    # Ensure 'Val' suffix
    if not dataset_name.endswith("Val"):
        dataset_name += "Val"
    
    filename = os.path.join(args.save_dir, f"head_{dataset_name}.pt")
    
    # Try to load existing head
    if os.path.exists(filename):
        print(f"Loading classification head for {args.model} on {dataset_name} from {filename}")
        return ClassificationHead.load(filename)
    
    print(f"Did not find classification head for {args.model} on {dataset_name} at {filename}")
    print(f"Building one from scratch...")
    
    # Build new head using CLIP text encoder
    model = ImageEncoder(args.model, keep_lang=True).model
    template = get_remote_sensing_template(dataset_name)
    
    logit_scale = model.logit_scale
    model.eval()
    model.to(args.device)
    
    print(f"Building classification head for {len(dataset_obj.classnames)} classes...")
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset_obj.classnames, bar_format=TQDM_BAR_FORMAT):
            # Clean class name (replace underscores, etc.)
            clean_name = classname.replace("_", " ").replace("&", " and ")
            
            texts = []
            for t in template:
                texts.append(t(clean_name))
            
            texts = open_clip.tokenize(texts).to(args.device)
            embeddings = model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            
            zeroshot_weights.append(embeddings)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(args.device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        
        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    
    print(f"Classification head shape: {zeroshot_weights.shape}")
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    
    # Save head
    os.makedirs(args.save_dir, exist_ok=True)
    classification_head.save(filename)
    print(f"Saved classification head to {filename}")
    
    return classification_head


def parse_class_names_from_readme(readme_path: Path) -> Dict[int, str]:
    """
    Parse class names from README.md file
    
    Args:
        readme_path: Path to README.md file
    
    Returns:
        Dictionary mapping label index to class name
    """
    if not readme_path.exists():
        return {}
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the class_label section using regex
        # Pattern: '0': class_name or "0": class_name
        pattern = r"['\"](\d+)['\"]:\s*(.+?)(?:\n|$)"
        matches = re.findall(pattern, content)
        
        if not matches:
            return {}
        
        # Create mapping
        class_names = {}
        for idx_str, name in matches:
            idx = int(idx_str)
            # Clean up the class name (remove quotes, extra spaces)
            name = name.strip().strip('"').strip("'")
            class_names[idx] = name
        
        return class_names
    except Exception as e:
        print(f"Warning: Could not parse README.md: {e}")
        return {}


class ParquetImageDataset(Dataset):
    """Dataset for loading images from parquet files"""
    
    def __init__(self, parquet_files, transform=None, readme_path=None, is_multilabel=False):
        """
        Args:
            parquet_files: List of parquet file paths or single file path
            transform: Optional transform to be applied on a sample
            readme_path: Optional path to README.md for class name mapping
            is_multilabel: If True, support multi-label classification
        """
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pandas and pyarrow are required for parquet support. "
                "Install with: pip install pandas pyarrow"
            )
        
        self.transform = transform
        self.is_multilabel = is_multilabel
        
        # Load parquet files
        if isinstance(parquet_files, (str, Path)):
            parquet_files = [parquet_files]
        
        dfs = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} samples from {len(parquet_files)} parquet file(s)")
        
        # Check columns
        if 'image' not in self.data.columns:
            raise ValueError(f"Parquet must have 'image' column. Found: {self.data.columns.tolist()}")
        
        # Check for different label column formats
        if 'label' in self.data.columns:
            # Standard format: single 'label' column
            self.labels = self.data['label'].values
            self.label_columns = ['label']
        elif 'label_1' in self.data.columns:
            # Multi-column format: label_1, label_2, label_3, ...
            # This is typically multi-label format (e.g., Million-AID, RSI-CB256)
            label_cols = [col for col in self.data.columns if col.startswith('label_')]
            label_cols.sort()  # label_1, label_2, label_3, ...
            print(f"Found multi-column labels: {label_cols}")
            
            # For multi-label datasets with multiple label columns
            if not self.is_multilabel:
                print(f"Warning: Dataset has {len(label_cols)} label columns but is_multilabel=False")
                print(f"         Setting is_multilabel=True automatically")
                self.is_multilabel = True
            
            # Combine all label columns into lists
            self.labels = self.data[label_cols].values.tolist()
            self.label_columns = label_cols
        else:
            raise ValueError(f"No label columns found. Available columns: {self.data.columns.tolist()}")
        
        # Extract unique labels
        if isinstance(self.labels, list) and len(self.labels) > 0:
            # Already a list (from multi-column format)
            pass
        else:
            # Convert to list if needed
            self.labels = list(self.labels)
        
        # Handle multi-label case
        if self.is_multilabel:
            # For multi-label, labels could be lists or arrays
            # Collect all unique class IDs
            all_classes = set()
            for label in self.labels:
                if isinstance(label, (list, tuple)):
                    all_classes.update(label)
                elif isinstance(label, int):
                    # Single label stored as int, treat as list with one element
                    all_classes.add(label)
                else:
                    # Try to convert to list
                    try:
                        label_list = list(label)
                        all_classes.update(label_list)
                    except:
                        all_classes.add(label)
            self.unique_labels = sorted(all_classes)
        else:
            # Single-label: standard behavior
            self.unique_labels = sorted(self.data['label'].unique())
        
        # Try to load class names from README
        self.class_name_mapping = {}
        if readme_path and Path(readme_path).exists():
            self.class_name_mapping = parse_class_names_from_readme(Path(readme_path))
            if self.class_name_mapping:
                print(f"Loaded {len(self.class_name_mapping)} class names from README.md")
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
        # Map labels to indices
        if self.is_multilabel:
            # For multi-label, store as-is (will be converted in __getitem__)
            self.targets = self.labels
        else:
            # For single-label, convert to indices
            self.targets = [self.label_to_idx[label] for label in self.labels]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_data = row['image']
        
        # Handle different image formats
        if isinstance(image_data, dict):
            # Hugging Face datasets format: {'bytes': b'...', 'path': None}
            if 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            else:
                raise ValueError(f"Unknown image dict format: {image_data.keys()}")
        elif isinstance(image_data, bytes):
            # Direct bytes
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            raise ValueError(f"Unknown image format: {type(image_data)}")
        
        # Handle label
        if self.is_multilabel:
            # Multi-label: create multi-hot encoding
            label_data = self.targets[idx]
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            
            # Convert label_data to list of label IDs
            if isinstance(label_data, (list, tuple)):
                label_ids = label_data
            elif isinstance(label_data, int):
                label_ids = [label_data]
            else:
                # Try to convert to list
                try:
                    label_ids = list(label_data)
                except:
                    label_ids = [label_data]
            
            # Set corresponding indices to 1
            for label_id in label_ids:
                if label_id in self.label_to_idx:
                    class_idx = self.label_to_idx[label_id]
                    multi_hot[class_idx] = 1.0
            
            label = multi_hot
        else:
            # Single-label: return class index
            label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        """Return list of class names in order"""
        class_names = []
        for label in self.unique_labels:
            # Try to get actual class name from mapping
            if self.class_name_mapping and label in self.class_name_mapping:
                class_names.append(self.class_name_mapping[label])
            else:
                # Fallback to label number as string
                class_names.append(str(label))
        return class_names


class RemoteSensingDataset:
    """
    Generic Remote Sensing Dataset Loader
    
    Automatically detects format:
    - ImageFolder format (directories with class names)
    - Parquet format (train/test parquet files)
    
    Required structure:
    For ImageFolder:
        dataset_root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                ...
    
    For Parquet:
        dataset_root/
            train-*.parquet
            test-*.parquet (or validation-*.parquet)
    
    Returns object with:
        - self.train_dataset
        - self.train_loader
        - self.test_dataset
        - self.test_loader
        - self.classnames
    """
    
    def __init__(
        self,
        dataset_name: str,
        preprocess: Callable,
        location: str = "./datasets",
        batch_size: int = 128,
        num_workers: int = 6,
        train_split: str = "train",
        test_split: str = "test",
        is_multilabel: bool = False,
    ):
        """
        Args:
            dataset_name: Name of the dataset folder
            preprocess: Transform function for images
            location: Root directory containing datasets
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            train_split: Name of train split ('train', 'data', etc.)
            test_split: Name of test split ('test', 'validation', etc.)
            is_multilabel: If True, support multi-label classification
        """
        self.dataset_name = dataset_name
        self.dataset_root = Path(location) / dataset_name
        self.is_multilabel = is_multilabel
        
        if not self.dataset_root.exists():
            raise ValueError(f"Dataset not found at {self.dataset_root}")
        
        print(f"Loading dataset: {dataset_name} from {self.dataset_root}")
        if self.is_multilabel:
            print(f"  Multi-label mode: ON")
        
        # Detect format
        self.format = self._detect_format()
        print(f"Detected format: {self.format}")
        
        # Load data
        if self.format == "imagefolder":
            self._load_imagefolder(preprocess, train_split, test_split)
        elif self.format == "parquet":
            self._load_parquet(preprocess, train_split, test_split)
        else:
            raise ValueError(f"Unknown format: {self.format}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        print(f"Dataset loaded successfully!")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Classes: {len(self.classnames)}")
        print(f"  Class names: {self.classnames[:5]}..." if len(self.classnames) > 5 else f"  Class names: {self.classnames}")
    
    def _has_image_files(self, directory):
        """Check if a directory contains image files (supports multiple formats)"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.TIF', '*.TIFF', '*.BMP']
        for ext in image_extensions:
            if list(directory.glob(ext)):
                return True
        return False
    
    def _detect_format(self):
        """Detect dataset format"""
        # Check for parquet files in data/ subdirectory
        data_dir = self.dataset_root / "data"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                return "parquet"
        
        # Check for parquet files in root
        parquet_files = list(self.dataset_root.glob("*.parquet"))
        if parquet_files:
            return "parquet"
        
        # Check for Images/ subdirectory (MLRSNet format)
        images_dir = self.dataset_root / "Images"
        if images_dir.exists() and images_dir.is_dir():
            subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
            if subdirs:
                return "imagefolder"
        
        # Check for data/ subdirectory with class folders (AID format)
        data_dir = self.dataset_root / "data"
        if data_dir.exists() and data_dir.is_dir():
            subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if subdirs:
                # Check if subdirs contain images
                first_subdir = subdirs[0]
                if self._has_image_files(first_subdir):
                    return "imagefolder"
        
        # Check for class folders in root (RSI-CB128 format)
        subdirs = [d for d in self.dataset_root.iterdir() if d.is_dir() and d.name != "data"]
        if subdirs:
            # Check if subdirs contain images
            first_subdir = subdirs[0]
            if self._has_image_files(first_subdir):
                return "imagefolder"
        
        raise ValueError(f"Cannot detect format for {self.dataset_root}")
    
    def _load_imagefolder(self, preprocess, train_split, test_split):
        """Load ImageFolder format dataset"""
        # Find the directory containing class folders
        possible_dirs = [
            self.dataset_root / "Images",  # MLRSNet
            self.dataset_root / "data",     # AID
            self.dataset_root,              # RSI-CB128 (root level)
        ]
        
        image_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                if subdirs:
                    # Verify it contains images
                    first_subdir = subdirs[0]
                    if self._has_image_files(first_subdir):
                        image_dir = dir_path
                        break
        
        if image_dir is None:
            raise ValueError(f"Cannot find image directory in {self.dataset_root}")
        
        print(f"Loading ImageFolder from: {image_dir}")
        
        # Load all data
        full_dataset = ImageFolder(root=image_dir, transform=preprocess)
        self.classnames = full_dataset.classes
        
        # Split into train/test (80/20 split)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        test_size = total_size - train_size
        
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )
        
        print(f"Split dataset: {train_size} train, {test_size} test")
    
    def _load_parquet(self, preprocess, train_split, test_split):
        """Load Parquet format dataset"""
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            data_dir = self.dataset_root
        
        # Look for README.md for class name mapping
        readme_path = self.dataset_root / "README.md"
        
        # Find train files
        train_files = sorted(data_dir.glob(f"{train_split}*.parquet"))
        if not train_files:
            raise ValueError(f"No {train_split} parquet files found in {data_dir}")
        
        print(f"Found {len(train_files)} train parquet file(s)")
        
        # Find test files
        test_files = sorted(data_dir.glob(f"{test_split}*.parquet"))
        if not test_files:
            # Try 'validation' if 'test' not found
            test_files = sorted(data_dir.glob("validation*.parquet"))
        
        if not test_files:
            print(f"Warning: No test/validation files found. Using 20% of train for test.")
            # Load all train data and split
            full_dataset = ParquetImageDataset(
                train_files, 
                transform=preprocess,
                readme_path=readme_path,
                is_multilabel=self.is_multilabel
            )
            
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            test_size = total_size - train_size
            
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=generator
            )
            
            self.classnames = full_dataset.get_class_names()
        else:
            print(f"Found {len(test_files)} test parquet file(s)")
            
            # Load train and test separately
            self.train_dataset = ParquetImageDataset(
                train_files, 
                transform=preprocess,
                readme_path=readme_path,
                is_multilabel=self.is_multilabel
            )
            self.test_dataset = ParquetImageDataset(
                test_files, 
                transform=preprocess,
                readme_path=readme_path,
                is_multilabel=self.is_multilabel
            )
            
            self.classnames = self.train_dataset.get_class_names()


# Dataset registry
REMOTE_SENSING_DATASETS = {
    "AID": RemoteSensingDataset,
    "CLRS": RemoteSensingDataset,
    "EuroSAT_RGB": RemoteSensingDataset,
    "Million-AID": RemoteSensingDataset,
    "MLRSNet": RemoteSensingDataset,
    "MultiScene": RemoteSensingDataset,
    "NWPU-RESISC45": RemoteSensingDataset,
    "Optimal-31": RemoteSensingDataset,
    "PatternNet": RemoteSensingDataset,
    "RS_C11": RemoteSensingDataset,
    "RSD46-WHU": RemoteSensingDataset,
    "RSI-CB128": RemoteSensingDataset,
    "RSI-CB256": RemoteSensingDataset,
    "RSSCN7": RemoteSensingDataset,
    "SAT-4": RemoteSensingDataset,
    "SAT-6": RemoteSensingDataset,
    "SIRI-WHU": RemoteSensingDataset,
    "UC_Merced": RemoteSensingDataset,
    "WHU-RS19": RemoteSensingDataset,
}


def split_remote_sensing_train_into_train_val(
    dataset,
    val_fraction=0.1,
    max_val_samples=5000,
    seed=0,
):
    """
    Split train dataset into train and validation sets
    Compatible with the standard get_dataset() Val split logic
    
    Args:
        dataset: RemoteSensingDataset object
        val_fraction: Fraction of train data to use for validation
        max_val_samples: Maximum number of validation samples
        seed: Random seed for reproducibility
    
    Returns:
        Modified dataset with train/val split
    """
    import copy
    from torch.utils.data import random_split, DataLoader
    
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size
    
    assert val_size > 0
    assert train_size > 0
    
    lengths = [train_size, val_size]
    
    trainset, valset = random_split(
        dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Split train dataset: {train_size} train, {val_size} val (seed={seed})")
    
    # Update dataset
    dataset.train_dataset = trainset
    dataset.train_loader = DataLoader(
        dataset.train_dataset,
        shuffle=True,
        batch_size=dataset.train_loader.batch_size,
        num_workers=dataset.train_loader.num_workers,
        pin_memory=True,
    )
    
    dataset.test_dataset = valset
    dataset.test_loader = DataLoader(
        dataset.test_dataset,
        batch_size=dataset.test_loader.batch_size,
        num_workers=dataset.test_loader.num_workers,
        pin_memory=True,
    )
    
    return dataset


def get_remote_sensing_dataset(
    dataset_name, 
    preprocess, 
    location="./datasets", 
    batch_size=128,
    num_workers=6,
    val_fraction=0.1,
    max_val_samples=5000,
    **kwargs
):
    """
    Get a remote sensing dataset
    
    Compatible with standard get_dataset() behavior:
    - If dataset_name ends with "Val", splits train data into train/val
    - Uses fixed seed (0) for reproducible splits
    
    Args:
        dataset_name: Name of the dataset (e.g., "AID" or "AIDVal")
        preprocess: Transform function
        location: Root directory containing datasets
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        val_fraction: Fraction of train data for validation (if Val suffix)
        max_val_samples: Maximum validation samples (if Val suffix)
        **kwargs: Additional arguments for dataset
    
    Returns:
        Dataset object with train_dataset, test_dataset, train_loader, test_loader, classnames
    """
    # Handle Val suffix (same logic as get_dataset())
    if dataset_name.endswith("Val"):
        base_dataset_name = dataset_name.replace("Val", "")
        print(f"Loading base dataset '{base_dataset_name}' and splitting into train/val...")
        
        # Load base dataset
        if base_dataset_name not in REMOTE_SENSING_DATASETS:
            raise ValueError(f"Unknown dataset: {base_dataset_name}. Available: {list(REMOTE_SENSING_DATASETS.keys())}")
        
        dataset_class = REMOTE_SENSING_DATASETS[base_dataset_name]
        dataset = dataset_class(
            base_dataset_name, 
            preprocess, 
            location, 
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        # Split train into train/val with fixed seed for reproducibility
        dataset = split_remote_sensing_train_into_train_val(
            dataset,
            val_fraction=val_fraction,
            max_val_samples=max_val_samples,
            seed=0,  # Fixed seed for reproducibility across all datasets
        )
        
        return dataset
    else:
        # Load dataset without Val split
        if dataset_name not in REMOTE_SENSING_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(REMOTE_SENSING_DATASETS.keys())}")
        
        dataset_class = REMOTE_SENSING_DATASETS[dataset_name]
        return dataset_class(
            dataset_name, 
            preprocess, 
            location, 
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

