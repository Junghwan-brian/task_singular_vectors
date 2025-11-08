import os
import csv
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image

from .common import ImageFolderWithPaths
from .imagenet import imagenet_classnames


def _read_synset_to_index(base_dir: str) -> Dict[str, int]:
    """
    Read LOC_synset_mapping.txt and build synset -> index mapping (0..999) in ImageNet order.
    Expects file at: {base_dir}/imagenet/LOC_synset_mapping.txt
    """
    mapping_file = os.path.join(base_dir, "imagenet", "LOC_synset_mapping.txt")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Missing synset mapping file: {mapping_file}")

    synset_to_idx = {}
    with open(mapping_file, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Each line: nXXXXXXXX description...
            synset = line.split(" ")[0]
            synset_to_idx[synset] = idx
    if len(synset_to_idx) != 1000:
        # Not fatal, but warn
        print(
            f"[warn] Expected 1000 synsets, found {len(synset_to_idx)} in {mapping_file}")
    return synset_to_idx


def _remap_imagefolder_labels_to_imagenet_indices(ds: ImageFolderWithPaths, synset_to_idx: Dict[str, int]):
    """
    Remap an ImageFolder dataset's labels to global ImageNet indices using folder synsets.
    """
    # ImageFolder stores dataset.samples as (path, target), where target is based on its own class_to_idx
    # We replace target using the folder (class) name mapped by synset_to_idx.
    class_to_idx_local = ds.class_to_idx  # e.g., {'n01440764': 0, ...}
    idx_to_class_local = {v: k for k, v in class_to_idx_local.items()}

    new_samples = []
    new_targets = []
    for path, local_target in ds.samples:
        synset = idx_to_class_local[local_target]
        global_idx = synset_to_idx.get(synset)
        if global_idx is None:
            raise KeyError(
                f"Synset '{synset}' not found in synset_to_idx mapping")
        new_samples.append((path, global_idx))
        new_targets.append(global_idx)

    ds.samples = new_samples
    ds.targets = new_targets


class _ImageNetValFromCSV(Dataset):
    """
    ImageNet validation set using flat folder and LOC_val_solution.csv for labels.
    Expects:
      - images at {base_dir}/imagenet/ILSVRC/Data/CLS-LOC/val/*.JPEG
      - labels at  {base_dir}/imagenet/LOC_val_solution.csv (first synset used)
    """

    def __init__(self, base_dir: str, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.img_dir = os.path.join(
            base_dir, "imagenet", "ILSVRC", "Data", "CLS-LOC", "val")
        self.csv_path = os.path.join(
            base_dir, "imagenet", "LOC_val_solution.csv")
        self.synset_to_idx = _read_synset_to_index(base_dir)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(
                f"ImageNet val directory not found: {self.img_dir}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"ImageNet val csv not found: {self.csv_path}")

        img_to_synset: Dict[str, str] = {}
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                # Row: ImageId, PredictionString
                # Use the first synset in PredictionString
                if len(row) < 2:
                    continue
                image_id = row[0].strip()  # e.g., ILSVRC2012_val_00000001
                pred_str = row[1].strip()
                if not pred_str:
                    continue
                first_synset = pred_str.split(" ")[0]
                img_to_synset[image_id + ".JPEG"] = first_synset

        # Build samples
        self.samples: List[Tuple[str, int]] = []
        for fname in os.listdir(self.img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            synset = img_to_synset.get(fname)
            if synset is None:
                # Skip files without label mapping
                continue
            idx = self.synset_to_idx.get(synset)
            if idx is None:
                continue
            self.samples.append((os.path.join(self.img_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class _ImageNetV2MFVal(Dataset):
    """
    ImageNetV2 matched-frequency format validation set.
    Expects directories 0..999 each containing images at:
      {base_dir}/imagenetv2-matched-frequency-format-val/{class_idx}/*.jpeg
    Labels are numeric directory names (0..999), matching ImageNet order.
    """

    def __init__(self, base_dir: str, transform=None):
        self.transform = transform
        self.root = os.path.join(
            base_dir, "imagenetv2-matched-frequency-format-val")
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"ImageNetV2-MF root not found: {self.root}")

        self.samples: List[Tuple[str, int]] = []
        # Iterate numeric dirs in numeric order
        for d in sorted(os.listdir(self.root), key=lambda x: int(x) if x.isdigit() else 1e9):
            if not d.isdigit():
                continue
            label = int(d)
            class_dir = os.path.join(self.root, d)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class ImageNetILSVRC:
    """
    ImageNet in official ILSVRC structure for training split.
    Train: {location}/imagenet/ILSVRC/Data/CLS-LOC/train/{synset}/... .
    Test: not used (prefer ImageNetILSVRCVal for evaluation).
    """

    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        train_dir = os.path.join(location, "imagenet",
                                 "ILSVRC", "Data", "CLS-LOC", "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet train directory not found: {train_dir}")

        synset_to_idx = _read_synset_to_index(location)
        self.train_dataset = ImageFolderWithPaths(
            train_dir, transform=self.preprocess)
        _remap_imagefolder_labels_to_imagenet_indices(
            self.train_dataset, synset_to_idx)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # For compatibility, set test to empty copy (not used)
        self.test_dataset = self.train_dataset
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageNetILSVRCVal:
    """
    Official ImageNet validation split using LOC_val_solution.csv labels
    """

    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = _ImageNetValFromCSV(
            location, transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageNetA:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=128, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        root = os.path.join(location, "imagenet-a", "imagenet-a")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"ImageNet-A root not found: {root}")
        synset_to_idx = _read_synset_to_index(location)

        self.train_dataset = ImageFolderWithPaths(
            root, transform=self.preprocess)
        _remap_imagefolder_labels_to_imagenet_indices(
            self.train_dataset, synset_to_idx)
        self.test_dataset = self.train_dataset

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageNetR:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=128, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        root = os.path.join(location, "imagenet-r", "imagenet-r")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"ImageNet-R root not found: {root}")
        synset_to_idx = _read_synset_to_index(location)

        self.train_dataset = ImageFolderWithPaths(
            root, transform=self.preprocess)
        _remap_imagefolder_labels_to_imagenet_indices(
            self.train_dataset, synset_to_idx)
        self.test_dataset = self.train_dataset

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageNetSketch:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=128, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        root = os.path.join(location, "ImageNet-Sketch", "sketch")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"ImageNet-Sketch root not found: {root}")
        synset_to_idx = _read_synset_to_index(location)

        self.train_dataset = ImageFolderWithPaths(
            root, transform=self.preprocess)
        _remap_imagefolder_labels_to_imagenet_indices(
            self.train_dataset, synset_to_idx)
        self.test_dataset = self.train_dataset

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ImageNetV2MFVal:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=128, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = _ImageNetV2MFVal(
            location, transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
