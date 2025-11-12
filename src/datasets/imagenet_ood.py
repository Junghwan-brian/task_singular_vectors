import os
import csv
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from .common import ImageFolderWithPaths
from .imagenet import imagenet_classnames


def _read_synset_to_index(base_dir: str) -> Dict[str, int]:
    """
    Read synset -> alphabetical index mapping (0..999).
    
    Note: ImageNet classnames are in alphabetical synset order, so we must map
    each synset (WNID) to its alphabetical index, NOT to ILSVRC2012_ID.
    
    Preferred: {base_dir}/imagenet/LOC_synset_mapping.txt (wnid per line in alphabetical order)
    Fallback:  {base_dir}/imagenet/imagenet-1k/data/meta.mat (requires scipy)
    """
    txt_mapping = os.path.join(base_dir, "imagenet", "LOC_synset_mapping.txt")
    if os.path.exists(txt_mapping):
        synset_to_idx: Dict[str, int] = {}
        with open(txt_mapping, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                synset = line.split(" ")[0]
                synset_to_idx[synset] = idx
        if len(synset_to_idx) != 1000:
            print(f"[warn] Expected 1000 synsets, found {len(synset_to_idx)} in {txt_mapping}")
        return synset_to_idx

    # Fallback to meta.mat: build alphabetical mapping
    meta_mat_path = os.path.join(base_dir, "imagenet", "imagenet-1k", "data", "meta.mat")
    if not os.path.exists(meta_mat_path):
        raise FileNotFoundError(
            f"Missing synset mapping: neither {txt_mapping} nor {meta_mat_path} exists."
        )
    try:
        import scipy.io as sio  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "scipy is required to read meta.mat for ImageNet synset mapping. "
            "Please install scipy or provide LOC_synset_mapping.txt."
        ) from e

    meta = sio.loadmat(meta_mat_path, squeeze_me=True, struct_as_record=False)
    synsets = meta.get("synsets", None)
    if synsets is None:
        raise RuntimeError(f"meta.mat at {meta_mat_path} missing 'synsets'")
    
    # Collect all leaf synsets (those with ILSVRC2012_ID)
    all_wnids = []
    for s in synsets:
        try:
            wnid = str(s.WNID)
            ilsvrc2012_id = int(s.ILSVRC2012_ID)
        except AttributeError:
            # Some entries might be non-leaf nodes; skip those
            continue
        if 1 <= ilsvrc2012_id <= 1000:
            all_wnids.append(wnid)
    
    if len(all_wnids) != 1000:
        raise RuntimeError(
            f"Expected 1000 leaf synsets in meta.mat, found {len(all_wnids)}."
        )
    
    # Sort alphabetically to get the correct index order
    sorted_wnids = sorted(all_wnids)
    synset_to_idx: Dict[str, int] = {wnid: idx for idx, wnid in enumerate(sorted_wnids)}
    
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


class _ImageNetValFromTxt(Dataset):
    """
    ImageNet validation set using flat folder and validation_ground_truth.txt for labels.
    Expects:
      - images at {base_dir}/imagenet/imagenet-1k/*.JPEG
      - labels at  {base_dir}/imagenet/imagenet-1k/data/ILSVRC2012_validation_ground_truth.txt
      - meta at {base_dir}/imagenet/imagenet-1k/data/meta.mat (required for ILSVRC ID -> synset mapping)
    
    Note: The ground truth file contains ILSVRC2012_ID (1..1000), which must be mapped
    to the alphabetical synset index (0..999) used by classnames.
    """

    def __init__(self, base_dir: str, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.img_dir = os.path.join(base_dir, "imagenet", "imagenet-1k")
        self.gt_path = os.path.join(
            base_dir, "imagenet", "imagenet-1k", "data", "ILSVRC2012_validation_ground_truth.txt"
        )
        self.meta_path = os.path.join(
            base_dir, "imagenet", "imagenet-1k", "data", "meta.mat"
        )

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"ImageNet-1k directory not found: {self.img_dir}")
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"Validation ground truth not found: {self.gt_path}")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path} (required for label mapping)")

        # Build mapping: ILSVRC2012_ID (1..1000) -> alphabetical synset index (0..999)
        ilsvrc_to_synset_idx = self._build_ilsvrc_to_synset_mapping()

        # Read labels (ILSVRC2012_ID: 1..1000) and convert to synset index (0..999)
        labels: List[int] = []
        with open(self.gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ilsvrc_id = int(line)
                if not (1 <= ilsvrc_id <= 1000):
                    raise ValueError(f"Invalid ILSVRC ID {ilsvrc_id} in {self.gt_path}")
                synset_idx = ilsvrc_to_synset_idx.get(ilsvrc_id)
                if synset_idx is None:
                    raise ValueError(f"No synset mapping found for ILSVRC ID {ilsvrc_id}")
                labels.append(synset_idx)

        # Gather image names sorted by numeric id to align with labels file
        fnames = [fn for fn in os.listdir(self.img_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
        # Expect ILSVRC2012_val_00000001.JPEG naming
        def _extract_num(name: str) -> int:
            # fallback: extract last 8 digits in name
            import re
            m = re.search(r"(\d{8})", name)
            return int(m.group(1)) if m else 10**9

        fnames_sorted = sorted(fnames, key=_extract_num)
        if len(fnames_sorted) < len(labels):
            raise RuntimeError(
                f"Number of images ({len(fnames_sorted)}) is smaller than number of labels ({len(labels)})."
            )
        if len(fnames_sorted) != len(labels):
            # Allow extra files but enforce prefix match if not equal
            print(
                f"[warn] Number of images ({len(fnames_sorted)}) != labels ({len(labels)}). "
                f"Using first {len(labels)} images in sorted order."
            )
            fnames_sorted = fnames_sorted[: len(labels)]

        self.samples: List[Tuple[str, int]] = [
            (os.path.join(self.img_dir, fname), labels[i]) for i, fname in enumerate(fnames_sorted)
        ]

    def _build_ilsvrc_to_synset_mapping(self) -> Dict[int, int]:
        """
        Build mapping from ILSVRC2012_ID (1..1000) to alphabetical synset index (0..999).
        
        Returns:
            Dict mapping ILSVRC2012_ID -> synset_index
        """
        try:
            import scipy.io as sio  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "scipy is required to read meta.mat for ImageNet label mapping. "
                "Please install scipy: pip install scipy"
            ) from e

        meta = sio.loadmat(self.meta_path, squeeze_me=True, struct_as_record=False)
        synsets = meta.get("synsets", None)
        if synsets is None:
            raise RuntimeError(f"meta.mat at {self.meta_path} missing 'synsets' field")

        # Build two mappings:
        # 1. ILSVRC2012_ID -> WNID (synset string)
        # 2. WNID (synset) -> alphabetical index (0..999)
        
        ilsvrc_to_wnid = {}
        all_wnids = []
        
        for s in synsets:
            try:
                wnid = str(s.WNID)
                ilsvrc2012_id = int(s.ILSVRC2012_ID)
            except AttributeError:
                # Some entries might be non-leaf nodes; skip those
                continue
            
            if 1 <= ilsvrc2012_id <= 1000:
                ilsvrc_to_wnid[ilsvrc2012_id] = wnid
                all_wnids.append(wnid)
        
        if len(ilsvrc_to_wnid) != 1000:
            raise RuntimeError(
                f"Expected 1000 leaf synsets in meta.mat, found {len(ilsvrc_to_wnid)}"
            )
        
        # Sort WNIDs alphabetically to get synset index (0..999)
        sorted_wnids = sorted(set(all_wnids))
        wnid_to_synset_idx = {wnid: idx for idx, wnid in enumerate(sorted_wnids)}
        
        # Combine: ILSVRC2012_ID -> synset_index
        ilsvrc_to_synset_idx = {}
        for ilsvrc_id, wnid in ilsvrc_to_wnid.items():
            synset_idx = wnid_to_synset_idx[wnid]
            ilsvrc_to_synset_idx[ilsvrc_id] = synset_idx
        
        return ilsvrc_to_synset_idx

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
      {base_dir}/ImageNetV2MFVal/{class_idx}/*.jpeg
    Labels are numeric directory names (0..999), matching ImageNet order.
    """

    def __init__(self, base_dir: str, transform=None):
        self.transform = transform
        self.root = os.path.join(
            base_dir, "ImageNetV2MFVal")
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
    ImageNet validation split sourced from imagenet-1k flat folder with ground truth txt.
    """

    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=6):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = imagenet_classnames

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = _ImageNetValFromTxt(
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

        root = os.path.join(location, "imagenet", "ImageNetA")
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

        root = os.path.join(location, "imagenet", "ImageNetR")
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

        root = os.path.join(location, "imagenet", "ImageNetSketch")
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
            os.path.join(location, "imagenet"), transform=self.preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
