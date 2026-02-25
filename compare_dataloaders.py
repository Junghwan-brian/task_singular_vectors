#!/usr/bin/env python3
"""
Compare how ImageNet val_loader vs OOD loader are created.
"""

print("="*80)
print("COMPARISON: ImageNet Val Loader vs OOD Loader Creation")
print("="*80)

print("\n### IMAGENET VAL LOADER (line 677-687):")
print("""
def _build_loader(dataset, is_train: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=is_train,
        num_workers=int(getattr(cfg, "num_workers", 2)),
        pin_memory=True,
    )

# ImageNet validation dataset
_eval_val_obj = ImageNetILSVRCVal(
    ImageEncoder(cfg.model).val_preprocess,  # ← uses model's val_preprocess
    location=cfg.data_location,
    batch_size=cfg.batch_size,
    num_workers=...)
base_val_dataset = _eval_val_obj.test_dataset

val_full_loader = _build_loader(base_val_dataset, is_train=False)
""")

print("\n### OOD LOADER (line 445-446):")
print("""
ood_loader = get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)

# Inside get_ood_dataloader (line 169-180):
def get_ood_dataloader(dataset_name: str, preprocess, cfg):
    dataset = get_dataset(
        dataset_name,
        preprocess,  # ← receives model.image_encoder.val_preprocess
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    return get_dataloader(dataset, is_train=False, args=cfg, image_encoder=None)

# Inside get_dataloader (line 127-133):
def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader  # ← returns dataset's loader
    return dataloader
""")

print("\n" + "="*80)
print("KEY DIFFERENCE")
print("="*80)
print("""
ImageNet val_loader:
  1. Create ImageNetILSVRCVal dataset with model.val_preprocess
  2. Get dataset.test_dataset (the actual dataset object)
  3. Wrap in DataLoader directly

OOD loader:
  1. Create OOD dataset (ImageNetA, etc.) with model.val_preprocess
  2. Call get_dataloader(..., image_encoder=None)
  3. Returns dataset.test_loader (ALREADY CREATED DataLoader)

The OOD dataset's test_loader was created WHEN THE DATASET WAS INITIALIZED!
Let's check what happens in ImageNetA.__init__()...
""")

print("\n### IMAGENET OOD DATASET INIT (imagenet_ood.py line 356-388):")
print("""
class ImageNetA:
    def __init__(self, preprocess, location=..., batch_size=128, ...):
        self.preprocess = preprocess
        self.batch_size = batch_size
        ...
        self.train_dataset = ImageFolderWithPaths(root, transform=self.preprocess)
        _remap_imagefolder_labels_to_imagenet_indices(self.train_dataset, synset_to_idx)
        self.test_dataset = self.train_dataset
        
        # test_loader is created HERE with the preprocess passed to __init__
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,  # ← uses dataset's batch_size (128)
            shuffle=False,
            ...
        )
""")

print("\n" + "="*80)
print("POTENTIAL ISSUE")
print("="*80)
print("""
OOD dataset's test_loader uses:
  - batch_size from dataset init (might be different!)
  - transform from preprocess passed to __init__

But this should be fine... the preprocess is correct (model.image_encoder.val_preprocess).

So the transform is correct, batch_size might differ but shouldn't affect accuracy.

The bug must be elsewhere...
""")

print("\n" + "="*80)
print("WAIT - LET'S CHECK SOMETHING")
print("="*80)
print("""
When we call get_ood_dataloader(), we pass model.image_encoder.val_preprocess.

But WHICH model.image_encoder?

Let's trace back:
  - Line 446: get_ood_dataloader(ood_name, model.image_encoder.val_preprocess, cfg)
  - 'model' here is the trained model

So model.image_encoder should be the encoder used during training (frozen but correct).
The preprocess should be correct.

Unless... wait, let me check if model.image_encoder changes between ImageNet and OOD eval.
""")

print("\n" + "="*80)
print("HYPOTHESIS: Model is recreated or reset")
print("="*80)
print("""
What if, between ImageNet evaluation and OOD evaluation,
the model object is somehow reset or recreated?

Or what if model.image_encoder is reassigned?

Let's check the code flow...
""")


