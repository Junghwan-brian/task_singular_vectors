#!/usr/bin/env python3
"""
Minimal reproduction of the OOD evaluation bug.
This creates a tiny test to understand what's happening.
"""
import torch
import sys
import os
sys.path.insert(0, '/workspace')

from src.models import ImageEncoder, ImageClassifier, get_classification_head
from omegaconf import OmegaConf

print("="*80)
print("MINIMAL REPRODUCTION OF OOD BUG")
print("="*80)

model_name = "ViT-B-32"
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.create({
    "data_location": "~/data",
    "batch_size": 4,
    "device": device,
    "save_dir": "./models/checkpoints",
    "model": model_name,
})

print(f"\n1. Initial Setup")
print(f"   Model: {model_name}")
print(f"   Device: {device}")

# Load model
image_encoder = ImageEncoder(model_name).to(device)
classification_head = get_classification_head(cfg, "ImageNet")
model = ImageClassifier(image_encoder, classification_head).to(device)

print(f"\n2. Get initial classification head ID")
initial_head_id = id(model.classification_head)
initial_weight_sum = model.classification_head.weight.sum().item()
print(f"   Classification head ID: {initial_head_id}")
print(f"   Weight sum: {initial_weight_sum:.6f}")
print(f"   Weight mean: {model.classification_head.weight.mean().item():.6f}")

print(f"\n3. Simulate training (modify classification head)")
# Freeze encoder
for p in image_encoder.parameters():
    p.requires_grad = False
    
# Unfreeze head
model.classification_head.weight.requires_grad_(True)
model.classification_head.bias.requires_grad_(True)

# Modify weights directly to simulate training
with torch.no_grad():
    model.classification_head.weight += 0.1
    model.classification_head.bias += 0.01

trained_weight_sum = model.classification_head.weight.sum().item()
print(f"   After 'training' - Weight sum: {trained_weight_sum:.6f}")
print(f"   Weight changed: {abs(trained_weight_sum - initial_weight_sum) > 0.001}")

print(f"\n4. Check if head ID is still the same")
after_train_head_id = id(model.classification_head)
print(f"   Classification head ID: {after_train_head_id}")
print(f"   Same object: {after_train_head_id == initial_head_id}")

print(f"\n5. Simulate evaluate_encoder_with_dataloader")
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

# Create tiny dummy dataloader
class TinyDataset:
    def __len__(self):
        return 8
    
    def __getitem__(self, idx):
        return {"images": torch.randn(3, 224, 224), "labels": idx % 1000}

tiny_loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)

print(f"\n   Before evaluation:")
print(f"     model.classification_head ID: {id(model.classification_head)}")
print(f"     Weight sum: {model.classification_head.weight.sum().item():.6f}")

# Call evaluation
metrics = evaluate_encoder_with_dataloader(
    model.image_encoder, model.classification_head, tiny_loader, device
)

print(f"\n   After evaluation:")
print(f"     model.classification_head ID: {id(model.classification_head)}")
print(f"     Weight sum: {model.classification_head.weight.sum().item():.6f}")
print(f"     Accuracy: {metrics['top1']:.4f}")

print(f"\n6. Now simulate OOD evaluation with get_ood_dataloader")
def get_ood_dataloader_simple(preprocess):
    """Simplified version"""
    from src.datasets.registry import get_dataset
    from src.datasets.common import get_dataloader
    
    # This will try to load real ImageNetA, but will fail if data doesn't exist
    try:
        dataset = get_dataset(
            "ImageNetA",
            preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
        )
        return get_dataloader(dataset, is_train=False, args=cfg, image_encoder=None)
    except Exception as e:
        print(f"     (Could not load real ImageNetA: {e})")
        print(f"     Using dummy loader instead")
        return tiny_loader

print(f"\n   Before OOD evaluation:")
print(f"     model.image_encoder ID: {id(model.image_encoder)}")
print(f"     model.classification_head ID: {id(model.classification_head)}")
print(f"     Weight sum: {model.classification_head.weight.sum().item():.6f}")

try:
    ood_loader = get_ood_dataloader_simple(model.image_encoder.val_preprocess)
    
    print(f"\n   After creating OOD loader:")
    print(f"     model.image_encoder ID: {id(model.image_encoder)}")
    print(f"     model.classification_head ID: {id(model.classification_head)}")
    print(f"     Weight sum: {model.classification_head.weight.sum().item():.6f}")
    
    ood_metrics = evaluate_encoder_with_dataloader(
        model.image_encoder, model.classification_head, ood_loader, device
    )
    
    print(f"\n   After OOD evaluation:")
    print(f"     model.classification_head ID: {id(model.classification_head)}")
    print(f"     Weight sum: {model.classification_head.weight.sum().item():.6f}")
    print(f"     OOD Accuracy: {ood_metrics['top1']:.4f}")

except Exception as e:
    print(f"   Error during OOD evaluation: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"If the classification_head ID changes or weight_sum reverts to initial value,")
print(f"that's where the bug is!")
print(f"\nInitial weight sum: {initial_weight_sum:.6f}")
print(f"Trained weight sum: {trained_weight_sum:.6f}")
print(f"Final weight sum: {model.classification_head.weight.sum().item():.6f}")


