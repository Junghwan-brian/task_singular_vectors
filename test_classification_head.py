#!/usr/bin/env python3
"""
Test if classification head is actually trained and used for evaluation.
"""
import torch
import sys
sys.path.insert(0, '/workspace')

from src.models import ImageEncoder, ImageClassifier, get_classification_head
from omegaconf import OmegaConf

print("="*80)
print("TEST: Classification Head Training and Evaluation")
print("="*80)

model_name = "ViT-B-32"
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.create({
    "data_location": "~/data",
    "batch_size": 128,
    "device": device,
    "save_dir": "./models/checkpoints",
    "model": model_name,
})

print(f"\n1. Loading {model_name}...")
image_encoder = ImageEncoder(model_name).to(device)
classification_head = get_classification_head(cfg, "ImageNet")
model = ImageClassifier(image_encoder, classification_head).to(device)

print(f"   Encoder output dim: {image_encoder.output_dim if hasattr(image_encoder, 'output_dim') else 'unknown'}")
print(f"   Head weight shape: {classification_head.weight.shape}")
print(f"   Head bias shape: {classification_head.bias.shape}")

# Get initial weights
initial_weight = classification_head.weight.data.clone()
initial_bias = classification_head.bias.data.clone()
print(f"   Initial weight mean: {initial_weight.mean():.6f}")
print(f"   Initial bias mean: {initial_bias.mean():.6f}")

print(f"\n2. Freezing encoder, unfreezing head...")
for p in image_encoder.parameters():
    p.requires_grad = False
classification_head.weight.requires_grad_(True)
classification_head.bias.requires_grad_(True)

print(f"   Encoder frozen: {not any(p.requires_grad for p in image_encoder.parameters())}")
print(f"   Head trainable: {classification_head.weight.requires_grad}")

print(f"\n3. Simulating training (10 steps)...")
optimizer = torch.optim.AdamW(classification_head.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Simulate training
losses = []
for step in range(10):
    # Dummy data
    dummy_features = torch.randn(16, 512).to(device)  # 16 samples, 512 features
    dummy_labels = torch.randint(0, 1000, (16,)).to(device)
    
    # Forward
    logits = classification_head(dummy_features)
    loss = loss_fn(logits, dummy_labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

print(f"   Losses: {losses[:3]} ... {losses[-3:]}")
print(f"   Loss decreased: {losses[0] > losses[-1]}")

# Check if weights changed
weight_changed = not torch.allclose(initial_weight, classification_head.weight.data)
bias_changed = not torch.allclose(initial_bias, classification_head.bias.data)
print(f"   Weight changed: {weight_changed}")
print(f"   Bias changed: {bias_changed}")
print(f"   New weight mean: {classification_head.weight.data.mean():.6f}")
print(f"   New bias mean: {classification_head.bias.data.mean():.6f}")

print(f"\n4. Testing evaluate_encoder_with_dataloader...")
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

# Create dummy dataloader
class DummyDataset:
    def __len__(self):
        return 32
    
    def __getitem__(self, idx):
        img = torch.randn(3, 224, 224)
        label = idx % 1000
        return {"images": img, "labels": label}

dummy_dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=8)

# Evaluate with trained head
print(f"   Evaluating with trained classification head...")
metrics = evaluate_encoder_with_dataloader(
    image_encoder, classification_head, dummy_dataloader, device
)
print(f"   Accuracy: {metrics['top1']:.4f}")

# Now evaluate with a NEW zero-shot head to compare
print(f"\n5. Creating NEW zero-shot head...")
new_zero_shot_head = get_classification_head(cfg, "ImageNet")
print(f"   New head is same as trained: {torch.allclose(new_zero_shot_head.weight, classification_head.weight)}")

print(f"   Evaluating with NEW zero-shot head...")
metrics_zero = evaluate_encoder_with_dataloader(
    image_encoder, new_zero_shot_head, dummy_dataloader, device
)
print(f"   Accuracy: {metrics_zero['top1']:.4f}")

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Trained head accuracy: {metrics['top1']:.4f}")
print(f"Zero-shot head accuracy: {metrics_zero['top1']:.4f}")
print(f"Same result: {abs(metrics['top1'] - metrics_zero['top1']) < 0.001}")

if abs(metrics['top1'] - metrics_zero['top1']) < 0.001:
    print(f"\n⚠️  WARNING: Results are IDENTICAL!")
    print(f"This suggests get_classification_head() returns the SAME head every time,")
    print(f"NOT a newly initialized zero-shot head.")
else:
    print(f"\n✓ Results are different as expected.")

print(f"\n{'='*80}")
print(f"HYPOTHESIS")
print(f"{'='*80}")
print(f"If get_classification_head() returns a CACHED head instead of creating new one,")
print(f"then ALL evaluations (ImageNet and OOD) use the SAME head.")
print(f"This would explain why ImageNet results differ (trained head)")
print(f"but OOD results are identical (they all use the same cached zero-shot head).")


