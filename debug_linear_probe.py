#!/usr/bin/env python3
"""
Debug Linear Probe to check if classification head is actually trained
and used for OOD evaluation.
"""
import torch
import sys
import os
sys.path.insert(0, '/workspace')

from src.models import ImageEncoder, ImageClassifier, get_classification_head
from omegaconf import OmegaConf

print("="*80)
print("DEBUG: Linear Probe Classification Head Training")
print("="*80)

# Load model
model_name = "ViT-B-32"
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.create({
    "data_location": "~/data",
    "batch_size": 128,
    "device": device,
    "save_dir": "./models/checkpoints",
    "model": model_name,
})

print(f"\nLoading {model_name}...")
image_encoder = ImageEncoder(model_name).to(device)

# Get classification head
classification_head = get_classification_head(cfg, "ImageNet")
model = ImageClassifier(image_encoder, classification_head).to(device)

print("\n" + "="*80)
print("Initial Classification Head State")
print("="*80)
print(f"Weight shape: {model.classification_head.weight.shape}")
print(f"Bias shape: {model.classification_head.bias.shape}")
print(f"Weight requires_grad: {model.classification_head.weight.requires_grad}")
print(f"Bias requires_grad: {model.classification_head.bias.requires_grad}")
print(f"First 5 weights of class 0: {model.classification_head.weight[0, :5]}")

# Check if this is zero-shot or trained
initial_weights = model.classification_head.weight.data.clone()

# Now check what happens in Linear Probe training
print("\n" + "="*80)
print("Simulating Linear Probe Training Setup")
print("="*80)

# Freeze image encoder
for p in image_encoder.parameters():
    p.requires_grad = False
    
# Unfreeze classification head
model.classification_head.weight.requires_grad_(True)
model.classification_head.bias.requires_grad_(True)

print(f"After training setup:")
print(f"  Encoder params require_grad: {any(p.requires_grad for p in image_encoder.parameters())}")
print(f"  Head weight requires_grad: {model.classification_head.weight.requires_grad}")
print(f"  Head bias requires_grad: {model.classification_head.bias.requires_grad}")

# Simulate one training step
dummy_input = torch.randn(4, 512).to(device)  # 4 samples, 512 features (ViT-B-32)
dummy_labels = torch.randint(0, 1000, (4,)).to(device)

optimizer = torch.optim.AdamW(model.classification_head.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Forward pass (simulating what happens during training)
logits = model.classification_head(dummy_input)
loss = loss_fn(logits, dummy_labels)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"\nAfter one training step:")
print(f"  Loss: {loss.item():.6f}")
print(f"  Weights changed: {not torch.allclose(initial_weights, model.classification_head.weight.data)}")
print(f"  First 5 weights of class 0 (after): {model.classification_head.weight[0, :5]}")

print("\n" + "="*80)
print("Now let's check what evaluate_encoder_with_dataloader does")
print("="*80)

# Check the evaluation function
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

print("\nReading evaluate_encoder_with_dataloader source code...")
import inspect
source = inspect.getsource(evaluate_encoder_with_dataloader)
print(source)

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)
print("evaluate_encoder_with_dataloader creates a NEW ImageClassifier:")
print("  model = ImageClassifier(image_encoder, classification_head)")
print("\nThis means it uses whatever image_encoder and classification_head are passed in.")
print("So the question is: are we passing the TRAINED classification_head?")

print("\n" + "="*80)
print("Checking actual baseline training code...")
print("="*80)
print("In imagenet_baselines_train.py line 282-284:")
print("  ood_metrics = evaluate_encoder_with_dataloader(")
print("      model.image_encoder, model.classification_head, ood_loader, cfg.device")
print("  )")
print("\nThis SHOULD work correctly - it passes model.classification_head")
print("which has been trained.")

print("\n" + "="*80)
print("HYPOTHESIS: The problem might be elsewhere...")
print("="*80)
print("Let's check if get_classification_head is being called again somewhere")
print("and overwriting the trained head.")


