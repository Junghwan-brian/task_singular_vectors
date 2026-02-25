"""
Script to save one sample image per class for each remote sensing dataset.
Saves images in organized folders: {output_dir}/{dataset_name}/{class}.png
Also saves metadata.json in each dataset folder.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
import json
import random
# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    REMOTE_SENSING_DATASETS
)
from src.datasets.registry import get_dataset
# Simple preprocessing that converts to PIL and then back to tensor
def get_simple_preprocess():
    """Get a simple preprocessing function that normalizes images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
def tensor_to_pil(tensor):
    """Convert a tensor image to PIL Image"""
    if tensor.dim() == 3:
        img = tensor
    elif tensor.dim() == 2:
        # Grayscale image [H, W]
        img = tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
    else:
        img = tensor.squeeze(0)
    
    # Handle grayscale images (1 channel)
    if img.shape[0] == 1:
        img = img.squeeze(0)  # [H, W]
        img_np = (img.numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='L').convert('RGB')
    else:
        # RGB images
        img = img.permute(1, 2, 0)  # [H, W, C]
        img_np = (img.numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
def save_one_sample_per_class(dataset_name, location="./datasets", output_dir="./test_outputs"):
    """
    Load a dataset and save one sample per class in a dedicated folder with metadata
    Args:
        dataset_name: Name of the dataset
        location: Root directory containing datasets
        output_dir: Directory to save the samples
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    try:
        # Get preprocessing
        preprocess = get_simple_preprocess()
        # Load dataset
        dataset = get_remote_sensing_dataset(
            dataset_name,
            preprocess,
            location=location,
            batch_size=1,
            num_workers=0,  # Use 0 workers for simplicity
        )
        # Get class names
        classnames = dataset.classnames
        print(f"Found {len(classnames)} classes: {classnames}")
        # Collect metadata
        metadata = {
            "dataset_name": dataset_name,
            "num_classes": len(classnames),
            "class_names": classnames,
            "train_samples": len(dataset.train_dataset),
            "test_samples": len(dataset.test_dataset),
        }
        # Collect one sample per class from train dataset
        class_samples = defaultdict(list)
        # Iterate through train dataset
        print("Collecting samples from train dataset...")
        for idx in range(len(dataset.train_dataset)):
            try:
                img, label = dataset.train_dataset[idx]
                # Handle different label types
                if isinstance(label, torch.Tensor):
                    if label.dim() == 0:  # scalar tensor
                        label_idx = int(label.item())
                    elif label.dim() == 1 and len(label) == 1:  # 1D tensor with 1 element
                        label_idx = int(label[0].item())
                    elif label.dim() == 1 and len(label) > 1:  # multi-hot encoding
                        # For multi-label, take the first positive label
                        label_idx = int(torch.where(label > 0)[0][0].item())
                    else:
                        label_idx = int(label.item())
                else:
                    label_idx = int(label)
                # Store sample if we don't have one for this class yet
                if label_idx not in class_samples and label_idx < len(classnames):
                    class_samples[label_idx] = img
                    print(f"  Collected sample for class {label_idx}: {classnames[label_idx]}")
                # Stop if we have samples for all classes
                if len(class_samples) == len(classnames):
                    break
            except Exception as e:
                print(f"  Warning: Error processing sample {idx}: {e}")
                continue
        # Create dataset-specific folder
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"\nSaving samples to {dataset_dir}...")
        # Save samples
        saved_classes = []
        for class_idx, img_tensor in class_samples.items():
            class_name = classnames[class_idx]
            # Clean class name for filename
            clean_class_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in class_name)
            filename = f"{clean_class_name}.png"
            filepath = os.path.join(dataset_dir, filename)
            # Convert tensor to PIL and save
            pil_img = tensor_to_pil(img_tensor)
            pil_img.save(filepath)
            print(f"  Saved: {filename}")
            saved_classes.append({
                "class_idx": class_idx,
                "class_name": class_name,
                "filename": filename
            })
        # Add saved classes info to metadata
        metadata["saved_classes"] = saved_classes
        metadata["num_saved_classes"] = len(saved_classes)
        # Save metadata as JSON
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: metadata.json")
        print(f"\n✓ Successfully processed {dataset_name}: {len(class_samples)}/{len(classnames)} classes saved")
        return True
    except Exception as e:
        print(f"\n✗ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
def main():
    """Main function to process all datasets"""
    # Get list of all remote sensing datasets
    all_datasets = sorted(REMOTE_SENSING_DATASETS.keys())
    print(f"Found {len(all_datasets)} remote sensing datasets")
    print(f"Datasets: {all_datasets}")
    # Process each dataset
    success_count = 0
    failed_datasets = []
    for dataset_name in all_datasets:
        success = save_one_sample_per_class(
            dataset_name,
            location="../datasets",  # Relative to test_outputs folder
            output_dir="."  # Save in test_outputs folder (current directory when run from there)
        )
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(all_datasets)} datasets")
    if failed_datasets:
        print(f"\nFailed datasets ({len(failed_datasets)}):")
        for name in failed_datasets:
            print(f"  - {name}")
    else:
        print("\n✓ All datasets processed successfully!")
def save_general_dataset_samples(location="./datasets", output_dir="./examples/general", total_samples=30):
    """
    일반 데이터셋들로부터 샘플을 수집하여 저장합니다.
    각 데이터셋에서 최소 1개씩 샘플을 수집합니다.
    
    Args:
        location: 데이터셋 루트 디렉토리
        output_dir: 샘플을 저장할 디렉토리
        total_samples: 수집할 총 샘플 수
    """
    # Remote sensing 데이터셋 제외한 일반 데이터셋 목록
    general_datasets = [
        'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST', 'EMNIST',
        'STL10', 'SVHN', 'GTSRB', 'Flowers102', 'Food101', 'DTD',
        'OxfordIIITPet', 'SUN397', 'Cars', 'FGVCAircraft', 
        'Caltech101', 'Caltech256', 'CUB200', 'PascalVOC'
    ]
    
    print(f"\n{'='*60}")
    print(f"일반 데이터셋 샘플링 시작")
    print(f"목표 샘플 수: {total_samples}")
    print(f"{'='*60}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플 수집
    collected_samples = []
    preprocess = get_simple_preprocess()
    
    # 데이터셋당 샘플 수 계산 (균등 분배, 여유있게 설정)
    samples_per_dataset = max(2, (total_samples // len(general_datasets)) + 1)
    
    for dataset_name in general_datasets:
        try:
            print(f"\n처리 중: {dataset_name}")
            
            # 데이터셋 로드
            dataset = get_dataset(
                dataset_name,
                preprocess,
                location=location,
                batch_size=1,
                num_workers=0,
            )
            
            # 샘플 수집
            num_samples = min(samples_per_dataset, len(dataset.train_dataset))
            indices = random.sample(range(len(dataset.train_dataset)), num_samples)
            
            for idx in indices:
                if len(collected_samples) >= total_samples:
                    break
                    
                try:
                    img, label = dataset.train_dataset[idx]
                    
                    # 레이블 처리
                    if isinstance(label, torch.Tensor):
                        label_idx = int(label.item())
                    else:
                        label_idx = int(label)
                    
                    # 클래스 이름 가져오기
                    if label_idx < len(dataset.classnames):
                        class_name = dataset.classnames[label_idx]
                    else:
                        class_name = f"class_{label_idx}"
                    
                    # 파일명 생성
                    sample_id = len(collected_samples)
                    clean_dataset_name = "".join(c if c.isalnum() else '_' for c in dataset_name)
                    clean_class_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in class_name)
                    filename = f"{sample_id:03d}_{clean_dataset_name}_{clean_class_name}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # 이미지 저장
                    pil_img = tensor_to_pil(img)
                    pil_img.save(filepath)
                    
                    collected_samples.append({
                        'filename': filename,
                        'dataset': dataset_name,
                        'class_name': class_name,
                        'label': label_idx
                    })
                    
                    print(f"  저장됨: {filename}")
                    
                except Exception as e:
                    print(f"  경고: 샘플 {idx} 처리 중 오류: {e}")
                    continue
            
            if len(collected_samples) >= total_samples:
                break
                
        except Exception as e:
            print(f"  오류: {dataset_name} 로드 실패: {e}")
            continue
    
    # 메타데이터 저장
    metadata = {
        'total_samples': len(collected_samples),
        'samples': collected_samples
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 완료: {len(collected_samples)}개 샘플 저장됨")
    print(f"  저장 위치: {output_dir}")
    return collected_samples


def create_image_grid(image_dir, output_path, num_images=25, grid_size=(5, 5), seed=42):
    """
    디렉토리에서 이미지를 랜덤하게 샘플링하여 그리드 형태로 저장합니다.
    
    Args:
        image_dir: 이미지가 있는 디렉토리
        output_path: 그리드 이미지를 저장할 경로
        num_images: 샘플링할 이미지 수
        grid_size: 그리드 크기 (rows, cols)
        seed: 랜덤 시드
    """
    random.seed(seed)
    
    # 이미지 파일 목록 가져오기
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if len(image_files) < num_images:
        print(f"경고: {image_dir}에 {num_images}개 이미지가 부족합니다. ({len(image_files)}개만 있음)")
        num_images = len(image_files)
    
    # 랜덤 샘플링
    sampled_images = random.sample(image_files, num_images)
    
    # 그리드 생성
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(sampled_images):
            img = Image.open(sampled_images[idx])
            ax.imshow(img)
        ax.axis('off')
    
    # 저장
    plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
    plt.close()
    
    print(f"✓ 그리드 이미지 저장됨: {output_path}")
    print(f"  샘플 수: {len(sampled_images)}")


def main_general_samples():
    """일반 데이터셋 샘플을 수집하고 그리드 이미지를 생성합니다."""
    # 1. 일반 데이터셋 샘플 수집
    print("\n" + "="*60)
    print("STEP 1: 일반 데이터셋 샘플 수집")
    print("="*60)
    save_general_dataset_samples(
        location="/workspace/datasets",
        output_dir="/workspace/examples/general",
        total_samples=30
    )
    
    # 2. 그리드 이미지 생성
    print("\n" + "="*60)
    print("STEP 2: 그리드 이미지 생성")
    print("="*60)
    
    # General domain 그리드
    print("\nGeneral domain 그리드 생성 중...")
    create_image_grid(
        image_dir="/workspace/examples/general",
        output_path="/workspace/examples/general_domain_5x5.png",
        num_images=25,
        grid_size=(5, 5),
        seed=42
    )
    
    # Remote sensing 그리드
    print("\nRemote sensing 그리드 생성 중...")
    create_image_grid(
        image_dir="/workspace/examples/rs",
        output_path="/workspace/examples/remote_sensing_5x5.png",
        num_images=25,
        grid_size=(5, 5),
        seed=42
    )
    
    print("\n" + "="*60)
    print("완료!")
    print("="*60)
    print("생성된 파일:")
    print("  - /workspace/examples/general_domain_5x5.png")
    print("  - /workspace/examples/remote_sensing_5x5.png")


if __name__ == "__main__":
    # 기존 remote sensing 샘플링을 원할 경우: main()
    # 일반 데이터셋 샘플링 및 그리드 생성: main_general_samples()
    main_general_samples()















