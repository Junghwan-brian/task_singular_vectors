#!/usr/bin/env python3
"""
ImageNet 및 OOD 데이터셋에 대한 Zero-shot CLIP 성능 평가 스크립트

모델별로 ImageNet-1k validation 및 OOD 데이터셋 
(ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2)에서 
zero-shot CLIP 분류 성능을 측정합니다.

사용법:
    python imagenet_zeroshot_eval.py
    python imagenet_zeroshot_eval.py --models ViT-B-16 ViT-L-14
    python imagenet_zeroshot_eval.py --output-dir results/zeroshot
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import open_clip

from src.models.modeling import ImageEncoder, ClassificationHead, ImageClassifier
from src.datasets.imagenet_ood import (
    ImageNetILSVRCVal, 
    ImageNetA, 
    ImageNetR, 
    ImageNetSketch, 
    ImageNetV2MFVal
)
from src.datasets.templates import get_templates
from src.datasets.common import maybe_dictionarize

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_zeroshot_classification_head(
    model: nn.Module,
    classnames: List[str],
    templates: List,
    device: torch.device
) -> ClassificationHead:
    """
    CLIP 텍스트 인코더를 사용하여 zero-shot classification head를 구축합니다.
    
    Args:
        model: CLIP model (text encoder 포함)
        classnames: 클래스 이름 리스트
        templates: 텍스트 프롬프트 템플릿
        device: 디바이스
        
    Returns:
        ClassificationHead 객체
    """
    logger.info(f"Building zero-shot classification head for {len(classnames)} classes...")
    
    logit_scale = model.logit_scale
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Building text embeddings"):
            texts = []
            for template in templates:
                texts.append(template(classname))
            
            # Tokenize and encode
            texts = open_clip.tokenize(texts).to(device)
            embeddings = model.encode_text(texts)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            
            # Average over templates
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            
            zeroshot_weights.append(embeddings)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        
        # Apply logit scale
        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
    
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    return classification_head


def evaluate_zeroshot(
    model: ImageClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str
) -> Dict[str, float]:
    """
    Zero-shot 성능을 평가합니다.
    
    Args:
        model: ImageClassifier 모델
        dataloader: 평가 데이터로더
        device: 디바이스
        dataset_name: 데이터셋 이름
        
    Returns:
        평가 메트릭 (top1 accuracy)
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    logger.info(f"Evaluating on {dataset_name}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_name}"):
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(images)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"{dataset_name} - Accuracy: {accuracy * 100:.2f}%")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_model_on_datasets(
    model_name: str,
    data_location: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    특정 모델을 모든 데이터셋에서 평가합니다.
    
    Args:
        model_name: 모델 이름 (예: "ViT-B-16")
        data_location: 데이터 위치
        device: 디바이스
        batch_size: 배치 사이즈
        num_workers: 데이터로더 워커 수
        
    Returns:
        데이터셋별 평가 결과
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"{'='*80}\n")
    
    # Load CLIP model
    logger.info(f"Loading CLIP model: {model_name}")
    image_encoder = ImageEncoder(model_name, keep_lang=True)
    clip_model = image_encoder.model
    preprocess = image_encoder.val_preprocess
    
    # Get ImageNet templates
    templates = get_templates("ImageNet")
    
    results = {}
    
    # 데이터셋 정의
    datasets_info = [
        ("ImageNetVal", ImageNetILSVRCVal),
        ("ImageNetA", ImageNetA),
        ("ImageNetR", ImageNetR),
        ("ImageNetSketch", ImageNetSketch),
        ("ImageNetV2", ImageNetV2MFVal),
    ]
    
    for dataset_name, dataset_class in datasets_info:
        try:
            logger.info(f"\n--- {dataset_name} ---")
            
            # Load dataset
            dataset = dataset_class(
                preprocess=preprocess,
                location=data_location,
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            # Build classification head
            classification_head = build_zeroshot_classification_head(
                model=clip_model,
                classnames=dataset.classnames,
                templates=templates,
                device=device
            )
            
            # Create image classifier
            classifier = ImageClassifier(
                image_encoder=image_encoder,
                classification_head=classification_head
            )
            
            # Evaluate
            metrics = evaluate_zeroshot(
                model=classifier,
                dataloader=dataset.test_loader,
                device=device,
                dataset_name=dataset_name
            )
            
            results[dataset_name] = metrics
            
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            results[dataset_name] = {"error": str(e)}
    
    return results


def save_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str
):
    """
    평가 결과를 저장합니다.
    
    Args:
        all_results: 모든 모델/데이터셋 결과
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_path / "zeroshot_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {json_path}")
    
    # Create summary table
    summary_path = output_path / "zeroshot_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Zero-shot CLIP Performance on ImageNet and OOD Datasets\n")
        f.write("="*80 + "\n\n")
        
        # Header
        datasets = ["ImageNetVal", "ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2"]
        f.write(f"{'Model':<20}")
        for ds in datasets:
            f.write(f"{ds:<18}")
        f.write("\n" + "-"*80 + "\n")
        
        # Results
        for model_name, model_results in sorted(all_results.items()):
            f.write(f"{model_name:<20}")
            for ds in datasets:
                if ds in model_results and "accuracy" in model_results[ds]:
                    acc = model_results[ds]["accuracy"] * 100
                    f.write(f"{acc:>6.2f}%           ")
                else:
                    f.write(f"{'N/A':<18}")
            f.write("\n")
    
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("Zero-shot CLIP Performance Summary")
    print("="*80)
    with open(summary_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet 및 OOD 데이터셋에서 Zero-shot CLIP 성능 평가"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ViT-B-16", "ViT-L-14", "ViT-B-32"],
        help="평가할 모델 리스트 (기본: ViT-B-16 ViT-L-14 ViT-B-32)"
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="데이터셋 루트 경로"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="imagenet_results",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="배치 사이즈"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="데이터로더 워커 수"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="디바이스 (cuda 또는 cpu)"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Evaluate all models
    all_results = {}
    
    for model_name in args.models:
        try:
            results = evaluate_model_on_datasets(
                model_name=model_name,
                data_location=args.data_location,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}
    
    # Save results
    save_results(all_results, args.output_dir)
    
    logger.info("\n✓ Zero-shot evaluation completed!")


if __name__ == "__main__":
    main()

