#!/usr/bin/env python3
"""
ImageNet 실험 결과를 집계하고 시각화하는 스크립트.
모델, shot 설정, 메소드별로 결과를 그룹화하고 비교합니다.

ImageNet-1k validation split 및 OOD 데이터셋(ImageNetA, ImageNetR, ImageNetSketch, ImageNetV2) 결과를 처리합니다.
"""

import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_energy_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Energy config tag를 파싱하여 하이퍼파라미터를 추출합니다.
    형식: imagenet_energy_{sigma_lr}_{topk}_{init_mode}_{warmup_ratio}_{sigma_wd}_{k}shot
    예시: imagenet_energy_0p001_12_average_0p1_0p0_16shot
    """
    if not config_tag.startswith('imagenet_energy_'):
        return {}
    
    parts = config_tag.split('_')
    if len(parts) < 8:
        return {}
    
    # imagenet_energy_{sigma_lr}_{topk}_{init_mode}_{warmup_ratio}_{sigma_wd}_{k}shot
    return {
        'sigma_lr': parts[2].replace('p', '.'),
        'topk': parts[3],
        'init_mode': parts[4],
        'warmup_ratio': parts[5].replace('p', '.'),
        'sigma_wd': parts[6].replace('p', '.'),
        'k': parts[7].replace('shot', ''),
    }


def parse_atlas_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Atlas config tag를 파싱하여 하이퍼파라미터를 추출합니다.
    형식: imagenet_atlas_{num_basis}_{lr}_{k}shot
    예시: imagenet_atlas_17_0p1_16shot
    """
    if not config_tag.startswith('imagenet_atlas_'):
        return {}
    
    parts = config_tag.split('_')
    if len(parts) < 5:
        return {}
    
    # imagenet_atlas_{num_basis}_{lr}_{k}shot
    return {
        'num_basis': parts[2],
        'lr': parts[3].replace('p', '.'),
        'k': parts[4].replace('shot', ''),
    }


def parse_baseline_config_tag(config_tag: str) -> Dict[str, str]:
    """
    Baseline config tag를 파싱하여 하이퍼파라미터를 추출합니다.
    
    LinearProbe: imagenet_baseline_lp_{lr}_{epochs}_{wd}_{k}shot
    TIP: imagenet_baseline_tip_{wd}_{k}shot
    LP++: imagenet_baseline_lpp_{wd}_{k}shot
    LoRA: imagenet_baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}_{k}shot
    """
    if not config_tag.startswith('imagenet_baseline_'):
        return {}
    
    parts = config_tag.split('_')
    
    if 'lp_' in config_tag and 'lpp_' not in config_tag:
        # LinearProbe: imagenet_baseline_lp_{lr}_{epochs}_{wd}_{k}shot
        if len(parts) >= 7:
            return {
                'method': 'LinearProbe',
                'lr': parts[3].replace('p', '.'),
                'epochs': parts[4],
                'wd': parts[5].replace('p', '.'),
                'k': parts[6].replace('shot', ''),
            }
    elif 'tip_' in config_tag:
        # TIP: imagenet_baseline_tip_{wd}_{k}shot
        if len(parts) >= 5:
            return {
                'method': 'TIP',
                'wd': parts[3].replace('p', '.'),
                'k': parts[4].replace('shot', ''),
            }
    elif 'lpp_' in config_tag:
        # LP++: imagenet_baseline_lpp_{wd}_{k}shot
        if len(parts) >= 5:
            return {
                'method': 'LP++',
                'wd': parts[3].replace('p', '.'),
                'k': parts[4].replace('shot', ''),
            }
    elif 'lora_' in config_tag:
        # LoRA: imagenet_baseline_lora_{r}_{alpha}_{lr}_{epochs}_{wd}_{k}shot
        if len(parts) >= 9:
            return {
                'method': 'LoRA',
                'r': parts[3],
                'alpha': parts[4].replace('p', '.'),
                'lr': parts[5].replace('p', '.'),
                'epochs': parts[6],
                'wd': parts[7].replace('p', '.'),
                'k': parts[8].replace('shot', ''),
            }
    
    return {}


def load_json_results(json_path: str) -> Optional[Dict[str, Any]]:
    """JSON 파일에서 결과를 로드합니다."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"JSON 로드 실패 {json_path}: {e}")
        return None


def extract_accuracies_from_results(data: Dict[str, Any]) -> Dict[str, float]:
    """
    결과 데이터에서 정확도를 추출합니다.
    
    Returns:
        {
            'imagenet': 65.5,
            'imagenet_a': 45.2,
            'imagenet_r': 70.1,
            'imagenet_sketch': 55.3,
            'imagenet_v2': 60.8,
        }
    """
    accuracies = {}
    
    # all_eval_accuracies 필드가 있으면 우선 사용 (모든 데이터셋 결과 포함)
    if 'all_eval_accuracies' in data and isinstance(data['all_eval_accuracies'], dict):
        all_eval = data['all_eval_accuracies']
        dataset_name_map = {
            'imagenet-a': 'imagenet_a',
            'imagenet-r': 'imagenet_r',
            'imagenet-sketch': 'imagenet_sketch',
            'imagenet-v2': 'imagenet_v2',
            'imageneta': 'imagenet_a',
            'imagenetr': 'imagenet_r',
            'imagenetsketch': 'imagenet_sketch',
            'imagenetv2': 'imagenet_v2',
            'imagenetv2mfval': 'imagenet_v2',
            'imagenetilsvrcval': 'imagenet',
            'imagenetilsvrc': 'imagenet',
        }
        
        for dataset_key, acc_value in all_eval.items():
            dataset_normalized = dataset_name_map.get(dataset_key.lower(), dataset_key.lower())
            if isinstance(acc_value, (int, float)):
                acc = float(acc_value)
                accuracies[dataset_normalized] = acc * 100 if acc <= 1.0 else acc
        
        return accuracies
    
    # 기존 방식: 개별 필드에서 추출
    # ID 데이터셋 (ImageNet-1k val)
    if 'final_accuracy' in data:
        acc = data['final_accuracy']
        accuracies['imagenet'] = acc * 100 if acc <= 1.0 else acc
    elif 'validation_history' in data and data['validation_history']:
        best_acc = max(record.get('accuracy', 0) for record in data['validation_history'])
        accuracies['imagenet'] = best_acc * 100 if best_acc <= 1.0 else best_acc
    
    # OOD 데이터셋
    # 'ood_results'와 'ood_accuracies' 둘 다 체크 (키 이름 불일치 대응)
    ood_datasets = data.get('ood_accuracies', data.get('ood_results', {}))
    dataset_name_map = {
        'imagenet-a': 'imagenet_a',
        'imagenet-r': 'imagenet_r',
        'imagenet-sketch': 'imagenet_sketch',
        'imagenet-v2': 'imagenet_v2',
        'imageneta': 'imagenet_a',
        'imagenetr': 'imagenet_r',
        'imagenetsketch': 'imagenet_sketch',
        'imagenetv2': 'imagenet_v2',
        'imagenetv2mfval': 'imagenet_v2',
    }
    
    for dataset_key, results in ood_datasets.items():
        dataset_normalized = dataset_name_map.get(dataset_key.lower(), dataset_key.lower())
        if isinstance(results, dict) and 'accuracy' in results:
            acc = results['accuracy']
            accuracies[dataset_normalized] = acc * 100 if acc <= 1.0 else acc
        elif isinstance(results, (int, float)):
            acc = results
            accuracies[dataset_normalized] = acc * 100 if acc <= 1.0 else acc
    
    return accuracies


def discover_imagenet_results(model_location: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    ImageNet 실험 결과를 탐색하고 구조화합니다.
    
    Returns:
        {
            'ViT-B-32': {
                '4shots': [
                    {
                        'method': 'Energy',
                        'config_tag': 'imagenet_energy_0p001_12_average_0p1_0p0_4shot',
                        'hyperparams': {...},
                        'accuracies': {'imagenet': 65.5, 'imagenet_a': 45.2, ...},
                    },
                    ...
                ],
                '16shots': [...],
            },
            'ViT-L-14': {...},
        }
    """
    results = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(model_location):
        logger.error(f"모델 위치를 찾을 수 없습니다: {model_location}")
        return {}
    
    # 모델별로 순회
    for model_name in sorted(os.listdir(model_location)):
        model_path = os.path.join(model_location, model_name)
        if not os.path.isdir(model_path):
            continue
        
        logger.info(f"모델 처리 중: {model_name}")
        
        # ImageNet 데이터셋 폴더 찾기
        for dataset_entry in sorted(os.listdir(model_path)):
            # ImageNetILSVRCVal 또는 ImageNet 관련 폴더
            if 'imagenet' not in dataset_entry.lower():
                continue
            
            dataset_path = os.path.join(model_path, dataset_entry)
            if not os.path.isdir(dataset_path):
                continue
            
            # Config 디렉토리 순회
            for config_entry in sorted(os.listdir(dataset_path)):
                config_path = os.path.join(dataset_path, config_entry)
                if not os.path.isdir(config_path):
                    continue
                
                # Shot 디렉토리 순회
                for shot_entry in sorted(os.listdir(config_path)):
                    shot_path = os.path.join(config_path, shot_entry)
                    if not os.path.isdir(shot_path):
                        continue
                    
                    if not (shot_entry.endswith('shots') or shot_entry.endswith('shot')):
                        continue
                    
                    shot_name = shot_entry if shot_entry.endswith('shots') else f"{shot_entry}s"
                    
                    # 결과 파일 찾기
                    process_imagenet_shot_directory(
                        results, model_name, shot_name, shot_path, config_entry
                    )
    
    return dict(results)


def process_imagenet_shot_directory(
    results: Dict,
    model_name: str,
    shot_name: str,
    shot_path: str,
    config_tag: str
) -> None:
    """ImageNet shot 디렉토리를 처리하고 결과를 추출합니다."""
    # Energy 결과
    energy_json = os.path.join(shot_path, 'energy_results_imagenet.json')
    if os.path.exists(energy_json):
        data = load_json_results(energy_json)
        if data:
            accuracies = extract_accuracies_from_results(data)
            hyperparams = parse_energy_config_tag(config_tag)
            
            if accuracies and hyperparams:
                results[model_name][shot_name].append({
                    'method': 'Energy',
                    'config_tag': config_tag,
                    'hyperparams': hyperparams,
                    'accuracies': accuracies,
                })
                logger.debug(f"Energy 결과 추가: {model_name}/{shot_name} - {accuracies.get('imagenet', 0):.2f}%")
    
    # Atlas 결과
    atlas_json = os.path.join(shot_path, 'atlas_results_imagenet.json')
    if os.path.exists(atlas_json):
        data = load_json_results(atlas_json)
        if data:
            accuracies = extract_accuracies_from_results(data)
            hyperparams = parse_atlas_config_tag(config_tag)
            
            if accuracies and hyperparams:
                results[model_name][shot_name].append({
                    'method': 'Atlas',
                    'config_tag': config_tag,
                    'hyperparams': hyperparams,
                    'accuracies': accuracies,
                })
                logger.debug(f"Atlas 결과 추가: {model_name}/{shot_name} - {accuracies.get('imagenet', 0):.2f}%")
    
    # Baseline 결과
    baseline_json = os.path.join(shot_path, 'baseline_results_imagenet.json')
    if os.path.exists(baseline_json):
        data = load_json_results(baseline_json)
        if data:
            accuracies = extract_accuracies_from_results(data)
            hyperparams = parse_baseline_config_tag(config_tag)
            
            if accuracies and hyperparams:
                method_name = hyperparams.pop('method', 'Baseline')
                results[model_name][shot_name].append({
                    'method': method_name,
                    'config_tag': config_tag,
                    'hyperparams': hyperparams,
                    'accuracies': accuracies,
                })
                logger.debug(f"{method_name} 결과 추가: {model_name}/{shot_name} - {accuracies.get('imagenet', 0):.2f}%")


def format_method_label(method: str, hyperparams: Dict[str, str]) -> str:
    """메소드 라벨을 포맷팅합니다."""
    if method == 'Energy':
        lr = hyperparams.get('sigma_lr', '?')
        topk = hyperparams.get('topk', '?')
        init = hyperparams.get('init_mode', '?')
        warmup = hyperparams.get('warmup_ratio', '?')
        wd = hyperparams.get('sigma_wd', '?')
        return f"Energy(lr={lr}, k={topk}, init={init}, w={warmup}, wd={wd})"
    elif method == 'Atlas':
        lr = hyperparams.get('lr', '?')
        num_basis = hyperparams.get('num_basis', '?')
        return f"Atlas(lr={lr}, basis={num_basis})"
    elif method == 'LinearProbe':
        lr = hyperparams.get('lr', '?')
        epochs = hyperparams.get('epochs', '?')
        return f"LinearProbe(lr={lr}, ep={epochs})"
    elif method == 'LoRA':
        r = hyperparams.get('r', '?')
        alpha = hyperparams.get('alpha', '?')
        lr = hyperparams.get('lr', '?')
        return f"LoRA(r={r}, α={alpha}, lr={lr})"
    elif method in ['TIP', 'LP++']:
        wd = hyperparams.get('wd', '?')
        return f"{method}(wd={wd})"
    else:
        return method


def visualize_model_shot_results(
    model_name: str,
    shot_name: str,
    results: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """특정 모델/shot 조합의 결과를 시각화합니다."""
    if not results:
        logger.warning(f"시각화할 결과가 없습니다: {model_name}/{shot_name}")
        return
    
    # 데이터셋 목록
    dataset_order = ['imagenet', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2']
    dataset_display = {
        'imagenet': 'ImageNet',
        'imagenet_a': 'ImageNet-A',
        'imagenet_r': 'ImageNet-R',
        'imagenet_sketch': 'ImageNet-Sketch',
        'imagenet_v2': 'ImageNet-V2',
    }
    
    # ImageNet ID 정확도로 정렬
    results_sorted = sorted(results, key=lambda x: x['accuracies'].get('imagenet', 0), reverse=True)
    
    # 각 메소드의 라벨과 정확도 준비
    method_labels = [format_method_label(r['method'], r['hyperparams']) for r in results_sorted]
    
    # 데이터셋별로 서브플롯 생성
    n_datasets = len([d for d in dataset_order if any(d in r['accuracies'] for r in results_sorted)])
    
    if n_datasets == 0:
        logger.warning(f"시각화할 데이터셋이 없습니다: {model_name}/{shot_name}")
        return
    
    fig_height = max(8, len(results) * 0.4 + 3)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, fig_height))
    
    if n_datasets == 1:
        axes = [axes]
    
    plot_idx = 0
    for dataset_key in dataset_order:
        # 이 데이터셋에 대한 결과가 있는지 확인
        if not any(dataset_key in r['accuracies'] for r in results_sorted):
            continue
        
        ax = axes[plot_idx]
        accuracies = [r['accuracies'].get(dataset_key, 0) for r in results_sorted]
        
        # 막대 그래프 생성
        y_positions = range(len(method_labels))
        bars = ax.barh(y_positions, accuracies, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 최고 결과 강조
        if bars and accuracies:
            max_acc = max(accuracies)
            max_idx = accuracies.index(max_acc)
            bars[max_idx].set_color('darkgreen')
            bars[max_idx].set_linewidth(1.5)
        
        # 레이블 및 스타일링
        ax.set_yticks(y_positions)
        ax.set_yticklabels(method_labels, fontsize=8)
        ax.set_xlabel('정확도 (%)', fontsize=10, fontweight='bold')
        ax.set_title(dataset_display[dataset_key], fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 정확도 값 표시
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                width = bar.get_width()
                ax.text(
                    width + 1, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.1f}%',
                    ha='left', va='center',
                    fontsize=7
                )
        
        plot_idx += 1
    
    # 전체 제목
    shot_display = shot_name.replace('shots', '-shot')
    fig.suptitle(f'{model_name} | {shot_display}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"imagenet_{model_name}_{shot_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"저장됨: {output_path}")


def select_best_configs_per_method(
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Dict[str, Dict[str, Tuple[str, Dict[str, float]]]]]:
    """
    각 모델/shot/메소드 조합에서 최고 성능 설정을 선택합니다 (ImageNet ID 기준).
    
    Returns:
        {
            'ViT-B-32': {
                '16shots': {
                    'Energy': (config_tag, accuracies),
                    'Atlas': (config_tag, accuracies),
                    'LinearProbe': (config_tag, accuracies),
                    ...
                },
            },
        }
    """
    best_configs = defaultdict(lambda: defaultdict(dict))
    
    for model_name, shots in all_results.items():
        for shot_name, results in shots.items():
            # 메소드별로 그룹화
            method_results = defaultdict(list)
            for result in results:
                method_results[result['method']].append(result)
            
            # 각 메소드에서 최고 성능 선택 (ImageNet ID 기준)
            for method, method_list in method_results.items():
                best_result = max(method_list, key=lambda x: x['accuracies'].get('imagenet', 0))
                best_configs[model_name][shot_name][method] = (
                    best_result['config_tag'],
                    best_result['accuracies']
                )
    
    return dict(best_configs)


def create_summary_table(
    best_configs: Dict[str, Dict[str, Dict[str, Tuple[str, Dict[str, float]]]]],
    output_dir: str
) -> None:
    """최고 성능 설정을 요약한 테이블을 생성합니다."""
    # 데이터셋 순서
    dataset_order = ['imagenet', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2']
    dataset_display = {
        'imagenet': 'ImageNet',
        'imagenet_a': 'ImageNet-A',
        'imagenet_r': 'ImageNet-R',
        'imagenet_sketch': 'ImageNet-Sketch',
        'imagenet_v2': 'ImageNet-V2',
    }
    
    # Shot 순서
    shot_order = ['4shots', '16shots']
    
    # 모델 목록
    models = sorted(best_configs.keys())
    
    # 메소드 목록 수집
    all_methods = set()
    for model_data in best_configs.values():
        for shot_data in model_data.values():
            all_methods.update(shot_data.keys())
    methods = sorted(all_methods)
    
    # 테이블 데이터 구조: method -> model -> shot -> dataset -> accuracy
    table_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for model_name in models:
        for shot_name in shot_order:
            if shot_name not in best_configs[model_name]:
                continue
            
            for method_name in methods:
                if method_name not in best_configs[model_name][shot_name]:
                    continue
                
                _, accuracies = best_configs[model_name][shot_name][method_name]
                for dataset in dataset_order:
                    if dataset in accuracies:
                        table_data[method_name][model_name][shot_name][dataset] = accuracies[dataset]
    
    # CSV 파일 생성
    csv_path = os.path.join(output_dir, 'imagenet_summary_table.csv')
    with open(csv_path, 'w') as f:
        # 헤더
        header = ['Method', 'Model', 'Shot'] + [dataset_display[d] for d in dataset_order]
        f.write(','.join(header) + '\n')
        
        # 데이터 행
        for method in methods:
            for model in models:
                for shot in shot_order:
                    row = [method, model, shot.replace('shots', '')]
                    for dataset in dataset_order:
                        acc = table_data[method][model][shot].get(dataset, None)
                        row.append(f"{acc:.2f}" if acc is not None else "-")
                    f.write(','.join(row) + '\n')
    
    logger.info(f"요약 테이블 저장됨: {csv_path}")
    
    # 시각화 테이블 생성
    fig = plt.figure(figsize=(20, max(8, len(methods) * len(models) * len(shot_order) * 0.3)))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # 테이블 컨텐츠 준비
    table_content = []
    for method in methods:
        for model in models:
            for shot in shot_order:
                row = [
                    method if model == models[0] and shot == shot_order[0] else '',
                    model if shot == shot_order[0] else '',
                    shot.replace('shots', '')
                ]
                for dataset in dataset_order:
                    acc = table_data[method][model][shot].get(dataset, None)
                    row.append(f"{acc:.2f}" if acc is not None else "-")
                table_content.append(row)
    
    # 테이블 생성
    table = ax.table(
        cellText=table_content,
        colLabels=header,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(len(header)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # 각 데이터셋 열에서 최고값 강조
    for col_idx in range(3, len(header)):
        values = []
        for row_idx in range(len(table_content)):
            cell_text = table_content[row_idx][col_idx]
            if cell_text != "-":
                try:
                    values.append((float(cell_text), row_idx))
                except:
                    pass
        
        if values:
            max_val, max_row_idx = max(values, key=lambda x: x[0])
            cell = table[(max_row_idx + 1, col_idx)]
            cell.set_facecolor('#FFEB3B')
            cell.set_text_props(weight='bold')
    
    plt.title('ImageNet 실험 결과 요약 (최고 성능 설정)', fontsize=14, fontweight='bold', pad=20)
    
    # 저장
    png_path = os.path.join(output_dir, 'imagenet_summary_table.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"요약 테이블 이미지 저장됨: {png_path}")


def save_best_configs_json(
    best_configs: Dict[str, Dict[str, Dict[str, Tuple[str, Dict[str, float]]]]],
    output_dir: str
) -> None:
    """최고 성능 설정을 JSON 파일로 저장합니다."""
    output_path = os.path.join(output_dir, 'imagenet_best_configs.json')
    
    # 직렬화 가능한 형식으로 변환
    serializable = {}
    for model_name, shots in best_configs.items():
        serializable[model_name] = {}
        for shot_name, methods in shots.items():
            serializable[model_name][shot_name] = {}
            for method_name, (config_tag, accuracies) in methods.items():
                serializable[model_name][shot_name][method_name] = {
                    'config_tag': config_tag,
                    'accuracies': accuracies
                }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    logger.info(f"최고 성능 설정 저장됨: {output_path}")


def create_per_shot_csv(
    best_configs: Dict[str, Dict[str, Dict[str, Tuple[str, Dict[str, float]]]]],
    output_dir: str
) -> None:
    """shot(=k-shot) 별로 모든 모델/메소드의 최고 성능을 CSV로 저장합니다."""
    # 데이터셋 순서
    dataset_order = ['imagenet', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2']
    dataset_display = {
        'imagenet': 'ImageNet',
        'imagenet_a': 'ImageNet-A',
        'imagenet_r': 'ImageNet-R',
        'imagenet_sketch': 'ImageNet-Sketch',
        'imagenet_v2': 'ImageNet-V2',
    }
    
    # 모델/메소드/샷 목록 수집
    models = sorted(best_configs.keys())
    all_methods = set()
    shot_set = set()
    for model_data in best_configs.values():
        for shot_name, methods in model_data.items():
            shot_set.add(shot_name)
            all_methods.update(methods.keys())
    
    def _shot_sort_key(s: str):
        try:
            return int(s.replace('shots', '').replace('shot', ''))
        except Exception:
            return s
    
    shot_order = sorted(shot_set, key=_shot_sort_key)
    methods = sorted(all_methods)
    
    # shot별 CSV 생성
    for shot in shot_order:
        csv_path = os.path.join(output_dir, f'imagenet_by_shot_{shot}.csv')
        with open(csv_path, 'w') as f:
            header = ['Method', 'Model'] + [dataset_display[d] for d in dataset_order]
            f.write(','.join(header) + '\n')
            
            for method in methods:
                for model in models:
                    row = [method, model]
                    accuracies = None
                    if model in best_configs and shot in best_configs[model] and method in best_configs[model][shot]:
                        _, accuracies = best_configs[model][shot][method]
                    for dataset in dataset_order:
                        val = accuracies.get(dataset) if accuracies else None
                        row.append(f"{val:.2f}" if val is not None else "-")
                    f.write(','.join(row) + '\n')
        
        logger.info(f"Shot별 CSV 저장됨: {csv_path}")


def create_per_model_csv(
    best_configs: Dict[str, Dict[str, Dict[str, Tuple[str, Dict[str, float]]]]],
    output_dir: str
) -> None:
    """모델(백본) 별로 모든 shot/메소드의 최고 성능을 CSV로 저장합니다."""
    # 데이터셋 순서
    dataset_order = ['imagenet', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2']
    dataset_display = {
        'imagenet': 'ImageNet',
        'imagenet_a': 'ImageNet-A',
        'imagenet_r': 'ImageNet-R',
        'imagenet_sketch': 'ImageNet-Sketch',
        'imagenet_v2': 'ImageNet-V2',
    }
    
    # 모델/메소드/샷 목록 수집
    models = sorted(best_configs.keys())
    all_methods = set()
    shot_set = set()
    for model_data in best_configs.values():
        for shot_name, methods in model_data.items():
            shot_set.add(shot_name)
            all_methods.update(methods.keys())
    
    def _shot_sort_key(s: str):
        try:
            return int(s.replace('shots', '').replace('shot', ''))
        except Exception:
            return s
    
    shot_order = sorted(shot_set, key=_shot_sort_key)
    methods = sorted(all_methods)
    
    # 모델별 CSV 생성
    for model in models:
        csv_path = os.path.join(output_dir, f'imagenet_by_model_{model}.csv')
        with open(csv_path, 'w') as f:
            header = ['Method', 'Shot'] + [dataset_display[d] for d in dataset_order]
            f.write(','.join(header) + '\n')
            
            for method in methods:
                for shot in shot_order:
                    row = [method, shot.replace('shots', '')]
                    accuracies = None
                    if method in best_configs.get(model, {}).get(shot, {}):
                        _, accuracies = best_configs[model][shot][method]
                    for dataset in dataset_order:
                        val = accuracies.get(dataset) if accuracies else None
                        row.append(f"{val:.2f}" if val is not None else "-")
                    f.write(','.join(row) + '\n')
        
        logger.info(f"모델별 CSV 저장됨: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='ImageNet 실험 결과를 집계하고 시각화합니다'
    )
    parser.add_argument(
        '--model_location',
        type=str,
        default='./models/checkpoints',
        help='모델 체크포인트가 있는 루트 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./imagenet_results',
        help='시각화 결과를 저장할 디렉토리'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='처리할 특정 모델들 (예: ViT-B-32 ViT-L-14)'
    )
    parser.add_argument(
        '--shots',
        nargs='*',
        default=None,
        help='처리할 특정 shot들 (예: 4shots 16shots)'
    )
    parser.add_argument(
        '--summary_only',
        action='store_true',
        help='요약 테이블만 생성 (개별 차트 스킵)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ImageNet 결과 집계 및 시각화")
    logger.info("=" * 80)
    logger.info(f"모델 위치: {args.model_location}")
    logger.info(f"출력 디렉토리: {args.output_dir}")
    
    # 결과 탐색
    logger.info("\n결과 탐색 중...")
    all_results = discover_imagenet_results(args.model_location)
    
    if not all_results:
        logger.error("결과를 찾을 수 없습니다!")
        return
    
    # 필터 적용
    if args.models:
        all_results = {k: v for k, v in all_results.items() if k in args.models}
    if args.shots:
        shots_normalized = [s if s.endswith('shots') else f"{s}shots" for s in args.shots]
        for model in all_results:
            all_results[model] = {
                k: v for k, v in all_results[model].items() if k in shots_normalized
            }
    
    logger.info(f"\n{len(all_results)}개 모델 발견")
    for model_name, shots in all_results.items():
        total_results = sum(len(results) for results in shots.values())
        logger.info(f"  {model_name}: {len(shots)}개 shot 설정, {total_results}개 결과")
    
    # 최고 성능 설정 선택
    logger.info("\n메소드별 최고 성능 설정 선택 중...")
    best_configs = select_best_configs_per_method(all_results)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 요약 테이블 생성
    logger.info("\n요약 테이블 생성 중...")
    create_summary_table(best_configs, args.output_dir)
    
    # 최고 성능 설정 JSON 저장
    logger.info("\n최고 성능 설정 JSON 저장 중...")
    save_best_configs_json(best_configs, args.output_dir)
    
    # shot/모델별 CSV 저장
    logger.info("\nk-shot 및 모델별 CSV 저장 중...")
    create_per_shot_csv(best_configs, args.output_dir)
    create_per_model_csv(best_configs, args.output_dir)
    
    # 개별 시각화 생성
    if not args.summary_only:
        total_combinations = sum(len(shots) for shots in all_results.values())
        logger.info(f"\n개별 시각화 생성 중... (총 {total_combinations}개 조합)")
        
        count = 0
        for model_name, shots in sorted(all_results.items()):
            for shot_name, results in sorted(shots.items()):
                count += 1
                logger.info(f"[{count}/{total_combinations}] 처리 중: {model_name}/{shot_name} ({len(results)}개 결과)")
                visualize_model_shot_results(model_name, shot_name, results, args.output_dir)
    
    logger.info("=" * 80)
    logger.info(f"✓ 완료! 결과가 {args.output_dir}에 저장되었습니다")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

