import os
import time
import json
import logging
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import sys
import torchvision
from tqdm.auto import tqdm
import math

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import (
    ALL_DATASETS,
    TQDM_BAR_FORMAT,
    get_energy_finetuned_path,
    get_finetuned_path,
    get_zeroshot_path,
)
from src.datasets import get_dataloader, maybe_dictionarize, get_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.sigma_param import SigmaParametrization
from src.eval.eval import eval_single_dataset
import torch
from src.models.task_vectors import NonLinearTaskVector
from torch.nn.utils.stateless import functional_call
from src.utils.utils import cosine_lr
from atlas_reverse import train_adapter

from src.datasets.remote_sensing import sample_k_shot_indices


def save_k_shot_indices(indices, save_dir, dataset_name, k, seed):
    """Save k-shot indices to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    with open(indices_path, 'w') as f:
        json.dump({"indices": indices, "dataset": dataset_name, "k": k, "seed": seed}, f)
    return indices_path


def load_k_shot_indices(save_dir, k, seed):
    """Load k-shot indices from a JSON file if it exists."""
    indices_path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    if os.path.exists(indices_path):
        with open(indices_path, 'r') as f:
            data = json.load(f)
            return data["indices"]
    return None


def subsample_from_larger_k(larger_indices, dataset, target_k, seed):
    """Subsample target_k indices per class from a larger k-shot set.
    
    This is much faster than re-sampling from the entire dataset.
    Uses deterministic selection (first target_k samples per class).
    """
    import numpy as np
    
    # Extract labels for the larger indices
    labels = []
    for idx in larger_indices:
        _, label = dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        elif isinstance(label, np.ndarray):
            label = int(label)
        labels.append(int(label))
    
    # Group indices by class
    class_to_indices = {}
    for idx, label in zip(larger_indices, labels):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # Select first target_k from each class (deterministic)
    selected_indices = []
    for label in sorted(class_to_indices.keys()):
        class_indices = class_to_indices[label][:target_k]
        selected_indices.extend(class_indices)
    
    return selected_indices



class AdapterCompatibleClassifier(torch.nn.Module):
    """Wrap an ImageClassifier to expose return_features inference for adapters."""

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    @property
    def image_encoder(self):
        return self.base_model.image_encoder

    @property
    def classification_head(self):
        return self.base_model.classification_head

    @property
    def train_preprocess(self):
        return getattr(self.base_model, "train_preprocess", None)

    @property
    def val_preprocess(self):
        return getattr(self.base_model, "val_preprocess", None)

    def freeze_head(self):
        return self.base_model.freeze_head()

    def forward(self, inputs, return_features: bool = False):
        features = self.image_encoder(inputs)
        logits = self.classification_head(features)
        if return_features:
            norm = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return logits, features / norm
        return logits

def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in str(val).replace('.', 'p'))


def build_energy_config_tag(cfg) -> str:
    num_tasks_minus_one = _sanitize_value(max(int(getattr(cfg, "num_tasks", 0)) - 1, 0))
    lr_part = _sanitize_value(cfg.sigma_lr)
    svd_part = _sanitize_value(getattr(cfg, "svd_keep_topk", 2))
    init_mode_part = _sanitize_value(getattr(cfg, "initialize_sigma", "average"))
    warmup_ratio_part = _sanitize_value(getattr(cfg, "warmup_ratio", 0.1))
    return f"energy_{num_tasks_minus_one}_{lr_part}_{svd_part}_{init_mode_part}_{warmup_ratio_part}"


def normalize_adapter_choice(value: str) -> str:
    val = (value or "none").strip().lower()
    if val in {"lp++", "lpp"}:
        return "lp++"
    if val == "tip":
        return "tip"
    return "none"


def adapter_path_tag(display: str) -> str:
    display = (display or "none").strip().lower()
    if display in {"", "none"}:
        return "none"
    if display == "lp++":
        return "lp++"
    return display.replace(" ", "_")


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def setup_simple_logger(name: str = __name__) -> logging.Logger:
    """Setup a clean logger with minimal formatting (no INFO/DEBUG prefixes)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with simple format (just the message)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


# Dataset-specific epochs for sigma training (general datasets)
SIGMA_EPOCHS_PER_DATASET = {
    # "Cars": 35,
    "DTD": 76,
    # "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    # "RESISC45": 15,
    # "SUN397": 14,
    "SVHN": 4,
    "CIFAR10": 6,
    "CIFAR100": 6,
    "STL10": 60,
    "Food101": 4,
    "Flowers102": 147,
    # "FER2013": 10,
    "PCAM": 1,
    "OxfordIIITPet": 82,
    "RenderedSST2": 39,
    "EMNIST": 2,
    "FashionMNIST": 5,
    "KMNIST": 5,
}


def compute_eval_epochs(total_epochs: int, max_evals: int = 5) -> set:
    total_epochs = max(int(total_epochs), 1)
    if total_epochs <= max_evals:
        return set(range(total_epochs))
    eval_epochs = {
        min(total_epochs - 1, int(round(i * (total_epochs - 1) / (max_evals - 1))))
        for i in range(max_evals)
    }
    return eval_epochs

def compute_and_sum_svd_mem_reduction_sum(task_vectors, config):
    """
    여러 태스크 벡터의 2D 가중치에 대해 SVD를 수행,
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과
    '재계산된 평균' sigma(diag)로 재구성.
    """
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 3)))
    with torch.no_grad():
        new_vector = {}
        # 공통 필터 함수(임베딩류 제외)
        def is_matrix_key(tv0, key):
            return (
                tv0.vector[key].ndim == 2 and
                all(t not in key for t in ("text_projection",
                    "positional", "token_embedding"))
            )
        # 키 순회
        tv0 = task_vectors[0]
        for key in tv0.vector:
            # 2D 행렬이 아니거나 제외 키면: 단순 평균
            if not is_matrix_key(tv0, key):
                avg = None
                for i, tv in enumerate(task_vectors):
                    vec = tv.vector[key].to(device)
                    avg = vec.clone() if i == 0 else avg + (vec - avg) / (i + 1)
                new_vector[key] = avg
                continue
            # -------- SVD 축 모으기 준비 --------
            # 첫 태스크에서 모양/순위 파악
            vec0 = task_vectors[0].vector[key].to(device)
            u0, s0, vh0 = torch.linalg.svd(vec0, full_matrices=False)
            m = int(u0.shape[0])
            r = int(s0.shape[0])          # 유효 랭크 상한
            n = int(vh0.shape[1])
            if r == 0:
                # 드물지만 0-rank 보호장치
                new_vector[key] = torch.zeros_like(vec0)
                continue
            # 사용할 태스크 수: r 보다 많은 태스크를 모두 쓰면 k가 0이 될 수 있으니 cap
            num_used = min(num_tasks, r)
            # 태스크당 축 수 k: floor로 잡으면 k*num_used <= r 보장
            # k = max(1, r // num_used)
            max_per_task = max(1, r // num_used)
            k = min(desired_k, max_per_task)
            if desired_k > max_per_task:
                print(
                    "[SVD] Requested %d comps/task but only %d available for %s; using %d.",
                    desired_k,
                    max_per_task,
                    key,
                    k,
                )
            chunks = int(k * num_used)    # <= r 보장
            # print(f"chunks: {chunks}")
            # 버퍼를 '정수 차원'으로 명시해 생성
            sum_u = torch.zeros((m, chunks), device=device, dtype=u0.dtype)
            # sum_s = torch.zeros((chunks,), device=device, dtype=s0.dtype) # [수정] 더 이상 sum_s 필요 없음
            sum_v = torch.zeros((chunks, n), device=device, dtype=vh0.dtype)
            # 각 태스크에서 상위 k개 축만 수집
            for i, tv in enumerate(task_vectors[:num_used]):
                vec = tv.vector[key].to(device)
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                # 실제 s 길이가 r보다 작을 수도 있으므로 k를 매 태스크마다 보정
                r_i = int(s.shape[0])
                k_i = min(k, r_i)  # 안전 클램프
                start = i * k
                end = start + k_i  # 마지막 태스크에서 k_i < k일 수 있음
                sum_u[:, start:end] = u[:, :k_i]
                # sum_s[start:end] = s[:k_i] # [수정] 더 이상 sum_s 필요 없음
                sum_v[start:end, :] = vh[:k_i, :]
            # 직교화(각 집합에 대해 다시 SVD → U*Vh)
            u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_orth = u_u @ vh_u          # (m × chunks)
            V_orth = u_v @ vh_v          # (chunks × n)
            # -------- [수정된 Sigma 계산 로직 시작] --------
            # 1. 각 태스크 벡터(M_i)를 공통 기저(U_orth, V_orth)로 투영하여
            #    최적의 대각 행렬(sigma_task)을 찾습니다.
            #    M_i ≈ U_orth @ Sigma_i @ V_orth 이므로,
            #    Sigma_i' = U_orth.T @ M_i @ V_orth.T 를 계산합니다.
            all_sigma_diags = []
            # U_orth (m, chunks) -> U_orth.T (chunks, m)
            # V_orth (chunks, n) -> V_orth.T (n, chunks)
            U_orth_T = U_orth.T
            V_orth_T = V_orth.T
            # "모든" 태스크 벡터에 대해 반복 (num_used 뿐만 아니라)
            for tv in task_vectors:
                M_i = tv.vector[key].to(device)  # 원본 태스크 행렬 (m, n)
                # Sigma_i_prime = (U_orth.T @ M_i) @ V_orth.T
                # (chunks, m) @ (m, n) @ (n, chunks) -> (chunks, chunks)
                Sigma_i_prime = (U_orth_T @ M_i) @ V_orth_T
                # "이를 diagonal로 만들도록 해야해" -> 대각 성분만 추출
                sigma_task_diag = torch.diag(Sigma_i_prime)  # (chunks,)
                all_sigma_diags.append(sigma_task_diag)
            # 2. 모든 태스크의 sigma_task_diag를 평균냅니다.
            if not all_sigma_diags:
                # 엣지 케이스 방어
                Sigma = torch.zeros(
                    (chunks, chunks), device=device, dtype=u0.dtype)
            else:
                # (num_tasks, chunks)
                stacked_sigmas = torch.stack(all_sigma_diags, dim=0)
                # (chunks,)
                mean_sigma_diag = torch.sum(stacked_sigmas, dim=0)
                # 최종 Sigma: 평균낸 대각 성분으로 대각 행렬 생성
                Sigma = torch.diag(mean_sigma_diag)   # (chunks, chunks)
            # -------- [수정된 Sigma 계산 로직 끝] --------
            # 이후 단계에서 SigmaParametrization(U, V, sigma)로 사용
            new_vector[key] = [U_orth, Sigma, V_orth]
    return new_vector

def compute_and_sum_svd_mem_reduction_average(task_vectors, config):
    """
    여러 태스크 벡터의 2D 가중치에 대해 SVD를 수행,
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과
    '재계산된 평균' sigma(diag)로 재구성.
    """
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 3)))
    with torch.no_grad():
        new_vector = {}
        # 공통 필터 함수(임베딩류 제외)
        def is_matrix_key(tv0, key):
            return (
                tv0.vector[key].ndim == 2 and
                all(t not in key for t in ("text_projection",
                    "positional", "token_embedding"))
            )
        # 키 순회
        tv0 = task_vectors[0]
        for key in tv0.vector:
            # 2D 행렬이 아니거나 제외 키면: 단순 평균
            if not is_matrix_key(tv0, key):
                avg = None
                for i, tv in enumerate(task_vectors):
                    vec = tv.vector[key].to(device)
                    avg = vec.clone() if i == 0 else avg + (vec - avg) / (i + 1)
                new_vector[key] = avg
                continue
            # -------- SVD 축 모으기 준비 --------
            # 첫 태스크에서 모양/순위 파악
            vec0 = task_vectors[0].vector[key].to(device)
            u0, s0, vh0 = torch.linalg.svd(vec0, full_matrices=False)
            m = int(u0.shape[0])
            r = int(s0.shape[0])          # 유효 랭크 상한
            n = int(vh0.shape[1])
            if r == 0:
                # 드물지만 0-rank 보호장치
                new_vector[key] = torch.zeros_like(vec0)
                continue
            # 사용할 태스크 수: r 보다 많은 태스크를 모두 쓰면 k가 0이 될 수 있으니 cap
            num_used = min(num_tasks, r)
            # 태스크당 축 수 k: floor로 잡으면 k*num_used <= r 보장
            # k = max(1, r // num_used)
            max_per_task = max(1, r // num_used)
            k = min(desired_k, max_per_task)
            if desired_k > max_per_task:
                print(
                    "[SVD] Requested %d comps/task but only %d available for %s; using %d.",
                    desired_k,
                    max_per_task,
                    key,
                    k,
                )
            chunks = int(k * num_used)    # <= r 보장
            # print(f"chunks: {chunks}")
            # 버퍼를 '정수 차원'으로 명시해 생성
            sum_u = torch.zeros((m, chunks), device=device, dtype=u0.dtype)
            # sum_s = torch.zeros((chunks,), device=device, dtype=s0.dtype) # [수정] 더 이상 sum_s 필요 없음
            sum_v = torch.zeros((chunks, n), device=device, dtype=vh0.dtype)
            # 각 태스크에서 상위 k개 축만 수집
            for i, tv in enumerate(task_vectors[:num_used]):
                vec = tv.vector[key].to(device)
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                # 실제 s 길이가 r보다 작을 수도 있으므로 k를 매 태스크마다 보정
                r_i = int(s.shape[0])
                k_i = min(k, r_i)  # 안전 클램프
                start = i * k
                end = start + k_i  # 마지막 태스크에서 k_i < k일 수 있음
                sum_u[:, start:end] = u[:, :k_i]
                # sum_s[start:end] = s[:k_i] # [수정] 더 이상 sum_s 필요 없음
                sum_v[start:end, :] = vh[:k_i, :]
            # 직교화(각 집합에 대해 다시 SVD → U*Vh)
            u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_orth = u_u @ vh_u          # (m × chunks)
            V_orth = u_v @ vh_v          # (chunks × n)
            # -------- [수정된 Sigma 계산 로직 시작] --------
            # 1. 각 태스크 벡터(M_i)를 공통 기저(U_orth, V_orth)로 투영하여
            #    최적의 대각 행렬(sigma_task)을 찾습니다.
            #    M_i ≈ U_orth @ Sigma_i @ V_orth 이므로,
            #    Sigma_i' = U_orth.T @ M_i @ V_orth.T 를 계산합니다.
            all_sigma_diags = []
            # U_orth (m, chunks) -> U_orth.T (chunks, m)
            # V_orth (chunks, n) -> V_orth.T (n, chunks)
            U_orth_T = U_orth.T
            V_orth_T = V_orth.T
            # "모든" 태스크 벡터에 대해 반복 (num_used 뿐만 아니라)
            for tv in task_vectors:
                M_i = tv.vector[key].to(device)  # 원본 태스크 행렬 (m, n)
                # Sigma_i_prime = (U_orth.T @ M_i) @ V_orth.T
                # (chunks, m) @ (m, n) @ (n, chunks) -> (chunks, chunks)
                Sigma_i_prime = (U_orth_T @ M_i) @ V_orth_T
                # "이를 diagonal로 만들도록 해야해" -> 대각 성분만 추출
                sigma_task_diag = torch.diag(Sigma_i_prime)  # (chunks,)
                all_sigma_diags.append(sigma_task_diag)
            # 2. 모든 태스크의 sigma_task_diag를 평균냅니다.
            if not all_sigma_diags:
                # 엣지 케이스 방어
                Sigma = torch.zeros(
                    (chunks, chunks), device=device, dtype=u0.dtype)
            else:
                # (num_tasks, chunks)
                stacked_sigmas = torch.stack(all_sigma_diags, dim=0)
                # (chunks,)
                mean_sigma_diag = torch.mean(stacked_sigmas, dim=0)
                # 최종 Sigma: 평균낸 대각 성분으로 대각 행렬 생성
                Sigma = torch.diag(mean_sigma_diag)   # (chunks, chunks)
            # -------- [수정된 Sigma 계산 로직 끝] --------
            # 이후 단계에서 SigmaParametrization(U, V, sigma)로 사용
            new_vector[key] = [U_orth, Sigma, V_orth]
    return new_vector

def compute_and_sum_svd_mem_reduction_tsvm(task_vectors, config):
    """
    여러 태스크 벡터의 2D 가중치에 대해 SVD를 수행,
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과 sigma(diag)로 재구성.
    """
    logger = setup_simple_logger(__name__)
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 2)))
    with torch.no_grad():
        new_vector = {}
        # 공통 필터 함수(임베딩류 제외)
        def is_matrix_key(tv0, key):
            return (
                tv0.vector[key].ndim == 2 and
                all(t not in key for t in ("text_projection", "positional", "token_embedding"))
            )
        # 키 순회
        tv0 = task_vectors[0]
        for key in tv0.vector:
            # 2D 행렬이 아니거나 제외 키면: 단순 평균
            if not is_matrix_key(tv0, key):
                avg = None
                for i, tv in enumerate(task_vectors):
                    vec = tv.vector[key].to(device)
                    avg = vec.clone() if i == 0 else avg + (vec - avg) / (i + 1)
                new_vector[key] = avg
                continue
            # -------- SVD 축 모으기 준비 --------
            # 첫 태스크에서 모양/순위 파악
            vec0 = task_vectors[0].vector[key].to(device)
            u0, s0, vh0 = torch.linalg.svd(vec0, full_matrices=False)
            m = int(u0.shape[0])
            r = int(s0.shape[0])          # 유효 랭크 상한
            n = int(vh0.shape[1])
            if r == 0:
                # 드물지만 0-rank 보호장치
                new_vector[key] = torch.zeros_like(vec0)
                continue
            # 사용할 태스크 수: r 보다 많은 태스크를 모두 쓰면 k가 0이 될 수 있으니 cap
            num_used = min(num_tasks, r)
            # # 태스크당 축 수 k: floor로 잡으면 k*num_used <= r 보장
            # k = max(1, r // num_used)
            # 원하는 축 개수(desired_k)를 가능한 범위로 클램프
            if num_used == 0:
                continue
            max_per_task = max(1, r // num_used)
            k = min(desired_k, max_per_task)
            if desired_k > max_per_task:
                logger.warning(
                    "[SVD] Requested %d comps/task but only %d available for %s; using %d.",
                    desired_k,
                    max_per_task,
                    key,
                    k,
                )
            chunks = int(k * num_used)    # <= r 보장
            # 버퍼를 '정수 차원'으로 명시해 생성
            sum_u = torch.zeros((m, chunks), device=device, dtype=u0.dtype)
            sum_s = torch.zeros((chunks,), device=device, dtype=s0.dtype)
            sum_v = torch.zeros((chunks, n), device=device, dtype=vh0.dtype)
            # 각 태스크에서 상위 k개 축만 수집
            for i, tv in enumerate(task_vectors[:num_used]):
                vec = tv.vector[key].to(device)
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                # 실제 s 길이가 r보다 작을 수도 있으므로 k를 매 태스크마다 보정
                r_i = int(s.shape[0])
                k_i = min(k, r_i)  # 안전 클램프
                start = i * k
                end = start + k_i  # 마지막 태스크에서 k_i < k일 수 있음
                sum_u[:, start:end] = u[:, :k_i]
                sum_s[start:end]    = s[:k_i]
                sum_v[start:end, :] = vh[:k_i, :]
            # 직교화(각 집합에 대해 다시 SVD → U*Vh)
            u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_orth = u_u @ vh_u          # (m × chunks)
            V_orth = u_v @ vh_v          # (chunks × n)
            # sigma는 대각으로 유지(여기서는 단순 concat된 sum_s)
            Sigma = torch.diag(sum_s)
            # 이후 단계에서 SigmaParametrization(U, V, sigma)로 사용

            new_vector[key] = [U_orth, Sigma, V_orth]
    return new_vector


def run_energy(cfg: DictConfig) -> None:
    """
    Run energy-based training on held-out test dataset.
    All configuration should be complete before calling this function.
    """
    logger = setup_simple_logger(__name__)
    
    with open_dict(cfg):
        # Auto-set sigma_epochs if not provided
        test_ds = cfg.test_dataset
        if test_ds and test_ds in SIGMA_EPOCHS_PER_DATASET and cfg.sigma_epochs is None:
            cfg.sigma_epochs = SIGMA_EPOCHS_PER_DATASET[test_ds]
            logger.info(f"✓ Auto-set sigma_epochs={cfg.sigma_epochs} for {test_ds}")
        
        if cfg.sigma_epochs is None:
            cfg.sigma_epochs = 10
        
        # Normalize adapter choice
        cfg.adapter = normalize_adapter_choice(cfg.adapter)
        
        # Set adapter lr/wd to sigma values if not specified
        if cfg.adapter_lr is None:
            cfg.adapter_lr = cfg.sigma_lr
        if cfg.adapter_wd is None:
            cfg.adapter_wd = cfg.sigma_wd
        
        cfg.adapter_display = cfg.adapter if cfg.adapter != "none" else "none"
        
        # Auto-generate config_tag if not provided
        if not cfg.config_tag:
            cfg.config_tag = build_energy_config_tag(cfg)
    
    # exclude held-out test dataset from basis building
    test_ds = cfg.test_dataset
    
    # Determine which datasets to use for basis construction
    logger.info("Using DATASETS_ALL for basis construction (leave-one-out mode)")
    if hasattr(cfg, 'DATASETS_ALL') and cfg.DATASETS_ALL:
        base_list = list(cfg.DATASETS_ALL)
    else:
        base_list = ALL_DATASETS[:cfg.num_tasks]

    
    # Remove test_dataset from basis list
    if test_ds in base_list:
        base_list = [d for d in base_list if d != test_ds]

    cfg.DATASETS = base_list
    cfg.num_tasks = len(base_list)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in base_list]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    cfg.config_tag = getattr(cfg, "config_tag", build_energy_config_tag(cfg))
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Using config tag: {cfg.config_tag}")
    adapter_display = cfg.adapter_display if hasattr(cfg, "adapter_display") else "none"
    adapter_tag = adapter_path_tag(adapter_display)
    logger.info(f"Adapter option: {adapter_display}")

    # TSV Merge Orthogonalization
    # Load fine-tuned checkpoints for basis tasks
    logger.info("Loading fine-tuned checkpoints for basis tasks:")
    logger.info(f"datasets: {cfg.DATASETS_VAL}")
    logger.info(f"model: {cfg.model}")
    
    ft_checks = []
    for dataset in cfg.DATASETS_VAL:
        path = get_finetuned_path(cfg.model_location, dataset, model=cfg.model)
        if os.path.exists(path):
            logger.info(f"✓ {path} exists")
            ft_checks.append(torch.load(path, map_location="cpu"))
        else:
            logger.error(f"✗ {path} does not exist")
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
    
    # Load or create pretrained (zeroshot) checkpoint
    # Use the first dataset to store zeroshot model (convention)
    first_dataset = cfg.DATASETS_VAL[0] if cfg.DATASETS_VAL else "dummy"
    zeroshot_path = get_zeroshot_path(cfg.model_location, first_dataset, model=cfg.model)
    
    logger.info(f"Loading zeroshot model from: {zeroshot_path}")
    ptm_check = torch.load(zeroshot_path, map_location="cpu")

    overall_start = time.time()

    # Create the task vectors
    task_vectors = [
        NonLinearTaskVector(cfg.model, ptm_check, check) for check in ft_checks
    ]

    # create final task vector with timing
    svd_start = time.time()
    init_mode = getattr(cfg, "initialize_sigma", "average").lower()
    if init_mode == "tsvm":
        logger.info("Using TSVM SVD initialization")
        svd_dict = compute_and_sum_svd_mem_reduction_tsvm(task_vectors, cfg)
    elif init_mode == "average":
        logger.info("Using average SVD initialization")
        svd_dict = compute_and_sum_svd_mem_reduction_average(task_vectors, cfg)
    elif init_mode == "sum":
        logger.info("Using sum SVD initialization")
        svd_dict = compute_and_sum_svd_mem_reduction_sum(task_vectors, cfg)
    svd_time = time.time() - svd_start
    logger.info(f"Computed SVD bases in {svd_time:.2f}s")

    # Export SVD bases and initial sigma diagonals for sigma-only finetuning
    basis = {}
    for key, value in svd_dict.items():
        # Expect list [U_orth, diag(sum_s), V_orth] for 2D weights
        if isinstance(value, list) and len(value) == 3:
            U_orth, diag_s, V_orth = value
            sigma_vec = torch.diagonal(diag_s).clone().detach().cpu()
            basis[key] = {
                "U": U_orth.clone().detach().cpu(),
                "V": V_orth.clone().detach().cpu(),
                "sigma": sigma_vec,
            }


    # If a held-out test dataset is provided, train sigma only on that dataset and evaluate
    sigma_trainable_params = None

    if test_ds:
        logger.info(f"Sigma-only finetuning on held-out dataset: {test_ds}")

        # Build sigma modules from exported basis in-memory
        sigma_modules = torch.nn.ModuleDict()
        # Map sanitized keys (valid for Module names) back to original state_dict keys
        sigma_key_map = {}
        for key, fv in basis.items():
            if all(k in fv for k in ("U", "V", "sigma")):
                U, V, sigma = fv["U"], fv["V"], fv["sigma"]
                if U.ndim == 2 and V.ndim == 2 and sigma.ndim == 1:
                    # ModuleDict keys cannot contain '.', so sanitize
                    safe_key = key.replace(".", "_")
                    # Avoid accidental collisions after sanitization
                    if safe_key in sigma_key_map:
                        suffix = 1
                        candidate = f"{safe_key}_{suffix}"
                        while candidate in sigma_key_map:
                            suffix += 1
                            candidate = f"{safe_key}_{suffix}"
                        safe_key = candidate
                    sigma_key_map[safe_key] = key
                    sigma_modules[safe_key] = SigmaParametrization(U, V, sigma)
        sigma_modules = sigma_modules.cuda()

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in sigma_modules.parameters() if p.requires_grad)
        logger.info(f"=" * 80)
        logger.info(f"Number of trainable sigma parameters: {trainable_params:,}")
        logger.info(f"Number of sigma modules: {len(sigma_modules)}")
        logger.info(f"=" * 80)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sigma_trainable_params = trainable_params
        
        # Use train split for sigma finetuning and Val split for evaluation
        val_dataset_name = test_ds + "Val"
        
        # Determine k-shot folder structure
        k = int(cfg.train_k)
        shot_folder = f"{k}shots" if k > 0 else "fullshots"
        
        # Ensure save_dir exists in cfg for classification head management
        with open_dict(cfg):
            if "save_dir" not in cfg:
                cfg.save_dir = os.path.join(
                    cfg.model_location, 
                    cfg.model
                )
        
        # Load pretrained encoder (keep original for later comparison)
        pretrained_encoder = ImageEncoder(cfg.model).cuda()
        image_encoder = ImageEncoder(cfg.model).cuda()
        
        # Load dataset with consistent train/val split
        logger.info(f"Loading dataset for training: {val_dataset_name}")
        train_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ] + image_encoder.train_preprocess.transforms[-3:])

        dataset_train = get_dataset(
            test_ds,
            train_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
        )
        
        classification_head = get_classification_head(cfg, test_ds)
        
        model = ImageClassifier(image_encoder, classification_head).cuda()
        model.freeze_head()

        train_loader = get_dataloader(
            dataset_train, is_train=True, args=cfg, image_encoder=None)

        # Optionally restrict to k training samples per class if cfg.train_k > 0
        if k is not None and k > 0:
            logger.info(f"Applying k-shot sampling: {k} samples per class")
            try:
                seed = int(getattr(cfg, 'seed', 1))
                
                # Create directory for saving indices
                indices_save_dir = os.path.join(cfg.model_location, cfg.model, val_dataset_name)
                
                # Try to load existing k-shot indices
                selected_indices = load_k_shot_indices(indices_save_dir, k, seed)
                
                if selected_indices is not None:
                    logger.info(f"✓ Loaded existing {k}-shot indices (seed={seed})")
                else:
                    # Try to subsample from larger k (e.g., 16-shot)
                    larger_k = 16
                    if k < larger_k:
                        larger_indices = load_k_shot_indices(indices_save_dir, larger_k, seed)
                        if larger_indices is not None:
                            logger.info(f"✓ Subsampling from existing {larger_k}-shot indices to {k}-shot")
                            selected_indices = subsample_from_larger_k(larger_indices, dataset_train, k, seed)
                            # Save for future use
                            indices_path = save_k_shot_indices(selected_indices, indices_save_dir, val_dataset_name, k, seed)
                            logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
                        else:
                            # Sample new indices from scratch
                            logger.info(f"Sampling new {k}-shot indices from full dataset (seed={seed})")
                            selected_indices = sample_k_shot_indices(
                                dataset_train,
                                k,
                                seed=seed,
                                verbose=True,
                                progress_desc=f"{test_ds} {k}-shot",
                            )
                            # Save the indices for future use
                            indices_path = save_k_shot_indices(selected_indices, indices_save_dir, val_dataset_name, k, seed)
                            logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
                    else:
                        # k >= larger_k, need to sample from scratch
                        logger.info(f"Sampling new {k}-shot indices from full dataset (seed={seed})")
                        selected_indices = sample_k_shot_indices(
                            dataset_train,
                            k,
                            seed=seed,
                            verbose=True,
                            progress_desc=f"{test_ds} {k}-shot",
                        )
                        # Save the indices for future use
                        indices_path = save_k_shot_indices(selected_indices, indices_save_dir, val_dataset_name, k, seed)
                        logger.info(f"✓ Saved {k}-shot indices to {indices_path}")
                
                # Get base dataset
                base_dataset = getattr(dataset_train, "train_dataset", None)
                if base_dataset is None:
                    base_dataset = getattr(train_loader, "dataset", None)
                
                if base_dataset is not None:
                    num_workers = getattr(train_loader, "num_workers", 8)
                    collate_fn = getattr(train_loader, "collate_fn", None)
                    train_loader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(base_dataset, selected_indices),
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn,
                    )
                    logger.info(
                        f"✓ Created {k}-shot dataloader with {len(selected_indices)} samples")
                else:
                    logger.warning(
                        "Could not locate base train_dataset for k-shot subsetting; using full loader instead.")
            except Exception as e:
                logger.error(f"Failed to apply k-shot sampling: {e}")
                logger.warning("Falling back to full training set")

        config_dir = os.path.join(
            cfg.model_location,
            cfg.model,
            val_dataset_name,
            cfg.config_tag,
        )
        os.makedirs(config_dir, exist_ok=True)
        energy_save_dir = os.path.join(config_dir, shot_folder)
        os.makedirs(energy_save_dir, exist_ok=True)

        params = [p for p in sigma_modules.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)
        
        # Use cosine annealing scheduler (same as Atlas)
        num_batches = len(train_loader)
        total_steps = int(cfg.sigma_epochs) * num_batches
        scheduler = cosine_lr(optimizer, cfg.sigma_lr, int(cfg.warmup_ratio * total_steps), total_steps)

        # capture cloned base parameters and buffers for functional_call
        base_params = {
            name: p.detach().clone()
            for name, p in model.image_encoder.named_parameters()
        }
        base_buffers = {
            name: b.detach().clone()
            for name, b in model.image_encoder.named_buffers()
        }
        base_state_dict = {
            name: tensor.detach().clone()
            for name, tensor in model.image_encoder.state_dict().items()
        }

        # Prepare evaluation schedule and history tracking
        eval_epochs = compute_eval_epochs(int(cfg.sigma_epochs))
        loss_history = []
        val_history = []
        eval_counter = 0

        def record_validation(stage: str, epoch_value, accuracy_value):
            """Record evaluation metadata for result serialization."""
            nonlocal eval_counter
            record = {
                "stage": stage,
                "epoch": int(epoch_value),
                "accuracy": float(accuracy_value),
                "elapsed_seconds": float(time.time() - overall_start),
                "evaluation_index": int(eval_counter),
            }
            val_history.append(record)
            eval_counter += 1
            return record

        # Log zeroshot accuracy before any sigma updates
        model.eval()
        with torch.no_grad():
            pretrained_metrics = eval_single_dataset(pretrained_encoder, val_dataset_name, cfg)
            pretrained_acc = pretrained_metrics['top1']
            logger.info(f"Pretrained encoder validation accuracy: {pretrained_acc * 100:.2f}%")
            record_validation("pretrained", -2, pretrained_acc)

            eval_params = {}
            for name, p in base_params.items():
                eval_params[name] = p.clone()
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in eval_params and module.sigma.numel() > 0:
                    delta = module().to(eval_params[orig_key].device)
                    if eval_params[orig_key].shape == delta.shape:
                        eval_params[orig_key] = eval_params[orig_key] + delta

            model.image_encoder.load_state_dict(eval_params, strict=False)
            zeroshot_metrics = eval_single_dataset(model.image_encoder, val_dataset_name, cfg)
            zeroshot_acc = zeroshot_metrics['top1']
            logger.info(f"Zeroshot encoder validation accuracy: {zeroshot_acc * 100:.2f}%")
            record_validation("zeroshot", -1, zeroshot_acc)
            model.image_encoder.load_state_dict(base_state_dict, strict=False)

        model.train()
        
        logger.info(f"Starting sigma fine-tuning for {cfg.sigma_epochs} epochs...")
        logger.info(f"Train dataset size: {len(train_loader.dataset)}, Batch size: {cfg.batch_size}, Steps per epoch: {len(train_loader)}")
        
        epoch_times = []  # Track training time per epoch (excluding validation)
        
        for epoch in range(int(cfg.sigma_epochs)):
            epoch_start = time.time()
            model.train()
            for i, batch in enumerate(train_loader):
                step = epoch * num_batches + i
                
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()

                # build delta map with autograd connectivity
                delta_map = {}
                for safe_key, module in sigma_modules.items():
                    orig_key = sigma_key_map.get(safe_key, safe_key)
                    if orig_key in base_params and module.sigma.numel() > 0:
                        delta = module()
                        if delta.shape == base_params[orig_key].shape:
                            delta_map[orig_key] = delta
                # combine base params with delta so that grads flow into sigma
                # detach base params to avoid tracking grads into frozen encoder
                params_map = {}
                for name, p in base_params.items():
                    if name in delta_map:
                        params_map[name] = p.detach() + delta_map[name]
                    else:
                        params_map[name] = p.detach()

                # forward using functional_call to preserve graph w.r.t sigma
                def encoder_forward(mod, x):
                    merged = {}
                    merged.update(base_buffers)
                    merged.update(params_map)
                    return functional_call(mod, merged, (x,))

                features = encoder_forward(model.image_encoder, inputs)
                logits = model.classification_head(features)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler(step)  # Update learning rate at each step

                loss_history.append(
                    {
                        "epoch": int(epoch),
                        "iteration": int(i),
                        "loss": float(loss.item()),
                    }
                )
                
                if (i == 0):
                    try:
                        grad_sum = 0.0
                        for p in params:
                            if p.grad is not None:
                                grad_sum += float(p.grad.detach().abs().sum().item())
                        logger.info(
                            f"[sigma] epoch {epoch} {i + 1}/{len(train_loader)} loss {loss.item():.4f} grad_sum {grad_sum:.4e} lr {optimizer.param_groups[0]['lr']:.6f}")
                    except Exception:
                        logger.info(
                            f"[sigma] epoch {epoch} {i + 1}/{len(train_loader)} loss {loss.item():.4f} lr {optimizer.param_groups[0]['lr']:.6f}")

            # Record training time for this epoch (before validation)
            epoch_train_time = time.time() - epoch_start
            epoch_times.append(epoch_train_time)

            if epoch in eval_epochs:
                model.eval()
                with torch.no_grad():
                    eval_params = {}
                    for name, p in base_params.items():
                        eval_params[name] = p.clone()
                    for safe_key, module in sigma_modules.items():
                        orig_key = sigma_key_map.get(safe_key, safe_key)
                        if orig_key in eval_params and module.sigma.numel() > 0:
                            delta = module().to(eval_params[orig_key].device)
                            if eval_params[orig_key].shape == delta.shape:
                                eval_params[orig_key] = eval_params[orig_key] + delta

                    model.image_encoder.load_state_dict(eval_params, strict=False)

                    val_metrics = eval_single_dataset(model.image_encoder, val_dataset_name, cfg)
                    val_acc = val_metrics['top1']

                    logger.info(f"[sigma] epoch {epoch} validation accuracy: {val_acc * 100:.2f}%")
                    record_validation("epoch", epoch, val_acc)

                    model.image_encoder.load_state_dict(base_state_dict, strict=False)

                model.train()

        # Finalize weights and save: materialize the final deltas onto base params
        with torch.no_grad():
            materialized = {}
            for name, p in base_params.items():
                materialized[name] = p.clone()
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in materialized and module.sigma.numel() > 0:
                    delta = module().to(materialized[orig_key].device)
                    if materialized[orig_key].shape == delta.shape:
                        materialized[orig_key] = materialized[orig_key] + delta
            # load back into encoder
            model.image_encoder.load_state_dict(materialized, strict=False)
        
        # Save with k-shot folder structure
        os.makedirs(energy_save_dir, exist_ok=True)
        energy_path = os.path.join(energy_save_dir, "energy.pt")
        model.image_encoder.save(energy_path)
        logger.info(f"Saved energy-trained encoder to {energy_path}")
        
        # Final evaluation on validation set
        logger.info("\n" + "=" * 100)
        logger.info("Final evaluation on validation set...")
        logger.info("=" * 100 + "\n")
        
        model.eval()
        with torch.no_grad():
            final_metrics = eval_single_dataset(model.image_encoder, val_dataset_name, cfg)
            final_acc = final_metrics['top1']
        
        logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")
        record_validation("final", int(cfg.sigma_epochs), final_acc)

        # Use minimum epoch training time (excluding validation)
        min_epoch_time = min(epoch_times) if epoch_times else 0.0
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        logger.info(f"Training time per epoch - Min: {min_epoch_time:.2f}s, Avg: {avg_epoch_time:.2f}s")

        gpu_peak_mem_mb = None
        if torch.cuda.is_available():
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

        # Optional adapter fine-tuning (TIP / LP++) after sigma training
        adapter_summary = None
        adapter_result_tag = adapter_path_tag(adapter_display)
        if cfg.adapter != "none":
            adapter_args = SimpleNamespace(
                adapter=cfg.adapter,
                batch_size=int(cfg.batch_size),
                num_grad_accumulation=int(getattr(cfg, "num_grad_accumulation", 1)),
                wd=float(getattr(cfg, "adapter_wd", cfg.sigma_wd)),
                lr=float(getattr(cfg, "adapter_lr", cfg.sigma_lr)),
                k=int(getattr(cfg, "train_k", 0)),
            )
            # Get validation loader for adapter training
            val_dataset = get_dataset(
                test_ds,
                model.image_encoder.val_preprocess,
                location=cfg.data_location,
                batch_size=cfg.batch_size,
            )
            val_loader = get_dataloader(
                val_dataset, is_train=False, args=cfg, image_encoder=None
            )
            
            adapter_ready_model = AdapterCompatibleClassifier(model)
            adapter_summary = train_adapter(
                adapter_ready_model, train_loader, val_loader, adapter_args, logger, energy_save_dir
            )
            if adapter_summary:
                adapter_result_tag = adapter_path_tag(adapter_summary["adapter_type"])

        # Save results to JSON
        results_path = os.path.join(energy_save_dir, f"energy_results_{adapter_result_tag}.json")
        results = {
            "target_dataset": test_ds,
            "final_accuracy": float(final_acc),
            "k_shot": k,
            "model": cfg.model,
            "sigma_epochs": cfg.sigma_epochs,
            "sigma_lr": cfg.sigma_lr,
            "svd_keep_topk": getattr(cfg, "svd_keep_topk", 2),
            "initialize_sigma": getattr(cfg, "initialize_sigma", None),
            "adapter_choice": cfg.adapter,
            "training_time": min_epoch_time,
            "avg_epoch_time": avg_epoch_time,
            "all_epoch_times": epoch_times,
            "trainable_params": sigma_trainable_params,
            "batch_size": cfg.batch_size,
            "gpu_peak_mem_mb": gpu_peak_mem_mb,
            "loss_history": loss_history,
            "validation_history": val_history,
            "evaluation_schedule": [int(ep) for ep in sorted(eval_epochs)],
            "pretrained_accuracy": float(pretrained_acc),
            "zeroshot_accuracy": float(zeroshot_acc),
            "config_tag": cfg.config_tag,
            "adapter_results": adapter_summary,
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"✓ Saved results to {results_path}")


if __name__ == "__main__":
    from src.datasets.registry import registry as DATASET_REGISTRY
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Energy-based task vector merging for general datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    allowed_test_datasets = sorted(
        [name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")]
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        # choices=allowed_test_datasets,
        help="Held-out dataset to train on (sigma epochs auto-set by dataset size)",
    )
    
    # Config and model
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_reverse.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Vision backbone (e.g., ViT-B-32, ViT-B-16)"
    )
    
    # Training hyperparameters
    parser.add_argument("--sigma_epochs", type=int, help="Number of sigma training epochs")
    parser.add_argument("--sigma_lr", type=float, help="Learning rate for sigma optimization")
    parser.add_argument("--sigma_wd", type=float, help="Weight decay for sigma optimization")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--k",
        type=int,
        dest="train_k",
        help="K-shot samples per class (0=fullshot)"
    )
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for sigma learning rate")
    
    # SVD and initialization
    parser.add_argument(
        "--svd_keep_topk",
        type=int,
        help="Number of singular vectors to keep per task"
    )
    parser.add_argument(
        "--initialize_sigma",
        type=str,
        choices=["average", "sum", "tsvm"],
        help="Initialization strategy for sigma basis"
    )

    # Adapter options
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["none", "tip", "lp++"],
        help="Optional adapter after sigma training"
    )
    parser.add_argument("--adapter_lr", type=float, help="Adapter learning rate")
    parser.add_argument("--adapter_wd", type=float, help="Adapter weight decay")
    parser.add_argument(
        "--adapter_grad_accum",
        type=int,
        dest="num_grad_accumulation",
        help="Gradient accumulation steps for adapter"
    )
    
    # Other
    parser.add_argument("--config_tag", type=str, help="Custom tag for output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for k-shot sampling")
    
    args = parser.parse_args()
    
    # Load config file
    cfg = load_config(args.config_file)
    
    # Merge CLI arguments (only non-None values override config)
    cli_overrides = {k: v for k, v in vars(args).items() 
                     if v is not None and k != "config_file"}
    
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    
    # Validate required fields
    if not cfg.get("test_dataset"):
        parser.error("--test_dataset is required")
    
    OmegaConf.set_struct(cfg, True)
    run_energy(cfg)

