import os
import time
import json
import logging
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import sys
import torchvision

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import ALL_DATASETS
from src.datasets import get_dataloader, maybe_dictionarize
from src.models import ImageClassifier, ImageEncoder
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path, get_energy_finetuned_path
from src.utils.sigma_param import SigmaParametrization
import torch
from src.models.task_vectors import ImageEncoder, NonLinearTaskVector
from torch.nn.utils.stateless import functional_call
from atlas_remote_sensing import train_adapter_remote


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
    k_part = _sanitize_value(getattr(cfg, "train_k", 0))
    lr_part = _sanitize_value(cfg.sigma_lr)
    svd_part = _sanitize_value(getattr(cfg, "svd_keep_topk", 2))
    init_mode_part = _sanitize_value(getattr(cfg, "initialize_sigma", "average"))
    return f"energy_{num_tasks_minus_one}_{k_part}_{lr_part}_{svd_part}_{init_mode_part}"


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

# Import remote sensing specific modules
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
    clean_dataset_logs,
    sample_k_shot_indices,
)
from src.eval.eval_remote_sensing_comparison import (
    evaluate_encoder_with_dataloader,
    visualize_sigma_matrices,
)


# Dataset-specific epochs for sigma training
# These match the fine-tuning epochs from finetune_remote_sensing_datasets.py
SIGMA_EPOCHS_PER_DATASET = {
    "AID": 10,              # ~10,000 train samples, 600x600
    "CLRS": 10,             # ~30,000 train samples, 256x256
    "EuroSAT_RGB": 12,      # ~21,600 train samples, 64x64
    "MLRSNet": 15,          # ~17,000 train samples, 256x256
    "NWPU-RESISC45": 15,    # ~25,200 train samples, 256x256
    "Optimal-31": 50,       # ~6,200 train samples, 256x256
    "PatternNet": 20,       # ~10,000 train samples, 256x256
    "RS_C11": 60,           # ~5,000 train samples, 512x512
    "RSD46-WHU": 20,        # ~10,000 train samples, 256x256
    "RSI-CB128": 15,        # ~18,000 train samples, 128x128
    "RSSCN7": 80,           # ~2,800 train samples, 400x400
    "SAT-4": 5,             # ~60,000 train samples, 28x28
    "SIRI-WHU": 100,        # ~2,400 train samples, 200x200
    "UC_Merced": 100,       # ~2,100 train samples, 256x256
    "WHU-RS19": 150,        # ~1,000 train samples, 600x600
}
# SIGMA_EPOCHS_PER_DATASET = {
#     "AID": 5,              # ~10,000 train samples, 600x600
#     "CLRS": 5,             # ~30,000 train samples, 256x256
#     "EuroSAT_RGB": 5,      # ~21,600 train samples, 64x64
#     "MLRSNet": 5,          # ~17,000 train samples, 256x256
#     "NWPU-RESISC45": 5,    # ~25,200 train samples, 256x256
#     "Optimal-31": 5,       # ~6,200 train samples, 256x256
#     "PatternNet": 5,       # ~10,000 train samples, 256x256
#     "RS_C11": 5,           # ~5,000 train samples, 512x512
#     "RSD46-WHU": 5,        # ~10,000 train samples, 256x256
#     "RSI-CB128": 5,        # ~18,000 train samples, 128x128
#     "RSSCN7": 5,           # ~2,800 train samples, 400x400
#     "SAT-4": 5,             # ~60,000 train samples, 28x28
#     "SIRI-WHU": 5,        # ~2,400 train samples, 200x200
#     "UC_Merced": 5,       # ~2,100 train samples, 256x256
#     "WHU-RS19": 5,        # ~1,000 train samples, 600x600
# }


def compute_eval_epochs(total_epochs: int, max_evals: int = 5) -> set:
    total_epochs = max(int(total_epochs), 1)
    if total_epochs <= max_evals:
        return set(range(total_epochs))
    eval_epochs = {
        min(total_epochs - 1, int(round(i * (total_epochs - 1) / (max_evals - 1))))
        for i in range(max_evals)
    }
    return eval_epochs


def compute_and_sum_svd_mem_reduction_average(task_vectors, config):
    """
    여러 태스크 벡터의 2D 가중치에 대해 SVD를 수행,
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과
    '재계산된 평균' sigma(diag)로 재구성.
    """
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    print(f"DATSETS: {datasets}")
    print("Computing SVD...")
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
                Sigma = torch.diag(mean_sigma_diag)  # (chunks, chunks)
            # -------- [수정된 Sigma 계산 로직 끝] --------
            # 이후 단계에서 SigmaParametrization(U, V, sigma)로 사용
            new_vector[key] = [U_orth, Sigma, V_orth]
    return new_vector

def compute_and_sum_svd_mem_reduction_tsvm(task_vectors, config):
    """
    여러 태스크 벡터의 2D 가중치에 대해 SVD를 수행,
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과 sigma(diag)로 재구성.
    """
    logger = logging.getLogger(__name__)
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    print(f"DATSETS: {datasets}")
    print("Computing SVD...")
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

    # Use Hydra's logger (already configured)
    logger = logging.getLogger(__name__)
    
    # add defaults for test-time sigma finetuning
    with open_dict(cfg):
        if "test_dataset" not in cfg:
            cfg.test_dataset = ""
        
        # override test_dataset from environment variable first
        if os.environ.get("ENERGY_TEST_DATASET", ""):
            cfg.test_dataset = os.environ["ENERGY_TEST_DATASET"]
        
        # Set sigma_epochs based on test_dataset (matches fine-tuning epochs)
        if cfg.test_dataset and cfg.test_dataset in SIGMA_EPOCHS_PER_DATASET:
            cfg.sigma_epochs = SIGMA_EPOCHS_PER_DATASET[cfg.test_dataset]
            logger.info(f"✓ Auto-set sigma_epochs={cfg.sigma_epochs} for {cfg.test_dataset}")
        elif "sigma_epochs" not in cfg:
            cfg.sigma_epochs = 10  # default fallback
        
        if "sigma_lr" not in cfg:
            cfg.sigma_lr = 5e-6
        if "sigma_wd" not in cfg:
            cfg.sigma_wd = 0.0
        # scheduler defaults
        if "sigma_lr_step_size" not in cfg:
            cfg.sigma_lr_step_size = 1
        if "sigma_lr_gamma" not in cfg:
            cfg.sigma_lr_gamma = 1.0
        if "batch_size" not in cfg:
            cfg.batch_size = 128
        if "device" not in cfg:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        # number of training samples per class to use (0 => use all)
        if "train_k" not in cfg:
            cfg.train_k = 0
        if "num_grad_accumulation" not in cfg:
            cfg.num_grad_accumulation = 1
        if "adapter" not in cfg:
            cfg.adapter = "none"
        if "adapter_lr" not in cfg:
            cfg.adapter_lr = cfg.sigma_lr
        if "adapter_wd" not in cfg:
            cfg.adapter_wd = cfg.sigma_wd

        # override from environment variables (manual override if needed)
        if os.environ.get("ENERGY_SIGMA_EPOCHS", ""):
            try:
                cfg.sigma_epochs = int(os.environ["ENERGY_SIGMA_EPOCHS"])
                logger.info(f"Manual override: sigma_epochs={cfg.sigma_epochs}")
            except Exception:
                pass
        if os.environ.get("ENERGY_SIGMA_LR", ""):
            try:
                cfg.sigma_lr = float(os.environ["ENERGY_SIGMA_LR"])
            except Exception:
                pass
        if os.environ.get("ENERGY_SIGMA_WD", ""):
            try:
                cfg.sigma_wd = float(os.environ["ENERGY_SIGMA_WD"])
            except Exception:
                pass
        if os.environ.get("ENERGY_SIGMA_LR_STEP_SIZE", ""):
            try:
                cfg.sigma_lr_step_size = int(
                    os.environ["ENERGY_SIGMA_LR_STEP_SIZE"])
            except Exception:
                pass
        if os.environ.get("ENERGY_SIGMA_LR_GAMMA", ""):
            try:
                cfg.sigma_lr_gamma = float(os.environ["ENERGY_SIGMA_LR_GAMMA"])
            except Exception:
                pass
        if os.environ.get("ENERGY_BATCH_SIZE", ""):
            try:
                cfg.batch_size = int(os.environ["ENERGY_BATCH_SIZE"])
            except Exception:
                pass
        if os.environ.get("ENERGY_TRAIN_K", ""):
            try:
                cfg.train_k = int(os.environ["ENERGY_TRAIN_K"])
            except Exception:
                pass
        if os.environ.get("ENERGY_ADAPTER", ""):
            cfg.adapter = os.environ["ENERGY_ADAPTER"]
        cfg.adapter = normalize_adapter_choice(cfg.adapter)
        cfg.adapter_display = cfg.adapter if cfg.adapter != "none" else "none"
        cfg.adapter_lr = float(getattr(cfg, "adapter_lr", cfg.sigma_lr))
        cfg.adapter_wd = float(getattr(cfg, "adapter_wd", cfg.sigma_wd))

        env_config_tag = os.environ.get("ENERGY_CONFIG_TAG", "").strip()
        if env_config_tag:
            cfg.config_tag = env_config_tag
        elif "config_tag" not in cfg or not cfg.config_tag:
            cfg.config_tag = build_energy_config_tag(cfg)
    
    # exclude held-out test dataset from basis building
    test_ds = cfg.test_dataset
    
    # Determine which datasets to use for basis construction
    # if cfg.DATASETS == "" or not cfg.DATASETS:
        # If DATASETS is empty or not specified, use DATASETS_ALL
    logger.info("Using DATASETS_ALL for basis construction (leave-one-out mode)")
    base_list = list(cfg.DATASETS_ALL)

    
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
    else:
        logger.info("Using average SVD initialization")
        svd_dict = compute_and_sum_svd_mem_reduction_average(task_vectors, cfg)
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
        
        # Load remote sensing dataset with consistent train/val split (same as Atlas)
        logger.info(f"Loading remote sensing dataset for training: {val_dataset_name}")
        train_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ] + image_encoder.train_preprocess.transforms[-3:])

        dataset_train = get_remote_sensing_dataset(
            val_dataset_name,
            train_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=6,
        )
        
        classification_head = get_remote_sensing_classification_head(cfg, val_dataset_name, dataset_train)
        
        model = ImageClassifier(image_encoder, classification_head).cuda()
        model.freeze_head()

        train_loader = get_dataloader(
            dataset_train, is_train=True, args=cfg, image_encoder=None)

        # Optionally restrict to k training samples per class if cfg.train_k > 0
        if k is not None and k > 0:
            logger.info(f"Applying k-shot sampling: {k} samples per class")
            try:
                selected_indices = sample_k_shot_indices(dataset_train, k, seed=0, verbose=True)
                
                # Get base dataset
                base_dataset = getattr(dataset_train, "train_dataset", None)
                if base_dataset is None:
                    base_dataset = getattr(train_loader, "dataset", None)
                
                if base_dataset is not None:
                    num_workers = getattr(train_loader, "num_workers", 6)
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
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.sigma_lr_step_size), gamma=float(cfg.sigma_lr_gamma)
        )

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

        # Load validation dataset for per-epoch evaluation
        logger.info(f"Loading validation dataset: {val_dataset_name}")
        val_dataset = get_remote_sensing_dataset(
            val_dataset_name,
            image_encoder.val_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=6,
        )
        val_loader = get_dataloader(
            val_dataset, is_train=False, args=cfg, image_encoder=None)
        
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

        # Prepare visualization directory
        visualization_dir = os.path.join(
            energy_save_dir, f"sigma_visualization_{adapter_tag}"
        )
        os.makedirs(visualization_dir, exist_ok=True)

        # Log zeroshot accuracy before any sigma updates
        model.eval()
        with torch.no_grad():
            pretrained_metrics = evaluate_encoder_with_dataloader(model.image_encoder, classification_head, val_loader, cfg.device)
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
            zeroshot_metrics = evaluate_encoder_with_dataloader(model.image_encoder, classification_head, val_loader, cfg.device)
            zeroshot_acc = zeroshot_metrics['top1']
            logger.info(f"Zeroshot encoder validation accuracy: {zeroshot_acc * 100:.2f}%")
            record_validation("zeroshot", -1, zeroshot_acc)
            model.image_encoder.load_state_dict(base_state_dict, strict=False)

        sigma_records = []
        records = visualize_sigma_matrices(
            sigma_modules,
            sigma_key_map,
            epoch=-1,
            save_path=os.path.join(visualization_dir, "sigma_epoch_-1.png"),
            title=f"{test_ds} ({shot_folder})",
        )
        if records:
            sigma_records.extend(records)
        model.train()
        
        logger.info(f"Starting sigma fine-tuning for {cfg.sigma_epochs} epochs...")
        logger.info(f"Train dataset size: {len(train_loader.dataset)}, Batch size: {cfg.batch_size}, Steps per epoch: {len(train_loader)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        for epoch in range(int(cfg.sigma_epochs)):
            model.train()
            for i, batch in enumerate(train_loader):
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

            # step scheduler at end of epoch
            scheduler.step()

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

                    val_metrics = evaluate_encoder_with_dataloader(
                        model.image_encoder, classification_head, val_loader, cfg.device)
                    val_acc = val_metrics['top1']

                    logger.info(f"[sigma] epoch {epoch} validation accuracy: {val_acc * 100:.2f}%")
                    record_validation("epoch", epoch, val_acc)

                    model.image_encoder.load_state_dict(base_state_dict, strict=False)

                records = visualize_sigma_matrices(
                    sigma_modules,
                    sigma_key_map,
                    epoch=epoch,
                    save_path=os.path.join(visualization_dir, f"sigma_epoch_{epoch:03d}.png"),
                    title=f"{test_ds} ({shot_folder})",
                )
                if records:
                    sigma_records.extend(records)
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
        # e.g., models/checkpoints_remote_sensing/ViT-B-32/MLRSNetVal/16shot/energy.pt

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
            final_metrics = evaluate_encoder_with_dataloader(
                model.image_encoder, classification_head, val_loader, cfg.device)
            final_acc = final_metrics['top1']
        
        logger.info(f"Final validation accuracy: {final_acc * 100:.2f}%")
        record_validation("final", int(cfg.sigma_epochs), final_acc)

        training_time = time.time() - overall_start
        gpu_peak_mem_mb = None
        if torch.cuda.is_available():
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")

        records = visualize_sigma_matrices(
            sigma_modules,
            sigma_key_map,
            epoch="final",
            save_path=os.path.join(visualization_dir, "sigma_epoch_final.png"),
            title=f"{test_ds} ({shot_folder})",
        )
        if records:
            sigma_records.extend(records)

        # sigma 변화 선 그래프 저장 (Early/Middle/Late 한 장)
        if sigma_records:
            from collections import defaultdict
            import numpy as np
            import matplotlib.pyplot as plt

            def epoch_order(val):
                if isinstance(val, int):
                    return val
                if isinstance(val, str):
                    if val == "final":
                        return 10**6
                    try:
                        return int(val)
                    except Exception:
                        return -10**6
                return 0

            series_by_label = defaultdict(list)
            for rec in sigma_records:
                epoch_val = rec.get("epoch")
                epoch_label = "final" if epoch_val == "final" else f"epoch {epoch_val}"
                series_by_label[rec["label"]].append(
                    (epoch_order(epoch_val), epoch_label, rec["sigma_values"])
                )

            ordered_labels = [lbl for lbl in ["Early", "Middle", "Late"] if lbl in series_by_label]
            if not ordered_labels:
                ordered_labels = sorted(series_by_label.keys())
            if ordered_labels:
                fig, axes = plt.subplots(1, len(ordered_labels), figsize=(6 * len(ordered_labels), 4))
                if len(ordered_labels) == 1:
                    axes = [axes]
                for ax, lbl in zip(axes, ordered_labels):
                    curves = sorted(series_by_label[lbl], key=lambda x: x[0])
                    if not curves:
                        continue
                    x = np.arange(len(curves[0][2]))
                    for _, epoch_label, sigma_vals in curves:
                        ax.plot(x, sigma_vals, label=epoch_label)
                    ax.set_xlabel("Diagonal Index", fontsize=12)
                    ax.set_ylabel("Sigma Value", fontsize=12)
                    ax.set_title(f"{lbl} Layer", fontsize=14, fontweight="bold")
                    ax.legend(fontsize=10)
                plt.tight_layout()
                overview_path = os.path.join(visualization_dir, "sigma_line_overview.png")
                plt.savefig(overview_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()

        # Optional adapter fine-tuning (TIP / LP++) after sigma training
        adapter_summary = None
        adapter_result_tag = adapter_tag
        if cfg.adapter != "none":
            adapter_args = SimpleNamespace(
                adapter=cfg.adapter,
                batch_size=int(cfg.batch_size),
                num_grad_accumulation=int(getattr(cfg, "num_grad_accumulation", 1)),
                wd=float(getattr(cfg, "adapter_wd", cfg.sigma_wd)),
                lr=float(getattr(cfg, "adapter_lr", cfg.sigma_lr)),
                k=int(getattr(cfg, "train_k", 0)),
            )
            adapter_ready_model = AdapterCompatibleClassifier(model)
            adapter_summary = train_adapter_remote(
                adapter_ready_model, train_loader, val_loader, adapter_args, logger, energy_save_dir
            )
            if adapter_summary:
                adapter_result_tag = adapter_path_tag(adapter_summary["adapter_type"])

        # Save results to JSON
        results_path = os.path.join(
            energy_save_dir, f"energy_results_{adapter_result_tag}.json"
        )
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
            "training_time": training_time,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    parser = argparse.ArgumentParser(add_help=False)
    from src.datasets.remote_sensing import REMOTE_SENSING_DATASETS
    allowed_test_datasets = sorted(list(REMOTE_SENSING_DATASETS.keys()))
    
    parser.add_argument(
        "--test_dataset",
        type=str,
        choices=allowed_test_datasets,
        default='EuroSAT_RGB',
        help="Held-out dataset to sigma-finetune on; one of %(choices)s. "
             "Sigma epochs will be automatically set based on dataset size.",
    )
    parser.add_argument("--config_file", type=str, default="config/config_remote_sensing.yaml")
    parser.add_argument("--sigma_epochs", type=int, default=None,
                        help="Manual override for sigma epochs (optional, auto-determined by default)")
    parser.add_argument("--sigma_lr", type=float, default=1e-3)
    parser.add_argument("--sigma_wd", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sigma-lr-step-size", type=int, default=1)
    parser.add_argument("--sigma-lr-gamma", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=0,
                        help="Number of samples per class for few-shot learning (0 or None = use all)")
    parser.add_argument("--config_tag", type=str, default=None,
                        help="Optional tag to group outputs for this configuration")
    parser.add_argument("--initialize_sigma", type=str, default=None,
                        choices=["average", "tsvm"],
                        help="Initialization strategy for sigma basis")
    parser.add_argument("--adapter", type=str, default=None,
                        choices=["none", "tip", "lp++"],
                        help="Optional adapter to train after sigma optimization (TIP or LP++)")
    parser.add_argument("--adapter_lr", type=float, default=None,
                        help="Learning rate for adapter training (defaults to sigma_lr)")
    parser.add_argument("--adapter_wd", type=float, default=None,
                        help="Weight decay for adapter training (defaults to sigma_wd)")
    parser.add_argument("--adapter_grad_accum", type=int, default=None,
                        help="Gradient accumulation steps for adapter training (default: 1)")
    args = parser.parse_args()


    if not args.test_dataset:
        parser.error("--test_dataset must be specified")

    cfg = load_config(args.config_file)

    if args.test_dataset:
        cfg.test_dataset = args.test_dataset
    if args.sigma_epochs is not None:
        cfg.sigma_epochs = args.sigma_epochs
    if args.sigma_lr is not None:
        cfg.sigma_lr = args.sigma_lr
    if args.sigma_wd is not None:
        cfg.sigma_wd = args.sigma_wd
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "sigma_lr_step_size", None) is not None:
        cfg.sigma_lr_step_size = args.sigma_lr_step_size
    if getattr(args, "sigma_lr_gamma", None) is not None:
        cfg.sigma_lr_gamma = args.sigma_lr_gamma
    if getattr(args, "k", None) is not None:
        cfg.train_k = args.k
    if getattr(args, "initialize_sigma", None) is not None:
        cfg.initialize_sigma = args.initialize_sigma
    if getattr(args, "config_tag", None):
        cfg.config_tag = args.config_tag
    if getattr(args, "adapter", None) is not None:
        cfg.adapter = args.adapter
    if getattr(args, "adapter_lr", None) is not None:
        cfg.adapter_lr = args.adapter_lr
    if getattr(args, "adapter_wd", None) is not None:
        cfg.adapter_wd = args.adapter_wd
    if getattr(args, "adapter_grad_accum", None) is not None:
        cfg.num_grad_accumulation = args.adapter_grad_accum

    OmegaConf.set_struct(cfg, True)
    run_energy(cfg)
