import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import sys

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import ALL_DATASETS
from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.datasets.common import SubsetSampler
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.variables_and_paths import get_finetuned_path
from src.utils.sigma_param import SigmaParametrization
import torch
from src.models.task_vectors import NonLinearTaskVector
from src.eval.aggregation import get_all_checkpoints
from torch.nn.utils.stateless import functional_call
from atlas_src.utils import get_n_shots, IndexWrapper, TIPWrapper, LPPWrapper, _RepeatSampler
import time

import math
import torch
import torchvision
import random
import numpy as np


class FunctionalEncoderWrapper(torch.nn.Module):
    """
    평가 시점에만 베이스 파라미터와 현재 시그마 델타를 합성해 forward 하는 래퍼.
    베이스 모듈의 파라미터는 수정하지 않으며, functional_call로만 합성합니다.
    """

    def __init__(self, base_encoder: torch.nn.Module, buffers: dict, params: dict):
        super().__init__()
        self.base_encoder = base_encoder
        self._buffers_map = buffers
        self._params_map = params
        # preserve preprocessors for downstream dataloader construction
        self.train_preprocess = getattr(base_encoder, "train_preprocess", None)
        self.val_preprocess = getattr(base_encoder, "val_preprocess", None)

    def forward(self, x):
        merged = {}
        merged.update(self._buffers_map)
        merged.update(self._params_map)
        return functional_call(self.base_encoder, merged, (x,))


def compute_and_sum_svd_mem_reduction(task_vectors, config):
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
            sum_s = torch.zeros((chunks,), device=device,
                                dtype=s0.dtype)  # [수정] 더 이상 sum_s 필요 없음
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
                sum_s[start:end] = s[:k_i]  # [수정] 더 이상 sum_s 필요 없음
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
            # (num_tasks, chunks)
            stacked_sigmas = torch.stack(all_sigma_diags, dim=0)
            # (chunks,)
            mean_sigma_diag = torch.mean(stacked_sigmas, dim=0)

            # 태스크별 에너지 분산
            # task_energy = torch.linalg.norm(
            #     stacked_sigmas, dim=1)  # (num_tasks,)
            # cv = (task_energy.std() / (task_energy.mean()+1e-12)).item()
            # cv가 0.5~1.0↑면 스케일 이질성 큼 → 단순 평균 부적합
            # print(f"CV: {cv}")

            # 최종 Sigma: 평균낸 대각 성분으로 대각 행렬 생성
            Sigma = torch.diag(mean_sigma_diag)  # (chunks, chunks)
            # Sigma = torch.diag(sum_s)
            # 이후 단계에서 SigmaParametrization(U, V, sigma)로 사용
            new_vector[key] = [U_orth, Sigma, V_orth]

    return new_vector


def train_adapter(ddp_model, ddp_loader, cfg, train_dataset_name, val_dataset_name, logger):
    try:
        # Build caches (logits/features) over the (possibly k-shot) training loader
        ddp_model = ddp_model.to(cfg.device)
        all_features, all_labels, all_indexes, all_logits = [], [], [], []
        ddp_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(ddp_loader):
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(cfg.device)
                logits, features = ddp_model(inputs, return_features=True)
                labels = batch["labels"]
                all_features.append(features.detach().cpu())
                all_labels.append(labels)
                # batch may contain 'index' when using IndexWrapper
                if "index" in batch:
                    all_indexes.append(batch["index"])
                else:
                    # fallback: synthetic incremental index
                    if len(all_indexes) == 0:
                        all_indexes.append(torch.arange(len(inputs)))
                    else:
                        start = int(torch.cat(all_indexes).numel())
                        all_indexes.append(torch.arange(
                            start, start + len(inputs)))
                all_logits.append(logits.detach().cpu())

        logits_cache = torch.cat(all_logits)
        features_cache = torch.cat(all_features)
        labels = torch.cat(all_labels)
        indexes = torch.cat(all_indexes)
        indexes_to_i = {indexes[i].item(): i for i in range(len(indexes))}

        # Wrap model with adapter
        adapter_model = ddp_model
        if cfg.adapter == 'lpp':
            shots = int(cfg.k_shot) if int(cfg.k_shot) > 0 else 0
            adapter_model = LPPWrapper(
                adapter_model, features_cache, labels, shots)
            epochs = 300
            lr = adapter_model.lr_temp
        elif cfg.adapter == 'tip':
            adapter_model = TIPWrapper(adapter_model, features_cache, labels)
            lr = 1e-3
            epochs = 10
        else:
            raise NotImplementedError(f"Adapter {cfg.adapter} unknown")

        adapter_model = adapter_model.to(cfg.device)
        ddp_model = adapter_model

        params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.sigma_wd)
        num_batches = len(ddp_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * num_batches)
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        print_every = 100
        # repeat sampler like atlas to simulate multiple epochs
        try:
            ddp_loader._DataLoader__initialized = False
            ddp_loader.batch_sampler = _RepeatSampler(
                ddp_loader.batch_sampler, epochs)
            ddp_loader._DataLoader__initialized = True
        except Exception:
            pass

        ddp_model.train()
        for i, batch in enumerate(ddp_loader):
            epoch = i // max(1, num_batches)
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(cfg.device)

            # map indexes to cache positions
            if "index" in batch:
                ids = [indexes_to_i[j.item()] for j in batch['index']]
            else:
                # fallback if index missing
                start_id = (i % len(indexes))
                ids = list(
                    range(start_id, min(start_id + len(inputs), len(indexes))))

            l_cache, f_cache = logits_cache[ids].to(
                inputs), features_cache[ids].to(inputs)

            logits = ddp_model(inputs, l_cache, f_cache)
            labels_b = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if (i + 1) % print_every == 0:
                logger.info(
                    f"[adapter:{cfg.adapter}] epoch {epoch} {i + 1}/{len(ddp_loader)} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}")

        try:
            scheduler.step()
        except Exception:
            pass

        # Evaluate adapter-wrapped model on validation split
        metrics = eval_single_dataset(
            ddp_model.model.image_encoder, val_dataset_name, cfg, model=ddp_model)
        logger.info(
            f"Adapter '{cfg.adapter}' Acc on {val_dataset_name}: {metrics['top1']*100:.2f}%")

        # Save adapter weights (only trainable params)
        adapter_coefs = {k: v for k, v in ddp_model.state_dict(
        ).items() if hasattr(v, 'requires_grad') and v.requires_grad}
        save_dir = cfg.save_dir if hasattr(
            cfg, 'save_dir') else os.path.join(cfg.model_location, cfg.model)
        os.makedirs(save_dir, exist_ok=True)
        adapter_path = os.path.join(
            save_dir, f"adapter_{train_dataset_name}_{cfg.adapter}.pt")
        torch.save(adapter_coefs, adapter_path)
        logger.info(f"Saved adapter weights to {adapter_path}")

    except Exception as e:
        logger.exception(f"Adapter training failed: {e}")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    # add defaults for test-time sigma finetuning
    with open_dict(cfg):
        if "test_dataset" not in cfg:
            cfg.test_dataset = ""
        if "sigma_epochs" not in cfg:
            cfg.sigma_epochs = 5
        if "sigma_wd" not in cfg:
            cfg.sigma_wd = 0.0
        # scheduler defaults
        if "sigma_lr_step_size" not in cfg:
            cfg.sigma_lr_step_size = 1
        if "sigma_lr_gamma" not in cfg:
            cfg.sigma_lr_gamma = 0.5

        cfg.sigma_lr = float(os.environ["ENERGY_SIGMA_LR"])
        cfg.sigma_wd = float(os.environ["ENERGY_SIGMA_WD"])
        cfg.sigma_lr_step_size = int(os.environ["ENERGY_SIGMA_LR_STEP_SIZE"])
        cfg.sigma_lr_gamma = float(os.environ["ENERGY_SIGMA_LR_GAMMA"])
        cfg.batch_size = int(os.environ["ENERGY_BATCH_SIZE"])
        # atlas 기본과 동일하게 디바이스 자동 선택(고정 GPU 인덱스 제거)
        cfg.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        # few-shot 옵션 (0이면 전체 사용)
        cfg.k_shot = int(os.environ.get("ENERGY_K_SHOT", "0"))
        cfg.sigma_epochs = int(os.environ.get("ENERGY_SIGMA_EPOCHS", "5"))
        cfg.svd_keep_topk = int(os.environ.get("ENERGY_SVD_KEEP_TOPK", "3"))
        cfg.seed = int(os.environ.get("ENERGY_SEED", "1"))
        # adapter (optional)
        cfg.adapter = os.environ.get("ENERGY_ADAPTER", "")

        # override from environment variables (set by argparse in __main__)
        if os.environ.get("ENERGY_TEST_DATASET", ""):
            cfg.test_dataset = os.environ["ENERGY_TEST_DATASET"]

    # atlas 기본과 경로 정렬: model_location/save_dir/data_location 기본값 통일
    with open_dict(cfg):
        if not hasattr(cfg, "model_location") or cfg.model_location in (None, ""):
            # atlas의 기본 동작: args.save가 있으면 우선, 없으면 고정 경로로 폴백
            base_root = getattr(cfg, "save", None)
            if base_root is None:
                base_root = os.path.expanduser(
                    "/disk3/junghwan/task_vector/models/checkpoints")
            cfg.model_location = base_root
        if not hasattr(cfg, "save_dir") or cfg.save_dir in (None, ""):
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        if not hasattr(cfg, "data_location") or cfg.data_location in (None, ""):
            cfg.data_location = os.path.expanduser("datasets")

    # exclude held-out test dataset from basis building
    test_ds = cfg.test_dataset
    if cfg.DATASETS == "":
        base_list = ALL_DATASETS[: cfg.num_tasks]
    else:
        base_list = list(cfg.DATASETS)
    if test_ds in base_list:
        base_list = [d for d in base_list if d != test_ds]
    cfg.DATASETS = base_list
    cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    # set up experiment logging
    logger = logging.getLogger("energy_train")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler = logging.FileHandler("energy_train.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    # Prevent duplicate logs from propagating to the root/Hydra loggers
    logger.propagate = False

    logger.info(cfg.method.full_name)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)

    # Set deterministic seed to match atlas for k-shot reproducibility
    try:
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # TSV Merge Orthogonalization
    ft_checks, ptm_check = get_all_checkpoints(cfg)
    # Create the task vectors
    task_vectors = [
        NonLinearTaskVector(cfg.model, ptm_check, check) for check in ft_checks
    ]

    # create final task vector
    svd_dict = compute_and_sum_svd_mem_reduction(task_vectors, cfg)

    # Export SVD bases and initial sigma diagonals for sigma-only finetuning
    try:
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
        if len(basis) > 0:
            basis_path = f"svd_basis{'_' + test_ds if test_ds else ''}.pth"
            torch.save(basis, basis_path)
            logger.info(
                f"Saved SVD basis for sigma finetuning with {len(basis)} entries -> {basis_path}")
        else:
            logger.warning("No SVD basis entries found to export.")
    except Exception as e:
        logger.exception(f"Failed to export SVD basis: {e}")

    # If a held-out test dataset is provided, train sigma only on that dataset and evaluate
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
        sigma_modules = sigma_modules.to(cfg.device)

        # Use train split for sigma finetuning and Val split for evaluation
        train_dataset_name = test_ds
        val_dataset_name = test_ds + "Val"
        image_encoder = ImageEncoder(cfg.model).to(cfg.device)
        # ensure save_dir exists in cfg for classification head management
        with open_dict(cfg):
            if "save_dir" not in cfg:
                cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        classification_head = get_classification_head(cfg, train_dataset_name)
        model = ImageClassifier(
            image_encoder, classification_head).to(cfg.device)
        model.freeze_head()

        preprocess_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                size=224, scale=(0.5, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ] + model.train_preprocess.transforms[-3:])

        dataset = get_dataset(
            train_dataset_name,
            preprocess_fn,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
        )
        # 기본 학습 로더 (atlas와 동일한 k-shot 추출 방식 적용)
        train_loader = get_dataloader(
            dataset, is_train=True, args=cfg, image_encoder=None)

        if int(cfg.k_shot) > 0:
            try:
                target_dataset = val_dataset_name
                int_idx_path = os.path.join(
                    cfg.save_dir, target_dataset, f"{int(cfg.k_shot)}_shots_{int(cfg.seed)}.pt")
                os.makedirs(os.path.dirname(int_idx_path), exist_ok=True)
                if os.path.isfile(int_idx_path) and int(cfg.seed) == 1:
                    to_keep = torch.load(int_idx_path)
                else:
                    to_keep = get_n_shots(
                        dataset.train_dataset, int(cfg.k_shot),
                        model.classification_head.out_features, cfg)
                    torch.save(to_keep, int_idx_path)

                r = len(to_keep) / int(cfg.batch_size)
                if r < 10:
                    over_sampling = 10/r
                    over_sampling = int(over_sampling) + 1
                    logger.info(f"Oversampling {over_sampling} times")
                    to_keep = torch.cat([to_keep] * over_sampling)

                index_dataset = IndexWrapper(dataset.train_dataset)
                sampler = torch.utils.data.SubsetRandomSampler(to_keep)
                train_loader = torch.utils.data.DataLoader(
                    index_dataset, batch_size=int(cfg.batch_size), sampler=sampler, num_workers=8)
                logger.info(
                    f"Applied atlas-style k-shot sampling: k={cfg.k_shot}, total_selected={len(to_keep)}")
            except Exception as e:
                logger.exception(f"k-shot 샘플링 구성 실패: {e}. 기본 전체 데이터로 진행합니다.")

        params = [p for p in sigma_modules.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.sigma_lr_step_size), gamma=float(cfg.sigma_lr_gamma)
        )

        # capture base parameters and buffers for functional_call
        base_params = dict(model.image_encoder.named_parameters())
        base_buffers = dict(model.image_encoder.named_buffers())
        energy_train_start_time = time.time()
        print(f"batch size: {cfg.batch_size}")
        print(f"train_loader length: {len(train_loader)}")
        for epoch in range(int(cfg.sigma_epochs)):
            model.train()
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(cfg.device)
                labels = batch["labels"].to(cfg.device)
                print(f"inputs shape: {inputs.shape}")

                # build delta map with autograd connectivity
                delta_map = {}
                for safe_key, module in sigma_modules.items():
                    orig_key = sigma_key_map.get(safe_key, safe_key)
                    if orig_key in base_params and module.sigma.numel() > 0:
                        delta = module()
                        if delta.shape == base_params[orig_key].shape:
                            # ensure delta is on the same device as the base parameter
                            delta_map[orig_key] = delta.to(
                                base_params[orig_key].device)

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

                if (i + 1) % 100 == 0:
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
            epoch_time = time.time() - start_time
            print(f"Epoch time: {epoch_time:.3f} seconds")

            # step scheduler at end of epoch
            scheduler.step()
            # Evaluate on held-out dataset
            # rebuild delta and params map for eval (no grad)
            with torch.no_grad():
                eval_delta_map = {}
                for safe_key, module in sigma_modules.items():
                    orig_key = sigma_key_map.get(safe_key, safe_key)
                    if orig_key in base_params and module.sigma.numel() > 0:
                        d = module()
                        if d.shape == base_params[orig_key].shape:
                            eval_delta_map[orig_key] = d.to(
                                base_params[orig_key].device)

                eval_params_map = {}
                for name, p in base_params.items():
                    if name in eval_delta_map:
                        eval_params_map[name] = p.detach() + \
                            eval_delta_map[name]
                    else:
                        eval_params_map[name] = p.detach()

                wrapped_encoder = FunctionalEncoderWrapper(
                    model.image_encoder, base_buffers, eval_params_map
                )

            metrics = eval_single_dataset(
                wrapped_encoder, val_dataset_name, cfg)
            logger.info(
                f"Epoch: {epoch} test Acc on {val_dataset_name}: {metrics['top1']*100:.2f}%")
        energy_train_end_time = time.time()
        print(
            f"Energy train time: {energy_train_end_time - energy_train_start_time:.2f} seconds")
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
        ft_path = get_finetuned_path(
            cfg.model_location, train_dataset_name, cfg.model)
        model.image_encoder.save(ft_path)
        logger.info(f"Saved sigma-finetuned encoder to {ft_path}")

        # Evaluate on held-out dataset
        metrics = eval_single_dataset(
            model.image_encoder, val_dataset_name, cfg)
        logger.info(
            f"Test Acc on {val_dataset_name}: {metrics['top1']*100:.2f}%")

        # Optional: Adapter training (TIP/LPP) similar to atlas.py
        if isinstance(cfg.adapter, str) and cfg.adapter in ("lpp", "tip"):
            try:
                logger.info(
                    f"Training adapter '{cfg.adapter}' on {train_dataset_name}")
                train_adapter(model, train_loader, cfg,
                              train_dataset_name, val_dataset_name, logger)
            except Exception as e:
                logger.exception(f"Adapter training failed: {e}")


if __name__ == "__main__":
    # Lightweight argparse wrapper; passes values via environment variables to avoid Hydra conflicts
    parser = argparse.ArgumentParser(add_help=False)
    from src.datasets.registry import registry as DATASET_REGISTRY
    allowed_test_datasets = sorted(
        [name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")]
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        choices=allowed_test_datasets,
        default='CIFAR10',
        help="Held-out dataset to sigma-finetune on; one of %(choices)s",
    )
    parser.add_argument("--sigma_epochs", type=int, default=5)
    parser.add_argument("--sigma_lr", type=float, default=1e-3)
    parser.add_argument("--sigma_wd", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--k_shot",
        type=int,
        default=0,
        help="클래스별 학습 샘플 수. 0이면 전체 사용",
    )
    # scheduler hyperparams
    parser.add_argument("--sigma_lr_step_size", type=int, default=1)
    parser.add_argument("--sigma_lr_gamma", type=float, default=0.5)
    parser.add_argument("--svd_keep_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adapter",
        type=str,
        choices=["lpp", "tip", ""],
        default="",
        help="Adapter to train after sigma finetuning on held-out dataset (lpp/tip)",
    )
    # parse_known_args: leave Hydra args intact
    args, unknown = parser.parse_known_args()

    # Remove recognized app-specific args so Hydra doesn't see them
    # and complain about unrecognized arguments.
    sys.argv = [sys.argv[0]] + unknown

    os.environ["ENERGY_TEST_DATASET"] = args.test_dataset
    os.environ["ENERGY_SIGMA_EPOCHS"] = str(args.sigma_epochs)
    os.environ["ENERGY_SIGMA_LR"] = str(args.sigma_lr)
    os.environ["ENERGY_SIGMA_WD"] = str(args.sigma_wd)
    os.environ["ENERGY_BATCH_SIZE"] = str(args.batch_size)
    os.environ["ENERGY_SIGMA_LR_STEP_SIZE"] = str(args.sigma_lr_step_size)
    os.environ["ENERGY_SIGMA_LR_GAMMA"] = str(args.sigma_lr_gamma)
    os.environ["ENERGY_K_SHOT"] = str(args.k_shot)
    os.environ["ENERGY_SVD_KEEP_TOPK"] = str(args.svd_keep_topk)
    os.environ["ENERGY_SEED"] = str(args.seed)
    os.environ["ENERGY_ADAPTER"] = args.adapter if args.adapter is not None else ""
    my_app()
