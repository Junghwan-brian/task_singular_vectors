import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse
import sys

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import ALL_DATASETS
from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.variables_and_paths import get_finetuned_path
from src.utils.sigma_param import SigmaParametrization
import torch
from src.models.task_vectors import NonLinearTaskVector
from src.eval.aggregation import get_all_checkpoints
from torch.nn.utils.stateless import functional_call

import math
import torch


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
    각 태스크에서 k개의 축만 모아 직교 기반(U_orth, V_orth)과 sigma(diag)로 재구성.
    """
    device = config.device
    datasets = list(config.DATASETS)
    num_tasks = int(len(datasets))
    print(f"DATSETS: {datasets}")
    print("Computing SVD...")

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
            k = max(1, r // num_used)
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
                sum_s[start:end] = s[:k_i]
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
        cfg.device = "cuda:3" if torch.cuda.is_available() else "cpu"

        # override from environment variables (set by argparse in __main__)
        if os.environ.get("ENERGY_TEST_DATASET", ""):
            cfg.test_dataset = os.environ["ENERGY_TEST_DATASET"]

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

        dataset = get_dataset(
            train_dataset_name,
            model.train_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
        )
        train_loader = get_dataloader(
            dataset, is_train=True, args=cfg, image_encoder=None)

        params = [p for p in sigma_modules.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.sigma_lr_step_size), gamma=float(cfg.sigma_lr_gamma)
        )

        # capture base parameters and buffers for functional_call
        base_params = dict(model.image_encoder.named_parameters())
        base_buffers = dict(model.image_encoder.named_buffers())

        for epoch in range(int(cfg.sigma_epochs)):
            model.train()
            for i, batch in enumerate(train_loader):
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(cfg.device)
                labels = batch["labels"].to(cfg.device)

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
    # scheduler hyperparams
    parser.add_argument("--sigma-lr-step-size", type=int, default=1)
    parser.add_argument("--sigma-lr-gamma", type=float, default=0.5)
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

    my_app()
