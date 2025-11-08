import os
import time
import json
import math
import random
import logging
from typing import Dict, Iterator, List, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
try:
    from torch.func import functional_call as fcall
except Exception:
    from torch.nn.utils.stateless import functional_call as fcall
import torchvision
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse

from src.utils.variables_and_paths import (
    ALL_DATASETS,
)
from src.datasets import get_dataloader, maybe_dictionarize, get_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.eval.eval import eval_single_dataset


def setup_simple_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _build_train_transform(image_encoder: ImageEncoder) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        + image_encoder.train_preprocess.transforms[-3:]
    )


def _make_dataloader_for_dataset(
    dataset_name: str,
    image_encoder: ImageEncoder,
    cfg: DictConfig,
) -> torch.utils.data.DataLoader:
    train_preprocess = _build_train_transform(image_encoder)
    dataset_train = get_dataset(
        dataset_name,
        train_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    loader = get_dataloader(dataset_train, is_train=True,
                            args=cfg, image_encoder=None)
    return loader


def _next_two_batches(
    loader: torch.utils.data.DataLoader,
    iters: Dict[str, Iterator],
    key: str,
) -> Tuple[Dict, Dict]:
    def _ensure_iter():
        try:
            return iters[key]
        except KeyError:
            iters[key] = iter(loader)
            return iters[key]

    it = _ensure_iter()
    try:
        batch_a = next(it)
    except StopIteration:
        iters[key] = iter(loader)
        it = iters[key]
        batch_a = next(it)
    try:
        batch_b = next(it)
    except StopIteration:
        iters[key] = iter(loader)
        it = iters[key]
        batch_b = next(it)
    return maybe_dictionarize(batch_a), maybe_dictionarize(batch_b)


def forward_features_with_params(
    encoder: ImageEncoder,
    params_map: Dict[str, torch.Tensor],
    buffers_map: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    merged = {}
    merged.update(buffers_map)
    merged.update(params_map)
    return fcall(encoder, merged, (x,))


def maml_meta_train(
    cfg: DictConfig,
    logger: logging.Logger,
) -> ImageEncoder:
    with open_dict(cfg):
        cfg.data_location = os.path.expanduser(cfg.data_location)
        if cfg.get("seed") is None:
            cfg.seed = 1
        if cfg.get("batch_size") is None:
            cfg.batch_size = 64
        # heads.py가 사용하는 save_dir 보장
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        os.makedirs(cfg.save_dir, exist_ok=True)
        # meta settings defaults
        cfg.meta_iterations = int(getattr(cfg, "meta_iterations", 2000))
        cfg.meta_batch_size = int(getattr(cfg, "meta_batch_size", 4))
        cfg.inner_steps = int(getattr(cfg, "inner_steps", 1))
        cfg.inner_lr = float(getattr(cfg, "inner_lr", 1e-2))
        cfg.meta_lr = float(getattr(cfg, "meta_lr", 5e-5))
        cfg.meta_wd = float(getattr(cfg, "meta_wd", 0.0))
        cfg.grad_clip = float(getattr(cfg, "grad_clip", 1.0))
        cfg.save_model_dir = getattr(
            cfg,
            "save_model_dir",
            "/disk3/junghwan/task_vector/maml/models",
        )
        cfg.save_result_dir = getattr(
            cfg,
            "save_result_dir",
            "/disk3/junghwan/task_vector/maml/results",
        )

    set_seed(int(cfg.seed))

    assert cfg.test_dataset, "test_dataset must be provided"

    # 구성: 15개 전체 중 테스트 제외(= 14개)로 meta-train
    if hasattr(cfg, "DATASETS_ALL") and cfg.DATASETS_ALL:
        base_list = list(cfg.DATASETS_ALL)
    else:
        base_list = list(ALL_DATASETS)
    if cfg.test_dataset in base_list:
        base_list = [d for d in base_list if d != cfg.test_dataset]
    # 안전장치: 최소 1개 이상
    assert len(
        base_list) >= 1, "No training datasets available after excluding test dataset"

    logger.info(
        f"Meta-train tasks (train split only, exclude test): {base_list}")
    logger.info(f"Model: {cfg.model}")

    # 모형 준비 (메타 파라미터 = encoder 파라미터)
    image_encoder = ImageEncoder(cfg.model).cuda()
    # 메타학습을 위해 인코더 파라미터의 미분 허용 보장
    for p in image_encoder.parameters():
        p.requires_grad_(True)
    meta_optimizer = torch.optim.AdamW(
        image_encoder.parameters(), lr=cfg.meta_lr, weight_decay=cfg.meta_wd
    )

    # 각 태스크(train split)용 로더/이터레이터
    loaders: Dict[str, torch.utils.data.DataLoader] = {}
    for ds in base_list:
        loaders[ds] = _make_dataloader_for_dataset(ds, image_encoder, cfg)
    iters: Dict[str, Iterator] = {}

    # 학습 루프
    image_encoder.train()
    start_time = time.time()
    for it in range(int(cfg.meta_iterations)):
        # 메타 배치: 무작위 태스크 일부 샘플
        task_batch = random.sample(base_list, k=min(
            cfg.meta_batch_size, len(base_list)))

        # 모형의 현재 파라미터/버퍼 스냅샷(메타 시작점)
        base_params = {n: p for n, p in image_encoder.named_parameters()}
        base_buffers = {n: b for n, b in image_encoder.named_buffers()}

        meta_optimizer.zero_grad(set_to_none=True)
        outer_losses: List[torch.Tensor] = []

        for task_name in task_batch:
            loader = loaders[task_name]
            support_batch, query_batch = _next_two_batches(
                loader, iters, task_name)
            xs = support_batch["images"].cuda()
            ys = support_batch["labels"].cuda()
            xq = query_batch["images"].cuda()
            yq = query_batch["labels"].cuda()

            # 태스크별 분류기 헤드 생성 (태스크 전용, 메타 파라미터 아님)
            head = get_classification_head(cfg, task_name).cuda()

            # fast weights (encoder/head) 초기화
            fast_encoder = {n: p for n, p in base_params.items()}
            fast_head = {n: p for n, p in head.named_parameters()}

            # inner-loop: support로 적응 (고차미분 유지)
            for _ in range(int(cfg.inner_steps)):
                feats = forward_features_with_params(
                    image_encoder, fast_encoder, base_buffers, xs)
                logits = fcall(head, fast_head, (feats,))
                loss_inner = nn.functional.cross_entropy(logits, ys)
                grads_encoder = torch.autograd.grad(
                    loss_inner, list(fast_encoder.values()), create_graph=True, allow_unused=True
                )
                grads_head = torch.autograd.grad(
                    loss_inner, list(fast_head.values()), create_graph=True, allow_unused=True
                )
                # SGD step on copies
                updated_encoder = {}
                for (name, param), grad in zip(fast_encoder.items(), grads_encoder):
                    if grad is None:
                        grad = torch.zeros_like(param)
                    updated_encoder[name] = param - cfg.inner_lr * grad
                fast_encoder = updated_encoder
                updated_head = {}
                for (name, param), grad in zip(fast_head.items(), grads_head):
                    if grad is None:
                        grad = torch.zeros_like(param)
                    updated_head[name] = param - cfg.inner_lr * grad
                fast_head = updated_head

            # outer loss: query로 계산
            with torch.set_grad_enabled(True):
                feats_q = forward_features_with_params(
                    image_encoder, fast_encoder, base_buffers, xq)
                logits_q = fcall(head, fast_head, (feats_q,))
                loss_outer = nn.functional.cross_entropy(logits_q, yq)
            outer_losses.append(loss_outer)

        if not outer_losses:
            continue
        outer_loss = torch.stack(outer_losses).mean()
        outer_loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            clip_grad_norm_(image_encoder.parameters(),
                            max_norm=float(cfg.grad_clip))
        meta_optimizer.step()

        if (it + 1) % 50 == 0:
            elapsed = time.time() - start_time
            ips = (it + 1) / max(1e-6, elapsed)
            logging.info(
                f"[MAML] iter {it + 1}/{cfg.meta_iterations} "
                f"outer_loss {outer_loss.item():.4f} it/s {ips:.2f}"
            )

    image_encoder.eval()
    return image_encoder


def finetune_on_test_dataset(
    encoder: ImageEncoder,
    cfg: DictConfig,
    logger: logging.Logger,
) -> Dict:
    with open_dict(cfg):
        cfg.finetune_epochs = int(getattr(cfg, "finetune_epochs", 20))
        cfg.finetune_lr = float(getattr(cfg, "finetune_lr", 5e-5))
        cfg.finetune_wd = float(getattr(cfg, "finetune_wd", 0.0))
        cfg.lr_step_size = int(getattr(cfg, "lr_step_size", 0))
        cfg.lr_gamma = float(getattr(cfg, "lr_gamma", 0.1))
        # heads.py가 사용하는 save_dir 보장
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        os.makedirs(cfg.save_dir, exist_ok=True)

    # 데이터/헤드
    train_preprocess = _build_train_transform(encoder)
    dataset_train = get_dataset(
        cfg.test_dataset,
        train_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    train_loader = get_dataloader(
        dataset_train, is_train=True, args=cfg, image_encoder=None)
    val_dataset_name = cfg.test_dataset + "Val"

    head = get_classification_head(cfg, cfg.test_dataset).cuda()
    model = ImageClassifier(encoder, head).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.finetune_lr, weight_decay=cfg.finetune_wd)
    scheduler = None
    if cfg.lr_step_size and cfg.lr_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.lr_step_size), gamma=float(cfg.lr_gamma))

    train_loss_history: List[float] = []
    val_acc_history: List[float] = []

    # 검증 데이터 로더 (파인튜닝된 head를 사용해 평가)
    val_preprocess = model.image_encoder.val_preprocess
    dataset_val = get_dataset(
        val_dataset_name,
        val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    val_loader = get_dataloader(
        dataset_val, is_train=False, args=cfg, image_encoder=None)

    logger.info(
        f"Start fine-tuning on test dataset {cfg.test_dataset} for {cfg.finetune_epochs} epochs")
    for epoch in range(int(cfg.finetune_epochs)):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = maybe_dictionarize(batch)
            x = batch["images"].cuda()
            y = batch["labels"].cuda()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        train_loss_history.append(avg_train_loss)

        # 검증: validation split에서 accuracy (현재 head 포함)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                b = maybe_dictionarize(batch)
                x = b["images"].cuda()
                y = b["labels"].cuda()
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        val_acc = float(correct / max(1, total))
        val_acc_history.append(val_acc)

        logger.info(
            f"[FT] epoch {epoch + 1}/{cfg.finetune_epochs} "
            f"train_loss {avg_train_loss:.4f} val_acc {val_acc * 100:.2f}%"
        )

    return {
        "train_loss_history": train_loss_history,
        "val_acc_history": val_acc_history,
    }


def run_maml(cfg: DictConfig) -> None:
    logger = setup_simple_logger(__name__)

    # 메타 학습
    encoder = maml_meta_train(cfg, logger)

    # 모델 저장 (테스트 데이터 이름 사용)
    os.makedirs(cfg.save_model_dir, exist_ok=True)
    model_save_path = os.path.join(
        cfg.save_model_dir, f"{cfg.test_dataset}.pt")
    encoder.save(model_save_path)
    logger.info(f"✓ Saved meta-trained encoder to {model_save_path}")

    # 테스트 데이터로 파인튜닝 및 결과 수집
    results = finetune_on_test_dataset(encoder, cfg, logger)

    # 결과 저장
    os.makedirs(cfg.save_result_dir, exist_ok=True)
    result_path = os.path.join(cfg.save_result_dir, f"{cfg.test_dataset}.json")
    payload = {
        "test_dataset": cfg.test_dataset,
        "model": cfg.model,
        "batch_size": cfg.batch_size,
        "meta_iterations": cfg.meta_iterations,
        "meta_batch_size": cfg.meta_batch_size,
        "inner_steps": cfg.inner_steps,
        "inner_lr": cfg.inner_lr,
        "meta_lr": cfg.meta_lr,
        "meta_wd": cfg.meta_wd,
        "finetune_epochs": cfg.finetune_epochs,
        "finetune_lr": cfg.finetune_lr,
        "finetune_wd": cfg.finetune_wd,
        "train_loss_history": results["train_loss_history"],
        "val_acc_history": results["val_acc_history"],
    }
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"✓ Saved results to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAML meta-training for fast adaptation across datasets (train split only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config_reverse.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="Flowers102",
        help="Held-out dataset to exclude from meta-training and adapt on",
    )
    parser.add_argument("--model", type=str,
                        help="Vision backbone (e.g., ViT-B-32)")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for loaders")
    parser.add_argument("--seed", type=int, help="Random seed")

    # meta settings
    parser.add_argument("--meta_iterations", type=int,
                        help="Number of meta-iterations (outer steps)")
    parser.add_argument("--meta_batch_size", type=int,
                        help="Number of tasks per meta-iteration")
    parser.add_argument("--inner_steps", type=int,
                        help="Inner-loop gradient steps")
    parser.add_argument("--inner_lr", type=float,
                        help="Inner-loop learning rate")
    parser.add_argument("--meta_lr", type=float,
                        help="Outer-loop learning rate")
    parser.add_argument("--meta_wd", type=float,
                        help="Outer-loop weight decay")
    parser.add_argument("--grad_clip", type=float,
                        help="Gradient clipping max norm")

    # finetune settings
    parser.add_argument("--finetune_epochs", type=int,
                        help="Fine-tuning epochs on test dataset")
    parser.add_argument("--finetune_lr", type=float,
                        help="Fine-tuning learning rate")
    parser.add_argument("--finetune_wd", type=float,
                        help="Fine-tuning weight decay")
    parser.add_argument("--lr_step_size", type=int,
                        help="LR scheduler step size for fine-tuning")
    parser.add_argument("--lr_gamma", type=float,
                        help="LR scheduler gamma for fine-tuning")

    # save dirs
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="/disk3/junghwan/task_vector/maml/models",
        help="Directory to save meta-trained models",
    )
    parser.add_argument(
        "--save_result_dir",
        type=str,
        default="/disk3/junghwan/task_vector/maml/results",
        help="Directory to save result JSONs",
    )

    args = parser.parse_args()
    cfg = load_config(args.config_file)
    cli_overrides = {k: v for k, v in vars(
        args).items() if v is not None and k != "config_file"}
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))
    if not cfg.get("test_dataset"):
        parser.error("--test_dataset is required")
    OmegaConf.set_struct(cfg, True)
    run_maml(cfg)
