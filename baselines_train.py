import os
import sys
import time
import logging
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
import torchvision
import math

from src.utils.variables_and_paths import ALL_DATASETS, get_finetuned_path
from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from atlas_src.utils import get_n_shots, IndexWrapper, TIPWrapper, LPPWrapper, _RepeatSampler


class LoRALinear(torch.nn.Module):
    def __init__(self, base_linear: torch.nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.r = int(max(1, r))
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_A = torch.nn.Parameter(torch.zeros(in_f, self.r))
        self.lora_B = torch.nn.Parameter(torch.zeros(self.r, out_f))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.dropout = torch.nn.Dropout(
            p=float(dropout)) if dropout and dropout > 0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.linear(x, self.base.weight, self.base.bias)
        lora_delta = self.dropout(x) @ self.lora_A @ self.lora_B
        return y + self.scaling * lora_delta

    @property
    def weight(self):
        # expose base linear's weight for external modules that reference .weight directly
        return self.base.weight

    @property
    def bias(self):
        # expose base linear's bias for external modules that reference .bias directly
        return self.base.bias


def apply_lora_to_module(module: torch.nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
        else:
            apply_lora_to_module(child, r=r, alpha=alpha, dropout=dropout)


def build_train_loader(cfg, model, train_dataset_name):
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
    train_loader = get_dataloader(
        dataset, is_train=True, args=cfg, image_encoder=None)

    if int(cfg.k_shot) > 0:
        try:
            val_dataset_name = train_dataset_name + \
                "Val" if not train_dataset_name.endswith(
                    "Val") else train_dataset_name
            int_idx_path = os.path.join(
                cfg.save_dir, val_dataset_name, f"{int(cfg.k_shot)}_shots_{int(cfg.seed)}.pt")
            os.makedirs(os.path.dirname(int_idx_path), exist_ok=True)
            if os.path.isfile(int_idx_path) and int(cfg.seed) == 1:
                to_keep = torch.load(int_idx_path)
            else:
                to_keep = get_n_shots(dataset.train_dataset, int(cfg.k_shot),
                                      model.classification_head.out_features, cfg)
                torch.save(to_keep, int_idx_path)

            r = len(to_keep) / int(cfg.batch_size)
            if r < 10:
                over_sampling = int(10 / r) + 1
                to_keep = torch.cat([to_keep] * over_sampling)

            index_dataset = IndexWrapper(dataset.train_dataset)
            sampler = torch.utils.data.SubsetRandomSampler(to_keep)
            train_loader = torch.utils.data.DataLoader(
                index_dataset, batch_size=int(cfg.batch_size), sampler=sampler, num_workers=8
            )
        except Exception as e:
            logging.getLogger("baselines_train").exception(
                f"k-shot 샘플링 실패: {e}. 전체 데이터로 진행합니다.")

    return train_loader


def cache_features(ddp_model, ddp_loader, device):
    all_features, all_labels, all_indexes, all_logits = [], [], [], []
    ddp_model = ddp_model.to(device)
    ddp_model.eval()
    with torch.no_grad():
        for batch in ddp_loader:
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            # Use model's normalized feature path for TIP/LPP stability
            logits, features = ddp_model(inputs, return_features=True)
            labels = batch["labels"]
            all_features.append(features.detach().cpu())
            all_labels.append(labels)
            if "index" in batch:
                all_indexes.append(batch["index"])
            else:
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
    return logits_cache, features_cache, labels, indexes_to_i


class ReturnFeaturesClassifier(torch.nn.Module):
    def __init__(self, base: ImageClassifier):
        super().__init__()
        self.image_encoder = base.image_encoder
        self.classification_head = base.classification_head
        self.train_preprocess = getattr(base, "train_preprocess", None)
        self.val_preprocess = getattr(base, "val_preprocess", None)

    def forward(self, inputs, return_features: bool = False):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        if return_features:
            return outputs, features / (features.norm(dim=-1, keepdim=True) + 1e-12)
        return outputs

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)


def eval_adapter_dataset(adapter_model: torch.nn.Module, dataset_name: str, cfg) -> dict:
    # use adapter's underlying classifier for preprocess
    base_model = adapter_model.model if hasattr(
        adapter_model, 'model') else adapter_model
    preprocess = getattr(base_model, 'val_preprocess', None)
    dataset = get_dataset(
        dataset_name,
        preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=cfg, image_encoder=None)
    device = cfg.device

    adapter_model.eval()
    top1, correct, n = 0.0, 0.0, 0.0
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)
            logits = adapter_model(x)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
    top1 = correct / max(1, n)
    return {"top1": top1}


def train_linear_probe(model, train_loader, cfg, train_dataset_name, logger):
    image_encoder = model.image_encoder
    # Freeze encoder, train head
    for p in image_encoder.parameters():
        p.requires_grad = False
    model.classification_head.weight.requires_grad_(True)
    model.classification_head.bias.requires_grad_(True)

    params = [p for p in model.classification_head.parameters()
              if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(
        cfg.lp_lr), weight_decay=float(cfg.lp_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(
        cfg.lp_lr_step_size), gamma=float(cfg.lp_lr_gamma))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = cfg.device
    model = model.to(device)
    model.train()

    num_batches = len(train_loader)
    print_every = 100
    for epoch in range(int(cfg.lp_epochs)):
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if (i + 1) % print_every == 0:
                logger.info(
                    f"[linear_probe] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    # Evaluate on validation split
    metrics = eval_single_dataset(
        model.image_encoder, train_dataset_name + "Val", cfg)
    logger.info(
        f"LinearProbe Acc on {train_dataset_name}Val: {metrics['top1']*100:.2f}%")

    # Save head only
    save_dir = cfg.save_dir if hasattr(
        cfg, 'save_dir') else os.path.join(cfg.model_location, cfg.model)
    os.makedirs(os.path.join(
        save_dir, train_dataset_name + "Val"), exist_ok=True)
    head_path = os.path.join(
        save_dir, train_dataset_name + "Val", f"linear_probe_head.pt")
    model.classification_head.save(head_path)
    logger.info(f"Saved linear probe head to {head_path}")


def train_tip_or_lpp(model, train_loader, cfg, train_dataset_name, logger, adapter: str):
    # Ensure classifier supports return_features for adapter wrappers
    ddp_model = ReturnFeaturesClassifier(model).to(cfg.device)

    # Build caches (logits/features) over the (possibly k-shot) training loader
    logits_cache, features_cache, labels, indexes_to_i = cache_features(
        ddp_model, train_loader, cfg.device)

    adapter_model = ddp_model
    if adapter == 'lpp':
        shots = int(cfg.k_shot) if int(cfg.k_shot) > 0 else 0
        adapter_model = LPPWrapper(
            adapter_model, features_cache, labels, shots)
        epochs = 300
        # device로 먼저 이동한 후 optimizer 생성
        adapter_model = adapter_model.to(cfg.device)
        # LPP는 adapter와 alpha_vec에 대해 서로 다른 learning rate 사용
        param_groups = [
            {'params': adapter_model.adapter.parameters(), 'lr': adapter_model.lr_temp},
            {'params': [adapter_model.alpha_vec], 'lr': adapter_model.lr_alpha}
        ]
    elif adapter == 'tip':
        adapter_model = TIPWrapper(adapter_model, features_cache, labels)
        epochs = 10
        # device로 먼저 이동한 후 optimizer 생성
        adapter_model = adapter_model.to(cfg.device)
        # TIP는 adapter와 beta_alpha 파라미터 모두 학습
        # TIP의 원 논리대로 adapter는 고정, beta_alpha만 학습
        param_groups = [
            {'params': [adapter_model.beta_alpha], 'lr': 1e-3}
        ]
    else:
        raise NotImplementedError(f"Adapter {adapter} unknown")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.adapter_wd)
    num_batches = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * num_batches)
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # 학습 가능한 모든 파라미터 수집 (gradient clipping용)
    params = [p for p in adapter_model.parameters() if p.requires_grad]

    # 명시적 에폭 반복으로 학습 반복을 보장
    adapter_model.train()
    total_cache = int(logits_cache.size(0))
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            step = epoch * max(1, num_batches) + i
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(cfg.device)

            if "index" in batch:
                ids = [indexes_to_i[j.item()] for j in batch['index']]
            else:
                # wrap-around indexing to always match batch size
                start_id = int(step % max(1, total_cache))
                ids = [int((start_id + t) % max(1, total_cache))
                       for t in range(len(inputs))]

            l_cache, f_cache = logits_cache[ids].to(
                inputs), features_cache[ids].to(inputs)
            logits = adapter_model(inputs, l_cache, f_cache)
            labels_b = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if (i + 1) % 100 == 0:
                # LPP와 TIP 모두 여러 param_groups를 가지므로 각각의 lr 출력
                lrs = [f"{pg['lr']:.6f}" for pg in optimizer.param_groups]
                logger.info(
                    f"[adapter:{adapter}] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lrs {lrs}")

            scheduler.step()

    # Evaluate adapter-wrapped model on validation split
    metrics = eval_adapter_dataset(
        adapter_model, train_dataset_name + "Val", cfg)
    logger.info(
        f"Adapter '{adapter}' Acc on {train_dataset_name}Val: {metrics['top1']*100:.2f}%")

    # Save adapter weights (only trainable params)
    adapter_coefs = {k: v for k, v in adapter_model.state_dict(
    ).items() if hasattr(v, 'requires_grad') and v.requires_grad}
    save_dir = cfg.save_dir if hasattr(
        cfg, 'save_dir') else os.path.join(cfg.model_location, cfg.model)
    os.makedirs(save_dir, exist_ok=True)
    adapter_path = os.path.join(
        save_dir, f"adapter_{train_dataset_name}_{adapter}.pt")
    torch.save(adapter_coefs, adapter_path)
    logger.info(f"Saved adapter weights to {adapter_path}")


def train_lora(model, train_loader, cfg, train_dataset_name, logger):
    # Apply LoRA to encoder
    apply_lora_to_module(model.image_encoder, r=int(cfg.lora_r), alpha=float(
        cfg.lora_alpha), dropout=float(cfg.lora_dropout))

    # Freeze base encoder params; LoRA params are trainable; optionally train head
    for p in model.image_encoder.parameters():
        if not isinstance(p, torch.nn.Parameter):
            continue
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, LoRALinear):
            # ensure LoRA params require grad
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
            if hasattr(module, 'base'):
                module.base.weight.requires_grad_(False)
                if module.base.bias is not None:
                    module.base.bias.requires_grad_(False)
        elif isinstance(module, torch.nn.Linear):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

    if bool(getattr(cfg, 'lora_tune_head', True)):
        model.classification_head.weight.requires_grad_(True)
        model.classification_head.bias.requires_grad_(True)
    else:
        model.classification_head.weight.requires_grad_(False)
        model.classification_head.bias.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(
        cfg.lora_lr), weight_decay=float(cfg.lora_wd))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(
        cfg.lora_lr_step_size), gamma=float(cfg.lora_lr_gamma))
    loss_fn = torch.nn.CrossEntropyLoss()

    device = cfg.device
    model = model.to(device)
    model.train()

    num_batches = len(train_loader)
    print_every = 100
    for epoch in range(int(cfg.lora_epochs)):
        for i, batch in enumerate(train_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if (i + 1) % print_every == 0:
                logger.info(
                    f"[lora] epoch {epoch} {i + 1}/{num_batches} loss {loss.item():.6f} lr {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

    # Evaluate on validation split
    metrics = eval_single_dataset(
        model.image_encoder, train_dataset_name + "Val", cfg)
    logger.info(
        f"LoRA Acc on {train_dataset_name}Val: {metrics['top1']*100:.2f}%")

    # Save encoder (with LoRA) and optionally head
    ft_path = get_finetuned_path(
        cfg.model_location, train_dataset_name, cfg.model)
    os.makedirs(os.path.dirname(ft_path), exist_ok=True)
    model.image_encoder.save(ft_path)
    logger.info(f"Saved LoRA-updated encoder to {ft_path}")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    # Defaults and env overrides similar to energy_train
    with open_dict(cfg):
        if "sigma_epochs" not in cfg:
            cfg.sigma_epochs = 5
        if "sigma_wd" not in cfg:
            cfg.sigma_wd = 0.0
        cfg.batch_size = int(os.environ.get(
            "BASELINE_BATCH_SIZE", os.environ.get("ENERGY_BATCH_SIZE", "256")))
        cfg.device = os.environ.get("BASELINE_DEVICE", None)
        if not cfg.device:
            cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.k_shot = int(os.environ.get("BASELINE_K_SHOT", "0"))
        cfg.seed = int(os.environ.get("BASELINE_SEED", "1"))
        cfg.adapter_wd = float(os.environ.get("BASELINE_ADAPTER_WD", "0.0"))

        # linear probe hyperparams
        cfg.lp_epochs = int(os.environ.get("BASELINE_LP_EPOCHS", "10"))
        cfg.lp_lr = float(os.environ.get("BASELINE_LP_LR", "1e-3"))
        cfg.lp_wd = float(os.environ.get("BASELINE_LP_WD", "0.0"))
        cfg.lp_lr_step_size = int(os.environ.get(
            "BASELINE_LP_LR_STEP_SIZE", "1"))
        cfg.lp_lr_gamma = float(os.environ.get("BASELINE_LP_LR_GAMMA", "0.5"))

        # LoRA hyperparams
        cfg.lora_epochs = int(os.environ.get("BASELINE_LORA_EPOCHS", "5"))
        cfg.lora_lr = float(os.environ.get("BASELINE_LORA_LR", "1e-4"))
        cfg.lora_wd = float(os.environ.get("BASELINE_LORA_WD", "0.0"))
        cfg.lora_lr_step_size = int(os.environ.get(
            "BASELINE_LORA_LR_STEP_SIZE", "1"))
        cfg.lora_lr_gamma = float(os.environ.get(
            "BASELINE_LORA_LR_GAMMA", "0.5"))
        cfg.lora_r = int(os.environ.get("BASELINE_LORA_R", "8"))
        cfg.lora_alpha = float(os.environ.get("BASELINE_LORA_ALPHA", "16.0"))
        cfg.lora_dropout = float(os.environ.get(
            "BASELINE_LORA_DROPOUT", "0.0"))
        cfg.lora_tune_head = bool(
            int(os.environ.get("BASELINE_LORA_TUNE_HEAD", "1")))

        # baseline choice and optional single dataset
        cfg.baseline_method = os.environ.get("BASELINE_METHOD", "linear_probe")
        cfg.target_dataset = os.environ.get("BASELINE_DATASET", "")

        # paths alignment
        if not hasattr(cfg, "model_location") or cfg.model_location in (None, ""):
            base_root = getattr(cfg, "save", None)
            if base_root is None:
                base_root = os.path.expanduser(
                    "/disk3/junghwan/task_vector/models/checkpoints")
            cfg.model_location = base_root
        if not hasattr(cfg, "save_dir") or cfg.save_dir in (None, ""):
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        if not hasattr(cfg, "data_location") or cfg.data_location in (None, ""):
            cfg.data_location = os.path.expanduser("datasets")

    # Dataset selection: use cfg.DATASETS or ALL_DATASETS
    if cfg.DATASETS == "":
        base_list = ALL_DATASETS[: cfg.num_tasks]
    else:
        base_list = list(cfg.DATASETS)

    if cfg.target_dataset:
        if cfg.target_dataset in base_list:
            base_list = [cfg.target_dataset]
        else:
            base_list = [cfg.target_dataset]
    cfg.DATASETS = base_list
    cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    # logging
    logger = logging.getLogger("baselines_train")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler("baselines_train.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info(cfg.method.full_name)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)

    # seed
    try:
        import random
        import numpy as np
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # iterate datasets
    for train_dataset_name in cfg.DATASETS:
        val_dataset_name = train_dataset_name + "Val"
        image_encoder = ImageEncoder(cfg.model).to(cfg.device)
        with open_dict(cfg):
            if "save_dir" not in cfg:
                cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        classification_head = get_classification_head(cfg, train_dataset_name)
        model = ImageClassifier(
            image_encoder, classification_head).to(cfg.device)

        model.freeze_head()  # keep head frozen by default; specific trainers will adjust

        train_loader = build_train_loader(cfg, model, train_dataset_name)

        logger.info(
            f"Training baseline '{cfg.baseline_method}' on {train_dataset_name}")
        if cfg.baseline_method == 'linear_probe':
            train_linear_probe(model, train_loader, cfg,
                               train_dataset_name, logger)
        elif cfg.baseline_method == 'tip_adapter':
            train_tip_or_lpp(model, train_loader, cfg,
                             train_dataset_name, logger, adapter='tip')
        elif cfg.baseline_method == 'lp_plus_plus':
            train_tip_or_lpp(model, train_loader, cfg,
                             train_dataset_name, logger, adapter='lpp')
        elif cfg.baseline_method == 'lora':
            train_lora(model, train_loader, cfg, train_dataset_name, logger)
        else:
            raise NotImplementedError(
                f"Unknown baseline method: {cfg.baseline_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    from src.datasets.registry import registry as DATASET_REGISTRY
    allowed_datasets = sorted(
        [name for name in DATASET_REGISTRY.keys() if not name.endswith("Val")])

    parser.add_argument(
        "--baseline_method",
        type=str,
        choices=["linear_probe", "tip_adapter", "lp_plus_plus", "lora"],
        default="linear_probe",
        help="Baseline to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=allowed_datasets + [""],
        default="CIFAR10",
        help="Train only on this dataset (otherwise iterate cfg.DATASETS)",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_shot", type=int, default=0,
                        help="클래스별 학습 샘플 수. 0이면 전체 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    # linear probe hparams
    parser.add_argument("--lp_epochs", type=int, default=10)
    parser.add_argument("--lp_lr", type=float, default=1e-3)
    parser.add_argument("--lp_wd", type=float, default=0.0)
    parser.add_argument("--lp_lr_step_size", type=int, default=1)
    parser.add_argument("--lp_lr_gamma", type=float, default=0.5)

    # TIP/LPP hparams
    parser.add_argument("--adapter_wd", type=float, default=0.0)

    # LoRA hparams
    parser.add_argument("--lora_epochs", type=int, default=5)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--lora_wd", type=float, default=0.0)
    parser.add_argument("--lora_lr_step_size", type=int, default=1)
    parser.add_argument("--lora_lr_gamma", type=float, default=0.5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_tune_head", type=int, default=1)

    args, unknown = parser.parse_known_args()

    # pass via environment vars to avoid Hydra conflicts
    sys.argv = [sys.argv[0]] + unknown

    os.environ["BASELINE_METHOD"] = args.baseline_method
    os.environ["BASELINE_DATASET"] = args.dataset
    os.environ["BASELINE_BATCH_SIZE"] = str(args.batch_size)
    os.environ["BASELINE_K_SHOT"] = str(args.k_shot)
    os.environ["BASELINE_SEED"] = str(args.seed)
    if args.device:
        os.environ["BASELINE_DEVICE"] = args.device

    os.environ["BASELINE_LP_EPOCHS"] = str(args.lp_epochs)
    os.environ["BASELINE_LP_LR"] = str(args.lp_lr)
    os.environ["BASELINE_LP_WD"] = str(args.lp_wd)
    os.environ["BASELINE_LP_LR_STEP_SIZE"] = str(args.lp_lr_step_size)
    os.environ["BASELINE_LP_LR_GAMMA"] = str(args.lp_lr_gamma)

    os.environ["BASELINE_ADAPTER_WD"] = str(args.adapter_wd)

    os.environ["BASELINE_LORA_EPOCHS"] = str(args.lora_epochs)
    os.environ["BASELINE_LORA_LR"] = str(args.lora_lr)
    os.environ["BASELINE_LORA_WD"] = str(args.lora_wd)
    os.environ["BASELINE_LORA_LR_STEP_SIZE"] = str(args.lora_lr_step_size)
    os.environ["BASELINE_LORA_LR_GAMMA"] = str(args.lora_lr_gamma)
    os.environ["BASELINE_LORA_R"] = str(args.lora_r)
    os.environ["BASELINE_LORA_ALPHA"] = str(args.lora_alpha)
    os.environ["BASELINE_LORA_DROPOUT"] = str(args.lora_dropout)
    os.environ["BASELINE_LORA_TUNE_HEAD"] = str(int(args.lora_tune_head))

    my_app()
