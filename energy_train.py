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
from src.models.task_vectors import ImageEncoder, NonLinearTaskVector
from src.eval.aggregation import get_all_checkpoints
from torch.nn.utils.stateless import functional_call


def compute_and_sum_svd_mem_reduction(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print(f"DATSETS: {config.DATASETS}")
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key and 'positional' not in key and 'token_embedding' not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                # u_u @ v_u 로 sum_u에 대한 orthogonal matrix 생성
                u_orth = u_u @ v_u
                v_orth = u_v @ v_v
                new_vector[key] = [
                    u_orth,
                    torch.diag(sum_s),
                    v_orth
                ]

    return new_vector


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    # add defaults for test-time sigma finetuning
    with open_dict(cfg):
        if "test_dataset" not in cfg:
            cfg.test_dataset = ""
        if "sigma_epochs" not in cfg:
            cfg.sigma_epochs = 1
        if "sigma_lr" not in cfg:
            cfg.sigma_lr = 1e-4
        if "sigma_wd" not in cfg:
            cfg.sigma_wd = 0.0
        # scheduler defaults
        if "sigma_lr_step_size" not in cfg:
            cfg.sigma_lr_step_size = 1
        if "sigma_lr_gamma" not in cfg:
            cfg.sigma_lr_gamma = 0.9
        if "batch_size" not in cfg:
            cfg.batch_size = 128
        if "device" not in cfg:
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        # override from environment variables (set by argparse in __main__)
        if os.environ.get("ENERGY_TEST_DATASET", ""):
            cfg.test_dataset = os.environ["ENERGY_TEST_DATASET"]
        if os.environ.get("ENERGY_SIGMA_EPOCHS", ""):
            try:
                cfg.sigma_epochs = int(os.environ["ENERGY_SIGMA_EPOCHS"])
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
        sigma_modules = sigma_modules.cuda()

        # Use train split for sigma finetuning and Val split for evaluation
        train_dataset_name = test_ds
        val_dataset_name = test_ds + "Val"
        image_encoder = ImageEncoder(cfg.model).cuda()
        # ensure save_dir exists in cfg for classification head management
        with open_dict(cfg):
            if "save_dir" not in cfg:
                cfg.save_dir = os.path.join(cfg.model_location, cfg.model)
        classification_head = get_classification_head(cfg, train_dataset_name)
        model = ImageClassifier(image_encoder, classification_head).cuda()
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
            logger.info(
                f"[sigma] epoch {epoch} end lr {optimizer.param_groups[0]['lr']:.6f}")

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
    parser.add_argument("--sigma_epochs", type=int, default=None)
    parser.add_argument("--sigma_lr", type=float, default=None)
    parser.add_argument("--sigma_wd", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    # scheduler hyperparams
    parser.add_argument("--sigma-lr-step-size", type=int, default=None)
    parser.add_argument("--sigma-lr-gamma", type=float, default=None)
    # parse_known_args: leave Hydra args intact
    args, unknown = parser.parse_known_args()

    # Remove recognized app-specific args so Hydra doesn't see them
    # and complain about unrecognized arguments.
    sys.argv = [sys.argv[0]] + unknown

    if args.test_dataset:
        os.environ["ENERGY_TEST_DATASET"] = args.test_dataset
    if args.sigma_epochs is not None:
        os.environ["ENERGY_SIGMA_EPOCHS"] = str(args.sigma_epochs)
    if args.sigma_lr is not None:
        os.environ["ENERGY_SIGMA_LR"] = str(args.sigma_lr)
    if args.sigma_wd is not None:
        os.environ["ENERGY_SIGMA_WD"] = str(args.sigma_wd)
    if args.batch_size is not None:
        os.environ["ENERGY_BATCH_SIZE"] = str(args.batch_size)
    if getattr(args, "sigma_lr_step_size", None) is not None:
        os.environ["ENERGY_SIGMA_LR_STEP_SIZE"] = str(args.sigma_lr_step_size)
    if getattr(args, "sigma_lr_gamma", None) is not None:
        os.environ["ENERGY_SIGMA_LR_GAMMA"] = str(args.sigma_lr_gamma)

    my_app()
