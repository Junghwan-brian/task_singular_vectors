import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import argparse

from src.eval.aggregation import create_task_vector
from src.utils.variables_and_paths import ALL_DATASETS
from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.variables_and_paths import get_finetuned_path
from src.utils.sigma_param import SigmaParametrization
import torch


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    # add defaults for test-time sigma finetuning
    with open_dict(cfg):
        if "test_dataset" not in cfg:
            cfg.test_dataset = ""
        if "sigma_epochs" not in cfg:
            cfg.sigma_epochs = 5
        if "sigma_lr" not in cfg:
            cfg.sigma_lr = 1e-3
        if "sigma_wd" not in cfg:
            cfg.sigma_wd = 0.0
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

    # create final task vector
    task_vector_dict, eval_masks, svd_dict = create_task_vector(cfg)

    # Export SVD bases and initial sigma diagonals for sigma-only finetuning
    try:
        basis = {}
        for key, value in task_vector_dict.vector.items():
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
        for key, fv in basis.items():
            if all(k in fv for k in ("U", "V", "sigma")):
                U, V, sigma = fv["U"], fv["V"], fv["sigma"]
                if U.ndim == 2 and V.ndim == 2 and sigma.ndim == 1:
                    sigma_modules[key] = SigmaParametrization(U, V, sigma)
        sigma_modules = sigma_modules.cuda()

        test_dataset_name = test_ds + "Val"
        image_encoder = ImageEncoder(cfg.model).cuda()
        classification_head = get_classification_head(cfg, test_dataset_name)
        model = ImageClassifier(image_encoder, classification_head).cuda()
        model.freeze_head()

        dataset = get_dataset(
            test_dataset_name,
            model.train_preprocess,
            location=cfg.data_location,
            batch_size=cfg.batch_size,
        )
        train_loader = get_dataloader(
            dataset, is_train=True, args=cfg, image_encoder=None)

        params = [p for p in sigma_modules.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)

        base_sd = model.image_encoder.state_dict()

        def apply_sigma_deltas(base_state_dict, modules: torch.nn.ModuleDict):
            with torch.no_grad():
                new_sd = {k: v.clone() for k, v in base_state_dict.items()}
                for k, m in modules.items():
                    if k in new_sd and m.sigma.numel() > 0:
                        delta = m().to(new_sd[k].device)
                        if new_sd[k].shape == delta.shape:
                            new_sd[k] = new_sd[k] + delta
                return new_sd

        for epoch in range(int(cfg.sigma_epochs)):
            model.train()
            for i, batch in enumerate(train_loader):
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()

                new_sd = apply_sigma_deltas(base_sd, sigma_modules)
                model.image_encoder.load_state_dict(new_sd)

                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                if i % 100 == 0:
                    logger.info(
                        f"[sigma] epoch {epoch} iter {i} loss {loss.item():.4f}")

        # Finalize weights and save
        final_sd = apply_sigma_deltas(base_sd, sigma_modules)
        model.image_encoder.load_state_dict(final_sd)
        ft_path = get_finetuned_path(
            cfg.model_location, test_dataset_name, cfg.model)
        model.image_encoder.save(ft_path)
        logger.info(f"Saved sigma-finetuned encoder to {ft_path}")

        # Evaluate on held-out dataset
        metrics = eval_single_dataset(
            model.image_encoder, test_dataset_name, cfg)
        logger.info(
            f"Test Acc on {test_dataset_name}: {metrics['top1']*100:.2f}%")


if __name__ == "__main__":
    # Lightweight argparse wrapper; passes values via environment variables to avoid Hydra conflicts
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test-dataset", type=str, default="")
    parser.add_argument("--sigma-epochs", type=int, default=None)
    parser.add_argument("--sigma-lr", type=float, default=None)
    parser.add_argument("--sigma-wd", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    # parse_known_args: leave Hydra args intact
    args, _ = parser.parse_known_args()

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

    my_app()
