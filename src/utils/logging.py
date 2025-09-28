import os
import json
import logging
import hydra
from argparse import Namespace
from pprint import pprint

from src.utils.distributed import is_main_process


def setup_logging(filename: str = "main.log", level: int = logging.INFO) -> logging.Logger:
    """Initialize and return a module-level logger.

    Creates a rotating file logger and a console logger if not already present.
    """
    logger = logging.getLogger("task_singular_vectors")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file handler fails (e.g., no permission), continue with console only
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_dict(dictionary: dict):
    if is_main_process():
        logger = logging.getLogger("task_singular_vectors")
        logger.info(json.dumps(dictionary))


def log_results(final_results, args):
    if args.method.name == "tall_mask":
        mask_suffix = f"tall_mask_ties" if args.method.use_ties else f"tall_mask_ta"
    elif args.method.name == "mag_masking":
        mask_suffix = "mag_mask"
    elif args.method.name == "consensus":
        mask_suffix = (
            f"k_{args.method.prun_thre_k}_ties"
            if args.method.use_ties
            else f"k_{args.method.prun_thre_k}_ta"
        )
    elif args.method.name == "TSVM":
        mask_suffix = "svd"
    else:
        mask_suffix = ""

    if not os.path.exists(f"results/{args.save_subfolder}"):
        os.makedirs(f"results/{args.save_subfolder}")

    if args.num_tasks < 8:
        cleaned_datasets = (
            str(args.DATASETS)
            .translate(str.maketrans("", "", '[]",'))
            .replace(" ", "_")
            .replace("'", "")
        )
        save_file = f"results/{args.save_subfolder}/{args.model}_{args.num_tasks}tasks_{args.method.full_name}_nonlinear_additions_{mask_suffix}_{cleaned_datasets}.json"
    else:
        save_file = f"results/{args.save_subfolder}/{args.model}_{args.num_tasks}tasks_{args.method.full_name}_nonlinear_additions_{mask_suffix}.json"

    with open(save_file, "w") as f:
        json.dump(final_results, f, indent=4)
    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_save_file = f"{args.method.full_name}_nonlinear_additions.json"
    hydra_save_file = os.path.join(hydra_dir, hydra_save_file)
    json.dump(final_results, open(hydra_save_file, "w"), indent=4)

    print("saved results to: ", save_file)
    print("saved results to: ", hydra_save_file)
    logger = logging.getLogger("task_singular_vectors")
    logger.info(f"Results saved to {save_file} and {hydra_save_file}")
