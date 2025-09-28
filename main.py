import os
from pprint import pprint

import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from src.eval.aggregation import create_task_vector
from src.eval.eval_utils import perform_eval_with_merged_vector
from src.utils.variables_and_paths import ALL_DATASETS


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    if cfg.DATASETS == "":
        cfg.DATASETS = ALL_DATASETS[: cfg.num_tasks]
    else:
        cfg.num_tasks = len(cfg.DATASETS)
    cfg.DATASETS_VAL = [dataset + "Val" for dataset in cfg.DATASETS]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    # set up experiment logging
    logger = logging.getLogger("task_singular_vectors")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler = logging.FileHandler("main.log")
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
    print("*" * 100)
    print("*" * 37, "Created task vector dict", "*" * 37)
    print("*" * 100)
    print("\n" * 3)

    # perform evaluation and log results
    print("*" * 100)
    print("*" * 39, "Starting Evaluation.", "*" * 39)
    print("*" * 100)
    additive_accuracies = perform_eval_with_merged_vector(
        cfg, task_vector_dict, eval_masks, svd_dict
    )
    pprint(additive_accuracies, width=1)
    logger.info(f"additive_accuracies: {additive_accuracies}")


if __name__ == "__main__":
    my_app()
