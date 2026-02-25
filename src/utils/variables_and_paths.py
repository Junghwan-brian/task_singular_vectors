from pathlib import Path
from typing import Literal

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENCLIP_CACHEDIR = Path(
    Path.home(), "openclip-cachedir", "open_clip").as_posix()
CACHEDIR = None

ALL_DATASETS = [
    "DTD",
    "GTSRB",
    "MNIST",
    "SVHN",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "CIFAR100",
    "PCAM",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "EMNIST",
    "CUB200",
    "FGVCAircraft",
    "Country211"
]

DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def cleanup_dataset_name(dataset_name: str):
    return dataset_name.replace("Val", "") + "Val"


def get_zeroshot_path(root, dataset, model):
    return Path(root, model, f"nonlinear_zeroshot.pt").as_posix()


def get_finetuned_path(root, dataset, model):
    base_dir = Path(root, model, cleanup_dataset_name(dataset))
    nonlinear_path = base_dir / "nonlinear_finetuned.pt"
    linear_path = base_dir / "finetuned.pt"
    # 우선순위: nonlinear_finetuned.pt > finetuned.pt
    if nonlinear_path.exists():
        return nonlinear_path.as_posix()
    if linear_path.exists():
        return linear_path.as_posix()
    # 둘 다 없으면 기본 경로(nonlinear_finetuned.pt)를 반환하여 상위 로직에서 존재 여부 체크
    return nonlinear_path.as_posix()


def get_energy_finetuned_path(root, dataset, model):
    return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_energy_finetuned.pt").as_posix()


def get_single_task_accuracies_path(model):
    return Path("results/single_task", model, f"nonlinear_ft_accuracies.json").as_posix()
