import os
import json
from pprint import pprint

from src.eval.eval import eval_single_dataset
from src.models.task_vectors import NonLinearTaskVector
from src.utils.args import parse_arguments
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

import torch

args = parse_arguments()
args.save_dir = os.path.join(args.model_location, args.model)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print("*" * 100)

pprint(args.__dict__, width=1)

accuracies = {}

PERCENTAGES = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    # 0.06, ViT-L
    0.07,
    # 0.08, ViT-L
    # 0.09, ViT-L
    0.1,
    # 0.125, ViT-L
    0.15,
    0.2,
    0.3,
    0.5,
]

DATASETS = [
    "MNIST",
    # "Cars",
    # "DTD",
    # "EuroSAT",
    "GTSRB",
    "RESISC45",
    # "SUN397",
    "SVHN",
    "PCAM",
    "CIFAR100",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "FER2013",
    "CIFAR10",
    "Food101",
    "RenderedSST2",
    "EMNIST",
    "FashionMNIST",
    "KMNIST",
]

# load pretrained checkpoint
pretrained_checkpoint = get_zeroshot_path(
    args.model_location, "MNIST", args.model)

# evaluate each task sequentially
for dataset in DATASETS:

    print("\n" * 3)
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    # load finetuned checkpoint
    finetuned_checkpoint = get_finetuned_path(
        args.model_location, dataset, args.model)

    task_vector = NonLinearTaskVector(
        args.model, pretrained_checkpoint, finetuned_checkpoint
    )

    directory = f"results/single_task_rank/{args.model}/{dataset}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_file = f"single_task_ft_accuracies_rank.json"
    save_path = os.path.join(directory, save_file)

    # Load existing data from file
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            result_dict = json.load(f)
    else:
        result_dict = {}

    for sv_reduction in PERCENTAGES:  # np.linspace(0.0, 0.10, 11)[1:]:
        print(f"\nsv_reduction percentage: {sv_reduction}\n")
        new_vector = {}
        accuracies = {}
        for key in task_vector.vector:
            new_vector[key] = {}
            # for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u, s, v = torch.linalg.svd(
                    task_vector.vector[key], full_matrices=False)

                # print(f"Computed SVD for {key}...")
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
                reduced_index_s = int(s.shape[0] * sv_reduction)

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, 0:reduced_index_s] = u[:, :reduced_index_s]
                sum_s[0:reduced_index_s] = s[:reduced_index_s]
                # select only the first reduced_index_s rows of v and place them
                sum_v[0:reduced_index_s, :] = v[:reduced_index_s, :]

                new_vector[key] = torch.linalg.multi_dot(
                    (sum_u, torch.diag(sum_s), sum_v)
                )
            else:
                new_vector[key] = task_vector.vector[key]

        new_task_vector = NonLinearTaskVector(args.model, vector=new_vector)

        image_encoder = new_task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=1.0
        )

        for split in ["val", "test"]:
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            if eval_dataset not in accuracies:
                accuracies[eval_dataset] = {}
            accuracies[eval_dataset] = eval_single_dataset(
                image_encoder, eval_dataset, args
            )["top1"]
            print()
        result_dict[sv_reduction] = accuracies

    with open(save_path, "w") as f:
        f.write(json.dumps(result_dict, sort_keys=False, indent=4) + "\n")

    pprint(accuracies, width=1)
    print("File saved at: ", save_path)
