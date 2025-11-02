"""Learn the coefficients on task vectors
under the few-shot setting for a dataset
and find the optimal combination.

Paul Albert <paul.albert@adelaide.edu.au>
Fred Zhang <frederic.zhang@adelaide.edu.au>

Australian Institute for Machine Learning
"""

from warnings import filterwarnings
import torchvision
import torch
import json
import time
import sys
import argparse
import os
from torch.cuda.amp import GradScaler
from atlas_src.modeling import ImageEncoder, ImageClassifier
from atlas_src.composition import WeightedImageEncoder
from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import (
    ALL_DATASETS,
    get_zeroshot_path,
    get_finetuned_path,
)
from src.utils.utils import cosine_lr
from atlas_src.utils import get_n_shots, TIPWrapper, LPPWrapper, IndexWrapper, _RepeatSampler
from atlas_src.args import parse_arguments
from src.eval.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.models.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
import random
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# use task vectors and paths from energy (src.*) environment

# utils: mix atlas wrappers with energy scheduler utils


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def is_main_process():
    return True


def main(rank, args):
    # Harmonize save/model locations with energy environment
    if not hasattr(args, 'model_location') or args.model_location is None:
        base_root = args.save if args.save is not None else os.path.expanduser(
            "/disk3/junghwan/task_vector/models/checkpoints")
        args.model_location = base_root
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)

    # Load the individual task vectors.
    # Use datasets defined in energy environment
    pool = list(ALL_DATASETS)
    task_vectors = {}
    for dataset in pool:
        # Resolve checkpoints using energy path helpers
        pretrained_checkpoint = get_zeroshot_path(
            args.model_location, "MNISTVal", args.model)
        finetuned_checkpoint = get_finetuned_path(
            args.model_location, dataset, args.model)
        task_vectors[dataset] = NonLinearTaskVector(
            args.model, pretrained_checkpoint, finetuned_checkpoint)

    args.rank = rank
    # Set deterministic seed for reproducibility (match energy style)
    if hasattr(args, 'seed') and args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if os.path.exists(args.log_path):
        with open(args.log_path, 'r') as f:
            comp_acc = json.load(f)
    else:
        comp_acc = {}

    for dataset, epochs in args.target_datasets.items():
        args.target_dataset = dataset + "Val"
        args.epochs = epochs
        zs_json_path = os.path.join(
            args.save_dir, f"{dataset}Val", "zeroshot_accuracies.json")
        # if os.path.isfile(zs_json_path):
        #     with open(zs_json_path, 'r') as f:
        #         args.zs_acc = json.load(f)
        #     comp_acc[f"{dataset}Val_zeroshot"] = args.zs_acc[f"{dataset}Val"]
        # else:
        #     if not hasattr(args, 'zs_acc'):
        args.zs_acc = {}

        if type(args.subsample) == float:
            data_amount = f"{args.subsample*100}%"
        else:
            data_amount = f"{args.subsample} shots"

        print("=" * 100)
        print(
            f"Learning task vector coefficients on {dataset} with {args.model} - {data_amount}")
        print("=" * 100)

        comp_acc = train(task_vectors, args, comp_acc)


def train(task_vectors, args, comp_acc={}):
    target_dataset = args.target_dataset

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, task_vectors, blockwise=args.blockwise_coef, partition=args.partition,
    )

    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    # TIP's more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + model.train_preprocess.transforms[-3:])

    dataset = get_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    if type(args.subsample) == int:
        int_idx_path = os.path.join(
            args.save_dir, target_dataset, f"{args.subsample}_shots_{args.seed}.pt")
        if os.path.isfile(int_idx_path) and args.seed == 1:
            to_keep = torch.load(int_idx_path)
        else:
            to_keep = get_n_shots(
                dataset.train_dataset, args.subsample, classification_head.out_features, args)
            os.makedirs(os.path.dirname(int_idx_path), exist_ok=True)
            torch.save(to_keep, int_idx_path)

        r = len(to_keep) / args.batch_size
        if r < 10:
            over_sampling = 10/r
            over_sampling = int(over_sampling) + 1
            print(f"Oversampling {over_sampling} times")
            to_keep = torch.cat([to_keep] * over_sampling)

    else:
        float_idx_path = os.path.join(
            args.save_dir, target_dataset, f"{args.subsample}_{args.seed}.pt")
        if os.path.isfile(float_idx_path) and args.seed == 1:
            to_keep = torch.load(float_idx_path)
        else:
            dataset_index = torch.arange(len(dataset.train_dataset))
            to_keep = torch.randperm(len(dataset_index))[
                : int(len(dataset_index) * args.subsample)]
            os.makedirs(os.path.dirname(float_idx_path), exist_ok=True)
            torch.save(to_keep, float_idx_path)

    index_dataset = IndexWrapper(dataset.train_dataset)
    sampler = torch.utils.data.SubsetRandomSampler(to_keep)
    data_loader = torch.utils.data.DataLoader(
        index_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)

    # Single-process training
    ddp_loader = data_loader
    ddp_model = model

    num_batches = len(ddp_loader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()
    if is_main_process():
        # Always recompute zero-shot to match energy behavior
        comp_acc[f"{target_dataset}_zeroshot"] = eval_single_dataset(
            image_encoder, target_dataset, args)["top1"]
        zs_json_path = os.path.join(
            args.save_dir, f"{target_dataset}", "zeroshot_accuracies.json")
        os.makedirs(os.path.dirname(zs_json_path), exist_ok=True)
        with open(zs_json_path, 'w') as f:
            json.dump(
                {f"{target_dataset}": comp_acc[f"{target_dataset}_zeroshot"]}, f, indent=4)
        args.zs_acc[f"{target_dataset}"] = comp_acc[f"{target_dataset}_zeroshot"]

        print(
            f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%.")

    best_coef = ddp_model.image_encoder.coef.data.clone()
    best_acc = args.zs_acc[target_dataset]
    atlas_start_time = time.time()
    print(f"batch size: {args.batch_size}")
    print(f"data_loader length: {len(ddp_loader)}")
    for epoch in range(args.epochs):
        start_time = time.time()
        for i, batch in enumerate(ddp_loader):

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            print(f"inputs shape: {inputs.shape}")
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"           # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}",   # noqa: E501
                    flush=True,
                )

        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.3f} seconds")
        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.image_encoder
            coef = ddp_model.image_encoder.coef
            acc = eval_single_dataset(
                image_encoder, target_dataset, args)["top1"]
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()

    atlas_end_time = time.time()
    print(f"Atlas time: {atlas_end_time - atlas_start_time:.2f} seconds")
    if is_main_process():
        comp_acc[target_dataset] = best_acc
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.image_encoder
        image_encoder.coef = torch.nn.Parameter(best_coef)
        comp_acc[target_dataset] = eval_single_dataset(
            image_encoder, target_dataset, args)["top1"]
        with open(args.log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)
        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path)
        else:
            heads = {}

        heads[target_dataset] = best_coef
        torch.save(heads, args.head_path)

    if args.adapter is not None:
        comp_acc = train_adapter(
            ddp_model, ddp_loader, args, comp_acc, which=args.adapter)
    return comp_acc


def train_adapter(ddp_model, ddp_loader, args, comp_acc, which='lpp'):
    # Extracting features:
    all_features, all_labels, all_indexes, all_logits = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(ddp_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()

            logits, features = ddp_model(inputs, return_features=True)
            labels = batch["labels"]

            all_features.append(features.detach().cpu())
            all_labels.append(labels)
            all_indexes.append(batch["index"])
            all_logits.append(logits.detach().cpu())

    logits_cache = torch.cat(all_logits)
    features_cache = torch.cat(all_features)
    labels = torch.cat(all_labels)
    indexes = torch.cat(all_indexes)
    indexes_to_i = {indexes[i].item(): i for i in range(len(indexes))}

    model = ddp_model
    if which == 'lpp':
        if type(args.subsample) == float:
            shots = 100
        else:
            shots = args.subsample
        model = LPPWrapper(model, features_cache, labels, shots)
        epochs = 300
        lr = model.lr_temp
    elif which == 'tip':
        model = TIPWrapper(model, features_cache, labels)
        lr = 1e-3
        epochs = 10
    else:
        raise NotImplementedError(f"Adapter {which} unknown")

    model = model.cuda()
    ddp_model = model

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    num_batches = len(ddp_loader)
    scheduler = cosine_lr(
        optimizer, lr, 0,
        epochs * num_batches // args.num_grad_accumulation,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    print_every = 100
    ddp_loader._DataLoader__initialized = False
    ddp_loader.batch_sampler = _RepeatSampler(ddp_loader.batch_sampler, epochs)
    ddp_loader._DataLoader__initialized = True

    for i, batch in enumerate(ddp_loader):
        start_time = time.time()
        epoch = i // num_batches
        step = (
            i // args.num_grad_accumulation
            + epoch * num_batches // args.num_grad_accumulation
        )

        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        data_time = time.time() - start_time

        ids = [indexes_to_i[i.item()] for i in batch['index']]
        l_cache, f_cache = logits_cache[ids].to(
            inputs), features_cache[ids].to(inputs)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = ddp_model(inputs, l_cache, f_cache)
            labels = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels)
            loss = loss / args.num_grad_accumulation

        if (i + 1) % args.num_grad_accumulation == 0:
            scheduler(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        batch_time = time.time() - start_time

        if (
            step % print_every == 0
            and ((i + 1) % args.num_grad_accumulation == 0)
            and is_main_process()
        ):
            percent_complete = 100 * (i + 1) / len(ddp_loader)
            print(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(ddp_loader)}]\t"           # noqa: E501
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                flush=True,
            )

    if is_main_process():
        # comp_acc[target_dataset+f"_{which}"] = best_acc
        target_dataset = args.target_dataset
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.model.image_encoder
        comp_acc[target_dataset+f"_{which}"] = eval_single_dataset(
            image_encoder, target_dataset, args, model=ddp_model)["top1"]
        with open(args.log_path, 'w') as f:
            json.dump(comp_acc, f, indent=4)

        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path)
        else:
            heads = {}

        adapter_coefs = {
            k: v for k, v in ddp_model.state_dict().items() if v.requires_grad}
        heads[target_dataset] = adapter_coefs
        torch.save(heads, args.head_path)

    return comp_acc


if __name__ == "__main__":

    # Lightweight argparse wrapper to override key hyperparameters via CLI
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--model", type=str, default=None)
    wrapper.add_argument("--batch_size", type=int, default=256)
    wrapper.add_argument("--lr", type=float, default=None)
    wrapper.add_argument("--wd", type=float, default=None)
    wrapper.add_argument("--epochs", type=int, default=None)
    wrapper.add_argument("--subsample", type=str, default=None)
    wrapper.add_argument("--world_size", type=int, default=None)
    wrapper.add_argument("--port", type=int, default=None)
    wrapper.add_argument("--data_location", type=str,
                         default='/home/junghwan/task_singular_vectors/datasets')
    wrapper.add_argument("--model_location", type=str, default=None)
    wrapper.add_argument("--seed", type=int, default=42)
    wrapper.add_argument("--print_every", type=int, default=None)
    # dataset controls
    wrapper.add_argument("--datasets", type=str, default='CIFAR10',
                         help="Comma-separated dataset names without 'Val'")
    wrapper.add_argument("--epochs_per_task", type=int, default=10)
    # adapter controls
    wrapper.add_argument(
        "--adapter",
        type=str,
        choices=["lpp", "tip"],
        default=None,
        help="Enable adapter training: choose 'lpp' or 'tip' (default: disabled)",
    )

    cli_args, unknown = wrapper.parse_known_args()
    # leave unknowns for atlas_src.args.parse_arguments
    sys.argv = [sys.argv[0]] + unknown

    args = parse_arguments()

    # Overlay CLI overrides
    if cli_args.model is not None:
        args.model = cli_args.model
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    if cli_args.lr is not None:
        args.lr = cli_args.lr
    if cli_args.wd is not None:
        args.wd = cli_args.wd
    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.world_size is not None:
        args.world_size = cli_args.world_size
    if cli_args.port is not None:
        args.port = cli_args.port
    if cli_args.data_location is not None:
        args.data_location = cli_args.data_location
    # Align data path handling with energy_train.py: default to "datasets" and expanduser
    try:
        if cli_args.data_location is None:
            args.data_location = os.path.expanduser("datasets")
        else:
            args.data_location = os.path.expanduser(args.data_location)
    except Exception:
        pass
    if cli_args.model_location is not None:
        args.model_location = cli_args.model_location
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.print_every is not None:
        args.print_every = cli_args.print_every
    if hasattr(cli_args, "adapter") and cli_args.adapter is not None:
        args.adapter = cli_args.adapter
    if cli_args.subsample is not None:
        # allow int or float
        try:
            if "." in cli_args.subsample:
                args.subsample = float(cli_args.subsample)
            else:
                args.subsample = int(cli_args.subsample)
        except Exception:
            pass

    # Build target_datasets from datasets option or ALL_DATASETS
    if cli_args.datasets is not None and len(cli_args.datasets.strip()) > 0:
        ds_list = [d.strip()
                   for d in cli_args.datasets.split(",") if len(d.strip()) > 0]
    else:
        ds_list = list(ALL_DATASETS)
    # Use --epochs if provided, otherwise use --epochs_per_task
    epochs_to_use = cli_args.epochs if cli_args.epochs is not None else (
        cli_args.epochs_per_task if cli_args.epochs_per_task is not None else 10)
    target_datasets = {ds: epochs_to_use for ds in ds_list}
    args.target_datasets = target_datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    args.lr = 1e-1
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    # args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    args.logdir += f"{args.model}"
    if type(args.subsample) == float:
        args.logdir += f"/{args.subsample*100:.0f}perc"
    else:
        args.logdir += f"/{args.subsample}shots"
        # few-shot: keep epochs per task as given

    # model_location/save_dir harmonization is handled in main(); keep args.save as-is
    if args.seed is not None:
        args.logdir += f"/{args.seed}"

    args.head_path = os.path.join(args.logdir, "learned_composition.pt")
    args.log_path = os.path.join(args.logdir, "learned_composition.json")

    os.makedirs(args.logdir, exist_ok=True)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
