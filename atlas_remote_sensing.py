"""Learn the coefficients on task vectors for remote sensing datasets
under the few-shot setting and find the optimal combination.

Adapted from atlas.py for remote sensing datasets.
Uses get_remote_sensing_dataset() instead of get_dataset().
"""

import os
import argparse
import sys
import time
import json
import torch
import torchvision
import logging
import subprocess
from omegaconf import OmegaConf

from torch.cuda.amp import GradScaler
from atlas_src.modeling import ImageEncoder, ImageClassifier
from atlas_src.composition import WeightedImageEncoder

# Task vectors from energy environment
from src.models.task_vectors import NonLinearTaskVector
from src.utils.variables_and_paths import (
    get_zeroshot_path,
    get_finetuned_path,
)

# Utils
from src.utils.utils import cosine_lr
from src.datasets.common import get_dataloader, maybe_dictionarize

# Remote sensing specific imports
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
    REMOTE_SENSING_DATASETS,
    sample_k_shot_indices,
)

# Evaluation function
from src.eval.eval_remote_sensing_comparison import evaluate_encoder_with_dataloader

# Dataset-specific epochs for Atlas training (matching fine-tuning epochs)
ATLAS_EPOCHS_PER_DATASET = {
    "AID": 30,              # ~10,000 train samples, 600x600
    "CLRS": 10,             # ~30,000 train samples, 256x256
    "EuroSAT_RGB": 15,      # ~21,600 train samples, 64x64
    "MLRSNet": 15,          # ~17,000 train samples, 256x256
    "NWPU-RESISC45": 15,    # ~25,200 train samples, 256x256
    "Optimal-31": 50,       # ~6,200 train samples, 256x256
    "PatternNet": 20,       # ~10,000 train samples, 256x256
    "RS_C11": 60,           # ~5,000 train samples, 512x512
    "RSD46-WHU": 20,        # ~10,000 train samples, 256x256
    "RSI-CB128": 15,        # ~18,000 train samples, 128x128
    "RSSCN7": 80,           # ~2,800 train samples, 400x400
    "SAT-4": 5,             # ~60,000 train samples, 28x28
    "SAT-6": 10,            # ~40,000 train samples, 28x28
    "SIRI-WHU": 100,        # ~2,400 train samples, 200x200
    "UC_Merced": 100,       # ~2,100 train samples, 256x256
    "WHU-RS19": 150,        # ~1,000 train samples, 600x600
}


def compute_eval_epochs(total_epochs: int, max_evals: int = 5) -> set:
    total_epochs = max(int(total_epochs), 1)
    if total_epochs <= max_evals:
        return set(range(total_epochs))
    eval_epochs = {
        min(total_epochs - 1, int(round(i * (total_epochs - 1) / (max_evals - 1))))
        for i in range(max_evals)
    }
    return eval_epochs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_single(args):
    """Entry point for single-GPU atlas training."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)

    if not hasattr(args, 'model_location') or args.model_location is None:
        args.model_location = os.path.expanduser("./models/checkpoints_remote_sensing")
    if not hasattr(args, 'save_dir') or args.save_dir is None:
        args.save_dir = os.path.join(args.model_location, args.model)

    test_ds = getattr(args, 'test_dataset', None)
    if hasattr(args, 'basis_datasets') and args.basis_datasets:
        pool = list(args.basis_datasets)
        logger.info(f"Using {len(pool)} explicitly specified basis datasets")
    elif test_ds and test_ds in REMOTE_SENSING_DATASETS:
        pool = [d for d in REMOTE_SENSING_DATASETS.keys() if d != test_ds]
        logger.info(f"Leave-one-out mode: using {len(pool)} datasets as basis (excluding {test_ds})")
    else:
        pool = list(REMOTE_SENSING_DATASETS.keys())
        logger.info(f"Using all {len(pool)} remote sensing datasets as basis")

    args.basis_datasets_list = pool
    comp_acc = {}

    if hasattr(args, 'target_datasets') and args.target_datasets:
        target_list = list(args.target_datasets.items())
    elif test_ds:
        default_epochs = getattr(args, 'epochs_per_task', 10)
        target_list = [(test_ds, default_epochs)]
    else:
        logger.error("No target dataset specified. Use --test_dataset or --datasets")
        return

    logger.info(f"Target datasets: {[d for d, _ in target_list]}")

    for dataset, epochs in target_list:
        args.target_dataset = dataset + "Val"
        if dataset in ATLAS_EPOCHS_PER_DATASET:
            args.epochs = ATLAS_EPOCHS_PER_DATASET[dataset]
            logger.info(f"✓ Auto-set epochs={args.epochs} for {dataset} (dataset-specific)")
        else:
            args.epochs = epochs
            logger.info(f"Using default epochs={args.epochs} for {dataset}")

        zs_json_path = os.path.join(args.save_dir, f"{dataset}Val", "zeroshot_accuracies.json")
        if os.path.isfile(zs_json_path):
            with open(zs_json_path, 'r') as f:
                args.zs_acc = json.load(f)
            comp_acc[f"{dataset}Val_zeroshot"] = args.zs_acc.get(f"{dataset}Val", 0.0)
        else:
            if not hasattr(args, 'zs_acc'):
                args.zs_acc = {}

        k = getattr(args, 'k', 0)
        data_amount = f"{k} shots per class" if k > 0 else "full dataset"
        logger.info("=" * 100)
        logger.info(f"Learning task vector coefficients on {dataset} with {args.model} - {data_amount}")
        logger.info("=" * 100)

        comp_acc = train_single_task(args, comp_acc, logger)


def train_single_task(args, comp_acc=None, logger=None):
    """Train atlas coefficients on remote sensing dataset"""

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

    if comp_acc is None:
        comp_acc = {}
    
    target_dataset = args.target_dataset

    logger.info(f"Loading task vectors for {len(args.basis_datasets_list)} basis datasets...")

    task_vectors = {}
    for dataset in args.basis_datasets_list:
        # Use Val suffix for consistency with fine-tuning
        dataset_val = dataset + "Val"
        pretrained_checkpoint_path = get_zeroshot_path(
            args.model_location, dataset_val, args.model)
        finetuned_checkpoint_path = get_finetuned_path(
            args.model_location, dataset_val, args.model)
        
        if os.path.exists(pretrained_checkpoint_path) and os.path.exists(finetuned_checkpoint_path):
            pretrained_state = torch.load(pretrained_checkpoint_path, map_location="cpu")
            finetuned_state = torch.load(finetuned_checkpoint_path, map_location="cpu")

            task_vectors[dataset] = NonLinearTaskVector(
                args.model, pretrained_state, finetuned_state)

            logger.info(f"✓ Loaded task vector for {dataset}")
        else:
            logger.warning(f"✗ Missing checkpoints for {dataset}")

    if not task_vectors:
        logger.error("No task vectors loaded; aborting.")
        return comp_acc or {}

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task (leave-one-out)
    available_task_vectors = [
        v for k, v in task_vectors.items() if orig_dataset != k]
    
    logger.info(f"Using {len(available_task_vectors)} task vectors for composition")
    logger.info(f"Target dataset: {orig_dataset} (held out)")

    # Create WeightedImageEncoder with task vectors
    image_encoder = ImageEncoder(args)
    image_encoder = WeightedImageEncoder(
        image_encoder, available_task_vectors, 
        blockwise=args.blockwise_coef, 
        partition=args.partition,
    )

    # TIP's more aggressive random crop with horizontal flip
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), 
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + image_encoder.train_preprocess.transforms[-3:])

    # Load remote sensing dataset
    train_dataset = get_remote_sensing_dataset(
        target_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Get classification head for remote sensing dataset
    classification_head = get_remote_sensing_classification_head(args, target_dataset, train_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    eval_dataset = get_remote_sensing_dataset(
        target_dataset,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=8,
    )

    # Few-shot sampling using unified k-shot function
    k = getattr(args, 'k', 0)
    logger.info("Using full training dataset (no k-shot sampling)" if k <= 0 else f"Applying k-shot sampling: {k} samples per class")
    if k > 0:
        selected_indices = sample_k_shot_indices(train_dataset, k, seed=0, verbose=True)
        base_dataset = getattr(train_dataset, "train_dataset", train_dataset)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_dataset, selected_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
        )
    else:
        train_loader = train_dataset.train_loader

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 80)
    logger.info(f"Trainable parameters (atlas): {trainable_params:,}")
    logger.info(f"Number of task vectors: {len(available_task_vectors)}")
    logger.info("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_batches = len(train_loader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()
    
    # Load validation dataset using get_dataloader (unified evaluation approach)
    val_loader = get_dataloader(eval_dataset, is_train=False, args=args, image_encoder=None)
    
    # Evaluate zeroshot accuracy using unified evaluation function
    if f"{target_dataset}_zeroshot" not in comp_acc.keys():
        logger.info(f"Evaluating zero-shot accuracy on {target_dataset}...")

        image_encoder.eval()
        classification_head.eval()

        zeroshot_metrics = evaluate_encoder_with_dataloader(
            image_encoder, classification_head, val_loader, 'cuda')
        zeroshot_acc = zeroshot_metrics['top1']

        comp_acc[f"{target_dataset}_zeroshot"] = zeroshot_acc
        args.zs_acc[f"{target_dataset}"] = zeroshot_acc

        image_encoder.train()
        classification_head.train()

    logger.info(
        f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%.")

    best_coef = model.image_encoder.coef.data.clone()
    best_acc = args.zs_acc[target_dataset]
    eval_epochs = compute_eval_epochs(args.epochs)
    loss_history = []
    train_start = time.time()

    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inputs)
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

                global_step = epoch * num_batches + i
                loss_history.append(
                    {
                        "epoch": int(epoch),
                        "iteration": int(i),
                        "global_step": int(global_step),
                        "loss": float(loss.item()),
                    }
                )

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
            ):
                percent_complete = 100 * (i + 1) / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )

        # Evaluate after selected epochs using unified evaluation function
        if epoch in eval_epochs:
            image_encoder = model.image_encoder
            coef = model.image_encoder.coef
            
            image_encoder.eval()
            classification_head.eval()
            
            # Use unified evaluation function
            metrics = evaluate_encoder_with_dataloader(
                image_encoder, classification_head, val_loader, 'cuda')
            acc = metrics['top1']
            
            # Set back to train mode
            image_encoder.train()
            classification_head.train()
            
            logger.info(f"Epoch {epoch}: Accuracy = {100*acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()
                logger.info(f"✓ New best accuracy: {100*best_acc:.2f}%")

    training_time = time.time() - train_start
    comp_acc[target_dataset] = best_acc
    target_dataset_clean = target_dataset.replace("Val", "")
    image_encoder = model.image_encoder
    image_encoder.coef = torch.nn.Parameter(best_coef)

    # Final evaluation using unified evaluation function
    image_encoder.eval()
    classification_head.eval()

    final_metrics = evaluate_encoder_with_dataloader(
        image_encoder, classification_head, val_loader, 'cuda')
    final_acc = final_metrics['top1']

    comp_acc[target_dataset_clean] = final_acc

    logger.info(f"Final test accuracy: {100*final_acc:.2f}%")

    # Save coefficients to dataset-specific directory
    k = getattr(args, 'k', 0)
    shot_folder = f"{k}shot" if k > 0 else "fullshot"

    save_dir = os.path.join(
        args.model_location,
        args.model,
        target_dataset,
        shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)

    atlas_path = os.path.join(save_dir, "atlas.pt")
    torch.save(best_coef, atlas_path)
    logger.info(f"✓ Saved learned atlas coefficients to {atlas_path}")

    log_path = os.path.join(save_dir, "atlas_results.json")
    gpu_peak_mem_mb = None
    if torch.cuda.is_available():
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak GPU memory during training: {gpu_peak_mem_mb:.2f} MB")
    result_log = {
        "target_dataset": target_dataset_clean,
        "final_accuracy": final_acc,
        "best_val_accuracy": best_acc,
        "k_shot": k,
        "model": args.model,
        "epochs": args.epochs,
        "training_time": training_time,
        "trainable_params": trainable_params,
        "batch_size": args.batch_size,
        "gpu_peak_mem_mb": gpu_peak_mem_mb,
        "loss_history": loss_history,
    }
    with open(log_path, 'w') as f:
        json.dump(result_log, f, indent=4)
    logger.info(f"✓ Saved results to {log_path}")

    return comp_acc


if __name__ == "__main__":
    
    # Load config first to get default values
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_remote_sensing.yaml")
    config = OmegaConf.load(config_path)
    
    # Get default model from config
    default_model = config.get("model", "ViT-B-32")

    # Lightweight argparse wrapper to override key hyperparameters via CLI
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--model", type=str, default=default_model,
                         help=f"Model architecture (default: {default_model} from config)")
    wrapper.add_argument("--batch_size", type=int, default=None)
    wrapper.add_argument("--lr", type=float, default=None)
    wrapper.add_argument("--wd", type=float, default=None)
    wrapper.add_argument("--epochs", type=int, default=None)
    wrapper.add_argument("--k", type=int, default=0,
                         help="k-shot per class (e.g., 16 = 16 samples per class). "
                              "Set to 0 to use full dataset (default: 16)")
    wrapper.add_argument("--data_location", type=str, default="./datasets")
    wrapper.add_argument("--model_location", type=str, default="./models/checkpoints_remote_sensing")
    wrapper.add_argument("--seed", type=int, default=1)
    wrapper.add_argument("--print_every", type=int, default=10)
    
    # Dataset controls (similar to energy_train_remote_sensing.py)
    wrapper.add_argument("--test_dataset", type=str, default=None,
                         help="Single dataset to train on (leave-one-out with others as basis)")
    wrapper.add_argument("--datasets", type=str, default=None,
                         help="Comma-separated dataset names without 'Val' (alternative to --test_dataset)")
    wrapper.add_argument("--epochs_per_task", type=int, default=10)
    wrapper.add_argument("--run_all", action="store_true",
                         help="Run atlas for all datasets in DATASETS_ALL (leave-one-out)")
    
    # Atlas-specific options
    wrapper.add_argument("--blockwise_coef", action="store_true", default=True,
                         help="Learn coefficient per parameter block")
    wrapper.add_argument("--partition", type=int, default=None,
                         help="Partition size for fine-grained coefficient learning")

    cli_args, unknown = wrapper.parse_known_args()

    # Import atlas args parser for remaining arguments
    from atlas_src.args import parse_arguments
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
    if cli_args.data_location is not None:
        args.data_location = cli_args.data_location
    if cli_args.model_location is not None:
        args.model_location = cli_args.model_location
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.print_every is not None:
        args.print_every = cli_args.print_every
    if cli_args.blockwise_coef is not None:
        args.blockwise_coef = cli_args.blockwise_coef
    if cli_args.partition is not None:
        args.partition = cli_args.partition
    
    # Set k-shot parameter
    if cli_args.k is not None:
        args.k = cli_args.k
    else:
        args.k = 16  # default: 16-shot

    # Default hyperparameters (following atlas.py)
    args.lr = 1e-1
    args.batch_size = 32 if args.model == "ViT-L-14" else 512
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.epochs_per_task = cli_args.epochs_per_task
    
    # If --run_all is specified, run for each dataset in DATASETS_ALL sequentially
    if cli_args.run_all:
        if "DATASETS_ALL" not in config or not config.DATASETS_ALL:
            print("ERROR: DATASETS_ALL not found or empty in config_remote_sensing.yaml")
            sys.exit(1)
        
        all_datasets = list(config.DATASETS_ALL)
        total = len(all_datasets)
        
        print("\n" + "=" * 100)
        print(f"🚀 RUNNING ATLAS FOR ALL {total} DATASETS (Leave-One-Out)")
        print("=" * 100)
        print(f"Datasets: {', '.join(all_datasets)}")
        k_str = f"{args.k}-shot per class" if args.k > 0 else "full dataset"
        print(f"Config: model={args.model}, k={k_str}, epochs_per_task={args.epochs_per_task}")
        print("=" * 100 + "\n")
        
        failed_datasets = []
        
        for idx, test_dataset in enumerate(all_datasets, 1):
            # Get dataset-specific epochs
            dataset_epochs = ATLAS_EPOCHS_PER_DATASET.get(test_dataset, args.epochs_per_task)
            
            print("\n" + "=" * 100)
            print(f"📊 [{idx}/{total}] Processing: {test_dataset} as TARGET dataset")
            print(f"    Basis datasets: {total - 1} datasets (all except {test_dataset})")
            print(f"    Training epochs: {dataset_epochs} (auto-determined based on dataset size)")
            print("=" * 100 + "\n")
            
            # Build command to run this script with specific test_dataset
            cmd = [
                sys.executable,  # Python interpreter
                __file__,        # This script
                "--test_dataset", test_dataset,
                "--model", args.model,
                "--k", str(args.k),
                "--epochs_per_task", str(args.epochs_per_task),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--wd", str(args.wd if hasattr(args, 'wd') else 0.0),
                "--data_location", args.data_location,
                "--model_location", args.model_location,
                "--seed", str(args.seed),
            ] + unknown  # Pass through any remaining args
            
            if cli_args.blockwise_coef:
                cmd.append("--blockwise_coef")
            if cli_args.partition is not None:
                cmd.extend(["--partition", str(cli_args.partition)])
            
            print(f"🔧 Running command: {' '.join(cmd)}\n")
            
            # Explicitly pass environment variables to subprocess (includes CUDA_VISIBLE_DEVICES)
            env = os.environ.copy()
            
            try:
                result = subprocess.run(cmd, check=True, env=env)
                print(f"\n✅ [{idx}/{total}] Successfully completed: {test_dataset}")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ [{idx}/{total}] FAILED: {test_dataset} (exit code: {e.returncode})")
                failed_datasets.append(test_dataset)
            except Exception as e:
                print(f"\n❌ [{idx}/{total}] ERROR: {test_dataset} - {str(e)}")
                failed_datasets.append(test_dataset)
            
            print("=" * 100)
        
        # Final summary
        print("\n" + "=" * 100)
        print("🏁 ATLAS TRAINING COMPLETED FOR ALL DATASETS")
        print("=" * 100)
        print(f"Total datasets: {total}")
        print(f"Successful: {total - len(failed_datasets)}")
        print(f"Failed: {len(failed_datasets)}")
        if failed_datasets:
            print(f"Failed datasets: {', '.join(failed_datasets)}")
        print("=" * 100 + "\n")
        
        sys.exit(0 if len(failed_datasets) == 0 else 1)
    
    else:
        # Single dataset mode or explicit datasets list
        if cli_args.test_dataset:
            # Single test dataset mode (leave-one-out)
            args.test_dataset = cli_args.test_dataset
            args.basis_datasets = [d for d in config.DATASETS_ALL if d != cli_args.test_dataset]
            print(f"Leave-one-out mode: Test dataset = {args.test_dataset}")
            print(f"Basis datasets: {len(args.basis_datasets)} datasets")
        elif cli_args.datasets is not None and len(cli_args.datasets.strip()) > 0:
            # Multiple datasets specified explicitly
            ds_list = [d.strip() for d in cli_args.datasets.split(",") if len(d.strip()) > 0]
            target_datasets = {ds: args.epochs_per_task for ds in ds_list}
            args.target_datasets = target_datasets
            print(f"Training on {len(ds_list)} specified datasets")
        else:
            # Default: use a subset or all from config
            print("No --test_dataset or --datasets specified. Using default subset.")
            ds_list = ["EuroSAT_RGB", "AID", "CLRS"]  # Small default subset for testing
            target_datasets = {ds: args.epochs_per_task for ds in ds_list}
            args.target_datasets = target_datasets

        # Setup logging directory (legacy - results now saved per-dataset)
        if not hasattr(args, 'logdir'):
            args.logdir = os.path.join("logs", "atlas_remote_sensing", args.model)
        else:
            args.logdir += f"/{args.model}"
        
        if args.k > 0:
            args.logdir += f"/{args.k}shots"
        else:
            args.logdir += "/fullshot"

        if args.seed is not None:
            args.logdir += f"/{args.seed}"

        # Legacy paths (kept for compatibility, actual results saved per-dataset)
        args.head_path = os.path.join(args.logdir, "learned_composition_remote_sensing.pt")
        args.log_path = os.path.join(args.logdir, "learned_composition_remote_sensing.json")

        os.makedirs(args.logdir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.logdir, "atlas_remote_sensing.log")),
            ]
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger(__name__).addHandler(console_handler)

        run_single(args)
