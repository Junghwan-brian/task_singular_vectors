"""
Fine-tuning script for Remote Sensing Datasets

Supports both single-label and multi-label classification:
- Single-label: Each image has ONE class label (e.g., "Forest", "Airport")
  Uses CrossEntropyLoss for training
- Multi-label: Each image can have MULTIPLE class labels (e.g., "Residential" + "Vegetation")
  Uses BCEWithLogitsLoss for training

The script automatically handles both types by separating datasets into:
  - train_datasets_single: Standard classification datasets
  - train_datasets_multi: Multi-label classification datasets
"""
import os
import sys
import time
import gc
import json
import copy
from pathlib import Path

import torch
import logging
import re

# Import from existing codebase
from src.datasets import get_dataloader, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder
from src.utils import parse_arguments, setup_logging
from src.utils.distributed import (
    cleanup_ddp,
    distribute_loader,
    is_main_process,
    setup_ddp,
)
from src.utils.utils import LabelSmoothing, cosine_lr
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path

# Import remote sensing specific modules
from src.datasets.remote_sensing import (
    get_remote_sensing_dataset,
    get_remote_sensing_classification_head,
    clean_dataset_logs,
)


def cli_flag_provided(flag: str) -> bool:
    """Return True if a CLI flag like '--batch-size' was explicitly provided."""
    long_form = f"--{flag}"
    return any(arg == long_form or arg.startswith(f"{long_form}=") for arg in sys.argv)


def finetune_remote_sensing(rank, args, is_multilabel=False):
    """
    Fine-tune on a remote sensing dataset
    
    Args:
        rank: GPU rank for distributed training
        args: Training arguments
        is_multilabel: If True, use BCEWithLogitsLoss for multi-label classification
                      If False, use CrossEntropyLoss for single-label classification
    """
    # Record start time for main process
    if rank == 0:
        training_start_time = time.time()
    
    setup_ddp(rank, args.world_size, port=args.port)
    
    train_dataset = args.train_dataset
    
    if is_main_process():
        log_filename = f"finetune_remote_sensing_{args.model}_{'multilabel' if is_multilabel else 'singlelabel'}.log"
        setup_logging(filename=log_filename)
        
        # Clean previous logs for this dataset only
        clean_dataset_logs(log_filename, train_dataset)
        
        print(f"\n{'='*100}")
        print(f"Starting fine-tuning for {train_dataset}")
        print(f"Log file: {log_filename}")
        print(f"{'='*100}\n")
    
    ft_path = get_finetuned_path(args.model_location, train_dataset, args.model)
    zs_path = get_zeroshot_path(args.model_location, train_dataset, args.model)
    
    # Initialize image encoder
    image_encoder = ImageEncoder(args.model)
    
    # Get dataset (without Val suffix for loading)
    dataset_name_no_val = train_dataset.replace("Val", "")
    print(f"\n{'='*100}")
    print(f"Loading dataset: {dataset_name_no_val}")
    print(f"{'='*100}\n")
    
    # Load remote sensing dataset
    dataset = get_remote_sensing_dataset(
        dataset_name_no_val,
        preprocess=image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=6,
        is_multilabel=is_multilabel,  # Pass multi-label flag
    )
    
    # Log total number of training samples for visibility
    try:
        train_size = len(dataset.train_loader.dataset)
    except AttributeError:
        train_size = len(getattr(dataset, "train_dataset", []))
    print(f"Total training samples for {dataset_name_no_val}: {train_size}")
    
    # Get classification head
    classification_head = get_remote_sensing_classification_head(args, train_dataset, dataset)
    model = ImageClassifier(image_encoder, classification_head)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of trainable parameters is {num_params/1e6:.2f}M")
    
    model.freeze_head()
    model = model.cuda()
    
    print_every = 100
    
    # Use the dataset's train_loader directly
    data_loader = dataset.train_loader
    num_batches = len(data_loader)
    
    # Distribute the data and model across the GPUs
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True, output_device=rank
    )
    
    print(f"Hello from process {rank}")
    
    # Set loss function based on label type
    if is_multilabel:
        print("Using BCEWithLogitsLoss for multi-label classification")
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        print("Using CrossEntropyLoss for single-label classification")
        if args.ls > 0:
            loss_fn = LabelSmoothing(args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )
    
    # Saving zero-shot model
    if is_main_process():
        ckpdir = os.path.join(args.save_dir, train_dataset)
        os.makedirs(ckpdir, exist_ok=True)
        model_path = get_zeroshot_path(args.model_location, train_dataset, args.model)
        ddp_model.module.image_encoder.save(model_path)
        print(f"Saved zero-shot model to {model_path}")
    
    print(f"\n{'='*100}")
    print(f"Starting fine-tuning for {args.epochs} epochs")
    print(f"{'='*100}\n")
    
    for epoch in range(args.epochs):
        ddp_model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()
            
            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )
            
            # Handle batch format
            if isinstance(batch, dict):
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()
            elif isinstance(batch, (list, tuple)):
                inputs, labels = batch
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")
            
            data_time = time.time() - start_time
            
            logits = ddp_model(inputs)
            
            # For multi-label, labels should be float (one-hot or multi-hot encoded)
            # For single-label, labels should be long (class indices)
            if is_multilabel:
                # If labels are not already in multi-hot format, convert them
                if labels.dtype == torch.long and len(labels.shape) == 1:
                    # Convert class indices to one-hot encoding
                    num_classes = logits.shape[1]
                    labels_onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
                    labels = labels_onehot
                labels = labels.float()
            
            loss = loss_fn(logits, labels)
            loss.backward()
            
            # Collect loss for epoch average
            epoch_losses.append(loss.item())
            
            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            batch_time = time.time() - start_time
            
            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = get_finetuned_path(
                    args.model_location, train_dataset, args.model
                ).replace(".pt", f"_{step}.pt")
                ddp_model.module.image_encoder.save(model_path)
            
            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )
                logging.getLogger("task_singular_vectors").info(
                    {
                        "dataset": train_dataset,
                        "epoch": epoch,
                        "step": step,
                        "train/loss": loss.item(),
                        "train/data_time": data_time,
                        "train/batch_time": batch_time,
                    }
                )
        
        # Log epoch summary
        if is_main_process() and len(epoch_losses) > 0:
            epoch_time = time.time() - epoch_start_time
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\n{'='*100}")
            print(f"Epoch {epoch}/{args.epochs - 1} completed in {epoch_time:.2f}s")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"{'='*100}\n")
            
            logging.getLogger("task_singular_vectors").info(
                {
                    "dataset": train_dataset,
                    "epoch": epoch,
                    "epoch_summary": True,
                    "train/avg_loss": avg_loss,
                    "train/epoch_time": epoch_time,
                }
            )
    
    # Save final finetuned model (only on main process)
    if is_main_process():
        ft_path = get_finetuned_path(args.model_location, train_dataset, args.model)
        zs_path = get_zeroshot_path(args.model_location, train_dataset, args.model)
        
        image_encoder = ddp_model.module.image_encoder
        image_encoder.save(ft_path)
        print(f"\nSaved finetuned model to {ft_path}")
        
        # Calculate and save training time information
        training_time = time.time() - training_start_time
        training_info = {
            'dataset': train_dataset.replace("Val", ""),
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_grad_accumulation': args.num_grad_accumulation,
            'learning_rate': args.lr,
            'world_size': args.world_size,
            'total_time_seconds': training_time,
            'total_time_minutes': training_time / 60,
            'total_time_hours': training_time / 3600,
            'is_multilabel': is_multilabel
        }
        
        # Save training info as JSON
        info_path = ft_path.replace('.pt', '_training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=4)
        print(f"Saved training info to {info_path}")
        print(f"Total training time: {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)")
        
        # Store return values before cleanup
        return_values = (zs_path, ft_path)
    else:
        return_values = None
    
    # Cleanup DDP for ALL processes (critical for memory management)
    cleanup_ddp()
    
    # Additional cleanup
    del ddp_model
    del ddp_loader
    torch.cuda.empty_cache()
    
    # Return after cleanup (only main process has values)
    if return_values is not None:
        return return_values


if __name__ == "__main__":
    base_args = parse_arguments()
    batch_size_overridden = cli_flag_provided("batch-size")
    
    # ============================================================================
    # Single-Label Datasets (standard image classification)
    # ============================================================================
    train_datasets_single = [
        # "AID",              # ImageFolder
        # "CLRS",             # Parquet: single 'label' column
        # "EuroSAT_RGB",      # Parquet: single 'label' column
        # "MLRSNet",          # ImageFolder
        # "NWPU-RESISC45",    # Parquet: single 'label' column
        # "Optimal-31",       # Parquet: single 'label' column
        # "PatternNet",       # Parquet: single 'label' column
        # "RS_C11",           # Parquet: single 'label' column
        # "RSD46-WHU",        # Parquet: single 'label' column
        "RSI-CB128",        # ImageFolder (45 classes)
        # "RSSCN7",           # ImageFolder
        # "SAT-4",            # Parquet: single 'label' column
        # "SAT-6",            # Parquet: single 'label' column
        # "SIRI-WHU",         # Parquet: single 'label' column
        # "UC_Merced",        # Parquet: single 'label' column
        # "WHU-RS19",         # Parquet: single 'label' column
    ]
    
    # ============================================================================
    # Multi-Label Datasets (images can have multiple labels)
    # ============================================================================
    train_datasets_multi = None
    # train_datasets_multi = [
    #     "MultiScene",       # Parquet: 'label' column with arrays [4, 5, 6, 15, 16, 21, 22]
    #     "Million-AID",      # Parquet: label_1, label_2, label_3 columns (3 labels per image)
    #     "RSI-CB256",        # Parquet: label_1, label_2 columns (2 labels per image)
    # ]
    
    epochs = {
        "AID": 60,              # ~10,000 train samples, 600x600
        "CLRS": 10,             # ~30,000 train samples, 256x256
        "EuroSAT_RGB": 12,      # ~21,600 train samples, 64x64
        "MLRSNet": 15,          # ~17,000 train samples, 256x256
        "NWPU-RESISC45": 15,    # ~25,200 train samples, 256x256
        "Optimal-31": 50,       # ~6,200 train samples, 256x256
        "PatternNet": 20,       # ~10,000 train samples, 256x256
        "RS_C11": 60,           # ~5,000 train samples, 512x512
        "RSD46-WHU": 20,        # ~10,000 train samples, 256x256
        "RSI-CB128": 15,        # ~18,000 train samples, 128x128
        "RSSCN7": 80,           # ~2,800 train samples, 400x400
        "SAT-4": 5,             # ~60,000 train samples, 28x28
        "SAT-6": 10,            # ~32,000 train samples, 28x28
        "SIRI-WHU": 100,        # ~2,400 train samples, 200x200
        "UC_Merced": 100,       # ~2,100 train samples, 256x256
        "WHU-RS19": 150,        # ~1,000 train samples, 600x600
        "MultiScene": 10,       # ~20,000 train samples, 512x512 (multi-label)
        "Million-AID": 2,       # Large dataset, 350x350 (multi-label, 3 labels per image)
        "RSI-CB256": 10,        # ~30,000 train samples, 256x256 (multi-label, 2 labels per image)
    }

    run_all_models = getattr(base_args, "run_all", False)
    models_to_run = (
        ["ViT-B-16", "ViT-B-32", "ViT-L-14"] if run_all_models else [base_args.model]
    )

    overall_results = []

    for model_name in models_to_run:
        args_template = copy.deepcopy(base_args)
        args_template.model = model_name
        args_template.save_dir = os.path.join(
            args_template.model_location, args_template.model
        )
        
        print("\n" + "=" * 100)
        print(f"RUNNING FINE-TUNING FOR MODEL: {model_name}")
        print("=" * 100 + "\n")

        # ----------------------------------------------------------------------
        # Single-label datasets
        # ----------------------------------------------------------------------
        single_trained = 0
        single_skipped = 0

        if train_datasets_single:
            print("\n" + "=" * 100)
            print(
                f"[{model_name}] SINGLE-LABEL DATASETS: {len(train_datasets_single)} datasets"
            )
            print("=" * 100 + "\n")

            for dataset in train_datasets_single:
                args = copy.deepcopy(args_template)
                args.lr = 1e-5
                args.epochs = epochs.get(dataset, 20)
                args.train_dataset = dataset + "Val"
                args.data_location = args_template.data_location
                args.save_dir = os.path.join(args.model_location, args.model)

                ft_path = get_finetuned_path(
                    args.model_location, args.train_dataset, args.model
                )
                if os.path.exists(ft_path):
                    print("\n" + "=" * 100)
                    print(
                        f"[{model_name} | SINGLE-LABEL] SKIPPING {dataset} - Already fine-tuned"
                    )
                    print(f"Checkpoint exists: {ft_path}")

                    info_path = ft_path.replace(".pt", "_training_info.json")
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            info = json.load(f)
                        print(
                            f"Previous training time: {info.get('total_time_minutes', 'N/A'):.2f} minutes"
                        )
                    print("=" * 100 + "\n")
                    single_skipped += 1
                    continue

                args.world_size = torch.cuda.device_count()

                if not batch_size_overridden:
                    args.batch_size = 1024 if args.model == "ViT-L-14" else 2048
                args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

                print("\n" + "=" * 100)
                print(
                    f"[{model_name} | SINGLE-LABEL] Finetuning on {dataset} for {args.epochs} epochs"
                )
                print(
                    f"Batch size: {args.batch_size} | Grad accumulation: {args.num_grad_accumulation}"
                )
                print(f"Using {args.world_size} GPUs with DDP")
                print("=" * 100 + "\n")

                torch.multiprocessing.spawn(
                    finetune_remote_sensing, args=(args, False), nprocs=args.world_size
                )

                single_trained += 1

                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)

                print("\n" + "=" * 100)
                print(
                    f"[{model_name} | SINGLE-LABEL] Completed fine-tuning on {dataset}"
                )
                print("GPU Memory cleaned up")
                print("=" * 100 + "\n")

        # ----------------------------------------------------------------------
        # Multi-label datasets
        # ----------------------------------------------------------------------
        multi_trained = 0
        multi_skipped = 0

        if train_datasets_multi:
            print("\n" + "=" * 100)
            print(
                f"[{model_name}] MULTI-LABEL DATASETS: {len(train_datasets_multi)} datasets"
            )
            print("=" * 100 + "\n")

            for dataset in train_datasets_multi:
                args = copy.deepcopy(args_template)
                args.lr = 1e-5
                args.epochs = epochs.get(dataset, 20)
                args.train_dataset = dataset + "Val"
                args.data_location = args_template.data_location
                args.save_dir = os.path.join(args.model_location, args.model)

                ft_path = get_finetuned_path(
                    args.model_location, args.train_dataset, args.model
                )
                if os.path.exists(ft_path):
                    print("\n" + "=" * 100)
                    print(
                        f"[{model_name} | MULTI-LABEL] SKIPPING {dataset} - Already fine-tuned"
                    )
                    print(f"Checkpoint exists: {ft_path}")

                    info_path = ft_path.replace(".pt", "_training_info.json")
                    if os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            info = json.load(f)
                        print(
                            f"Previous training time: {info.get('total_time_minutes', 'N/A'):.2f} minutes"
                        )
                    print("=" * 100 + "\n")
                    multi_skipped += 1
                    continue

                args.world_size = torch.cuda.device_count()

                if not batch_size_overridden:
                    args.batch_size = 64 if args.model == "ViT-L-14" else 128
                args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1

                print("\n" + "=" * 100)
                print(
                    f"[{model_name} | MULTI-LABEL] Finetuning on {dataset} for {args.epochs} epochs"
                )
                print(
                    f"Batch size: {args.batch_size} | Grad accumulation: {args.num_grad_accumulation}"
                )
                print(f"Using {args.world_size} GPUs with DDP")
                print("=" * 100 + "\n")

                torch.multiprocessing.spawn(
                    finetune_remote_sensing, args=(args, True), nprocs=args.world_size
                )

                multi_trained += 1

                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)

                print("\n" + "=" * 100)
                print(
                    f"[{model_name} | MULTI-LABEL] Completed fine-tuning on {dataset}"
                )
                print("GPU Memory cleaned up")
                print("=" * 100 + "\n")

        # ----------------------------------------------------------------------
        # Summary per model
        # ----------------------------------------------------------------------
        print("\n" + "=" * 100)
        print(f"FINE-TUNING SUMMARY ({model_name})")
        print("=" * 100)
        print("Single-label datasets:")
        print(f"  Total: {len(train_datasets_single) if train_datasets_single else 0}")
        print(f"  Trained: {single_trained}")
        print(f"  Skipped (already fine-tuned): {single_skipped}")
        print("\nMulti-label datasets:")
        print(f"  Total: {len(train_datasets_multi) if train_datasets_multi else 0}")
        print(f"  Trained: {multi_trained}")
        print(f"  Skipped (already fine-tuned): {multi_skipped}")
        print(f"\nGrand Total:")
        print(f"  Trained: {single_trained + multi_trained}")
        print(f"  Skipped: {single_skipped + multi_skipped}")
        print("=" * 100 + "\n")

        overall_results.append(
            {
                "model": model_name,
                "single_trained": single_trained,
                "single_skipped": single_skipped,
                "multi_trained": multi_trained,
                "multi_skipped": multi_skipped,
            }
        )

    if len(overall_results) > 1:
        total_single_trained = sum(r["single_trained"] for r in overall_results)
        total_single_skipped = sum(r["single_skipped"] for r in overall_results)
        total_multi_trained = sum(r["multi_trained"] for r in overall_results)
        total_multi_skipped = sum(r["multi_skipped"] for r in overall_results)

        print("\n" + "=" * 100)
        print("AGGREGATED FINE-TUNING SUMMARY (ALL MODELS)")
        print("=" * 100)
        print("Single-label datasets:")
        print(f"  Trained: {total_single_trained}")
        print(f"  Skipped (already fine-tuned): {total_single_skipped}")
        print("\nMulti-label datasets:")
        print(f"  Trained: {total_multi_trained}")
        print(f"  Skipped (already fine-tuned): {total_multi_skipped}")
        print(f"\nGrand Total:")
        print(f"  Trained: {total_single_trained + total_multi_trained}")
        print(f"  Skipped: {total_single_skipped + total_multi_skipped}")
        print("=" * 100 + "\n")
