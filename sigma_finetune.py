import os
import time
from typing import Dict

import torch

from src.datasets import get_dataloader, get_dataset, maybe_dictionarize
from src.eval.eval import eval_single_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils import parse_arguments, setup_logging
from src.utils.variables_and_paths import get_finetuned_path, get_zeroshot_path
from src.utils.sigma_param import SigmaParametrization


def build_sigma_modules(basis: Dict[str, Dict[str, torch.Tensor]]):
    modules = torch.nn.ModuleDict()
    for key, fv in basis.items():
        if not ("U" in fv and "V" in fv and "sigma" in fv):
            continue
        U, V, sigma = fv["U"], fv["V"], fv["sigma"]
        if U.ndim == 2 and V.ndim == 2 and sigma.ndim == 1:
            modules[key] = SigmaParametrization(U, V, sigma)
    return modules


def apply_sigma_deltas_to_encoder(image_encoder: ImageEncoder, sigma_modules: torch.nn.ModuleDict):
    with torch.no_grad():
        sd = image_encoder.state_dict()
        for key, module in sigma_modules.items():
            if key in sd and module.sigma.numel() > 0:
                delta = module().to(sd[key].device)
                if sd[key].shape == delta.shape:
                    sd[key] = sd[key] + delta
        image_encoder.load_state_dict(sd)


def finetune_sigma(args):
    setup_logging(filename="sigma_finetune.log")

    # Load base encoder (zeroshot)
    train_dataset = args.train_dataset
    image_encoder = ImageEncoder(args.model)

    # Load basis exported by energy_train.py
    basis_path = os.path.join(os.getcwd(), "svd_basis.pth")
    assert os.path.exists(
        basis_path), f"Missing {basis_path}. Run energy_train.py first."
    basis = torch.load(basis_path, map_location="cpu")

    sigma_modules = build_sigma_modules(basis)
    sigma_modules = sigma_modules.cuda()

    classification_head = get_classification_head(args, train_dataset)
    model = ImageClassifier(image_encoder.cuda(), classification_head)
    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    data_loader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=None)

    # Only optimize sigma parameters
    params = [p for p in sigma_modules.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    print("Trainable sigma params:", sum(p.numel() for p in params))

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(data_loader):
            start_time = time.time()

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()

            # Reconstruct encoder weights with current sigma each step
            apply_sigma_deltas_to_encoder(model.image_encoder, sigma_modules)

            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if i % print_every == 0:
                print(
                    f"epoch {epoch} iter {i} loss {loss.item():.4f} time {time.time()-start_time:.3f}")

    # Save zeroshot and finetuned encoders
    ckpdir = os.path.join(args.save_dir, train_dataset)
    os.makedirs(ckpdir, exist_ok=True)
    zs_path = get_zeroshot_path(args.model_location, train_dataset, args.model)
    ft_path = get_finetuned_path(
        args.model_location, train_dataset, args.model)
    model.image_encoder.save(ft_path)

    # Evaluate
    test_accuracy = eval_single_dataset(
        model.image_encoder, train_dataset, args)
    print("Accuracy:", test_accuracy)


if __name__ == "__main__":
    args = parse_arguments()
    # Expect single dataset name (e.g., MNISTVal)
    if isinstance(args.train_dataset, list):
        args.train_dataset = args.train_dataset[0]
    args.save_dir = os.path.join(args.model_location, args.model)
    finetune_sigma(args)
