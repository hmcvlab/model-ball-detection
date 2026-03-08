"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torchvision import models
from tqdm import tqdm

from ball_detector import aux, model

ROOT = Path(__file__).parent


@dataclass
class TrainParameter:
    """Dataclass for model parameters."""

    epochs: int = 10
    batch_size: int = 4
    lr: float = 0.0001
    weight_decay: float = 0.01
    warmup_epochs: int = 0
    momentum: float = 0.9


@dataclass
class TrainResults:
    """Dataclass to store training results, including metrics and logs."""

    final_model: torch.nn.Module | None = None
    logs: list = field(default_factory=list)


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start training...")
    t_file = args.dataset / "train.coco.json"
    v_file = args.dataset / "valid.coco.json"

    # Load transformations
    aug_params = aux.Augmentation()

    # Load model
    if args.file_model:
        model_data = model.load_from_file(args.file_model)
    elif args.torch_model:
        model_data = model.load_from_torchvision(args.torch_model)
    elif args.yolo_model:
        model_data = model.load_from_torchhub("ultralytics/yolov5", args.yolo_model)
    else:
        raise ValueError("Either --file-model or --default-model must be set.")

    # Patch model_data
    with open(t_file, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    model_data.cats = {c["id"]: c["name"] for c in cats}

    # Load transforms
    t_transforms = model_data.transforms
    if args.augment:
        model_data.with_augmentation = True
        t_transforms += aux.augmentation_transforms(aug_params)

    # Load dataset
    t_loader = aux.load_dataset(t_file, t_transforms, shuffle=True)
    v_loader = aux.load_dataset(v_file, model_data.transforms, shuffle=False)

    result = _train(model_data.ai_model, t_loader, v_loader, params=TrainParameter())

    # Export model
    file = model.filename(args.dir_output, model_data.architecture)
    model.save(file, result.final_model, model_data)

    # Export logs
    df = pd.DataFrame(result.logs)
    df.to_csv(file.with_suffix(".csv"), index=False)


def _train(
    ai_model: torch.nn.Module,
    train_loader,
    val_loader,
    params: TrainParameter,
) -> TrainResults:
    """Train a torch model using the given data loaders."""

    optimizer = torch.optim.AdamW(
        [p for p in ai_model.parameters() if p.requires_grad],
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    # Set up learning rate scheduler
    warmup_steps = params.warmup_epochs * len(train_loader)
    if warmup_steps > 0:

        def lr_lambda(step):
            warmup_factor = 0.1
            if step >= warmup_steps:
                return 1.0
            alpha = step / warmup_steps
            return warmup_factor + alpha * (1.0 - warmup_factor)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    result = TrainResults()
    best_loss = float("inf")
    for epoch in tqdm(range(params.epochs), desc="Epoch", unit="epoch"):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Batch", unit="batch", leave=False)

        ai_model.train()
        for step, (images, targets) in enumerate(progress_bar):
            images, targets = aux.to_device(images, targets)

            loss_dict = ai_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += losses.item()

            # Save logs
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=running_loss / (step + 1), lr=f"{current_lr:.2e}"
            )

        n_batches = len(train_loader)
        avg_val_loss = _validate(ai_model, val_loader)

        result.logs.append(
            {
                "step": (epoch + 1) * n_batches,
                "epoch": epoch,
                "loss": running_loss / n_batches,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            result.final_model = deepcopy(ai_model)

    return result


def _train_yolo(
    ai_model: torch.nn.Module,
    train_loader,
    val_loader,
    params: TrainParameter,
) -> TrainResults:
    """Train a torch model using the given data loaders."""

    optimizer = _smart_optimizer(ai_model, params)

    result = TrainResults()
    best_loss = float("inf")
    for epoch in tqdm(range(params.epochs), desc="Epoch", unit="epoch"):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Batch", unit="batch", leave=False)

        ai_model.train()
        for step, (images, targets) in enumerate(progress_bar):
            # images, targets = aux.to_device(images, targets)

            loss_dict = ai_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

            # Save logs
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=running_loss / (step + 1), lr=f"{current_lr:.2e}"
            )

        n_batches = len(train_loader)
        avg_val_loss = _validate(ai_model, val_loader)

        result.logs.append(
            {
                "step": (epoch + 1) * n_batches,
                "epoch": epoch,
                "loss": running_loss / n_batches,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            result.final_model = deepcopy(ai_model.state_dict())

    return result


def _smart_optimizer(ai_model, params: TrainParameter):
    """Initializes YOLOv5 smart optimizer with 3 parameter groups for different decay
    configurations.

    Groups are 0) weights with decay, 1) weights no decay, 2) biases no decay.
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(
        v for k, v in torch.nn.__dict__.items() if "Norm" in k
    )  # normalization layers, i.e. BatchNorm2d()
    for v in ai_model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    optimizer = torch.optim.AdamW(
        g[2], lr=params.lr, betas=(params.momentum, 0.999), weight_decay=0.0
    )
    optimizer.add_param_group(
        {"params": g[0], "weight_decay": params.weight_decay}
    )  # add g0 with weight_decay
    optimizer.add_param_group(
        {"params": g[1], "weight_decay": 0.0}
    )  # add g1 (BatchNorm2d weights)
    return optimizer


def _validate(ai_model, loader: torch.utils.data.DataLoader):
    """Run validation loop and return average validation loss.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (DataLoader): Validation data loader.

    Returns:
        float: Average validation loss.
    """

    ai_model.train()  # Returns model training output

    val_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = aux.to_device(images, targets)
            loss_dict = ai_model(images, targets)
            losses = sum(loss_dict.values())
            val_loss += losses.item()
    return val_loss / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=aux.DATA_ROOT / "datasets/accurate-balls"
    )
    parser.add_argument(
        "--dir-output", type=Path, default=aux.DATA_ROOT / "models/torch"
    )
    parser.add_argument("--augment", action="store_true", default=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-model", type=Path)
    group.add_argument("--torch-model", choices=models.list_models(models.detection))
    group.add_argument("--yolo-model", choices=torch.hub.list("ultralytics/yolov5"))

    main(parser.parse_args())
