"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torchvision.transforms import v2
from tqdm import tqdm

from ball_detector import aux, model

ROOT = Path(__file__).parent
SEED = 42


@dataclass
class Parameter:
    """Dataclass for model parameters."""

    epochs: int = 10
    batch_size: int = 4
    lr: float = 0.0001
    weight_decay: float = 0.01
    warmup_epochs: int = 0
    momentum: float = 0.9


# pylint: disable=too-many-instance-attributes
@dataclass
class Augmentation:
    """Dataclass for image augmentation parameters.

    Normalization values for MASK-R-CNN:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    """

    seed: int = SEED
    noise_sigma: float | None = 0.05
    jitter_hue: float | None = 0.05
    jitter_brightness: float | None = 0.05
    p_jitter: float | None = 0.2
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None

    # Geometric augmentations
    p_zoom: float | None = None
    p_hflip: float | None = None
    p_vflip: float | None = None
    p_distort: float | None = None


def run(
    model_data: model.Data,
    train_loader,
    val_loader,
    params: Parameter,
) -> model.Data:
    """Train a torch model using the given data loaders."""
    logger.info(f"Start training of {model_data.name} model...")
    ai_model = model_data.ai_model
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

    logs = []
    best_loss = float("inf")
    for epoch in tqdm(range(params.epochs), desc="Epoch", unit="epoch"):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Batch", unit="batch", leave=False)

        ai_model.train()
        for step, (images, targets) in enumerate(progress_bar):
            images, targets = aux.to_device(images, targets, model_data.device)

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

        logs.append(
            {
                "step": (epoch + 1) * n_batches,
                "epoch": epoch,
                "loss": running_loss / n_batches,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_data.ai_model = deepcopy(ai_model)

    model_data.logs = logs
    return model_data


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
            images, targets = aux.to_device(images, targets, ai_model.device)
            loss_dict = ai_model(images, targets)
            losses = sum(loss_dict.values())
            val_loss += losses.item()
    return val_loss / len(loader)


def augmentation_transforms(params: Augmentation) -> list[v2.Transform]:
    """Generate torch transformation object for validate and train."""
    transform = []

    # Geometric augmentations
    if params.p_hflip is not None:
        transform += [v2.RandomHorizontalFlip(params.p_hflip)]

    if params.p_vflip is not None:
        transform += [v2.RandomVerticalFlip(params.p_vflip)]

    if params.p_zoom is not None:
        transform += [v2.RandomZoomOut(fill=0, p=params.p_zoom, side_range=[1, 2])]

    if params.p_distort is not None:
        transform += [v2.RandomPhotometricDistort(p=params.p_distort)]
    transform += [v2.SanitizeBoundingBoxes()]

    # Pixel augmentations must be after convert to float
    if params.jitter_brightness is not None and params.jitter_hue is not None:
        jitter = v2.ColorJitter(
            brightness=params.jitter_brightness, hue=params.jitter_hue
        )
        transform += [v2.RandomApply([jitter], p=params.p_jitter)]

    if params.noise_sigma is not None:
        transform += [v2.GaussianNoise(sigma=params.noise_sigma)]

    return transform
