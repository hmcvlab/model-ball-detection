"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torchvision import datasets
from torchvision.transforms import v2

SEED = 42


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


def _collate_fn(batch):
    """Default collate function."""
    return tuple(zip(*batch))


def load_dataset(file: Path, transform, shuffle=True):
    """Wrapper for torch-based dataset"""
    dataset_coco = datasets.CocoDetection(file.parent, str(file), transform=transform)
    dataset_coco = datasets.wrap_dataset_for_transforms_v2(
        dataset_coco, target_keys=("boxes", "labels", "masks", "image_id")
    )
    return torch.utils.data.DataLoader(
        dataset_coco,
        batch_size=6,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=_collate_fn,
    )


def load_model():
    pass


def add_transform(params: Augmentation | None, train: bool):
    """Generate torch transformation object for validate and train."""
    transform = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if params is None or not train:
        return v2.Compose(transform)

    # Processing that should only be done during training
    if train:
        # Geometric augemntations
        if params.p_hflip is not None:
            transform += [v2.RandomHorizontalFlip(params.p_hflip)]

        if params.p_vflip is not None:
            transform += [v2.RandomVerticalFlip(params.p_vflip)]

        if params.p_zoom is not None:
            transform += [v2.RandomZoomOut(fill=0, p=params.p_zoom, side_range=[1, 2])]

        if params.p_distort is not None:
            transform += [v2.RandomPhotometricDistort(p=params.p_distort)]
        transform += [v2.SanitizeBoundingBoxes()]

        # Pixel augementations must be after convert to float
        if params.jitter_brightness is not None and params.jitter_hue is not None:
            jitter = v2.ColorJitter(
                brightness=params.jitter_brightness, hue=params.jitter_hue
            )
            transform += [v2.RandomApply([jitter], p=params.p_jitter)]

        if params.noise_sigma is not None:
            transform += [v2.GaussianNoise(sigma=params.noise_sigma)]

    # Normalization has to be done last for training and validation
    if params.normalize_mean is not None and params.normalize_std is not None:
        logger.warning("Normalization has been loaded BUT IS NOT APPLIED")

    return v2.Compose(transform)
