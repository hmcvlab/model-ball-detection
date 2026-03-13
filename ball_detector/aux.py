"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import v2

SEED = 42
DATA_ROOT = Path("/mnt/data")


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


def to_device(images: list[torch.Tensor], targets: list[dict], device: str):
    """Move images and targets to device."""
    keys = ["boxes", "labels"]
    images = [img.to(device) for img in images]
    targets = [{k: t[k].to(device) for k in keys} for t in targets]
    return images, targets


def load_dataset(file: Path, transforms: list[v2.Transform] | v2.Compose, shuffle=True):
    """Wrapper for torch-based dataset"""
    transforms = v2.Compose(transforms) if isinstance(transforms, list) else transforms
    dataset_coco = datasets.CocoDetection(file.parent, str(file), transforms=transforms)
    dataset_coco = datasets.wrap_dataset_for_transforms_v2(
        dataset_coco, target_keys=("boxes", "labels", "image_id")
    )

    return torch.utils.data.DataLoader(
        dataset_coco,
        batch_size=4,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=_collate_fn,
    )


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

