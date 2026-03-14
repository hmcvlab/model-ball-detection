"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import v2

DATA_ROOT = Path("/mnt/data")


def file_benchmark(file_holdout: Path):
    """Return path to benchmark file."""
    return DATA_ROOT / f"analysis/{file_holdout.parent.stem}_benchmark.csv".replace(
        "-", "_"
    )


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


def _collate_fn(batch):
    """Default collate function."""
    return tuple(zip(*batch))
