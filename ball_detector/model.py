"""
Created on 2026-03-07
Copyright (c) 2026 Munich University of Applied Sciences

This module contains functions for model handling e.g. load, save
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelData:
    """Dataclass for model data, including weights and metadata."""

    ai_model: torch.nn.Module
    architecture: str
    dataset: str = "coco"
    with_augmentation: bool = False


def save(file: Path, weights: torch.nn.Module, model_data: ModelData):
    """Structure data for export."""

    # Generate output structure
    model_data.ai_model = weights
    output = asdict(model_data)
    torch.save(output, file)


def load_from_torch(architecture: str) -> ModelData:
    """Load models either from file or from torchvision."""

    # Check if architecture is supported:
    det_models = models.list_models(module=models.detection)
    if architecture not in det_models:
        raise ValueError(
            f"Architecture {architecture} is not in:" + "\n".join(det_models)
        )

    model = models.get_model(architecture, weights="DEFAULT")

    model = model.to(DEVICE)
    return ModelData(ai_model=model, architecture=architecture)


def load_from_file(file_model: Path) -> ModelData:
    """Load model from pth file."""
    model_data = torch.load(file_model, weights_only=False, map_location=DEVICE)
    model_data = ModelData(**model_data)
    return model_data


def filename(dir_output: Path, name: str):
    """Add a suffix wit an index if the output folder already exists."""

    file_new = None
    for idx in range(100):
        file_new = dir_output / f"{name}-{idx:02d}.pth"
        if not file_new.exists():
            break

    logger.info(f"Output file: {file_new}")
    dir_output.mkdir(parents=True, exist_ok=True)
    return file_new
