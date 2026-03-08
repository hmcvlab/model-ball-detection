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
from torchvision.transforms import v2

from ball_detector import aux

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelData:
    """Dataclass for model data, including weights and metadata."""

    ai_model: torch.nn.Module
    architecture: str
    source: str
    transforms: torch.nn.Module
    dataset: str = "coco"
    with_augmentation: bool = False


def save(file: Path, weights: torch.nn.Module, model_data: ModelData):
    """Structure data for export."""

    # Generate output structure
    model_data.ai_model = weights
    output = asdict(model_data)
    torch.save(output, file)


def load_from_torchvision(architecture: str) -> ModelData:
    """Load models either from file or from torchvision."""

    # Check if architecture is supported:
    det_models = models.list_models(module=models.detection)
    if architecture not in det_models:
        raise ValueError(
            f"Architecture {architecture} is not in:" + "\n".join(det_models)
        )

    model = models.get_model(architecture, weights="DEFAULT")
    model = model.to(DEVICE)

    # Get weights to load transforms
    logger.info(f"Loading weights for {architecture}")
    weights_enum = models.get_model_weights(architecture)
    w_str = f"{weights_enum.__name__}.COCO_V1"
    logger.info(f"Loading weights {w_str}")
    transforms = models.get_weight(w_str).transforms()

    # Generate transforms
    transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize((255, 255)),  # Adjust the input size according to your needs
        ]
    )

    return ModelData(
        ai_model=model,
        architecture=architecture,
        source="torch",
        transforms=transforms,
    )


def load_from_torchhub(repo: str, model_name: str):
    """Load model from torchhub."""

    repo_models = torch.hub.list(repo)
    if model_name not in repo_models:
        raise ValueError(f"Model {model_name} is not in:" + "\n".join(repo_models))

    torch.hub.set_dir(aux.DATA_ROOT / "models/torchhub")
    model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)
    model = model.to(DEVICE)

    transforms = v2.Compose(
        [
            # v2.ToTensor(),
            # v2.Resize((320, 640)),  # Adjust the input size according to your needs
            v2.ToPILImage(),
        ]
    )
    # transforms = None

    return ModelData(
        ai_model=model,
        architecture=model_name,
        source="torchhub",
        transforms=transforms,
    )


def load_from_file(file_model: Path) -> ModelData:
    """Load model from pth file."""
    model_data = torch.load(file_model, weights_only=False, map_location=DEVICE)
    model_data.setdefault("source", file_model.stem)
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
