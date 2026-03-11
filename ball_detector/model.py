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


@dataclass
class ModelData:
    """Dataclass for model data, including weights and metadata."""

    ai_model: torch.nn.Module
    name: str
    source: str
    transforms: list[v2.Transform]
    cats: dict[int, str]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    with_augmentation: bool = False

    def __post_init__(self):
        self.ai_model = self.ai_model.to(self.device)


def save(file: Path, weights: torch.nn.Module, model_data: ModelData):
    """Structure data for export."""
    if not isinstance(weights, torch.nn.Module):
        raise ValueError("Weights must be a torch.nn.Module")

    # Generate output structure
    model_data.ai_model = weights
    output = asdict(model_data)
    torch.save(output, file)


def load_from_torchvision(name: str, device: str) -> ModelData:
    """Load models either from file or from torchvision."""
    logger.info(f"Loading model for {name}...")

    # Check if architecture is supported:
    det_models = models.list_models(module=models.detection)
    if name not in det_models:
        raise ValueError(f"Architecture {name} is not in:" + "\n".join(det_models))

    # Get model
    model = models.get_model(name, weights="DEFAULT")

    # Get weights
    weights_enum = models.get_model_weights(name)
    weights = models.get_weight(f"{weights_enum.__name__}.COCO_V1")

    return ModelData(
        ai_model=model,
        name=name,
        source="torch",
        transforms=[v2.ToImage(), v2.ToDtype(torch.float, scale=True)],
        cats=dict(enumerate(weights.meta["categories"])),
        device=device,
    )


def load_from_torchhub(repo: str, model_name: str, device: str) -> ModelData:
    """Load model from torchhub."""

    repo_models = torch.hub.list(repo)
    if model_name not in repo_models:
        raise ValueError(f"Model {model_name} is not in:" + "\n".join(repo_models))

    dir_models = Path(".cache/torchhub")
    dir_models.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(dir_models)
    model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)

    return ModelData(
        ai_model=model,
        name=model_name,
        source="torchhub",
        transforms=[v2.ToPILImage()],
        cats=model.names,
        device=device,
    )


def load_from_file(file_model: Path, device: str) -> ModelData:
    """Load model from pth file."""
    logger.info(f"Loading model from {file_model}")
    model_data = torch.load(file_model, weights_only=False, map_location=device)
    model_data["source"] = "file"
    model_data["name"] = file_model.stem
    model_data["transforms"] = [v2.ToImage(), v2.ToDtype(torch.float, scale=True)]
    return ModelData(**model_data)


def filename(dir_output: Path, name: str):
    """Add a suffix wit an index if the output folder already exists."""

    file_new = None
    for idx in range(100):
        file_new = dir_output / f"{name}_{idx:02d}.pth"
        if not file_new.exists():
            break

    logger.info(f"Output file: {file_new}")
    dir_output.mkdir(parents=True, exist_ok=True)
    return file_new
