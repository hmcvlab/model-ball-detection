"""
Created on 2026-03-07
Copyright (c) 2026 Munich University of Applied Sciences

This module contains functions for model handling e.g. load, save
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torchvision import models
from torchvision.transforms import v2
from ultralytics import YOLO


@dataclass
class Data:
    """Dataclass for model data, including weights and metadata."""

    ai_model: torch.nn.Module
    name: str
    source: str
    transforms: list[v2.Transform]
    cats: dict[int, str]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    with_augmentation: bool = False
    logs: list = field(default_factory=list)

    def __post_init__(self):
        self.ai_model = self.ai_model.to(self.device)

    def export(self, file: Path):
        """Export model to pth file."""
        data4export = asdict(self)
        logs = data4export.pop("logs")

        # Save model to pth
        torch.save(data4export, file.with_suffix(".pth"))

        # Save logs to csv
        df_logs = pd.DataFrame(logs)
        df_logs.to_csv(file.with_suffix(".csv"), index=False)


def load_from_torchvision(name: str, device: str) -> Data:
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

    return Data(
        ai_model=model,
        name=name,
        source="torch",
        transforms=[v2.ToImage(), v2.ToDtype(torch.float, scale=True)],
        cats=dict(enumerate(weights.meta["categories"])),
        device=device,
    )


def load_from_torchhub(repo: str, model_name: str, device: str) -> Data:
    """Load model from torchhub."""

    repo_models = torch.hub.list(repo)
    if model_name not in repo_models:
        raise ValueError(f"Model {model_name} is not in:" + "\n".join(repo_models))

    dir_models = Path("/tmp/torchhub")
    dir_models.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(dir_models)
    model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True)

    return Data(
        ai_model=model,
        name=model_name,
        source="torchhub",
        transforms=[v2.ToPILImage()],
        cats=model.names,
        device=device,
    )


def load_from_file(file_model: Path, device: str) -> Data:
    """Load model from pth file."""
    logger.info(f"Loading model from {file_model}")
    model_data = torch.load(file_model, weights_only=False, map_location=device)

    # Handle custom model files differently
    if "train_args" in model_data:
        return Data(
            ai_model=YOLO(file_model, task="detect"),
            name=file_model.stem,
            source="file",
            transforms=[v2.ToPILImage()],
            cats=model_data["model"].names,
            device=device,
        )

    model_data["source"] = "file"
    model_data["name"] = file_model.stem
    model_data["transforms"] = [v2.ToImage(), v2.ToDtype(torch.float, scale=True)]
    return Data(**model_data)


def filename(dir_output: Path, name: str, suffix: str = ".pth") -> Path:
    """Add a suffix wit an index if the output folder already exists."""

    file_new = dir_output / f"{name}_00.pth"
    for idx in range(1, 99, 1):
        if not file_new.exists():
            logger.info(f"Output file: {file_new}")
            return file_new
        file_new = dir_output / f"{name}_{idx:02d}{suffix}"

    raise ValueError("Unable to find a non-existing output file!")
