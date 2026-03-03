"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from pathlib import Path

import torch
from loguru import logger as log
from torchvision import datasets, models
from torchvision.transforms import v2

from model import aux

ROOT = Path(__file__).parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""

    # Load transformations
    aug_params = aux.Augmentation()
    t_transform = aux.add_transform(aug_params, train=True)
    v_transform = aux.add_transform(aug_params, train=False)

    # Load dataset
    t_loader = aux.load_dataset(args.dataset / "train", t_transform, shuffle=True)
    v_loader = aux.load_dataset(args.dataset / "valid", v_transform, shuffle=False)

    # Load model
    model_settings = {"weights": None, "weights_backbone": None}
    model_data = torch.load(args.model, weights_only=False, map_location=DEVICE)
    model = models.detection.maskrcnn_resnet50_fpn(**model_settings)
    model.load_state_dict(model_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    main(parser.parse_args())
