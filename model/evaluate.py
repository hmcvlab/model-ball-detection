"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger as log
from torchvision import datasets, models
from torchvision.transforms import v2

from model import aux

ROOT = Path(__file__).parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Load dataset
    loader = aux.load_dataset(args.holdout, transform, shuffle=False)

    # Load model
    model = aux.load_model_from_file(args.model)
    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--holdout", type=Path)
    main(parser.parse_args())
