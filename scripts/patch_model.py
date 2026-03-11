"""
Created on 2026-03-11
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from loguru import logger

from ball_detector import model


def main(args):
    """Load model data and replace legacy fields:

    architecture -> name
    """
    file_model = Path(args.model)
    logger.info(f"Loading model from {file_model}")
    model_data = torch.load(file_model, weights_only=False, map_location=args.device)

    # Validate model, if valid just say ok and end script
    if _smoke_test(model_data, args.device):
        logger.info("Model is valid -> nothing to do.")
        return
    logger.info(pformat(model_data, indent=2, compact=True))
    logger.warning("Model is not valid -> patching...")

    # Patch model
    model_data["name"] = model_data["architecture"]
    model_data.pop("architecture")

    # If no error occurred, patch was successful
    if _smoke_test(model_data, args.device):
        tmp_model = model.ModelData(**model_data)
        torch.save(asdict(tmp_model), file_model)
        logger.info("Patching was successful!")
    else:
        logger.error("Patching failed!")


def _smoke_test(model_args: dict, device: str) -> bool:
    """Smoke test for model."""
    try:
        tmp_model = model.ModelData(**model_args)
        tmp_model.ai_model.eval()
        tmp_model.ai_model.to(device)
        with torch.no_grad():
            tmp_model.ai_model(torch.rand(1, 3, 224, 224).to(device))
    except ValueError as e:
        logger.error(e)
        logger.error(pformat(model_args, indent=2, compact=True))
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    main(parser.parse_args())
