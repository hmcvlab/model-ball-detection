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
    model_data = torch.load(file_model, weights_only=False, map_location=model.DEVICE)

    # Validate model, if valid just say ok and end script
    try:
        _ = model.ModelData(**model_data)
        logger.info("Model is valid.")
        return
    except TypeError:
        # Print info about model
        logger.info(pformat(model_data, indent=2, compact=True))
        logger.warning("Model is not valid -> patching...")

    # Patch model
    model_data["name"] = model_data["architecture"]
    model_data.pop("architecture")

    # Validate and save if valid
    tmp_model = model.ModelData(**model_data)
    tmp_model.ai_model.eval()
    torch.save(asdict(tmp_model), file_model)
    logger.info("Patching was successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    main(parser.parse_args())
