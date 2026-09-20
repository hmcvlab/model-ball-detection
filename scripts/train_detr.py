"""
Created on 2026-03-13
Copyright (c) 2026 Munich University of Applied Sciences

Script to train a PyTorch model for object detection.
"""

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import detr
import numpy as np
import torch
from loguru import logger

from ball_detector import aux


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start training...")
    t_file = args.dataset / "train.coco.json"
    v_file = args.dataset / "valid.coco.json"
    h_file = args.dataset / "holdout.coco.json"

    train_params = detr.parameters.Train.from_args(args)
    model_params = detr.parameters.Model.from_args(args)
    loss_params = detr.parameters.Loss.from_args(args)
    aug_params = detr.parameters.Augmentation.from_args(args)

    # Fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    categories = detr.io.load_categories(t_file)

    # Load model
    if detr.io.is_legacy_model(args.model):
        logger.warning(f"Model {args.model} is in legacy format, converting...")
        model_data = detr.io.load_model_legacy(
            args.model,
            device=args.device,
            categories=categories,
            model_params=model_params,
            loss_params=loss_params,
            train_params=train_params,
        )
    else:
        model_data = detr.io.load_model(
            args.model, device=args.device, categories=categories
        )
    model_data.meta = detr.ModelMeta(
        categories=categories,
        dataset=str(t_file.relative_to(t_file.parents[1])),
        model_type=model_data.meta.model_type,
        subtype=model_data.meta.subtype,
        train_params=train_params,
        settings={
            "model_params": asdict(model_params),
            "loss_params": asdict(loss_params),
            "aug_params": asdict(aug_params),
        },
    )
    logger.info(f"Training parameters: {model_data.meta.train_params}")

    # Load transforms
    if args.augment:
        t_transforms = detr.transforms.augmentation(aug_params)
    else:
        t_transforms = detr.transforms.default()
    v_transforms = detr.transforms.default()

    # Load dataset
    is_segm = model_data.meta.model_type is detr.ModelType.DETR_SEGM
    t_loader = detr.dataset.load(
        t_file,
        t_transforms,
        return_masks=is_segm,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    v_loader = detr.dataset.load(
        v_file,
        v_transforms,
        return_masks=is_segm,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    h_loader = detr.dataset.load(
        h_file,
        v_transforms,
        return_masks=is_segm,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )

    model_data.set_device(args.device)
    new_model = detr.train.run(
        model_data,
        t_loader,
        v_loader,
        device=args.device,
        params=train_params,
        v_file=v_file,
    )

    # Evaluate on validation and holdout split
    outputs = detr.coco.inference(new_model, v_loader, device=args.device)
    stats = detr.coco.run_eval(v_file, outputs, iou_type="bbox")
    logger.info(f"Validation metrics: {stats}")

    outputs = detr.coco.inference(new_model, h_loader, device=args.device)
    stats = detr.coco.run_eval(h_file, outputs, iou_type="bbox")
    logger.info(f"Holdout metrics: {stats}")

    # Export model + logs
    args.dir_output.mkdir(parents=True, exist_ok=True)
    file_model = detr.io.output_filename(args.dir_output, new_model.name)
    logger.info(f"Training completed. Exporting model to {file_model}")
    detr.io.save_model(new_model, file_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=aux.DATASET_DIR)
    parser.add_argument(
        "--dir-output", type=Path, default=aux.MODEL_DIR / "detr"
    )
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--model",
        type=Path,
        default=aux.MODEL_DIR / "detr/detr-r50-e632da11.pth",
    )

    # Add training, model, loss and augmentation parameters from dataclasses
    detr.parameters.add_args(parser, detr.parameters.Train)
    detr.parameters.add_args(parser, detr.parameters.Model)
    detr.parameters.add_args(parser, detr.parameters.Loss)
    detr.parameters.add_args(parser, detr.parameters.Augmentation)

    main(parser.parse_args())
