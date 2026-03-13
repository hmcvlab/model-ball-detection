"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from torchvision import io, models

from ball_detector import aux, coco, draw, model

ROOT = Path(__file__).parent


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start evaluation...")

    # Load model
    if args.file_model:
        ai_models = [
            model.load_from_file(file, args.device)
            for file in args.file_model.parent.glob(args.file_model.name)
        ]
    elif args.torch_model:
        ai_models = [model.load_from_torchvision(args.torch_model, args.device)]
    elif args.yolo_model:
        ai_models = [
            model.load_from_torchhub("ultralytics/yolov5", args.yolo_model, args.device)
        ]
    else:
        raise ValueError("Either --file-model or --default-model must be set.")

    file_benchmark = (
        aux.DATA_ROOT / f"analysis/{args.holdout.parent.stem}_benchmark.csv"
    )
    for model_data in ai_models:
        # Load dataset
        transforms = model_data.transforms
        loader = aux.load_dataset(args.holdout, transforms=transforms, shuffle=False)

        # Run inference
        try:
            if args.yolo_model:
                results = coco.inference_yolo(model_data, loader)
            else:
                results = coco.inference_torch(model_data, loader)
        except ValueError as e:
            logger.error(e)
            continue


        # Run COCO evaluation
        stats_coco = coco.run_eval(args.holdout, results)

        # Save samples
        file_out = Path("tmp") / f"{model_data.name}-{model_data.source}.jpg"
        file_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output file: {file_out}")
        img = draw.sample_with_boxes(args.holdout, results)
        io.write_jpeg(img, str(file_out))

        # Save results
        stats = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": model_data.name,
            "source": model_data.source,
        } | stats_coco
        df_stats = pd.DataFrame(stats, index=[0])
        df_stats.to_csv(
            file_benchmark, index=False, mode="a", header=not file_benchmark.exists()
        )

    # Cleanup csv file by dropping duplicates
    df_stats = pd.read_csv(file_benchmark)
    df_stats = df_stats.drop_duplicates(
        subset=["name", "source", "ap_mean", "ar_max_100"]
    ).sort_values(["ap_mean", "ar_max_100"], ascending=False)
    df_stats.to_csv(file_benchmark, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        type=Path,
        default=aux.DATA_ROOT / "datasets/accurate-balls/holdout.coco.json",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-model", type=Path)
    group.add_argument(
        "--torch-model",
        choices=models.list_models(models.detection),
    )
    group.add_argument("--yolo-model", choices=torch.hub.list("ultralytics/yolov5"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    main(parser.parse_args())
