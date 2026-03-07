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
from pycocotools.cocoeval import COCOeval
from torchvision import models, ops
from tqdm import tqdm

from ball_detector import aux, model

ROOT = Path(__file__).parent


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start evaluation...")

    # Load model
    if args.file_model:
        ai_models = [
            model.load_from_file(file)
            for file in args.file_model.parent.glob(args.file_model.name)
        ]
    elif args.torch_model:
        ai_models = [model.load_from_torchvision(args.torch_model)]
    elif args.yolov5_model:
        ai_models = [model.load_from_torchhub("ultralytics/yolov5", args.yolov5_model)]
    else:
        raise ValueError("Either --file-model or --default-model must be set.")

    file_benchmark = (
        aux.DATA_ROOT / f"analysis/{args.holdout.parent.stem}_benchmark.csv"
    )
    for model_data in ai_models:
        # Load dataset
        loader = aux.load_dataset(
            args.holdout, transforms=model_data.transforms, shuffle=False
        )

        # Run inference
        if args.yolov5_model:
            results = _inference_yolo(model_data.ai_model, loader)
        else:
            results = _inference_torch(model_data.ai_model, loader)

        # Save results
        coco_gt = COCO(args.holdout)
        stats = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": model_data.source,
            "architecture": model_data.architecture,
        }
        stats = stats | run_coco(coco_gt, results)
        df_stats = pd.DataFrame(stats, index=[0])
        df_stats.to_csv(
            file_benchmark, index=False, mode="a", header=not file_benchmark.exists()
        )

    logger.info("Done!")


def _inference_yolo(
    ai_model: torch.nn.Module, loader: torch.utils.data.DataLoader
) -> list:
    """Runs COCO evaluation and returns results in coco compatible format."""
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        with torch.no_grad():
            outputs = ai_model(images)

        for out, target in zip(outputs.pandas().xywh, targets):
            for _, row in out.iterrows():
                bbox = row[["xcenter", "ycenter", "width", "height"]].tolist()
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "category_id": int(row["class"]),
                        "bbox": bbox,
                        "score": float(row["confidence"]),
                    }
                )

    return results


def _inference_torch(ai_model, loader) -> list:
    """Runs COCO evaluation and prints results to stdout.

    Args:
        model: A PyTorch instance segmentation model.
        data_loader: A PyTorch data loader for the COCO dataset.
    """
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        images_tensor = torch.stack(images).to(model.DEVICE)
        with torch.no_grad():
            outputs = ai_model(images_tensor)

        for out, target in zip(outputs, targets):
            # Convert boxes using torch
            out["boxes"] = ops.box_convert(out["boxes"], in_fmt="xyxy", out_fmt="xywh")

            outputs = {k: v.detach().cpu() for k, v in out.items()}
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score),
                    }
                )

    return results


def run_coco(coco_gt: COCO, results: list) -> dict:
    """Runs COCO evaluation and prints results to stdout."""
    if len(results) == 0:
        logger.warning("No results to evaluate.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Convert stats to dict
    coco_eval.stats = coco_eval.stats.round(3)
    return {
        "ap_mean": coco_eval.stats[0],
        "ap_50": coco_eval.stats[1],
        "ap_75": coco_eval.stats[2],
        "ap_small": coco_eval.stats[3],
        "ap_medium": coco_eval.stats[4],
        "ap_large": coco_eval.stats[5],
        "ar_max_1": coco_eval.stats[6],
        "ar_max_10": coco_eval.stats[7],
        "ar_max_100": coco_eval.stats[8],
        "ar_small": coco_eval.stats[9],
        "ar_medium": coco_eval.stats[10],
        "ar_large": coco_eval.stats[11],
    }


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
    group.add_argument("--yolov5-model", choices=torch.hub.list("ultralytics/yolov5"))

    main(parser.parse_args())
