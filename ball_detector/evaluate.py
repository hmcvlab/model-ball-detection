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
        model_data = model.load_from_file(args.file_model)
        model_name = args.file_model.stem
    elif args.torch_model:
        model_data = model.load_from_torch(args.torch_model)
        model_name = args.torch_model
    else:
        raise ValueError("Either --file-model or --default-model must be set.")
    transforms = aux.add_transform(None, train=False)

    # Load dataset
    loader = aux.load_dataset(args.holdout, transforms=transforms, shuffle=False)

    # Run inference
    results = _inference(model_data.ai_model, loader)

    # Save results
    coco_gt = COCO(args.holdout)
    stats = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "architecture": model_data.architecture,
    }
    stats = stats | run_coco(coco_gt, results)
    df_stats = pd.DataFrame(stats, index=[0])
    file_benchmark = args.holdout.parent / "benchmark.csv"
    df_stats.to_csv(file_benchmark, index=False, mode="a")

    logger.info("Done!")


def _inference(ai_model, loader) -> list:
    """Runs COCO evaluation and prints results to stdout.

    Args:
        model: A PyTorch instance segmentation model.
        data_loader: A PyTorch data loader for the COCO dataset.
    """
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        images = list(image.to(model.DEVICE) for image in images)
        with torch.no_grad():
            outputs = ai_model(images)

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

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file-model", type=Path)
    group.add_argument(
        "--torch-model",
        choices=models.list_models(models.detection),
        default="ssd300_vgg16",
    )

    main(parser.parse_args())
