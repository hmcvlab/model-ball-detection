"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import ops
from tqdm import tqdm

from ball_detector import aux, hough, model

ROOT = Path(__file__).parent


def inference_yolo(model_data: model.Data, loader: torch.utils.data.DataLoader) -> list:
    """Runs COCO evaluation and returns results in coco compatible format."""
    ai_model = model_data.ai_model
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        with torch.no_grad():
            outputs = ai_model(images)

        for out, target in zip(outputs, targets):
            data = out.boxes
            boxes = ops.box_convert(data.xyxy, in_fmt="xyxy", out_fmt="xywh")
            for box, score, cat in zip(boxes, data.conf, data.cls):
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "bbox": box.tolist(),
                        "score": float(score),
                        "category_id": int(cat),
                    }
                )

    # Add name to results
    df = pd.DataFrame(results)
    df["name"] = df["category_id"].map(model_data.cats)
    results = df.to_dict("records")
    return results


def inference_torch(
    model_data: model.Data, loader: torch.utils.data.DataLoader
) -> list:
    """Runs COCO evaluation and prints results to stdout.

    Args:
        model: A PyTorch instance segmentation model.
        data_loader: A PyTorch data loader for the COCO dataset.
    """
    ai_model = model_data.ai_model
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        images, target = aux.to_device(images, targets, model_data.device)
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

    # Add name to results
    df = pd.DataFrame(results)
    df["name"] = df["category_id"].map(model_data.cats)
    results = df.to_dict("records")
    return results


def inference_hough(loader: torch.utils.data.DataLoader) -> list:
    """Run hough circle detection on dataset."""
    results = []
    for images, targets in tqdm(loader, desc="Evaluating hough"):

        for img, target in zip(images, targets):
            img = np.array(img)
            res = hough.circles(img)
            res["image_id"] = int(target["image_id"])
            results += res.head(len(target["boxes"])).to_dict("records")

    return results


def run_eval(file_holdout: Path, results: list[dict]) -> dict:
    """Runs COCO evaluation and prints results to stdout."""
    coco_gt = COCO(file_holdout)
    results = _adapt_results_to_coco(results, coco_gt)

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


def _adapt_results_to_coco(results: list[dict], coco_gt: COCO) -> list[dict]:
    """The categories in the model might be different from the ones in the COCO dataset.
    Hence we want to select certain categories 'sports ball' and 'ball' only and adapt
    their ID's to match the ones in the dataset."""
    df = pd.DataFrame(results)
    if "sports ball" not in df["name"].unique():
        logger.warning("No 'sports ball' found in results -> skipping adaptation.")
        return results

    # Only keep detections of 'sports ball' and map to ball
    cats_gt = {cat["name"]: cat["id"] for cat in coco_gt.cats.values()}
    df = df[df["name"].isin(["sports ball"])]
    df["category_id"] = cats_gt["ball"]
    df["name"] = "ball"

    return df.to_dict("records")
