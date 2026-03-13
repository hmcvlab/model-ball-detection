"""
Created on 2026-03-13
Copyright (c) 2026 robominds GmbH

Module containing functions for visualizations.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torchvision import io, ops, utils


def colors(n: int):
    """Return colors fitting to class map."""
    cls_values = np.array(list(range(n)), dtype=np.uint8)
    cls_values = cv2.normalize(cls_values, None, 0, 255, cv2.NORM_MINMAX)
    all_colors = cv2.applyColorMap(cls_values, cv2.COLORMAP_RAINBOW).squeeze()
    return {idx: tuple(map(int, color)) for idx, color in enumerate(all_colors)}


def draw_sample_with_boxes(
    file_gt: Path, results: list[dict], image_id: int = -1
) -> torch.Tensor:
    """Draw results into image tensor."""
    coco_gt = COCO(file_gt)
    df = pd.DataFrame(results)
    all_colors = colors(int(df["category_id"].max()) + 1)
    while image_id not in df["image_id"].unique():
        image_id = random.choice(list(coco_gt.imgs.keys()))

    # Load image
    filename = file_gt.parent / coco_gt.imgs[image_id]["file_name"]
    img = io.read_image(filename, io.ImageReadMode.RGB)

    # Extract ground truth boxes
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    df_gt = pd.DataFrame(coco_gt.loadAnns(ann_ids))
    df_gt["name"] = df_gt["category_id"].map(coco_gt.cats)
    df_gt["colors"] = "green"
    img = draw_object(img, df_gt)

    # Extract boxes and labels from results
    df = df[df["image_id"] == image_id]
    df_det = df[df["score"] >= min(0.5, df["score"].max())].copy()
    df_det["colors"] = df_det["category_id"].map(all_colors)
    return draw_object(img, df_det)


def draw_object(img: torch.Tensor, df: pd.DataFrame):
    """Draw all objects stored in dataframe into image tensor."""
    box_colors = df["colors"].tolist()
    boxes = torch.Tensor(df["bbox"].tolist())
    boxes = ops.box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")

    # Extract labels with cat name and score
    if "score" in df.columns:
        labels = [
            f"{name}: {score:.2f}" for name, score in zip(df["name"], df["score"])
        ]
    else:
        labels = [f"{name}" for name in df["name"]]

    return utils.draw_bounding_boxes(
        img, boxes, labels, width=2, colors=box_colors, fill_labels=True
    )
