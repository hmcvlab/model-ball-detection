"""
Created on 2026-03-10
Copyright (c) 2026 Munich University of Applied Sciences
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, Image

from ball_detector import aux


def coco_annotations(length: int, tmp_path: Path):
    """Create a random coco annotation with pre-defined length"""

    data = {"images": [], "annotations": [], "categories": [], "info": {}}
    data["info"] = {"year": 2026, "version": 1, "description": "Test annotation"}

    for idx in range(length):
        file_name = tmp_path / f"images/{idx}.png"
        width = 20
        height = 10
        image = np.ones((height, width, 3), np.uint8) * 255
        file_name.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(file_name), image)

        # Add images data
        images = {
            "id": idx,
            "file_name": str(file_name),
            "width": width,
            "height": height,
        }

        # Add annotations data
        annotations = {
            "id": idx,
            "image_id": idx,
            "bbox": [5, 4, 7, 7],
            "category_id": 1,
        }

        # Add categories data
        categories = {"id": 0, "name": "cat"}

        # Append data
        data["images"].append(images)
        data["annotations"].append(annotations)
        data["categories"].append(categories)

    file_annotation = tmp_path / "coco.json"
    with file_annotation.open("w", encoding="utf-8") as file:
        file.write(json.dumps(data))

    return data


@pytest.mark.parametrize(
    "n",
    [
        (2),
        (3),
    ],
)
def test_load_dataset(n, tmp_path):
    """Test if colors creates a dict of tuples."""
    # Arrange
    data = coco_annotations(n, tmp_path)
    transformations = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

    # Act
    loader = aux.load_dataset(tmp_path / "coco.json", transformations)
    image, target = next(iter(loader))

    # Assert
    assert len(data["images"]) == len(image) == len(target)
    assert isinstance(image[0], Image)
    assert isinstance(target[0], dict)
    assert isinstance(target[0]["boxes"], BoundingBoxes)
    assert isinstance(target[0]["labels"], torch.Tensor)
