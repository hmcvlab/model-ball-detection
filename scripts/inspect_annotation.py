"""Copyright (c) 2024 Munich University of Applied Sciences"""

import argparse
import pprint
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torchvision import datasets, utils
from torchvision.transforms import v2
from tqdm import tqdm

from ball_detector import draw


def main(args: argparse.Namespace) -> None:
    """Load COCO images and annotations, draw annotations, and save visualizations."""

    target_keys = ("boxes", "labels")
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = datasets.CocoDetection(args.file.parent, args.file, transforms=transforms)
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Load categories
    cats = {x["id"]: x["name"] for x in dataset._dataset.coco.cats.values()}
    logger.info("Found the following categories:\n" + pprint.pformat(cats))

    # Get colors
    all_colors = draw.colors(len(cats))

    for n, (img, target) in tqdm(enumerate(loader), desc="Visualize..."):
        # Since we have only
        img = img[0]
        boxes = target.get("boxes")[0]
        labels = target.get("labels")[0].tolist()

        colors = list(map(all_colors.get, labels))
        labels = list(map(cats.get, labels))
        tqdm.write(f"{img=}")
        img = utils.draw_bounding_boxes(
            img, boxes, labels, colors=colors, fill_labels=True
        )

        if args.type == "seg":
            masks = target.get("masks")[0].to(torch.bool)
            img = utils.draw_segmentation_masks(img, masks, colors=colors, alpha=0.5)

        file_output = Path("tmp/samples") / f"{n}-{args.file.stem}.png"
        file_output.parent.mkdir(exist_ok=True, parents=True)
        utils.save_image(img, file_output)
        tqdm.write(f"Saved {file_output}")

        if n >= args.samples:
            break


def get_colors(labels: list[str]) -> list[tuple]:
    """Create a list of colors where each color is associated with a label."""
    colormap = cv2.applyColorMap(
        np.arange(0, max(labels) + 1, dtype=np.uint8), cv2.COLORMAP_RAINBOW
    )
    colormap = colormap.squeeze().tolist()
    return [tuple(colormap[label]) for label in labels]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to COCO <name>.coco.json",
        default="/mnt/data/datasets/accurate-balls/train.coco.json",
    )
    parser.add_argument(
        "--samples", type=int, help="Number of samples to visualize", default=1
    )
    parser.add_argument("--type", choices=["bbox", "seg"], default="bbox")
    main(parser.parse_args())
