"""
Created on 2026-03-11
Copyright (c) 2026 Munich University of Applied Sciences

Script to automatically convert coco-json annotations into ultralytics format and train.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import io, ops, utils
from tqdm import tqdm
from ultralytics import YOLO


def main(args: argparse.Namespace):
    """Entrypoint"""

    file_coco = args.dir_dataset.parent / f"{args.dir_dataset.name}-yolo/coco.yaml"
    if not file_coco.exists():
        coco2yolo(args.dir_dataset, file_coco)

    model = YOLO(f"tmp/models/{args.model}", task="detect")
    model.train(data=file_coco, epochs=10, batch=4)


def coco2yolo(dir_input: Path, file_out: Path):
    """Convert coco.json files into ultralytics format
    root:
        images/
            train/
            val/
            holdout/
        labels/
            train/
            val/
            holdout/
    """
    dir_root = file_out.parent
    dir_images = dir_root / "images"
    dir_labels = dir_root / "labels"
    dir_root.mkdir(exist_ok=True)
    dir_images.mkdir(exist_ok=True)
    dir_labels.mkdir(exist_ok=True)

    # Process each split
    coco_data = {}
    names_split = ["train", "val", "holdout"]
    for split in tqdm(names_split, desc="Convert dataset"):

        # Rename split to yolo format valid -> val
        file_stem = "valid" if split == "val" else split

        # Load COCO data
        file = dir_input / f"{file_stem}.coco.json"
        with open(file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Convert to DataFrames
        df_imgs = pd.DataFrame(coco_data["images"]).set_index("id")
        df_annos = pd.DataFrame(coco_data["annotations"])

        # Create folders
        dir_images_name = dir_images / split
        dir_labels_name = dir_labels / split
        dir_images_name.mkdir(exist_ok=True)
        dir_labels_name.mkdir(exist_ok=True)

        for img_id, img_row in df_imgs.iterrows():
            df4img = df_annos[df_annos["image_id"] == img_id]

            # boxes: category_id, x, y, w, h
            cats = df4img["category_id"].to_list()
            boxes = _boxes2cxcywh_normalized(df4img["bbox"].to_list(), img_row)
            annos = np.column_stack((cats, boxes)).tolist()

            # Copy image
            file_img = dir_input / img_row["file_name"]
            shutil.copy2(file_img, dir_images_name)

            # Generate txt-file with annotations
            file_txt = dir_labels_name / f"{file_img.stem}.txt"
            np.savetxt(
                file_txt,
                annos,
                fmt="%d %.6f %.6f %.6f %.6f",
                delimiter=" ",
                newline="\n",
            )

            # Draw sample
            # if img_id == 1:
            #     img = io.read_image(file_img)
            #     boxes = ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            #     img = utils.draw_bounding_boxes(img, boxes)
            #     file_sample = Path(f"tmp/samples/yolo/{file_img.stem}.png")
            #     file_sample.parent.mkdir(exist_ok=True, parents=True)
            #     io.write_png(img, file_sample)

    # Export config
    data = {"path": str(dir_root)}
    for split in names_split:
        data[split] = str(dir_images.joinpath(split).relative_to(dir_root))
    data["names"] = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    with open(file_out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


def _boxes2cxcywh_normalized(boxes: list[list[float]], img: pd.Series) -> np.ndarray:
    """Convert boxes from xyxy to cxcywh normalized"""
    width, height = img["width"], img["height"]
    boxes_t = torch.Tensor(boxes)
    boxes_t = ops.box_convert(boxes_t, in_fmt="xywh", out_fmt="cxcywh")
    norm_factor = torch.Tensor([width, height, width, height])
    boxes_t = boxes_t / norm_factor
    assert boxes_t.max() <= 1
    assert boxes_t.min() >= 0
    return boxes_t.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["yolov5s.pt", "yolov5m.pt", "yolov5l.pt"],
        default="yolov5s.pt",
    )
    parser.add_argument(
        "--dir-dataset",
        type=Path,
        help="Input path",
        default="/mnt/data/datasets/accurate-balls",
    )
    main(parser.parse_args())
