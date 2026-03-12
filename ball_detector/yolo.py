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
import yaml
from tqdm import tqdm
from ultralytics import YOLO


def main(args: argparse.Namespace):
    """Entrypoint"""

    file_coco = args.dir_dataset / "yolo/coco.yaml"
    if not file_coco.exists():
        coco2yolo(args.dir_dataset, file_coco)

    model = YOLO(args.model)
    model.train(data=file_coco, epochs=10, batch=4)


def coco2yolo(dir_input: Path, file_out: Path):
    """Merges coco.json files into a single ndjson file:
    {
        "type": "image",
        "file": "image1.jpg",
        "url": "https://www.url.com/path/to/image1.jpg",
        "width": 640,
        "height": 480,
        "split": "train",
        "annotations": {
            "boxes": [
                [0, 0.525, 0.376, 0.284, 0.418],
                [1, 0.735, 0.298, 0.193, 0.337]
            ]
        }
    }
    """
    dir_root = file_out.parent
    dir_images = dir_root / "images"
    dir_labels = dir_root / "labels"
    dir_root.mkdir(exist_ok=True)
    dir_images.mkdir(exist_ok=True)
    dir_labels.mkdir(exist_ok=True)

    # Process each split
    coco_data = {}
    names_split = ["train", "valid", "holdout"]
    for split in tqdm(names_split, desc="Convert dataset"):

        # Load COCO data
        file = dir_input / f"{split}.coco.json"
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
            boxes = np.array(df4img["bbox"].to_list())
            annos = np.column_stack((cats, boxes)).tolist()

            # Copy image
            file_img = dir_input / img_row["file_name"]
            shutil.copy2(file_img, dir_images_name)

            # Generate txt-file with annotations
            file_txt = dir_labels_name / file_img.with_suffix(".txt")
            np.savetxt(
                file_txt,
                annos,
                fmt="%d %.6f %.6f %.6f %.6f",
                delimiter=" ",
                newline="\n",
            )

    # Export config
    data = {"path": str(dir_root)}
    for split in names_split:
        data[split] = dir_labels.joinpath(split).relative_to(dir_root)
    data["names"] = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    with open(file_out, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov5s.pt", "yolov5m", "yolov5l"])
    parser.add_argument("--dir-dataset", type=Path, help="Input path")
    main(parser.parse_args())
