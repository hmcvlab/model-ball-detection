"""
Created on 2026-03-11
Copyright (c) 2026 Munich University of Applied Sciences

Script to automatically convert coco-json annotations into ultralytics format and train.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def main(args: argparse.Namespace):
    """Entrypoint"""

    file_ndjson = args.dir_dataset / "annotations.ndjson"
    if not file_ndjson.exists():
        coco2ndjson(args.dir_dataset, file_ndjson)

    model = YOLO(args.model)
    model.train(data=file_ndjson, epochs=10, batch=16)


def coco2ndjson(dir_input: Path, file_out: Path):
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
    file_train = dir_input / "train.coco.json"
    file_val = dir_input / "valid.coco.json"
    file_holdout = dir_input / "holdout.coco.json"

    ndjson_data = []
    is_first = True
    # Process each split
    for split_name, json_file in tqdm(
        [
            ("train", file_train),
            ("val", file_val),
            ("test", file_holdout),
        ],
        desc="Convert dataset",
    ):

        # Load COCO data
        with open(json_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Convert to DataFrames
        df_imgs = pd.DataFrame(coco_data["images"]).set_index("id")
        df_annos = pd.DataFrame(coco_data["annotations"])

        if is_first:
            is_first = False
            info = coco_data["info"]
            ndjson_data.append(
                {
                    "type": "dataset",
                    "task": "detect",
                    "name": dir_input.name,
                    "description": info["description"],
                    "version": info["version"],
                    "created_at": info["date_created"],
                    "updated_at": str(datetime.now().isoformat()),
                    "class_names": {
                        str(c["id"]): c["name"] for c in coco_data["categories"]
                    },
                }
            )

        for img_id, img_row in df_imgs.iterrows():
            df4img = df_annos[df_annos["image_id"] == img_id]

            # boxes: category_id, x, y, w, h
            cats = df4img["category_id"].to_list()
            boxes = np.array(df4img["bbox"].to_list())
            annos = np.column_stack((cats, boxes)).tolist()
            img_data = {
                "type": "image",
                "file": img_row["file_name"],
                "width": img_row["width"],
                "height": img_row["height"],
                "split": split_name,
                "annotations": {"boxes": annos},
            }

            ndjson_data.append(img_data)

    # Save file
    with open(file_out, "w", encoding="utf-8") as f:
        for img_data in ndjson_data:
            f.write(json.dumps(img_data) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov5s.pt", "yolov5m", "yolov5l"])
    parser.add_argument("--dir-dataset", type=Path, help="Input path")
    main(parser.parse_args())
