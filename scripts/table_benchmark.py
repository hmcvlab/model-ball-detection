"""
Created on 2026-03-14
Copyright (c) 2026 Munich University of Applied Sciences

Script to generate a table in TeX format from benchmark results.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ball_detector import aux

COLMAP = {
    "name": "Model",
    "ap_mean": r"\textbf{mAP}",
    "ap_medium": r"\textbf{AP}$^{m}$",
    "ap_large": r"\textbf{AP}$^{l}$",
    "ar_max_10": r"\textbf{AR}$^{10}$",
}
FAMILY_REPLACE = {
    "yolov5": "YOLOv5",
    "yolo11": "YOLO11",
    "yolo26": "YOLO26",
    "hough": "Hough",
    "detr": "DETR",
    "fasterrcnn": "Faster R-CNN",
    "fcos": "FCOS",
    "retinanet": "RetinaNet",
    "ssd300": "SSD300",
    "ssdlite": "SSDLite",
}
NAME_REPLACE = {
    " fpn": "",
    " large": "",
    " v3": "",
    "320 ": "",
    " 320": "-s",
    "rt-": "RT-",
    "resnet50": "R50",
    "mobilenet": "MobileNet",
}


def _get_family(name):
    """If any of the family-substrings is inside the name, return the family, otherwise
    misc."""
    for f in [
        "yolov5",
        "yolo11",
        "yolo26",
        "hough",
        "detr",
        "fasterrcnn",
        "fcos",
        "retinanet",
        "ssd300",
        "ssdlite",
    ]:
        if f in name:
            return pd.Series([f, name.replace(f, "").strip()])
    return "misc"


def _export(df: pd.DataFrame, file: Path):
    """Export dataframe to tex file."""
    # Map column names
    df = df.rename(columns=COLMAP)

    # Find the maximum value for each column
    max_values = df.max()

    # Create a formatter function for multi-index columns
    def bold_max_multi(val, col):
        if val == max_values[col]:
            return f"\\textbf{{{val:.2f}}}"  # Format to 2 decimal places
        return f"{val:.2f}"  # Format to 2 decimal places

    # Create formatters dictionary for each column
    formatters = {col: lambda x, col=col: bold_max_multi(x, col) for col in df.columns}

    # Export to LaTeX with the formatters
    latex_table = df.to_latex(
        formatters=formatters,
        escape=False,  # Don't escape LaTeX commands
        bold_rows=False,
        multicolumn=True,  # Handle multi-index columns
        multicolumn_format="c",  # Center align the multicolumns
        multirow=True,  # If you have multi-index rows
        na_rep="-",
    )

    # Write to file
    file.parent.mkdir(exist_ok=True, parents=True)
    with open(file, "w", encoding="utf-8") as f:
        f.write(latex_table)


def main(args):
    """Entrypoint"""
    file_benchmark = aux.file_benchmark(args.file_holdout)

    # Extract information from holdout file
    with open(args.file_holdout, "r", encoding="utf-8") as f:
        data = json.load(f)
    df_anno = pd.DataFrame(data["annotations"])

    # Group by image and count number of annotations
    df_anno = df_anno[["image_id", "id"]].groupby("image_id").count()
    df_anno = df_anno.reset_index()
    df_anno = df_anno.rename(columns={"id": "n_anno"})
    print(df_anno.value_counts("n_anno"))

    # Read benchmark
    df = pd.read_csv(file_benchmark)[list(COLMAP.keys())]
    df["tuned"] = df["name"].str.contains("_00")
    df["name"] = df["name"].str.replace("_00", "")
    df["name"] = df["name"].str.replace("_", " ")
    df[["family", "name"]] = df["name"].apply(_get_family)
    df = df.sort_values(["family", "name"])
    df["family"] = df["family"].map(FAMILY_REPLACE)

    for old, new in NAME_REPLACE.items():
        df["name"] = df["name"].str.replace(old, new)

    # Split dataframes by tuned/pretrained
    df_tune_indexed = df[df["tuned"]].set_index(["family", "name"])
    df_pretrained_indexed = df[~df["tuned"]].set_index(["family", "name"])
    df_tune_indexed = df_tune_indexed.drop("tuned", axis=1)
    df_pretrained_indexed = df_pretrained_indexed.drop("tuned", axis=1)

    # Create multi-index columns for each dataframe
    for col in df_tune_indexed.columns:
        df_tune_indexed[col] = df_tune_indexed[col]
    for col in df_pretrained_indexed.columns:
        df_pretrained_indexed[col] = df_pretrained_indexed[col]

    # Create column multi-index
    df_tune_indexed.columns = pd.MultiIndex.from_product(
        [["tune"], df_tune_indexed.columns]
    )
    df_pretrained_indexed.columns = pd.MultiIndex.from_product(
        [["pretrained"], df_pretrained_indexed.columns]
    )

    # Concatenate horizontally (axis=1)
    combined = pd.concat([df_tune_indexed, df_pretrained_indexed], axis=1)

    print(combined)

    # Export to latex and mark highest value on each column bold
    _export(combined, Path("tmp/tables/benchmark.tex"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-holdout",
        type=str,
        default=aux.DATA_ROOT / "datasets/accurate-balls/holdout.coco.json",
    )
    main(parser.parse_args())
