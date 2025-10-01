"""
Created on Wed Oct 01 2025
Copyright (c) 2025 Munich University of Applied Sciences

This script generate UUIDs for all images inside each zip file in the tmp/
directory.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
from loguru import logger as log

ROOT = Path(__file__).parent


def main(args: argparse.Namespace) -> None:
    """Search recursively for zip files and generate UUIDs for all images."""
    log.remove()
    log.add(sys.stdout, level="DEBUG" if args.debug else "INFO")
    log.info("Start renaming images...")

    for folder in filter(
        lambda x: x.name in ["train", "valid", "test"], args.source.glob("*")
    ):
        log.info(f"Found directory: {folder}")
        new_folder = folder.parent / "new" / folder.name
        new_folder.mkdir(exist_ok=True, parents=True)

        # Load annotation into pandas dataframe
        file_data = next(folder.glob("*.coco.json"))
        with file_data.open(mode="r", encoding="utf-8") as file:
            data = json.load(file)
        df_images = pd.DataFrame(data["images"])
        log.debug(df_images)

        # Iterate over image entries and rename image to uuid in annotation and rename
        # image
        for entry in data["images"]:
            old_name = folder / entry["file_name"]
            new_name = new_folder / f"{uuid4()}.jpg"
            entry["file_name"] = new_name.name
            shutil.copy2(old_name, new_name)
            log.debug(f"Rename: {old_name} -> {new_name}")

        # Save new data file to new directory
        new_file_data = new_folder / file_data.name
        with new_file_data.open(mode="w", encoding="utf-8") as file:
            json.dump(data, file)

        log.info(f"Save new annotation file to {new_file_data}")

    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=ROOT / "tmp/accurate-ball-detection", type=Path
    )
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
