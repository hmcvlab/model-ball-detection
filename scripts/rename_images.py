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
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from loguru import logger as log
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    """Search recursively for zip files and generate UUIDs for all images."""
    log.remove()
    log.add(sys.stdout, level="DEBUG" if args.debug else "INFO")
    log.info("Start renaming images...")

    dir_old = args.source
    dir_new = args.source.parent / f"{dir_old.name}-patched"
    dir_new.mkdir(exist_ok=True)

    for file_data in tqdm(
        dir_old.glob("*.coco.json"),
        desc="Process coco file",
        position=0,
        leave=True,
        total=3,
    ):

        # Load annotation into pandas dataframe
        with file_data.open(mode="r", encoding="utf-8") as file:
            data = json.load(file)
        df_images = pd.DataFrame(data["images"])
        log.debug(df_images)

        # Update infos
        data["info"]["version"] = "v3"
        data["info"]["year"] = 2026
        data["info"]["description"] = (
            "A dataset with images containing accurately annotated balls and few"
            " samples of cars."
        )
        data["info"]["date_modified"] = datetime.now().isoformat()
        data["info"]["contributor"] = "Valentino Behret, Simon Weber"
        data["info"]["url"] = "https://zenodo.org/records/17071583"

        # Iterate over image entries and rename image to uuid in annotation and rename
        # image
        for entry in tqdm(
            data["images"], desc="Rename images", position=1, leave=False
        ):
            old_name = dir_old / entry["file_name"]
            new_name = dir_new / f"images/{uuid4()}.jpg"
            entry["file_name"] = str(new_name.relative_to(dir_new))
            new_name.parent.mkdir(exist_ok=True)
            shutil.copy2(old_name, new_name)
            log.debug(f"Rename: {old_name} -> {new_name}")

        # Save new data file to new directory
        new_file_data = dir_new / file_data.name
        with new_file_data.open(mode="w", encoding="utf-8") as file:
            json.dump(data, file)

        log.info(f"Save new annotation file to {new_file_data}")

    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="/mnt/data/datasets/accurate-balls", type=Path
    )
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
