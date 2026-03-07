"""
Created on 2026-03-07
Copyright (c) 2026 Munich University of Applied Sciences

Script to find images recursively and patch relative path into coco.json
"""

import argparse
import json
from pathlib import Path

from loguru import logger
from pycocotools.coco import COCO
from tqdm import tqdm


def main(args: argparse.Namespace):
    """Entrypoint"""

    # Create a backup in case smoke test fails
    file_backup = args.file.with_suffix(".backup.json")
    file_backup.write_text(args.file.read_text(), encoding="utf-8")
    logger.info(f"Created backup {file_backup}")

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for image in tqdm(data["images"], desc="Patching images"):
        file_old = Path(image["file_name"])
        file_new = next(args.file.parent.rglob(file_old.name))

        if not file_new:
            raise ValueError(f"Image {file_old} not found.")

        image["file_name"] = str(file_new.relative_to(args.file.parent))

    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # Test annotations and delete backup if passed
    _smoke_test(args.file)
    if file_backup.exists():
        file_backup.unlink()


def _smoke_test(file: Path):
    """Test if annotations are valid."""
    coco_holdout = COCO(file)
    coco_holdout.info()
    logger.info("Annotations are valid!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, help="Input path")
    main(parser.parse_args())
