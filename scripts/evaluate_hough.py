"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from torchvision import io
from torchvision.transforms import v2

from ball_detector import aux, coco, draw

ROOT = Path(__file__).parent


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start evaluation...")

    file_benchmark = (
        aux.DATA_ROOT / f"analysis/{args.holdout.parent.stem}_benchmark.csv"
    )
    # Load dataset
    transforms = [v2.ToPILImage()]
    loader = aux.load_dataset(args.holdout, transforms=transforms, shuffle=False)

    # Run inference
    results = coco.inference_hough(loader)

    # Run COCO evaluation
    stats_coco = coco.run_eval(args.holdout, results)

    # Save samples
    file_out = Path("tmp") / "hough.jpg"
    file_out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output file: {file_out}")
    img = draw.sample_with_boxes(args.holdout, results)
    io.write_jpeg(img, str(file_out))

    # Save results
    stats = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": "hough",
        "source": "code",
    } | stats_coco
    df_stats = pd.DataFrame(stats, index=[0])
    df_stats.to_csv(
        file_benchmark, index=False, mode="a", header=not file_benchmark.exists()
    )

    # Cleanup csv file by dropping duplicates
    df_stats = pd.read_csv(file_benchmark)
    df_stats = df_stats.drop_duplicates(
        subset=["name", "source", "ap_mean", "ar_max_100"]
    ).sort_values(["ap_mean", "ar_max_100"], ascending=False)
    df_stats.to_csv(file_benchmark, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        type=Path,
        default=aux.DATA_ROOT / "datasets/accurate-balls/holdout.coco.json",
    )
    main(parser.parse_args())
