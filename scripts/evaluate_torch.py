"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from torchvision import io, models

from ball_detector import aux, coco, draw, model

ROOT = Path(__file__).parent


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start evaluation...")

    # Load model
    if args.file_model:
        ai_models = [
            model.load_from_file(file, args.device)
            for file in args.file_model.parent.glob(args.file_model.name)
        ]
    elif args.torch_model:
        ai_models = [model.load_from_torchvision(args.torch_model, args.device)]
    else:
        raise ValueError("Either --file-model or --default-model must be set.")

    file_benchmark = aux.file_benchmark(args.holdout)
    for model_data in ai_models:
        # Load dataset
        transforms = model_data.transforms
        loader = aux.load_dataset(args.holdout, transforms=transforms, shuffle=False)

        # Run inference
        try:
            results = coco.inference_torch(model_data, loader)
        except RuntimeError:
            # Try again on cpu
            logger.warning(f"Trying to run {model_data.name} on cpu...")
            model_data.device = "cpu"
            results = coco.inference_torch(model_data, loader)

        # Run COCO evaluation
        stats_coco = coco.run_eval(args.holdout, results)

        # Save samples
        file_out = Path("tmp") / f"{model_data.name}-{model_data.source}.jpg"
        file_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output file: {file_out}")
        img = draw.sample_with_boxes(args.holdout, results)
        io.write_jpeg(img, str(file_out))

        # Save results
        stats = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": model_data.name,
            "source": model_data.source,
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-model", type=Path)
    group.add_argument(
        "--torch-model",
        choices=models.list_models(models.detection),
    )

    main(parser.parse_args())
