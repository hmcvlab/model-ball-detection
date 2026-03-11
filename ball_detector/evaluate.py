"""
Created on 2026-03-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import io, models, ops, utils
from tqdm import tqdm

from ball_detector import aux, model

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
    elif args.yolo_model:
        ai_models = [
            model.load_from_torchhub("ultralytics/yolov5", args.yolo_model, args.device)
        ]
    else:
        raise ValueError("Either --file-model or --default-model must be set.")

    file_benchmark = (
        aux.DATA_ROOT / f"analysis/{args.holdout.parent.stem}_benchmark.csv"
    )
    for model_data in ai_models:
        # Load dataset
        transforms = model_data.transforms
        loader = aux.load_dataset(args.holdout, transforms=transforms, shuffle=False)

        # Run inference
        try:
            if args.yolo_model:
                results = _inference_yolo(model_data, loader)
            else:
                results = _inference_torch(model_data, loader)
        except ValueError as e:
            logger.error(e)
            continue

        # Add name to results
        df = pd.DataFrame(results)
        df["name"] = df["category_id"].map(model_data.cats)
        results = df.to_dict("records")

        # Run COCO evaluation
        coco_gt = COCO(args.holdout)
        results = _adapt_results_to_coco(results, coco_gt)
        stats_coco = run_coco(coco_gt, results)

        # Save samples
        file_out = Path("tmp") / f"{model_data.name}-{model_data.source}.jpg"
        file_out.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output file: {file_out}")
        img = sample_with_draw_boxes(args.holdout, results)
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


def _adapt_results_to_coco(results: list[dict], coco_gt: COCO) -> list[dict]:
    """The categories in the model might be different from the ones in the COCO dataset.
    Hence we want to select certain categories 'sports ball' and 'ball' only and adapt
    their ID's to match the ones in the dataset."""
    df = pd.DataFrame(results)
    if "sports ball" not in df["name"].unique():
        logger.warning("No 'sports ball' found in results -> skipping adaptation.")
        return results

    # Only keep detections of 'sports ball' and map to ball
    cats_gt = {cat["name"]: cat["id"] for cat in coco_gt.cats.values()}
    df = df[df["name"].isin(["sports ball"])]
    df["category_id"] = cats_gt["ball"]
    df["name"] = "ball"

    return df.to_dict("records")


def _inference_yolo(
    model_data: model.ModelData, loader: torch.utils.data.DataLoader
) -> list:
    """Runs COCO evaluation and returns results in coco compatible format."""
    ai_model = model_data.ai_model
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        with torch.no_grad():
            outputs = ai_model(images)

        for out, target in zip(outputs.xyxy, targets):
            boxes = ops.box_convert(out[:, :4], in_fmt="xyxy", out_fmt="xywh")
            for row, box in zip(out, boxes):
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "bbox": box.tolist(),
                        "score": float(row[4]),
                        "category_id": int(row[5]),
                    }
                )

    return results


def _inference_torch(
    model_data: model.ModelData, loader: torch.utils.data.DataLoader
) -> list:
    """Runs COCO evaluation and prints results to stdout.

    Args:
        model: A PyTorch instance segmentation model.
        data_loader: A PyTorch data loader for the COCO dataset.
    """
    ai_model = model_data.ai_model
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):

        images, target = aux.to_device(images, targets, model_data.device)
        with torch.no_grad():
            outputs = ai_model(images)

        for out, target in zip(outputs, targets):
            # Convert boxes using torch
            out["boxes"] = ops.box_convert(out["boxes"], in_fmt="xyxy", out_fmt="xywh")

            outputs = {k: v.detach().cpu() for k, v in out.items()}
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                results.append(
                    {
                        "image_id": int(target["image_id"]),
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score),
                    }
                )

    return results


def run_coco(coco_gt: COCO, results: list) -> dict:
    """Runs COCO evaluation and prints results to stdout."""
    if len(results) == 0:
        logger.warning("No results to evaluate.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.useCats = 0
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Convert stats to dict
    coco_eval.stats = coco_eval.stats.round(3)
    return {
        "ap_mean": coco_eval.stats[0],
        "ap_50": coco_eval.stats[1],
        "ap_75": coco_eval.stats[2],
        "ap_small": coco_eval.stats[3],
        "ap_medium": coco_eval.stats[4],
        "ap_large": coco_eval.stats[5],
        "ar_max_1": coco_eval.stats[6],
        "ar_max_10": coco_eval.stats[7],
        "ar_max_100": coco_eval.stats[8],
        "ar_small": coco_eval.stats[9],
        "ar_medium": coco_eval.stats[10],
        "ar_large": coco_eval.stats[11],
    }


def sample_with_draw_boxes(
    file_gt: Path, results: list[dict], image_id: int = -1
) -> torch.Tensor:
    """Draw results into image tensor."""
    coco_gt = COCO(file_gt)
    df = pd.DataFrame(results)
    all_colors = aux.colors(int(df["category_id"].max()) + 1)
    while image_id not in df["image_id"].unique():
        image_id = random.choice(list(coco_gt.imgs.keys()))

    # Load image
    filename = file_gt.parent / coco_gt.imgs[image_id]["file_name"]
    img = io.read_image(filename, io.ImageReadMode.RGB)

    # Extract ground truth boxes
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    df_gt = pd.DataFrame(coco_gt.loadAnns(ann_ids))
    df_gt["name"] = df_gt["category_id"].map(coco_gt.cats)
    df_gt["colors"] = "green"
    img = _draw_object(img, df_gt)

    # Extract boxes and labels from results
    df = df[df["image_id"] == image_id]
    df_det = df[df["score"] >= min(0.5, df["score"].max())].copy()
    df_det["colors"] = df_det["category_id"].map(all_colors)
    return _draw_object(img, df_det)


def _draw_object(img: torch.Tensor, df: pd.DataFrame):
    """Draw all objects stored in dataframe into image tensor."""
    colors = df["colors"].tolist()
    boxes = torch.Tensor(df["bbox"].tolist())
    boxes = ops.box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")

    # Extract labels with cat name and score
    if "score" in df.columns:
        labels = [
            f"{name}: {score:.2f}" for name, score in zip(df["name"], df["score"])
        ]
    else:
        labels = [f"{name}" for name in df["name"]]

    return utils.draw_bounding_boxes(
        img, boxes, labels, width=2, colors=colors, fill_labels=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        type=Path,
        default=aux.DATA_ROOT / "datasets/accurate-balls/holdout.coco.json",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-model", type=Path)
    group.add_argument(
        "--torch-model",
        choices=models.list_models(models.detection),
    )
    group.add_argument("--yolo-model", choices=torch.hub.list("ultralytics/yolov5"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

    main(parser.parse_args())
