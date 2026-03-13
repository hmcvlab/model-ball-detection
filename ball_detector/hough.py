"""
Created on 2026-03-13
Copyright (c) 2026 Munich University of Applied Sciences
"""

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import scipy
from loguru import logger


@dataclass
class Settings:
    """Settings for calibration"""

    hough_gauss_kernel: int = 17
    hough_median_kernel: int = 11
    hough_threshold_lower: int = 67
    hough_threshold_upper: int = 24


def circles(img: np.ndarray, config: Settings = Settings()) -> pd.DataFrame:
    """Detect circles using hough transformation"""
    img = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = (config.hough_gauss_kernel, config.hough_gauss_kernel)
    img = cv2.medianBlur(img, config.hough_median_kernel)
    img = cv2.GaussianBlur(src=img, ksize=kernel, sigmaX=0)
    tmp_circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=config.hough_threshold_lower,
        param2=config.hough_threshold_upper,
        minRadius=5,
        maxRadius=300,
    )
    if tmp_circles is None:
        raise RuntimeWarning("No circles found! - Tune parameters.")
    tmp_circles = np.squeeze(np.uint16(np.around(tmp_circles))).reshape(-1, 3)

    circles_df = pd.DataFrame(tmp_circles, columns=["x", "y", "radius"])
    circles_df["variance"] = circles_df.apply(lambda x: pixel_variance(img, x), axis=1)
    circles_df["variance"] = circles_df["variance"] / circles_df["variance"].max() * 255
    circles_df = circles_df.sort_values(by="variance")
    circles_df["score"] = circles_df.apply(
        lambda x: normalized_pixel_entropy(img, x), axis=1
    )

    # Convert x, y, radius to bboxes xywh
    circles_df["bbox"] = circles_df.apply(
        lambda x: [
            x["x"] - x["radius"],
            x["y"] - x["radius"],
            x["radius"] * 2,
            x["radius"] * 2,
        ],
        axis=1,
    )

    # Assign labels and names
    circles_df["category_id"] = 1
    circles_df["name"] = "ball"

    return circles_df


def pixel_variance(img: np.ndarray, row: pd.Series):
    """Compute pixel variance"""
    mask = np.zeros_like(img)
    cv2.circle(mask, (row["x"], row["y"]), row["radius"], 255, -1)
    pixels = img[mask == 255]
    return np.var(pixels)


def normalized_pixel_entropy(img: np.ndarray, row: pd.Series):
    """Compute pixel entropy"""
    x = np.round(row["x"]).astype(int)
    y = np.round(row["y"]).astype(int)
    radius = np.round(row["radius"]).astype(int)
    mask = np.zeros_like(img)
    cv2.circle(mask, (x, y), radius, 255, -1)
    pixels = img[mask == 255]

    _, counts = np.unique(pixels, return_counts=True)
    entropy = scipy.stats.entropy(counts, base=2) / np.log2(np.min([256, len(pixels)]))
    if 0 > entropy > 1:
        logger.error(f"Entropy out of range: {entropy}")
    if len(pixels) < 256:
        logger.warning(f"Low pixel count: {len(pixels)}")
    return entropy
