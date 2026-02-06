#!/usr/bin/env python3
"""
Leaf segmentation v0 (simple + robust)

Pipeline per image:
1) Read image
2) Detect illuminated ROI from Red channel (R > 100)
   - fill holes to make ROI solid
   - keep largest component
   - measure ROI diameter (enclosing circle + equivalent diameter from area)
3) Segment leaves inside ROI from Red channel (R <= 100)
   - clean with dilate then erode
   - connected components + area filter (>= 100,000 px)
4) Save CSV + QC masks + overlay

Requires: opencv-python, numpy, pandas
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class Settings:
    red_thresh: int = 100                 # illuminated: R > 100; leaves: R <= 100
    roi_close_kernel: int = 31            # helps join edges + smooth ROI
    roi_close_iters: int = 2

    leaf_min_area_px: int = 100_000       # user-specified for now
    leaf_morph_kernel: int = 9            # for dilate/erode smoothing
    leaf_dilate_iters: int = 2
    leaf_erode_iters: int = 2


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def fill_holes(binary_255: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask (uint8 0/255).
    """
    h, w = binary_255.shape
    flood = binary_255.copy()

    # Flood fill from border on inverted mask to find holes
    inv = cv2.bitwise_not(flood)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, mask, seedPoint=(0, 0), newVal=255)

    # Holes are the parts NOT reached by flood fill (invert back)
    inv_filled = cv2.bitwise_not(inv)
    filled = cv2.bitwise_or(binary_255, inv_filled)
    return filled


def keep_largest_component(binary_255: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary image (uint8 0/255).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_255, connectivity=8)
    if num <= 1:
        return binary_255

    # component 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_255)
    out[labels == idx] = 255
    return out


def detect_roi_from_red(img_bgr: np.ndarray, s: Settings) -> Tuple[np.ndarray, Tuple[int, int, int], dict]:
    """
    ROI mask from Red channel: illuminated pixels are R > red_thresh.
    Returns:
      roi_mask_255, (cx, cy, r), metrics dict
    """
    red = img_bgr[:, :, 2]

    roi = np.zeros_like(red, dtype=np.uint8)
    roi[red > s.red_thresh] = 255

    # Close to smooth and join gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.roi_close_kernel, s.roi_close_kernel))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=s.roi_close_iters)

    # Fill holes (critical because leaves can look "dark" like background)
    roi = fill_holes(roi)

    # Keep largest component (handles noise + multiple bright regions)
    roi = keep_largest_component(roi)

    # Compute circle metrics from contour
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No illuminated ROI detected using Red threshold. Try adjusting red_thresh.")

    c = max(contours, key=cv2.contourArea)
    (cx_f, cy_f), r_f = cv2.minEnclosingCircle(c)
    cx, cy, r = int(round(cx_f)), int(round(cy_f)), int(round(r_f))

    roi_area_px = int(np.count_nonzero(roi))
    roi_diameter_px = float(2.0 * r)
    roi_diameter_eq_px = float(np.sqrt(4.0 * roi_area_px / np.pi))

    metrics = {
        "roi_area_px": roi_area_px,
        "roi_radius_px": float(r),
        "roi_diameter_px": roi_diameter_px,
        "roi_diameter_eq_px": roi_diameter_eq_px,
        "roi_cx_px": int(cx),
        "roi_cy_px": int(cy),
    }

    return roi, (cx, cy, r), metrics


def segment_leaves_from_red(img_bgr: np.ndarray, roi_mask_255: np.ndarray, s: Settings) -> Tuple[np.ndarray, int]:
    """
    Leaves: inside ROI, Red <= red_thresh. Returns:
      leaf_mask_255 (all kept leaves merged), n_leaves_kept
    """
    red = img_bgr[:, :, 2]

    leaf = np.zeros_like(red, dtype=np.uint8)
    leaf[(roi_mask_255 > 0) & (red <= s.red_thresh)] = 255

    # Dilate then erode (ImageJ-ish smoothing)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.leaf_morph_kernel, s.leaf_morph_kernel))
    leaf = cv2.dilate(leaf, k, iterations=s.leaf_dilate_iters)
    leaf = cv2.erode(leaf, k, iterations=s.leaf_erode_iters)

    # Connected components and area filter
    num, labels, stats, _ = cv2.connectedComponentsWithStats(leaf, connectivity=8)

    out = np.zeros_like(leaf)
    kept = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= s.leaf_min_area_px:
            out[labels == i] = 255
            kept += 1

    return out, kept


def make_overlay(img_bgr: np.ndarray, roi_circle: Tuple[int, int, int], roi_mask_255: np.ndarray, leaf_mask_255: np.ndarray) -> np.ndarray:
    """
    Overlay:
      - ROI circle in green
      - ROI edge in light green
      - leaf edges in white
    """
    out = img_bgr.copy()
    cx, cy, r = roi_circle

    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)

    # ROI edge
    roi_edges = cv2.Canny(roi_mask_255, 50, 150)
    out[roi_edges > 0] = (0, 255, 100)

    # Leaf edges
    leaf_edges = cv2.Canny(leaf_mask_255, 50, 150)
    out[leaf_edges > 0] = (255, 255, 255)

    return out


def process_one(path: Path, out_dir: Path, s: Settings) -> dict:
    img = read_image(path)

    roi_mask, roi_circle, roi_metrics = detect_roi_from_red(img, s)
    leaf_mask, n_leaves = segment_leaves_from_red(img, roi_mask, s)

    qc_masks = out_dir / "QC_masks"
    qc_overlays = out_dir / "QC_overlays"
    qc_masks.mkdir(parents=True, exist_ok=True)
    qc_overlays.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    cv2.imwrite(str(qc_masks / f"{stem}_roi.png"), roi_mask)
    cv2.imwrite(str(qc_masks / f"{stem}_leaves.png"), leaf_mask)

    overlay = make_overlay(img, roi_circle, roi_mask, leaf_mask)
    cv2.imwrite(str(qc_overlays / f"{stem}_overlay.png"), overlay)

    rec = {
        "image": path.name,
        "n_leaves_kept": int(n_leaves),
        **roi_metrics,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder containing images")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--glob", default="*.JPG", help="Glob pattern (default: *.JPG)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = Settings()

    paths = sorted(in_dir.rglob(args.glob))
    if not paths:
        raise SystemExit(f"No images found in {in_dir} with glob {args.glob}")

    rows = []
    for p in paths:
        try:
            rec = process_one(p, out_dir, s)
            rows.append(rec)
            print(f"Processed {p.name}: leaves={rec['n_leaves_kept']} roi_d_eq={rec['roi_diameter_eq_px']:.1f}px")
        except Exception as e:
            print(f"FAILED {p.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "leaf_results_v0.csv", index=False)
        print(f"Saved: {out_dir / 'leaf_results_v0.csv'}")


if __name__ == "__main__":
    main()
