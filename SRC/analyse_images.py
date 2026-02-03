#!/usr/bin/env python3
"""
Fluorescence spray coverage quantification.

For each image:
1) Detect illuminated circular ROI
2) Segment leaves within ROI
3) Compute fluorescence coverage per leaf
4) Save CSV + QC overlays

Requires: opencv-python, numpy, pandas
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd


# ------------------------- Settings ------------------------- #

@dataclass
class Settings:
    # --- Circle detection ---
    circle_thresh: int = 20          # grayscale threshold to separate black border vs illuminated circle
    circle_margin_px: int = 8        # expand fitted circle mask by this many pixels

    # --- Leaf segmentation (HSV) ---
    leaf_v_max: int = 90             # leaves are dark: keep pixels with V <= leaf_v_max
    leaf_s_min: int = 30             # avoid very low-saturation noise
    leaf_min_area_px: int = 8_000    # filter tiny components (tune per image resolution) # CONFIG FILE
    leaf_max_area_px: int = 500_000  # filter weird huge components

    # Morphology
    morph_kernel: int = 7            # odd number recommended
    morph_iters: int = 2

    # --- Fluorescence detection ---
    fluor_ratio_thresh: float = 1.25 # (G+B)/(R+1) threshold
    fluor_min_component_px: int = 20 # remove tiny specks if you want (tune)
    fluor_morph_kernel: int = 3
    fluor_morph_iters: int = 1

    # --- Debug ---
    save_debug: bool = True


# ------------------------- Core functions ------------------------- #

def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_illuminated_circle(img_bgr: np.ndarray, s: Settings) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Returns: circle_mask (uint8 0/255), (cx, cy, r)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # threshold: illuminated region is > black surround
    _, bw = cv2.threshold(gray, s.circle_thresh, 255, cv2.THRESH_BINARY)

    # clean up small holes/noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_kernel, s.morph_kernel))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=s.morph_iters)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No illuminated region detected (circle). Try lowering circle_thresh.")

    # largest contour = illuminated circle region
    c = max(contours, key=cv2.contourArea)
    (cx_f, cy_f), r_f = cv2.minEnclosingCircle(c)
    cx, cy, r = int(round(cx_f)), int(round(cy_f)), int(round(r_f)) + s.circle_margin_px

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
    return mask, (cx, cy, r)


def segment_leaves(img_bgr: np.ndarray, circle_mask: np.ndarray, s: Settings) -> np.ndarray:
    """
    Returns a labeled image (int32) where 0 = background, 1..N = leaf objects.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # candidate leaf pixels: dark (low V) and not totally desaturated
    leaf_bw = (V <= s.leaf_v_max) & (S >= s.leaf_s_min) & (circle_mask > 0)
    leaf_bw = leaf_bw.astype(np.uint8) * 255

    # morphology to fill gaps and remove pepper noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_kernel, s.morph_kernel))
    leaf_bw = cv2.morphologyEx(leaf_bw, cv2.MORPH_CLOSE, k, iterations=s.morph_iters)
    leaf_bw = cv2.morphologyEx(leaf_bw, cv2.MORPH_OPEN, k, iterations=max(1, s.morph_iters - 1))

    # connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_bw, connectivity=8)

    # filter by area and relabel sequentially
    out = np.zeros(labels.shape, dtype=np.int32)
    new_id = 1
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if s.leaf_min_area_px <= area <= s.leaf_max_area_px:
            out[labels == i] = new_id
            new_id += 1

    return out


def fluorescence_mask(img_bgr: np.ndarray, leaf_mask: np.ndarray, s: Settings) -> np.ndarray:
    """
    Compute fluorescent pixels within a single leaf mask (uint8 0/255).
    Uses ratio (G+B)/(R+1).
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    ratio = (g + b) / (r + 1.0)

    fluor = (ratio >= s.fluor_ratio_thresh) & (leaf_mask > 0)
    fluor = fluor.astype(np.uint8) * 255

    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.fluor_morph_kernel, s.fluor_morph_kernel))
    fluor = cv2.morphologyEx(fluor, cv2.MORPH_OPEN, k, iterations=s.fluor_morph_iters)

    # remove tiny connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fluor, connectivity=8)
    clean = np.zeros_like(fluor)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= s.fluor_min_component_px:
            clean[labels == i] = 255

    return clean


def make_overlay(img_bgr: np.ndarray, circle: Tuple[int, int, int], leaf_labels: np.ndarray, fluor_all: np.ndarray) -> np.ndarray:
    """
    QC overlay:
    - circle boundary in green
    - leaf boundaries in white
    - fluorescence pixels in cyan
    """
    out = img_bgr.copy()
    cx, cy, r = circle
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)

    # leaf boundaries
    leaf_bw = (leaf_labels > 0).astype(np.uint8) * 255
    edges = cv2.Canny(leaf_bw, 50, 150)
    out[edges > 0] = (255, 255, 255)

    # fluorescence highlight (cyan in BGR = (255,255,0))
    out[fluor_all > 0] = (255, 255, 0)

    return out


def process_image(path: Path, out_dir: Path, s: Settings) -> pd.DataFrame:
    img = read_image(path)
    circle_mask, circle = detect_illuminated_circle(img, s)
    leaf_labels = segment_leaves(img, circle_mask, s)
    roi_area_px = int(np.count_nonzero(circle_mask))
    cx, cy, r = circle
    roi_diameter_px = float(2.0 * r)
    roi_diameter_eq_px = float(np.sqrt(4.0 * roi_area_px / np.pi))
    

    records = []
    fluor_all = np.zeros(leaf_labels.shape, dtype=np.uint8)

    n_leaves = int(leaf_labels.max())
    for leaf_id in range(1, n_leaves + 1):
        leaf_mask = (leaf_labels == leaf_id).astype(np.uint8) * 255
        leaf_area = int(np.count_nonzero(leaf_mask))

        fluor = fluorescence_mask(img, leaf_mask, s)
        fluor_area = int(np.count_nonzero(fluor))
        fluor_all = cv2.bitwise_or(fluor_all, fluor)

        coverage = 100.0 * fluor_area / leaf_area if leaf_area > 0 else np.nan

        records.append({
            "image": path.name,
            "leaf_id": leaf_id,
            "leaf_area_px": leaf_area,
            "fluor_area_px": fluor_area,
            "coverage_pct": coverage,
            "roi_area_px": roi_area_px,
            "roi_diameter_px": roi_diameter_px,
            "roi_diameter_eq_px": roi_diameter_eq_px,
            "roi_cx_px": cx,
            "roi_cy_px": cy,
        })

    if s.save_debug:
        overlay = make_overlay(img, circle, leaf_labels, fluor_all)
        out_png = out_dir / "qc_overlays" / f"{path.stem}_overlay.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), overlay)

        # Also save intermediate masks if you want
        masks_dir = out_dir / "qc_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(masks_dir / f"{path.stem}_circle.png"), circle_mask)
        cv2.imwrite(str(masks_dir / f"{path.stem}_leaves.png"), (leaf_labels > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(masks_dir / f"{path.stem}_fluor.png"), fluor_all)

    return pd.DataFrame.from_records(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder with images (jpg/png/tif)")
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

    all_dfs = []
    for p in paths:
        try:
            df = process_image(p, out_dir, s)
            all_dfs.append(df)
            print(f"Processed {p.name}: {df.shape[0]} leaves")
        except Exception as e:
            print(f"FAILED {p.name}: {e}")

    if all_dfs:
        res = pd.concat(all_dfs, ignore_index=True)
        # One ROI diameter per image; broadcast back to all rows
        roi_per_image = (
            res[["image", "roi_diameter_eq_px"]]
            .drop_duplicates()
            .set_index("image")
        )
        median_d = float(roi_per_image["roi_diameter_eq_px"].median())
        
        roi_per_image["roi_pct_diff"] = 100.0 * (roi_per_image["roi_diameter_eq_px"] - median_d) / median_d
        # Threshold: tune; 10% is a sensible starting point
        thr = 10.0
        roi_per_image["roi_comment"] = np.where(
            np.abs(roi_per_image["roi_pct_diff"]) >= thr,
            "roi_diameter_outlier",
            ""
        )
        
        # merge back onto all leaf rows
        res = res.merge(roi_per_image.reset_index(), on="image", how="left")

        res.to_csv(out_dir / "leaf_fluorescence_results.csv", index=False)
        print(f"Saved: {out_dir / 'leaf_fluorescence_results.csv'}")


if __name__ == "__main__":
    main()
