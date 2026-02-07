#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# -----------------------
# -----------------------
# CONFIG
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "DATA" / "raw"
PROCESSED_DIR = BASE_DIR / "DATA" / "processed"
OUTPUTS_DIR = BASE_DIR / "DATA" / "outputs"

CSV_OUT = OUTPUTS_DIR / "leaf_fluorescence_results.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (inclusive)
ILLUM_RED_MIN = 5
LEAF_RED_MAX = 120
FLUOR_BLUE_MIN = 40

# Scale adjustment for masks (to reduce processing time)
ILLUM_SCALE = 0.25   # 0.5 or 0.25; 0.25 is usually much faster

# Scale calibration: diameter of illuminated region in cm
# Measure the diameter of the full illuminated circle in cm on a ruler image, then set ILLUM_DIAMETER_CM to the true physical diameter in cm. This allows you to convert pixel measurements to real-world units if needed.
ILLUM_DIAMETER_CM = 18.35  # <-- set this from your ruler image (true physical diameter of the illuminated circle)


# Morphology / filtering
ILLUM_SHRINK_PX = 150  # large shrink to ensure leaf ROIs are well within illumination (tune if needed)
MIN_LEAF_AREA_PX = 100_000
MIN_FLUOR_BLOB_AREA_PX = 20

OUT_SUFFIX = "_processed.png"

print("analyse_images.py started")

print("BASE_DIR:", BASE_DIR)
print("RAW_DIR:", RAW_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("CSV_OUT:", CSV_OUT)

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

images = sorted({
    p.resolve() for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    for p in RAW_DIR.glob(ext)
})

print(f"Found {len(images)} images")

# -----------------------
# HELPERS
# -----------------------
def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def largest_connected_component(mask01: np.ndarray) -> np.ndarray:
    """mask01 is 0/1 uint8. Returns 0/1 mask of largest component."""
    if mask01.max() == 0:
        return mask01.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if n <= 1:
        return mask01.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)

def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a binary mask.
    mask01: uint8 0/1
    returns: uint8 0/1
    """
    mask255 = (mask01 * 255).astype(np.uint8)

    h, w = mask255.shape
    flood = mask255.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from (0,0) assuming it's background
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    # Invert flood-filled image to get holes, then OR with original
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask255, holes)

    return (filled > 0).astype(np.uint8)


def shrink_mask_erode(mask01: np.ndarray, shrink_px: int) -> np.ndarray:
    """
    Shrink uniformly using cv2.erode with a small elliptical kernel and N iterations.
    mask01: uint8 0/1
    """
    if shrink_px <= 0:
        return mask01.copy()

    mask255 = (mask01 * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask255, kernel, iterations=shrink_px)
    return (eroded > 0).astype(np.uint8)

# def shrink_mask_by_px(mask01: np.ndarray, shrink_px: int) -> np.ndarray:
#     """
#     Shrink binary mask uniformly by N pixels using cv2.erode.
#     mask01 must be 0/1 uint8.
#     """
#     if shrink_px <= 0:
#         return mask01.copy()

#     mask255 = (mask01 * 255).astype(np.uint8)

#     # Small circular-ish kernel
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#     eroded = cv2.erode(mask255, kernel, iterations=shrink_px)

#     return (eroded > 0).astype(np.uint8)


# def max_feret_diameter_px(mask01: np.ndarray, max_points: int = 600) -> float:
#     """Max Feret via convex hull point-pair max distance (on boundary)."""
#     cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not cnts:
#         return 0.0
#     cnt = max(cnts, key=cv2.contourArea)
#     hull = cv2.convexHull(cnt)
#     pts = hull.reshape(-1, 2).astype(np.float32)
#     if pts.shape[0] == 0:
#         return 0.0
#     if pts.shape[0] > max_points:
#         idx = np.linspace(0, pts.shape[0]-1, max_points).astype(int)
#         pts = pts[idx]
#     diff = pts[:, None, :] - pts[None, :, :]
#     d2 = (diff ** 2).sum(axis=2)
#     return float(np.sqrt(d2.max()))

def connected_components(mask01: np.ndarray):
    return cv2.connectedComponentsWithStats(mask01, connectivity=8)

def draw_contours(img_bgr: np.ndarray, mask01: np.ndarray, color_bgr, thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(out, cnts, -1, color_bgr, thickness)
    return out

def mask_centroid_xy(mask01: np.ndarray) -> tuple[int, int] | None:
    """
    Centroid of a binary mask (0/1). Returns (x, y) in pixel coords, or None if empty.
    """
    m = mask01.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

# -----------------------
# MAIN
# -----------------------
def analyse_image(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {path}")

    b = img_bgr[:, :, 0]
    r = img_bgr[:, :, 2]

    # 1) Illumination ROI (UN-ERODED) from Red >= 5
    # --- Illumination mask at reduced scale (for speed) ---
    h, w = r.shape[:2]
    scale = float(ILLUM_SCALE)

    r_small = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    illum0_small = (r_small >= ILLUM_RED_MIN).astype(np.uint8)
    illum_small = largest_connected_component(illum0_small)

    # Diameter for zoom QC on the UN-SHRUNK mask (computed on small mask, scaled to full-res px)
    cnts, _ = cv2.findContours(illum_small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        illum_diam = float(max(bw, bh)) / scale
    else:
        illum_diam = 0.0

    # Per-image scale (handles zoom changes)
    px_per_cm = illum_diam / ILLUM_DIAMETER_CM if ILLUM_DIAMETER_CM > 0 else np.nan
    cm2_per_px = (1.0 / (px_per_cm ** 2)) if px_per_cm and px_per_cm > 0 else np.nan

    # Smooth + fill holes on the small mask
    k_close = max(3, int(round(31 * scale)) | 1)  # odd, >=3
    illum_small = cv2.morphologyEx(
        illum_small, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    )
    illum_small = fill_holes(illum_small)

    # Shrink on the small mask
    shrink_small = max(1, int(round(ILLUM_SHRINK_PX * scale)))
    illum_shrunk_small = shrink_mask_erode(illum_small, shrink_small)

    # Resize masks back to full resolution for downstream steps + overlays
    illum = cv2.resize(illum_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    illum_shrunk = cv2.resize(illum_shrunk_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # 2) Leaf ROIs within shrunk illumination: Red <= 120
    leaf0 = ((r <= LEAF_RED_MAX) & (illum_shrunk > 0)).astype(np.uint8)

    # Light cleanup similar to ImageJ open/close (tune if needed)
    leaf0 = cv2.morphologyEx(leaf0, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    leaf0 = cv2.morphologyEx(leaf0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n, labels, stats, _ = connected_components(leaf0)
    leaf_masks = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= MIN_LEAF_AREA_PX:
            m = (labels == lab).astype(np.uint8)
            leaf_masks.append((lab, m, area))

    total_leaves = len(leaf_masks)

    # Visual base: Blue channel grayscale -> BGR
    vis = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

    rows = []
    if total_leaves == 0:
        rows.append(dict(
            filename=path.name,
            illumination_diameter_px=illum_diam,
            total_leaves=0,
            leaf_id=0,
            leaf_area_px=0,
            fluorescence_area_px=0,
            fluorescence_blob_count=0,
            leaf_coverage_pct=0.0
        ))
    else:
        for i, (_lab, leaf_m, leaf_area) in enumerate(leaf_masks, start=1):
            # 3) fluorescence within leaf: Blue >= 35
            fluor0 = ((b >= FLUOR_BLUE_MIN) & (leaf_m > 0)).astype(np.uint8)

            # filter tiny fluorescence blobs
            nf, flab, fstats, _ = connected_components(fluor0)

            # areas for labels 1..nf-1 (skip background)
            areas = fstats[1:, cv2.CC_STAT_AREA]
            keep = areas >= MIN_FLUOR_BLOB_AREA_PX

            blob_count = int(keep.sum())
            fluor_area = int(areas[keep].sum())

            # Build a lookup table for labels to keep (fast)
            lut = np.zeros(nf, dtype=bool)
            # labels kept are +1 because areas starts at label 1
            lut[1:][keep] = True

            fluor_clean = lut[flab].astype(np.uint8)


            coverage = (100.0 * fluor_area / leaf_area) if leaf_area > 0 else 0.0

            leaf_area_cm2 = leaf_area * cm2_per_px if np.isfinite(cm2_per_px) else np.nan
            fluor_area_cm2 = fluor_area * cm2_per_px if np.isfinite(cm2_per_px) else np.nan


            # fill fluorescence in red
            vis[fluor_clean > 0] = (0, 0, 255)

            rows.append(dict(
                filename=path.name,
                illumination_diameter_px=illum_diam,
                illumination_diameter_cm=ILLUM_DIAMETER_CM,
                px_per_cm=px_per_cm,
                total_leaves=total_leaves,
                leaf_id=i,
                leaf_area_px=leaf_area,
                leaf_area_cm2=leaf_area_cm2,
                fluorescence_area_px=fluor_area,
                fluorescence_area_cm2=fluor_area_cm2,
                fluorescence_blob_count=blob_count,
                leaf_coverage_pct=coverage
            ))


    # 4/5) overlays: illum outline yellow, leaf outlines green
    # Pre-shrink illumination (cyan)
    vis = draw_contours(vis, illum, (255, 255, 0), thickness=2)

    # Post-shrink illumination (yellow)
    vis = draw_contours(vis, illum_shrunk, (0, 255, 255), thickness=3)

    for leaf_id, (_lab, leaf_m, _area) in enumerate(leaf_masks, start=1):
        vis = draw_contours(vis, leaf_m, (0, 255, 0), thickness=2)  # green outline

        c = mask_centroid_xy(leaf_m)
        if c is not None:
            x, y = c
            # draw a small black shadow for readability, then white text
            cv2.putText(vis, str(leaf_id), (x - 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(leaf_id), (x - 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


    return rows, vis

def main():
    ensure_dirs()
    imgs = sorted({
    p.resolve() for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
    for p in RAW_DIR.glob(ext)
    })
    all_rows = []

    for p in imgs:
        try:
            rows, vis = analyse_image(p)
            all_rows.extend(rows)
            out_img = PROCESSED_DIR / f"{p.stem}{OUT_SUFFIX}"
            cv2.imwrite(str(out_img), vis)
            print(f"Processed {p.name} -> {out_img}")
        except Exception as e:
            all_rows.append(dict(
                filename=p.name,
                illumination_diameter_px=float("nan"),
                total_leaves=0,
                leaf_id=0,
                leaf_area_px=0,
                fluorescence_area_px=0,
                fluorescence_blob_count=0,
                leaf_coverage_pct=0.0,
                error=str(e)
            ))
            print(f"ERROR {p.name}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"Wrote {CSV_OUT} ({len(df)} rows)")

if __name__ == "__main__":
    main()
