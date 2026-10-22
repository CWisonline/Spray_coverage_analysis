#!/usr/bin/env python3
"""
Leaf spray coverage analysis from UV-fluorescence images.

Pipeline (per image)
--------------------
1. Lit region      : the illuminated circle (B+G+R above a low threshold).
2. Background card : red-fluorescing card, found by R/B ratio > card_ratio_min.
                     Its convex hull defines the sample area.
3. Leaves          : objects inside the card hull with R/B < leaf_ratio_max.
                     Ratios are used instead of absolute levels because they
                     are largely unaffected by vignetting and exposure.
4. Flat field      : the card's red fluorescence is smoothed and interpolated
                     under the leaves to give a relative UV-excitation map;
                     the blue (tracer) channel is divided by it.
5. Deposits        : 'local'    - pixels exceeding the local leaf background
                                  by k robust SDs (no controls needed), or
                     'absolute' - corrected blue >= a threshold calibrated on
                                  untreated control leaves.
                     Optionally require a tracer-like colour (higher G/B than
                     the leaf), which rejects bright midribs.
6. Outputs         : per-leaf CSV, per-image summary CSV, QC overlay images.

Run from anywhere:
    python SRC/analyse_images.py
    python SRC/analyse_images.py --input "D:/trial/images" --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

log = logging.getLogger("spray_coverage")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "paths": {
        "raw_dir": "DATA/raw",
        "processed_dir": "DATA/processed",
        "outputs_dir": "DATA/outputs",
    },
    "scale": {
        "method": "card_height",
        "card_height_cm": 21.0,
        "px_per_cm": 96.5,
        "illum_diameter_cm": 18.35,
        "illum_min_sum": 15,
    },
    "segmentation": {
        "card_ratio_min": 1.0,
        "leaf_ratio_max": 0.6,
        "card_edge_margin_px": 15,
        "saturation_level": 250,
        "min_leaf_area_cm2": 2.0,
        "min_solidity": 0.85,
        "max_leaf_blue_median": 150,
        "split_erode_px": 25,
    },
    "flat_field": {"enabled": True, "smoothing_sigma_px": 60,
                   "normalise_between_images": True, "card_reference_level": 100.0},
    "deposits": {
        "method": "local",
        "k_sigma": 4.0,
        "min_delta": 8.0,
        "background_window_px": 61,
        "envelope_radius_px": 40,
        "absolute_thresholds": {"Upper": 55.0, "Under": 80.0, "default": 80.0},
        "absolute_colour_margin": 0.04,
        "colour_model": {"intercept": 0.188, "card_green_red": 0.323, "under": -0.018},
        "leaf_side_column": "Leaf_side",
        "leaf_side_aliases": {"Upper": "Upper", "Top": "Upper", "Adaxial": "Upper",
                              "Under": "Under", "Lower": "Under", "Abaxial": "Under"},
        "absolute_threshold": 60.0,
        "colour_filter": True,
        "green_blue_margin": 0.03,
        "detection_limit_pct": None,
        "detection_limit_by": None,
        "detection_limit_values": None,
        "min_deposit_area_px": 5,
        "leaf_edge_exclusion_px": 6,
        "colour_blur_sigma": 2.5,
        "vein_filter": False,
        "vein_threshold_fraction": 0.5,
        "vein_min_length_px": 60,
        "vein_min_elongation": 4.0,
        "vein_dilate_px": 3,
    },
    "output": {"overlay_scale": 0.5, "save_debug_images": False},
    "processing": {"reference_width_px": 2576, "processing_width": None},
    "metadata": {"path": None, "sheet": None, "id_column": "Photo_number",
                 "filename_pattern": r"DSCN0*(\d+)",
                 "summary_factors": ["Spray_rate", "Canopy_location", "Leaf_side"]},
}


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    if not path.exists():
        log.warning("Config file %s not found - using defaults.", path)
        return cfg
    if yaml is None:
        sys.exit("PyYAML is required to read config.yaml: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        sys.exit(f"{path} is not a valid YAML mapping.")
    return _deep_update(cfg, user)


def resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    idx = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes of every object in a 0/1 mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
    return out


def disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(np.uint8)
    # Distance transform gives an exact circular erosion of any radius.
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return (dist > r).astype(np.uint8)


def remove_small(mask: np.ndarray, min_area: int):
    """Remove components < min_area. Returns cleaned mask, n kept, areas, labels."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= min_area
    return keep[lab].astype(np.uint8), int(keep.sum()), areas[keep[1:]]


def split_touching(mask: np.ndarray, erode_px: int, min_seed_area: float):
    """Separate leaves that touch at narrow contacts. Each object is eroded;
    if it breaks into >= 2 sizeable cores, pixels are reassigned to the cores
    by watershed and a 1-px cut line is drawn between them.
    Returns (new mask, number of extra objects created)."""
    if erode_px <= 0:
        return mask, 0
    core = erode(mask, erode_px)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    seeds = np.zeros(mask.shape, np.int32)
    k = 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_seed_area:
            k += 1
            seeds[lab == i] = k + 1          # labels >= 2; 1 = background
    n_obj, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    extra = k - (n_obj - 1)
    if extra <= 0:
        return mask, 0
    seeds[mask == 0] = 1
    ws = cv2.watershed(cv2.merge([mask.astype(np.uint8) * 255] * 3), seeds)
    out = ((ws > 1) & (mask > 0)).astype(np.uint8)   # ws == -1 are cut lines
    # Objects that had no seed (small leaves fully eroded) are restored as-is
    unseeded = (mask > 0) & (ws <= 1) & (cv2.dilate((ws == -1).astype(np.uint8), disk(1)) == 0)
    out |= unseeded.astype(np.uint8)
    return out, int(extra)


def measure_card_height(card: np.ndarray) -> float:
    """Vertical extent (px) of the red card, as the median over the central
    columns. Leaves lying on the card do not affect it (only the top-most
    and bottom-most card pixels in each column are used)."""
    h, w = card.shape
    ext = []
    for x in range(int(w * 0.35), int(w * 0.65), 10):
        ys = np.flatnonzero(card[:, x])
        if ys.size > 0.2 * h:
            ext.append(ys[-1] - ys[0] + 1)
    return float(np.median(ext)) if ext else float("nan")


def leaf_solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return float("nan")
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    return float(mask.sum() / hull_area) if hull_area > 0 else float("nan")


def long_thin_components(mask: np.ndarray, min_length: float, min_elongation: float) -> np.ndarray:
    """Return the components of a 0/1 mask that are long (minAreaRect long side
    >= min_length px) and thin (long/short side >= min_elongation, or a low
    fill of their rotated box, as for curved or branching veins)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if max(bw, bh) < min_length * 0.7:
            continue
        pts = np.column_stack(np.nonzero(lab[y:y + bh, x:x + bw] == i))[:, ::-1].astype(np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(a, b), max(min(a, b), 1.0)
        fill = area / max(a * b, 1.0)
        if long_side >= min_length and (long_side / short_side >= min_elongation or fill < 0.25):
            out[lab == i] = 1
    return out


def robust_sd(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def masked_smooth(values: np.ndarray, weight: np.ndarray, sigma: float, work_scale: float = 0.25) -> np.ndarray:
    """Normalised convolution: Gaussian-smooth `values` using only pixels where
    weight > 0, filling gaps (e.g. under leaves) from the surroundings."""
    h, w = values.shape
    v = cv2.resize(values * weight, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(weight.astype(np.float32), None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    s = max(1.0, sigma * work_scale)
    num = cv2.GaussianBlur(v, (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)
    # Second, wider pass fills any remaining large gaps
    num2 = cv2.GaussianBlur(v, (0, 0), s * 4)
    den2 = cv2.GaussianBlur(m, (0, 0), s * 4)
    fine = num / np.maximum(den, 1e-6)
    coarse = num2 / np.maximum(den2, 1e-6)
    alpha = np.clip(den / 0.5, 0.0, 1.0)            # smooth blend, no hard edges
    out = alpha * fine + (1.0 - alpha) * coarse
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class ImageResult:
    leaf_rows: list
    summary: dict
    overlay: np.ndarray
    debug: dict


def analyse_image(path: Path, cfg: dict, capture: list | None = None,
                  leaf_side: str | None = None) -> ImageResult:
    """Analyse one image. If `capture` is a list, per-leaf pixel arrays are
    appended to it (used by calibrate_controls.py)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("could not read image")
    native_width = img.shape[1]
    pw = cfg.get("processing", {}).get("processing_width")
    if pw and img.shape[1] != int(pw):
        f0 = float(pw) / img.shape[1]
        img = cv2.resize(img, None, fx=f0, fy=f0, interpolation=cv2.INTER_AREA)
    # Pixel-based settings are defined at reference_width_px; scale them to
    # the width actually processed so behaviour is resolution-independent.
    S = img.shape[1] / float(cfg.get("processing", {}).get("reference_width_px", 2576))

    def P(v):  # scale a length in px
        return float(v) * S
    f = img.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    h, w = B.shape
    sc, sg, ff, dp = cfg["scale"], cfg["segmentation"], cfg["flat_field"], cfg["deposits"]
    warnings: list[str] = []

    saturated = (img >= sg["saturation_level"]).any(axis=2)

    # 1) lit region -------------------------------------------------------
    lit_small = cv2.resize(((B + G + R) >= sc["illum_min_sum"]).astype(np.uint8), None,
                           fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    lit_small = largest_component(cv2.morphologyEx(lit_small, cv2.MORPH_CLOSE, disk(4)))
    lit_small = fill_holes(lit_small)
    lit = cv2.resize(lit_small, (w, h), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(lit_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    illum_diam_px = np.nan
    if cnts:
        (_, _), rad = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        illum_diam_px = 2.0 * rad / 0.25
        x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if sc["method"] == "illumination_circle" and (
                x <= 1 or y <= 1 or (x + bw) >= lit_small.shape[1] - 1 or (y + bh) >= lit_small.shape[0] - 1):
            warnings.append("lit circle touches image border (scale may be wrong)")

    # 2) background card --------------------------------------------------
    ratio = (R + 1.0) / (B + 1.0)
    card_height_px = measure_card_height(((ratio > sg["card_ratio_min"]) & (lit > 0)).astype(np.uint8))

    if sc["method"] == "fixed":
        px_per_cm = float(sc["px_per_cm"])
    elif sc["method"] == "card_height":
        px_per_cm = card_height_px / float(sc["card_height_cm"])
    elif sc["method"] == "illumination_circle":
        px_per_cm = illum_diam_px / float(sc["illum_diameter_cm"])
    else:
        raise ValueError(f"unknown scale.method '{sc['method']}'")
    cm2_per_px = 1.0 / px_per_cm**2 if np.isfinite(px_per_cm) and px_per_cm > 0 else np.nan

    card = ((ratio > sg["card_ratio_min"]) & (lit > 0) & ~saturated).astype(np.uint8)
    card = cv2.morphologyEx(card, cv2.MORPH_OPEN, disk(max(1, round(P(2)))))
    # Leaves lying across the card (tip to petiole) split the visible red area
    # into several pieces, so keep every large piece, not just the largest.
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(card, connectivity=8)
    min_piece = 0.005 * h * w
    keep_c = [i for i in range(1, n_c) if st_c[i, cv2.CC_STAT_AREA] >= min_piece]
    card = np.isin(lab_c, keep_c).astype(np.uint8)
    if card.sum() == 0:
        raise RuntimeError("background card not found (check segmentation.card_ratio_min)")
    cnts, _ = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(cnts))
    sample_area = np.zeros((h, w), np.uint8)
    cv2.fillPoly(sample_area, [hull], 1)
    sample_area = erode(sample_area & lit, int(round(P(sg["card_edge_margin_px"]))))

    # 3) leaves -----------------------------------------------------------
    leaf = ((ratio < sg["leaf_ratio_max"]) & (sample_area > 0) & ~saturated).astype(np.uint8)
    # Opening removes specks; no closing step, because closing can bridge
    # narrow gaps between neighbouring leaves. Holes are filled per object.
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, disk(max(1, round(P(2)))))
    leaf = fill_holes(leaf) & sample_area
    min_leaf_px = (sg["min_leaf_area_cm2"] / cm2_per_px) if np.isfinite(cm2_per_px) else 20000
    leaf, n_split = split_touching(leaf, int(round(P(sg["split_erode_px"]))), min_leaf_px * 0.15)
    if n_split:
        warnings.append(f"{n_split} touching-leaf contact(s) split - check overlay")
    n, lab, st, cen = cv2.connectedComponentsWithStats(leaf, connectivity=4)
    leaf_ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_leaf_px]
    # Reject very bright objects (e.g. the blue fluorescent reference strip):
    # leaves are dark in blue relative to these.
    max_b = float(sg.get("max_leaf_blue_median", 150))
    bright = [i for i in leaf_ids if np.median(B[lab == i]) >= max_b]
    if bright:
        warnings.append(f"{len(bright)} very bright object(s) excluded (reference strip?)")
        leaf_ids = [i for i in leaf_ids if i not in bright]
    # sort top-to-bottom, then left-to-right for stable numbering
    leaf_ids.sort(key=lambda i: (round(cen[i][1] / (h / 6)), cen[i][0]))
    edge_band = (sample_area > 0) & (erode(sample_area, 3) == 0)

    # 4) flat field from card fluorescence ---------------------------------
    if ff["enabled"]:
        card_ref = (card > 0) & (erode(sample_area, 0) > 0) & (cv2.dilate(leaf, disk(round(P(8)))) == 0)
        illum_map = masked_smooth(R, card_ref.astype(np.float32), P(ff["smoothing_sigma_px"]))
        card_level = float(np.median(R[card_ref])) if card_ref.any() else np.nan
        # Normalise to a fixed card level so that corrected blue values (and
        # hence thresholds) are comparable between images, despite exposure
        # or UV-output differences.
        target = float(ff["card_reference_level"]) if ff.get("normalise_between_images", True) else card_level
        illum_map = np.clip(illum_map / max(target, 1e-6), 0.1, 10.0)
    else:
        illum_map = np.ones_like(B)
        card_level = float(np.median(R[card > 0]))
    Bc = B / illum_map
    # Card colour (G/R) tracks camera white balance / black level, and
    # predicts the colour of clean leaf in this image (hybrid method).
    card_gr = float(np.median((G[card > 0] + 1.0) / (R[card > 0] + 1.0)))

    # 5) deposits ---------------------------------------------------------
    leaf_all = np.isin(lab, leaf_ids)
    leaf_core = erode(leaf_all.astype(np.uint8), int(round(P(dp["leaf_edge_exclusion_px"])))) > 0
    # Colour (G/B) is evaluated on smoothed channels: JPEG 4:2:0 stores colour
    # at half resolution, so per-pixel colour ratios at sharp edges (midribs,
    # veins) are unreliable and create false 'tracer-coloured' fringes.
    cs = P(dp.get("colour_blur_sigma", 0.8))
    Bs = cv2.GaussianBlur(B, (0, 0), cs)
    Gs = cv2.GaussianBlur(G, (0, 0), cs)
    gb = (Gs + 1.0) / (Bs + 1.0)

    if dp["method"] == "local":
        # Median background computed on leaf pixels only (non-leaf replaced by
        # the leaf median so the filter is not pulled up by the bright card).
        fill_val = float(np.median(Bc[leaf_core])) if leaf_core.any() else 0.0
        tmp = np.where(leaf_all, Bc, fill_val).astype(np.float32)
        ws = 0.5
        k = int(P(dp["background_window_px"]) * ws) | 1
        small = cv2.resize(tmp, None, fx=ws, fy=ws, interpolation=cv2.INTER_AREA)
        bg = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), max(3, min(k, 255)))
        bg = cv2.resize(bg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        resid = Bc - bg
    elif dp["method"] in ("envelope", "hybrid"):
        # Rolling-ball style background: morphological opening (erode then
        # dilate) traces the LOWER envelope of the leaf, so it follows slow
        # brightness changes but is not lifted by bright deposits smaller than
        # the structuring element - even when deposits cover much of the leaf.
        ws = 0.5
        fill = masked_smooth(Bc, leaf_core.astype(np.float32), P(30))
        tmp = np.where(leaf_core, Bc, fill).astype(np.float32)
        small = cv2.resize(tmp, None, fx=ws, fy=ws, interpolation=cv2.INTER_AREA)
        r = max(2, int(round(P(dp.get("envelope_radius_px", 40)) * ws)))
        bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, disk(r))
        bg = cv2.GaussianBlur(bg, (0, 0), r / 2.0)
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
        resid = Bc - bg
    elif dp["method"] == "absolute":
        resid = Bc - float(dp["absolute_threshold"])
    else:
        raise ValueError(f"unknown deposits.method '{dp['method']}'")

    # hybrid: per-leaf-side fixed threshold and predicted clean-leaf colour
    aliases = {str(k).lower(): v for k, v in (dp.get("leaf_side_aliases") or {}).items()}
    side = aliases.get(str(leaf_side).strip().lower(), leaf_side) if leaf_side is not None else None
    thr_tab = dp.get("absolute_thresholds") or {}
    if side in thr_tab:
        abs_T = float(thr_tab[side])
    else:
        abs_T = float(thr_tab.get("default", 80.0))
        if dp["method"] == "hybrid":
            warnings.append("leaf side unknown: conservative (Under) fixed threshold used")
    cm = dp.get("colour_model") or {}
    gb_pred = (float(cm.get("intercept", 0.188)) + float(cm.get("card_green_red", 0.323)) * card_gr
               + (float(cm.get("under", -0.018)) if side == "Under" else 0.0))

    vis_base = img.copy()
    leaf_rows = []
    deposit_all = np.zeros((h, w), np.uint8)
    for out_id, li in enumerate(leaf_ids, start=1):
        lm = lab == li
        core = lm & leaf_core
        if core.sum() == 0:
            continue
        bg_pix = core
        if dp["method"] == "local":
            sd = robust_sd(resid[core])
            thr = max(dp["k_sigma"] * sd, dp["min_delta"])
            cand = core & (resid >= thr)
            thr_abs = float(np.median(bg[core])) + thr
        elif dp["method"] in ("envelope", "hybrid"):
            # Noise and offset from the LOWER tail only (P5, P20), assuming the
            # darkest ~50% of the leaf is clean: robust to dense deposits.
            p5, p20 = np.percentile(resid[core], [5, 20])
            sd = max((p20 - p5) / 0.803, 0.5)
            mu = p20 + 0.842 * sd
            thr = mu + max(dp["k_sigma"] * sd, dp["min_delta"])
            cand = core & (resid >= thr)
            bg_pix = core & (resid < mu)
            thr_abs = float(np.median(bg[core])) + thr
        else:
            sd, thr = robust_sd(Bc[core]), float(dp["absolute_threshold"])
            cand = core & (resid >= 0)
            thr_abs = thr
        # colour reference from background (non-deposit) pixels of this leaf
        gb_leaf = float(np.median(gb[bg_pix])) if bg_pix.any() else float(np.median(gb[core]))
        if dp["colour_filter"]:
            cand &= gb >= (gb_leaf + dp["green_blue_margin"])
        vein_px = 0
        if dp.get("vein_filter", False):
            # Veins/midribs/creases: long, thin structures brighter than the
            # leaf. Found with a looser threshold (no colour test); candidates
            # touching them are rejected.
            loose = (core & (resid >= dp.get("vein_threshold_fraction", 0.5) * thr)).astype(np.uint8)
            veins = long_thin_components(loose, P(dp.get("vein_min_length_px", 60)),
                                         dp.get("vein_min_elongation", 4.0))
            veins = cv2.dilate(veins, disk(int(round(P(dp.get("vein_dilate_px", 3)))))) > 0
            vein_px = int((veins & core).sum())
            n_v, lab_v = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
            bad = np.unique(lab_v[veins & (lab_v > 0)])
            cand &= ~np.isin(lab_v, bad)
        fixed = np.zeros_like(cand)
        if dp["method"] == "hybrid":
            # Control-referenced test: exposure-corrected blue above the level
            # never reached by unsprayed leaves of this leaf side, AND colour
            # above the clean-leaf colour predicted from the card. Detects
            # films covering most of a leaf, which leaf-relative tests cannot.
            fixed = core & (Bc >= abs_T) & ((gb - gb_pred) >= float(dp.get("absolute_colour_margin", 0.04)))
            cand = cand | fixed
        dep, n_dep, dep_areas = remove_small(cand, int(round(dp["min_deposit_area_px"] * S * S)))
        dep = dep.astype(bool)
        deposit_all |= dep
        if capture is not None:
            capture.append({"filename": path.name, "leaf_id": out_id,
                            "z": (resid[core] / max(sd, 1e-6)).astype(np.float32),
                            "delta": resid[core].astype(np.float32),
                            "dgb": (gb[core] - gb_leaf).astype(np.float32),
                            "gb": gb[core].astype(np.float32),
                            "bc": Bc[core].astype(np.float32)})

        leaf_px = int(lm.sum())
        core_px = int(core.sum())
        dep_px = int(dep.sum())
        coverage = 100.0 * dep_px / core_px if core_px else 0.0
        # Flag only substantial contact with the card edge (a petiole or
        # leaf tip crossing the edge is harmless).
        contact_len = (lm & edge_band).sum() / 3.0          # edge band is 3 px wide
        leaf_cnts = cv2.findContours(lm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        perimeter = cv2.arcLength(max(leaf_cnts, key=cv2.contourArea), True)
        touches_edge = bool(contact_len > 0.02 * perimeter)
        solidity = leaf_solidity(lm)
        leaf_rows.append({
            "filename": path.name,
            "leaf_id": out_id,
            "leaf_area_px": leaf_px,
            "leaf_area_cm2": leaf_px * cm2_per_px,
            "analysed_area_px": core_px,
            "deposit_area_px": dep_px,
            "deposit_area_cm2": dep_px * cm2_per_px,
            "coverage_pct": coverage,
            "deposit_count": n_dep,
            "deposits_per_cm2": n_dep / (core_px * cm2_per_px) if core_px and np.isfinite(cm2_per_px) else np.nan,
            "median_deposit_area_px": float(np.median(dep_areas)) if n_dep else 0.0,
            "leaf_background_blue": float(np.median(Bc[core])),
            "leaf_noise_sd": sd,
            "threshold_used": thr,
            "coverage_fixed_pct": 100.0 * float((dep & fixed).sum()) / core_px if core_px else 0.0,
            "leaf_side_used": side,
            "fixed_threshold_corrected_blue": abs_T if dp["method"] == "hybrid" else np.nan,
            "predicted_clean_green_blue": gb_pred,
            # effective threshold as a corrected and as a RAW blue grey level
            # (raw = what you would read off the original image at this leaf)
            "threshold_corrected_blue": thr_abs,
            "threshold_raw_blue_equiv": thr_abs * float(np.median(illum_map[core])),
            "leaf_raw_blue_median": float(np.median(B[core])),
            "leaf_green_blue": gb_leaf,
            "excess_fluorescence_sum": float(np.clip(resid[dep], 0, None).sum()),
            "mean_illum_factor": float(np.mean(illum_map[core])),
            "touches_card_edge": touches_edge,
            "solidity": solidity,
            "vein_excluded_px": vein_px,
        })
        if touches_edge:
            warnings.append(f"leaf {out_id} touches the card edge (part may be excluded)")
        if np.isfinite(solidity) and solidity < sg.get("min_solidity", 0.85):
            warnings.append(f"leaf {out_id} irregular outline (solidity {solidity:.2f}): "
                            f"possibly overlapping leaves or a damaged leaf - check overlay")

    if not leaf_rows:
        warnings.append("no leaves detected")
    frac_sat = float(saturated[leaf_all].mean()) if leaf_all.any() else 0.0
    blue_med = float(np.median(B[leaf_all])) if leaf_all.any() else np.nan
    if np.isfinite(blue_med) and blue_med < 50:
        warnings.append(f"leaf blue level low ({blue_med:.0f}/255): image under-exposed, weak deposit contrast")

    tot_core = sum(r["analysed_area_px"] for r in leaf_rows)
    tot_dep = sum(r["deposit_area_px"] for r in leaf_rows)
    summary = {
        "filename": path.name,
        "n_leaves": len(leaf_rows),
        "px_per_cm": px_per_cm,
        "scale_method": sc["method"],
        "card_height_px": card_height_px,
        "native_width_px": native_width,
        "processed_width_px": img.shape[1],
        "illumination_diameter_px": illum_diam_px,
        "total_leaf_area_cm2": sum(r["leaf_area_cm2"] for r in leaf_rows),
        "pooled_coverage_pct": 100.0 * tot_dep / tot_core if tot_core else np.nan,
        "detection_limit_pct": dp.get("detection_limit_pct"),
        "mean_leaf_coverage_pct": float(np.mean([r["coverage_pct"] for r in leaf_rows])) if leaf_rows else np.nan,
        "total_deposit_count": sum(r["deposit_count"] for r in leaf_rows),
        "leaf_raw_blue_median": blue_med,
        "card_red_median": card_level,
        "card_green_red": card_gr,
        "leaf_saturated_fraction": frac_sat,
        "deposit_method": dp["method"],
        "warnings": "; ".join(warnings),
    }

    # 6) QC overlay -------------------------------------------------------
    vis = cv2.convertScaleAbs(vis_base, alpha=2.0, beta=0)   # brighten for viewing
    vis[deposit_all > 0] = (255, 0, 255)                      # deposits: magenta
    hull_c, _ = cv2.findContours(sample_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, hull_c, -1, (0, 255, 255), 3)       # sample area: yellow
    for out_id, li in enumerate(leaf_ids, start=1):
        lm = (lab == li).astype(np.uint8)
        c, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, c, -1, (0, 255, 0), 3)          # leaves: green
        cx, cy = map(int, cen[li])
        cov = next((r["coverage_pct"] for r in leaf_rows if r["leaf_id"] == out_id), 0.0)
        label = f"{out_id}: {cov:.1f}%"
        for col, th in (((0, 0, 0), 8), ((255, 255, 255), 3)):
            cv2.putText(vis, label, (cx - 90, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, th, cv2.LINE_AA)
    s = float(cfg["output"]["overlay_scale"])
    if s != 1.0:
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    debug = {}
    if cfg["output"]["save_debug_images"]:
        debug["flatfield"] = cv2.normalize(illum_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        r8 = np.clip(resid * 8 + 128, 0, 255).astype(np.uint8)
        r8[~leaf_all] = 0
        debug["residual"] = r8
    return ImageResult(leaf_rows, summary, vis, debug)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quantify fluorescent spray coverage on leaf images.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml", help="YAML config (default: repo config.yaml)")
    p.add_argument("--input", type=Path, help="folder of raw images (overrides config)")
    p.add_argument("--output", type=Path, help="folder for CSV outputs (overrides config)")
    p.add_argument("--processed", type=Path, help="folder for QC overlays (overrides config)")
    p.add_argument("--debug", action="store_true", help="also save flat-field and residual images")
    p.add_argument("--metadata", type=Path, help="trial spreadsheet (.xlsx/.csv) to join on photo number")
    p.add_argument("--sheet", help="sheet name in the metadata workbook (e.g. Cherry)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def apply_detection_limit(leaf: pd.DataFrame, img: pd.DataFrame, dp: dict):
    """Set above_detection_limit using a single limit or one per stratum
    (e.g. per Leaf_side), as configured."""
    if leaf.empty:
        return leaf, img
    by = dp.get("detection_limit_by")
    values = dp.get("detection_limit_values") or {}
    default = dp.get("detection_limit_pct")
    if by and by in leaf.columns and values:
        lim = leaf[by].map(lambda v: values.get(str(v), default))
    else:
        lim = pd.Series(default, index=leaf.index)
    leaf["detection_limit_pct"] = lim
    leaf["above_detection_limit"] = np.where(lim.notna(), leaf["coverage_pct"] > lim.astype(float), np.nan)
    if not img.empty and "filename" in img.columns:
        frac = leaf.groupby("filename")["above_detection_limit"].mean()
        img["pct_leaves_above_LOD"] = img["filename"].map(100 * frac)
    return leaf, img


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    if args.debug:
        cfg["output"]["save_debug_images"] = True

    # Command-line paths are relative to the current folder; config paths are
    # relative to the repository root.
    raw_dir = args.input.resolve() if args.input else resolve(cfg["paths"]["raw_dir"])
    out_dir = args.output.resolve() if args.output else resolve(cfg["paths"]["outputs_dir"])
    proc_dir = args.processed.resolve() if args.processed else resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    log.info("Input folder : %s", raw_dir)
    log.info("Outputs      : %s", out_dir)
    log.info("QC overlays  : %s", proc_dir)
    if not raw_dir.is_dir():
        log.error("Input folder does not exist: %s", raw_dir)
        return 1
    images = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    log.info("Found %d image(s)", len(images))
    if not images:
        log.error("No images found. Supported extensions: %s", ", ".join(sorted(IMAGE_EXTS)))
        return 1

    md = cfg.get("metadata", {})
    meta_path = args.metadata or (resolve(md["path"]) if md.get("path") else None)
    meta, side_of = None, {}
    if meta_path:
        import metadata as M
        pattern = md.get("filename_pattern", r"DSCN0*(\d+)")
        idc = md.get("id_column", "Photo_number")
        meta = M.load_metadata(meta_path, args.sheet or md.get("sheet"), idc)
        log.info("Metadata: %s (%d rows)", meta_path, len(meta))
        sc_ = cfg["deposits"].get("leaf_side_column", "Leaf_side")
        if sc_ in meta.columns:
            side_of = dict(zip(meta[idc], meta[sc_]))
    elif cfg["deposits"]["method"] == "hybrid":
        log.warning("No metadata: leaf side unknown, so the conservative (Under) fixed threshold is used. "
                    "Give --metadata for full sensitivity on upper surfaces.")

    leaf_rows, summaries = [], []
    for i, p in enumerate(images, 1):
        try:
            side = side_of.get(M.photo_number(p.name, pattern)) if side_of else None
            res = analyse_image(p, cfg, leaf_side=side)
            leaf_rows.extend(res.leaf_rows)
            summaries.append(res.summary)
            cv2.imwrite(str(proc_dir / f"{p.stem}_qc.jpg"), res.overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            for name, im in res.debug.items():
                cv2.imwrite(str(proc_dir / f"{p.stem}_{name}.png"), im)
            msg = f"[{i}/{len(images)}] {p.name}: {res.summary['n_leaves']} leaves, " \
                  f"pooled coverage {res.summary['pooled_coverage_pct']:.2f}%"
            log.info(msg + (f"  WARNING: {res.summary['warnings']}" if res.summary["warnings"] else ""))
        except Exception as e:  # keep going on a bad image
            log.error("[%d/%d] %s: ERROR %s", i, len(images), p.name, e)
            summaries.append({"filename": p.name, "n_leaves": 0, "warnings": f"ERROR: {e}"})

    leaf_df, img_df = pd.DataFrame(leaf_rows), pd.DataFrame(summaries)
    if meta_path:
        issues = M.check_metadata(meta, [p.name for p in images], pattern, idc)
        issues.to_csv(out_dir / "metadata_issues.csv", index=False)
        leaf_df = M.attach(leaf_df, meta, pattern, idc)
        img_df = M.attach(img_df, meta, pattern, idc)
    leaf_df, img_df = apply_detection_limit(leaf_df, img_df, cfg["deposits"])
    if meta_path and not img_df.empty:
        factors = md.get("summary_factors", ["Spray_rate", "Canopy_location", "Leaf_side"])
        tabs = []
        for k in range(1, len(factors) + 1):
            t = M.treatment_summary(img_df, leaf_df, factors[:k])
            if not t.empty:
                t.insert(0, "grouping", " x ".join(factors[:k]))
                tabs.append(t)
        if tabs:
            pd.concat(tabs, ignore_index=True).to_csv(out_dir / "treatment_summary.csv", index=False)
            log.info("Wrote %s", out_dir / "treatment_summary.csv")

    leaf_csv = out_dir / "leaf_coverage_results.csv"
    summ_csv = out_dir / "image_summary.csv"
    leaf_df.to_csv(leaf_csv, index=False)
    img_df.to_csv(summ_csv, index=False)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        if yaml is not None:
            yaml.safe_dump(cfg, fh, sort_keys=False)
    log.info("Wrote %s (%d leaves)", leaf_csv, len(leaf_rows))
    log.info("Wrote %s (%d images)", summ_csv, len(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
