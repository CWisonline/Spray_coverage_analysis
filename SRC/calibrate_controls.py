#!/usr/bin/env python3
"""
Calibrate deposit detection from UNSPRAYED control leaves.

Runs the normal analysis on a folder of control images (no tracer) and
reports how often the current settings detect 'deposits' where none exist.

Outputs (in --output, default DATA/calibration):
  controls_leaf_stats.csv   one row per control leaf: background level, noise,
                            colour, percentiles of corrected blue, and the
                            false-positive coverage under the current settings
  exceedance_table.csv      % of control-leaf pixels passing each combination
                            of k_sigma and green_blue_margin (min_delta applied)
  calibration_report.txt    summary and recommended detection limit

Usage:
    python SRC/calibrate_controls.py --input DATA/controls
    # With the trial spreadsheet: pre-spray photos are picked out of the full
    # image folder automatically and a limit is computed per leaf side.
    python SRC/calibrate_controls.py --input DATA/raw --metadata DATA/Photo_numbers.xlsx --sheet Cherry

The recommended detection limit is the 95th percentile of the per-leaf
false-positive coverage. Copy it to deposits.detection_limit_pct in
config.yaml; coverage values at or below it cannot be distinguished from an
unsprayed leaf.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_images as A  # noqa: E402

import metadata as M  # noqa: E402

K_GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
M_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]


def bootstrap_lod(df: pd.DataFrame, q: float, n: int = 2000, seed: int = 1):
    """Percentile of per-leaf false positives with a 95% CI from resampling
    whole images (leaves within an image are not independent)."""
    groups = [g["false_pos_coverage_pct"].to_numpy() for _, g in df.groupby("filename")]
    if len(groups) < 3:
        return float(np.percentile(df["false_pos_coverage_pct"], q)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [np.percentile(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), q)
         for _ in range(n)]
    return float(np.percentile(df["false_pos_coverage_pct"], q)), *np.percentile(b, [2.5, 97.5])


def stratified_lines(df: pd.DataFrame, args) -> list[str]:
    out = []
    if args.stratify and args.stratify in df.columns:
        out += [f"Detection limit by {args.stratify} (p{args.percentile:g}; 95% CI from resampling images):"]
        vals = {}
        for k, g in df.groupby(args.stratify):
            lod, lo, hi = bootstrap_lod(g, args.percentile)
            n_img = g.filename.nunique()
            ok = n_img >= 10
            if ok:
                vals[str(k)] = round(max(lod, hi if np.isfinite(hi) else lod), 3)
            out.append(f"  {k:10s} images {n_img:3d} leaves {len(g):4d}  background median "
                       f"{g.background_blue_corr.median():5.1f}  LOD {lod:.3f}% (CI {lo:.3f}-{hi:.3f}%)"
                       + ("" if ok else "  <- fewer than 10 images: not reliable, not suggested"))
        if not vals:
            out += ["  No stratum has >= 10 control images; keep the single detection_limit_pct.", ""]
            vals = None
        out += [] if vals is None else ["  Suggested config (upper CI used, conservative):",
                f"    detection_limit_by: {args.stratify}",
                "    detection_limit_values: {" + ", ".join(f"{k}: {v}" for k, v in vals.items()) + "}", ""]
    if args.compare and args.compare in df.columns:
        keys = [c for c in [args.stratify, args.compare] if c and c in df.columns]
        t = df.groupby(keys).agg(images=("filename", "nunique"), leaves=("leaf_id", "count"),
                                 background_median=("background_blue_corr", "median"),
                                 noise_sd_median=("noise_sd", "median"),
                                 fp_mean_pct=("false_pos_coverage_pct", "mean"),
                                 fp_p95_pct=("false_pos_coverage_pct", lambda x: np.percentile(x, 95)))
        out += [f"Comparison by {args.compare} (does it change the unsprayed baseline?):",
                t.round(3).to_string(), ""]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate detection settings on unsprayed control leaves.")
    ap.add_argument("--input", type=Path, required=True, help="folder of control (unsprayed) images")
    ap.add_argument("--config", type=Path, default=A.REPO_ROOT / "config.yaml")
    ap.add_argument("--output", type=Path, default=A.REPO_ROOT / "DATA" / "calibration")
    ap.add_argument("--percentile", type=float, default=95.0,
                    help="percentile of per-leaf false-positive coverage used as detection limit (default 95)")
    ap.add_argument("--metadata", type=Path, help="trial spreadsheet; selects control images and strata")
    ap.add_argument("--sheet", help="sheet name (e.g. Cherry)")
    ap.add_argument("--control-column", default="Comments", help="metadata column identifying controls")
    ap.add_argument("--control-value", default="Pre-spray", help="value in that column marking controls")
    ap.add_argument("--stratify", default="Leaf_side", help="metadata column for separate limits ('' = none)")
    ap.add_argument("--compare", default="UV_filter", help="metadata column to compare within strata ('' = none)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("calibrate")

    cfg = A.load_config(args.config)
    if cfg["deposits"]["method"] != "local":
        log.warning("deposits.method is '%s'; calibration statistics refer to the local method.",
                    cfg["deposits"]["method"])
    in_dir = args.input.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in A.IMAGE_EXTS)
    meta = None
    if args.metadata:
        md = cfg.get("metadata", {})
        pattern, idc = md.get("filename_pattern", r"DSCN0*(\d+)"), md.get("id_column", "Photo_number")
        meta = M.load_metadata(args.metadata, args.sheet or md.get("sheet"), idc)
        ctrl_ids = set(meta.loc[meta[args.control_column].astype(str).str.strip().str.lower()
                                == args.control_value.lower(), idc])
        images = [p for p in images if M.photo_number(p.name, pattern) in ctrl_ids]
        log.info("Metadata: %d control rows ('%s' = '%s'); %d matching images found",
                 len(ctrl_ids), args.control_column, args.control_value, len(images))
    if not images:
        log.error("No images in %s", in_dir)
        return 1
    log.info("Calibrating on %d control image(s) from %s", len(images), in_dir)

    rows, caps, summaries = [], [], []
    for i, p in enumerate(images, 1):
        cap: list = []
        try:
            res = A.analyse_image(p, cfg, capture=cap)
        except Exception as e:
            log.error("[%d/%d] %s: ERROR %s", i, len(images), p.name, e)
            continue
        summaries.append(res.summary)
        for r, c in zip(res.leaf_rows, cap):
            bc = c["bc"]
            rows.append({
                "filename": r["filename"], "leaf_id": r["leaf_id"],
                "leaf_area_cm2": r["leaf_area_cm2"],
                "background_blue_corr": r["leaf_background_blue"],
                "noise_sd": r["leaf_noise_sd"],
                "leaf_green_blue": r["leaf_green_blue"],
                "blue_corr_p99": float(np.percentile(bc, 99)),
                "blue_corr_p99_9": float(np.percentile(bc, 99.9)),
                "blue_corr_max": float(bc.max()),
                "false_pos_coverage_pct": r["coverage_pct"],
                "false_pos_deposit_count": r["deposit_count"],
                "solidity": r["solidity"],
            })
            caps.append(c)
        log.info("[%d/%d] %s: %d leaves, false-positive coverage %.3f%%",
                 i, len(images), p.name, res.summary["n_leaves"], res.summary["pooled_coverage_pct"])

    if not rows:
        log.error("No leaves analysed.")
        return 1
    df = pd.DataFrame(rows)
    if meta is not None:
        df = M.attach(df, meta, pattern, idc)
    df.to_csv(out_dir / "controls_leaf_stats.csv", index=False)

    # Pixel-level exceedance table (before the minimum-deposit-size filter)
    z = np.concatenate([c["z"] for c in caps])
    d = np.concatenate([c["delta"] for c in caps])
    g = np.concatenate([c["dgb"] for c in caps])
    md = float(cfg["deposits"]["min_delta"])
    tab = pd.DataFrame(
        [[100.0 * np.mean((z >= k) & (g >= m) & (d >= md)) for m in M_GRID] for k in K_GRID],
        index=[f"k_sigma={k}" for k in K_GRID], columns=[f"gb_margin={m}" for m in M_GRID])
    tab.to_csv(out_dir / "exceedance_table.csv")

    fp = df["false_pos_coverage_pct"].to_numpy()
    lod = float(np.percentile(fp, args.percentile))
    bg = df["background_blue_corr"]
    dp = cfg["deposits"]
    lines = [
        "CONTROL CALIBRATION REPORT",
        "=" * 60,
        f"Images: {len(summaries)}   Leaves: {len(df)}   Pixels: {z.size:,}",
        f"Settings: k_sigma={dp['k_sigma']}, min_delta={dp['min_delta']}, "
        f"green_blue_margin={dp['green_blue_margin']}, colour_blur_sigma={dp.get('colour_blur_sigma')}, "
        f"min_deposit_area_px={dp['min_deposit_area_px']}, vein_filter={dp.get('vein_filter')}",
        "",
        "Leaf background (flat-field corrected blue; card normalised to "
        f"{cfg['flat_field']['card_reference_level']}):",
        f"  min {bg.min():.1f}  p25 {bg.quantile(.25):.1f}  median {bg.median():.1f}  "
        f"p75 {bg.quantile(.75):.1f}  max {bg.max():.1f}",
        f"  99.9th percentile of corrected blue per leaf: "
        f"{df['blue_corr_p99_9'].min():.0f} - {df['blue_corr_p99_9'].max():.0f}",
        "  -> a single absolute threshold is only usable if this range is narrow.",
        "",
        "False-positive coverage on control leaves (current settings):",
        f"  mean of leaves {fp.mean():.4f}%",
        f"  median {np.median(fp):.4f}%  p95 {np.percentile(fp, 95):.4f}%  max {fp.max():.4f}%",
        f"  leaves with zero detections: {(fp == 0).sum()} / {len(fp)}",
        "",
        f"RECOMMENDED detection limit (p{args.percentile:g} of per-leaf false positives): {lod:.3f}%",
        f"  -> set deposits.detection_limit_pct: {lod:.3f} in config.yaml",
        "",
        *stratified_lines(df, args),
        "Pixel exceedance on control leaves (% of pixels; before size filter):",
        tab.round(4).to_string(),
        "",
        "Caveats: the limit applies only to images taken with the same camera",
        "settings, card, tracer and leaf surface as the controls. Re-run when",
        "any of these change, and include controls in every imaging session.",
    ]
    report = "\n".join(lines)
    (out_dir / "calibration_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    log.info("\nWrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
