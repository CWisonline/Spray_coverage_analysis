"""
Link image results to the trial metadata spreadsheet (e.g. Photo_numbers.xlsx).

Images are matched on the camera's sequential file number: 'DSCN0923.JPG'
matches the row with Photo_number = 923 (leading zeros ignored).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("spray_coverage")


def load_metadata(path: Path, sheet: str | None, id_column: str = "Photo_number") -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    # tidy column names: 'UV _filter' -> 'UV_filter'
    df.columns = [re.sub(r"\s+", "", str(c)) for c in df.columns]
    id_column = re.sub(r"\s+", "", id_column)
    if id_column not in df.columns:
        raise ValueError(f"metadata has no '{id_column}' column (columns: {list(df.columns)})")
    df = df.dropna(subset=[id_column]).copy()
    df[id_column] = df[id_column].astype(int)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    return df


def photo_number(filename: str, pattern: str) -> int | None:
    m = re.search(pattern, filename, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def check_metadata(meta: pd.DataFrame, filenames: list[str], pattern: str,
                   id_column: str = "Photo_number") -> pd.DataFrame:
    """Return a table of data-quality issues (also logged)."""
    issues = []
    ids = meta[id_column]
    for v in ids[ids.duplicated()].unique():
        issues.append(("duplicate_photo_number", int(v), "appears more than once in metadata"))
    diffs = ids.diff()
    for idx in np.flatnonzero(diffs.to_numpy() < 0):
        issues.append(("out_of_sequence", int(ids.iloc[idx]),
                       f"follows {int(ids.iloc[idx - 1])} - possible typing error"))
    nums = {photo_number(f, pattern): f for f in filenames}
    for n, f in nums.items():
        if n is None:
            issues.append(("unparsed_filename", -1, f"{f}: no photo number found"))
        elif n not in set(ids):
            issues.append(("image_not_in_metadata", n, f))
    df = pd.DataFrame(issues, columns=["issue", "photo_number", "detail"])
    for _, r in df.iterrows():
        log.warning("METADATA %s: %s %s", r.issue, r.photo_number, r.detail)
    return df


def attach(df: pd.DataFrame, meta: pd.DataFrame, pattern: str, id_column: str = "Photo_number") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.insert(1, id_column, [photo_number(f, pattern) for f in df["filename"]])
    return df.merge(meta, on=id_column, how="left")


def treatment_summary(img: pd.DataFrame, leaf: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Descriptive summary by treatment factors. Image means are the unit
    (leaves within an image are sub-samples)."""
    factors = [f for f in factors if f in img.columns]
    if not factors:
        return pd.DataFrame()
    img = img.copy()
    for f in factors:
        img[f] = img[f].fillna("(none)")
    g = img.groupby(factors, dropna=False)
    out = g["mean_leaf_coverage_pct"].agg(n_images="count", mean_coverage_pct="mean",
                                          sd_coverage_pct="std", median_coverage_pct="median",
                                          min_coverage_pct="min", max_coverage_pct="max")
    if "above_detection_limit" in leaf.columns:
        lf = leaf.copy()
        for f in factors:
            lf[f] = lf[f].fillna("(none)")
        out["n_leaves"] = lf.groupby(factors, dropna=False)["leaf_id"].count()
        out["pct_leaves_above_LOD"] = 100 * lf.groupby(factors, dropna=False)["above_detection_limit"].mean()
    return out.reset_index()
