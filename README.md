# Leaf Spray Coverage Analysis

Python image-analysis pipeline for quantifying fluorescent spray-deposit
coverage on leaves (cherry and blueberry trials; any broad leaf on the card). Intended for use with a fluorescent
tracer added to the spray tank, e.g. DropSight UView or NIAB's edible
fluorescent tracer. Leaves are photographed on a red-fluorescing background
card in the dedicated imaging box (built-in UV LEDs, narrow band-pass filter).

## Project structure

```
SRC/analyse_images.py      analysis script
SRC/calibrate_controls.py  detection-limit calibration on unsprayed leaves
SRC/metadata.py            joins results to the trial spreadsheet
config.yaml                all tunable parameters (paths, scale, thresholds)
DATA/controls/             unsprayed control images (not tracked)
DATA/calibration/          calibration report and tables
DATA/raw/               raw fluorescence images (not tracked by git)
DATA/processed/         QC overlay images (not tracked)
DATA/outputs/           CSV results (not tracked)
```

## Setup (once)

```powershell
# Windows PowerShell, from the repository folder
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS / Linux: `source .venv/bin/activate` instead of the second line.

## Running

The script is in `SRC/`, so include the folder in the command. Paths are
resolved relative to the repository, so it can be run from any folder.

```powershell
python SRC/analyse_images.py                      # uses DATA/raw and config.yaml
python SRC/analyse_images.py --input "D:\Trial 2026\images"
python SRC/analyse_images.py --debug              # also save flat-field / residual maps
python SRC/analyse_images.py --help
```

Supported formats: .jpg, .jpeg, .png, .tif, .tiff.

### With the trial spreadsheet

Keep the camera's original file names (`DSCN0923.JPG`); they are matched to
`Photo_number` in the spreadsheet (leading zeros ignored).

```powershell
python SRC/analyse_images.py --input "D:\Cherry\images" --metadata DATA/Photo_numbers.xlsx --sheet Cherry
```
Adds the spreadsheet columns (Plot, Spray_rate, Tree, Canopy_location,
Leaf_side, UV_filter, Comments) to both CSVs, writes `treatment_summary.csv`
(descriptive statistics by Spray_rate / Canopy_location / Leaf_side, using
image means as the unit) and `metadata_issues.csv` (duplicate or
out-of-sequence photo numbers, images without a spreadsheet row).
The summary is descriptive only: for inference use a mixed model with
Plot (and Tree) as random effects, since leaves and images are sub-samples.

## Outputs

* `DATA/outputs/leaf_coverage_results.csv` - one row per leaf: leaf area,
  analysed area, deposit area, **coverage_pct**, deposit count and density,
  leaf background level, noise, threshold used, flags.
* `DATA/outputs/image_summary.csv` - one row per image: number of leaves,
  scale, pooled and mean coverage, exposure indicators and **warnings**.
* `DATA/outputs/config_used.yaml` - exact settings used (keep with results).
* `DATA/processed/<image>_qc.jpg` - overlay: leaves (green, labelled with
  coverage), detected deposits (magenta), analysed card area (yellow).
  **Always inspect these before using the numbers.**

## Method

1. **Leaves** are segmented with the red/blue colour ratio (card R/B > 1,
   leaf R/B < 0.6) inside the convex hull of the red card. Ratios are largely
   insensitive to vignetting and exposure, unlike fixed grey-level thresholds.
2. **Flat-field correction**: the card's own red fluorescence maps the UV
   excitation and lens fall-off; it is smoothed, interpolated under the
   leaves, and used to correct the blue (tracer) channel. Each image is also
   scaled to a common card level, so thresholds are comparable between images.
3. **Deposits** (`deposits.method`):
   * `envelope` (default) - background is the rolling-ball (morphological
     opening) lower envelope of the leaf, and noise is estimated from the
     darkest part of the leaf, so dense or patchy deposits do not raise the
     threshold. Deposit = background + max(`k_sigma` x noise, `min_delta`)
     and tracer-coloured. The threshold actually used on each leaf is
     reported as `threshold_raw_blue_equiv` (raw blue grey level), so it can
     be compared with what you read off the original image.
   * `local` - pixels brighter than the local leaf median by
     `k_sigma` robust SDs (and at least `min_delta` grey levels). Needs no
     controls, but will under-estimate very high or continuous (film) cover.
   * `absolute` - corrected blue >= `absolute_threshold`, which should be
     set from **untreated control leaves** imaged with identical settings.
   * `colour_filter` keeps only pixels with a higher green/blue ratio than
     the leaf (tracer emission), which rejects the bright, bluer midribs.
     The colour test is applied to smoothed channels (`colour_blur_sigma`)
     because JPEG 4:2:0 stores colour at half resolution, which otherwise
     creates false tracer-coloured fringes along veins.
4. **Detection limit**: coverage at or below `detection_limit_pct` cannot be
   distinguished from an unsprayed leaf (`above_detection_limit` column).
5. **Scale**: `card_height` (default) uses the measured height of the red
   A4 card (21.0 cm) in every image (~57 px/cm in the current setup).
   The illuminated circle is NOT a valid scale reference (it is set by the
   lens/filter optics and varied ~30% between images while the card varied <3%).

## Calibrating with unsprayed controls

```powershell
python SRC/calibrate_controls.py --input DATA/controls
# or select the pre-spray photos from the full set via the spreadsheet,
# with a separate limit per leaf side and a UV-filter on/off comparison:
python SRC/calibrate_controls.py --input "D:\Cherry\images" --metadata DATA/Photo_numbers.xlsx --sheet Cherry
```
Writes `DATA/calibration/calibration_report.txt`, per-leaf statistics and a
false-positive table for alternative `k_sigma` / `green_blue_margin` values.
Copy the recommended limit into `deposits.detection_limit_pct`. Re-run
whenever camera settings, card, tracer, crop or leaf surface change.

Current calibration (Oct 2026): 18 unsprayed cherry images, 59 leaves.
False-positive coverage per leaf: median 0.000%, p95 0.040% (bootstrap 95%
CI 0.023-0.055%), max 0.072%. Detection limit set to 0.05%.
NOTE: this was derived from re-compressed copies of the photos (uploaded
via chat, ~150 kB each). Re-run on the camera originals, ideally with the
spreadsheet so that Upper and Under surfaces get their own limits.

## Known limitations

* Deposits larger than ~2x `envelope_radius_px` (40 px, ~7 mm) or covering
  more than ~50% of a leaf start to be absorbed into the background estimate;
  very faint films below ~6 corrected grey levels are not detected.
* The earlier `local` (median) method under-estimated heavily sprayed leaves
  3-4 fold, because deposits raised the leaf median and noise estimate.
* Unsprayed background differs ~2-fold between leaf surfaces (higher on the
  underside), so an absolute intensity threshold is not used by default.
* Deposits on veins are kept (spray collects there); occasional false
  positives along midribs remain and are covered by the detection limit.

## Imaging recommendations

* Fix camera settings: manual exposure, ISO, aperture and **white balance**;
  shoot RAW (or TIFF) if possible. Auto-exposure/auto-WB change channel
  values between images.
* Expose so that leaf background is well above ~50/255 but deposits do not
  saturate; check the `leaf_raw_blue_median` and warnings columns.
* Focus on the leaf surface; blur spreads small deposits below threshold.
* Leaves must not touch or overlap each other or the card edge
  (overlaps are flagged by low `solidity`).
* Image the same leaf surface (adaxial or abaxial) throughout, or record it:
  unsprayed-leaf background differed ~2-fold between control image sets.
* Include untreated control leaves in every imaging session.
