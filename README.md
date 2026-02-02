# Leaf Spray Coverage Analysis

Python-based image analysis pipeline for quantifying fluorescent spray
coverage on apple leaves.
Intended for use with a fluorescent tracer added into spray tanks, e.g. DropSight's UView, or Niab's edible fluorescent tracer.
Leaves or other materials should be photographed using the dedicated imaging box, with built-in UV LEDs, and a narrow band pass filter.
Use a ruler or similar to record a calibration image.

## Project structure

SRC/ Python analysis scripts
DATA/raw/ Raw fluorescent images (not tracked)
DATA/processed/ Processed/masked images (not tracked)
DATA/outputs/ CSV outputs and summary tables


## Requirements

- Python ≥ 3.10
- OpenCV
- NumPy
- pandas
- scikit-image

Install dependencies:

```bash
pip install -r requirements.txt



