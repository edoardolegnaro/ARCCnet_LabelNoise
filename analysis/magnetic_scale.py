"""Estimate the MDI/HMI field ratio and weak-field noise on paired cutouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from arccnet.models import preprocessing_common as pp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("processed", type=Path)
parser.add_argument("overlap_rows", type=Path)
parser.add_argument("data_folder", type=Path)
args = parser.parse_args()
indices = json.loads(args.overlap_rows.read_text())["mdi_view_source_index"]
rows = pd.read_parquet(args.processed).loc[indices]
measurements = []
for row in rows.itertuples():
    fields = []
    for path in (row.path_image_cutout_hmi, row.path_image_cutout_mdi):
        fits = pp.resolve_cutout_fits_path(
            path, data_folder=args.data_folder, dataset_folder="arccnet-ar-classification-v20251016"
        )
        fields.append(np.abs(np.nan_to_num(pp.load_fits_hdu_data(fits).astype(float))))
    hmi, mdi = fields
    # The two cutouts cover the same solid angle with different pixel areas.
    flux_ratio = mdi[mdi > 100].sum() * (hmi.size / mdi.size) / hmi[hmi > 100].sum()
    measurements.append((flux_ratio, 1.4826 * np.median(hmi[hmi < 50]), 1.4826 * np.median(mdi[mdi < 50])))
print(pd.DataFrame(measurements, columns=["flux_ratio", "hmi_noise_G", "mdi_noise_G"]).median())
