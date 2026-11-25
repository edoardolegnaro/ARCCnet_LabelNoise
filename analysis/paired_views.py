"""Compare paired HMI and MDI predictions for the same observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("hmi_predictions", type=Path)
parser.add_argument("mdi_predictions", type=Path)
parser.add_argument("overlap_rows", type=Path)
args = parser.parse_args()
indices = json.loads(args.overlap_rows.read_text())["mdi_view_source_index"]
hmi, mdi = [
    pd.read_csv(path).set_index("source_index", verify_integrity=True).loc[indices]
    for path in (args.hmi_predictions, args.mdi_predictions)
]
if not hmi[["number", "fold", "true_label"]].equals(mdi[["number", "fold", "true_label"]]):
    raise ValueError("The paired views disagree on region, fold or catalogue class")
rank = {"Alpha": 0, "Beta": 1, "Beta-Gamma": 2}
scores = (
    pd.DataFrame(
        {
            "accuracy": mdi.correct.astype(int) - hmi.correct.astype(int),
            "class_shift": mdi.pred_label.map(rank) - hmi.pred_label.map(rank),
            "n": 1,
        }
    )
    .groupby(hmi.number)
    .sum()
)
print("HMI and MDI accuracy:", hmi.correct.mean(), mdi.correct.mean())
print("Same predicted class:", hmi.pred_label.eq(mdi.pred_label).mean())
rng = np.random.default_rng(0)
samples = []
for _ in range(2000):
    resample = scores.iloc[rng.integers(len(scores), size=len(scores))].sum()
    samples.append(resample[["accuracy", "class_shift"]] / resample.n)
print("MDI minus HMI:", scores[["accuracy", "class_shift"]].sum() / scores.n.sum())
print("95% region-bootstrap intervals:", pd.DataFrame(samples).quantile([0.025, 0.975]))
