import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Evaluate single-instrument transfer at fixed NOAA folds.")
parser.add_argument("revision", type=Path)
parser.add_argument("reference_predictions", type=Path)
args = parser.parse_args()
REV, PAPER = args.revision, args.reference_predictions
C = ["Alpha", "Beta", "Beta-Gamma"]


def mf1(t, p):
    t, p = np.asarray(t), np.asarray(p)
    s = []
    for c in C:
        tp = ((t == c) & (p == c)).sum()
        d = 2 * tp + ((t != c) & (p == c)).sum() + ((t == c) & (p != c)).sum()
        if d:
            s.append(2 * tp / d)
    return float(np.mean(s))


def inst(df):
    return np.where(df.path_image_cutout_hmi.fillna("").astype(str).str.strip() != "", "HMI", "MDI")


def fold_means(df):
    out = {}
    for i in ("HMI", "MDI"):
        sub = df[df.inst == i]
        f = sub.groupby("fold").apply(lambda g: mf1(g.true_label, g.pred_label), include_groups=False)
        out[i] = (f.mean(), f.min(), f.max(), (sub.true_label != sub.pred_label).mean(), f)
    return out


models = {"both instruments (paper, seed 42)": PAPER}
for s in (43, 44, 45, 46):
    p = REV / f"seed_ensemble/oof/seed_{s}/label_noise_diagnostics.csv"
    if p.exists():
        models[f"both instruments (seed {s})"] = p
for run in ("hmi_only_112x224", "mdi_only_112x224"):
    p = REV / f"instrument/cross_eval/{run}/label_noise_diagnostics.csv"
    if p.exists():
        models[run] = p

rows, folds = [], {}
for name, p in models.items():
    df = pd.read_csv(p)
    df["inst"] = inst(df)
    r = fold_means(df)
    folds[name] = {i: r[i][4] for i in r}
    rows.append(
        {
            "model": name,
            "n": len(df),
            "HMI_macro_f1": r["HMI"][0],
            "HMI_range": f"{r['HMI'][1]:.3f}-{r['HMI'][2]:.3f}",
            "HMI_error": r["HMI"][3],
            "MDI_macro_f1": r["MDI"][0],
            "MDI_range": f"{r['MDI'][1]:.3f}-{r['MDI'][2]:.3f}",
            "MDI_error": r["MDI"][3],
        }
    )
    # consistency: re-inferred own-instrument predictions vs the training run's test_predictions.csv
    run = name if name.endswith("_112x224") else None
    if run:
        own = "HMI" if run.startswith("hmi") else "MDI"
        tp = sorted(
            glob.glob(
                str(
                    REV
                    / f"instrument/{run}/swin_tiny_patch4_window7_224.ms_in1k/both/fold_*/logs/fold_*_evaluation/test_predictions.csv"
                )
            )
        )
        if tp:
            t = pd.concat([pd.read_csv(f) for f in tp]).set_index("source_index")
            m = df.set_index("source_index").loc[t.index]
            agree = (m.pred_idx.to_numpy() == t.prediction.to_numpy()).mean()
            print(
                f"{run}: re-inferred {own} predictions agree with training-run test predictions for {agree:.4%} of {len(t)} rows"
            )
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
ref = "both instruments (paper, seed 42)"
for name in folds:
    if name == ref:
        continue
    for i in ("HMI", "MDI"):
        d = (folds[name][i] - folds[ref][i]).dropna()
        print(f"{name:34s} on {i}: delta vs paper {d.mean():+.4f} (folds better {(d > 0).sum()}/8)")
