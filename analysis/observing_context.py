"""Measure the association of cutout context and temporal smoothing with held-out errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("predictions", type=Path)
parser.add_argument("processed", type=Path)
parser.add_argument("catalogue", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
DIAG, PROCESSED, CATALOG, OUT = args.predictions, args.processed, args.catalogue, args.output
CLASSES = ["Alpha", "Beta", "Beta-Gamma"]
N_BOOT = 1000
RNG = np.random.default_rng(42)


def macro_f1(true, pred) -> float:
    true, pred = np.asarray(true), np.asarray(pred)
    scores = []
    for c in CLASSES:
        tp = np.sum((true == c) & (pred == c))
        denom = 2 * tp + np.sum((true != c) & (pred == c)) + np.sum((true == c) & (pred != c))
        if denom:
            scores.append(2 * tp / denom)
    return float(np.mean(scores))


def box_center_and_half(df: pd.DataFrame, inst: str) -> tuple[np.ndarray, np.ndarray]:
    tr = np.vstack(df[f"top_right_cutout_{inst}"].to_numpy()).astype(float)
    bl = np.vstack(df[f"bottom_left_cutout_{inst}"].to_numpy()).astype(float)
    return (tr + bl) / 2.0, (tr - bl) / 2.0


diag = pd.read_csv(DIAG)
proc = pd.read_parquet(PROCESSED)
df = diag.set_index("source_index").join(
    proc[
        [
            "top_right_cutout_hmi",
            "bottom_left_cutout_hmi",
            "top_right_cutout_mdi",
            "bottom_left_cutout_mdi",
            "latitude_hmi.mask",
            "longitude_hmi.mask",
            "latitude_mdi.mask",
            "longitude_mdi.mask",
            "target_time.jd1",
            "target_time.jd2",
            "number",
        ]
    ].rename(columns={"number": "number_proc", "target_time.jd1": "jd1", "target_time.jd2": "jd2"}),
    how="left",
)
assert (df["number"] == df["number_proc"]).all(), "diagnostics/processed join mismatch"
df["inst"] = np.where(df["path_image_cutout_hmi"].fillna("").astype(str).str.strip() != "", "hmi", "mdi")
df["error"] = (df["true_label"] != df["pred_label"]).astype(int)

# position in the instrument actually used by the image loader (fall back if masked)
use_hmi = (df["inst"] == "hmi") & ~df["longitude_hmi.mask"].astype(bool)
use_mdi = ~use_hmi & ~df["longitude_mdi.mask"].astype(bool)
df["lat"] = np.where(use_hmi, df["latitude_hmi"], np.where(use_mdi, df["latitude_mdi"], np.nan))
df["lon"] = np.where(use_hmi, df["longitude_hmi"], np.where(use_mdi, df["longitude_mdi"], np.nan))
print(f"rows {len(df):,}; position available {df.lon.notna().mean():.4f}; |lon| max {df.lon.abs().max():.1f}")

cat = pd.read_parquet(CATALOG)
cat = cat[cat["region_type"].isin(["AR", "IA"])].copy()
counts = {"n_ar_in_fov": np.zeros(len(df), int), "n_ar_ia_in_fov": np.zeros(len(df), int)}
df = df.reset_index()
for inst in ("hmi", "mdi"):
    rows = df[df["inst"] == inst]
    c_cent, _ = box_center_and_half(cat, inst)
    valid = np.abs(c_cent).sum(axis=1) > 0
    cat_i = cat.loc[valid, ["number", "region_type", "target_time.jd1", "target_time.jd2"]].copy()
    cat_i["cx"], cat_i["cy"] = c_cent[valid, 0], c_cent[valid, 1]
    groups = {k: g for k, g in cat_i.groupby(["target_time.jd1", "target_time.jd2"])}
    r_cent, r_half = box_center_and_half(rows, inst)
    for j, (idx, row) in enumerate(rows.iterrows()):
        g = groups.get((row["jd1"], row["jd2"]))
        if g is None:
            continue
        g = g[g["number"] != row["number"]]
        inside = (np.abs(g["cx"].to_numpy() - r_cent[j, 0]) <= r_half[j, 0]) & (
            np.abs(g["cy"].to_numpy() - r_cent[j, 1]) <= r_half[j, 1]
        )
        counts["n_ar_ia_in_fov"][idx] = int(inside.sum())
        counts["n_ar_in_fov"][idx] = int((inside & (g["region_type"].to_numpy() == "AR")).sum())
for k, v in counts.items():
    df[k] = v
same_day = cat[cat.region_type == "AR"].groupby(["target_time.jd1", "target_time.jd2"]).size()
print(f"AR regions per catalogue day: median {same_day.median():.0f}, max {same_day.max()}")


def standardized_table(frame: pd.DataFrame, bin_col: str, label: str) -> pd.DataFrame:
    weights = frame["true_label"].value_counts(normalize=True).reindex(CLASSES)
    bins = [b for b in frame[bin_col].cat.categories if (frame[bin_col] == b).any()]
    regions = frame["number"].unique()
    r_index = {r: i for i, r in enumerate(regions)}
    ri = frame["number"].map(r_index).to_numpy()
    bi = frame[bin_col].cat.codes.to_numpy()
    ci = frame["true_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    n = np.zeros((len(regions), len(frame[bin_col].cat.categories), 3))
    e = np.zeros_like(n)
    np.add.at(n, (ri, bi, ci), 1)
    np.add.at(e, (ri, bi, ci), frame["error"].to_numpy())

    def rates(nn: np.ndarray, ee: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class = ee / nn  # bins x classes
        w = np.where(nn > 0, weights.to_numpy()[None, :], 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        return np.nansum(per_class * w, axis=1)

    point = rates(n.sum(0), e.sum(0))
    boot = np.empty((N_BOOT, n.shape[1]))
    for b in range(N_BOOT):
        wts = RNG.multinomial(len(regions), np.full(len(regions), 1 / len(regions)))
        boot[b] = rates(np.tensordot(wts, n, axes=1), np.tensordot(wts, e, axes=1))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = []
    for k, cat_name in enumerate(frame[bin_col].cat.categories):
        if cat_name not in bins:
            continue
        sub = frame[frame[bin_col] == cat_name]
        out.append(
            {
                "factor": label,
                "bin": str(cat_name),
                "n_obs": len(sub),
                "share_alpha": (sub.true_label == "Alpha").mean(),
                "share_beta_gamma": (sub.true_label == "Beta-Gamma").mean(),
                "raw_error_rate": sub["error"].mean(),
                "std_error_rate": point[k],
                "std_ci_low": lo[k],
                "std_ci_high": hi[k],
            }
        )
    return pd.DataFrame(out)


df["abs_lon_bin"] = pd.cut(
    df["lon"].abs(), [0, 15, 30, 45, 60, 90], right=False, labels=["0-15", "15-30", "30-45", "45-60", ">=60"]
)
df["area_bin"] = pd.cut(df["area"], [-1, 30, 100, 300, 1e9], labels=["<=30", "31-100", "101-300", ">300"])
df["spots_bin"] = pd.cut(
    df["number_of_sunspots"], [-1, 2, 5, 10, 20, 1e9], labels=["1-2", "3-5", "6-10", "11-20", ">20"]
)
df["ar_fov_bin"] = pd.cut(df["n_ar_in_fov"], [-1, 0, 1, 1e9], labels=["0", "1", ">=2"])
df["ar_ia_fov_bin"] = pd.cut(df["n_ar_ia_in_fov"], [-1, 0, 1, 1e9], labels=["0", "1", ">=2"])
df["inst_bin"] = pd.Categorical(df["inst"].str.upper(), categories=["HMI", "MDI"])

tables = [
    standardized_table(df, "abs_lon_bin", "|longitude| (deg)"),
    standardized_table(df, "area_bin", "area (MSH)"),
    standardized_table(df, "spots_bin", "number of spots"),
    standardized_table(df, "ar_fov_bin", "other sunspot regions in cutout"),
    standardized_table(df, "ar_ia_fov_bin", "other sunspot regions or plages in cutout"),
    standardized_table(df, "inst_bin", "instrument"),
]
res = pd.concat(tables, ignore_index=True)
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "r1_4_context_factors.csv", index=False)
pd.set_option("display.width", 200)
print(f"\noverall error rate {df.error.mean():.4f}")
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

df["day"] = pd.to_datetime(df["date_only"])
probs = ["prob_alpha", "prob_beta", "prob_beta_gamma"]
base_fold = df.groupby("fold").apply(lambda g: macro_f1(g.true_label, g.pred_label), include_groups=False)
print("\nTemporal context (average a region's probabilities over neighbouring days):")
print(
    f"  single observation: fold-mean macro-F1 {base_fold.mean():.4f}; pooled {macro_f1(df.true_label, df.pred_label):.4f}; error rate {df.error.mean():.4f}"
)
rows_tc = [
    {
        "window_days": 0,
        "fold_mean_macro_f1": base_fold.mean(),
        "pooled_macro_f1": macro_f1(df.true_label, df.pred_label),
        "error_rate": df.error.mean(),
    }
]
for w in (1, 2, 3):
    smoothed = np.zeros((len(df), 3))
    for _, g in df.groupby(["fold", "number"]):
        d = g["day"].to_numpy()
        p = g[probs].to_numpy()
        near = np.abs(d[:, None] - d[None, :]) <= np.timedelta64(w, "D")
        smoothed[g.index.to_numpy()] = near @ p / near.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[smoothed.argmax(axis=1)]
    tmp = df.assign(pred_s=pred)
    fm = tmp.groupby("fold").apply(lambda g: macro_f1(g.true_label, g.pred_s), include_groups=False)
    err = (tmp.true_label != tmp.pred_s).mean()
    near_tr = tmp.is_near_label_transition.astype(bool)
    print(
        f"  +/-{w} d: fold-mean macro-F1 {fm.mean():.4f} (folds better {(fm > base_fold).sum()}/8); pooled {macro_f1(tmp.true_label, tmp.pred_s):.4f}; "
        f"error rate {err:.4f} (near transition {(tmp.true_label != tmp.pred_s)[near_tr].mean():.4f} vs {df.error[near_tr].mean():.4f}; "
        f"elsewhere {(tmp.true_label != tmp.pred_s)[~near_tr].mean():.4f} vs {df.error[~near_tr].mean():.4f})"
    )
    rows_tc.append(
        {
            "window_days": w,
            "fold_mean_macro_f1": fm.mean(),
            "pooled_macro_f1": macro_f1(tmp.true_label, tmp.pred_s),
            "error_rate": err,
        }
    )
pd.DataFrame(rows_tc).to_csv(OUT / "r1_4_temporal_context.csv", index=False)
df[
    [
        "source_index",
        "fold",
        "number",
        "inst",
        "lat",
        "lon",
        "area",
        "number_of_sunspots",
        "n_ar_in_fov",
        "n_ar_ia_in_fov",
        "error",
    ]
].to_csv(OUT / "r1_4_context_per_observation.csv", index=False)

crowded = df.n_ar_in_fov.gt(0)
print(df.groupby(["true_label", crowded.rename("crowded")]).error.agg(["size", "mean"]))
print(pd.crosstab([df.true_label, df.spots_bin], df.pred_label, normalize="index"))
