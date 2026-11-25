from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Summarize geometry runs and ensemble predictions.")
parser.add_argument("revision", type=Path)
parser.add_argument("reference_predictions", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
REV = args.revision
MODEL = "swin_tiny_patch4_window7_224.ms_in1k"
PAPER_DIAG = args.reference_predictions
CLASSES = ["Alpha", "Beta", "Beta-Gamma"]
OUT = args.output


def macro_f1(true, pred) -> float:
    true, pred = np.asarray(true), np.asarray(pred)
    scores = []
    for c in CLASSES:
        tp = np.sum((true == c) & (pred == c))
        denom = 2 * tp + np.sum((true != c) & (pred == c)) + np.sum((true == c) & (pred != c))
        if denom:
            scores.append(2 * tp / denom)
    return float(np.mean(scores)) if scores else float("nan")


def fold_scores(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.Series:
    return df.groupby("fold").apply(lambda g: macro_f1(g[true_col], g[pred_col]), include_groups=False)


def loader_instrument(df: pd.DataFrame) -> pd.Series:
    """HMI when the preferred HMI cutout path is set (image-loader rule), else MDI."""
    hmi = df["path_image_cutout_hmi"].fillna("").astype(str).str.strip() != ""
    return pd.Series(np.where(hmi, "HMI", "MDI"), index=df.index)


def describe(name: str, scores: pd.Series) -> dict:
    return {
        "run": name,
        "n_folds": int(scores.notna().sum()),
        "macro_f1_mean": scores.mean(),
        "macro_f1_min": scores.min(),
        "macro_f1_max": scores.max(),
    }


paper = pd.read_csv(PAPER_DIAG)
paper["instrument_loader"] = loader_instrument(paper)
baseline = fold_scores(paper, "true_label", "pred_label")
baseline_inst = {
    inst: fold_scores(paper[paper.instrument_loader == inst], "true_label", "pred_label") for inst in ("HMI", "MDI")
}


def load_test_predictions(run_root: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(run_root / MODEL / "both/fold_*/logs/fold_*_evaluation/test_predictions.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["true_label"] = df["target"].map(dict(enumerate(CLASSES)))
    df["pred_label"] = df["prediction"].map(dict(enumerate(CLASSES)))
    return df


rows = [describe("paper (seed 42, 112x224 keep-ratio)", baseline)]
rows += [describe(f"paper on {inst} rows", s) for inst, s in baseline_inst.items()]
experiments = [
    "instrument/hmi_only_112x224",
    "instrument/hmi_only_224x448",
    "resolution/both_224x448",
    "aspect_ratio/square_pad_224x224",
    "aspect_ratio/square_stretch_224x224",
]
for name in experiments:
    preds = load_test_predictions(REV / name)
    if preds is None:
        rows.append({"run": name, "n_folds": 0})
        continue
    scores = fold_scores(preds, "true_label", "pred_label")
    is_hmi_only = name.startswith("instrument/hmi_only")
    reference = baseline_inst["HMI"] if is_hmi_only else baseline
    paired = (scores - reference.reindex(scores.index)).dropna()
    row = describe(name, scores)
    row.update(
        {
            "reference": "paper on HMI rows" if is_hmi_only else "paper",
            "delta_mean_vs_reference": paired.mean(),
            "folds_better": int((paired > 0).sum()),
        }
    )
    rows.append(row)
    if not is_hmi_only:
        for inst in ("HMI", "MDI"):
            sub = fold_scores(preds[preds.instrument == inst], "true_label", "pred_label")
            d = (sub - baseline_inst[inst].reindex(sub.index)).dropna()
            r = describe(f"{name} on {inst} rows", sub)
            r.update(
                {
                    "reference": f"paper on {inst} rows",
                    "delta_mean_vs_reference": d.mean(),
                    "folds_better": int((d > 0).sum()),
                }
            )
            rows.append(r)

OUT.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "r1_2_input_and_instrument_summary.csv", index=False)
print("\n=== R1.2: input policy and instrument (fold-mean macro-F1; paired deltas on identical folds) ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def load_members(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    members = {}
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(
                path,
                usecols=[
                    "source_index",
                    "fold",
                    "true_label",
                    "pred_label",
                    *[f"prob_{c.lower().replace('-', '_')}" for c in CLASSES],
                ],
            )
            members[name] = df.set_index("source_index").sort_index()
    return members


def ensemble_report(label: str, members: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    if len(members) < 2:
        print(f"\n=== {label}: {len(members)} member(s) available; need at least 2 ===")
        return None
    index = paper.set_index("source_index").sort_index().index
    for name, member in members.items():
        if not member.index.is_unique or not member.index.equals(index):
            raise ValueError(f"{name} does not cover the reference observations")
        for column in ("fold", "true_label"):
            if not member[column].equals(paper.set_index("source_index").loc[index, column]):
                raise ValueError(f"{name} disagrees with the reference {column}")
    ref = paper.set_index("source_index").loc[index]
    prob_cols = [f"prob_{c.lower().replace('-', '_')}" for c in CLASSES]
    probs = np.mean([m.loc[index, prob_cols].to_numpy() for m in members.values()], axis=0)
    preds = np.stack([m.loc[index, "pred_label"].to_numpy() for m in members.values()], axis=1)
    true = ref["true_label"].to_numpy()

    out = pd.DataFrame(index=pd.Index(index, name="source_index"))
    out["fold"] = ref["fold"].to_numpy()
    out["true_label"] = true
    out["ensemble_pred"] = np.array(CLASSES)[probs.argmax(axis=1)]
    out["ensemble_conf"] = probs.max(axis=1)
    out["n_members"] = preds.shape[1]
    out["n_agree_label"] = (preds == true[:, None]).sum(axis=1)
    out["unanimous_wrong"] = (preds == preds[:, :1]).all(axis=1) & (preds[:, 0] != true)
    out["unanimous_pred"] = np.where(out["unanimous_wrong"], preds[:, 0], "")
    for col in (
        "instrument_loader",
        "is_near_label_transition",
        "nearest_other_label",
        "nearest_other_label_days",
        "is_high_conf_error",
        "number",
        "dates",
    ):
        out[col] = ref[col].to_numpy()
    # Check both adjacent catalogue days, including ties with different labels.
    catalogue = pd.MultiIndex.from_arrays(
        [paper["number"], pd.to_datetime(paper["dates"]).dt.normalize(), paper["true_label"]]
    )
    days = pd.to_datetime(out["dates"]).dt.normalize()
    matches = np.zeros(len(out), dtype=bool)
    for offset in (-1, 1):
        candidates = pd.MultiIndex.from_arrays([out["number"], days + pd.Timedelta(days=offset), out["unanimous_pred"]])
        matches |= candidates.isin(catalogue)
    out["unanimous_matches_adjacent_label_1d"] = out["unanimous_wrong"] & matches

    member_scores = {
        n: fold_scores(m.loc[index].reset_index(), "true_label", "pred_label").mean() for n, m in members.items()
    }
    ens_scores = fold_scores(out.reset_index(), "true_label", "ensemble_pred")
    errors = out[out["true_label"] != out["ensemble_pred"]]
    unan = out[out["unanimous_wrong"]]
    print(f"\n=== {label}: {len(members)} members, {len(out):,} observations ===")
    print("member fold-mean macro-F1:", {k: round(v, 4) for k, v in member_scores.items()})
    print(
        f"mean-probability ensemble macro-F1: {ens_scores.mean():.4f} [{ens_scores.min():.4f}-{ens_scores.max():.4f}]"
    )
    print(
        f"label agreed by all members: {(out.n_agree_label == out.n_members).mean():.1%}; by none: {(out.n_agree_label == 0).mean():.1%}"
    )
    print(
        f"unanimous disagreement with the catalogue (all members predict the same other class): {len(unan):,} ({len(unan) / len(out):.1%})"
    )
    if len(unan):
        print("  true -> predicted:", unan.groupby(["true_label", "unanimous_pred"]).size().to_dict())
        print(
            f"  near a label transition (2 d): {unan.is_near_label_transition.mean():.1%}  vs all ensemble errors {errors.is_near_label_transition.mean():.1%}  vs all obs {out.is_near_label_transition.mean():.1%}"
        )
        print(
            f"  predicted class = catalogue class of same NOAA region within 1 d: {unan.unanimous_matches_adjacent_label_1d.mean():.1%}"
        )
        print(
            f"  instrument: {unan.instrument_loader.value_counts(normalize=True).round(3).to_dict()}  (all obs {out.instrument_loader.value_counts(normalize=True).round(3).to_dict()})"
        )
        print(
            f"  also a seed-42 high-confidence error: {unan.is_high_conf_error.mean():.1%}; mean ensemble confidence {unan.ensemble_conf.mean():.3f}"
        )
    return out


seed_paths = {"seed42": PAPER_DIAG}
seed_paths.update(
    {f"seed{s}": REV / f"seed_ensemble/oof/seed_{s}/label_noise_diagnostics.csv" for s in (43, 44, 45, 46)}
)
seeds = load_members(seed_paths)
if len(seeds) != 5:
    raise ValueError("Expected the reference and four additional seeds")
seed_table = ensemble_report("Seed ensemble", seeds)
if seed_table is not None:
    seed_table.to_csv(OUT / "r1_3_seed_ensemble_per_observation.csv")

committee_paths = {
    Path(p).parent.name: Path(p) for p in sorted(glob.glob(str(REV / "committee/*/label_noise_diagnostics.csv")))
}
committee_table = ensemble_report(
    "R1.3 architecture committee (top-10 paper configurations)", load_members(committee_paths)
)
if len(committee_paths) != 10:
    raise ValueError("Expected ten committee members")
if committee_table is not None:
    committee_table.to_csv(OUT / "r1_3_committee_per_observation.csv")
