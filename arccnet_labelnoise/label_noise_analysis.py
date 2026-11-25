"""Label-noise diagnostics for Hale paper sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import arccnet_labelnoise.config as config
from arccnet_labelnoise.dataset import HaleDataset, get_fold_data
from arccnet_labelnoise.lightning_model import HaleLightningModel
from arccnet_labelnoise.paper_training import (
    apply_hale_config,
    dataset_cache_path,
    safe_job_component,
)

DEFAULT_METADATA_COLUMNS = (
    "number",
    "dates",
    "date_only",
    "time",
    "target_time.jd1",
    "target_time.jd2",
    "latitude_hmi",
    "longitude_hmi",
    "latitude_mdi",
    "longitude_mdi",
    "area",
    "number_of_sunspots",
    "magnetic_class",
    "mcintosh_class",
    "label",
    "grouped_labels",
    "path_image_cutout_hmi",
    "path_image_cutout_mdi",
    "processed_path_image_mag_hmi",
    "processed_path_image_mag_mdi",
    "processed_path_image_cont_hmi",
    "processed_path_image_cont_mdi",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Completed Hale paper sweep root.")
    parser.add_argument("--model-name", default="swin_tiny_patch4_window7_224.ms_in1k")
    parser.add_argument("--data-type", default="both", choices=["magnetogram", "continuum", "both"])
    parser.add_argument("--classes", default="a-b-bg")
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--folds", default=None, help="Comma/range fold subset, e.g. 1 or 1-3. Default: all folds.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-folder", default=config.DATA_FOLDER)
    parser.add_argument("--dataset-folder", default=config.DATASET_FOLDER)
    parser.add_argument("--df-file-name", default=config.DF_FILE_NAME)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--low-margin-threshold", type=float, default=0.10)
    parser.add_argument("--transition-days", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke checks.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run-root>/paper_metrics/label_noise_analysis/<model>/<data_type>",
    )
    parser.add_argument("--paired-mdi", action="store_true", help="Use the complete MDI view of paired observations.")
    return parser.parse_args()


def _parse_fold_spec(spec: str | None, n_folds: int) -> list[int]:
    if not spec:
        return list(range(1, n_folds + 1))
    folds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    invalid = [fold for fold in folds if fold < 1 or fold > n_folds]
    if invalid:
        raise ValueError(f"Invalid fold(s) {invalid}; expected 1-{n_folds}")
    return list(dict.fromkeys(folds))


def _configure(args: argparse.Namespace) -> None:
    config_args = SimpleNamespace(
        classes=args.classes,
        model_name=args.model_name,
        data_type=args.data_type,
        max_epochs=config.MAX_EPOCHS,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_folds=args.n_folds,
        random_state=args.random_state,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        monitor=config.CHECKPOINT_MONITOR,
        monitor_mode=config.CHECKPOINT_MODE,
        optimizer=config.OPTIMIZER,
        loss_policy=config.LOSS_POLICY,
        focal_loss_gamma=config.FOCAL_LOSS_GAMMA,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        auto_lr_find=False,
        lr_find_min_lr=config.LR_FIND_MIN_LR,
        lr_find_max_lr=config.LR_FIND_MAX_LR,
        lr_find_num_training=config.LR_FIND_NUM_TRAINING,
        transformer_rgb_adapter=True,
        transformer_imagenet_normalize=True,
        accelerator="gpu",
        devices=1,
        data_folder=args.data_folder,
        dataset_folder=args.dataset_folder,
        df_file_name=args.df_file_name,
    )
    apply_hale_config(config_args, data_type=args.data_type)
    config.ENABLE_TENSORBOARD = False
    config.ENABLE_CSV = False
    config.ENABLE_COMET = False
    config.NUM_WORKERS = args.num_workers
    config.BATCH_SIZE = args.batch_size


def _load_dataset() -> pd.DataFrame:
    path = dataset_cache_path()
    if not path.exists():
        raise FileNotFoundError(f"Processed Hale dataset cache not found: {path}")
    df = pd.read_parquet(path)
    if "model_labels" not in df.columns:
        if "grouped_labels" not in df.columns:
            raise ValueError("Processed dataset has neither 'model_labels' nor 'grouped_labels'.")
        label_to_index = {label: idx for idx, label in enumerate(config.class_names)}
        df = df.copy()
        df["model_labels"] = df["grouped_labels"].map(label_to_index)
        missing = df["model_labels"].isna()
        if missing.any():
            labels = sorted(str(value) for value in df.loc[missing, "grouped_labels"].dropna().unique())
            raise ValueError(f"Found grouped_labels not present in configured class_names: {labels}")
        df["model_labels"] = df["model_labels"].astype(int)
    return df


def _load_evaluation(run_root: Path, model_name: str, data_type: str, fold: int) -> dict:
    eval_path = (
        run_root
        / safe_job_component(model_name)
        / data_type
        / f"fold_{fold:02d}"
        / "logs"
        / f"fold_{fold}_evaluation"
        / "evaluation.json"
    )
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")
    return json.loads(eval_path.read_text())


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def _metadata_for_row(row: pd.Series) -> dict:
    payload = {}
    for column in DEFAULT_METADATA_COLUMNS:
        if column in row.index:
            payload[column] = row[column]
    return payload


def _infer_instrument(row: pd.Series) -> str:
    for instrument in ("hmi", "mdi"):
        path = row.get(f"path_image_cutout_{instrument}")
        if pd.notna(path) and str(path).strip():
            return instrument.upper()
    return "unknown"


def _predict_fold(
    run_root: Path,
    df: pd.DataFrame,
    model_name: str,
    data_type: str,
    fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None = None,
) -> list[dict]:
    evaluation = _load_evaluation(run_root, model_name, data_type, fold)
    checkpoint = evaluation["best_checkpoint"]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Best checkpoint from evaluation does not exist: {checkpoint}")

    config.MODEL_NAME = model_name
    config.DATA_TYPE = data_type
    model = HaleLightningModel.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    _, _, test_df = get_fold_data(df, fold)
    dataset = HaleDataset(test_df, data_type=data_type, is_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = evaluation["class_names"]
    rows: list[dict] = []
    offset = 0
    stop = False
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            preds_np = probabilities.argmax(axis=1)
            sorted_probs = np.sort(probabilities, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]

            for i in range(len(labels_np)):
                if max_samples is not None and len(rows) >= max_samples:
                    stop = True
                    break
                source = test_df.iloc[offset + i]
                true_idx = int(labels_np[i])
                pred_idx = int(preds_np[i])
                row = {
                    "fold": fold,
                    "test_row_position": offset + i,
                    "source_index": source.name,
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "true_label": class_names[true_idx],
                    "pred_label": class_names[pred_idx],
                    "correct": pred_idx == true_idx,
                    "confidence": float(probabilities[i, pred_idx]),
                    "top2_margin": float(margins[i]),
                    "prob_alpha": float(probabilities[i, 0]),
                    "prob_beta": float(probabilities[i, 1]),
                    "prob_beta_gamma": float(probabilities[i, 2]),
                    "instrument": _infer_instrument(source),
                }
                row.update(_metadata_for_row(source))
                rows.append(row)
            offset += len(labels_np)
            if stop:
                break
    return rows


def _add_diagnostic_flags(
    df: pd.DataFrame, high_conf: float, low_margin: float, transition_days: float
) -> pd.DataFrame:
    out = df.copy()
    out["is_high_conf_error"] = (~out["correct"]) & (out["confidence"] >= high_conf)
    out["is_low_margin"] = out["top2_margin"] <= low_margin
    beta_bg = {"Beta", "Beta-Gamma"}
    out["is_beta_beta_gamma_flip"] = ~out["correct"] & out["true_label"].isin(beta_bg) & out["pred_label"].isin(beta_bg)

    if "time" in out.columns:
        out["_time_numeric"] = pd.to_numeric(out["time"], errors="coerce")
    else:
        out["_time_numeric"] = np.nan
    out["nearest_other_label_days"] = np.nan
    out["nearest_other_label"] = ""
    out["is_near_label_transition"] = False

    for _, group in out.dropna(subset=["_time_numeric"]).groupby("number", dropna=True):
        ordered = group.sort_values("_time_numeric")
        labels = ordered["true_label"].to_numpy()
        times = ordered["_time_numeric"].to_numpy(dtype=float)
        indices = ordered.index.to_numpy()
        for pos, idx in enumerate(indices):
            different = np.flatnonzero(labels != labels[pos])
            if different.size == 0:
                continue
            nearest_pos = different[np.argmin(np.abs(times[different] - times[pos]))]
            days = abs(float(times[nearest_pos] - times[pos]))
            out.loc[idx, "nearest_other_label_days"] = days
            out.loc[idx, "nearest_other_label"] = labels[nearest_pos]
            out.loc[idx, "is_near_label_transition"] = days <= transition_days

    out["label_noise_score"] = (
        out["is_high_conf_error"].astype(int) * 3
        + out["is_beta_beta_gamma_flip"].astype(int) * 3
        + out["is_near_label_transition"].astype(int) * 2
        + out["is_low_margin"].astype(int)
    )
    return out.drop(columns=["_time_numeric"])


def _write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for name, mask in {
        "all": pd.Series(True, index=df.index),
        "errors": ~df["correct"],
        "high_conf_errors": df["is_high_conf_error"],
        "low_margin": df["is_low_margin"],
        "beta_beta_gamma_flips": df["is_beta_beta_gamma_flip"],
        "near_label_transition": df["is_near_label_transition"],
    }.items():
        sub = df.loc[mask]
        rows.append(
            {
                "group": name,
                "n": len(sub),
                "fraction": len(sub) / len(df) if len(df) else 0.0,
                "mean_confidence": sub["confidence"].mean() if len(sub) else np.nan,
                "mean_top2_margin": sub["top2_margin"].mean() if len(sub) else np.nan,
                "beta_beta_gamma_flip_count": int(sub["is_beta_beta_gamma_flip"].sum()) if len(sub) else 0,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "label_noise_summary.csv", index=False)

    temporal = (
        df.groupby("number", dropna=False)
        .agg(
            n_samples=("number", "size"),
            n_true_labels=("true_label", "nunique"),
            n_pred_labels=("pred_label", "nunique"),
            n_errors=("correct", lambda s: int((~s).sum())),
            n_high_conf_errors=("is_high_conf_error", "sum"),
            n_beta_beta_gamma_flips=("is_beta_beta_gamma_flip", "sum"),
            n_near_label_transition=("is_near_label_transition", "sum"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
        .sort_values(["n_high_conf_errors", "n_beta_beta_gamma_flips", "n_errors"], ascending=False)
    )
    temporal.to_csv(output_dir / "temporal_region_summary.csv", index=False)


def main() -> None:
    args = _parse_args()
    _configure(args)
    run_root = Path(args.run_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_root / "paper_metrics" / "label_noise_analysis" / safe_job_component(args.model_name) / args.data_type
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _choose_device(args.device)
    dataset = _load_dataset()
    if args.paired_mdi:
        available = lambda key: dataset[key].fillna("").str.strip().ne("")
        both = available("path_image_cutout_hmi") & available("path_image_cutout_mdi")
        paired = dataset.loc[both].copy()
        paired["path_image_cutout_hmi"] = ""
        if "path_image_cutout_hmi.mask" in paired:
            paired["path_image_cutout_hmi.mask"] = True
        from .data_preparation import complete_fov_mask

        indices = paired.index[complete_fov_mask(paired)]
        dataset = dataset.drop(index=paired.index.difference(indices))
        for key in ("path_image_cutout_hmi", "path_image_cutout_hmi.mask"):
            if key in paired:
                dataset.loc[indices, key] = paired.loc[indices, key]
        (output_dir / "overlap_mdi_view_rows.json").write_text(
            json.dumps(
                {
                    "overlap_rows": len(paired),
                    "mdi_view_source_index": indices.tolist(),
                    "mdi_view_rows": len(indices),
                    "dropped_incomplete_mdi": len(paired) - len(indices),
                }
            )
        )
    all_rows: list[dict] = []
    for fold in _parse_fold_spec(args.folds, args.n_folds):
        rows = _predict_fold(
            run_root,
            dataset,
            args.model_name,
            args.data_type,
            fold,
            device,
            args.batch_size,
            args.num_workers,
            max_samples=max(0, args.max_samples - len(all_rows)) if args.max_samples is not None else None,
        )
        base = len(all_rows)
        for idx, row in enumerate(rows):
            row["global_row_position"] = base + idx
        all_rows.extend(rows)
        if args.max_samples is not None and len(all_rows) >= args.max_samples:
            break

    diagnostics = _add_diagnostic_flags(
        pd.DataFrame(all_rows),
        high_conf=args.high_confidence_threshold,
        low_margin=args.low_margin_threshold,
        transition_days=args.transition_days,
    )
    diagnostics = diagnostics.sort_values(
        ["label_noise_score", "is_high_conf_error", "is_beta_beta_gamma_flip", "confidence"],
        ascending=[False, False, False, False],
    )
    diagnostics.to_csv(output_dir / "label_noise_diagnostics.csv", index=False)
    diagnostics.loc[diagnostics["is_high_conf_error"]].to_csv(output_dir / "high_confidence_errors.csv", index=False)
    diagnostics.loc[diagnostics["is_low_margin"]].to_csv(output_dir / "low_margin_cases.csv", index=False)
    diagnostics.loc[diagnostics["is_beta_beta_gamma_flip"]].to_csv(
        output_dir / "beta_beta_gamma_flips.csv", index=False
    )
    diagnostics.loc[diagnostics["is_near_label_transition"]].to_csv(
        output_dir / "near_label_transition_cases.csv", index=False
    )
    _write_summary(diagnostics, output_dir)

    print(f"Rows analyzed: {len(diagnostics)}")
    print(f"Output directory: {output_dir}")
    print(f"Diagnostics: {output_dir / 'label_noise_diagnostics.csv'}")
    print(f"Summary: {output_dir / 'label_noise_summary.csv'}")


if __name__ == "__main__":
    main()
