"""Simplex-threshold tuning for completed Hale paper runs.

The tuning rule follows the SimplexTools idea:

    prediction = argmax(probability - tau)

For each fold, ``tau`` is selected on the validation split and then applied to
the held-out test split. This keeps the analysis as post-processing and avoids
using test labels for threshold selection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
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
    CLASSIFICATION_METRIC_KEYS,
    apply_hale_config,
    dataset_cache_path,
    safe_job_component,
)
from arccnet_labelnoise.metrics import compute_classification_metric_report

SUPPORTED_OBJECTIVES = ("macro_f1", "f1", "accuracy", "balanced_accuracy")
DEFAULT_MODEL_NAME = "swin_tiny_patch4_window7_224.ms_in1k"
DEFAULT_DATA_TYPE = "both"


def parse_fold_spec(spec: str | None, n_folds: int) -> list[int]:
    """Parse a fold spec such as ``1-3,5`` into ordered 1-based fold numbers."""
    if not spec:
        return list(range(1, n_folds + 1))
    folds: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    unique = list(dict.fromkeys(folds))
    invalid = [fold for fold in unique if fold < 1 or fold > n_folds]
    if invalid:
        raise ValueError(f"Invalid fold(s) {invalid}; expected 1-{n_folds}")
    if not unique:
        raise ValueError("At least one fold must be selected.")
    return unique


def generate_simplex_grid(n_classes: int, resolution: int) -> np.ndarray:
    """Generate all non-negative simplex grid points at a fixed integer resolution."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    def compositions(total: int, parts: int) -> Iterable[list[int]]:
        if parts == 1:
            yield [total]
            return
        for value in range(total + 1):
            for rest in compositions(total - value, parts - 1):
                yield [value, *rest]

    grid = np.asarray(list(compositions(int(resolution), int(n_classes))), dtype=np.float64)
    return grid / float(resolution)


def simplex_predict(probs: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Predict classes with the simplex threshold rule ``argmax(probs - tau)``."""
    probs = np.asarray(probs, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D; got shape {probs.shape}")
    if tau.shape != (probs.shape[1],):
        raise ValueError(f"tau must have shape {(probs.shape[1],)}; got {tau.shape}")
    return np.argmax(probs - tau.reshape(1, -1), axis=1).astype(int)


def _objective_key(objective: str) -> str:
    key = str(objective).strip().lower()
    if key == "f1":
        return "macro_f1"
    if key not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unsupported objective {objective!r}; expected one of {SUPPORTED_OBJECTIVES}")
    return key


def _metric_report(labels: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict:
    return compute_classification_metric_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        class_names=class_names,
    )


def _tie_break_key(tau: np.ndarray) -> tuple[float, tuple[float, ...]]:
    uniform = np.full_like(tau, 1.0 / len(tau), dtype=np.float64)
    return (float(np.linalg.norm(tau - uniform)), tuple(float(value) for value in tau))


def tune_simplex_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: list[str],
    resolution: int = 40,
    objective: str = "macro_f1",
) -> dict:
    """Select the best simplex threshold vector on validation probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D; got shape {probs.shape}")
    if labels.ndim != 1 or len(labels) != len(probs):
        raise ValueError("labels must be 1D and match the number of probability rows.")
    if probs.shape[1] != len(class_names):
        raise ValueError("class_names length must match the number of probability columns.")

    metric_key = _objective_key(objective)
    tau_points = generate_simplex_grid(probs.shape[1], resolution)
    best_tau = tau_points[0]
    best_report = _metric_report(labels, simplex_predict(probs, best_tau), class_names)
    best_score = float(best_report[metric_key])
    best_tie = _tie_break_key(best_tau)
    scores = []

    for tau in tau_points:
        preds = simplex_predict(probs, tau)
        report = _metric_report(labels, preds, class_names)
        score = float(report[metric_key])
        scores.append(score)
        tie = _tie_break_key(tau)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tie < best_tie):
            best_tau = tau
            best_report = report
            best_score = score
            best_tie = tie

    return {
        "tau": best_tau,
        "objective": metric_key,
        "objective_value": best_score,
        "candidate_count": int(len(tau_points)),
        "resolution": int(resolution),
        "validation_report": best_report,
        "scores": scores,
        "tau_points": tau_points,
    }


def evaluate_tau(probs: np.ndarray, labels: np.ndarray, tau: np.ndarray, class_names: list[str]) -> dict:
    """Evaluate a fixed simplex threshold vector."""
    return _metric_report(labels, simplex_predict(probs, tau), class_names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Completed Hale paper sweep root.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-type", default=DEFAULT_DATA_TYPE, choices=["magnetogram", "continuum", "both"])
    parser.add_argument("--classes", default="a-b-bg")
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--folds", default=None, help="Comma/range fold subset, e.g. 1 or 1-3. Default: all folds.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-folder", default=config.DATA_FOLDER)
    parser.add_argument("--dataset-folder", default=config.DATASET_FOLDER)
    parser.add_argument("--df-file-name", default=config.DF_FILE_NAME)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--objective", default="macro_f1", choices=SUPPORTED_OBJECTIVES)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for smoke checks.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-predictions", action="store_true", help="Write per-sample validation/test CSV files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <run-root>/paper_metrics/simplex_tuning/<model>/<data_type>",
    )
    return parser.parse_args()


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
        delta_aux_weight=0.0,
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


def _split_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch[:2]
    return images, labels


def _predict_dataframe(
    model: HaleLightningModel,
    split_df: pd.DataFrame,
    *,
    data_type: str,
    split: str,
    fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> pd.DataFrame:
    dataset = HaleDataset(split_df, data_type=data_type, is_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    rows: list[dict] = []
    offset = 0
    stop = False
    with torch.no_grad():
        for batch in loader:
            images, labels = _split_batch(batch)
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds_np = probabilities.argmax(axis=1).astype(int)

            for i, label in enumerate(labels_np):
                if max_samples is not None and len(rows) >= max(0, int(max_samples)):
                    stop = True
                    break
                source = split_df.iloc[offset + i]
                row = {
                    "fold": int(fold),
                    "split": split,
                    "row_position": int(offset + i),
                    "source_index": int(source.name) if isinstance(source.name, int | np.integer) else str(source.name),
                    "true_idx": int(label),
                    "argmax_pred_idx": int(preds_np[i]),
                }
                for class_idx, class_name in enumerate(config.class_names):
                    safe_name = class_name.lower().replace("-", "_")
                    row[f"prob_{safe_name}"] = float(probabilities[i, class_idx])
                rows.append(row)
            offset += len(labels_np)
            if stop:
                break

    return pd.DataFrame(rows)


def _prob_columns(class_names: list[str]) -> list[str]:
    return [f"prob_{name.lower().replace('-', '_')}" for name in class_names]


def _run_fold(
    run_root: Path,
    dataset: pd.DataFrame,
    *,
    model_name: str,
    data_type: str,
    fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    resolution: int,
    objective: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    evaluation = _load_evaluation(run_root, model_name, data_type, fold)
    checkpoint = evaluation["best_checkpoint"]
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Best checkpoint from evaluation does not exist: {checkpoint}")

    config.MODEL_NAME = model_name
    config.DATA_TYPE = data_type
    class_names = list(evaluation["class_names"])
    model = HaleLightningModel.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    _, val_df, test_df = get_fold_data(dataset, fold)
    val_predictions = _predict_dataframe(
        model,
        val_df,
        data_type=data_type,
        split="val",
        fold=fold,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    test_predictions = _predict_dataframe(
        model,
        test_df,
        data_type=data_type,
        split="test",
        fold=fold,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )

    prob_cols = _prob_columns(class_names)
    val_probs = val_predictions[prob_cols].to_numpy(dtype=np.float64)
    test_probs = test_predictions[prob_cols].to_numpy(dtype=np.float64)
    val_labels = val_predictions["true_idx"].to_numpy(dtype=int)
    test_labels = test_predictions["true_idx"].to_numpy(dtype=int)

    baseline_tau = np.full(len(class_names), 1.0 / len(class_names), dtype=np.float64)
    baseline_val = evaluate_tau(val_probs, val_labels, baseline_tau, class_names)
    baseline_test = evaluate_tau(test_probs, test_labels, baseline_tau, class_names)
    tuning = tune_simplex_threshold(
        val_probs,
        val_labels,
        class_names=class_names,
        resolution=resolution,
        objective=objective,
    )
    tuned_tau = tuning["tau"]
    tuned_val = tuning["validation_report"]
    tuned_test = evaluate_tau(test_probs, test_labels, tuned_tau, class_names)
    test_predictions["simplex_pred_idx"] = simplex_predict(test_probs, tuned_tau)
    val_predictions["simplex_pred_idx"] = simplex_predict(val_probs, tuned_tau)

    row = {
        "fold": int(fold),
        "model_name": model_name,
        "data_type": data_type,
        "objective": _objective_key(objective),
        "resolution": int(resolution),
        "candidate_count": int(tuning["candidate_count"]),
        "n_val": int(len(val_predictions)),
        "n_test": int(len(test_predictions)),
        "best_checkpoint": checkpoint,
    }
    for idx, class_name in enumerate(class_names):
        row[f"tau_{class_name.lower().replace('-', '_')}"] = float(tuned_tau[idx])
    for prefix, report in {
        "baseline_val": baseline_val,
        "simplex_val": tuned_val,
        "baseline_test": baseline_test,
        "simplex_test": tuned_test,
    }.items():
        for key in CLASSIFICATION_METRIC_KEYS:
            row[f"{prefix}_{key}"] = float(report[key])

    return row, val_predictions, test_predictions


def _summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in fold_df.columns
        if (col.startswith("baseline_") or col.startswith("simplex_")) and pd.api.types.is_numeric_dtype(fold_df[col])
    ]
    rows = []
    for col in metric_cols:
        rows.append(
            {
                "metric": col,
                "mean": float(fold_df[col].mean()),
                "std": float(fold_df[col].std(ddof=1)) if len(fold_df) > 1 else float("nan"),
                "min": float(fold_df[col].min()),
                "max": float(fold_df[col].max()),
                "count": int(fold_df[col].count()),
            }
        )
    return pd.DataFrame(rows)


def _write_confusion_matrix(path: Path, report: dict) -> None:
    class_names = report["class_names"]
    cm = pd.DataFrame(report["confusion_matrix"], index=class_names, columns=class_names)
    cm.to_csv(path)


def _write_outputs(
    output_dir: Path,
    fold_rows: list[dict],
    all_val_predictions: list[pd.DataFrame],
    all_test_predictions: list[pd.DataFrame],
    *,
    class_names: list[str],
    save_predictions: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_simplex_metrics.csv", index=False)
    summary_df = _summarize_folds(fold_df)
    summary_df.to_csv(output_dir / "simplex_summary.csv", index=False)

    test_predictions = pd.concat(all_test_predictions, ignore_index=True)
    val_predictions = pd.concat(all_val_predictions, ignore_index=True)
    baseline_report = _metric_report(
        test_predictions["true_idx"].to_numpy(dtype=int),
        test_predictions["argmax_pred_idx"].to_numpy(dtype=int),
        class_names,
    )
    simplex_report = _metric_report(
        test_predictions["true_idx"].to_numpy(dtype=int),
        test_predictions["simplex_pred_idx"].to_numpy(dtype=int),
        class_names,
    )
    _write_confusion_matrix(output_dir / "aggregate_confusion_matrix_baseline.csv", baseline_report)
    _write_confusion_matrix(output_dir / "aggregate_confusion_matrix_simplex.csv", simplex_report)

    if save_predictions:
        val_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold_count": int(len(fold_df)),
        "class_names": class_names,
        "fold_metrics_csv": str(output_dir / "fold_simplex_metrics.csv"),
        "summary_csv": str(output_dir / "simplex_summary.csv"),
        "aggregate_baseline_test_report": baseline_report,
        "aggregate_simplex_test_report": simplex_report,
        "fold_rows": fold_rows,
    }
    (output_dir / "simplex_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure(args)
    run_root = Path(args.run_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_root / "paper_metrics" / "simplex_tuning" / safe_job_component(args.model_name) / args.data_type
    )
    folds = parse_fold_spec(args.folds, args.n_folds)
    device = _choose_device(args.device)
    dataset = _load_dataset()

    fold_rows = []
    all_val_predictions = []
    all_test_predictions = []
    class_names = list(config.class_names)
    for fold in folds:
        row, val_predictions, test_predictions = _run_fold(
            run_root,
            dataset,
            model_name=args.model_name,
            data_type=args.data_type,
            fold=fold,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            resolution=args.resolution,
            objective=args.objective,
        )
        fold_rows.append(row)
        all_val_predictions.append(val_predictions)
        all_test_predictions.append(test_predictions)
        class_names = _load_evaluation(run_root, args.model_name, args.data_type, fold)["class_names"]
        tau_values = [round(float(row[f"tau_{name.lower().replace('-', '_')}"]), 3) for name in class_names]
        print(
            f"Fold {fold}: baseline test macro-F1={row['baseline_test_macro_f1']:.4f}, "
            f"simplex test macro-F1={row['simplex_test_macro_f1']:.4f}, "
            f"tau={tau_values}"
        )

    _write_outputs(
        output_dir,
        fold_rows,
        all_val_predictions,
        all_test_predictions,
        class_names=class_names,
        save_predictions=args.save_predictions,
    )
    print(f"Output directory: {output_dir}")
    print(f"Fold metrics: {output_dir / 'fold_simplex_metrics.csv'}")
    print(f"Summary: {output_dir / 'simplex_summary.csv'}")


if __name__ == "__main__":
    main()
