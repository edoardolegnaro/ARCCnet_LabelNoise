"""Checkpoint-only comparison of flat and conditionally routed Hale predictions.

The flat model is evaluated on its held-out fold checkpoints, aligned by source
row with an existing conditional-delta OOF export, and compared with a paired
NOAA-active-region cluster bootstrap.  No model fitting or threshold tuning is
performed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytorch_lightning as pl
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
from arccnet_labelnoise.metrics import (
    compute_classification_metric_report,
    normalize_metric_key,
)

TAXONOMY_NAMES = ("Alpha", "Beta", "Beta-Delta", "Beta-Gamma", "Beta-Gamma-Delta")
BASE_CLASS_NAMES = ("Alpha", "Beta", "Beta-Gamma")
TAXONOMY_TO_BASE = np.asarray([0, 1, 1, 2, 2], dtype=int)
DEFAULT_MODEL_NAME = "swin_tiny_patch4_window7_224.ms_in1k"
DEFAULT_DATA_TYPE = "both"
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-run-root", required=True)
    parser.add_argument("--conditional-oof-predictions", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-type", choices=["magnetogram", "continuum", "both"], default=DEFAULT_DATA_TYPE)
    parser.add_argument("--folds", default="1-8")
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--processed-dataset-filename",
        default=None,
        help="Override the cache filename recorded in the completed flat job.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <flat-run-root>/paper_metrics/flat_vs_conditional_delta",
    )
    return parser.parse_args()


def parse_fold_spec(value: str, n_folds: int) -> list[int]:
    """Parse a comma/range fold specification into unique ordered folds."""
    folds: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Fold range end must be >= start. Got: {part}")
            folds.extend(range(start, end + 1))
        else:
            folds.append(int(part))
    folds = list(dict.fromkeys(folds))
    if not folds or min(folds) < 1 or max(folds) > int(n_folds):
        raise ValueError(f"Folds must be within 1-{n_folds}. Got: {folds}")
    return folds


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def _configure_deterministic_inference(seed: int) -> None:
    """Match the deterministic CUDA flags used by the original Lightning test pass."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pl.seed_everything(seed, workers=True, verbose=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _job_directory(run_root: Path, model_name: str, data_type: str, fold: int) -> Path:
    return run_root / safe_job_component(model_name) / data_type / f"fold_{fold:02d}"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_from_completed_job(
    run_root: Path,
    model_name: str,
    data_type: str,
    fold: int,
    processed_dataset_filename: str | None,
) -> dict:
    job = _read_json(_job_directory(run_root, model_name, data_type, fold) / "job_config.json")
    if processed_dataset_filename is not None:
        job["processed_dataset_filename"] = processed_dataset_filename
    apply_hale_config(SimpleNamespace(**job), data_type=data_type)
    config.ENABLE_TENSORBOARD = False
    config.ENABLE_CSV = False
    config.ENABLE_COMET = False
    return job


def _load_dataset() -> pd.DataFrame:
    path = dataset_cache_path()
    if not path.exists():
        raise FileNotFoundError(f"Processed Hale dataset cache not found: {path}")
    dataset = pd.read_parquet(path)
    if "model_labels" not in dataset:
        label_to_index = {label: index for index, label in enumerate(config.class_names)}
        dataset = dataset.copy()
        dataset["model_labels"] = dataset["grouped_labels"].map(label_to_index)
        if dataset["model_labels"].isna().any():
            raise ValueError("The processed dataset contains labels outside the configured flat taxonomy.")
        dataset["model_labels"] = dataset["model_labels"].astype(int)
    return dataset


def _evaluation_path(run_root: Path, model_name: str, data_type: str, fold: int) -> Path:
    return (
        _job_directory(run_root, model_name, data_type, fold) / "logs" / f"fold_{fold}_evaluation" / "evaluation.json"
    )


def predict_flat_fold(
    run_root: Path,
    dataset: pd.DataFrame,
    *,
    model_name: str,
    data_type: str,
    fold: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Export one held-out flat fold and verify its stored confusion matrix."""
    evaluation = _read_json(_evaluation_path(run_root, model_name, data_type, fold))
    class_names = tuple(evaluation.get("class_names") or ())
    if class_names != TAXONOMY_NAMES:
        raise ValueError(f"Expected flat taxonomy {TAXONOMY_NAMES}; found {class_names}")
    checkpoint = Path(evaluation["best_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint does not exist: {checkpoint}")

    config.MODEL_NAME = model_name
    config.DATA_TYPE = data_type
    _, _, test_df = get_fold_data(dataset, fold)
    loader = DataLoader(
        HaleDataset(test_df, data_type=data_type, is_training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = HaleLightningModel.load_from_checkpoint(checkpoint, map_location=device)
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision="16-mixed" if device.type == "cuda" else "32-true",
        deterministic=True,
        benchmark=False,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model, dataloaders=loader)

    targets = np.asarray(model.test_targets, dtype=int)
    predictions = np.asarray(model.test_predictions, dtype=int)
    probabilities = torch.softmax(torch.cat(model.test_logits, dim=0).float(), dim=1).numpy()
    if len(test_df) != len(targets):
        raise ValueError(f"Fold {fold} metadata/output length mismatch: {len(test_df)} != {len(targets)}")

    report = compute_classification_metric_report(
        targets,
        predictions,
        labels=list(range(len(TAXONOMY_NAMES))),
        class_names=TAXONOMY_NAMES,
    )
    expected_confusion = np.asarray(evaluation["confusion_matrix"], dtype=int)
    actual_confusion = np.asarray(report["confusion_matrix"], dtype=int)
    if not np.array_equal(actual_confusion, expected_confusion):
        raise ValueError(
            f"Fold {fold} checkpoint inference did not reproduce the stored confusion matrix: "
            f"expected={expected_confusion.tolist()}, actual={actual_confusion.tolist()}"
        )

    metadata_columns = [
        column for column in ("number", "dates", "time", "label", "grouped_labels") if column in test_df
    ]
    rows = test_df[metadata_columns].reset_index(drop=True).copy()
    rows.insert(0, "source_index", test_df.index.to_numpy())
    rows.insert(1, "fold", int(fold))
    rows.insert(2, "model_name", model_name)
    rows.insert(3, "data_type", data_type)
    rows["flat_target"] = targets
    rows["flat_prediction"] = predictions
    rows["flat_true_label"] = np.asarray(TAXONOMY_NAMES, dtype=object)[targets]
    rows["flat_predicted_label"] = np.asarray(TAXONOMY_NAMES, dtype=object)[predictions]
    for class_index, class_name in enumerate(TAXONOMY_NAMES):
        rows[f"prob_{normalize_metric_key(class_name)}"] = probabilities[:, class_index]

    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _require_unique_source_indices(frame: pd.DataFrame, name: str) -> None:
    if "source_index" not in frame:
        raise ValueError(f"{name} predictions lack source_index.")
    duplicates = frame["source_index"].duplicated(keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, "source_index"].head().tolist()
        raise ValueError(f"{name} predictions contain duplicate source indices, including {values}")


def align_flat_and_routed_predictions(flat: pd.DataFrame, routed: pd.DataFrame) -> pd.DataFrame:
    """Return a strictly validated, one-to-one paired prediction table."""
    required_flat = {"source_index", "fold", "number", "flat_true_label", "flat_predicted_label"}
    required_routed = {
        "source_index",
        "fold",
        "number",
        "routed_true_label",
        "routed_predicted_label",
    }
    missing_flat = sorted(required_flat - set(flat.columns))
    missing_routed = sorted(required_routed - set(routed.columns))
    if missing_flat or missing_routed:
        raise ValueError(f"Missing alignment columns: flat={missing_flat}, routed={missing_routed}")
    _require_unique_source_indices(flat, "Flat")
    _require_unique_source_indices(routed, "Routed")
    if set(flat["source_index"]) != set(routed["source_index"]):
        raise ValueError("Flat and routed predictions do not cover the same source indices.")

    routed_columns = [
        "source_index",
        "fold",
        "number",
        "routed_true_label",
        "routed_predicted_label",
    ]
    aligned = flat.merge(
        routed[routed_columns],
        on="source_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_flat", "_routed"),
    )
    checks = {
        "fold": aligned["fold_flat"].eq(aligned["fold_routed"]),
        "NOAA number": aligned["number_flat"].eq(aligned["number_routed"]),
        "taxonomy target": aligned["flat_true_label"].eq(aligned["routed_true_label"]),
    }
    failed = [name for name, mask in checks.items() if not mask.all()]
    if failed:
        raise ValueError(f"Flat/routed alignment failed for: {', '.join(failed)}")
    aligned = aligned.drop(columns=["fold_routed", "number_routed"]).rename(
        columns={"fold_flat": "fold", "number_flat": "number"}
    )
    return aligned.sort_values("source_index").reset_index(drop=True)


def confusion_matrices_by_cluster(
    aligned: pd.DataFrame,
    *,
    true_column: str,
    prediction_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one fixed-label confusion matrix per NOAA active region."""
    label_to_index = {name: index for index, name in enumerate(TAXONOMY_NAMES)}
    true_indices = aligned[true_column].map(label_to_index)
    predicted_indices = aligned[prediction_column].map(label_to_index)
    if true_indices.isna().any() or predicted_indices.isna().any():
        raise ValueError(f"Unknown taxonomy label in {true_column} or {prediction_column}.")
    cluster_codes, cluster_values = pd.factorize(aligned["number"], sort=True)
    matrices = np.zeros((len(cluster_values), len(TAXONOMY_NAMES), len(TAXONOMY_NAMES)), dtype=np.int64)
    np.add.at(matrices, (cluster_codes, true_indices.astype(int), predicted_indices.astype(int)), 1)
    return np.asarray(cluster_values), matrices


def collapse_confusion_matrices(confusion: np.ndarray) -> np.ndarray:
    """Collapse five-class confusion matrices to Alpha/Beta/Beta-Gamma."""
    confusion = np.asarray(confusion)
    if confusion.shape[-2:] != (len(TAXONOMY_NAMES), len(TAXONOMY_NAMES)):
        raise ValueError(f"Expected trailing confusion shape (5, 5); got {confusion.shape}")
    collapsed = np.zeros((*confusion.shape[:-2], len(BASE_CLASS_NAMES), len(BASE_CLASS_NAMES)), dtype=confusion.dtype)
    for true_index, true_base in enumerate(TAXONOMY_TO_BASE):
        for predicted_index, predicted_base in enumerate(TAXONOMY_TO_BASE):
            collapsed[..., true_base, predicted_base] += confusion[..., true_index, predicted_index]
    return collapsed


def metrics_from_confusion(confusion: np.ndarray, *, include_per_class: bool = True) -> dict[str, np.ndarray]:
    """Compute fixed-label metrics for one or a batch of confusion matrices."""
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positives = np.diagonal(confusion, axis1=-2, axis2=-1)
    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    precision = np.divide(
        true_positives,
        predicted,
        out=np.zeros_like(true_positives),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros_like(true_positives),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    total = confusion.sum(axis=(-2, -1))
    metrics = {
        "accuracy": np.divide(
            true_positives.sum(axis=-1),
            total,
            out=np.zeros_like(total),
            where=total != 0,
        ),
        "balanced_accuracy": recall.mean(axis=-1),
        "macro_f1": f1.mean(axis=-1),
    }
    if include_per_class:
        for index, class_name in enumerate(TAXONOMY_NAMES):
            metrics[f"f1_{normalize_metric_key(class_name)}"] = f1[..., index]
    return metrics


def comparison_metrics(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Compute exact-taxonomy and collapsed-base metrics."""
    metrics = metrics_from_confusion(confusion)
    collapsed = metrics_from_confusion(collapse_confusion_matrices(confusion), include_per_class=False)
    metrics["collapsed_macro_f1"] = collapsed["macro_f1"]
    metrics["collapsed_accuracy"] = collapsed["accuracy"]
    return metrics


def paired_cluster_bootstrap(
    flat_cluster_confusion: np.ndarray,
    routed_cluster_confusion: np.ndarray,
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Paired percentile bootstrap over NOAA-region confusion matrices."""
    flat_cluster_confusion = np.asarray(flat_cluster_confusion, dtype=np.int64)
    routed_cluster_confusion = np.asarray(routed_cluster_confusion, dtype=np.int64)
    if flat_cluster_confusion.shape != routed_cluster_confusion.shape:
        raise ValueError("Flat and routed cluster confusion arrays must have the same shape.")
    if flat_cluster_confusion.ndim != 3 or flat_cluster_confusion.shape[1:] != (5, 5):
        raise ValueError("Cluster confusion arrays must have shape (n_clusters, 5, 5).")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")

    flat_point = comparison_metrics(flat_cluster_confusion.sum(axis=0))
    routed_point = comparison_metrics(routed_cluster_confusion.sum(axis=0))
    metric_names = list(flat_point)
    flat_samples = {name: np.empty(n_replicates, dtype=np.float64) for name in metric_names}
    routed_samples = {name: np.empty(n_replicates, dtype=np.float64) for name in metric_names}

    rng = np.random.default_rng(seed)
    n_clusters = len(flat_cluster_confusion)
    for start in range(0, n_replicates, chunk_size):
        stop = min(start + chunk_size, n_replicates)
        draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        flat_confusion = flat_cluster_confusion[draws].sum(axis=1)
        routed_confusion = routed_cluster_confusion[draws].sum(axis=1)
        flat_metrics = comparison_metrics(flat_confusion)
        routed_metrics = comparison_metrics(routed_confusion)
        for name in metric_names:
            flat_samples[name][start:stop] = flat_metrics[name]
            routed_samples[name][start:stop] = routed_metrics[name]

    rows = []
    for name in metric_names:
        flat_values = flat_samples[name]
        routed_values = routed_samples[name]
        difference = flat_values - routed_values
        rows.append(
            {
                "metric": name,
                "flat_estimate": float(flat_point[name]),
                "routed_estimate": float(routed_point[name]),
                "difference_flat_minus_routed": float(flat_point[name] - routed_point[name]),
                "flat_ci_low": float(np.quantile(flat_values, 0.025)),
                "flat_ci_high": float(np.quantile(flat_values, 0.975)),
                "routed_ci_low": float(np.quantile(routed_values, 0.025)),
                "routed_ci_high": float(np.quantile(routed_values, 0.975)),
                "difference_ci_low": float(np.quantile(difference, 0.025)),
                "difference_ci_high": float(np.quantile(difference, 0.975)),
                "bootstrap_probability_flat_greater": float(np.mean(difference > 0.0)),
                "n_replicates": int(n_replicates),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def _write_new_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_new_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    flat_run_root = Path(args.flat_run_root)
    conditional_path = Path(args.conditional_oof_predictions)
    output_dir = (
        Path(args.output_dir) if args.output_dir else flat_run_root / "paper_metrics" / "flat_vs_conditional_delta"
    )
    folds = parse_fold_spec(args.folds, args.n_folds)
    if folds != list(range(1, args.n_folds + 1)):
        raise ValueError("The final paired OOF comparison requires all configured folds.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {output_dir}")

    job = _configure_from_completed_job(
        flat_run_root,
        args.model_name,
        args.data_type,
        folds[0],
        args.processed_dataset_filename,
    )
    if tuple(config.class_names) != TAXONOMY_NAMES:
        raise ValueError(f"Configured classes are not the expected flat taxonomy: {config.class_names}")
    dataset = _load_dataset()
    _configure_deterministic_inference(int(job.get("random_state", DEFAULT_BOOTSTRAP_SEED)))
    device = _choose_device(args.device)

    flat_folds = []
    for fold in folds:
        predictions = predict_flat_fold(
            flat_run_root,
            dataset,
            model_name=args.model_name,
            data_type=args.data_type,
            fold=fold,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        flat_folds.append(predictions)
        print(f"Fold {fold}: exported {len(predictions):,} held-out predictions")
    flat_oof = pd.concat(flat_folds, ignore_index=True)
    _require_unique_source_indices(flat_oof, "Flat OOF")

    routed_oof = pd.read_csv(conditional_path)
    aligned = align_flat_and_routed_predictions(flat_oof, routed_oof)
    flat_clusters, flat_confusion = confusion_matrices_by_cluster(
        aligned,
        true_column="flat_true_label",
        prediction_column="flat_predicted_label",
    )
    routed_clusters, routed_confusion = confusion_matrices_by_cluster(
        aligned,
        true_column="routed_true_label",
        prediction_column="routed_predicted_label",
    )
    if not np.array_equal(flat_clusters, routed_clusters):
        raise ValueError("Flat and routed predictions produced different NOAA cluster ordering.")
    bootstrap = paired_cluster_bootstrap(
        flat_confusion,
        routed_confusion,
        n_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )

    flat_path = output_dir / "flat_oof_predictions.csv"
    aligned_path = output_dir / "paired_oof_predictions.csv"
    bootstrap_path = output_dir / "paired_noaa_cluster_bootstrap.csv"
    _write_new_csv(flat_oof, flat_path)
    _write_new_csv(aligned, aligned_path)
    _write_new_csv(bootstrap, bootstrap_path)
    _write_new_json(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "flat_run_root": str(flat_run_root),
            "conditional_oof_predictions": str(conditional_path),
            "processed_dataset": str(dataset_cache_path()),
            "model_name": args.model_name,
            "data_type": args.data_type,
            "folds": folds,
            "n_observations": int(len(aligned)),
            "n_noaa_regions": int(len(flat_clusters)),
            "class_names": list(TAXONOMY_NAMES),
            "class_support": aligned["flat_true_label"].value_counts().reindex(TAXONOMY_NAMES).to_dict(),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "job_config": job,
            "flat_oof_predictions_csv": str(flat_path),
            "paired_oof_predictions_csv": str(aligned_path),
            "bootstrap_summary_csv": str(bootstrap_path),
        },
        output_dir / "analysis_manifest.json",
    )
    print(bootstrap.to_string(index=False))
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
