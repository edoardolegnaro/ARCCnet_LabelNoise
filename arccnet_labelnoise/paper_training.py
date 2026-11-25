"""Prepare, train and evaluate a Hale benchmark fold."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from numbers import Number
from pathlib import Path

import pandas as pd

from arccnet.models.cutouts.hale.data_preparation import validate_fold_separation
from arccnet_labelnoise import config
from arccnet_labelnoise.trainer import HaleTrainer
from arccnet_labelnoise.metrics import (
    compute_binary_classification_metric_report,
    compute_classification_metric_report,
    normalize_metric_key,
)
from arccnet_labelnoise.labels import CLASS_NAMES_DICT, LABEL_MAPPING_DICT

TEST_RESULT_METRIC_KEYS = (
    "test_loss",
    "test_acc",
    "test_accuracy",
    "test_balanced_acc",
    "test_balanced_accuracy",
    "test_f1",
    "test_macro_f1",
    "test_weighted_f1",
    "test_macro_precision",
    "test_macro_recall",
)
CLASSIFICATION_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)
DELTA_METRIC_KEYS = tuple(f"delta_{key}" for key in CLASSIFICATION_METRIC_KEYS)
DELTA_PROBABILITY_METRIC_KEYS = ("average_precision", "roc_auc", "positive_prevalence")


def dataset_cache_tag(dataset_folder: str) -> str:
    """Return the filename-safe cache tag used by Hale processed datasets."""
    return str(dataset_folder).strip(os.sep).replace(os.sep, "_").replace("/", "_")


def processed_dataset_filename(
    classes: str,
    dataset_folder: str,
    n_folds: int,
    random_state: int,
    spatial_policy_tag: str = config.SPATIAL_POLICY_TAG,
) -> str:
    """Return the standard processed-dataset filename for one label policy."""
    return (
        f"processed_dataset_{dataset_cache_tag(dataset_folder)}_{classes}_"
        f"{spatial_policy_tag}_{int(n_folds)}-splits_rs-{int(random_state)}.parquet"
    )


def matched_processed_dataset_filename(
    classes: str,
    split_reference_classes: str,
    dataset_folder: str,
    n_folds: int,
    random_state: int,
    spatial_policy_tag: str = config.SPATIAL_POLICY_TAG,
) -> str:
    """Return a distinct filename for labels using another policy's folds."""
    return (
        f"processed_dataset_{dataset_cache_tag(dataset_folder)}_{classes}_"
        f"{spatial_policy_tag}_matched-splits-from-{split_reference_classes}_"
        f"{int(n_folds)}-splits_rs-{int(random_state)}.parquet"
    )


def apply_hale_config(args: argparse.Namespace, *, data_type: str | None = None, run_dir: Path | None = None) -> None:
    """Apply CLI settings to the Hale config module for this process."""
    classes = getattr(args, "classes", config.classes)
    if classes not in LABEL_MAPPING_DICT:
        raise ValueError(f"Unsupported Hale classes: {classes}")

    config.classes = classes
    config.class_names = CLASS_NAMES_DICT.get(classes, ["Unknown"])
    config.label_mapping = LABEL_MAPPING_DICT[classes]
    config.NUM_CLASSES = len(config.class_names)

    config.MODEL_NAME = getattr(args, "model_name", config.MODEL_NAME)
    config.DATA_TYPE = data_type or getattr(args, "data_type", config.DATA_TYPE)
    config.MAX_EPOCHS = int(getattr(args, "max_epochs", config.MAX_EPOCHS))
    config.BATCH_SIZE = int(getattr(args, "batch_size", config.BATCH_SIZE))
    config.NUM_WORKERS = int(getattr(args, "num_workers", config.NUM_WORKERS))
    config.N_FOLDS = int(getattr(args, "n_folds", config.N_FOLDS))
    config.RANDOM_STATE = int(getattr(args, "random_state", config.RANDOM_STATE))
    config.LEARNING_RATE = float(getattr(args, "learning_rate", config.LEARNING_RATE))
    config.EARLY_STOPPING_PATIENCE = int(getattr(args, "patience", config.EARLY_STOPPING_PATIENCE))
    config.EARLY_STOPPING_MONITOR = getattr(args, "monitor", config.EARLY_STOPPING_MONITOR)
    config.CHECKPOINT_MONITOR = getattr(args, "monitor", config.CHECKPOINT_MONITOR)
    config.LR_SCHEDULER_MONITOR = getattr(args, "monitor", config.LR_SCHEDULER_MONITOR)
    config.EARLY_STOPPING_MODE = getattr(args, "monitor_mode", config.EARLY_STOPPING_MODE)
    config.CHECKPOINT_MODE = getattr(args, "monitor_mode", config.CHECKPOINT_MODE)
    config.LR_SCHEDULER_MODE = getattr(args, "monitor_mode", config.LR_SCHEDULER_MODE)
    config.OPTIMIZER = getattr(args, "optimizer", config.OPTIMIZER)
    config.LOSS_POLICY = getattr(args, "loss_policy", config.LOSS_POLICY)
    config.FOCAL_LOSS_GAMMA = float(getattr(args, "focal_loss_gamma", config.FOCAL_LOSS_GAMMA))
    config.DELTA_AUX_WEIGHT = float(getattr(args, "delta_aux_weight", config.DELTA_AUX_WEIGHT))
    config.DELTA_AUX_MASK_ALPHA = bool(getattr(args, "delta_aux_mask_alpha", config.DELTA_AUX_MASK_ALPHA))
    config.DELTA_POS_WEIGHT_POWER = float(getattr(args, "delta_pos_weight_power", config.DELTA_POS_WEIGHT_POWER))
    config.GRADIENT_CLIP_VAL = float(getattr(args, "gradient_clip_val", config.GRADIENT_CLIP_VAL))
    config.AUTO_LR_FIND = bool(getattr(args, "auto_lr_find", config.AUTO_LR_FIND))
    config.LR_FIND_MIN_LR = float(getattr(args, "lr_find_min_lr", config.LR_FIND_MIN_LR))
    config.LR_FIND_MAX_LR = float(getattr(args, "lr_find_max_lr", config.LR_FIND_MAX_LR))
    config.LR_FIND_NUM_TRAINING = int(getattr(args, "lr_find_num_training", config.LR_FIND_NUM_TRAINING))
    config.TRANSFORMER_RGB_ADAPTER = bool(getattr(args, "transformer_rgb_adapter", config.TRANSFORMER_RGB_ADAPTER))
    config.TRANSFORMER_IMAGENET_NORMALIZE = bool(
        getattr(args, "transformer_imagenet_normalize", config.TRANSFORMER_IMAGENET_NORMALIZE)
    )
    config.ACCELERATOR = getattr(args, "accelerator", config.ACCELERATOR)
    config.DEVICES = int(getattr(args, "devices", config.DEVICES))
    config.DATA_FOLDER = getattr(args, "data_folder", config.DATA_FOLDER)
    config.DATASET_FOLDER = getattr(args, "dataset_folder", config.DATASET_FOLDER)
    config.DF_FILE_NAME = getattr(args, "df_file_name", config.DF_FILE_NAME)
    config.IMAGE_TARGET_HEIGHT = int(getattr(args, "input_height", config.IMAGE_TARGET_HEIGHT))
    config.IMAGE_TARGET_WIDTH = int(getattr(args, "input_width", config.IMAGE_TARGET_WIDTH))
    config.IMAGE_TARGET_SIZE = (config.IMAGE_TARGET_HEIGHT, config.IMAGE_TARGET_WIDTH)
    config.RESIZE_MODE = getattr(args, "resize_mode", config.RESIZE_MODE)
    config.INSTRUMENT_FILTER = getattr(args, "instrument", config.INSTRUMENT_FILTER)

    config.ENABLE_COMET = False
    config.ENABLE_CSV = True

    config.DATASET_CACHE_TAG = dataset_cache_tag(config.DATASET_FOLDER)
    processed_filename_override = getattr(args, "processed_dataset_filename", None)
    if processed_filename_override:
        processed_filename_path = Path(processed_filename_override)
        if processed_filename_path.name != str(processed_filename_override):
            raise ValueError("--processed-dataset-filename must be a filename, not a path.")
        config.PROCESSED_DATASET_FILENAME = processed_filename_path.name
    else:
        config.PROCESSED_DATASET_FILENAME = processed_dataset_filename(
            config.classes,
            config.DATASET_FOLDER,
            config.N_FOLDS,
            config.RANDOM_STATE,
        )
    config.PROJECT_NAME = f"arcaff-v20251016-{config.classes}"

    if run_dir is not None:
        config.LOG_DIR = str(run_dir / "logs")


def dataset_cache_path() -> Path:
    """Return the configured Hale processed-dataset cache path."""
    return Path(config.DATA_FOLDER) / config.PROCESSED_DATASET_FILENAME


def json_scalar(value):
    """Convert common metric scalar types into JSON/CSV friendly values."""
    if isinstance(value, Number):
        return float(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return float(value.detach().cpu())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_prepared_dataset() -> pd.DataFrame:
    """Load the cached processed dataset through the standard Hale trainer."""
    return HaleTrainer(class_names=config.class_names).prepare_dataset_once()


def prepare_dataset_command(args: argparse.Namespace) -> None:
    """Prepare or reuse the grouped Hale dataset cache."""
    apply_hale_config(args)
    processed_path = dataset_cache_path()
    explicit_filename = bool(getattr(args, "processed_dataset_filename", None))

    if processed_path.exists() and not args.force:
        print(f"Processed dataset already exists: {processed_path}")
        return
    if processed_path.exists() and args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing processed dataset: {processed_path}. "
            "Move it aside manually if you really want to regenerate it."
        )
    if explicit_filename:
        raise FileNotFoundError(
            f"Explicit processed dataset does not exist: {processed_path}. "
            "Create and verify it before launching training."
        )

    df = load_prepared_dataset()
    print(f"Prepared dataset rows: {len(df):,}")
    print(f"Processed dataset path: {processed_path}")


def prepare_matched_dataset_command(args: argparse.Namespace) -> Path:
    """Create a non-overwriting label cache using another policy's fold assignments."""
    if args.classes not in LABEL_MAPPING_DICT:
        raise ValueError(f"Unsupported Hale classes: {args.classes}")
    reference_classes = args.split_reference_classes
    if reference_classes not in LABEL_MAPPING_DICT:
        raise ValueError(f"Unsupported split-reference Hale classes: {reference_classes}")
    if args.classes == reference_classes:
        raise ValueError("Target and split-reference classes must differ.")

    data_folder = Path(args.data_folder)
    target_path = data_folder / processed_dataset_filename(
        args.classes,
        args.dataset_folder,
        args.n_folds,
        args.random_state,
    )
    reference_path = data_folder / processed_dataset_filename(
        reference_classes,
        args.dataset_folder,
        args.n_folds,
        args.random_state,
    )
    output_filename = getattr(args, "processed_dataset_filename", None) or matched_processed_dataset_filename(
        args.classes,
        reference_classes,
        args.dataset_folder,
        args.n_folds,
        args.random_state,
    )
    if Path(output_filename).name != str(output_filename):
        raise ValueError("--processed-dataset-filename must be a filename, not a path.")
    output_path = data_folder / output_filename

    missing_paths = [path for path in (target_path, reference_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Matched split preparation requires existing standard caches: {missing_text}")

    target = pd.read_parquet(target_path)
    reference = pd.read_parquet(reference_path)
    identity_columns = ["number", "dates", "time", "label"]
    missing_identity = [column for column in identity_columns if column not in target or column not in reference]
    if missing_identity:
        raise ValueError(f"Cannot verify cache observation identity; missing columns: {missing_identity}")
    if len(target) != len(reference) or any(
        not target[column].equals(reference[column]) for column in identity_columns
    ):
        raise ValueError("Target and split-reference caches do not contain identical observations in the same order.")

    fold_columns = [f"Fold {fold}" for fold in range(1, int(args.n_folds) + 1)]
    missing_folds = [column for column in fold_columns if column not in reference]
    if missing_folds:
        raise ValueError(f"Split-reference cache is missing fold columns: {missing_folds}")

    matched = target.copy()
    matched[fold_columns] = reference[fold_columns]
    validate_fold_separation(matched)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if not existing.equals(matched):
            raise FileExistsError(f"Refusing to overwrite a different matched dataset: {output_path}")
        print(f"Matched processed dataset already exists and was verified: {output_path}")
        return output_path

    matched.to_parquet(output_path, index=False)
    print(f"Matched processed dataset rows: {len(matched):,}")
    print(f"Target labels: {args.classes}")
    print(f"Fold assignments copied from: {reference_classes}")
    print(f"Matched processed dataset path: {output_path}")
    return output_path


def train_fold_command(args: argparse.Namespace) -> None:
    """Train one Hale fold for one modality."""
    run_dir = Path(args.run_root) / safe_job_component(args.model_name) / args.data_type / f"fold_{int(args.fold):02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_hale_config(args, data_type=args.data_type, run_dir=run_dir)

    job_config = {
        "created_utc": datetime.now(UTC).isoformat(),
        "fold": int(args.fold),
        "classes": config.classes,
        "class_names": config.class_names,
        "data_type": config.DATA_TYPE,
        "model_name": config.MODEL_NAME,
        "max_epochs": config.MAX_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "learning_rate": config.LEARNING_RATE,
        "optimizer": config.OPTIMIZER,
        "loss_policy": config.LOSS_POLICY,
        "focal_loss_gamma": config.FOCAL_LOSS_GAMMA,
        "delta_aux_weight": config.DELTA_AUX_WEIGHT,
        "delta_aux_mask_alpha": config.DELTA_AUX_MASK_ALPHA,
        "delta_pos_weight_power": config.DELTA_POS_WEIGHT_POWER,
        "monitor": config.CHECKPOINT_MONITOR,
        "monitor_mode": config.CHECKPOINT_MODE,
        "gradient_clip_val": config.GRADIENT_CLIP_VAL,
        "auto_lr_find": config.AUTO_LR_FIND,
        "transformer_rgb_adapter": config.TRANSFORMER_RGB_ADAPTER,
        "transformer_imagenet_normalize": config.TRANSFORMER_IMAGENET_NORMALIZE,
        "accelerator": config.ACCELERATOR,
        "devices": config.DEVICES,
        "data_folder": config.DATA_FOLDER,
        "dataset_folder": config.DATASET_FOLDER,
        "df_file_name": config.DF_FILE_NAME,
        "processed_dataset_filename": config.PROCESSED_DATASET_FILENAME,
        "spatial_policy_tag": config.SPATIAL_POLICY_TAG,
        "require_complete_fov": config.REQUIRE_COMPLETE_FOV,
        "expected_hmi_cutout_shape": config.EXPECTED_HMI_CUTOUT_SHAPE,
        "expected_mdi_cutout_shape": config.EXPECTED_MDI_CUTOUT_SHAPE,
        "model_input_size": config.IMAGE_TARGET_SIZE,
        "resize_mode": config.RESIZE_MODE,
        "instrument_filter": config.INSTRUMENT_FILTER,
        "random_state": config.RANDOM_STATE,
        "log_dir": config.LOG_DIR,
        "csv_enabled": config.ENABLE_CSV,
        "comet_enabled": config.ENABLE_COMET,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(run_dir / "job_config.json", job_config)
    write_json(run_dir / "job_status.json", {**job_config, "status": "running"})

    try:
        df = load_prepared_dataset()
        trainer = HaleTrainer(class_names=config.class_names)
        trainer_obj, _model, test_results = trainer.train_single_fold(df, fold_num=int(args.fold))
        best_checkpoint = getattr(getattr(trainer_obj, "checkpoint_callback", None), "best_model_path", None)
        evaluation_json = Path(config.LOG_DIR) / f"fold_{int(args.fold)}_evaluation" / "evaluation.json"

        result_payload = {
            **job_config,
            "status": "success",
            "finished_utc": datetime.now(UTC).isoformat(),
            "test_results": test_results,
            "best_checkpoint": best_checkpoint,
            "evaluation_json": str(evaluation_json),
        }
        write_json(run_dir / "job_result.json", result_payload)
        write_json(run_dir / "job_status.json", result_payload)
    except Exception as exc:
        write_json(
            run_dir / "job_status.json",
            {
                **job_config,
                "status": "failed",
                "finished_utc": datetime.now(UTC).isoformat(),
                "error": repr(exc),
            },
        )
        raise


def safe_job_component(value: str) -> str:
    """Return a filesystem-safe job path component."""
    return str(value).replace("/", "_").replace(":", "_")


def _find_job_config(evaluation_path: Path) -> dict:
    for parent in evaluation_path.parents:
        config_path = parent / "job_config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _flatten_metric_rows(run_root: Path) -> list[dict]:
    rows = []
    for evaluation_path in sorted(run_root.glob("**/fold_*_evaluation/evaluation.json")):
        with evaluation_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        job_config = _find_job_config(evaluation_path)
        metric_report = payload.get("classification_metric_report") or {}
        delta_metric_report = payload.get("delta_metric_report") or {}
        delta_routed_metric_report = payload.get("delta_routed_metric_report") or {}
        test_metrics = payload.get("test_metrics") or {}
        row = {
            "classes": payload.get("classes") or job_config.get("classes"),
            "data_type": payload.get("data_type") or job_config.get("data_type"),
            "model_name": payload.get("model_name") or job_config.get("model_name"),
            "fold": payload.get("fold") or job_config.get("fold"),
            "evaluation_path": str(evaluation_path),
            "best_checkpoint": payload.get("best_checkpoint"),
        }
        for key in TEST_RESULT_METRIC_KEYS:
            if key in test_metrics:
                row[key] = json_scalar(test_metrics[key])
        for key in CLASSIFICATION_METRIC_KEYS:
            if key in metric_report:
                row[f"test_{key}"] = json_scalar(metric_report[key])
            if key in delta_metric_report:
                row[f"test_delta_{key}"] = json_scalar(delta_metric_report[key])
            if key in delta_routed_metric_report:
                row[f"test_delta_routed_{key}"] = json_scalar(delta_routed_metric_report[key])
        _add_binary_report_scalars(row, delta_metric_report, "test_delta")
        for base_class, base_report in (delta_metric_report.get("by_base_class") or {}).items():
            _add_binary_report_scalars(
                row,
                base_report,
                f"test_delta_{normalize_metric_key(base_class)}",
            )
        _add_binary_report_scalars(
            row,
            delta_metric_report.get("base_class_only_control") or {},
            "test_delta_base_only_control",
        )
        for class_name, class_metrics in (delta_routed_metric_report.get("per_class") or {}).items():
            for metric_name in ("precision", "recall", "f1", "support"):
                if metric_name in class_metrics:
                    row[f"test_delta_routed_{normalize_metric_key(class_name)}_{metric_name}"] = json_scalar(
                        class_metrics[metric_name]
                    )
        rows.append(row)
    return rows


def _add_binary_report_scalars(row: dict, report: dict, prefix: str) -> None:
    """Flatten binary rare-event metrics into one collection row."""
    for key in (*CLASSIFICATION_METRIC_KEYS, *DELTA_PROBABILITY_METRIC_KEYS):
        if key in report:
            row[f"{prefix}_{key}"] = json_scalar(report[key])
    positive_metrics = (report.get("per_class") or {}).get("Delta") or {}
    for metric_name in ("precision", "recall", "f1", "support"):
        if metric_name in positive_metrics:
            row[f"{prefix}_positive_{metric_name}"] = json_scalar(positive_metrics[metric_name])


def _collect_delta_oof_predictions(run_root: Path, output_dir: Path) -> tuple[Path | None, Path | None]:
    """Pool fold-level delta predictions and compute out-of-fold rare-event metrics."""
    frames = []
    for prediction_path in sorted(run_root.glob("**/fold_*_evaluation/delta_predictions.csv")):
        frame = pd.read_csv(prediction_path)
        frame["prediction_path"] = str(prediction_path)
        frames.append(frame)
    if not frames:
        return None, None

    predictions = pd.concat(frames, ignore_index=True)
    predictions_path = output_dir / "delta_oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    metric_rows = []
    group_columns = ["model_name", "data_type"]
    for group_values, group in predictions.groupby(group_columns, dropna=False):
        common = dict(zip(group_columns, group_values, strict=True))
        eligibility = group["delta_eligible_ground_truth"]
        if eligibility.dtype != bool:
            eligibility = (
                eligibility.astype("string")
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
            if eligibility.isna().any():
                raise ValueError("delta_eligible_ground_truth contains non-boolean values.")
        eligible = group[eligibility.astype(bool)]
        scopes = [("eligible_overall", eligible)]
        scopes.extend(
            (
                f"within_{normalize_metric_key(base_class)}",
                eligible[eligible["base_target_name"].eq(base_class)],
            )
            for base_class in ("Beta", "Beta-Gamma")
        )
        for scope, subset in scopes:
            if subset.empty:
                continue
            report = compute_binary_classification_metric_report(
                subset["delta_target"],
                subset["delta_prediction"],
                subset["delta_probability"],
                class_names=("No-Delta", "Delta"),
            )
            row = {
                **common,
                "scope": scope,
                "n_observations": len(subset),
                "n_noaa_regions": subset["number"].nunique() if "number" in subset else None,
            }
            _add_binary_report_scalars(row, report, "delta")
            metric_rows.append(row)

        if not eligible.empty:
            base_only_predictions = eligible["base_target_name"].eq("Beta-Gamma").astype(int)
            control_report = compute_binary_classification_metric_report(
                eligible["delta_target"],
                base_only_predictions,
                base_only_predictions,
                class_names=("No-Delta", "Delta"),
            )
            control_row = {
                **common,
                "scope": "base_class_only_control",
                "n_observations": len(eligible),
                "n_noaa_regions": eligible["number"].nunique() if "number" in eligible else None,
            }
            _add_binary_report_scalars(control_row, control_report, "delta")
            metric_rows.append(control_row)

        taxonomy_names = ["Alpha", "Beta", "Beta-Delta", "Beta-Gamma", "Beta-Gamma-Delta"]
        label_to_index = {name: index for index, name in enumerate(taxonomy_names)}
        routed_report = compute_classification_metric_report(
            group["routed_true_label"].map(label_to_index),
            group["routed_predicted_label"].map(label_to_index),
            labels=list(range(len(taxonomy_names))),
            class_names=taxonomy_names,
        )
        routed_row = {
            **common,
            "scope": "predicted_base_routed_taxonomy",
            "n_observations": len(group),
            "n_noaa_regions": group["number"].nunique() if "number" in group else None,
        }
        for key in CLASSIFICATION_METRIC_KEYS:
            if key in routed_report:
                routed_row[f"routed_{key}"] = json_scalar(routed_report[key])
        for class_name in ("Beta-Delta", "Beta-Gamma-Delta"):
            metrics = routed_report["per_class"].get(class_name, {})
            for metric_name in ("precision", "recall", "f1", "support"):
                if metric_name in metrics:
                    routed_row[f"routed_{normalize_metric_key(class_name)}_{metric_name}"] = json_scalar(
                        metrics[metric_name]
                    )
        metric_rows.append(routed_row)

    metrics_path = output_dir / "delta_oof_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    return predictions_path, metrics_path


def collect_command(args: argparse.Namespace) -> None:
    """Collect fold JSON artifacts into paper-facing CSV/JSON summaries."""
    run_root = Path(args.run_root)
    output_dir = run_root / "paper_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _flatten_metric_rows(run_root)
    if not rows:
        raise FileNotFoundError(f"No fold evaluation artifacts found under {run_root}")

    fold_metrics_path = output_dir / "fold_metrics.csv"
    with fold_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "fold_metrics.json", {"rows": rows})

    df = pd.DataFrame(rows)
    numeric_cols = [
        col
        for col in df.columns
        if col.startswith("test_") and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = (
        df.groupby(["classes", "data_type", "model_name"], dropna=False)[numeric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary_path = output_dir / "summary_by_modality.csv"
    summary.to_csv(summary_path, index=False)
    delta_predictions_path, delta_oof_metrics_path = _collect_delta_oof_predictions(run_root, output_dir)

    write_json(
        output_dir / "collection_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "run_root": str(run_root),
            "fold_count": len(rows),
            "fold_metrics_csv": str(fold_metrics_path),
            "summary_csv": str(summary_path),
            "delta_oof_predictions_csv": str(delta_predictions_path) if delta_predictions_path else None,
            "delta_oof_metrics_csv": str(delta_oof_metrics_path) if delta_oof_metrics_path else None,
            "tensorboard_command": f"tensorboard --logdir {run_root}",
        },
    )
    (run_root / "tensorboard_command.txt").write_text(f"tensorboard --logdir {run_root}\n", encoding="utf-8")
    print(f"Fold metrics: {fold_metrics_path}")
    print(f"Summary: {summary_path}")
    if delta_oof_metrics_path:
        print(f"Delta OOF metrics: {delta_oof_metrics_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common Hale training arguments."""
    parser.add_argument("--classes", default=config.classes)
    parser.add_argument("--model-name", default=config.MODEL_NAME)
    parser.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--n-folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--monitor", default=config.CHECKPOINT_MONITOR)
    parser.add_argument("--monitor-mode", choices=["min", "max"], default=config.CHECKPOINT_MODE)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=config.OPTIMIZER)
    parser.add_argument(
        "--loss-policy",
        choices=["unweighted_ce", "balanced_ce", "sqrt_balanced_ce", "focal_loss"],
        default=config.LOSS_POLICY,
    )
    parser.add_argument("--focal-loss-gamma", type=float, default=config.FOCAL_LOSS_GAMMA)
    parser.add_argument("--delta-aux-weight", type=float, default=config.DELTA_AUX_WEIGHT)
    parser.add_argument(
        "--delta-aux-mask-alpha",
        action="store_true",
        default=config.DELTA_AUX_MASK_ALPHA,
        help="Exclude ground-truth Alpha samples from the auxiliary delta loss and metrics.",
    )
    parser.add_argument(
        "--delta-pos-weight-power",
        type=float,
        default=config.DELTA_POS_WEIGHT_POWER,
        help="Raise the eligible negative/positive ratio to this power for auxiliary BCE weighting.",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=config.GRADIENT_CLIP_VAL)
    parser.add_argument("--auto-lr-find", action="store_true", default=config.AUTO_LR_FIND)
    parser.add_argument("--lr-find-min-lr", type=float, default=config.LR_FIND_MIN_LR)
    parser.add_argument("--lr-find-max-lr", type=float, default=config.LR_FIND_MAX_LR)
    parser.add_argument("--lr-find-num-training", type=int, default=config.LR_FIND_NUM_TRAINING)
    parser.add_argument("--transformer-rgb-adapter", action="store_true", default=config.TRANSFORMER_RGB_ADAPTER)
    parser.add_argument(
        "--transformer-imagenet-normalize",
        action="store_true",
        default=config.TRANSFORMER_IMAGENET_NORMALIZE,
    )
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--data-folder", default=config.DATA_FOLDER)
    parser.add_argument("--dataset-folder", default=config.DATASET_FOLDER)
    parser.add_argument("--df-file-name", default=config.DF_FILE_NAME)
    parser.add_argument(
        "--processed-dataset-filename",
        default=None,
        help="Use this verified processed-cache filename instead of the standard policy-derived filename.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=config.IMAGE_TARGET_HEIGHT,
        help="Model input height in pixels (paper: 112).",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=config.IMAGE_TARGET_WIDTH,
        help="Model input width in pixels (paper: 224).",
    )
    parser.add_argument(
        "--resize-mode",
        choices=config.RESIZE_MODES,
        default=config.RESIZE_MODE,
        help="How the 2:1 cutout is mapped to the model input (paper: keep-ratio).",
    )
    parser.add_argument(
        "--instrument",
        choices=config.INSTRUMENT_FILTERS,
        default=config.INSTRUMENT_FILTER,
        help="Restrict to observations whose preferred cutout is HMI or MDI (paper: all).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the paper-training CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare/reuse the Hale processed dataset cache.")
    add_common_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Fail if the processed cache already exists; this script never overwrites prepared data.",
    )
    prepare_parser.set_defaults(func=prepare_dataset_command)

    matched_parser = subparsers.add_parser(
        "prepare-matched",
        help="Create/reuse a target-label cache with fold assignments copied from another label policy.",
    )
    add_common_arguments(matched_parser)
    matched_parser.add_argument("--split-reference-classes", required=True)
    matched_parser.set_defaults(func=prepare_matched_dataset_command)

    train_parser = subparsers.add_parser("train-fold", help="Train one Hale fold for one modality.")
    add_common_arguments(train_parser)
    train_parser.add_argument("--run-root", required=True)
    train_parser.add_argument("--data-type", choices=["magnetogram", "continuum", "both"], required=True)
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.set_defaults(func=train_fold_command)

    collect_parser = subparsers.add_parser("collect", help="Collect Hale fold artifacts into CSV tables.")
    collect_parser.add_argument("--run-root", required=True)
    collect_parser.set_defaults(func=collect_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
