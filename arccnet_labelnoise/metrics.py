"""Metric helpers for cutout classification pipelines."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_metric_key(value: object) -> str:
    """Return a stable lowercase key for metric names."""
    key = str(value).strip().lower()
    key = re.sub(r"[^0-9a-zA-Z]+", "_", key)
    key = re.sub(r"_+", "_", key)
    return key.strip("_") or "unknown"


def _resolve_labels_and_names(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] | None,
    class_names: Sequence[str] | None,
) -> tuple[list[int], list[str]]:
    if labels is None:
        if class_names:
            labels = list(range(len(class_names)))
        else:
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    resolved_labels = [int(label) for label in labels]
    if class_names is None or len(class_names) != len(resolved_labels):
        resolved_names = [f"Class_{label}" for label in resolved_labels]
    else:
        resolved_names = [str(name) for name in class_names]
    return resolved_labels, resolved_names


def compute_classification_metric_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Compute paper-facing classification metrics from true/predicted labels."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    resolved_labels, resolved_names = _resolve_labels_and_names(y_true_array, y_pred_array, labels, class_names)

    if y_true_array.size == 0:
        empty_cm = confusion_matrix([], [], labels=resolved_labels).tolist() if resolved_labels else []
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "weighted_precision": float("nan"),
            "weighted_recall": float("nan"),
            "weighted_f1": float("nan"),
            "labels": resolved_labels,
            "class_names": resolved_names,
            "support": {name: 0 for name in resolved_names},
            "per_class": {},
            "classification_report": {},
            "confusion_matrix": empty_cm,
        }

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=resolved_labels,
        target_names=resolved_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_array, y_pred_array, labels=resolved_labels)

    per_class = {}
    support = {}
    for class_name in resolved_names:
        metrics = report.get(class_name, {})
        if isinstance(metrics, dict):
            support[class_name] = int(metrics.get("support", 0))
            per_class[class_name] = {
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1-score", 0.0)),
                "support": support[class_name],
            }

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "macro_precision": float(
            precision_score(y_true_array, y_pred_array, labels=resolved_labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_array, y_pred_array, labels=resolved_labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, labels=resolved_labels, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true_array, y_pred_array, labels=resolved_labels, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true_array, y_pred_array, labels=resolved_labels, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, labels=resolved_labels, average="weighted", zero_division=0)
        ),
        "labels": resolved_labels,
        "class_names": resolved_names,
        "support": support,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def compute_binary_classification_metric_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    *,
    class_names: Sequence[str] = ("Negative", "Positive"),
) -> dict:
    """Compute binary classification metrics including rare-event probability scores.

    Average precision is the primary probability metric for an imbalanced binary
    task. ROC AUC is retained as a secondary discrimination measure. Both are
    reported as NaN when only one target class is present.
    """
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    if not (len(y_true_array) == len(y_pred_array) == len(y_score_array)):
        raise ValueError("Binary targets, predictions, and scores must have equal length.")

    report = compute_classification_metric_report(
        y_true_array,
        y_pred_array,
        labels=[0, 1],
        class_names=class_names,
    )
    report["positive_prevalence"] = float(y_true_array.mean()) if y_true_array.size else float("nan")
    if y_true_array.size and np.unique(y_true_array).size == 2:
        report["average_precision"] = float(average_precision_score(y_true_array, y_score_array))
        report["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array))
    else:
        report["average_precision"] = float("nan")
        report["roc_auc"] = float("nan")
    return report


def metric_report_to_scalars(metric_report: dict, *, prefix: str = "") -> dict[str, float]:
    """Flatten the scalar parts of a metric report for loggers."""
    key_prefix = f"{prefix}_" if prefix else ""
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    )
    scalars = {
        f"{key_prefix}{key}": float(metric_report[key])
        for key in scalar_keys
        if key in metric_report and isinstance(metric_report[key], int | float)
    }

    for class_name, metrics in metric_report.get("per_class", {}).items():
        class_key = normalize_metric_key(class_name)
        for metric_name in ("precision", "recall", "f1", "support"):
            value = metrics.get(metric_name)
            if isinstance(value, int | float):
                scalars[f"{key_prefix}{class_key}_{metric_name}"] = float(value)

    return scalars
