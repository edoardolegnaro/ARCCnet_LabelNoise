import numpy as np
import pandas as pd
import pytest

from arccnet_labelnoise.conditional_delta_comparison import (
    align_flat_and_routed_predictions,
    collapse_confusion_matrices,
    comparison_metrics,
    confusion_matrices_by_cluster,
    paired_cluster_bootstrap,
)


def _prediction_frames():
    flat = pd.DataFrame(
        {
            "source_index": [0, 1, 2, 3, 4, 5],
            "fold": [1, 1, 2, 2, 3, 3],
            "number": [10, 10, 20, 20, 30, 30],
            "flat_true_label": [
                "Alpha",
                "Beta",
                "Beta-Delta",
                "Beta-Gamma",
                "Beta-Gamma-Delta",
                "Beta",
            ],
            "flat_predicted_label": [
                "Alpha",
                "Beta",
                "Beta-Delta",
                "Beta-Gamma",
                "Beta-Gamma-Delta",
                "Beta",
            ],
        }
    )
    routed = pd.DataFrame(
        {
            "source_index": [5, 4, 3, 2, 1, 0],
            "fold": [3, 3, 2, 2, 1, 1],
            "number": [30, 30, 20, 20, 10, 10],
            "routed_true_label": [
                "Beta",
                "Beta-Gamma-Delta",
                "Beta-Gamma",
                "Beta-Delta",
                "Beta",
                "Alpha",
            ],
            "routed_predicted_label": [
                "Beta",
                "Beta-Gamma",
                "Beta-Gamma",
                "Beta-Gamma-Delta",
                "Beta",
                "Alpha",
            ],
        }
    )
    return flat, routed


def test_alignment_is_one_to_one_and_checks_targets():
    flat, routed = _prediction_frames()
    aligned = align_flat_and_routed_predictions(flat, routed)

    assert aligned["source_index"].tolist() == list(range(6))
    assert aligned["fold"].tolist() == flat["fold"].tolist()
    assert aligned["number"].tolist() == flat["number"].tolist()

    routed.loc[routed["source_index"] == 2, "routed_true_label"] = "Beta"
    with pytest.raises(ValueError, match="taxonomy target"):
        align_flat_and_routed_predictions(flat, routed)


def test_alignment_rejects_duplicate_source_indices():
    flat, routed = _prediction_frames()
    routed.loc[0, "source_index"] = routed.loc[1, "source_index"]

    with pytest.raises(ValueError, match="duplicate source indices"):
        align_flat_and_routed_predictions(flat, routed)


def test_confusion_collapse_preserves_counts_and_merges_delta_subclasses():
    confusion = np.arange(25).reshape(5, 5)
    collapsed = collapse_confusion_matrices(confusion)

    assert collapsed.shape == (3, 3)
    assert collapsed.sum() == confusion.sum()
    assert collapsed[1, 1] == confusion[1:3, 1:3].sum()
    assert collapsed[2, 2] == confusion[3:5, 3:5].sum()


def test_paired_cluster_bootstrap_is_deterministic_and_uses_fixed_classes():
    flat, routed = _prediction_frames()
    aligned = align_flat_and_routed_predictions(flat, routed)
    clusters, flat_confusion = confusion_matrices_by_cluster(
        aligned,
        true_column="flat_true_label",
        prediction_column="flat_predicted_label",
    )
    routed_clusters, routed_confusion = confusion_matrices_by_cluster(
        aligned,
        true_column="routed_true_label",
        prediction_column="routed_predicted_label",
    )

    first = paired_cluster_bootstrap(flat_confusion, routed_confusion, n_replicates=100, seed=7, chunk_size=11)
    second = paired_cluster_bootstrap(flat_confusion, routed_confusion, n_replicates=100, seed=7, chunk_size=11)

    assert clusters.tolist() == routed_clusters.tolist() == [10, 20, 30]
    pd.testing.assert_frame_equal(first, second)
    macro = first.set_index("metric").loc["macro_f1"]
    assert macro["flat_estimate"] == pytest.approx(1.0)
    assert macro["difference_flat_minus_routed"] > 0
    assert set(first["metric"]) >= {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "collapsed_macro_f1",
        "f1_beta_delta",
    }


def test_comparison_metrics_accepts_batched_confusion_matrices():
    confusion = np.stack([np.eye(5, dtype=int), np.eye(5, dtype=int) * 2])
    metrics = comparison_metrics(confusion)

    assert metrics["macro_f1"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["collapsed_macro_f1"].tolist() == pytest.approx([1.0, 1.0])
