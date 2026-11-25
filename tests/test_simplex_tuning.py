import numpy as np
import pytest

from arccnet_labelnoise import simplex_tuning


def test_generate_simplex_grid_has_expected_points():
    grid = simplex_tuning.generate_simplex_grid(n_classes=3, resolution=2)

    assert len(grid) == 6
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0.0)
    assert any(np.allclose(point, [0.5, 0.5, 0.0]) for point in grid)


def test_simplex_predict_matches_argmax_for_uniform_tau():
    probs = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.4, 0.5],
        ]
    )
    tau = np.array([1 / 3, 1 / 3, 1 / 3])

    assert simplex_tuning.simplex_predict(probs, tau).tolist() == [0, 1, 2]


def test_tune_simplex_threshold_can_improve_macro_f1():
    class_names = ["Alpha", "Beta", "Beta-Gamma"]
    labels = np.array([0, 1, 2, 2])
    probs = np.array(
        [
            [0.55, 0.30, 0.15],
            [0.25, 0.55, 0.20],
            [0.45, 0.15, 0.40],
            [0.40, 0.20, 0.39],
        ]
    )
    uniform_tau = np.array([1 / 3, 1 / 3, 1 / 3])
    baseline = simplex_tuning.evaluate_tau(probs, labels, uniform_tau, class_names)

    tuned = simplex_tuning.tune_simplex_threshold(
        probs,
        labels,
        class_names=class_names,
        resolution=4,
        objective="macro_f1",
    )

    assert tuned["objective_value"] > baseline["macro_f1"]
    assert tuned["tau"].shape == (3,)
    assert tuned["candidate_count"] == 15


def test_parse_fold_spec_validates_bounds():
    assert simplex_tuning.parse_fold_spec("1-3,5", n_folds=8) == [1, 2, 3, 5]

    with pytest.raises(ValueError, match="expected 1-8"):
        simplex_tuning.parse_fold_spec("1,9", n_folds=8)
