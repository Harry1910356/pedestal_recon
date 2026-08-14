from __future__ import annotations

import numpy as np

from lh_transitions.visualize_full_profile_v11 import (
    example_indices,
    radial_metrics,
    sample_metrics,
)


def test_sample_metrics_are_exact_for_exact_full_profiles() -> None:
    target = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
    metrics = sample_metrics(target, target.copy())
    np.testing.assert_allclose(metrics["rmse"], 0.0)
    np.testing.assert_allclose(metrics["mae"], 0.0)
    np.testing.assert_allclose(metrics["nrmse"], 0.0)
    np.testing.assert_allclose(metrics["r2"], 1.0)


def test_radial_metrics_preserve_the_profile_grid() -> None:
    target = np.zeros((2, 3))
    prediction = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, 3.0]])
    metrics = radial_metrics(target, prediction)
    np.testing.assert_allclose(metrics["rmse"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(metrics["mae"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(metrics["bias"], [0.0, 0.0, 3.0])


def test_example_indices_include_best_and_worst_combined_nrmse() -> None:
    metrics = {
        "Te": {"nrmse": np.array([0.4, 0.1, 0.3, 0.2])},
        "Ne": {"nrmse": np.array([0.4, 0.1, 0.3, 0.2])},
    }
    examples = example_indices(metrics)
    assert examples[0] == ("best", 1)
    assert examples[-1] == ("worst", 0)
