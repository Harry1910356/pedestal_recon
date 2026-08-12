from __future__ import annotations

import numpy as np

from lh_transitions.visualize_resmlp_v10 import sample_metrics, tier


def test_sample_metrics_are_exact_for_exact_prediction() -> None:
    target = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 7.0]])
    metrics = sample_metrics(target, target.copy())
    np.testing.assert_allclose(metrics["rmse"], 0.0)
    np.testing.assert_allclose(metrics["mae"], 0.0)
    np.testing.assert_allclose(metrics["nrmse"], 0.0)
    np.testing.assert_allclose(metrics["r2"], 1.0)


def test_tier_supports_higher_and_lower_is_better_metrics() -> None:
    assert tier(np.array([0.96, 0.90, 0.82, 0.50]), (0.95, 0.85, 0.80), True).tolist() == [
        "Excellent",
        "Good",
        "Acceptable",
        "Poor",
    ]
    assert tier(np.array([50.0, 100.0, 200.0, 300.0]), (75.0, 150.0, 250.0), False).tolist() == [
        "Excellent",
        "Good",
        "Acceptable",
        "Poor",
    ]
