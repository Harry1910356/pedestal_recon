from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lh_transitions.profile_recon_pedestal_aware_v12 import (
    EXPECTED_GRID_POINTS,
    RECONSTRUCTION_SCOPE,
    PedestalAwarePhysicalSlopeLoss,
    calibrate_pedestal_threshold,
    normalized_steepest_slope_score,
)


def test_v12_is_explicitly_a_full_profile_reconstructor() -> None:
    assert RECONSTRUCTION_SCOPE == "full_profile"
    assert EXPECTED_GRID_POINTS == 201


def test_balanced_checkpoints_declare_full_profile_scope() -> None:
    checkpoint_dir = Path(__file__).parents[1] / "v12_balanced_outputs"
    if not checkpoint_dir.exists():
        return
    for species in ("te", "ne"):
        checkpoint = torch.load(
            checkpoint_dir / f"best_resmlp_v12_{species}_pedestal_aware.pth",
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["reconstruction_scope"] == "full_profile"


def test_normalized_slope_score_separates_sharp_and_smooth_profiles() -> None:
    rho = np.linspace(0.0, 1.25, 201)
    smooth = 2.0 - rho
    sharp = 2.0 - rho - 0.8 / (
        1.0 + np.exp(np.clip(-(rho - 0.98) / 0.008, -60.0, 60.0))
    )
    score = normalized_steepest_slope_score(np.stack([smooth, sharp]), rho)
    assert score[1] > 5.0 * score[0]


def test_pedestal_loss_is_zero_for_exact_profiles() -> None:
    rho = np.linspace(0.0, 1.25, 201, dtype=np.float32)
    mean = np.zeros(201, dtype=np.float32)
    std = np.ones(201, dtype=np.float32)
    criterion = PedestalAwarePhysicalSlopeLoss(mean, std, rho, pedestal_threshold=2.0)
    target = torch.randn(3, 201)
    components = criterion.components(target.clone(), target)
    assert torch.allclose(components["total"], torch.tensor(0.0), atol=1e-6)


def test_sharp_region_error_receives_more_weight_than_flat_region_error() -> None:
    rho = np.linspace(0.0, 1.25, 201, dtype=np.float32)
    target_np = 2.0 - rho - 0.8 / (
        1.0 + np.exp(np.clip(-(rho - 0.98) / 0.008, -60.0, 60.0))
    )
    target = torch.tensor(target_np[None, :], dtype=torch.float32)
    criterion = PedestalAwarePhysicalSlopeLoss(
        np.zeros(201, dtype=np.float32), np.ones(201, dtype=np.float32), rho,
        pedestal_threshold=2.0, weight_gradient=0.0,
        weight_peak_magnitude=0.0, weight_peak_location=0.0,
    )
    flat_error = target.clone(); flat_error[:, 30] += 0.2
    steep_error = target.clone(); steep_error[:, np.argmin(abs(rho - 0.98))] += 0.2
    assert criterion(steep_error, target) > criterion(flat_error, target)


def test_threshold_calibration_returns_requested_population_boundary() -> None:
    rho = np.linspace(0.0, 1.25, 201, dtype=np.float32)
    step = 1.0 / (1.0 + np.exp(np.clip(-(rho - 0.98) / 0.01, -60.0, 60.0)))
    profiles = np.stack([2.0 - rho - a * step for a in np.linspace(0.0, 1.0, 20)])
    threshold = calibrate_pedestal_threshold(
        profiles, np.zeros(201), np.ones(201), rho, 0.8, 0.7, 1.1
    )
    scores = normalized_steepest_slope_score(profiles, rho)
    np.testing.assert_allclose(threshold, np.quantile(scores, 0.8))
