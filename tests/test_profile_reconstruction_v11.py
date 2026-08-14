from __future__ import annotations

import numpy as np
import torch

from lh_transitions.profile_recon_robust_sobolev_full_profile_v11 import (
    EXPECTED_GRID_POINTS,
    FullProfileDataset,
    ModelConfig,
    ProfileResMLPModel,
    species_fit_scalars,
)


def _write_dataset(path, samples: int = 4, features: int = 7) -> None:
    te = np.tile(np.arange(EXPECTED_GRID_POINTS, dtype=np.float32), (samples, 1))
    ne = te + 1000.0
    np.savez(
        path,
        X_train=np.zeros((samples, features), dtype=np.float32),
        Y_train=np.zeros((samples, 200), dtype=np.float32),
        Y_fit=np.concatenate([te, ne], axis=1),
        T_train=np.arange(samples, dtype=np.float32),
    )


def test_full_profile_dataset_uses_y_fit_instead_of_mtanh(tmp_path) -> None:
    path = tmp_path / "samples_train.npz"
    _write_dataset(path)

    te = FullProfileDataset(path, "Te")
    ne = FullProfileDataset(path, "Ne")

    assert te.Y.shape == (4, EXPECTED_GRID_POINTS)
    assert ne.Y.shape == (4, EXPECTED_GRID_POINTS)
    assert float(te.Y[0, -1]) == 200.0
    assert float(ne.Y[0, 0]) == 1000.0


def test_v11_model_outputs_all_201_profile_points() -> None:
    config = ModelConfig(input_dim=7, hidden_dim=16, num_blocks=1)
    model = ProfileResMLPModel(**vars(config)).eval()
    assert model(torch.randn(3, 7)).shape == (3, EXPECTED_GRID_POINTS)


def test_species_fit_scalars_split_the_402_point_contract(tmp_path) -> None:
    mean = np.arange(2 * EXPECTED_GRID_POINTS, dtype=np.float32)
    std = mean + 1.0
    np.savez(tmp_path / "standardization_scalars.npz", y_mean_fit=mean, y_std_fit=std)

    te_mean, te_std = species_fit_scalars(tmp_path, "Te")
    ne_mean, ne_std = species_fit_scalars(tmp_path, "Ne")

    assert len(te_mean) == len(ne_mean) == EXPECTED_GRID_POINTS
    assert te_mean[0] == 0.0 and te_std[-1] == EXPECTED_GRID_POINTS
    assert ne_mean[0] == EXPECTED_GRID_POINTS
    assert ne_std[-1] == 2 * EXPECTED_GRID_POINTS
