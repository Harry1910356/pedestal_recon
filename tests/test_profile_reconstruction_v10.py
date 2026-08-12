from __future__ import annotations

import json

import torch

from lh_transitions.profile_recon_robust_sobolev_mtanh_v10 import (
    CustomBalancedSobolevLoss,
    ModelConfig,
    ProfileResMLPModel,
    all_feature_indices,
)


def test_all_feature_indices_preserve_every_saved_column(tmp_path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    names = [f"feature_{index}" for index in range(112)]
    (train / "feature_names.json").write_text(
        json.dumps({"X_features": names}), encoding="utf-8"
    )

    indices, selected_names = all_feature_indices(tmp_path)

    assert indices == list(range(112))
    assert selected_names == names


def test_v10_model_is_v7_resmlp_with_full_input_width() -> None:
    config = ModelConfig(input_dim=112, output_dim=100, hidden_dim=32, num_blocks=2)
    model = ProfileResMLPModel(**vars(config)).eval()
    prediction = model(torch.randn(5, 112))
    assert prediction.shape == (5, 100)
    assert model.in_proj[0].in_features == 112


def test_v10_uses_v7_core_weighted_sobolev_loss() -> None:
    loss_fn = CustomBalancedSobolevLoss(weight_grad=20.0, delta=1.0, grid_points=100)
    target = torch.randn(4, 100)
    assert float(loss_fn(target, target)) == 0.0
    assert float(loss_fn.point_weights[0]) == 5.0
    assert float(loss_fn.point_weights[-1]) == 1.0
    assert float(loss_fn(target + 0.1, target)) > 0.0
