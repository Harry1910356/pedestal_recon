#!/usr/bin/env python3
"""Run v12 pedestal-aware checkpoints on complete Te/Ne profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lh_transitions.profile_recon_pedestal_aware_v12 import (
    EXPECTED_GRID_POINTS,
    RECONSTRUCTION_SCOPE,
    SPECIES,
    VERSION,
    ModelConfig,
    ProfileResMLPModel,
    all_feature_indices,
    choose_device,
)


def load_model(path: Path, species: str, device: torch.device) -> tuple[ProfileResMLPModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "version": VERSION,
        "reconstruction_scope": RECONSTRUCTION_SCOPE,
        "species": species,
        "grid_points": EXPECTED_GRID_POINTS,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"{path} has {key}={checkpoint.get(key)!r}; expected {value!r}")
    config = ModelConfig(**checkpoint["model_config"])
    model = ProfileResMLPModel(**asdict_compatible(config))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


def asdict_compatible(config: ModelConfig) -> dict[str, int | float]:
    return {
        "input_dim": config.input_dim,
        "output_dim": config.output_dim,
        "hidden_dim": config.hidden_dim,
        "num_blocks": config.num_blocks,
        "dropout_p": config.dropout_p,
    }


def predict_in_batches(
    model: ProfileResMLPModel,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run full-profile inference without loading the entire split onto the device."""
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            batches.append(model(batch).cpu().numpy())
    if not batches:
        return np.empty((0, EXPECTED_GRID_POINTS), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def run_prediction(
    data_root: Path,
    checkpoint_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    _, feature_names = all_feature_indices(data_root)
    with np.load(data_root / split / "samples_train.npz") as archive:
        features = np.asarray(archive["X_train"], dtype=np.float32)
        target_normalized = np.asarray(archive["Y_fit"], dtype=np.float32)
        times = np.asarray(archive["T_train"], dtype=np.float32)
    predictions = []
    for species in SPECIES:
        model, checkpoint = load_model(
            checkpoint_dir / f"best_resmlp_v12_{species.lower()}_pedestal_aware.pth",
            species,
            device,
        )
        if checkpoint["feature_names"] != feature_names:
            raise ValueError(f"{species} checkpoint feature contract differs from dataset")
        normalized = predict_in_batches(model, features, batch_size, device)
        predictions.append(normalized * checkpoint["y_std"] + checkpoint["y_mean"])
    with np.load(data_root / "standardization_scalars.npz") as scalars:
        target = target_normalized * scalars["y_std_fit"] + scalars["y_mean_fit"]
    return {
        "prediction": np.concatenate(predictions, axis=1),
        "target": target.astype(np.float32),
        "time_ms": times,
        "rho": np.linspace(0.0, 1.25, EXPECTED_GRID_POINTS, dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("intergral_v7_weighted"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("v12_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("v12_predictions"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_prediction(args.data_root.resolve(), args.checkpoint_dir.resolve(), args.split, args.batch_size, choose_device(args.device))
    output = args.output_dir / f"{args.split}_v12_full_profile.npz"
    np.savez_compressed(output, **result)
    (args.output_dir / f"{args.split}_v12_manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "reconstruction_scope": RECONSTRUCTION_SCOPE,
                "split": args.split,
                "samples": len(result["prediction"]),
                "output": str(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"v12 predictions written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
