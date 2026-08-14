#!/usr/bin/env python3
"""Run v11 Te/Ne checkpoints and save complete rho=0--1.25 profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lh_transitions.profile_recon_robust_sobolev_full_profile_v11 import (
    EXPECTED_GRID_POINTS,
    SPECIES,
    TARGET_KEY,
    ModelConfig,
    ProfileResMLPModel,
    all_feature_indices,
    choose_device,
)


def load_checkpoint(path: str | Path, species: str) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{path} is not a v11 checkpoint")
    expected = {
        "version": "v11",
        "target_key": TARGET_KEY,
        "species": species,
        "grid_points": EXPECTED_GRID_POINTS,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"{path} has {key}={checkpoint.get(key)!r}; expected {value!r}"
            )
    if checkpoint.get("feature_selection") != "all":
        raise ValueError(f"{path} was not trained with the v11 all-feature contract")
    return checkpoint


def build_model(checkpoint: dict, device: torch.device) -> ProfileResMLPModel:
    config = ModelConfig(**checkpoint["model_config"])
    if config.output_dim != EXPECTED_GRID_POINTS:
        raise ValueError(f"checkpoint output width is {config.output_dim}, expected 201")
    model = ProfileResMLPModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def predict_in_batches(
    model: ProfileResMLPModel,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            batches.append(model(batch).cpu().numpy())
    if not batches:
        return np.empty((0, EXPECTED_GRID_POINTS), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def run_prediction(
    data_root: str | Path,
    checkpoint_dir: str | Path,
    split: str,
    input_kind: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    data_root = Path(data_root)
    checkpoint_dir = Path(checkpoint_dir)
    _, feature_names = all_feature_indices(data_root)
    checkpoints = {
        species: load_checkpoint(
            checkpoint_dir / f"best_resmlp_v11_{species.lower()}_full_profile.pth",
            species,
        )
        for species in SPECIES
    }
    for species, checkpoint in checkpoints.items():
        if checkpoint.get("feature_names") != feature_names:
            raise ValueError(
                f"{species} checkpoint feature names/order differ from the current dataset"
            )

    if input_kind == "labeled":
        archive_path = data_root / split / "samples_train.npz"
        x_key, time_key = "X_train", "T_train"
    elif input_kind == "infer":
        archive_path = data_root / split / "samples_infer.npz"
        x_key, time_key = "X_infer", "T_infer"
    else:
        raise ValueError("input_kind must be 'labeled' or 'infer'")

    with np.load(archive_path) as archive:
        features = np.asarray(archive[x_key], dtype=np.float32)
        times = np.asarray(archive[time_key], dtype=np.float32)
        target_norm = (
            np.asarray(archive[TARGET_KEY], dtype=np.float32)
            if input_kind == "labeled" and TARGET_KEY in archive.files
            else None
        )
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError(
            f"{archive_path} has X shape {features.shape}; expected (*, {len(feature_names)})"
        )

    prediction_norm_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for species in SPECIES:
        prediction_norm = predict_in_batches(
            build_model(checkpoints[species], device), features, batch_size, device
        )
        mean = np.asarray(checkpoints[species]["y_mean"], dtype=np.float32)
        std = np.asarray(checkpoints[species]["y_std"], dtype=np.float32)
        prediction_norm_parts.append(prediction_norm)
        prediction_parts.append(prediction_norm * std + mean)

    result = {
        "prediction": np.concatenate(prediction_parts, axis=1),
        "prediction_normalized": np.concatenate(prediction_norm_parts, axis=1),
        "time_ms": times,
        "rho": np.linspace(0.0, 1.25, EXPECTED_GRID_POINTS, dtype=np.float32),
    }
    if target_norm is not None:
        with np.load(data_root / "standardization_scalars.npz") as scalars:
            mean = np.asarray(scalars["y_mean_fit"], dtype=np.float32)
            std = np.asarray(scalars["y_std_fit"], dtype=np.float32)
        result["target"] = target_norm * std + mean
        result["target_normalized"] = target_norm
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "intergral_v7_weighted"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "v11_outputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "v11_predictions")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--input-kind", choices=("labeled", "infer"), default="labeled")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_prediction(
        args.data_root.resolve(),
        args.checkpoint_dir.resolve(),
        args.split,
        args.input_kind,
        args.batch_size,
        choose_device(args.device),
    )
    output_path = output_dir / f"{args.split}_{args.input_kind}_v11_full_profile.npz"
    np.savez_compressed(output_path, **result)
    manifest = {
        "version": "v11",
        "split": args.split,
        "input_kind": args.input_kind,
        "sample_count": int(len(result["prediction"])),
        "prediction_shape": list(result["prediction"].shape),
        "rho_range": [0.0, 1.25],
        "grid_points_per_species": EXPECTED_GRID_POINTS,
        "output": str(output_path),
    }
    (output_dir / f"{args.split}_{args.input_kind}_v11_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"v11 predictions written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
