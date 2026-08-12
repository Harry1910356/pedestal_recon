#!/usr/bin/env python3
"""Evaluate the v10 all-feature, v7-style ResMLP models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lh_transitions.profile_recon_robust_sobolev_mtanh_v10 import (
    ModelConfig,
    ProfileResMLPModel,
    all_feature_indices,
    choose_device,
)


SPECIES = ("Te", "Ne")
COLORS = {"Te": "#d95f02", "Ne": "#1b75bc"}
DISPLAY_SCALE = {"Te": 1000.0, "Ne": 1.0}
UNITS = {"Te": "eV", "Ne": r"$10^{19}$ m$^{-3}$"}
R2_LIMITS = (0.95, 0.85, 0.80)
RMSE_LIMITS = {"Te": (75.0, 150.0, 250.0), "Ne": (0.10, 0.22, 0.35)}
TIER_NAMES = ("Excellent", "Good", "Acceptable", "Poor")
TIER_COLORS = ("#10b981", "#3b82f6", "#f59e0b", "#ef4444")


@dataclass
class Evaluation:
    keys: list[tuple[int, int]]
    target: np.ndarray
    prediction: np.ndarray


def safe_load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            f"{path} is not a new v10 all-feature checkpoint; retrain v10 first"
        )
    if checkpoint.get("version") != "v10":
        raise ValueError(f"{path} has version {checkpoint.get('version')!r}, expected 'v10'")
    if checkpoint.get("feature_selection") != "all":
        raise ValueError(f"{path} was not trained with all features")
    return checkpoint


def sample_keys(metadata_path: Path, times: np.ndarray) -> list[tuple[int, int]]:
    metadata = pl.read_parquet(metadata_path)
    keys: list[tuple[int, int] | None] = [None] * len(times)
    for row in metadata.iter_rows(named=True):
        start = int(row["train_start_idx"])
        stop = start + int(row["train_seq_len"])
        shot = int(row["shot_id"])
        for index in range(start, stop):
            keys[index] = (shot, int(round(float(times[index]) * 100.0)))
    if any(key is None for key in keys):
        raise RuntimeError("metadata does not cover every labeled sample")
    return [key for key in keys if key is not None]


def load_model(checkpoint: dict, device: torch.device) -> ProfileResMLPModel:
    config = ModelConfig(**checkpoint["model_config"])
    model = ProfileResMLPModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.in_proj[0].in_features != config.input_dim:
        raise RuntimeError("checkpoint input dimension is inconsistent")
    return model.to(device).eval()


def evaluate(
    data_root: Path,
    checkpoint_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> Evaluation:
    _, feature_names = all_feature_indices(data_root)
    checkpoints = {
        species: safe_load(checkpoint_dir / f"best_resmlp_v10_{species.lower()}.pth")
        for species in SPECIES
    }
    for species, checkpoint in checkpoints.items():
        if checkpoint.get("feature_names") != feature_names:
            raise ValueError(
                f"{species} checkpoint feature names/order differ from the current dataset"
            )
        if checkpoint["model_config"]["input_dim"] != len(feature_names):
            raise ValueError(
                f"{species} checkpoint expects {checkpoint['model_config']['input_dim']} "
                f"features; dataset contains {len(feature_names)}"
            )
    models = {species: load_model(checkpoints[species], device) for species in SPECIES}

    with np.load(data_root / split / "samples_train.npz") as archive:
        features = np.asarray(archive["X_train"], dtype=np.float32)
        target_norm = np.asarray(archive["Y_train"], dtype=np.float64)
        times = np.asarray(archive["T_train"])
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {features.shape[1]} columns; expected all {len(feature_names)} features"
        )

    predictions: dict[str, list[np.ndarray]] = {species: [] for species in SPECIES}
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            for species in SPECIES:
                predictions[species].append(models[species](batch).cpu().numpy())
    prediction_norm = np.concatenate(
        [np.concatenate(predictions[species], axis=0) for species in SPECIES], axis=1
    ).astype(np.float64)
    with np.load(data_root / "standardization_scalars.npz") as scalars:
        y_mean = np.asarray(scalars["y_mean"], dtype=np.float64)
        y_std = np.asarray(scalars["y_std"], dtype=np.float64)
    target = target_norm * y_std + y_mean
    prediction = prediction_norm * y_std + y_mean
    keys = sample_keys(data_root / split / "metadata.parquet", times)
    return Evaluation(keys=keys, target=target, prediction=prediction)


def sample_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=1))
    mae = np.mean(np.abs(error), axis=1)
    denominator = np.sum((target - np.mean(target, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - np.sum(error**2, axis=1) / np.maximum(denominator, 1e-12)
    robust_range = np.percentile(target, 95, axis=1) - np.percentile(target, 5, axis=1)
    nrmse = rmse / np.maximum(robust_range, 1e-8)
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}


def tier(values: np.ndarray, limits: tuple[float, float, float], higher: bool) -> np.ndarray:
    result = np.full(len(values), "Poor", dtype=object)
    if higher:
        result[values >= limits[2]] = "Acceptable"
        result[values >= limits[1]] = "Good"
        result[values >= limits[0]] = "Excellent"
    else:
        result[values <= limits[2]] = "Acceptable"
        result[values <= limits[1]] = "Good"
        result[values <= limits[0]] = "Excellent"
    return result


def statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def write_metrics(
    path: Path, keys: list[tuple[int, int]], metrics: dict[str, dict[str, np.ndarray]]
) -> None:
    fields = ["sample_index", "shot_id", "time_ms"]
    for species in SPECIES:
        fields.extend(
            f"{species.lower()}_{name}"
            for name in ("rmse", "mae", "nrmse", "r2", "rmse_tier", "r2_tier")
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (shot, centims) in enumerate(keys):
            row: dict[str, object] = {
                "sample_index": index,
                "shot_id": shot,
                "time_ms": centims / 100.0,
            }
            for species in SPECIES:
                for name in ("rmse", "mae", "nrmse", "r2", "rmse_tier", "r2_tier"):
                    row[f"{species.lower()}_{name}"] = metrics[species][name][index]
            writer.writerow(row)


def plot_summary(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, species in enumerate(SPECIES):
        for column, name in enumerate(("rmse", "r2")):
            values = metrics[species][name]
            finite = values[np.isfinite(values)]
            axes[row, column].hist(finite, bins=60, color=COLORS[species], alpha=0.8)
            axes[row, column].axvline(np.median(finite), color="black", linestyle="--")
            axes[row, column].set_title(f"{species} {name.upper()} distribution")
            axes[row, column].grid(alpha=0.25)
    fig.suptitle("v10 all-feature ResMLP — reconstruction distributions", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tiers(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, species in enumerate(SPECIES):
        for column, name in enumerate(("rmse_tier", "r2_tier")):
            counts = [int(np.sum(metrics[species][name] == tier_name)) for tier_name in TIER_NAMES]
            axes[row, column].bar(TIER_NAMES, counts, color=TIER_COLORS)
            axes[row, column].set_title(f"{species} {name.replace('_tier', '').upper()} tiers")
            axes[row, column].grid(axis="y", alpha=0.25)
    fig.suptitle("v10 all-feature ResMLP — quality tiers", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def example_indices(metrics: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, int]]:
    badness = metrics["Te"]["nrmse"] + metrics["Ne"]["nrmse"]
    order = np.argsort(badness)
    return [
        ("best", int(order[0])),
        ("good", int(order[round(0.10 * (len(order) - 1))])),
        ("bad", int(order[round(0.90 * (len(order) - 1))])),
        ("worst", int(order[-1])),
    ]


def plot_examples(
    path: Path, evaluation: Evaluation, metrics: dict[str, dict[str, np.ndarray]]
) -> None:
    examples = example_indices(metrics)
    grid = evaluation.target.shape[1] // 2
    rho = np.linspace(0.7, 1.25, grid)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for row, (label, index) in enumerate(examples):
        shot, centims = evaluation.keys[index]
        for column, species in enumerate(SPECIES):
            section = slice(column * grid, (column + 1) * grid)
            scale = DISPLAY_SCALE[species]
            axes[row, column].plot(
                rho, evaluation.target[index, section] * scale, color="black", label="target"
            )
            axes[row, column].plot(
                rho,
                evaluation.prediction[index, section] * scale,
                color=COLORS[species],
                label="v10 all features",
            )
            axes[row, column].set_title(
                f"{label} | shot {shot}, {centims / 100:.2f} ms | {species} | "
                f"R²={metrics[species]['r2'][index]:.3f}"
            )
            axes[row, column].set_ylabel(UNITS[species])
            axes[row, column].grid(alpha=0.25)
            if row == 0:
                axes[row, column].legend()
    fig.suptitle("v10 all-feature representative reconstructions", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "intergral_v7_weighted"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "v10_outputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "v10_visualizations")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate(
        data_root, checkpoint_dir, args.split, args.batch_size, choose_device(args.device)
    )
    grid = evaluation.target.shape[1] // 2
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for column, species in enumerate(SPECIES):
        section = slice(column * grid, (column + 1) * grid)
        scale = DISPLAY_SCALE[species]
        metrics[species] = sample_metrics(
            evaluation.target[:, section] * scale,
            evaluation.prediction[:, section] * scale,
        )
        metrics[species]["rmse_tier"] = tier(
            metrics[species]["rmse"], RMSE_LIMITS[species], False
        )
        metrics[species]["r2_tier"] = tier(metrics[species]["r2"], R2_LIMITS, True)

    prefix = args.split
    write_metrics(output_dir / f"{prefix}_v10_per_sample_metrics.csv", evaluation.keys, metrics)
    plot_summary(output_dir / f"{prefix}_v10_reconstruction_summary.png", metrics)
    plot_tiers(output_dir / f"{prefix}_v10_quality_tiers.png", metrics)
    plot_examples(output_dir / f"{prefix}_v10_examples.png", evaluation, metrics)
    _, feature_names = all_feature_indices(data_root)
    summary = {
        "model_version": "v10",
        "design": "v7_weighted_all_features",
        "feature_selection": "all",
        "input_dim": len(feature_names),
        "split": args.split,
        "sample_count": len(evaluation.keys),
        "metrics": {
            species: {
                name: statistics(metrics[species][name]) for name in ("rmse", "r2")
            }
            for species in SPECIES
        },
    }
    (output_dir / f"{prefix}_v10_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"v10 {args.split} visualizations written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
