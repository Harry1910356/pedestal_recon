#!/usr/bin/env python3
"""Evaluate and visualize v11 full-profile reconstructions over rho=0--1.25."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lh_transitions.predict_full_profile_v11 import run_prediction
from lh_transitions.profile_recon_robust_sobolev_full_profile_v11 import (
    EXPECTED_GRID_POINTS,
    SPECIES,
    all_feature_indices,
    choose_device,
)


COLORS = {"Te": "#d95f02", "Ne": "#1b75bc"}
DISPLAY_SCALE = {"Te": 1000.0, "Ne": 1.0}
UNITS = {"Te": "eV", "Ne": r"$10^{19}$ m$^{-3}$"}


@dataclass
class Evaluation:
    keys: list[tuple[int, float]]
    rho: np.ndarray
    target: np.ndarray
    prediction: np.ndarray


def sample_keys(
    metadata_path: str | Path, times_ms: np.ndarray
) -> list[tuple[int, float]]:
    """Map packed NPZ rows back to ``(shot_id, time_ms)`` identifiers."""
    path = Path(metadata_path)
    if not path.exists():
        return [(-1, float(time)) for time in times_ms]
    metadata = pl.read_parquet(path)
    keys: list[tuple[int, float] | None] = [None] * len(times_ms)
    for row in metadata.iter_rows(named=True):
        start = int(row["train_start_idx"])
        stop = min(start + int(row["train_seq_len"]), len(keys))
        shot = int(row["shot_id"])
        for index in range(start, stop):
            keys[index] = (shot, float(times_ms[index]))
    if any(key is None for key in keys):
        raise RuntimeError(f"{path} does not cover every labeled sample")
    return [key for key in keys if key is not None]


def evaluate(
    data_root: Path,
    checkpoint_dir: Path,
    split: str,
    batch_size: int,
    device,
) -> Evaluation:
    result = run_prediction(
        data_root,
        checkpoint_dir,
        split,
        "labeled",
        batch_size,
        device,
    )
    if "target" not in result:
        raise ValueError("v11 visualization requires labeled data containing Y_fit")
    target = np.asarray(result["target"], dtype=np.float64)
    prediction = np.asarray(result["prediction"], dtype=np.float64)
    rho = np.asarray(result["rho"], dtype=np.float64)
    if target.shape != prediction.shape or target.shape[1] != 2 * EXPECTED_GRID_POINTS:
        raise ValueError(
            f"expected matching (*, {2 * EXPECTED_GRID_POINTS}) target/prediction arrays; "
            f"received {target.shape} and {prediction.shape}"
        )
    if len(rho) != EXPECTED_GRID_POINTS or not np.isclose(rho[[0, -1]], [0.0, 1.25]).all():
        raise ValueError("v11 rho grid must contain 201 points spanning 0--1.25")
    keys = sample_keys(data_root / split / "metadata.parquet", result["time_ms"])
    return Evaluation(keys=keys, rho=rho, target=target, prediction=prediction)


def sample_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate one set of reconstruction metrics per time slice."""
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=1))
    mae = np.mean(np.abs(error), axis=1)
    denominator = np.sum((target - np.mean(target, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1.0 - np.sum(error**2, axis=1) / np.maximum(denominator, 1e-12)
    robust_range = np.percentile(target, 95, axis=1) - np.percentile(target, 5, axis=1)
    nrmse = rmse / np.maximum(robust_range, 1e-8)
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}


def radial_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate errors across samples at each radial grid point."""
    error = prediction - target
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "bias": np.mean(error, axis=0),
    }


def statistics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {name: float("nan") for name in ("mean", "std", "median", "p05", "p95")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def species_sections() -> dict[str, slice]:
    return {
        "Te": slice(0, EXPECTED_GRID_POINTS),
        "Ne": slice(EXPECTED_GRID_POINTS, 2 * EXPECTED_GRID_POINTS),
    }


def calculate_metrics(
    evaluation: Evaluation,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    sample: dict[str, dict[str, np.ndarray]] = {}
    radial: dict[str, dict[str, np.ndarray]] = {}
    for species, section in species_sections().items():
        scale = DISPLAY_SCALE[species]
        target = evaluation.target[:, section] * scale
        prediction = evaluation.prediction[:, section] * scale
        sample[species] = sample_metrics(target, prediction)
        radial[species] = radial_metrics(target, prediction)
    return sample, radial


def write_sample_metrics(
    path: Path,
    keys: list[tuple[int, float]],
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = ["sample_index", "shot_id", "time_ms"] + [
        f"{species.lower()}_{name}"
        for species in SPECIES
        for name in ("rmse", "mae", "nrmse", "r2")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (shot, time_ms) in enumerate(keys):
            row: dict[str, float | int] = {
                "sample_index": index,
                "shot_id": shot,
                "time_ms": time_ms,
            }
            for species in SPECIES:
                for name in ("rmse", "mae", "nrmse", "r2"):
                    row[f"{species.lower()}_{name}"] = float(metrics[species][name][index])
            writer.writerow(row)


def write_radial_metrics(
    path: Path,
    rho: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    data: dict[str, np.ndarray] = {"rho": rho}
    for species in SPECIES:
        for name in ("rmse", "mae", "bias"):
            data[f"{species.lower()}_{name}"] = metrics[species][name]
    pl.DataFrame(data).write_csv(path)


def plot_metric_distributions(
    path: Path, metrics: dict[str, dict[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, species in enumerate(SPECIES):
        for column, name in enumerate(("rmse", "r2")):
            values = metrics[species][name]
            finite = values[np.isfinite(values)]
            axes[row, column].hist(finite, bins=60, color=COLORS[species], alpha=0.82)
            if finite.size:
                axes[row, column].axvline(
                    np.median(finite), color="black", linestyle="--", label="median"
                )
                axes[row, column].legend()
            axes[row, column].set_title(f"{species} {name.upper()} distribution")
            axes[row, column].set_xlabel(UNITS[species] if name == "rmse" else "R²")
            axes[row, column].set_ylabel("Samples")
            axes[row, column].grid(alpha=0.25)
    fig.suptitle("v11 full-profile reconstruction metrics (ρ=0–1.25)", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_radial_errors(
    path: Path, rho: np.ndarray, metrics: dict[str, dict[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, species in enumerate(SPECIES):
        axes[row, 0].plot(rho, metrics[species]["rmse"], color=COLORS[species], lw=2)
        axes[row, 0].set_title(f"{species} RMSE versus ρ")
        axes[row, 0].set_ylabel(UNITS[species])
        axes[row, 1].plot(rho, metrics[species]["bias"], color=COLORS[species], lw=2)
        axes[row, 1].axhline(0.0, color="black", lw=1, linestyle="--")
        axes[row, 1].set_title(f"{species} mean bias versus ρ")
        axes[row, 1].set_ylabel(UNITS[species])
        for axis in axes[row]:
            axis.set_xlabel("ρ")
            axis.set_xlim(0.0, 1.25)
            axis.grid(alpha=0.25)
    fig.suptitle("v11 radial error structure", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def example_indices(metrics: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, int]]:
    """Select representative samples by combined Te/Ne normalized RMSE."""
    badness = metrics["Te"]["nrmse"] + metrics["Ne"]["nrmse"]
    finite = np.flatnonzero(np.isfinite(badness))
    if finite.size == 0:
        raise ValueError("no finite samples are available for example plots")
    order = finite[np.argsort(badness[finite])]
    positions = (0.0, 0.25, 0.75, 1.0)
    labels = ("best", "lower quartile", "upper quartile", "worst")
    return [
        (label, int(order[round(position * (len(order) - 1))]))
        for label, position in zip(labels, positions)
    ]


def plot_examples(
    path: Path,
    evaluation: Evaluation,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    examples = example_indices(metrics)
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    for row, (label, index) in enumerate(examples):
        shot, time_ms = evaluation.keys[index]
        identity = f"sample {index}" if shot < 0 else f"shot {shot}, {time_ms:.2f} ms"
        for column, species in enumerate(SPECIES):
            section = species_sections()[species]
            scale = DISPLAY_SCALE[species]
            axis = axes[row, column]
            axis.plot(
                evaluation.rho,
                evaluation.target[index, section] * scale,
                color="black",
                lw=2,
                label="target Y_fit",
            )
            axis.plot(
                evaluation.rho,
                evaluation.prediction[index, section] * scale,
                color=COLORS[species],
                lw=2,
                label="v11 prediction",
            )
            axis.set_title(
                f"{label} | {identity} | {species} | R²={metrics[species]['r2'][index]:.3f}"
            )
            axis.set_xlabel("ρ")
            axis.set_ylabel(UNITS[species])
            axis.set_xlim(0.0, 1.25)
            axis.grid(alpha=0.25)
            if row == 0:
                axis.legend()
    fig.suptitle("v11 representative full-profile reconstructions", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_population_profiles(path: Path, evaluation: Evaluation) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for column, species in enumerate(SPECIES):
        section = species_sections()[species]
        scale = DISPLAY_SCALE[species]
        target = evaluation.target[:, section] * scale
        prediction = evaluation.prediction[:, section] * scale
        target_lo, target_med, target_hi = np.quantile(target, [0.1, 0.5, 0.9], axis=0)
        pred_lo, pred_med, pred_hi = np.quantile(prediction, [0.1, 0.5, 0.9], axis=0)
        axis = axes[column]
        axis.fill_between(
            evaluation.rho, target_lo, target_hi, color="black", alpha=0.12
        )
        axis.plot(evaluation.rho, target_med, color="black", lw=2, label="target median")
        axis.fill_between(
            evaluation.rho, pred_lo, pred_hi, color=COLORS[species], alpha=0.18
        )
        axis.plot(
            evaluation.rho,
            pred_med,
            color=COLORS[species],
            lw=2,
            label="v11 median",
        )
        axis.set_title(f"{species} population profile (bands: 10–90%)")
        axis.set_xlabel("ρ")
        axis.set_ylabel(UNITS[species])
        axis.set_xlim(0.0, 1.25)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("v11 target and prediction populations", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "intergral_v7_weighted"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "v11_outputs")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "v11_visualizations"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    data_root = args.data_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate(
        data_root,
        checkpoint_dir,
        args.split,
        args.batch_size,
        choose_device(args.device),
    )
    sample, radial = calculate_metrics(evaluation)
    prefix = f"{args.split}_v11"
    write_sample_metrics(output_dir / f"{prefix}_per_sample_metrics.csv", evaluation.keys, sample)
    write_radial_metrics(output_dir / f"{prefix}_radial_metrics.csv", evaluation.rho, radial)
    plot_metric_distributions(output_dir / f"{prefix}_metric_distributions.png", sample)
    plot_radial_errors(output_dir / f"{prefix}_radial_errors.png", evaluation.rho, radial)
    plot_examples(output_dir / f"{prefix}_examples.png", evaluation, sample)
    plot_population_profiles(output_dir / f"{prefix}_population_profiles.png", evaluation)

    _, feature_names = all_feature_indices(data_root)
    summary = {
        "model_version": "v11",
        "design": "v10_all_features_full_profile",
        "target_key": "Y_fit",
        "rho_range": [0.0, 1.25],
        "grid_points_per_species": EXPECTED_GRID_POINTS,
        "feature_selection": "all",
        "input_dim": len(feature_names),
        "split": args.split,
        "sample_count": len(evaluation.keys),
        "metrics": {
            species: {
                name: statistics(sample[species][name])
                for name in ("rmse", "mae", "nrmse", "r2")
            }
            for species in SPECIES
        },
    }
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"v11 {args.split} visualizations written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
