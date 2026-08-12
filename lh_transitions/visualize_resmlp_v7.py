#!/usr/bin/env python3
"""Evaluate and visualize the v7_weighted ResMLP reconstruction model.

The output layout deliberately mirrors the post-training diagnostics produced
by ``ped_spline_v4.py``: a per-sample metric table, summary distributions,
quality-tier counts, and representative best/good/bad/worst profile plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_mtanh_models import MODEL_SPECS, evaluate_one  # noqa: E402


SPEC = next(spec for spec in MODEL_SPECS if spec.version == "v7_weighted")
SPECIES = ("Te", "Ne")
COLORS = {"Te": "#d95f02", "Ne": "#1b75bc"}
TIER_COLORS = {
    "Excellent": "#10b981",
    "Good": "#3b82f6",
    "Acceptable": "#f59e0b",
    "Poor": "#ef4444",
}
R2_LIMITS = (0.95, 0.85, 0.80)
RMSE_LIMITS = {"Te": (75.0, 150.0, 250.0), "Ne": (0.10, 0.22, 0.35)}
DISPLAY_SCALE = {"Te": 1000.0, "Ne": 1.0}
UNITS = {"Te": "eV", "Ne": r"$10^{19}$ m$^{-3}$"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, help="default: ROOT/intergral_v7_weighted")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, help="default: ROOT/v7_weighted_outputs")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--rank-threshold", type=int, default=50)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    error = pred - true
    rmse = np.sqrt(np.mean(error**2, axis=1))
    mae = np.mean(np.abs(error), axis=1)
    centered = true - np.mean(true, axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    r2 = 1.0 - np.sum(error**2, axis=1) / np.maximum(denominator, 1e-12)
    robust_range = np.percentile(true, 95, axis=1) - np.percentile(true, 5, axis=1)
    fallback = np.maximum(np.mean(np.abs(true), axis=1), 1e-8)
    nrmse = rmse / np.maximum(robust_range, 0.05 * fallback)
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def tier(values: np.ndarray, limits: tuple[float, float, float], higher_is_better: bool) -> np.ndarray:
    labels = np.full(len(values), "Poor", dtype=object)
    if higher_is_better:
        labels[values >= limits[2]] = "Acceptable"
        labels[values >= limits[1]] = "Good"
        labels[values >= limits[0]] = "Excellent"
    else:
        labels[values <= limits[2]] = "Acceptable"
        labels[values <= limits[1]] = "Good"
        labels[values <= limits[0]] = "Excellent"
    return labels


def write_metrics(
    path: Path,
    keys: list[tuple[int, int]],
    metrics: dict[str, dict[str, np.ndarray]],
    badness: np.ndarray,
) -> None:
    fields = ["sample_index", "shot_id", "time_ms", "badness_score"]
    for species in SPECIES:
        prefix = species.lower()
        fields.extend(
            [f"{prefix}_rmse", f"{prefix}_mae", f"{prefix}_nrmse", f"{prefix}_r2",
             f"{prefix}_rmse_tier", f"{prefix}_r2_tier"]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in np.argsort(-badness):
            shot, centims = keys[index]
            row: dict[str, object] = {
                "sample_index": int(index), "shot_id": shot,
                "time_ms": centims / 100.0, "badness_score": badness[index],
            }
            for species in SPECIES:
                prefix = species.lower()
                for metric in ("rmse", "mae", "nrmse", "r2"):
                    row[f"{prefix}_{metric}"] = metrics[species][metric][index]
                row[f"{prefix}_rmse_tier"] = metrics[species]["rmse_tier"][index]
                row[f"{prefix}_r2_tier"] = metrics[species]["r2_tier"][index]
            writer.writerow(row)


def distribution_panel(
    ax: plt.Axes,
    values: np.ndarray,
    species: str,
    metric: str,
) -> None:
    finite = values[np.isfinite(values)]
    if metric == "R2":
        # Keep all counts but collect the long negative tail in the first bin.
        lower = min(-1.0, float(np.quantile(finite, 0.01)))
        shown = np.clip(finite, lower, 1.0)
        limits = R2_LIMITS
        xlabel = r"Per-profile $R^2$"
    else:
        upper = float(np.quantile(finite, 0.99))
        shown = np.clip(finite, 0.0, upper)
        limits = RMSE_LIMITS[species]
        xlabel = f"RMSE [{UNITS[species]}]"
    ax.hist(shown, bins=60, color=COLORS[species], alpha=0.78, edgecolor="white", linewidth=0.35)
    ax.axvline(np.median(finite), color="black", linestyle="--", linewidth=1.8,
               label=f"median = {np.median(finite):.3g}")
    ax.axvline(np.mean(finite), color="#7c3aed", linestyle=":", linewidth=1.8,
               label=f"mean = {np.mean(finite):.3g}")
    for limit, color in zip(limits, ("#10b981", "#3b82f6", "#f59e0b")):
        ax.axvline(limit, color=color, linewidth=1.0, alpha=0.8)
    ax.set_title(f"{species} {metric} distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time slices")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(fontsize=8)


def plot_summary(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    distribution_panel(axes[0, 0], metrics["Te"]["rmse"], "Te", "RMSE")
    distribution_panel(axes[0, 1], metrics["Te"]["r2"], "Te", "R2")
    distribution_panel(axes[1, 0], metrics["Ne"]["rmse"], "Ne", "RMSE")
    distribution_panel(axes[1, 1], metrics["Ne"]["r2"], "Ne", "R2")
    fig.suptitle("v7_weighted ResMLP — test reconstruction distributions", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tiers(path: Path, metrics: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    labels = list(TIER_COLORS)
    for row, species in enumerate(SPECIES):
        for column, metric in enumerate(("rmse", "r2")):
            ax = axes[row, column]
            values = metrics[species][f"{metric}_tier"]
            counts = np.asarray([np.sum(values == label) for label in labels])
            bars = ax.bar(labels, counts, color=[TIER_COLORS[label] for label in labels])
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{count}\n{count / len(values):.1%}", ha="center", va="bottom", fontsize=9)
            ax.set_title(f"{species} {metric.upper()} quality tiers")
            ax.set_ylabel("Time slices")
            ax.set_ylim(0, max(counts) * 1.14 if max(counts) else 1)
            ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    fig.suptitle("v7_weighted ResMLP — quality-tier composition", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def example_indices(badness: np.ndarray) -> list[tuple[str, int]]:
    ordered = np.argsort(badness)
    positions = (
        ("best", 0.0),
        ("good", 0.10),
        ("bad", 0.90),
        ("worst", 1.0),
    )
    return [(label, int(ordered[round(fraction * (len(ordered) - 1))])) for label, fraction in positions]


def plot_example(
    path: Path,
    label: str,
    index: int,
    key: tuple[int, int],
    target: np.ndarray,
    prediction: np.ndarray,
    fit_target: np.ndarray | None,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    grid = target.shape[1] // 2
    rho = np.linspace(0.7, 1.25, grid)
    rho_fit = np.linspace(0.0, 1.25, fit_target.shape[1] // 2) if fit_target is not None else None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), constrained_layout=True)
    for column, species in enumerate(SPECIES):
        slc = slice(column * grid, (column + 1) * grid)
        scale = DISPLAY_SCALE[species]
        ax = axes[column]
        if fit_target is not None and rho_fit is not None:
            fit_grid = fit_target.shape[1] // 2
            fit_slc = slice(column * fit_grid, (column + 1) * fit_grid)
            mask = (rho_fit >= 0.7) & (rho_fit <= 1.25)
            ax.plot(rho_fit[mask], fit_target[index, fit_slc][mask] * scale,
                    color="#9ca3af", linewidth=1.4, label="201-point fit")
        ax.plot(rho, target[index, slc] * scale, color="black", linewidth=2.5, label="mtanh target")
        ax.plot(rho, prediction[index, slc] * scale, color=COLORS[species], linewidth=2.2,
                label="v7 prediction")
        ax.set_title(
            f"{species}: RMSE={metrics[species]['rmse'][index]:.3g} {UNITS[species]}, "
            f"R²={metrics[species]['r2'][index]:.3f}"
        )
        ax.set_xlabel(r"Normalized radius $\rho$")
        ax.set_ylabel(f"{species} [{UNITS[species]}]")
        ax.set_xlim(0.69, 1.26)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(fontsize=9)
    shot, centims = key
    fig.suptitle(
        f"{label.upper()} case — sample {index}, shot {shot}, {centims / 100:.2f} ms",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_examples_overview(
    path: Path,
    examples: list[tuple[str, int]],
    keys: list[tuple[int, int]],
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    grid = target.shape[1] // 2
    rho = np.linspace(0.7, 1.25, grid)
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 3.5 * len(examples)), constrained_layout=True)
    for row, (label, index) in enumerate(examples):
        shot, centims = keys[index]
        for column, species in enumerate(SPECIES):
            slc = slice(column * grid, (column + 1) * grid)
            scale = DISPLAY_SCALE[species]
            ax = axes[row, column]
            ax.plot(rho, target[index, slc] * scale, color="black", linewidth=2.3, label="target")
            ax.plot(rho, prediction[index, slc] * scale, color=COLORS[species], linewidth=2.0, label="v7")
            ax.set_title(
                f"{label.title()} | shot {shot}, {centims / 100:.2f} ms | {species} | "
                f"RMSE={metrics[species]['rmse'][index]:.3g}, R²={metrics[species]['r2'][index]:.3f}"
            )
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(f"{species} [{UNITS[species]}]")
            ax.grid(True, linestyle=":", alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("v7_weighted representative reconstruction cases", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)), "std": float(np.std(finite)),
        "min": float(np.min(finite)), "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)), "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)), "max": float(np.max(finite)),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data_root = (args.data_root or root / "intergral_v7_weighted").resolve()
    output = (args.output_dir or root / "v7_weighted_outputs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_dir = output / f"{args.split}_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Evaluating v7_weighted on {args.split} with {device} ...")
    evaluation = evaluate_one(
        root, SPEC, data_root, args.split, args.batch_size, device, args.rank_threshold
    )
    target = evaluation.target
    prediction = evaluation.prediction
    grid = target.shape[1] // 2
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for column, species in enumerate(SPECIES):
        slc = slice(column * grid, (column + 1) * grid)
        scale = DISPLAY_SCALE[species]
        metrics[species] = sample_metrics(target[:, slc] * scale, prediction[:, slc] * scale)
        metrics[species]["r2_tier"] = tier(metrics[species]["r2"], R2_LIMITS, True)
        metrics[species]["rmse_tier"] = tier(metrics[species]["rmse"], RMSE_LIMITS[species], False)
    badness = np.mean(
        np.vstack((
            rank01(metrics["Te"]["nrmse"]), rank01(metrics["Ne"]["nrmse"]),
            rank01(-metrics["Te"]["r2"]), rank01(-metrics["Ne"]["r2"]),
        )), axis=0,
    )
    write_metrics(output / f"{args.split}_reconstruction_metrics.csv", evaluation.keys, metrics, badness)
    plot_summary(output / f"{args.split}_reconstruction_summary.png", metrics)
    plot_tiers(output / f"{args.split}_quality_tiers.png", metrics)

    fit_target = None
    npz_path = data_root / args.split / "samples_train.npz"
    scalar_path = data_root / "standardization_scalars.npz"
    with np.load(npz_path) as archive, np.load(scalar_path) as scalars:
        if "Y_fit" in archive.files and "y_mean_fit" in scalars.files and "y_std_fit" in scalars.files:
            fit_target = np.asarray(archive["Y_fit"], dtype=np.float64) * scalars["y_std_fit"] + scalars["y_mean_fit"]

    examples = example_indices(badness)
    plot_examples_overview(
        output / f"{args.split}_examples_overview.png", examples, evaluation.keys,
        target, prediction, metrics,
    )
    example_records = []
    for label, index in examples:
        shot, centims = evaluation.keys[index]
        filename = f"{label}_sample_{index}_shot_{shot}_time_{centims / 100:.0f}.png"
        plot_example(
            examples_dir / filename, label, index, evaluation.keys[index],
            target, prediction, fit_target, metrics,
        )
        example_records.append({"label": label, "sample_index": index, "shot_id": shot, "time_ms": centims / 100.0})

    summary: dict[str, object] = {
        "model_version": "v7_weighted", "split": args.split,
        "sample_count": len(target), "device": str(device),
        "tier_thresholds": {
            "r2": {"excellent": 0.95, "good": 0.85, "acceptable": 0.80},
            "te_rmse_eV": {"excellent": 75, "good": 150, "acceptable": 250},
            "ne_rmse_1e19_m-3": {"excellent": 0.10, "good": 0.22, "acceptable": 0.35},
        },
        "metrics": {}, "examples": example_records,
    }
    for species in SPECIES:
        summary["metrics"][species] = {
            "rmse": statistics(metrics[species]["rmse"]),
            "r2": statistics(metrics[species]["r2"]),
            "rmse_tiers": {label: int(np.sum(metrics[species]["rmse_tier"] == label)) for label in TIER_COLORS},
            "r2_tiers": {label: int(np.sum(metrics[species]["r2_tier"] == label)) for label in TIER_COLORS},
        }
    (output / f"{args.split}_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote v7 diagnostics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
