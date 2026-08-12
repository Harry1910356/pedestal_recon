#!/usr/bin/env python3
"""Fairly compare the v2--v8 mtanh profile reconstruction checkpoints.

Each checkpoint is evaluated with the normalized input and output scalars from
the data version it was trained on.  Samples are aligned by (shot_id, time_ms),
and all metrics are then calculated against one common physical-space target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    version: str
    data_dir: str
    checkpoint_suffix: str
    inverse_transform: str = "identity"
    architecture: str = "resmlp"


# v4 checkpoints have no version suffix in this project.  v3 is listed so a
# future v3 pair is picked up without changing this program.
MODEL_SPECS = (
    ModelSpec("v2", "intergral_v2", "_v2"),
    ModelSpec("v3", "intergral_v3", "_v3"),
    ModelSpec("v4", "intergral_v4", ""),
    ModelSpec("v5", "intergral_v5", "_v5"),
    ModelSpec("v6_log", "intergral_v6_log", "_v6_log", "arcsinh"),
    ModelSpec("v7_weighted", "intergral_v7_weighted", "_v7_weighted"),
    ModelSpec("v8_pinn", "intergral_v8", "_v8_pinn", "arcsinh", "pinn"),
)


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.drop(self.act(self.linear1(x)))
        x = self.drop(self.linear2(x))
        return self.act(x + residual)


class ProfileResMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, blocks: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.blocks = nn.ModuleList(ResMLPBlock(hidden_dim) for _ in range(blocks))
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


class ProfilePINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, blocks: int):
        super().__init__()
        self.grid_points = output_dim
        self.drho = 0.55 / (output_dim - 1)
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.blocks = nn.ModuleList(ResMLPBlock(hidden_dim) for _ in range(blocks))
        self.deriv_proj = nn.Linear(hidden_dim, output_dim - 1)
        self.edge_proj = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        derivatives = self.softplus(self.deriv_proj(x))
        derivatives = derivatives.clone()
        derivatives[:, :5] = 0.0
        edge = self.edge_proj(x)
        integrated = torch.flip(
            torch.cumsum(torch.flip(derivatives, dims=[1]), dim=1), dims=[1]
        ) * self.drho
        return torch.cat((edge + integrated, edge), dim=1)


@dataclass
class Evaluation:
    spec: ModelSpec
    keys: list[tuple[int, int]]
    prediction: np.ndarray
    target: np.ndarray
    feature_count: int
    rank_threshold: int | None


def parse_overrides(values: Iterable[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires VERSION=VALUE, got: {value}")
        key, item = value.split("=", 1)
        result[key.strip()] = item.strip()
    return result


def checkpoint_paths(root: Path, spec: ModelSpec) -> tuple[Path, Path]:
    stem = "best_resmlp_robust_model"
    return (
        root / f"{stem}_te_mtanh{spec.checkpoint_suffix}.pth",
        root / f"{stem}_ne_mtanh{spec.checkpoint_suffix}.pth",
    )


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state_dict")
    if "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def model_from_state(state: dict[str, torch.Tensor], architecture: str) -> nn.Module:
    input_dim = int(state["in_proj.0.weight"].shape[1])
    hidden_dim = int(state["in_proj.0.weight"].shape[0])
    block_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"blocks\.(\d+)\.linear1\.weight$", key))
    }
    blocks = max(block_ids) + 1 if block_ids else 0
    if architecture == "pinn" or "deriv_proj.weight" in state:
        output_dim = int(state["deriv_proj.weight"].shape[0]) + 1
        model = ProfilePINN(input_dim, output_dim, hidden_dim, blocks)
    else:
        output_dim = int(state["out_proj.weight"].shape[0])
        model = ProfileResMLP(input_dim, output_dim, hidden_dim, blocks)
    model.load_state_dict(state, strict=True)
    return model.eval()


def feature_ranks(root: Path, data_root: Path) -> tuple[list[str], dict[str, tuple[float, ...]]]:
    with (data_root / "train" / "feature_names.json").open(encoding="utf-8") as handle:
        names = json.load(handle)["X_features"]
    rank_files = (
        root / "lh_transitions" / "stats_output" / "feature_importance_ranking_te.csv",
        root / "lh_transitions" / "stats_output" / "feature_importance_ranking_ne.csv",
    )
    ranks: dict[str, list[float]] = {name: [] for name in names}
    for path in rank_files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("Feature") in ranks:
                    ranks[row["Feature"]].extend(
                        (float(row.get("RF_Rank", 999)), float(row.get("Permutation_Rank", 999)))
                    )
    return names, {name: tuple(values) for name, values in ranks.items()}


def select_features(
    root: Path,
    data_root: Path,
    required: int,
    explicit_threshold: int | None,
) -> tuple[np.ndarray, int | None]:
    names, ranks = feature_ranks(root, data_root)
    if required == len(names):
        return np.arange(required), None
    if not any(ranks.values()):
        raise RuntimeError(
            f"checkpoint expects {required} inputs but no ranking CSV is available to select them"
        )

    def indices(threshold: int) -> np.ndarray:
        return np.asarray(
            [i for i, name in enumerate(names) if any(rank <= threshold for rank in ranks[name])],
            dtype=np.int64,
        )

    if explicit_threshold is not None:
        chosen = indices(explicit_threshold)
        if len(chosen) != required:
            raise RuntimeError(
                f"rank threshold {explicit_threshold} selects {len(chosen)} features; "
                f"checkpoint requires {required}"
            )
        return chosen, explicit_threshold

    candidates = [(abs(threshold - 50), -threshold, threshold, indices(threshold)) for threshold in range(1, 1000)]
    matches = [item for item in candidates if len(item[3]) == required]
    if not matches:
        raise RuntimeError(
            f"cannot reproduce the checkpoint's {required} input features from the current ranking CSVs; "
            "use --rank-threshold after restoring the ranking files used for training"
        )
    _, _, threshold, chosen = min(matches)
    return chosen, threshold


def sample_keys(metadata_path: Path, times: np.ndarray) -> list[tuple[int, int]]:
    metadata = pl.read_parquet(metadata_path)
    keys: list[tuple[int, int] | None] = [None] * len(times)
    for row in metadata.iter_rows(named=True):
        start = int(row["train_start_idx"])
        stop = start + int(row["train_seq_len"])
        shot = int(row["shot_id"])
        for index in range(start, stop):
            # T is float32 in the saved NPZ. Quantizing to 0.01 ms makes the
            # join stable while remaining much finer than the sampling period.
            keys[index] = (shot, int(round(float(times[index]) * 100.0)))
    if any(key is None for key in keys):
        raise RuntimeError(f"metadata does not cover all {len(times)} training samples")
    return [key for key in keys if key is not None]


def inverse_targets(values: np.ndarray, mean: np.ndarray, std: np.ndarray, transform: str) -> np.ndarray:
    physical = values * std + mean
    if transform == "arcsinh":
        physical = np.sinh(physical)
    return physical


def evaluate_one(
    root: Path,
    spec: ModelSpec,
    data_root: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    threshold: int | None,
) -> Evaluation:
    te_path, ne_path = checkpoint_paths(root, spec)
    te_state, ne_state = load_state(te_path), load_state(ne_path)
    te_model, ne_model = model_from_state(te_state, spec.architecture), model_from_state(ne_state, spec.architecture)
    input_dim = int(te_state["in_proj.0.weight"].shape[1])
    if int(ne_state["in_proj.0.weight"].shape[1]) != input_dim:
        raise RuntimeError("Te and Ne checkpoints use different input dimensions")

    feature_indices, resolved_threshold = select_features(root, data_root, input_dim, threshold)
    with np.load(data_root / split / "samples_train.npz") as data:
        x = np.asarray(data["X_train"][:, feature_indices], dtype=np.float32)
        y = np.asarray(data["Y_train"], dtype=np.float64)
        times = np.asarray(data["T_train"])
    scalars = np.load(data_root / "standardization_scalars.npz")
    y_mean, y_std = np.asarray(scalars["y_mean"]), np.asarray(scalars["y_std"])
    grid = len(y_mean) // 2
    if y.shape[1] != 2 * grid:
        raise RuntimeError(f"target width {y.shape[1]} does not match scalar width {len(y_mean)}")

    predictions: list[np.ndarray] = []
    te_model.to(device)
    ne_model.to(device)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred = torch.cat((te_model(batch), ne_model(batch)), dim=1)
            predictions.append(pred.cpu().numpy())
    pred_norm = np.concatenate(predictions).astype(np.float64)
    prediction = inverse_targets(pred_norm, y_mean, y_std, spec.inverse_transform)
    target = inverse_targets(y, y_mean, y_std, spec.inverse_transform)
    keys = sample_keys(data_root / split / "metadata.parquet", times)
    return Evaluation(spec, keys, prediction, target, input_dim, resolved_threshold)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else float("nan")
    scale = float(np.std(target))
    nrmse = rmse / scale if scale > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}


def align(evaluations: list[Evaluation]) -> tuple[list[tuple[int, int]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared = set(evaluations[0].keys)
    for evaluation in evaluations[1:]:
        shared.intersection_update(evaluation.keys)
    keys = sorted(shared)
    if not keys:
        raise RuntimeError("the available versions have no common (shot_id, time_ms) test samples")
    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for evaluation in evaluations:
        lookup = {key: index for index, key in enumerate(evaluation.keys)}
        indices = np.asarray([lookup[key] for key in keys])
        predictions[evaluation.spec.version] = evaluation.prediction[indices]
        targets[evaluation.spec.version] = evaluation.target[indices]
    return keys, predictions, targets


def save_results(
    output: Path,
    evaluations: list[Evaluation],
    keys: list[tuple[int, int]],
    predictions: dict[str, np.ndarray],
    reference_target: np.ndarray,
) -> list[dict[str, float | int | str]]:
    output.mkdir(parents=True, exist_ok=True)
    grid = reference_target.shape[1] // 2
    rows: list[dict[str, float | int | str]] = []
    per_sample_rows: list[dict[str, float | int | str]] = []
    for evaluation in evaluations:
        version = evaluation.spec.version
        pred = predictions[version]
        te = regression_metrics(reference_target[:, :grid], pred[:, :grid])
        ne = regression_metrics(reference_target[:, grid:], pred[:, grid:])
        score = (te["nrmse"] + ne["nrmse"]) / 2.0
        rows.append(
            {
                "version": version,
                "samples": len(keys),
                "features": evaluation.feature_count,
                "rank_threshold": evaluation.rank_threshold if evaluation.rank_threshold is not None else "all",
                "te_rmse": te["rmse"], "te_mae": te["mae"], "te_r2": te["r2"], "te_nrmse": te["nrmse"],
                "ne_rmse": ne["rmse"], "ne_mae": ne["mae"], "ne_r2": ne["r2"], "ne_nrmse": ne["nrmse"],
                "macro_nrmse": score,
            }
        )
        for index, (shot, time_centims) in enumerate(keys):
            te_rmse = float(np.sqrt(np.mean((pred[index, :grid] - reference_target[index, :grid]) ** 2)))
            ne_rmse = float(np.sqrt(np.mean((pred[index, grid:] - reference_target[index, grid:]) ** 2)))
            per_sample_rows.append(
                {"version": version, "shot_id": shot, "time_ms": time_centims / 100.0,
                 "te_rmse": te_rmse, "ne_rmse": ne_rmse}
            )
    rows.sort(key=lambda row: float(row["macro_nrmse"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    def write_csv(path: Path, records: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv(output / "model_ranking.csv", rows)
    write_csv(output / "per_sample_metrics.csv", per_sample_rows)
    return rows


def save_plots(
    output: Path,
    rows: list[dict[str, float | int | str]],
    keys: list[tuple[int, int]],
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    plot_samples: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    versions = [str(row["version"]) for row in rows]
    x = np.arange(len(versions))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, [float(row["te_nrmse"]) for row in rows], 0.36, label="Te")
    axes[0].bar(x + 0.18, [float(row["ne_nrmse"]) for row in rows], 0.36, label="Ne")
    axes[0].set_ylabel("NRMSE (lower is better)")
    axes[0].set_xticks(x, versions, rotation=25, ha="right")
    axes[0].legend()
    axes[1].bar(x - 0.18, [float(row["te_r2"]) for row in rows], 0.36, label="Te")
    axes[1].bar(x + 0.18, [float(row["ne_r2"]) for row in rows], 0.36, label="Ne")
    axes[1].set_ylabel("R2 (higher is better)")
    axes[1].set_xticks(x, versions, rotation=25, ha="right")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "metric_comparison.png", dpi=180)
    plt.close(fig)

    best = versions[0]
    grid = target.shape[1] // 2
    errors = np.mean((predictions[best] - target) ** 2, axis=1)
    count = max(1, min(plot_samples, len(keys)))
    quantiles = np.linspace(0.1, 0.9, count)
    ordered = np.argsort(errors)
    selected = [int(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]) for q in quantiles]
    rho = np.linspace(0.7, 1.25, grid)
    fig, axes = plt.subplots(count, 2, figsize=(12, 4 * count), squeeze=False)
    for row_index, sample_index in enumerate(selected):
        shot, time_centims = keys[sample_index]
        for species_index, (name, slc) in enumerate((("Te", slice(0, grid)), ("Ne", slice(grid, 2 * grid)))):
            ax = axes[row_index, species_index]
            ax.plot(rho, target[sample_index, slc], color="black", linewidth=2.5, label="target")
            for version in versions:
                ax.plot(rho, predictions[version][sample_index, slc], linewidth=1.4, label=version)
            ax.set_title(f"Shot {shot}, {time_centims / 100:.2f} ms — {name}")
            ax.set_xlabel("rho")
            ax.grid(alpha=0.25)
            if row_index == 0 and species_index == 1:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "reconstruction_comparison.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="project root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--versions", nargs="+", help="versions to include (default: all v2--v8)")
    parser.add_argument("--data-root", action="append", default=[], metavar="VERSION=PATH", help="override a version's data directory")
    parser.add_argument("--rank-threshold", action="append", default=[], metavar="VERSION=N", help="override feature-rank threshold")
    parser.add_argument("--reference-version", help="version whose physical target is used (default: newest evaluated)")
    parser.add_argument("--output", type=Path, default=Path("model_comparison_results"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--strict", action="store_true", help="fail instead of skipping incomplete versions")
    return parser


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        data_overrides = parse_overrides(args.data_root, "--data-root")
        threshold_values = parse_overrides(args.rank_threshold, "--rank-threshold")
        thresholds = {key: int(value) for key, value in threshold_values.items()}
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    wanted = set(args.versions or [spec.version for spec in MODEL_SPECS])
    unknown = wanted.difference(spec.version for spec in MODEL_SPECS)
    if unknown:
        print(f"error: unknown versions: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    device = choose_device(args.device)
    print(f"Using device: {device}")
    evaluations: list[Evaluation] = []
    skipped: list[str] = []
    for spec in MODEL_SPECS:
        if spec.version not in wanted:
            continue
        data_root = Path(data_overrides.get(spec.version, root / spec.data_dir)).resolve()
        te_path, ne_path = checkpoint_paths(root, spec)
        required = [data_root / args.split / "samples_train.npz", data_root / args.split / "metadata.parquet", data_root / "standardization_scalars.npz", te_path, ne_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            message = f"{spec.version}: missing {', '.join(missing)}"
            if args.strict:
                print(f"error: {message}", file=sys.stderr)
                return 1
            skipped.append(message)
            print(f"SKIP {message}")
            continue
        print(f"Evaluating {spec.version} from {data_root} ...")
        try:
            evaluation = evaluate_one(root, spec, data_root, args.split, args.batch_size, device, thresholds.get(spec.version))
        except Exception as error:
            if args.strict:
                raise
            message = f"{spec.version}: {type(error).__name__}: {error}"
            skipped.append(message)
            print(f"SKIP {message}")
            continue
        evaluations.append(evaluation)
        print(f"  {len(evaluation.keys)} samples, {evaluation.feature_count} features, rank threshold={evaluation.rank_threshold}")

    if not evaluations:
        print("error: no complete model/data pair could be evaluated", file=sys.stderr)
        return 1
    keys, predictions, targets = align(evaluations)
    available = {evaluation.spec.version for evaluation in evaluations}
    reference = args.reference_version or evaluations[-1].spec.version
    if reference not in available:
        print(f"error: reference version {reference!r} was not evaluated", file=sys.stderr)
        return 1
    rows = save_results(output, evaluations, keys, predictions, targets[reference])
    save_plots(output, rows, keys, predictions, targets[reference], args.plot_samples)
    report = {
        "split": args.split,
        "device": str(device),
        "reference_version": reference,
        "shared_samples": len(keys),
        "evaluated_versions": [evaluation.spec.version for evaluation in evaluations],
        "skipped": skipped,
        "ranking_metric": "macro_nrmse = mean(Te NRMSE, Ne NRMSE); lower is better",
    }
    (output / "run_info.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRanking (lower macro NRMSE is better):")
    for row in rows:
        print(f"  {row['rank']}. {row['version']:<14} macro_NRMSE={float(row['macro_nrmse']):.6f}  Te_R2={float(row['te_r2']):.6f}  Ne_R2={float(row['ne_r2']):.6f}")
    print(f"\nCompared {len(keys)} shared samples; results written to {output}")
    if len(rows) == 1:
        print("Warning: only one complete version was available, so this is not yet a cross-version comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
