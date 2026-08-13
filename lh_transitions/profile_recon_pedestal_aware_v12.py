#!/usr/bin/env python3
"""Fine-tune v11 full-profile models with a pedestal-aware physical-slope loss.

v11 inherited a core-to-edge decreasing weight from the cropped-profile model.
On the full rho=0--1.25 grid that weighting suppresses the pedestal.  v12 keeps
the same ResMLP architecture but computes derivatives in physical units and
uses the target slope to focus value/gradient loss around sharp pedestal drops.
It also supervises the magnitude and radial location of the steepest drop.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


VERSION = "v12"
RECONSTRUCTION_SCOPE = "full_profile"
SPECIES = ("Te", "Ne")
TARGET_KEY = "Y_fit"
RHO_MIN = 0.0
RHO_MAX = 1.25
EXPECTED_GRID_POINTS = 201


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    """Resolve an explicit device name or select the best available device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def all_feature_indices(data_root: str | Path) -> tuple[list[int], list[str]]:
    """Return all saved input features in their dataset column order."""
    path = Path(data_root) / "train" / "feature_names.json"
    if not path.exists():
        raise FileNotFoundError(f"missing feature definition: {path}")
    names = json.loads(path.read_text(encoding="utf-8")).get("X_features", [])
    if not names:
        raise ValueError(f"{path} does not contain a non-empty X_features list")
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate X feature names")
    return list(range(len(names))), names


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    output_dim: int = EXPECTED_GRID_POINTS
    hidden_dim: int = 256
    num_blocks: int = 4
    dropout_p: float = 0.1


class ResMLPBlock(nn.Module):
    """Residual MLP block used by the full-profile reconstructor."""

    def __init__(self, hidden_dim: int, dropout_p: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_p)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.act(self.linear1(values))
        values = self.drop(values)
        values = self.linear2(values)
        values = self.drop(values)
        return self.act(values + residual)


class ProfileResMLPModel(nn.Module):
    """Residual MLP that reconstructs one complete 201-point profile."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = EXPECTED_GRID_POINTS,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout_p)
        )
        self.blocks = nn.ModuleList(
            [ResMLPBlock(hidden_dim, dropout_p) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.in_proj(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out_proj(hidden)


class FullProfileDataset(Dataset):
    """Load all input features and one species from the normalized Y_fit target."""

    def __init__(self, npz_path: str | Path, species: str) -> None:
        path = Path(npz_path)
        if species not in SPECIES:
            raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
        with np.load(path) as archive:
            required = {"X_train", TARGET_KEY, "T_train"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} is missing required arrays: {sorted(missing)}")
            features = np.asarray(archive["X_train"], dtype=np.float32)
            full_target = np.asarray(archive[TARGET_KEY], dtype=np.float32)
            times = np.asarray(archive["T_train"], dtype=np.float32)
        if full_target.ndim != 2 or full_target.shape[1] % 2:
            raise ValueError(f"Y_fit must contain equal Te/Ne sections, got {full_target.shape}")
        if len(features) != len(full_target) or len(times) != len(full_target):
            raise ValueError("X_train, Y_fit, and T_train sample counts do not match")
        self.grid_points = full_target.shape[1] // 2
        section = slice(0, self.grid_points) if species == "Te" else slice(self.grid_points, None)
        self.X = torch.from_numpy(features)
        self.Y = torch.from_numpy(full_target[:, section].copy())
        self.T = torch.from_numpy(times)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index], self.T[index]


def species_fit_scalars(data_root: str | Path, species: str) -> tuple[np.ndarray, np.ndarray]:
    """Return full-profile denormalization arrays for one species."""
    with np.load(Path(data_root) / "standardization_scalars.npz") as scalars:
        mean = np.asarray(scalars["y_mean_fit"], dtype=np.float32)
        std = np.asarray(scalars["y_std_fit"], dtype=np.float32)
    if mean.shape != std.shape or mean.ndim != 1 or len(mean) % 2:
        raise ValueError("y_mean_fit/y_std_fit must be equal-length Te/Ne vectors")
    grid_points = len(mean) // 2
    section = slice(0, grid_points) if species == "Te" else slice(grid_points, None)
    return mean[section], std[section]


def validate_data_contract(
    data_root: Path,
    feature_names: list[str],
    train_dataset: FullProfileDataset,
    val_dataset: FullProfileDataset,
) -> None:
    """Validate the complete-profile grid and feature order across splits."""
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if dataset.X.shape[1] != len(feature_names):
            raise ValueError(
                f"{split_name} X has {dataset.X.shape[1]} columns; expected {len(feature_names)}"
            )
        if dataset.grid_points != EXPECTED_GRID_POINTS:
            raise ValueError(
                f"{split_name} Y_fit has {dataset.grid_points} points per species; "
                f"expected {EXPECTED_GRID_POINTS}"
            )
    for split_name in ("val", "test"):
        path = data_root / split_name / "feature_names.json"
        if path.exists():
            names = json.loads(path.read_text(encoding="utf-8")).get("X_features", [])
            if names != feature_names:
                raise ValueError(f"{split_name} feature order differs from train: {path}")


@dataclass(frozen=True)
class TrainingConfig:
    # Defaults are the balanced setting selected on the held-out pedestal set:
    # it recovers slope magnitude without the Ne profile degradation observed
    # with the more aggressive first trial.
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    weight_value: float = 8.0
    weight_gradient: float = 1.5
    weight_peak_magnitude: float = 0.5
    weight_peak_location: float = 0.1
    pedestal_point_boost: float = 4.0
    pedestal_sample_boost: float = 0.25
    pedestal_quantile: float = 0.8
    pedestal_rho_min: float = 0.7
    pedestal_rho_max: float = 1.1
    location_temperature: float = 0.2
    huber_delta: float = 0.5
    gradient_clip: float = 1.0
    seed: int = 42


def physical_profile_arrays(
    target_normalized: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(target_normalized, dtype=np.float64)
        * np.asarray(y_std, dtype=np.float64)[None, :]
        + np.asarray(y_mean, dtype=np.float64)[None, :]
    )


def normalized_steepest_slope_score(
    profiles: np.ndarray,
    rho: np.ndarray,
    rho_min: float = 0.7,
    rho_max: float = 1.1,
) -> np.ndarray:
    """Return max negative physical slope divided by full-profile range."""
    profiles = np.asarray(profiles, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    interval_rho = 0.5 * (rho[:-1] + rho[1:])
    interval_mask = (interval_rho >= rho_min) & (interval_rho <= rho_max)
    slopes = np.diff(profiles, axis=1) / np.diff(rho)[None, :]
    descent = np.maximum(-slopes[:, interval_mask], 0.0)
    scale = np.maximum(np.ptp(profiles, axis=1), 1e-8)
    return np.max(descent, axis=1) / scale


def calibrate_pedestal_threshold(
    target_normalized: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    rho: np.ndarray,
    quantile: float,
    rho_min: float,
    rho_max: float,
) -> float:
    profiles = physical_profile_arrays(target_normalized, y_mean, y_std)
    scores = normalized_steepest_slope_score(profiles, rho, rho_min, rho_max)
    return float(np.quantile(scores[np.isfinite(scores)], quantile))


class PedestalAwarePhysicalSlopeLoss(nn.Module):
    """Target-adaptive full-profile loss evaluated in physical derivative space."""

    def __init__(
        self,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        rho: np.ndarray,
        pedestal_threshold: float,
        *,
        weight_gradient: float = 4.0,
        weight_value: float = 1.0,
        weight_peak_magnitude: float = 2.0,
        weight_peak_location: float = 0.5,
        pedestal_point_boost: float = 8.0,
        pedestal_sample_boost: float = 1.0,
        pedestal_rho_min: float = 0.7,
        pedestal_rho_max: float = 1.1,
        location_temperature: float = 0.2,
        huber_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if location_temperature <= 0 or pedestal_threshold < 0:
            raise ValueError("location_temperature must be positive and threshold non-negative")
        self.weight_gradient = weight_gradient
        self.weight_value = weight_value
        self.weight_peak_magnitude = weight_peak_magnitude
        self.weight_peak_location = weight_peak_location
        self.pedestal_point_boost = pedestal_point_boost
        self.pedestal_sample_boost = pedestal_sample_boost
        self.pedestal_threshold = pedestal_threshold
        self.location_temperature = location_temperature
        self.huber_delta = huber_delta
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.register_buffer("rho", torch.as_tensor(rho, dtype=torch.float32))
        centers = 0.5 * (self.rho[:-1] + self.rho[1:])
        self.register_buffer(
            "pedestal_interval_mask",
            (centers >= pedestal_rho_min) & (centers <= pedestal_rho_max),
        )

    def components(
        self, prediction_normalized: torch.Tensor, target_normalized: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prediction_physical = prediction_normalized * self.y_std + self.y_mean
        target_physical = target_normalized * self.y_std + self.y_mean
        dr = torch.diff(self.rho)
        prediction_slope = torch.diff(prediction_physical, dim=-1) / dr
        target_slope = torch.diff(target_physical, dim=-1) / dr

        profile_scale = (
            target_physical.amax(dim=-1) - target_physical.amin(dim=-1)
        ).clamp_min(1e-8)
        prediction_slope_n = prediction_slope / profile_scale[:, None]
        target_slope_n = target_slope / profile_scale[:, None]
        target_descent = torch.relu(-target_slope_n)

        region_descent = target_descent[:, self.pedestal_interval_mask]
        region_max = region_descent.amax(dim=-1).clamp_min(1e-8)
        gate = (region_max >= self.pedestal_threshold).to(target_normalized.dtype)

        local_importance = torch.zeros_like(target_descent)
        local_importance[:, self.pedestal_interval_mask] = (
            region_descent / region_max[:, None]
        ).square()
        gradient_weights = 1.0 + self.pedestal_point_boost * local_importance
        point_importance = torch.maximum(
            F.pad(local_importance, (1, 0)), F.pad(local_importance, (0, 1))
        )
        point_weights = 1.0 + self.pedestal_point_boost * point_importance

        value_error = F.huber_loss(
            prediction_normalized,
            target_normalized,
            delta=self.huber_delta,
            reduction="none",
        )
        value = (value_error * point_weights).sum(dim=-1) / point_weights.sum(dim=-1)

        gradient_error = F.huber_loss(
            prediction_slope_n,
            target_slope_n,
            delta=self.huber_delta,
            reduction="none",
        )
        gradient = (gradient_error * gradient_weights).sum(dim=-1) / gradient_weights.sum(
            dim=-1
        )

        pred_region = prediction_slope_n[:, self.pedestal_interval_mask]
        true_region = target_slope_n[:, self.pedestal_interval_mask]
        tau = self.location_temperature
        prediction_peak = -tau * torch.logsumexp(-pred_region / tau, dim=-1)
        target_peak = -tau * torch.logsumexp(-true_region / tau, dim=-1)
        peak_magnitude = F.huber_loss(
            prediction_peak, target_peak, delta=self.huber_delta, reduction="none"
        )

        target_probability = torch.softmax(-true_region.detach() / tau, dim=-1)
        prediction_log_probability = torch.log_softmax(-pred_region / tau, dim=-1)
        target_log_probability = torch.log(target_probability.clamp_min(1e-12))
        peak_location = torch.sum(
            target_probability * (target_log_probability - prediction_log_probability), dim=-1
        )

        per_sample = (
            self.weight_value * value
            + self.weight_gradient * gradient
            + gate
            * (
                self.weight_peak_magnitude * peak_magnitude
                + self.weight_peak_location * peak_location
            )
        )
        sample_weights = 1.0 + self.pedestal_sample_boost * gate
        total = torch.sum(per_sample * sample_weights) / torch.sum(sample_weights)
        return {
            "total": total,
            "value": value.mean(),
            "gradient": gradient.mean(),
            "peak_magnitude": (peak_magnitude * gate).sum() / gate.sum().clamp_min(1.0),
            "peak_location": (peak_location * gate).sum() / gate.sum().clamp_min(1.0),
            "pedestal_fraction": gate.mean(),
        }

    def forward(
        self, prediction_normalized: torch.Tensor, target_normalized: torch.Tensor
    ) -> torch.Tensor:
        return self.components(prediction_normalized, target_normalized)["total"]


def load_v11_initialization(
    model: ProfileResMLPModel, checkpoint_dir: Path | None, species: str
) -> str | None:
    if checkpoint_dir is None:
        return None
    path = checkpoint_dir / f"best_resmlp_v11_{species.lower()}_full_profile.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "v11" or checkpoint.get("species") != species:
        raise ValueError(f"incompatible v11 initialization checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return str(path)


def train_species(
    species: str,
    data_root: Path,
    output_dir: Path,
    feature_names: list[str],
    training: TrainingConfig,
    device: torch.device,
    init_checkpoint_dir: Path | None,
) -> Path:
    train_dataset = FullProfileDataset(data_root / "train" / "samples_train.npz", species)
    val_dataset = FullProfileDataset(data_root / "val" / "samples_train.npz", species)
    validate_data_contract(data_root, feature_names, train_dataset, val_dataset)
    model_config = ModelConfig(input_dim=len(feature_names), output_dim=EXPECTED_GRID_POINTS)
    model = ProfileResMLPModel(**asdict(model_config)).to(device)
    initialized_from = load_v11_initialization(model, init_checkpoint_dir, species)
    y_mean, y_std = species_fit_scalars(data_root, species)
    rho = np.linspace(RHO_MIN, RHO_MAX, EXPECTED_GRID_POINTS, dtype=np.float32)
    threshold = calibrate_pedestal_threshold(
        train_dataset.Y.numpy(), y_mean, y_std, rho, training.pedestal_quantile,
        training.pedestal_rho_min, training.pedestal_rho_max,
    )
    criterion = PedestalAwarePhysicalSlopeLoss(
        y_mean, y_std, rho, threshold,
        weight_gradient=training.weight_gradient,
        weight_value=training.weight_value,
        weight_peak_magnitude=training.weight_peak_magnitude,
        weight_peak_location=training.weight_peak_location,
        pedestal_point_boost=training.pedestal_point_boost,
        pedestal_sample_boost=training.pedestal_sample_boost,
        pedestal_rho_min=training.pedestal_rho_min,
        pedestal_rho_max=training.pedestal_rho_max,
        location_temperature=training.location_temperature,
        huber_delta=training.huber_delta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    train_loader = DataLoader(train_dataset, batch_size=training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training.batch_size, shuffle=False)
    checkpoint_path = output_dir / f"best_resmlp_v12_{species.lower()}_pedestal_aware.pth"
    history_path = output_dir / f"training_history_v12_{species.lower()}.csv"
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    print(
        f"[{species}] v12 physical-slope threshold={threshold:.6g}; "
        f"initialized_from={initialized_from or 'scratch'}"
    )
    for epoch in range(1, training.epochs + 1):
        model.train()
        train_total = 0.0
        for features, targets, _ in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            optimizer.step()
            train_total += loss.detach().item() * len(features)
        train_loss = train_total / len(train_dataset)
        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for features, targets, _ in val_loader:
                features, targets = features.to(device), targets.to(device)
                val_total += criterion(model(features), targets).item() * len(features)
        val_loss = val_total / len(val_dataset)
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": learning_rate})
        print(f"[{species}] epoch {epoch:03d}/{training.epochs} train={train_loss:.6f} val={val_loss:.6f} lr={learning_rate:.3g}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "version": VERSION,
                    "reconstruction_scope": RECONSTRUCTION_SCOPE,
                    "design": "v11_pedestal_aware_physical_slope",
                    "target_key": TARGET_KEY, "profile_kind": "physical_fit",
                    "rho_min": RHO_MIN, "rho_max": RHO_MAX,
                    "grid_points": EXPECTED_GRID_POINTS, "feature_selection": "all",
                    "feature_names": feature_names, "kept_indices": list(range(len(feature_names))),
                    "model_config": asdict(model_config), "training_config": asdict(training),
                    "model_state": model.state_dict(), "species": species,
                    "y_mean": y_mean, "y_std": y_std,
                    "pedestal_threshold": threshold, "initialized_from": initialized_from,
                    "best_epoch": epoch, "best_val_loss": best_val_loss,
                }, checkpoint_path,
            )
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("intergral_v7_weighted"))
    parser.add_argument("--output-dir", type=Path, default=Path("v12_outputs"))
    parser.add_argument("--init-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--target", choices=("both", "Te", "Ne"), default="both")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-value", type=float, default=8.0)
    parser.add_argument("--weight-gradient", type=float, default=1.5)
    parser.add_argument("--weight-peak-magnitude", type=float, default=0.5)
    parser.add_argument("--weight-peak-location", type=float, default=0.1)
    parser.add_argument("--pedestal-point-boost", type=float, default=4.0)
    parser.add_argument("--pedestal-sample-boost", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    data_root=args.data_root.resolve(); output_dir=args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    _, feature_names=all_feature_indices(data_root)
    training=TrainingConfig(
        epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
        weight_value=args.weight_value,weight_gradient=args.weight_gradient,
        weight_peak_magnitude=args.weight_peak_magnitude,
        weight_peak_location=args.weight_peak_location,
        pedestal_point_boost=args.pedestal_point_boost,
        pedestal_sample_boost=args.pedestal_sample_boost,seed=args.seed,
    )
    device=choose_device(args.device)
    targets=SPECIES if args.target=="both" else (args.target,)
    checkpoints=[train_species(sp,data_root,output_dir,feature_names,training,device,args.init_checkpoint_dir.resolve() if args.init_checkpoint_dir else None) for sp in targets]
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "reconstruction_scope": RECONSTRUCTION_SCOPE,
                "design": "v11_pedestal_aware_physical_slope",
                "data_root": str(data_root),
                "device": str(device),
                "feature_names": feature_names,
                "training": asdict(training),
                "checkpoints": [str(path) for path in checkpoints],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"v12 pedestal-aware outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
