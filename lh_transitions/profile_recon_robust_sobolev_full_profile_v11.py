#!/usr/bin/env python3
"""Train v11 ResMLP models for the complete rho=0--1.25 plasma profiles.

Unlike v10, which learns the 100-point mtanh target stored as ``Y_train``,
v11 learns the 402-column physical-fit target stored as ``Y_fit``.  Separate
Te and Ne models each reconstruct all 201 radial points.  The input feature
contract and network architecture otherwise remain the all-feature v10 design.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lh_transitions.profile_recon_robust_sobolev_mtanh_v10 import (
    CustomBalancedSobolevLoss,
    ProfileResMLPModel,
    all_feature_indices,
    choose_device,
    set_seed,
)


SPECIES = ("Te", "Ne")
TARGET_KEY = "Y_fit"
RHO_MIN = 0.0
RHO_MAX = 1.25
EXPECTED_GRID_POINTS = 201


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    output_dim: int = EXPECTED_GRID_POINTS
    hidden_dim: int = 256
    num_blocks: int = 4
    dropout_p: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    weight_grad: float = 20.0
    huber_delta: float = 1.0
    seed: int = 42


class FullProfileDataset(Dataset):
    """Load all X features and one species from the normalized ``Y_fit`` target."""

    def __init__(self, npz_path: str | Path, species: str) -> None:
        path = Path(npz_path)
        if species not in SPECIES:
            raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
        if not path.exists():
            raise FileNotFoundError(f"data file not found: {path}")

        with np.load(path) as archive:
            required = {"X_train", TARGET_KEY, "T_train"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} is missing required arrays: {sorted(missing)}")
            features = np.asarray(archive["X_train"], dtype=np.float32)
            full_target = np.asarray(archive[TARGET_KEY], dtype=np.float32)
            times = np.asarray(archive["T_train"], dtype=np.float32)

        if full_target.ndim != 2 or full_target.shape[1] % 2:
            raise ValueError(
                f"{TARGET_KEY} must contain equally sized Te/Ne sections; "
                f"received shape {full_target.shape}"
            )
        if len(features) != len(full_target) or len(times) != len(full_target):
            raise ValueError("X_train, Y_fit and T_train sample counts do not match")

        self.grid_points = full_target.shape[1] // 2
        section = slice(0, self.grid_points) if species == "Te" else slice(self.grid_points, None)
        self.X = torch.from_numpy(features)
        self.Y = torch.from_numpy(full_target[:, section].copy())
        self.T = torch.from_numpy(times)
        self.species = species

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index], self.T[index]


def species_fit_scalars(data_root: str | Path, species: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the denormalization arrays corresponding to one Y_fit species."""
    if species not in SPECIES:
        raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
    path = Path(data_root) / "standardization_scalars.npz"
    with np.load(path) as scalars:
        missing = {"y_mean_fit", "y_std_fit"}.difference(scalars.files)
        if missing:
            raise KeyError(f"{path} is missing full-profile scalars: {sorted(missing)}")
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
    if train_dataset.X.shape[1] != len(feature_names):
        raise ValueError(
            f"train X has {train_dataset.X.shape[1]} columns but feature_names.json "
            f"declares {len(feature_names)}"
        )
    if val_dataset.X.shape[1] != len(feature_names):
        raise ValueError(f"val X has {val_dataset.X.shape[1]} columns; expected {len(feature_names)}")
    if train_dataset.grid_points != val_dataset.grid_points:
        raise ValueError("train and val Y_fit grids have different sizes")
    if train_dataset.grid_points != EXPECTED_GRID_POINTS:
        raise ValueError(
            f"v11 expects {EXPECTED_GRID_POINTS} full-profile points per species, "
            f"found {train_dataset.grid_points}"
        )
    for split in ("val", "test"):
        path = data_root / split / "feature_names.json"
        if path.exists():
            names = json.loads(path.read_text(encoding="utf-8")).get("X_features", [])
            if names != feature_names:
                raise ValueError(f"{split} feature order differs from train: {path}")


def train_species(
    species: str,
    data_root: Path,
    output_dir: Path,
    feature_names: list[str],
    training: TrainingConfig,
    device: torch.device,
) -> Path:
    train_dataset = FullProfileDataset(data_root / "train" / "samples_train.npz", species)
    val_dataset = FullProfileDataset(data_root / "val" / "samples_train.npz", species)
    validate_data_contract(data_root, feature_names, train_dataset, val_dataset)

    model_config = ModelConfig(
        input_dim=len(feature_names), output_dim=train_dataset.grid_points
    )
    model = ProfileResMLPModel(**asdict(model_config)).to(device)
    criterion = CustomBalancedSobolevLoss(
        weight_grad=training.weight_grad,
        delta=training.huber_delta,
        grid_points=model_config.output_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    train_loader = DataLoader(
        train_dataset, batch_size=training.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=training.batch_size, shuffle=False)

    y_mean, y_std = species_fit_scalars(data_root, species)
    if len(y_mean) != model_config.output_dim:
        raise ValueError("Y_fit scalar width does not match the model output width")
    checkpoint_path = output_dir / f"best_resmlp_v11_{species.lower()}_full_profile.pth"
    history_path = output_dir / f"training_history_{species.lower()}_full_profile.csv"
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(
        f"[{species}] v11 reconstructs {model_config.output_dim} points over "
        f"rho={RHO_MIN:g}--{RHO_MAX:g} using all {model_config.input_dim} features on {device}"
    )
    for epoch in range(1, training.epochs + 1):
        model.train()
        train_total = 0.0
        for features, targets, _ in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
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
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"[{species}] epoch {epoch:03d}/{training.epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} lr={learning_rate:.3g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "version": "v11",
                    "design": "v10_all_features_full_profile",
                    "target_key": TARGET_KEY,
                    "profile_kind": "physical_fit",
                    "rho_min": RHO_MIN,
                    "rho_max": RHO_MAX,
                    "grid_points": model_config.output_dim,
                    "feature_selection": "all",
                    "feature_names": feature_names,
                    "kept_indices": list(range(len(feature_names))),
                    "model_config": asdict(model_config),
                    "training_config": asdict(training),
                    "model_state": model.state_dict(),
                    "species": species,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "best_epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "intergral_v7_weighted"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "v11_outputs")
    parser.add_argument("--target", choices=("both", "Te", "Ne"), default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-grad", type=float, default=20.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    set_seed(args.seed)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, feature_names = all_feature_indices(data_root)
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        weight_grad=args.weight_grad,
        huber_delta=args.huber_delta,
        seed=args.seed,
    )
    device = choose_device(args.device)
    targets = SPECIES if args.target == "both" else (args.target,)
    checkpoints = [
        train_species(species, data_root, output_dir, feature_names, training, device)
        for species in targets
    ]
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "version": "v11",
                "design": "v10_all_features_full_profile",
                "target_key": TARGET_KEY,
                "profile_kind": "physical_fit",
                "rho_range": [RHO_MIN, RHO_MAX],
                "grid_points_per_species": EXPECTED_GRID_POINTS,
                "data_root": str(data_root),
                "device": str(device),
                "feature_selection": "all",
                "input_dim": len(feature_names),
                "feature_names": feature_names,
                "training": asdict(training),
                "checkpoints": [str(path) for path in checkpoints],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"v11 full-profile outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
