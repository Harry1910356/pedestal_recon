#!/usr/bin/env python3
"""Train v10 as the all-feature counterpart of v7_weighted.

This v10 deliberately repeats the v7 data contract, ResMLP architecture and
core-weighted Huber-Sobolev loss.  Its only model-selection difference is that
all columns in ``feature_names.json`` are used in their saved order.  Feature
importance CSV files and RF/permutation ranks are never read.
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lh_transitions.profile_recon_robust_sobolev_mtanh_v7_weighted import (
    CustomBalancedSobolevLoss,
    PlasmaProfileDataset,
    ProfileResMLPModel,
    ResMLPBlock,
    set_seed,
)


SPECIES = ("Te", "Ne")


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    output_dim: int = 100
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


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def all_feature_indices(data_root: str | Path) -> tuple[list[int], list[str]]:
    """Return every saved feature, preserving the dataset column order."""
    path = Path(data_root) / "train" / "feature_names.json"
    if not path.exists():
        raise FileNotFoundError(f"missing feature definition: {path}")
    names = json.loads(path.read_text(encoding="utf-8")).get("X_features", [])
    if not names:
        raise ValueError(f"{path} does not contain a non-empty X_features list")
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate X feature names")
    return list(range(len(names))), names


def _species_scalars(data_root: Path, species: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(data_root / "standardization_scalars.npz") as scalars:
        y_mean = np.asarray(scalars["y_mean"], dtype=np.float32)
        y_std = np.asarray(scalars["y_std"], dtype=np.float32)
    grid_points = len(y_mean) // 2
    section = slice(0, grid_points) if species == "Te" else slice(grid_points, None)
    return y_mean[section], y_std[section]


def _validate_full_feature_contract(
    data_root: Path, feature_names: list[str], train_dataset: PlasmaProfileDataset
) -> None:
    if train_dataset.X.shape[1] != len(feature_names):
        raise ValueError(
            f"train X has {train_dataset.X.shape[1]} columns but feature_names.json "
            f"declares {len(feature_names)}"
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
    """Train one v7-style model with the complete X matrix."""
    train_dataset = PlasmaProfileDataset(
        str(data_root / "train" / "samples_train.npz"),
        is_train=True,
        target_type=species,
        kept_indices=None,
    )
    val_dataset = PlasmaProfileDataset(
        str(data_root / "val" / "samples_train.npz"),
        is_train=True,
        target_type=species,
        kept_indices=None,
    )
    _validate_full_feature_contract(data_root, feature_names, train_dataset)
    if val_dataset.X.shape[1] != len(feature_names):
        raise ValueError(
            f"val X has {val_dataset.X.shape[1]} columns; expected {len(feature_names)}"
        )

    model_config = ModelConfig(
        input_dim=len(feature_names), output_dim=int(train_dataset.Y.shape[1])
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
        train_dataset, batch_size=training.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=training.batch_size, shuffle=False)

    y_mean, y_std = _species_scalars(data_root, species)
    checkpoint_path = output_dir / f"best_resmlp_v10_{species.lower()}.pth"
    history_path = output_dir / f"training_history_{species.lower()}.csv"
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    print(
        f"[{species}] v10 uses all {model_config.input_dim} features on {device}; "
        "no feature-ranking filter"
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
            train_total += float(loss) * len(features)
        train_loss = train_total / len(train_dataset)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for features, targets, _ in val_loader:
                features, targets = features.to(device), targets.to(device)
                val_total += float(criterion(model(features), targets)) * len(features)
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
                    "version": "v10",
                    "design": "v7_weighted_all_features",
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

    if not history:
        raise RuntimeError("training produced no epochs")
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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "v10_outputs")
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
                "version": "v10",
                "design": "v7_weighted_all_features",
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
    print(f"v10 outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
