# v11 full-profile reconstruction

v11 reconstructs the complete physical-fit profiles instead of the restricted
mtanh profiles used by v10:

| Version | NPZ target | Points per species | Rho range |
|---|---|---:|---:|
| v10 | `Y_train` | 100 | 0.7--1.25 |
| v11 | `Y_fit` | 201 | 0--1.25 |

The existing v7-weighted preprocessing output already contains everything v11
needs.  Each split's `samples_train.npz` must contain `X_train`, `Y_fit`, and
`T_train`; `standardization_scalars.npz` must contain `y_mean_fit` and
`y_std_fit`.  No new mtanh reconstruction or raw-data preprocessing is done by
v11.

Train both species with all saved input features:

```bash
uv run python -m lh_transitions.profile_recon_robust_sobolev_full_profile_v11 \
  --data-root intergral_v7_weighted \
  --output-dir v11_outputs \
  --target both \
  --epochs 100 \
  --batch-size 128 \
  --device auto
```

The output directory contains:

- `best_resmlp_v11_te_full_profile.pth`
- `best_resmlp_v11_ne_full_profile.pth`
- one training-history CSV per species
- `run_config.json`

Every checkpoint records `target_key=Y_fit`, `grid_points=201`, and
`rho_min=0.0`, `rho_max=1.25` so that it cannot silently be mistaken for a v10
mtanh checkpoint.

Run both checkpoints on labeled test samples:

```bash
uv run python -m lh_transitions.predict_full_profile_v11 \
  --data-root intergral_v7_weighted \
  --checkpoint-dir v11_outputs \
  --output-dir v11_predictions \
  --split test \
  --input-kind labeled
```

Use `--input-kind infer` to reconstruct the split's 1 ms inference grid.  The
saved NPZ contains a `(samples, 402)` physical-unit `prediction` matrix (201 Te
columns followed by 201 Ne columns), the shared 201-point `rho` vector and the
sample times.  For labeled input it also contains the denormalized `target`.

Visualize and evaluate the complete profiles on a labeled split:

```bash
uv run python -m lh_transitions.visualize_full_profile_v11 \
  --data-root intergral_v7_weighted \
  --checkpoint-dir v11_outputs \
  --output-dir v11_visualizations \
  --split test \
  --device auto
```

This produces sample-level and rho-level metric CSVs, a JSON summary, metric
distributions, radial RMSE/bias curves, representative complete-profile
examples, and target-versus-prediction population profiles.  The radial axis is
always the v11 range `rho=0--1.25`.
