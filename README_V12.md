# v12 pedestal-aware full-profile reconstruction

> **Scope: full-profile reconstruction.** v12 reconstructs the complete Te and
> Ne profiles on the 201-point `rho=0--1.25` grid. It is not a pedestal-only or
> cropped-profile model; the pedestal-aware objective improves the edge region
> while the network continues to predict every radial point.

v12 fine-tunes the v11 full-profile ResMLP checkpoints with a loss designed for
sharp pedestal drops.  The architecture, 112-feature input contract, 201-point
output grid, and physical `Y_fit` target are unchanged.

The v11 loss inherited a radial weight that decreases from core to edge and
computed first differences in point-wise standardized coordinates.  v12:

- denormalizes the target and prediction before calculating radial slopes;
- finds the steep pedestal intervals from the **true** profile over
  `rho=0.7--1.1`;
- increases value and gradient sensitivity locally around those intervals;
- supervises the magnitude and location of the steepest negative slope; and
- gives slope-selected pedestal samples additional weight.

The default values are the balanced setting selected after a held-out v11/v12
comparison.  They improve slope recovery while keeping full-profile NRMSE near
the v11 baseline.

Train from the existing v11 checkpoints by specifying their directory:

```bash
uv run python -m lh_transitions.profile_recon_pedestal_aware_v12 \
  --data-root intergral_v7_weighted \
  --init-checkpoint-dir v11_outputs \
  --output-dir v12_balanced_outputs \
  --target both \
  --device auto
```

Omit `--init-checkpoint-dir` to train the same full-profile architecture from
scratch.

The balanced pretrained checkpoints are included in `v12_balanced_outputs/`.
They carry the same explicit `reconstruction_scope=full_profile` metadata.

Predict the labeled test split:

```bash
uv run python -m lh_transitions.predict_full_profile_v12 \
  --data-root intergral_v7_weighted \
  --checkpoint-dir v12_balanced_outputs \
  --output-dir v12_balanced_predictions \
  --split test
```

The checked-in artifacts under `v12_comparison/` compare v11 and the
balanced v12 checkpoints on the same top-20% true-slope pedestal population.

Every v12 checkpoint, run configuration, and prediction manifest includes
`reconstruction_scope=full_profile` so downstream tools can distinguish this
model from cropped-profile reconstruction versions.
