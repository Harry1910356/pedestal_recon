# Comparison of the v7, v10, v11, and v12 Models

This document compares the four TCV Te/Ne profile-reconstruction models in this
repository. Each version trains separate electron-temperature (Te) and
electron-density (Ne) regressors. The table below therefore gives the output
width of one species model; concatenating Te and Ne doubles that width.

## At a glance

| Version | Input features | Training target | Output per species | Radial range | Main purpose |
|---|---:|---|---:|---|---|
| v7 weighted | 69 selected features | `Y_train` mtanh profile | 100 | `rho=0.7--1.25` | Compact cropped-profile baseline |
| v10 | All 112 features | `Y_train` mtanh profile | 100 | `rho=0.7--1.25` | Isolate the effect of using every input feature |
| v11 | All 112 features | `Y_fit` physical-fit profile | 201 | `rho=0--1.25` | Reconstruct the complete core-to-edge profile |
| v12 | All 112 features | `Y_fit` physical-fit profile | 201 | `rho=0--1.25` | Preserve full-profile reconstruction while improving pedestal slopes |

The combined output matrices contain 200 columns for v7/v10 and 402 columns
for v11/v12, ordered as all Te points followed by all Ne points.

## Inputs and outputs

All four models use the same preprocessed time-slice dataset. The saved input
contract contains 112 scalar and engineered features, including plasma state,
geometry, heating, gas, first and second temporal differences, and moving
averages.

- **v7 weighted** selects 69 columns using the union of the Te and Ne
  random-forest/permutation rankings at threshold 50. This reduces the input
  dimension but requires the ranking CSV files to reproduce the exact column
  order expected by the supplied checkpoints.
- **v10, v11, and v12** use all 112 columns in the order stored in
  `feature_names.json`.
- **v7 and v10** learn `Y_train`: mtanh curves evaluated at 100 points in the
  edge-oriented interval `rho=0.7--1.25`.
- **v11 and v12** learn `Y_fit`: the complete 201-point physical-fit profiles
  over `rho=0--1.25`. This includes the plasma core and the pedestal/edge.

## Network and objective differences

The basic network is intentionally stable across all versions: an input
projection to 256 hidden units, four residual MLP blocks with GELU activations
and dropout 0.1, and a linear output projection. The output layer has 100 units
for v7/v10 and 201 units for v11/v12.

| Version | Network change | Loss/objective change |
|---|---|---|
| v7 weighted | Four-block ResMLP; 69-dimensional input | Huber value loss plus first-difference loss, with weights decreasing from the inner end of the cropped grid to the edge |
| v10 | Same ResMLP as v7; input projection expands to 112 features | Same core-weighted Huber-Sobolev objective as v7 |
| v11 | Same all-feature ResMLP as v10; output expands from 100 to 201 points | Applies the v7/v10 core-weighted Huber-Sobolev objective to the full-profile `Y_fit` target |
| v12 | Same 112-to-201 ResMLP shape as v11 | Denormalizes profiles before computing physical radial slopes, identifies steep true-pedestal intervals in `rho=0.7--1.1`, locally boosts value and gradient errors, and supervises the magnitude and location of the steepest negative slope |

The balanced v12 checkpoints are fine-tuned from v11. Consequently, the main
v11-to-v12 experiment changes the training objective rather than the model
capacity, input contract, or output grid.

## Strengths and limitations

### v7 weighted

**Strengths**

- Uses fewer inputs, reducing dependence on weakly ranked features.
- Includes ready-to-use checkpoints and provides a simple cropped-profile
  baseline.
- Directly models smooth mtanh targets in the pedestal-oriented radial range.

**Limitations**

- Does not reconstruct the core below `rho=0.7`.
- Requires external ranking files to recover the exact 69-feature contract.
- A smooth mtanh target cannot represent every detail present in the physical
  201-point fits.

### v10

**Strengths**

- Uses all available features and removes the ranking-file dependency.
- Keeps the same target, architecture, and loss as v7, making v7 versus v10 a
  relatively clean feature-selection comparison.

**Limitations**

- Still predicts only the cropped 100-point mtanh profile.
- Additional inputs may add redundancy or noise.
- Checkpoints for the current all-feature implementation are not included and
  must be trained before inference.

### v11

**Strengths**

- Reconstructs all 201 radial points from core to edge.
- Uses the physical-fit target instead of the restricted mtanh curve.
- Provides a direct full-profile baseline with included checkpoints, training
  histories, prediction code, and evaluation tools.

**Limitations**

- Reuses a radial weighting scheme inherited from the cropped model; on the
  full grid this decreases toward the edge and can underemphasize the pedestal.
- Computes first differences in point-wise standardized coordinates rather
  than directly supervising physical slope magnitude and location.
- Sharp pedestal drops can therefore be predicted too smoothly.

### v12

**Strengths**

- Keeps the complete v11 output while explicitly improving pedestal-gradient
  magnitude and location.
- Computes slopes after denormalization and accounts for the physical radial
  spacing.
- On the held-out steep-pedestal population, it substantially improves Ne
  slope recovery and improves Te slope localization relative to v11.

**Limitations**

- The loss is more complex and introduces pedestal-region and weighting
  hyperparameters.
- The balanced checkpoint trades a small amount of Ne full-profile accuracy
  for better Ne pedestal-slope recovery.
- Pedestal selection is target-driven during training, so deployment still
  relies on the model learning this behavior from the input features alone.

## Choosing a version

- Choose **v7** for the supplied compact-input, cropped mtanh baseline.
- Choose **v10** when studying whether all 112 features improve the same
  cropped-profile task.
- Choose **v11** for a straightforward full-profile baseline and the best
  separation between output-scope changes and pedestal-aware loss changes.
- Choose **v12** when full-profile reconstruction is required and recovering
  sharp pedestal gradients is more important than minimizing every global Ne
  error metric.

For commands and checkpoint details, see [README.md](README.md),
[README_V11.md](README_V11.md), and [README_V12.md](README_V12.md). The held-out
v11/v12 pedestal metrics are available in
[`v12_comparison/v11_vs_v12_pedestal_metrics.csv`](v12_comparison/v11_vs_v12_pedestal_metrics.csv).
