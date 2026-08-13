# v7 Raw-Data Preprocessing Reference

This document describes the preprocessing behavior that is **actually implemented** in [`raw_ped_v7_weighted.py`](./raw_ped_v7_weighted.py), with emphasis on:

- which missing values are zero-filled and when;
- how NaNs, invalid profiles, and outliers are handled;
- how `X`, `Y_mtanh`, and `Y_fit` are normalized; and
- how the train/validation/test split avoids statistical leakage.

> This is an implementation reference, not an idealized pipeline specification. In particular, the module-level summary says that zero filling is delayed until after magnitude scaling, but the main pipeline zero-fills the four designated raw features before feature engineering and applies a second fallback fill after scaling. The sequence below follows the executable code.

---

## 1. Inputs and outputs

### 1.1 Inputs

| Data | Source | Main contents |
|---|---|---|
| Model inputs `X` | `TCV_DATAno<shot>build.parquet` | Time-dependent engineering, control, density, heating, gas, and diagnostic features |
| mtanh parameters | `TCV_zhang_<shot>.h5` | Te/Ne `alpha`, `offset`, `ped_height`, `ped_width`, and `ped_pos` |
| Reference profiles | `TCV_zhang_<shot>_raw.h5` | Te/Ne 201-point `fit` arrays and per-time-slice `counts_fit` |
| Flat-top windows | `flat_top_times.csv` | Start and end time retained for each shot |
| Mismatched slices | `mismatched_slices.csv` | `(shot, time_s)` pairs that must be invalidated |

### 1.2 Outputs

The pipeline writes one directory for each of `train`, `val`, and `test`:

| File | Contents |
|---|---|
| `samples_train.npz` | `X_train`, `Y_train`, `T_train`, and `Y_fit` |
| `samples_infer.npz` | `X_infer` and `T_infer` on the regular 1 ms inference grid |
| `metadata.parquet` | Shot identifiers, array offsets, sequence lengths, and time ranges |
| `feature_names.json` | X feature names and the 200 mtanh target names |
| `standardization_scalars.npz` | Magnitudes, means, standard deviations, and normalization masks |

Final numerical arrays are stored as `float32`.

---

## 2. End-to-end sequence

```text
Load X and Y
   |
   +-- retain the shot-specific flat-top interval; take abs(BZERO)
   +-- repair invalid TAU_conf values
   +-- resample X to TS times and to the regular 1 ms inference grid
   +-- apply TS profile-range and TS/Parquet density-consistency checks
   +-- zero-fill four designated raw features
   +-- generate first differences, second differences, and moving averages
   |
Split by shot into train / val / test
   |
   +-- use train to remove all-NaN feature columns
   +-- remove rows with unresolved NaNs, invalid targets, or density outliers
   +-- use train to calculate X magnitudes, means, and standard deviations
   +-- use train to calculate point-wise Y means and standard deviations
   +-- apply the same train-derived statistics to every split
```

Splitting is performed by **shot**, not by randomly shuffling individual time slices. The defaults are `0.7 / 0.2 / 0.1` with seed `42`. A fixed group of NBI2-active shots is split separately to reduce distribution imbalance.

---

## 3. Zero filling

### 3.1 Columns for which missing means physically zero

Only these four raw features interpret a missing value as a physical zero:

```python
ZERO_FILL_COLS = {"NBI", "NBI2", "GASmeas_D2", "GASmeas_N2"}
```

| Feature/data type | NaN handling |
|---|---|
| `NBI` | Fill with physical `0.0` |
| `NBI2` | Fill with physical `0.0` |
| `GASmeas_D2` | Fill with physical `0.0` |
| `GASmeas_N2` | Fill with physical `0.0` |
| Every other X feature | Do not impute; remove the row if a NaN remains after cleaning |
| Y profiles | Never impute; remove the complete time slice if any target point is NaN |

### 3.2 Actual order of zero filling

The main pipeline executes zero filling twice:

1. **Before feature engineering:** after time resampling and profile-quality checks, NaNs in the four raw features are replaced with physical `0.0`.
2. First/second differences and moving averages are calculated from this already-filled raw X matrix.
3. **After magnitude scaling:** the same four raw columns receive a second NaN-to-zero fallback. Under the current sequence, the first fill has normally removed these NaNs already.

Therefore, the current executable behavior is **zero-fill before feature engineering, then apply a fallback fill after magnitude scaling**.

### 3.3 A filled zero is generally not zero in normalized space

The inserted value is a physical zero before normalization. X subsequently undergoes:

```text
x_scaled = x_physical / magnitude
x_norm   = (x_scaled - train_mean) / train_std
```

The model input corresponding to an inserted physical zero is therefore normally:

```text
(0 - train_mean) / train_std
```

It is generally **not zero in normalized space**. This is physical-zero imputation, not mean imputation.

### 3.4 Derived features

- `NBI` and `NBI2` do not receive first- or second-difference features, but they do receive moving averages.
- `GASmeas_D2` and `GASmeas_N2` receive both difference and moving-average features.
- Because zero filling happens first, transitions between missing/zero and nonzero values are treated as real temporal changes by the derived features.

---

## 4. NaNs and invalid data

### 4.1 Missing feature columns

The raw feature schema and order are taken from the first common X/Y shot. If a later shot lacks one of these columns, the loader inserts that column and fills it with NaN.

After every split has been loaded:

- a feature column is considered all-NaN using the **training split only**;
- every train-all-NaN column is removed consistently from train, validation, test, and inference data; and
- a column that is not all-NaN in train is retained even if it is all-NaN in another split, in which case affected rows in that split are removed later.

### 4.2 NaNs during time interpolation

X is interpolated to:

- the actual TS/Y measurement times for supervised samples; and
- a regular grid with default spacing `target_dt_ms=1.0` for inference samples.

Outside the source time range, interpolation uses the first and last valid value of the column. A completely invalid column remains NaN.

Implementation caveat: `_resample_x()` calculates a `valid_mask`, but passes the complete original column, including internal NaNs, to `interp1d`. Internal NaNs may therefore propagate instead of being interpolated across. Any unresolved NaNs are removed by the strict row filter later.

### 4.3 Special handling of `TAU_conf`

The following values are marked invalid:

```text
TAU_conf <= 0
TAU_conf > 0.08 s
TAU_conf is NaN
```

Repair depends on the number of remaining valid points in the shot's flat-top interval:

| Number of valid points | Repair |
|---:|---|
| At least 2 | Linear interpolation over time; use the first/last valid value outside the valid range |
| 1 | Fill every invalid point with the only valid value |
| 0 | Fill every invalid point with the hard-coded TCV default `0.025 s` |

This repair occurs before X resampling and feature engineering.

The `0.08 s` cutoff is a hard-coded empirical cleaning threshold inherited by several preprocessing versions. The repository does not contain a derivation that establishes 80 ms as a universal physical limit. It should therefore be treated as a dataset-specific heuristic rather than a fundamental upper bound. `TAU_conf_calc` does not receive the same threshold-based repair.

### 4.4 How Y profiles become invalid

A Te or Ne profile remains or becomes NaN when:

- `_raw.h5` reports `counts_fit == 0` for the time slice;
- `counts_fit` is not the expected value `201`;
- the `(shot, time)` pair appears in `mismatched_slices.csv`; or
- the upstream HDF5 arrays already contain NaNs.

If any of the 200 `Y_mtanh` points or any of the 402 `Y_fit` points is NaN, the complete time slice is removed. Y is never zero-filled or mean-imputed.

### 4.5 Strict final row requirements

A supervised sample is retained only if:

```text
every non-zero-fill X feature is finite
AND all 200 Y_mtanh points are finite
AND all 402 Y_fit points are finite
AND the profile passes the physical-range check
AND all applicable Ne-related feature checks pass
```

Inference rows do not have Y, so only X completeness and Ne-related outlier checks apply.

### 4.6 Hard-coded range and outlier checks

#### TS profile range

The complete time slice is removed if any Te/Ne mtanh or 201-point fit value satisfies:

```text
value < 0  or  value > 15
```

The code assumes Te is in keV and Ne is in `10^19 m^-3`.

#### TS versus Parquet core density

The Ne fit at `rho=0` is compared with `Ne_core_avg`:

- if either value is nonpositive, remove the time slice;
- after multiplying the TS value by `1e19`, remove the time slice when `TS / Parquet > 10` or `< 0.1`, matching the threshold used to create `mismatched_slices.csv`.

#### Ne-related X features

For every retained training feature, the pipeline calculates the median absolute nonzero value as a typical scale. For feature names containing `ne`:

- difference features are rejected when `abs(x) > 100 * scale`;
- nondifference features are rejected when `abs(x) > 100 * scale`, `abs(x) < 0.01 * scale`, or `x <= 0`.

These scales are calculated from train and applied to all splits.

---

## 5. Normalization

Normalization parameters are derived from the cleaned **training split** and then reused for train, validation, and test.

### 5.1 X: magnitude scaling followed by Z-score normalization

For each retained X feature, the training data's finite, nonzero values define:

```text
magnitude_j = mean(abs(x_train_nonzero_j))
```

If no valid nonzero values exist, or if `magnitude < 1e-8`, the magnitude is set to `1.0`.

First apply magnitude scaling:

```text
x_scaled_j = x_j / magnitude_j
```

Then calculate the scaled training statistics:

```text
mean_j = mean(x_scaled_train_j)
std_j  = std(x_scaled_train_j)       # NumPy default: ddof=0
```

Normal columns receive:

```text
x_norm_j = (x_scaled_j - mean_j) / std_j
```

If the training standard deviation is below `1e-8`, `skip_norm_mask[j]` is set and the Z-score step is skipped. Magnitude scaling still applies. The code does not clip large normalized values.

### 5.2 `Y_mtanh`: point-wise Z-score normalization

The model target contains:

- 100 Te points on `rho=0.7–1.25`;
- 100 Ne points on `rho=0.7–1.25`;
- 200 columns in total.

Each species/radial point is normalized independently using train statistics:

```text
y_norm[:, j] = (y[:, j] - y_mean[j]) / y_std[j]
```

If `y_std[j] < 1e-8`, it is replaced with `1.0`.

### 5.3 `Y_fit`: point-wise Z-score normalization over 402 columns

The reference fit contains:

- 201 Te points on `rho=0–1.25`;
- 201 Ne points on `rho=0–1.25`;
- 402 columns in total.

It uses separate `y_mean_fit` and `y_std_fit` statistics. Each radial point is normalized independently, with standard deviations below `1e-8` replaced by `1.0`.

### 5.4 Inverse transforms

Restore a predicted mtanh profile with:

```text
Y_physical = Y_normalized * y_std + y_mean
```

Restore a reference fit with:

```text
Y_fit_physical = Y_fit_normalized * y_std_fit + y_mean_fit
```

For an X column that received a Z-score:

```text
x_scaled   = x_normalized * x_std + x_mean
x_physical = x_scaled * x_magnitude
```

For `skip_norm_mask=True` columns, skip the first line and multiply by `x_magnitude` only.

---

## 6. Saved standardization parameters

`standardization_scalars.npz` contains:

| Key | Meaning |
|---|---|
| `x_magnitude` | Mean absolute nonzero magnitude of each X column |
| `x_mean` | Training mean after magnitude scaling |
| `x_std` | Training population standard deviation after magnitude scaling |
| `y_mean` / `y_std` | Point-wise statistics for the 200-column mtanh target |
| `y_mean_fit` / `y_std_fit` | Point-wise statistics for the 402-column reference fit |
| `skip_norm_mask` | Near-constant X columns that skip Z-score normalization |
| `zero_fill_cols` | Raw feature names for which missing is interpreted as physically zero |

Inference must use this file and `feature_names.json` from the same data/checkpoint contract. Do not re-estimate these statistics from test or deployment data.

---

## 7. Quick-reference summary

| Question | Current v7 behavior |
|---|---|
| What happens to an ordinary missing X value? | It is not imputed; rows with unresolved NaNs are removed |
| What happens to missing NBI/NBI2/gas values? | Fill with physical zero before feature engineering; apply a fallback fill after scaling |
| What happens to a missing Y value? | Remove the complete time slice |
| What happens to a train-all-NaN feature? | Remove the column consistently from every split |
| How is X normalized? | Train-derived nonzero magnitude scaling, followed by a train-derived Z-score |
| How is Y normalized? | Train-derived Z-score independently at each species/radial point |
| Are validation or test statistics used? | No |
| Is clipping applied? | No |
| Does an inserted physical zero remain zero at model input? | Usually not; after the Z-score it becomes the normalized value corresponding to physical zero |
