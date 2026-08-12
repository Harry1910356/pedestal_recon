# TCV Profile Reconstruction: v7 and v10

This repository contains the source code, small data-contract files, and v7 checkpoints for reconstructing TCV electron-temperature (Te) and electron-density (Ne) profiles. Large processed arrays and raw source data are intentionally excluded from Git. See [EXCLUDED_FILES.md](EXCLUDED_FILES.md) for their exact sizes and expected locations.

## Model versions

| Version | Network and loss | Input features |
|---|---|---|
| `v7_weighted` | Four-block ResMLP with a core-weighted Huber-Sobolev loss | 69 of the 112 stored columns, selected by the union of Te/Ne random-forest and permutation ranks at threshold 50 |
| `v10` | The same dataset contract, ResMLP, and loss as v7 | All 112 stored columns in their saved order; no feature-importance ranking is read |

v10 imports the shared v7 dataset, network, and loss definitions so that its only intended experimental difference is feature selection.

## Data flow

```text
TCV_required_features_integrated/TCV_DATAno<shot>build.parquet
                                |
                                | X: time-dependent scalar features
                                |
TCV_Processed_H5_compare/TCV_zhang_<shot>.h5
                                |
                                | time + fitted mtanh parameters
                                | -> 100-point Te and Ne model targets
                                |
TCV_Processed_H5_compare/TCV_zhang_<shot>_raw.h5
                                |
                                | 201-point fitted reference profiles
                                | + raw Thomson-scattering observations
                                v
                lh_transitions/raw_ped_v7_weighted.py
                                |
                                v
                    intergral_v7_weighted/
              train, validation, and test NPZ datasets
                                |
                   +------------+------------+
                   |                         |
                   v                         v
            v7: ranked 69 features     v10: all 112 features
```

## Source data formats

### Integrated feature parquet files

Source directory in the original workspace:

```text
/Users/peizhengzhang/Desktop/LH_tran_24_Apr/LH_transitions/TCV_required_features_integrated
```

This directory is not included in the repository. At packaging time it contained 3,946 per-shot files totaling approximately 2.4 GiB. The expected naming convention is:

```text
TCV_DATAno<shot>build.parquet
```

Each parquet file is the X-side time series for one TCV shot. Sampled files contain 32 columns: `time` plus 31 numeric plasma, geometry, heating, gas, and diagnostic features:

```text
KAPPA, AREA, PD, Q95, LI, BZERO, SXRcore, BETAN, BETAP, POHM,
ECRH, P_baratron_div, P_baratron_mid, DELTA_TOP, DELTA_BOTTOM,
TAU_conf, TAU_conf_calc, IPLA, OddN, EvenN, NEavg, Ne_edge_avg,
Ne_core_avg, Wtot, Wth, NBI, NBI2, GASmeas_D2, GASmeas_N2,
GASmeas_Ne, GASmeas_Ar
```

The preprocessing code uses these files as follows:

1. It extracts the shot number from the filename.
2. It reads `time` in seconds, keeps only the shot-specific interval from `flat_top_times.csv`, converts time to milliseconds, and sorts the rows.
3. It converts `BZERO` to its absolute value and retains `IPLA` as a model feature.
4. The first usable shot defines the raw feature order. Missing columns in later shots are inserted as NaN so every shot has a consistent schema.
5. Invalid `TAU_conf` values are interpolated where possible.
6. The X time series is interpolated to the HDF5 target times for supervised samples and to a 1 ms grid for inference samples.
7. First and second time differences and moving-average features are added. NBI and NBI2 are excluded from derivative calculation; all continuous columns receive moving averages.
8. `NBI`, `NBI2`, `GASmeas_D2`, and `GASmeas_N2` are treated as physically zero when absent.
9. Columns that are entirely NaN in training and rows with unresolved NaNs are removed. Magnitude scaling and Z-score statistics are computed from the training split only.

After cleaning and feature engineering, the packaged `feature_names.json` files describe 112 X columns. v7 selects 69 of them; v10 uses all 112.

### Processed parameter HDF5 files

Example:

```text
TCV_Processed_H5_compare/TCV_zhang_55777.h5
```

These are compact, time-indexed fit-result files. Typical datasets include:

```text
time, shot
Te/{alpha,offset,ped_height,ped_width,ped_pos,boundary_rho,
    param_err,poly,poly_err,r2,rmse}
Ne/{alpha,offset,ped_height,ped_width,ped_pos,boundary_rho,
    param_err,poly,poly_err,r2,rmse}
```

`raw_ped_v7_weighted.py` reads `time` and the five mtanh parameters (`alpha`, `offset`, `ped_height`, `ped_width`, and `ped_pos`) for each species. It evaluates the mtanh expression on 100 radial points over `rho = 0.7–1.25` and concatenates Te and Ne into the 200-column normalized training target `Y_train`.

The main v7 target therefore comes from the fitted parameters in the direct `.h5` file, not from `_raw.h5`.

### Raw/reference HDF5 files

Example:

```text
TCV_Processed_H5_compare/TCV_zhang_55777_raw.h5
```

These larger companion files contain flattened fitted profiles and the underlying Thomson-scattering observations:

```text
shot
Te/{fit,counts_fit,raw_rho,raw_profile,raw_error_bar,counts}
Ne/{fit,counts_fit,raw_rho,raw_profile,raw_error_bar,counts}
```

During v7 dataset construction, `fit` and `counts_fit` are used to reconstruct a 201-point physical reference profile over `rho = 0–1.25` for each species. Te and Ne are concatenated into the 402-column `Y_fit` array. `Y_fit` is saved with each labeled split and is used for quality checks, reference curves, and later analysis; it is not the direct 200-column prediction target of the v7/v10 ResMLP.

The `raw_rho`, `raw_profile`, `raw_error_bar`, and `counts` fields are not used to construct `Y_train`. They are used by the optional post-training plotting routine in `profile_recon_robust_sobolev_mtanh_v7_weighted.py` to overlay raw Thomson-scattering points and error bars on reconstructed profiles.

### Are both HDF5 files used by v7?

Yes. The v7 loader scans direct `.h5` files and requires a matching `_raw.h5` file for the same shot. A shot is skipped if either member of the pair is unavailable.

At inspection time, the source directory contained:

| Item | Count |
|---|---:|
| Direct `.h5` files | 3,760 |
| `_raw.h5` files | 3,760 |
| Complete shot pairs | 3,760 |
| Unpaired files | 0 |
| Direct-file time slices | 314,881 |
| Slices with both Te and Ne `counts_fit == 201` | 280,922 |

All 3,760 pairs had the fields and array relationships required by the v7 loader. This does not mean that every pair or time slice appears in the final dataset. A sample is retained only if the shot also has a matching feature parquet file and the slice passes the flat-top, finite-value, physical-range, TS/core-density consistency, and mismatch-list filters.

The roles are complementary:

| Source | Primary v7 role |
|---|---|
| Direct `.h5` | Time axis and fitted mtanh parameters used to generate the 100-point Te and Ne model targets |
| `_raw.h5` | 201-point reference fits used for `Y_fit`; raw TS points used by optional diagnostic plots |
| Integrated parquet | Time-dependent X features used as model input |

## Repository contents

```text
.
├── README.md
├── EXCLUDED_FILES.md
├── pyproject.toml
├── compare_mtanh_models.py
├── flat_top_times.csv
├── mismatched_slices.csv
├── best_resmlp_robust_model_te_mtanh_v7_weighted.pth
├── best_resmlp_robust_model_ne_mtanh_v7_weighted.pth
├── intergral_v7_weighted/
│   ├── standardization_scalars.npz
│   ├── train/{feature_names.json,metadata.parquet}
│   ├── val/{feature_names.json,metadata.parquet}
│   └── test/{feature_names.json,metadata.parquet}
├── lh_transitions/
│   ├── raw_ped_v7_weighted.py
│   ├── profile_recon_robust_sobolev_mtanh_v7_weighted.py
│   ├── visualize_resmlp_v7.py
│   ├── profile_recon_robust_sobolev_mtanh_v10.py
│   ├── visualize_resmlp_v10.py
│   └── stats_output/feature_importance_ranking_{te,ne}.csv
└── tests/
```

## Installation

Python 3.11 is required. With uv:

```bash
uv sync --dev
uv run pytest -q
```

Alternatively:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Restoring the excluded processed arrays

Training or dataset-backed evaluation requires the relevant NPZ files to be downloaded from external storage and restored to these paths:

```text
intergral_v7_weighted/train/samples_train.npz
intergral_v7_weighted/train/samples_infer.npz
intergral_v7_weighted/val/samples_train.npz
intergral_v7_weighted/val/samples_infer.npz
intergral_v7_weighted/test/samples_train.npz
intergral_v7_weighted/test/samples_infer.npz
```

Only `test/samples_train.npz` is needed for labeled test evaluation. The inference NPZ files are not read by the current evaluation scripts.

## Evaluating v7

The repository includes the existing v7 Te and Ne checkpoints. After restoring `test/samples_train.npz`, run:

```bash
uv run python lh_transitions/visualize_resmlp_v7.py \
  --root . \
  --data-root intergral_v7_weighted \
  --split test \
  --output-dir v7_weighted_outputs \
  --device auto \
  --rank-threshold 50
```

The two ranking CSV files are required to reproduce the 69 input columns expected by the existing v7 checkpoints. The checkpoints contain network weights but do not contain the selected column indices.

## Training and evaluating v10

The current v10 is the all-feature counterpart of v7 and does not accept a rank threshold. Previous shape-balanced v10 checkpoints are incompatible with this implementation and are intentionally excluded. Retrain v10 with:

```bash
uv run python lh_transitions/profile_recon_robust_sobolev_mtanh_v10.py \
  --data-root intergral_v7_weighted \
  --output-dir v10_outputs \
  --target both \
  --epochs 100 \
  --batch-size 128 \
  --device auto
```

Then evaluate it with:

```bash
uv run python lh_transitions/visualize_resmlp_v10.py \
  --data-root intergral_v7_weighted \
  --checkpoint-dir v10_outputs \
  --output-dir v10_visualizations \
  --split test \
  --device auto
```

## Rebuilding the dataset from source data

Restore both excluded source directories before preprocessing:

```text
TCV_required_features_integrated/
TCV_Processed_H5_compare/
```

Run the parameterized function rather than relying on a machine-specific path:

```bash
uv run python -c "from lh_transitions.raw_ped_v7_weighted import build_reconstruction_dataset; build_reconstruction_dataset(x_folder='TCV_required_features_integrated', h5_folder='TCV_Processed_H5_compare', output_root='intergral_v7_weighted', flat_top_csv='flat_top_times.csv', split_ratios=(0.7, 0.2, 0.1), split_seed=42, target_dt_ms=1.0)"
```

## SSH security

Never commit or share an SSH private key, GitHub token, or password. Pushing requires only a local SSH key whose public key is registered with GitHub and a repository URL such as `git@github.com:account/repository.git`.
