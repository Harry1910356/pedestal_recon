"""
reconstruction_pipeline.py
===========================
Reconstruction dataset preprocessing for v7_weighted (based on v5, with per-column Z-score normalization restored).

Core processing rules:
1. Add missing columns, take the absolute value of BZERO, retain IPLA, and strictly remove NaN rows and columns.
2. Load every split into memory before cleaning.
3. Delay zero filling for designated columns until after magnitude scaling, then replace residual NaNs with zero.
4. Targets:
   - Y_mtanh: reconstruct 100 points over rho=0.7--1.25 from _raw.h5 and apply a per-column Z-score.
   - Y_fit: use the 201-point physical profile over rho=0--1.25 and apply a per-column Z-score.
5. Adaptive normalization: monitor training features with extremely small variance.
6. Feature engineering: add first- and second-order time derivatives and moving averages for continuous variables.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.interpolate import interp1d

# =============================================================================
# Constants and grid configuration
# =============================================================================

TIME_COLUMN      = "time"
TARGET_DT_MS     = 1.0

# These columns physically represent zero when absent; zero filling is delayed until after magnitude scaling.
ZERO_FILL_COLS = {"NBI", "NBI2", "GASmeas_D2", "GASmeas_N2"}

# --- Profile grid configuration ---
RHO_POINTS_FIT = 201

# v7_weighted uses the original rho range of 0.7--1.25.
RHO_MIN_MTANH = 0.7
RHO_MAX_MTANH = 1.25
RHO_POINTS_MTANH = 100
RHO_GRID_MTANH = np.linspace(RHO_MIN_MTANH, RHO_MAX_MTANH, RHO_POINTS_MTANH, dtype=np.float64)

# Mismatched-slice filtering configuration
FILTER_MISMATCHED_SLICES = True
MISMATCHED_SLICES_CSV = Path(__file__).resolve().parent.parent / "mismatched_slices.csv"

# Generate target column names dynamically (for example, Te_rho_0 ... Ne_rho_99).
Y_TARGET_COLUMNS: list[str] = []
for _sp in ["Te", "Ne"]:
    Y_TARGET_COLUMNS.extend([f"{_sp}_rho_{i}" for i in range(RHO_POINTS_MTANH)])


# =============================================================================
# Engineered-feature settings (time parameters for first/second derivatives and moving averages)
# =============================================================================
DT_DIFF_MS = 10.0       # Time interval used for derivative calculations, in ms
WINDOW_AVG_MS = 5.0    # Moving-average window, in ms

# Discrete/categorical features excluded from derivative and moving-average calculations
DISCRETE_COLUMNS: list[str] = []


# =============================================================================
# Vectorized engineered-feature calculations for 2D matrices
# =============================================================================

def compute_first_derivative_matrix(t_ms: np.ndarray, x_matrix: np.ndarray, dt: float) -> np.ndarray:
    if len(t_ms) < 2:
        return np.zeros_like(x_matrix)
    interp = interp1d(t_ms, x_matrix, axis=0, kind="linear", bounds_error=False, fill_value=(x_matrix[0], x_matrix[-1]))
    x_prev = interp(t_ms - dt)
    return (x_matrix - x_prev) / dt

def compute_second_derivative_matrix(t_ms: np.ndarray, x_matrix: np.ndarray, dt: float) -> np.ndarray:
    dx = compute_first_derivative_matrix(t_ms, x_matrix, dt)
    d2x = compute_first_derivative_matrix(t_ms, dx, dt)
    return d2x

def compute_moving_average_matrix(t_ms: np.ndarray, x_matrix: np.ndarray, window: float) -> np.ndarray:
    if window <= 0 or len(t_ms) < 2:
        return x_matrix.copy()
    
    n_samples, n_feats = x_matrix.shape
    dt_reg = 0.5
    t_start = t_ms[0] - window
    t_end = t_ms[-1]
    
    num_steps = int(np.ceil((t_end - t_start) / dt_reg)) + 1
    t_reg = np.linspace(t_start, t_end, num_steps)
    
    interp = interp1d(t_ms, x_matrix, axis=0, kind="linear", bounds_error=False, fill_value=(x_matrix[0], x_matrix[-1]))
    x_reg = interp(t_reg)
    
    k = int(round(window / dt_reg))
    if k <= 1:
        return x_matrix.copy()
        
    x_reg_padded = np.vstack([np.zeros((1, n_feats), dtype=x_reg.dtype), x_reg])
    cumsum = np.cumsum(x_reg_padded, axis=0)
    
    main_part = (cumsum[k:] - cumsum[:-k]) / k
    first_part = np.cumsum(x_reg[:k-1], axis=0) / np.arange(1, k)[:, np.newaxis]
    
    rolling_mean_reg = np.vstack([first_part, main_part])
    interp_back = interp1d(t_reg, rolling_mean_reg, axis=0, kind="linear", bounds_error=False, fill_value=(rolling_mean_reg[0], rolling_mean_reg[-1]))
    return interp_back(t_ms)

def compute_derivatives_for_matrix(
    t_ms: np.ndarray, 
    x_matrix: np.ndarray, 
    feature_cols: list[str], 
    discrete_cols: list[str], 
    dt: float
) -> tuple[np.ndarray, list[str]]:
    continuous_indices = [i for i, col in enumerate(feature_cols) if col not in discrete_cols and col not in {"NBI", "NBI2"}]
    if not continuous_indices:
        return np.empty((len(t_ms), 0), dtype=np.float32), []
        
    x_cont = x_matrix[:, continuous_indices]
    diff1 = compute_first_derivative_matrix(t_ms, x_cont, dt)
    diff2 = compute_second_derivative_matrix(t_ms, x_cont, dt)
    
    new_features = []
    new_names = []
    for idx_in_cont, col_idx in enumerate(continuous_indices):
        col_name = feature_cols[col_idx]
        new_features.append(diff1[:, idx_in_cont])
        new_features.append(diff2[:, idx_in_cont])
        new_names.append(f"{col_name}_diff1")
        new_names.append(f"{col_name}_diff2")
        
    if not new_features:
        return np.empty((len(t_ms), 0), dtype=np.float32), []
        
    return np.column_stack(new_features).astype(np.float32), new_names

def compute_averages_for_matrix(
    t_ms: np.ndarray, 
    x_matrix: np.ndarray, 
    feature_cols: list[str], 
    discrete_cols: list[str], 
    window: float
) -> tuple[np.ndarray, list[str]]:
    continuous_indices = [i for i, col in enumerate(feature_cols) if col not in discrete_cols]
    if not continuous_indices:
        return np.empty((len(t_ms), 0), dtype=np.float32), []
        
    x_cont = x_matrix[:, continuous_indices]
    avg_data = compute_moving_average_matrix(t_ms, x_cont, window)
    
    new_features = []
    new_names = []
    for idx_in_cont, col_idx in enumerate(continuous_indices):
        col_name = feature_cols[col_idx]
        new_features.append(avg_data[:, idx_in_cont])
        new_names.append(f"{col_name}_avg")
        
    if not new_features:
        return np.empty((len(t_ms), 0), dtype=np.float32), []
        
    return np.column_stack(new_features).astype(np.float32), new_names


# =============================================================================
# Vectorized mtanh profile reconstruction
# =============================================================================

def mtanh_ped_vectorized(par_matrix: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    alpha      = par_matrix[:, 0][:, np.newaxis]
    offset     = par_matrix[:, 1][:, np.newaxis]
    ped_height = par_matrix[:, 2][:, np.newaxis]
    ped_width  = par_matrix[:, 3][:, np.newaxis]
    ped_pos    = par_matrix[:, 4][:, np.newaxis]

    off = offset
    dped = ped_height - off
    
    safe_width = np.where(np.abs(ped_width) < 1e-6, 1e-6, ped_width)
    wid_2 = safe_width / 2.0
    xsym = ped_pos
    alpha_2 = alpha / 2.0

    z = (xsym - x_grid[np.newaxis, :]) / wid_2
    tanhz = np.tanh(z)
    
    y = (dped / 2.0) * (1.0 + alpha_2 * z) * (1.0 + tanhz) + off
    return y


# =============================================================================
# Y-data loading and reconstruction
# =============================================================================

def load_y_from_h5_folder(
    folder_path: str | Path,
    filter_mismatched: bool = False,
    mismatched_csv: str | Path | None = None
) -> dict[int, dict]:
    pattern   = os.path.join(str(folder_path), "TCV_zhang_*.h5")
    file_list = [f for f in glob.glob(pattern) if not f.endswith("_raw.h5")]

    if not file_list:
        print(f"[HDF5] Warning: no parameter .h5 files found in {folder_path}.")
        return {}

    mismatched_set = set()
    if filter_mismatched and mismatched_csv is not None:
        mismatched_csv = Path(mismatched_csv)
        if mismatched_csv.exists():
            try:
                mismatch_df = pl.read_csv(mismatched_csv)
                for row in mismatch_df.select(["shot", "time_s"]).iter_rows():
                    mismatched_set.add((int(row[0]), round(float(row[1]), 6)))
                print(f"[HDF5] Loaded {len(mismatched_set)} mismatched-slice records for filtering.")
            except Exception as e:
                print(f"[HDF5] Warning: failed to load mismatched-slice CSV: {e}")
        else:
            print(f"[HDF5] Warning: mismatched-slice CSV does not exist: {mismatched_csv}")

    all_data: dict[int, dict] = {}
    print(f"[HDF5] Found {len(file_list)} parameter files; matching them to _raw.h5 files and extracting profiles...")

    for file_path in file_list:
        try:
            file_name = os.path.basename(file_path)
            m = re.search(r"TCV_zhang_(\d+)\.h5", file_name)
            if not m:
                continue
            shot_num = int(m.group(1))

            raw_path = os.path.join(str(folder_path), f"TCV_zhang_{shot_num}_raw.h5")
            if not os.path.exists(raw_path):
                continue

            # Read time values and mtanh parameters.
            with h5py.File(file_path, "r") as f:
                time_array = np.squeeze(f["time"][()])
                if time_array.ndim == 0 and time_array.size == 1:
                    time_array = np.array([time_array.item()])
                
                if time_array.size == 0:
                    continue
                num_t = len(time_array)

                # Read Te/Ne mtanh parameters.
                # Te
                te_alpha = np.atleast_1d(np.squeeze(f["Te/alpha"][()]))
                te_offset = np.atleast_1d(np.squeeze(f["Te/offset"][()]))
                te_ped_height = np.atleast_1d(np.squeeze(f["Te/ped_height"][()]))
                te_ped_width = np.atleast_1d(np.squeeze(f["Te/ped_width"][()]))
                te_ped_pos = np.atleast_1d(np.squeeze(f["Te/ped_pos"][()]))

                # Ne
                ne_alpha = np.atleast_1d(np.squeeze(f["Ne/alpha"][()]))
                ne_offset = np.atleast_1d(np.squeeze(f["Ne/offset"][()]))
                ne_ped_height = np.atleast_1d(np.squeeze(f["Ne/ped_height"][()]))
                ne_ped_width = np.atleast_1d(np.squeeze(f["Ne/ped_width"][()]))
                ne_ped_pos = np.atleast_1d(np.squeeze(f["Ne/ped_pos"][()]))

            te_par_matrix = np.column_stack([te_alpha, te_offset, te_ped_height, te_ped_width, te_ped_pos])
            ne_par_matrix = np.column_stack([ne_alpha, ne_offset, ne_ped_height, ne_ped_width, ne_ped_pos])

            te_mtanh_curves = mtanh_ped_vectorized(te_par_matrix, RHO_GRID_MTANH)
            ne_mtanh_curves = mtanh_ped_vectorized(ne_par_matrix, RHO_GRID_MTANH)

            # Read the 201-point fitted values from _raw.h5.
            with h5py.File(raw_path, "r") as f_raw:
                te_fit = np.squeeze(f_raw["Te/fit"][()]) if "Te/fit" in f_raw else None
                te_counts = np.squeeze(f_raw["Te/counts_fit"][()]) if "Te/counts_fit" in f_raw else None
                ne_fit = np.squeeze(f_raw["Ne/fit"][()]) if "Ne/fit" in f_raw else None
                ne_counts = np.squeeze(f_raw["Ne/counts_fit"][()]) if "Ne/counts_fit" in f_raw else None

            if te_fit is None or te_counts is None or ne_fit is None or ne_counts is None:
                continue

            # Reconstruct the physical Te profile (201 points).
            te_curves = np.full((num_t, RHO_POINTS_FIT), np.nan, dtype=np.float32)
            cur_idx = 0
            for t_idx in range(num_t):
                cnt = int(te_counts[t_idx])
                if cnt == RHO_POINTS_FIT:
                    te_curves[t_idx] = te_fit[cur_idx : cur_idx + RHO_POINTS_FIT]
                    cur_idx += RHO_POINTS_FIT
                elif cnt == 0:
                    pass
                else:
                    te_curves[t_idx] = np.nan
                    cur_idx += cnt

            # Reconstruct the physical Ne profile (201 points).
            ne_curves = np.full((num_t, RHO_POINTS_FIT), np.nan, dtype=np.float32)
            cur_idx = 0
            for t_idx in range(num_t):
                cnt = int(ne_counts[t_idx])
                if cnt == RHO_POINTS_FIT:
                    ne_curves[t_idx] = ne_fit[cur_idx : cur_idx + RHO_POINTS_FIT]
                    cur_idx += RHO_POINTS_FIT
                elif cnt == 0:
                    pass
                else:
                    ne_curves[t_idx] = np.nan
                    cur_idx += cnt

            # Filter mismatched slices.
            if filter_mismatched and len(mismatched_set) > 0:
                for t_idx in range(num_t):
                    t_val = time_array[t_idx]
                    if (shot_num, round(t_val, 6)) in mismatched_set:
                        te_curves[t_idx] = np.nan
                        ne_curves[t_idx] = np.nan
                        te_mtanh_curves[t_idx] = np.nan
                        ne_mtanh_curves[t_idx] = np.nan

            # Unit handling
            # HDF5 stores Te in keV, so retaining Te_keV is equivalent to division by 1000 eV (1 keV).
            te_curves = te_curves * 1.0
            te_mtanh_curves = te_mtanh_curves * 1.0
            # Keep Ne unchanged.
            ne_curves = ne_curves * 1.0
            ne_mtanh_curves = ne_mtanh_curves * 1.0

            # Concatenate horizontally.
            y_mtanh_matrix = np.hstack([te_mtanh_curves, ne_mtanh_curves]) # (num_t, 200)
            y_fit_matrix = np.hstack([te_curves, ne_curves])             # (num_t, 402)

            all_data[shot_num] = {
                "time": time_array,
                "Y_mtanh": y_mtanh_matrix,
                "Y_fit": y_fit_matrix
            }

        except Exception as exc:
            print(f"[HDF5] Error reading/parsing profile data for shot {shot_num}: {exc}")

    return all_data


# =============================================================================
# X-data loading and processing
# =============================================================================

def load_flat_top_times(csv_path: str | Path) -> dict[int, tuple[float, float]]:
    df = pl.read_csv(csv_path)
    flat_top_dict = {}
    for row in df.iter_rows(named=True):
        shot = int(row["shot"])
        start = float(row["flat_top_start"])
        end = float(row["flat_top_end"])
        flat_top_dict[shot] = (start, end)
    return flat_top_dict

def load_x_parquet_folder(
    x_folder: str | Path, shot_glob: str = "TCV_DATAno*build.parquet"
) -> dict[int, Path]:
    pattern   = os.path.join(str(x_folder), shot_glob)
    file_list = glob.glob(pattern)
    shot_paths: dict[int, Path] = {}
    for fp in file_list:
        stem = Path(fp).stem
        m = re.search(r"\d+", stem)
        if m:
            shot_paths[int(m.group())] = Path(fp)
    return shot_paths

def load_x_shot(
    path: Path,
    flat_top_dict: dict[int, tuple[float, float]],
    expected_features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pl.read_parquet(path)

    m = re.search(r"\d+", path.stem)
    if not m:
        print(f"[load_x_shot] Warning: Cannot extract shot ID from path: {path}. Skipping.")
        return np.array([]), np.empty((0, 0)), []
    shot_id = int(m.group())

    if shot_id not in flat_top_dict:
        print(f"[load_x_shot] Warning: Shot {shot_id} not found in flat top times dict. Skipping.")
        return np.array([]), np.empty((0, 0)), []

    start_t, end_t = flat_top_dict[shot_id]
    if start_t == end_t:
        return np.array([]), np.empty((0, 0)), []

    df = df.filter((pl.col(TIME_COLUMN) >= start_t) & (pl.col(TIME_COLUMN) <= end_t))

    if "BZERO" in df.columns:
        df = df.with_columns(pl.col("BZERO").abs())

    df = df.with_columns((pl.col(TIME_COLUMN) * 1000.0).alias("t_ms")).sort("t_ms")
    
    if df.height < 2:
        return np.array([]), np.empty((0, 0)), []

    if expected_features is not None:
        missing_cols = [c for c in expected_features if c not in df.columns]
        if missing_cols:
            df = df.with_columns([pl.lit(np.nan).alias(c) for c in missing_cols])
        feature_cols = expected_features
    else:
        meta_cols = {"t_ms", TIME_COLUMN} 
        feature_cols = [
            c for c, dtype in zip(df.columns, df.dtypes)
            if c not in meta_cols and dtype.is_numeric()
        ]

    t_ms     = df["t_ms"].to_numpy().astype(np.float64)
    x_matrix = df.select(feature_cols).to_numpy().astype(np.float32)

    return t_ms, x_matrix, feature_cols


def assign_splits(shot_ids: list[int], ratios: tuple[float, float, float] = (0.7, 0.2, 0.1), seed: int = 42) -> dict[int, str]:
    nbi2_active = {76084, 77409, 77411, 79815, 79825, 79826, 79827}
    
    nbi2_group = [s for s in shot_ids if s in nbi2_active]
    other_group = [s for s in shot_ids if s not in nbi2_active]
    
    split_map = {}
    
    def split_group(ids, seed_offset):
        ids = sorted(ids)
        rng = random.Random(seed + seed_offset)
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            return {}
        
        total = sum(ratios)
        r_train, r_val = ratios[0]/total, ratios[1]/total
        
        n_train = max(1, round(n * r_train)) if n >= 3 else round(n * r_train)
        n_val   = max(1, round(n * r_val)) if n >= 3 else round(n * r_val)
        
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        
        g_map = {}
        for i, sid in enumerate(ids):
            if i < n_train:
                g_map[sid] = "train"
            elif i < n_train + n_val:
                g_map[sid] = "val"
            else:
                g_map[sid] = "test"
        return g_map
        
    split_map.update(split_group(nbi2_group, 1))
    split_map.update(split_group(other_group, 2))
    return split_map

def _resample_x(t_orig: np.ndarray, x_orig: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    n_feat = x_orig.shape[1]
    resampled = np.zeros((len(t_target), n_feat), dtype=np.float32)

    for col_idx in range(n_feat):
        valid_mask = ~np.isnan(x_orig[:, col_idx])
        if not np.any(valid_mask):
            resampled[:, col_idx] = np.nan
            continue
            
        left_fill  = x_orig[valid_mask, col_idx][0]
        right_fill = x_orig[valid_mask, col_idx][-1]

        interp = interp1d(
            t_orig, x_orig[:, col_idx],
            kind="linear",
            bounds_error=False,
            fill_value=(left_fill, right_fill),
        )
        resampled[:, col_idx] = interp(t_target)

    return resampled


# =============================================================================
# Main preprocessing function
# =============================================================================

def build_reconstruction_dataset(
    x_folder:           str | Path,
    h5_folder:          str | Path,
    output_root:        str | Path,
    split_ratios:       tuple[float, float, float] = (0.7, 0.2, 0.1),
    split_seed:         int   = 42,
    target_dt_ms:       float = TARGET_DT_MS,
    shot_glob:          str   = "TCV_DATAno*build.parquet",
    flat_top_csv:       str | Path | None = None,
) -> dict[str, Path]:

    x_folder, h5_folder, output_root = Path(x_folder), Path(h5_folder), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load flat_top_times.csv.
    if flat_top_csv is None:
        sibling_path = Path(__file__).resolve().parent.parent / "flat_top_times.csv"
        if sibling_path.exists():
            flat_top_csv = sibling_path
        elif Path("flat_top_times.csv").exists():
            flat_top_csv = Path("flat_top_times.csv")
        else:
            raise FileNotFoundError("flat_top_times.csv not found in typical locations.")
    else:
        flat_top_csv = Path(flat_top_csv)

    print(f"\n[pipeline] Loading flat-top time-window configuration: {flat_top_csv}")
    flat_top_dict = load_flat_top_times(flat_top_csv)

    print("\n===== Step 1 / 3: Load Y data and reconstruct mtanh and 201-point physical profiles =====")
    y_dict = load_y_from_h5_folder(
        h5_folder,
        filter_mismatched=FILTER_MISMATCHED_SLICES,
        mismatched_csv=MISMATCHED_SLICES_CSV
    )

    print("\n===== Step 2 / 3: Load X data (Parquet) =====")
    x_paths = load_x_parquet_folder(x_folder, shot_glob)

    common_shots = sorted(set(y_dict.keys()) & set(x_paths.keys()))
    if not common_shots:
        raise RuntimeError("[pipeline] X and Y have no shot_id values in common.")
    print(f"[pipeline] X and Y share {len(common_shots)} shots.")

    _, _, raw_feature_cols = load_x_shot(x_paths[common_shots[0]], flat_top_dict)
    
    # Use a synthetic calculation to obtain engineered-feature column names.
    t_fake = np.linspace(0.0, 100.0, 100)
    x_fake = np.zeros((100, len(raw_feature_cols)), dtype=np.float32)
    _, diff_names = compute_derivatives_for_matrix(t_fake, x_fake, raw_feature_cols, DISCRETE_COLUMNS, DT_DIFF_MS)
    _, avg_names = compute_averages_for_matrix(t_fake, x_fake, raw_feature_cols, DISCRETE_COLUMNS, WINDOW_AVG_MS)
    
    global_feature_cols = raw_feature_cols + diff_names + avg_names
    print(f"[pipeline] Raw X feature count: {len(raw_feature_cols)}")
    print(f"[pipeline] X feature count including engineered features: {len(global_feature_cols)}")

    print("\n===== Step 3 / 3: Assign splits and build cleaned samples =====")
    split_map = assign_splits(common_shots, ratios=split_ratios, seed=split_seed)
    outputs: dict[str, Path] = {}

    ordered_splits = ["train", "val", "test"]
    raw_splits = {}

    # ---------------------------------------------------------
    # Stage 1: load all datasets into memory.
    # ---------------------------------------------------------
    for split_name in ordered_splits:
        shot_ids_in_split = [s for s, sp in split_map.items() if sp == split_name]
        if not shot_ids_in_split:
            continue

        all_x_train, all_y_train, all_y_fit_train, all_t_train, all_shot_train = [], [], [], [], []
        all_x_infer, all_t_infer, all_shot_infer = [], [], []

        for shot_id in shot_ids_in_split:
            try:
                t_x, x_mat, _ = load_x_shot(x_paths[shot_id], flat_top_dict, expected_features=raw_feature_cols)
            except Exception as exc:
                continue

            if t_x.size < 2: continue

            # Clean TAU_conf anomalies: fill with time window linear interpolation
            if "TAU_conf" in raw_feature_cols:
                tau_idx = raw_feature_cols.index("TAU_conf")
                tau_vals = x_mat[:, tau_idx]
                is_anomaly = (tau_vals <= 0.0) | (tau_vals > 0.08) | np.isnan(tau_vals)
                if np.any(is_anomaly):
                     cleaned_tau = tau_vals.copy()
                     valid_mask = ~is_anomaly
                     t_valid = t_x[valid_mask]
                     tau_valid = tau_vals[valid_mask]
                     
                     if len(t_valid) >= 2:
                         interp = interp1d(
                             t_valid, tau_valid,
                             kind="linear",
                             bounds_error=False,
                             fill_value=(tau_valid[0], tau_valid[-1])
                         )
                         cleaned_tau[is_anomaly] = interp(t_x[is_anomaly])
                     elif len(t_valid) == 1:
                         cleaned_tau[is_anomaly] = tau_valid[0]
                     else:
                         cleaned_tau[is_anomaly] = 0.025  # TCV default
                     x_mat[:, tau_idx] = cleaned_tau

            shot_y = y_dict.get(shot_id)
            if not shot_y: continue

            y_mat_mtanh = shot_y["Y_mtanh"]
            y_mat_fit   = shot_y["Y_fit"]
            t_y = shot_y["time"] * 1000.0
            if len(t_y) < 2: continue

            t_min, t_max = float(t_x.min()), float(t_x.max())

            valid_y_mask   = (t_y >= t_min) & (t_y <= t_max)
            t_train        = t_y[valid_y_mask]
            y_mat_train_mtanh = y_mat_mtanh[valid_y_mask]
            y_mat_train_fit   = y_mat_fit[valid_y_mask]
            if len(t_train) < 2: continue
            
            # 1. Resample by interpolation.
            x_mat_train = _resample_x(t_x, x_mat, t_train)

            # TS profile sanity check and TS vs Parquet core density check
            keep_mask = np.ones(len(t_train), dtype=bool)
            
            # 1. Sanity check on TS fit profile values: discard if negative or absurdly large
            for t_idx in range(len(t_train)):
                te_mtanh = y_mat_train_mtanh[t_idx, :100]
                ne_mtanh = y_mat_train_mtanh[t_idx, 100:]
                te_fit   = y_mat_train_fit[t_idx, :201]
                ne_fit   = y_mat_train_fit[t_idx, 201:]
                
                # Te in keV units. Te > 15.0 corresponds to > 15 keV
                # Ne in 10^19 m^-3 units. Ne > 15.0 corresponds to > 15e19 m^-3
                if np.any(te_mtanh < 0.0) or np.any(te_mtanh > 15.0):
                    keep_mask[t_idx] = False
                elif np.any(ne_mtanh < 0.0) or np.any(ne_mtanh > 15.0):
                    keep_mask[t_idx] = False
                elif np.any(te_fit < 0.0) or np.any(te_fit > 15.0):
                    keep_mask[t_idx] = False
                elif np.any(ne_fit < 0.0) or np.any(ne_fit > 15.0):
                    keep_mask[t_idx] = False

            # 2. TS vs Parquet core density check: discard time steps where core density differs by 100x
            if "Ne_core_avg" in raw_feature_cols:
                ne_core_idx = raw_feature_cols.index("Ne_core_avg")
                for t_idx in range(len(t_train)):
                    if not keep_mask[t_idx]:
                        continue
                    ts_val = y_mat_train_fit[t_idx, 201]  # Ne fit at rho=0
                    pq_val = x_mat_train[t_idx, ne_core_idx]  # Ne_core_avg
                    
                    if not np.isnan(ts_val) and not np.isnan(pq_val):
                        if ts_val <= 0 or pq_val <= 0:
                            keep_mask[t_idx] = False
                        else:
                            ts_scaled = ts_val * 1e19
                            ratio = ts_scaled / pq_val
                            if ratio > 100.0 or ratio < 0.01:
                                keep_mask[t_idx] = False
                
            t_train = t_train[keep_mask]
            y_mat_train_mtanh = y_mat_train_mtanh[keep_mask]
            y_mat_train_fit = y_mat_train_fit[keep_mask]
            x_mat_train = x_mat_train[keep_mask]
                
            if len(t_train) < 2: continue

            t_infer     = np.arange(t_min, t_max, target_dt_ms, dtype=np.float64)
            x_mat_infer = _resample_x(t_x, x_mat, t_infer)

            # 2. Physically zero-fill ZERO_FILL_COLS.
            for c_idx, c_name in enumerate(raw_feature_cols):
                if c_name in ZERO_FILL_COLS:
                    x_mat_train[np.isnan(x_mat_train[:, c_idx]), c_idx] = 0.0
                    x_mat_infer[np.isnan(x_mat_infer[:, c_idx]), c_idx] = 0.0

            # 3. Calculate first/second derivatives and moving-average features.
            diff_train, _ = compute_derivatives_for_matrix(
                t_train, x_mat_train, raw_feature_cols, DISCRETE_COLUMNS, DT_DIFF_MS
            )
            avg_train, _ = compute_averages_for_matrix(
                t_train, x_mat_train, raw_feature_cols, DISCRETE_COLUMNS, WINDOW_AVG_MS
            )

            diff_infer, _ = compute_derivatives_for_matrix(
                t_infer, x_mat_infer, raw_feature_cols, DISCRETE_COLUMNS, DT_DIFF_MS
            )
            avg_infer, _ = compute_averages_for_matrix(
                t_infer, x_mat_infer, raw_feature_cols, DISCRETE_COLUMNS, WINDOW_AVG_MS
            )

            # 4. Concatenate raw and engineered features.
            x_mat_train_combined = np.hstack([x_mat_train, diff_train, avg_train]).astype(np.float32)
            x_mat_infer_combined = np.hstack([x_mat_infer, diff_infer, avg_infer]).astype(np.float32)

            all_x_train.append(x_mat_train_combined)
            all_y_train.append(y_mat_train_mtanh)
            all_y_fit_train.append(y_mat_train_fit)
            all_t_train.append(t_train.astype(np.float32))
            all_shot_train.append(np.full(len(t_train), shot_id, dtype=np.int32))

            all_x_infer.append(x_mat_infer_combined)
            all_t_infer.append(t_infer.astype(np.float32))
            all_shot_infer.append(np.full(len(t_infer), shot_id, dtype=np.int32))

        if not all_x_train:
            continue

        raw_splits[split_name] = {
            "X": np.concatenate(all_x_train, axis=0),
            "Y": np.concatenate(all_y_train, axis=0),
            "Y_fit": np.concatenate(all_y_fit_train, axis=0),
            "T": np.concatenate(all_t_train, axis=0),
            "S": np.concatenate(all_shot_train, axis=0),
            "X_inf": np.concatenate(all_x_infer, axis=0),
            "T_inf": np.concatenate(all_t_infer, axis=0),
            "S_inf": np.concatenate(all_shot_infer, axis=0),
            "shot_ids": shot_ids_in_split
        }

    if "train" not in raw_splits:
        raise RuntimeError("The training split is empty; statistics cannot be computed.")

    # ---------------------------------------------------------
    # Stage 2: clean all splits consistently by removing all-NaN columns and rows containing NaNs.
    # ---------------------------------------------------------
    nan_col_mask = np.all(np.isnan(raw_splits["train"]["X"]), axis=0)
    valid_cols_mask = ~nan_col_mask
    current_feature_cols = [c for c, m in zip(global_feature_cols, valid_cols_mask) if m]
    
    if np.any(nan_col_mask):
        dropped_cols = [c for c, m in zip(global_feature_cols, valid_cols_mask) if not m]
        print(f"\n  [cleaning] Removed {np.sum(nan_col_mask)} all-NaN feature columns.")

    # Calculate each valid training feature's typical scale (median absolute non-zero value).
    train_X_valid = raw_splits["train"]["X"][:, valid_cols_mask]
    global_scales = []
    for i in range(train_X_valid.shape[1]):
        col = train_X_valid[:, i]
        valid = col[(~np.isnan(col)) & (col != 0.0)]
        if len(valid) == 0:
            global_scales.append(1.0)
        else:
            global_scales.append(float(np.nanmedian(np.abs(valid))))
    global_scales = np.array(global_scales)

    cleaned_splits = {}
    for split_name, data in raw_splits.items():
        X = data["X"][:, valid_cols_mask]
        Y = data["Y"]
        Y_fit = data["Y_fit"]
        T = data["T"]
        S = data["S"]
        
        X_inf = data["X_inf"][:, valid_cols_mask]
        T_inf = data["T_inf"]
        S_inf = data["S_inf"]
        
        original_len = len(X)
        zero_fill_indices = [i for i, c in enumerate(current_feature_cols) if c in ZERO_FILL_COLS]
        check_col_mask = np.ones(X.shape[1], dtype=bool)
        if zero_fill_indices:
            check_col_mask[zero_fill_indices] = False
        
        # 100x Outlier check: ONLY applied to Ne-related features
        valid_row = np.ones(len(X), dtype=bool)
        for i in range(X.shape[1]):
            col_name = current_feature_cols[i]
            if "ne" in col_name.lower():
                scale = global_scales[i]
                if scale > 1e-10:
                    if "diff" in col_name.lower():
                        # Derivatives can naturally be negative or near zero
                        outlier_rows = np.abs(X[:, i]) > 100.0 * scale
                    else:
                        outlier_rows = (np.abs(X[:, i]) > 100.0 * scale) | (np.abs(X[:, i]) < 0.01 * scale) | (X[:, i] <= 0)
                    valid_row &= ~outlier_rows
                
        # Also check inf (inference split) for outliers
        valid_row_inf = np.ones(len(X_inf), dtype=bool)
        for i in range(X_inf.shape[1]):
            col_name = current_feature_cols[i]
            if "ne" in col_name.lower():
                scale = global_scales[i]
                if scale > 1e-10:
                    if "diff" in col_name.lower():
                        outlier_rows_inf = np.abs(X_inf[:, i]) > 100.0 * scale
                    else:
                        outlier_rows_inf = (np.abs(X_inf[:, i]) > 100.0 * scale) | (np.abs(X_inf[:, i]) < 0.01 * scale) | (X_inf[:, i] <= 0)
                    valid_row_inf &= ~outlier_rows_inf

        # Strictly discard rows containing NaNs in X, Y_mtanh, or Y_fit.
        valid_row &= ~np.isnan(X[:, check_col_mask]).any(axis=1) & ~np.isnan(Y).any(axis=1) & ~np.isnan(Y_fit).any(axis=1)
        valid_row_inf &= ~np.isnan(X_inf[:, check_col_mask]).any(axis=1)
        
        cleaned_splits[split_name] = {
            "X": X[valid_row], "Y": Y[valid_row], "Y_fit": Y_fit[valid_row], "T": T[valid_row], "S": S[valid_row],
            "X_inf": X_inf[valid_row_inf], "T_inf": T_inf[valid_row_inf], "S_inf": S_inf[valid_row_inf],
            "shot_ids": data["shot_ids"]
        }
        print(f"  [cleaning] {split_name}: removed slices with NaNs or 100x outliers; samples {original_len} -> {len(cleaned_splits[split_name]['X'])}")

    # ---------------------------------------------------------
    # Stage 3: magnitude-scale every column, fill zeros, and determine the normalization strategy.
    # ---------------------------------------------------------
    print("\n--- Calculating column-wise magnitude scaling and normalization strategy ---")
    train_X = cleaned_splits["train"]["X"].astype(np.float64)
    train_Y = cleaned_splits["train"]["Y"].astype(np.float64)
    train_Y_fit = cleaned_splits["train"]["Y_fit"].astype(np.float64)

    x_magnitude = np.ones(train_X.shape[1], dtype=np.float64)
    zero_fill_indices = [i for i, c in enumerate(current_feature_cols) if c in ZERO_FILL_COLS]

    for i in range(train_X.shape[1]):
        col = train_X[:, i]
        valid = col[(~np.isnan(col)) & (col != 0.0)]
        if len(valid) == 0:
            print(f"  [magnitude] '{current_feature_cols[i]}' is entirely NaN/0 in training; using magnitude=1.0 without scaling.")
            continue
        mag = np.mean(np.abs(valid))
        if mag < 1e-8:
            print(f"  [magnitude] '{current_feature_cols[i]}' has a very small mean absolute value ({mag:.2e}); using magnitude=1.0 without scaling.")
            continue
        x_magnitude[i] = mag
        print(f"  [magnitude] '{current_feature_cols[i]}' magnitude={mag:.4e}")

    for split_name, data in cleaned_splits.items():
        data["X"]     = data["X"].astype(np.float64) / x_magnitude
        data["X_inf"] = data["X_inf"].astype(np.float64) / x_magnitude
        for i in zero_fill_indices:
            data["X"][np.isnan(data["X"][:, i]), i]         = 0.0
            data["X_inf"][np.isnan(data["X_inf"][:, i]), i] = 0.0

    train_X = cleaned_splits["train"]["X"]

    x_mean = np.mean(train_X, axis=0)
    x_std  = np.std(train_X, axis=0)

    # Calculate per-column Z-score statistics for the generated physical profiles (Y_mtanh, 200 points).
    y_mean = np.mean(train_Y, axis=0)
    y_std  = np.std(train_Y, axis=0)
    y_std[y_std < 1e-8] = 1.0

    # Calculate per-column Z-score statistics for the physical reference fits (Y_fit, 402 points).
    y_mean_fit = np.mean(train_Y_fit, axis=0)
    y_std_fit  = np.std(train_Y_fit, axis=0)
    y_std_fit[y_std_fit < 1e-8] = 1.0

    skip_norm_mask = np.zeros(train_X.shape[1], dtype=bool)

    for i in range(train_X.shape[1]):
        if x_std[i] < 1e-8:
            global_min = min(np.min(cleaned_splits[s]["X"][:, i]) for s in cleaned_splits)
            global_max = max(np.max(cleaned_splits[s]["X"][:, i]) for s in cleaned_splits)

            if global_min >= -10.0 and global_max <= 10.0:
                skip_norm_mask[i] = True
                print(f"  [strategy] '{current_feature_cols[i]}' is a small constant after scaling [{global_min:.2f}, {global_max:.2f}]; skipping Z-score.")
            else:
                skip_norm_mask[i] = True
                print(f"  [warning] '{current_feature_cols[i]}' remains out of range after scaling [{global_min:.2e}, {global_max:.2e}]; "
                      f"normalization was skipped, so inspect this feature separately.")

    np.savez_compressed(
        output_root / "standardization_scalars.npz",
        x_magnitude=x_magnitude,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        y_mean_fit=y_mean_fit,
        y_std_fit=y_std_fit,
        skip_norm_mask=skip_norm_mask,
        zero_fill_cols=np.array(sorted(ZERO_FILL_COLS)),
    )
    
    # ---------------------------------------------------------
    # Stages 4 and 5: apply Z-scores and write output files.
    # ---------------------------------------------------------
    print("\n--- Applying normalization and generating datasets ---")
    norm_cols = ~skip_norm_mask

    for split_name, data in cleaned_splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        X     = data["X"]
        X_inf = data["X_inf"]
        Y     = data["Y"].astype(np.float64)
        Y_fit = data["Y_fit"].astype(np.float64)
        
        X[:, norm_cols] = (X[:, norm_cols] - x_mean[norm_cols]) / x_std[norm_cols]
        X_inf[:, norm_cols] = (X_inf[:, norm_cols] - x_mean[norm_cols]) / x_std[norm_cols]
        
        # Apply the corresponding Z-score normalization.
        Y = (Y - y_mean) / y_std
        Y_fit = (Y_fit - y_mean_fit) / y_std_fit
        
        X = X.astype(np.float32)
        X_inf = X_inf.astype(np.float32)
        Y = Y.astype(np.float32)
        Y_fit = Y_fit.astype(np.float32)
        
        all_metadata = []
        cur_train_idx = 0
        cur_infer_idx = 0
        T = data["T"]
        S = data["S"]
        T_inf = data["T_inf"]
        S_inf = data["S_inf"]

        for shot_id in data["shot_ids"]:
            n_train = np.sum(S == shot_id)
            n_infer = np.sum(S_inf == shot_id)
            if n_train == 0 and n_infer == 0:
                continue

            t_train_shot = T[S == shot_id]
            t_infer_shot = T_inf[S_inf == shot_id]

            t_min = float(min(t_train_shot.min() if n_train > 0 else float('inf'),
                              t_infer_shot.min() if n_infer > 0 else float('inf')))
            t_max = float(max(t_train_shot.max() if n_train > 0 else float('-inf'),
                              t_infer_shot.max() if n_infer > 0 else float('-inf')))

            all_metadata.append({
                "shot_id": shot_id,
                "train_start_idx": cur_train_idx, "train_seq_len": n_train,
                "infer_start_idx": cur_infer_idx, "infer_seq_len": n_infer,
                "start_time_ms": t_min if (n_train > 0 or n_infer > 0) else 0.0,
                "end_time_ms": t_max if (n_train > 0 or n_infer > 0) else 0.0,
            })
            cur_train_idx += n_train
            cur_infer_idx += n_infer

        np.savez_compressed(split_dir / "samples_train.npz", X_train=X, Y_train=Y, T_train=T, Y_fit=Y_fit)
        np.savez_compressed(split_dir / "samples_infer.npz", X_infer=X_inf, T_infer=T_inf)
        pl.DataFrame(all_metadata).write_parquet(split_dir / "metadata.parquet")

        with (split_dir / "feature_names.json").open("w", encoding="utf-8") as fh:
            json.dump({"X_features": current_feature_cols, "Y_targets": Y_TARGET_COLUMNS}, fh, indent=2, ensure_ascii=False)

        outputs[split_name] = split_dir
        print(f"  [complete] Saved the {split_name} dataset.")

    print("\n===== Preprocessing complete =====")
    return outputs

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    build_reconstruction_dataset(
        x_folder     = project_root / "TCV_required_features_integrated",
        h5_folder    = project_root / "TCV_Processed_H5_compare",
        output_root  = project_root / "intergral_v7_weighted",
        split_ratios = (0.7, 0.2, 0.1),
        split_seed   = 42,
        target_dt_ms = 1.0,
    )
