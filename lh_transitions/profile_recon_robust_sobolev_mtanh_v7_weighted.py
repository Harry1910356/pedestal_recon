import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import polars as pl
import h5py
import json
from pathlib import Path

# Setup sys.path or use Path to locate feature ranking
workspace_root = Path(__file__).resolve().parents[1]

def set_seed(seed: int = 1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_kept_feature_indices(data_root: str, rank_threshold: int = 50):
    feature_json_path = Path(data_root) / "train" / "feature_names.json"
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Missing feature names json file: {feature_json_path}")
    with open(feature_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    x_features = meta.get("X_features", [])

    te_csv = workspace_root / "lh_transitions" / "stats_output" / "feature_importance_ranking_te.csv"
    ne_csv = workspace_root / "lh_transitions" / "stats_output" / "feature_importance_ranking_ne.csv"

    if not te_csv.exists() or not ne_csv.exists():
        print("[*] Feature ranking CSV files not found. Using all features.")
        return list(range(len(x_features))), x_features

    import pandas as pd
    df_te = pd.read_csv(te_csv)
    df_ne = pd.read_csv(ne_csv)

    te_ranks = df_te.set_index("Feature")
    ne_ranks = df_ne.set_index("Feature")

    kept_indices = []
    kept_names = []
    for idx, name in enumerate(x_features):
        rank_te_rf = te_ranks.loc[name, "RF_Rank"] if name in te_ranks.index else 999
        rank_te_mlp = te_ranks.loc[name, "Permutation_Rank"] if name in te_ranks.index else 999
        rank_ne_rf = ne_ranks.loc[name, "RF_Rank"] if name in ne_ranks.index else 999
        rank_ne_mlp = ne_ranks.loc[name, "Permutation_Rank"] if name in ne_ranks.index else 999

        if (rank_te_rf <= rank_threshold or 
            rank_te_mlp <= rank_threshold or 
            rank_ne_rf <= rank_threshold or 
            rank_ne_mlp <= rank_threshold):
            kept_indices.append(idx)
            kept_names.append(name)

    print(f"[*] Filtered features: rank_threshold={rank_threshold}. Kept {len(kept_indices)}/{len(x_features)} features.")
    return kept_indices, kept_names

# =============================================================================
# 1. PyTorch dataset definition (loads Y_mtanh and Y_fit from v7_weighted)
# =============================================================================
class PlasmaProfileDataset(Dataset):
    def __init__(self, npz_path: str, is_train: bool = True, target_type: str = "both", kept_indices: list = None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Data file not found: {npz_path}. Check the preprocessing path.")
        
        data_dict = np.load(npz_path)
        self.target_type = target_type
        
        if is_train:
            self.X = torch.tensor(data_dict["X_train"], dtype=torch.float32)
            if kept_indices is not None:
                self.X = self.X[:, kept_indices]
            Y_full = data_dict["Y_train"]
            grid_points = Y_full.shape[1] // 2
            if target_type == "Te":
                self.Y = torch.tensor(Y_full[:, :grid_points], dtype=torch.float32)
            elif target_type == "Ne":
                self.Y = torch.tensor(Y_full[:, grid_points:], dtype=torch.float32)
            else:
                self.Y = torch.tensor(Y_full, dtype=torch.float32)
            
            # Load Y_fit (201 points) if available in npz
            if "Y_fit" in data_dict:
                self.Y_fit = torch.tensor(data_dict["Y_fit"], dtype=torch.float32)
            else:
                self.Y_fit = None

            self.T = torch.tensor(data_dict.get("T_train", data_dict.get("T_infer")), dtype=torch.float32)
        else:
            self.X = torch.tensor(data_dict["X_infer"], dtype=torch.float32)
            if kept_indices is not None:
                self.X = self.X[:, kept_indices]
            self.Y = None
            self.Y_fit = None
            self.T = torch.tensor(data_dict.get("T_infer"), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx], self.T[idx]
        return self.X[idx], self.T[idx]


# =============================================================================
# 2. Residual fully connected network (ResMLP)
# =============================================================================
class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.GELU()  
        self.drop = nn.Dropout(p=dropout_p)  

    def forward(self, x):
        residual = x
        out = self.act(self.linear1(x))
        out = self.drop(out)
        out = self.linear2(out)
        out = self.drop(out)
        return self.act(out + residual)  

class ProfileResMLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 100, hidden_dim: int = 256, num_blocks: int = 4, dropout_p: float = 0.1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p)
        )
        self.blocks = nn.ModuleList([
            ResMLPBlock(dim=hidden_dim, dropout_p=dropout_p) for _ in range(num_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


# =============================================================================
# 3. Physics-based Sobolev loss with core-focused weights
# =============================================================================
class CustomBalancedSobolevLoss(nn.Module):
    def __init__(self, weight_grad: float = 50.0, delta: float = 1.0, grid_points: int = 100):
        super().__init__()
        self.weight_grad = weight_grad
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        self.grid_points = grid_points
        
        # Define linearly decreasing core-to-edge weights: W_j = 1.0 + 4.0 * (1.0 - j / 99).
        # j=0 is closest to the core (rho=0.7) with weight 5.0; j=99 is closest to the edge (rho=1.25) with weight 1.0.
        w = 1.0 + 4.0 * (1.0 - torch.arange(grid_points, dtype=torch.float32) / (grid_points - 1))
        self.register_buffer("point_weights", w)
        
        # The gradient term spans grid_points - 1 intervals and uses analogous weights.
        w_grad = 1.0 + 4.0 * (1.0 - torch.arange(grid_points - 1, dtype=torch.float32) / (grid_points - 2)) if grid_points > 1 else torch.ones(1)
        self.register_buffer("grad_weights", w_grad)

    def forward(self, pred, target):
        num_species = pred.shape[-1] // self.grid_points
        pred_grid = pred.view(pred.size(0), num_species, self.grid_points)
        target_grid = target.view(target.size(0), num_species, self.grid_points)

        # 1. Huber loss on profile values
        loss_val_raw = self.huber(pred_grid, target_grid) # (batch, species, grid_points)
        loss_val = torch.mean(loss_val_raw * self.point_weights.view(1, 1, self.grid_points))

        # 2. Loss on gradients (first differences)
        pred_grad1 = pred_grid[:, :, 1:] - pred_grid[:, :, :-1]
        target_grad1 = target_grid[:, :, 1:] - target_grid[:, :, :-1]
        
        loss_grad_raw = (pred_grad1 - target_grad1) ** 2 # (batch, species, grid_points - 1)
        loss_grad = torch.mean(loss_grad_raw * self.grad_weights.view(1, 1, self.grid_points - 1))

        total_loss = loss_val + self.weight_grad * loss_grad
        return total_loss


# =============================================================================
# 4. Main training pipeline
# =============================================================================
def train_pipeline(data_root: str, target_type: str = "both", epochs: int = 100, batch_size: int = 128, lr: float = 1e-4, kept_indices: list = None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Current accelerator: {device} | Training target: {target_type}")

    train_dataset = PlasmaProfileDataset(os.path.join(data_root, "train", "samples_train.npz"), is_train=True, target_type=target_type, kept_indices=kept_indices)
    val_dataset   = PlasmaProfileDataset(os.path.join(data_root, "val", "samples_train.npz"), is_train=True, target_type=target_type, kept_indices=kept_indices)

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.X.shape[1]
    grid_points = train_dataset.Y.shape[1] // 2 if target_type == "both" else train_dataset.Y.shape[1]
    output_dim = grid_points if target_type in ["Te", "Ne"] else 2 * grid_points
    
    model = ProfileResMLPModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=256, num_blocks=4, dropout_p=0.1).to(device)
    criterion = CustomBalancedSobolevLoss(weight_grad=20.0, delta=1.0, grid_points=grid_points).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')

    print(f"\n--- Training core-weighted robust Huber-Sobolev model [{target_type}] ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{epochs}] | Train Weighted Loss: {train_loss:.5f} | Val Weighted Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = f"best_resmlp_robust_model_{target_type.lower()}_mtanh_v7_weighted.pth"
            torch.save(model.state_dict(), model_name)
            print(f"  [+] [{target_type}] Validation improved; saved new model to {model_name}.")

    print(f"\n[+] [{target_type}] Training complete. Best validation loss: {best_val_loss:.5f}")
    return model, input_dim


# =============================================================================
# 5. Comparison plotting in physical units
# =============================================================================
def evaluate_and_plot_with_raw(model_te, model_ne, data_root: str, h5_folder: str, input_dim: int, num_plots: int = 3, kept_indices: list = None, split_name: str = "test"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_te.to(device)
    model_te.load_state_dict(torch.load("best_resmlp_robust_model_te_mtanh_v7_weighted.pth", map_location=device))
    model_te.eval()

    model_ne.to(device)
    model_ne.load_state_dict(torch.load("best_resmlp_robust_model_ne_mtanh_v7_weighted.pth", map_location=device))
    model_ne.eval()

    dataset = PlasmaProfileDataset(os.path.join(data_root, split_name, "samples_train.npz"), is_train=True, target_type="both", kept_indices=kept_indices)
    scalars = np.load(os.path.join(data_root, "standardization_scalars.npz"))
    metadata = pl.read_parquet(os.path.join(data_root, split_name, "metadata.parquet"))
    
    y_mean = scalars["y_mean"]
    y_std = scalars["y_std"]
    y_mean_fit = scalars["y_mean_fit"]
    y_std_fit = scalars["y_std_fit"]
    
    grid_points_mtanh = len(y_mean) // 2       # 100
    grid_points_fit = len(y_mean_fit) // 2     # 201

    random_indices = np.random.choice(len(dataset), size=num_plots, replace=False)
    
    rho_100 = np.linspace(0.7, 1.25, grid_points_mtanh)
    rho_201 = np.linspace(0.0, 1.25, grid_points_fit)

    # Configure plot styling.
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    print(f"\n[*] Sampling {num_plots} cases from the independent test set and plotting them in physical units...")
    
    for count, idx in enumerate(random_indices, 1):
        row = metadata.filter((pl.col('train_start_idx') <= idx) & (pl.col('train_start_idx') + pl.col('train_seq_len') > idx))
        if len(row) == 0:
            print(f"No metadata found for index {idx}; skipping.")
            continue
            
        shot_id = row['shot_id'][0]
        
        test_x, test_y_norm, time_ms_tensor = dataset[idx]
        time_ms = float(time_ms_tensor)
        time_sec = time_ms / 1000.0

        with torch.no_grad():
            pred_y_norm_te = model_te(test_x.unsqueeze(0).to(device)).cpu().numpy().squeeze()
            pred_y_norm_ne = model_ne(test_x.unsqueeze(0).to(device)).cpu().numpy().squeeze()
            pred_y_norm = np.concatenate([pred_y_norm_te, pred_y_norm_ne])

        # pred_y_phys, true_y_phys_mtanh
        pred_y_phys = pred_y_norm * y_std + y_mean
        true_y_phys_mtanh = test_y_norm.numpy() * y_std + y_mean

        # true_y_phys_201 (from self.Y_fit)
        if dataset.Y_fit is not None:
            test_y_fit_norm = dataset.Y_fit[idx].numpy()
            true_y_phys_201 = test_y_fit_norm * y_std_fit + y_mean_fit
        else:
            print("Warning: Dataset does not contain Y_fit, skipping 201-point curves.")
            continue

        # Load the corresponding processed and raw HDF5 data.
        proc_path = os.path.join(h5_folder, f"TCV_zhang_{shot_id}.h5")
        raw_path = os.path.join(h5_folder, f"TCV_zhang_{shot_id}_raw.h5")
        
        if not os.path.exists(proc_path) or not os.path.exists(raw_path):
            print(f"HDF5 files for shot {shot_id} were not found; skipping this plot.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        species_configs = {
            "Te": {
                "ax": axes[0],
                "color_raw": "#ff7f0e",       # Warm bright orange (raw data points)
                "color_pred": "#b34d00",      # Brick red/deep orange (model mtanh reconstruction)
                "color_mtanh": "#2ca02c",     # Green (calculated mtanh target)
                "color_true": "k",            # Black (physical 201-point reference fit)
                "label": "Electron Temperature $T_e$",
                "unit": "eV",
                "slice_mtanh": slice(0, grid_points_mtanh),
                "slice_fit": slice(0, grid_points_fit)
            },
            "Ne": {
                "ax": axes[1],
                "color_raw": "#1f77b4",       # Classic deep blue (raw data points)
                "color_pred": "#004c8c",      # Saturated navy (model mtanh reconstruction)
                "color_mtanh": "#2ca02c",     # Green (calculated mtanh target)
                "color_true": "k",            # Black (physical 201-point reference fit)
                "label": "Electron Density $n_e$",
                "unit": "$10^{19}$ m$^{-3}$",
                "slice_mtanh": slice(grid_points_mtanh, 2 * grid_points_mtanh),
                "slice_fit": slice(grid_points_fit, 2 * grid_points_fit)
            }
        }

        with h5py.File(proc_path, "r") as fp, h5py.File(raw_path, "r") as fr:
            h5_times = fp["time"][:].flatten()
            t_idx = np.argmin(np.abs(h5_times - time_sec))
            
            for sp, cfg in species_configs.items():
                ax = cfg["ax"]
                sl_mtanh = cfg["slice_mtanh"]
                sl_fit = cfg["slice_fit"]

                if sp == "Te":
                    y_scale = 1000.0    # Convert to eV.
                    raw_scale = 1000.0  # Convert raw HDF5 values from keV to eV.
                else:
                    y_scale = 1.0
                    raw_scale = 1.0
                
                # 1. Load raw points and error bars, then restrict them to [0.7, 1.25].
                counts = fr[sp]["counts"][:].flatten()
                raw_rho = fr[sp]["raw_rho"][0]
                raw_profile = fr[sp]["raw_profile"][0]
                raw_error_bar = fr[sp]["raw_error_bar"][0]
                
                start_idx = int(np.sum(counts[:t_idx]))
                end_idx = int(np.sum(counts[:t_idx+1]))
                
                rho_t = raw_rho[start_idx:end_idx]
                profile_t = raw_profile[start_idx:end_idx]
                err_t = raw_error_bar[start_idx:end_idx]
                
                try:
                    bound_rho = float(np.squeeze(fp[sp]["boundary_rho"][t_idx]))
                except Exception:
                    bound_rho = np.nan
                
                # Retain raw measurements within 0.7--1.25.
                raw_mask = (rho_t >= 0.7) & (rho_t <= 1.25)
                rho_t_f = rho_t[raw_mask]
                profile_t_f = profile_t[raw_mask]
                err_t_f = err_t[raw_mask]
                
                # 2. Plot raw measurements with error bars.
                ax.errorbar(
                    rho_t_f, profile_t_f * raw_scale, yerr=err_t_f * raw_scale, fmt='o',
                    color=cfg["color_raw"], ecolor=cfg["color_raw"],
                    alpha=0.6, capsize=3, elinewidth=1, markersize=5,
                    label=rf"Raw Data [0.7, 1.25]", zorder=3
                )
                
                # 3. Plot the 201-point fitted profile over 0.7--1.25 as a reference.
                fit_mask = (rho_201 >= 0.7) & (rho_201 <= 1.25)
                ax.plot(rho_201[fit_mask], true_y_phys_201[sl_fit][fit_mask] * y_scale, 
                        color='grey', linestyle='-', linewidth=1.5, alpha=0.5, label='Reference (Fit Profile)', zorder=4)
                
                # 4. Plot the calculated mtanh target.
                ax.plot(rho_100, true_y_phys_mtanh[sl_mtanh] * y_scale, 
                        color='g', linestyle='--', linewidth=2.5, label='mtanh Target (Calculated)', zorder=5)
                
                # 5. Plot the neural-network reconstruction.
                ax.plot(rho_100, pred_y_phys[sl_mtanh] * y_scale, 
                        color=cfg["color_pred"], linestyle='-', linewidth=2.5, label='Recon Prediction', zorder=6)
                
                # 6. Add guide lines and apply styling.
                if not np.isnan(bound_rho):
                    ax.axvline(bound_rho, color='gray', linestyle='--', alpha=0.7, label=rf"Boundary \rho_b={bound_rho:.3f}")
                ax.axvspan(0.95, 1.05, color='gray', alpha=0.08, label='Narrow Pedestal')
                
                ax.set_title(f"{cfg['label']} Profile (Edge Region)", fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel(r"Normalized Radius \rho", fontsize=12)
                ax.set_ylabel(f"{cfg['label']} [{cfg['unit']}]", fontsize=12)
                ax.legend(fontsize=10, loc="upper right", frameon=True)
                ax.grid(True, linestyle=':', alpha=0.3)
                
                # Keep the x-axis close to the 0.7--1.25 range.
                ax.set_xlim(0.68, 1.27)
                
                # Add safe y-axis margins.
                p_min = np.min(profile_t_f * raw_scale) if len(profile_t_f) > 0 else np.min(pred_y_phys[sl_mtanh] * y_scale)
                p_max = np.max(profile_t_f * raw_scale) if len(profile_t_f) > 0 else np.max(pred_y_phys[sl_mtanh] * y_scale)
                
                y_min = min(p_min, np.min(pred_y_phys[sl_mtanh] * y_scale), np.min(true_y_phys_mtanh[sl_mtanh] * y_scale))
                y_max = max(p_max, np.max(pred_y_phys[sl_mtanh] * y_scale), np.max(true_y_phys_mtanh[sl_mtanh] * y_scale))
                y_range = y_max - y_min
                y_margin = y_range * 0.1 if y_range > 0 else 1.0
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

        fig.suptitle(f"TCV Shot {shot_id} ({split_name.upper()} SET) at {time_sec:.5f} s ({int(round(time_ms))} ms) - Core-focused Loss Weighting (v7_weighted)",
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        if split_name == "val":
            plot_path = f"comparison_edge_val_shot_{shot_id}_time_{int(round(time_ms))}_v7_weighted.png"
        else:
            plot_path = f"comparison_edge_shot_{shot_id}_time_{int(round(time_ms))}_v7_weighted.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  [+] Saved {split_name} comparison plot to {plot_path}")


def evaluate_r2_scores(model_te, model_ne, data_root: str, kept_indices: list = None):
    from sklearn.metrics import r2_score
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_te.to(device).eval()
    model_ne.to(device).eval()
    
    scalars = np.load(os.path.join(data_root, "standardization_scalars.npz"))
    y_mean = scalars["y_mean"]
    y_std = scalars["y_std"]
    grid_points = len(y_mean) // 2
    y_mean_te, y_mean_ne = y_mean[:grid_points], y_mean[grid_points:]
    y_std_te, y_std_ne = y_std[:grid_points], y_std[grid_points:]
    
    print("\n" + "="*80)
    print("      NEURAL NETWORK RECONSTRUCTION PERFORMANCE (R^2 Score on mtanh curves)")
    print("="*80)
    
    for set_name in ["val", "test"]:
        npz_path = os.path.join(data_root, set_name, "samples_train.npz")
        if not os.path.exists(npz_path):
            continue
            
        dataset = PlasmaProfileDataset(npz_path, is_train=True, target_type="both", kept_indices=kept_indices)
        X_t = dataset.X.to(device)
        Y_norm = dataset.Y.numpy()
        
        with torch.no_grad():
            pred_te_norm = model_te(X_t).cpu().numpy()
            pred_ne_norm = model_ne(X_t).cpu().numpy()
            
        # Convert to physical units
        true_te_phys = Y_norm[:, :grid_points] * y_std_te + y_mean_te
        true_ne_phys = Y_norm[:, grid_points:] * y_std_ne + y_mean_ne
        pred_te_phys = pred_te_norm * y_std_te + y_mean_te
        pred_ne_phys = pred_ne_norm * y_std_ne + y_mean_ne
        
        # Calculate R2 (column-wise uniform average)
        r2_te_col = r2_score(true_te_phys, pred_te_phys)
        r2_ne_col = r2_score(true_ne_phys, pred_ne_phys)
        r2_total_col = r2_score(
            np.concatenate([true_te_phys, true_ne_phys], axis=1),
            np.concatenate([pred_te_phys, pred_ne_phys], axis=1)
        )
        
        # Calculate R2 (flattened overall variance explained)
        r2_te_flat = r2_score(true_te_phys.flatten(), pred_te_phys.flatten())
        r2_ne_flat = r2_score(true_ne_phys.flatten(), pred_ne_phys.flatten())
        r2_total_flat = r2_score(
            np.concatenate([true_te_phys, true_ne_phys], axis=1).flatten(),
            np.concatenate([pred_te_phys, pred_ne_phys], axis=1).flatten()
        )
        
        print(f"Dataset: {set_name.capitalize():<5} (Column-Wise Avg)  | Te R^2: {r2_te_col:.4f} | Ne R^2: {r2_ne_col:.4f} | Total R^2: {r2_total_col:.4f}")
        print(f"Dataset: {set_name.capitalize():<5} (Flattened Shape)   | Te R^2: {r2_te_flat:.4f} | Ne R^2: {r2_ne_flat:.4f} | Total R^2: {r2_total_flat:.4f}")
        print("-" * 80)
    print("="*80 + "\n")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    DATA_ROOT_DIR = str(workspace_root / "intergral_v7_weighted")
    H5_FOLDER = str(workspace_root / "TCV_Processed_H5_compare")
    
    # 0. Get kept feature indices based on rank threshold 50
    kept_indices, kept_names = get_kept_feature_indices(DATA_ROOT_DIR, rank_threshold=50)
    in_features = len(kept_indices)
    
    # Get grid_points dynamically
    scalars = np.load(os.path.join(DATA_ROOT_DIR, "standardization_scalars.npz"))
    grid_points = len(scalars["y_mean"]) // 2
    print(f"[*] Dynamically detected grid points: {grid_points}")
    
    # 1. Train the electron-temperature model (Te).
    print("\n" + "="*50)
    print("      TRAINING ELECTRON TEMPERATURE (Te) MODEL (mtanh v7_weighted)")
    print("="*50)
    model_te, _ = train_pipeline(data_root=DATA_ROOT_DIR, target_type="Te", epochs=100, batch_size=128, lr=1e-4, kept_indices=kept_indices)
    
    # 2. Train the electron-density model (Ne).
    print("\n" + "="*50)
    print("      TRAINING ELECTRON DENSITY (Ne) MODEL (mtanh v7_weighted)")
    print("="*50)
    model_ne, _ = train_pipeline(data_root=DATA_ROOT_DIR, target_type="Ne", epochs=100, batch_size=128, lr=1e-4, kept_indices=kept_indices)
    
    # 3. Load both models, create comparison plots, and report R2 scores.
    eval_model_te = ProfileResMLPModel(input_dim=in_features, output_dim=grid_points, hidden_dim=256, num_blocks=4, dropout_p=0.1)
    eval_model_ne = ProfileResMLPModel(input_dim=in_features, output_dim=grid_points, hidden_dim=256, num_blocks=4, dropout_p=0.1)
    
    # Plot test set (10 plots)
    evaluate_and_plot_with_raw(eval_model_te, eval_model_ne, data_root=DATA_ROOT_DIR, h5_folder=H5_FOLDER, input_dim=in_features, num_plots=10, kept_indices=kept_indices, split_name="test")
    # Plot validation set (5 plots)
    evaluate_and_plot_with_raw(eval_model_te, eval_model_ne, data_root=DATA_ROOT_DIR, h5_folder=H5_FOLDER, input_dim=in_features, num_plots=5, kept_indices=kept_indices, split_name="val")
    
    evaluate_r2_scores(eval_model_te, eval_model_ne, DATA_ROOT_DIR, kept_indices=kept_indices)
