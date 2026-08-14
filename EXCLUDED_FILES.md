# Excluded Files

The following sizes were measured when this release repository was assembled. These files are not stored in Git.

## Processed arrays excluded because of size

| Original file | Size | Purpose and restore location |
|---|---:|---|
| `intergral_v7_weighted/train/samples_infer.npz` | 798,897,472 bytes (about 762 MiB) | Unlabeled 1 ms inference sequences for training shots |
| `intergral_v7_weighted/train/samples_train.npz` | 296,710,039 bytes (about 283 MiB) | Required for v7/v10/v11 training and v12 fine-tuning |
| `intergral_v7_weighted/val/samples_infer.npz` | 225,881,045 bytes (about 215 MiB) | Unlabeled validation-shot inference sequences |
| `intergral_v7_weighted/test/samples_infer.npz` | 118,440,100 bytes (about 113 MiB) | Unlabeled test-shot inference sequences |
| `intergral_v7_weighted/val/samples_train.npz` | 83,638,396 bytes (about 80 MiB) | Required for validation and checkpoint selection across all four versions |
| `intergral_v7_weighted/test/samples_train.npz` | 43,677,310 bytes (about 42 MiB) | Required for labeled test evaluation across all four versions |

The final two files are below GitHub's 100 MiB hard limit, but they are still excluded to keep datasets out of the source repository and to keep clones small.

## Excluded source-data directories

| Original directory | Local size | Contents |
|---|---:|---|
| `TCV_required_features_integrated/` | About 2.4 GiB | 3,946 per-shot parquet files containing the time-dependent X features |
| `TCV_Processed_H5_compare/` | About 1.9 GiB | 3,760 pairs of fitted-parameter `.h5` and reference/raw `_raw.h5` files used to construct Y |
| Complete `intergral_v7_weighted/` | About 1.5 GiB | The six large NPZ files plus the small metadata and scalar files that are included here |

Store these datasets in controlled object storage, institutional storage, or a data-publication service. Add the download URL, version, access restrictions, and checksums here when available:

```text
DATA_DOWNLOAD_URL_OR_INSTRUCTIONS=not yet provided
```

## Generated artifacts excluded for non-size reasons

| Original directory or file | Reason |
|---|---|
| `v7_weighted_outputs/` (about 5.8 MiB) | Derived plots, per-sample metrics, and reports can be regenerated from the v7 checkpoints and test data |
| `v10_visualizations/` (about 8.5 MiB) | Results belong to the previous shape-balanced v10 and do not describe the current all-feature v10 |
| `v10_outputs/best_resmlp_v10_{te,ne}.pth` | Checkpoints use the replaced v10 architecture and cannot be loaded by the current v7-style, 112-input model |
| Other old `v10_outputs/` history, configuration, and shape-label files | Artifacts of the replaced v10 workflow; regenerate them with the current implementation |
| `v11_predictions/` and `v11_visualizations/` | Derived predictions, plots, and metric reports can be regenerated from the included v11 checkpoints and restored test data |
| Root-level and auxiliary image-directory `*_v7_weighted.png` files | Duplicate or sampled plots rather than runtime dependencies |
| v7 scripts under `scratch/` | One-off analysis code with machine-specific paths; not part of the core training/evaluation path |

## If large files must be versioned

Do not add files larger than 100 MiB to normal Git history. Prefer external data storage with documented versions and checksums. If Git LFS is necessary, configure it before the first `git add`:

```bash
git lfs install
git lfs track "*.npz" "*.parquet"
git add .gitattributes
```

Confirm the GitHub account's LFS storage and download-bandwidth allowance before adopting this approach.
