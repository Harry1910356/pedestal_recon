# TCV profile reconstruction: v7 and v10

这个隔离包只包含 v7/v10 的代码、小型数据契约和可直接提交 GitHub 的模型文件，不包含大型训练数据或原始 TCV 数据。

## 两个版本

| 版本 | 网络与损失 | 输入特征 |
|---|---|---|
| v7_weighted | 4-block ResMLP；核心加权 Huber-Sobolev loss | 从112列中按 Te/Ne RF 与 permutation rank 阈值50选出的69列 |
| v10 | 与 v7 相同的 dataset、ResMLP 和 loss | 全部112列，不读取 feature-importance ranking |

v10 直接复用 v7 中的公共 dataset、网络和 loss 定义，避免两个实现逐渐不一致。

## 已包含的内容

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

六个大型 `samples_*.npz`、原始 parquet/HDF5 和生成结果没有复制。准确大小和应恢复的位置见 [EXCLUDED_FILES.md](EXCLUDED_FILES.md)。

## 安装

需要 Python 3.11。使用 uv：

```bash
uv sync --dev
uv run pytest -q
```

或使用普通 venv/pip：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## 恢复数据

代码和模型可以被 Git clone，但训练或数据集评估前，必须从项目外部存储取回所需 NPZ，并放回以下位置：

```text
intergral_v7_weighted/train/samples_train.npz
intergral_v7_weighted/train/samples_infer.npz
intergral_v7_weighted/val/samples_train.npz
intergral_v7_weighted/val/samples_infer.npz
intergral_v7_weighted/test/samples_train.npz
intergral_v7_weighted/test/samples_infer.npz
```

只做 test 评估时，仅需要 `test/samples_train.npz`；不需要 `samples_infer.npz`。建议在 `EXCLUDED_FILES.md` 中补充数据下载地址或机构存储位置，而不是把大文件提交普通 Git。

## v7 评估

发布包包含已有 v7 Te/Ne checkpoint。恢复 `test/samples_train.npz` 后运行：

```bash
uv run python lh_transitions/visualize_resmlp_v7.py \
  --root . \
  --data-root intergral_v7_weighted \
  --split test \
  --output-dir v7_weighted_outputs \
  --device auto \
  --rank-threshold 50
```

两份 ranking CSV 对 v7 checkpoint 是必要的，因为 checkpoint 接受69维输入，而数据保存112列。CSV 用来重建训练时采用的69列及其顺序。

## v10 训练与评估

当前 v10 是 v7 的全特征对照版本，固定使用全部112列，不接受 `--rank-threshold`。旧 shape-balanced v10 checkpoint 与新架构不兼容，因此没有放进隔离包；需要重新训练：

```bash
uv run python lh_transitions/profile_recon_robust_sobolev_mtanh_v10.py \
  --data-root intergral_v7_weighted \
  --output-dir v10_outputs \
  --target both \
  --epochs 100 \
  --batch-size 128 \
  --device auto
```

训练后评估：

```bash
uv run python lh_transitions/visualize_resmlp_v10.py \
  --data-root intergral_v7_weighted \
  --checkpoint-dir v10_outputs \
  --output-dir v10_visualizations \
  --split test \
  --device auto
```

## 从原始数据重新预处理

需要恢复 `TCV_required_features_integrated/` 和 `TCV_Processed_H5_compare/`。它们没有包含在 Git 包中。调用参数化函数，不要使用原脚本底部遗留的本机绝对路径：

```bash
uv run python -c "from lh_transitions.raw_ped_v7_weighted import build_reconstruction_dataset; build_reconstruction_dataset(x_folder='TCV_required_features_integrated', h5_folder='TCV_Processed_H5_compare', output_root='intergral_v7_weighted', flat_top_csv='flat_top_times.csv', split_ratios=(0.7, 0.2, 0.1), split_seed=42, target_dt_ms=1.0)"
```

## 推送到 GitHub

先在 GitHub 创建一个空仓库，不要勾选自动创建 README、license 或 `.gitignore`。然后在本目录执行：

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Add v7 and all-feature v10 profile reconstruction"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

推送前建议确认没有意外的大文件：

```bash
find . -type f -size +50M -not -path './.git/*'
```

正常结果应为空。

## SSH 安全

不要把 SSH 私钥、私钥文件内容、GitHub token 或密码提交到仓库或发给协作者。推送只需要：

1. 把本机 SSH 公钥添加到 GitHub 账号；
2. 本机 `ssh-agent` 能访问对应私钥；
3. 使用仓库的 SSH URL，例如 `git@github.com:account/repo.git`。

可以用下面的命令验证认证；它不会上传项目文件：

```bash
ssh -T git@github.com
```

