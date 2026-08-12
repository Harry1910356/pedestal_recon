# 未复制文件清单

以下大小来自隔离包创建时的本地文件。它们不在 GitHub 发布目录中。

## 因体积较大而排除的数据文件

| 原文件 | 大小 | 恢复位置/用途 |
|---|---:|---|
| `intergral_v7_weighted/train/samples_infer.npz` | 798,897,472 bytes（约762 MiB） | 全训练 shot 的1 ms无标签推理序列 |
| `intergral_v7_weighted/train/samples_train.npz` | 296,710,039 bytes（约283 MiB） | v7/v10 训练必需 |
| `intergral_v7_weighted/val/samples_infer.npz` | 225,881,045 bytes（约215 MiB） | validation 无标签推理序列 |
| `intergral_v7_weighted/test/samples_infer.npz` | 118,440,100 bytes（约113 MiB） | test 无标签推理序列 |
| `intergral_v7_weighted/val/samples_train.npz` | 83,638,396 bytes（约80 MiB） | validation 与 checkpoint 选择必需 |
| `intergral_v7_weighted/test/samples_train.npz` | 43,677,310 bytes（约42 MiB） | test 评估必需 |

虽然最后两个文件低于 GitHub 的100 MiB硬限制，它们仍被排除，以避免把数据集放入源码仓库并保持 clone 体积较小。

## 因整体体积较大而排除的原始数据目录

| 原目录 | 本地大小 | 内容 |
|---|---:|---|
| `TCV_required_features_integrated/` | 约2.4 GiB | 按 shot 保存的工程特征 parquet，是重新预处理 X 的输入 |
| `TCV_Processed_H5_compare/` | 约1.9 GiB | 处理后的 mtanh 参数与 `_raw.h5` TS 数据，是重新构造 Y 的输入 |
| 完整 `intergral_v7_weighted/` | 约1.5 GiB | 六个大型 NPZ 加已包含的小型 metadata/scalar 文件 |

这些数据应放在对象存储、机构存储或数据发布平台，并在此处补充下载地址和访问权限说明：

```text
DATA_DOWNLOAD_URL_OR_INSTRUCTIONS=尚未填写
```

## 非体积原因排除的产物

| 原目录/文件 | 原因 |
|---|---|
| `v7_weighted_outputs/`（约5.8 MiB） | 可由 checkpoint 和 test 数据重新生成；避免提交大量派生图表和逐样本 CSV |
| `v10_visualizations/`（约8.5 MiB） | 属于旧 shape-balanced v10 的派生结果，与当前全特征 v10 不一致 |
| `v10_outputs/best_resmlp_v10_{te,ne}.pth` | 属于被替换的旧 v10 架构，不能加载到当前 v7-style、112维输入模型 |
| `v10_outputs/` 中其他 history/config/shape label | 属于旧 v10 训练流程，应由当前代码重新生成 |
| 根目录和 `图片/` 下的 `*_v7_weighted.png` | 重复或抽样生成图，不属于运行依赖 |
| `scratch/` 中的 v7 分析脚本 | 含本机绝对路径的一次性分析代码，不属于核心训练/评估依赖 |

## 如果必须版本化大文件

不要把超过100 MiB的文件加入普通 Git 历史。可选方案：

1. 首选外部数据存储，在 README 中记录版本、校验和及下载方法；
2. 或在首次 `git add` 之前配置 Git LFS：

```bash
git lfs install
git lfs track "*.npz" "*.parquet"
git add .gitattributes
```

使用 Git LFS 前需要确认 GitHub 账户的 LFS 存储和下载带宽额度。

