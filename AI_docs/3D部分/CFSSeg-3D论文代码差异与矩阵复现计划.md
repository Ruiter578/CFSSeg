# CFSSeg 3D 论文-代码差异与矩阵复现计划

日期：2026-07-16
分支：`feature/cfsseg-code3d-integration`
范围：`SegACIL/CFSSeg-code3D` 的 ACL / closed-form 3D 复现实验。

## 结论

3D 部分不能只按“论文参数”或“作者脚本参数”单跑一组后就宣称严格复现。论文、作者脚本和 `main.py` 默认值之间存在可影响结果的差异，尤其是 `uncertain_t / tau`。

当前已完成的 S3DIS S0 `8-1` 首轮实验使用的是论文参数 `tau=0.0035`，最终结果为：

| split | tasks | tau | all mIoU | base mIoU | novel mIoU | 论文 all | 论文 base | 论文 novel |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 / cvfold 0 | 8-1 | 0.0035 | 38.63 | 47.87 | 23.85 | 41.66 | 49.77 | 28.69 |

当前结果比论文低约 `3.03` 个 all-mIoU 点。日志还显示所有增量 batch 的 `Uncertain points ratio` 都是 `0.0000`，说明 `tau=0.0035` 在当前实现和数据上没有触发不确定点筛选，必须把作者脚本里的 `tau=0.002` 一并复现对比。

## 论文中的 3D 设置

论文实验设置写明：

| 项 | 论文描述 |
| --- | --- |
| 数据集 | S3DIS、ScanNet |
| S3DIS 验证集 | Area 5 |
| ScanNet 划分 | 官方 train / validation split |
| 输入块 | 1m x 1m sliding window blocks |
| 每块点数 | 2048 |
| 3D 模型 | DGCNN |
| batch size | 32 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| epochs | 100 |
| 持续学习阶段 | freeze encoder，插入 RHL |
| 3D buffer / `dE` | 5000 |
| gamma | 1 |
| S3DIS tau | 0.0035 |
| ScanNet tau | 0.001 |
| S0 split | 原数据集标注顺序 |
| S1 split | 类别名字母顺序 |
| 3D CSS setting | disjoint |

论文 Table 1 / Table 2 的主要 3D 目标值如下。

### S0

| 数据集 | tasks | base | novel | all |
| --- | --- | ---: | ---: | ---: |
| S3DIS | 8-1 | 49.77 | 28.69 | 41.66 |
| S3DIS | 10-1 | 45.26 | 34.05 | 42.67 |
| S3DIS | 12-1 | 45.19 | 30.71 | 44.08 |
| ScanNet | 15-1 | 32.56 | 10.22 | 26.97 |
| ScanNet | 17-1 | 29.59 | 12.83 | 27.98 |
| ScanNet | 19-1 | 28.54 | 16.88 | 27.96 |

### S1

| 数据集 | tasks | base | novel | all |
| --- | --- | ---: | ---: | ---: |
| S3DIS | 8-1 | 51.33 | 30.72 | 43.40 |
| S3DIS | 10-1 | 45.16 | 35.98 | 43.04 |
| S3DIS | 12-1 | 45.65 | 27.53 | 44.25 |
| ScanNet | 15-1 | 28.20 | 16.09 | 25.18 |
| ScanNet | 17-1 | 28.43 | 15.01 | 26.42 |
| ScanNet | 19-1 | 28.71 | 11.84 | 27.84 |

## 作者代码中的差异

| 位置 | 代码值 | 论文值 | 判断 |
| --- | --- | --- | --- |
| `scripts/train.sh` S3DIS ACL | `UNCERTAIN_T_VALUES=(0.002)` | `0.0035` | 关键差异，必须复现两组 |
| `scripts/train.sh` ScanNet ACL | `0.0005` | `0.001` | 关键差异，必须复现两组 |
| `main.py` 默认 `--uncertain_t` | `0.0065` | S3DIS `0.0035` / ScanNet `0.001` | 不能依赖默认值，必须显式传参 |
| `train1.sh` / `increOurs` | S3DIS `0.0065`、ScanNet `0.0045` | 不对应 ACL 论文设置 | 先不作为主复现路径 |
| `utils/AnalyticLinear.py` 默认 gamma | `1e-1` | `1` | 类默认不同，但 ACL 实际调用传 `gamma=1`，可接受 |
| `train_ACL.py` 中 AIR buffer | `buffer_size=5000` | `dE=5000` | 对齐 |
| `train_ACL.py` 原 step 循环 | `range(1, STEP)` | 从 base step 开始完整复现 | 原写法更像只跑增量；已改为 `--start_step` 参数，默认 `0` |

## 为什么要做矩阵实验

2D 部分已经出现过“论文写 8192、源码/复现最佳值不同”的情况。3D 部分同样存在 tau 不一致，而且首轮论文 tau 下 `Uncertain points ratio` 全为 0。后续需要把“论文参数”和“作者脚本参数”都复现出来，再判断哪个更接近论文表格。

矩阵实验只改变一个关键因素：`uncertain_t / tau`。其他参数保持论文设置。

## 最小矩阵

第一阶段先做 S3DIS S0 `8-1`，因为它已经有一组论文 tau 首轮结果，可直接对比。

| 优先级 | 数据集 | split | tasks | tau | 来源 | 目的 |
| --- | --- | --- | --- | ---: | --- | --- |
| P0 | S3DIS | S0 / cvfold 0 | 8-1 | 0.0035 | 论文 | 已完成，作为 paper-aligned baseline |
| P0 | S3DIS | S0 / cvfold 0 | 8-1 | 0.002 | 作者 `train.sh` | 验证作者脚本值是否更接近论文结果 |
| P1 | S3DIS | S1 / cvfold 1 | 8-1 | 0.0035 | 论文 | 验证 S1 是否也稳定 |
| P1 | S3DIS | S1 / cvfold 1 | 8-1 | 0.002 | 作者 `train.sh` | 和 S1 论文值对比 |

## 完整表格复现矩阵

如果最小矩阵确认代码链路可靠，再扩展到完整 Table 1 / Table 2。

| 数据集 | split | tasks | tau 候选 |
| --- | --- | --- | --- |
| S3DIS | S0 / cvfold 0 | 8-1, 10-1, 12-1 | 0.0035, 0.002 |
| S3DIS | S1 / cvfold 1 | 8-1, 10-1, 12-1 | 0.0035, 0.002 |
| ScanNet | S0 / cvfold 0 | 15-1, 17-1, 19-1 | 0.001, 0.0005 |
| ScanNet | S1 / cvfold 1 | 15-1, 17-1, 19-1 | 0.001, 0.0005 |

这不是 8 个实验，而是完整参数核验会达到 `2 datasets x 2 splits x 3 tasks x 2 tau = 24` 组。若只跑每个数据集的代表性多步任务，则是 `S3DIS 8-1 x 2 splits x 2 tau + ScanNet 15-1 x 2 splits x 2 tau = 8` 组。

## 推荐启动命令

先在 GPU 2 上补 S3DIS S0 `8-1, tau=0.002`：

```bash
cd /TRS-SAS/linwei/SegACIL/CFSSeg-code3D
GPU_ID=2 \
PYTHON_BIN=/opt/conda/envs/segacil/bin/python \
RUN_GROUP=20260716_s3dis_s0_8_1_tau_compare \
TAU_VALUES="0.002" \
TASKS_VALUES="8-1" \
CVFOLDS="0" \
NUM_EPOCHS=100 \
bash scripts/run_acl_s3dis_matrix.sh
```

如果要同时跑论文 tau 和作者脚本 tau：

```bash
cd /TRS-SAS/linwei/SegACIL/CFSSeg-code3D
GPU_ID=2 \
PYTHON_BIN=/opt/conda/envs/segacil/bin/python \
RUN_GROUP=20260716_s3dis_s0_8_1_tau_compare \
TAU_VALUES="0.0035 0.002" \
TASKS_VALUES="8-1" \
CVFOLDS="0" \
NUM_EPOCHS=100 \
bash scripts/run_acl_s3dis_matrix.sh
```

输出会进入：

```text
/TRS-SAS/linwei/SegACIL/checkpoints_3d/s3dis/<RUN_GROUP>/tau_<tau>/log_acl_s3dis_cv<split>_tasks<tasks>/
```

每个新实验会自动写：

```text
run_manifest.json
result_summary.json
```

`.tar`、TensorBoard events 和完整日志仍然不进 Git；轻量 JSON 可以进 Git。

## 旧日志回填

首轮已完成实验可用工具回填：

```bash
cd /TRS-SAS/linwei/SegACIL/CFSSeg-code3D
python tools/extract_acl_results.py \
  --run-dir /TRS-SAS/linwei/SegACIL/checkpoints_3d/s3dis/log_acl_s3dis_cv0_tasks8-1 \
  --write
```

这会生成：

```text
/TRS-SAS/linwei/SegACIL/checkpoints_3d/s3dis/log_acl_s3dis_cv0_tasks8-1/result_summary.json
```

## 当前首轮实验判断

首轮实验不是无效实验，因为：

- split、tasks、batch size、点数、DGCNN、epoch、lr、weight decay 与论文主设置一致。
- 使用了论文 S3DIS tau `0.0035`。
- 训练从 base step 开始，跑完了 S0 `8-1` 的 6-step 流程。

但它不能单独作为“严格复现完成”的证据，因为：

- 结果低于论文表格，尤其 novel mIoU 差距较明显。
- 作者脚本默认 tau 是 `0.002`，尚未对比。
- `Uncertain points ratio` 全为 0，说明论文 tau 在当前链路下没有实际筛出不确定点。
- 旧代码此前没有 JSON/manifest，缺少可审计留痕。

下一步优先级：先跑 `tau=0.002` 的 S3DIS S0 `8-1` 对照，再决定是否扩大到 8 组代表实验或完整 24 组表格复现。
