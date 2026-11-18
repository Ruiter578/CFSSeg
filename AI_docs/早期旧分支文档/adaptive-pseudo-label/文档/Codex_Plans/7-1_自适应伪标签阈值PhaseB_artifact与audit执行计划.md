# 自适应伪标签阈值 Phase B：artifact 与 audit 执行计划

生成时间：2026-07-01
目标分支：`feature/adaptive-pseudo-label`
当前决策：暂不启动训练；先补齐 artifact 执行链与质量审计工具。

## 1. 本计划回答的三个事实问题

### 1.1 6-30 Phase A Codex plan 对应小白版五阶段的哪里

`Codex_Plans/6-30_自适应伪标签阈值PhaseA执行计划.md` 主要对应小白版工作流的：

```text
阶段一：Phase A 搜索基础设施与 fixed sweep
```

它在末尾提到 fixed sweep 完成后进入 adaptive low-q sweep，但它本身不是完整的阶段二执行文档。实际执行中，后续已经补充并跑完了阶段二：

```text
batch_global q=0.1 / 0.3 / 0.5
batch_class  q=0.1 / 0.3 / 0.5
batch_class q=0.3 + min_conf=0.6
```

### 1.2 当前是否只完成 Phase A

不是。当前已经完成：

```text
Phase A 代码基础设施：grid runner、summary 工具、grid summary 单测
Phase A fixed sweep：fixed0.6 / fixed0.8 / fixed0.9
Phase A adaptive low-q sweep：batch_global / batch_class q=0.1/0.3/0.5 等
```

Phase A 的实验结论已经足够明确：

```text
fixed0.6 当前最强；
batch_class q0.1 接近 fixed0.6，但没有超过；
继续扩大 batch-level quantile 网格优先级不高。
```

因此下一步不是直接扩大 Phase A，而是进入 Phase B：offline artifact calibration 和 pseudo-label quality audit。

### 1.3 小白版第 4 章四种“自适应”是否安排妥当

旧版五阶段安排不够完整：

| 第 4 章方向 | 旧工作流覆盖情况 | 当前修正 |
| --- | --- | --- |
| 接受比例自适应 | 阶段二覆盖 | 已完成，保留为消融 |
| 固定阈值校准 | 阶段一覆盖 | 已完成，`fixed0.6` 成为强 baseline |
| 离线校准 artifact | 只被笼统提到 | 提升为 Phase B 主任务 |
| 质量感知自适应 | 只部分落在 raw-mask audit | 先 audit，再决定 margin/cap/weighted 是否实现 |

## 2. 已确认的 Phase A 结果

同协议：VOC `15-5 overlap step1`，`deeplabv3_resnet101`，batch size 32，seed 1，`buffer=8196`，`gamma=1`。

共同 step0：

```text
checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
sha256 = 6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913
```

核心结果：

| strategy | all mIoU | old | new | accepted |
| --- | ---: | ---: | ---: | ---: |
| `fixed0.6` | **70.7731** | **79.6761** | **42.2836** | 94.9132% |
| `batch_class q0.1` | 70.7694 | 79.6758 | 42.2691 | 89.9997% |
| `fixed0.7` | 70.7383 | 79.6346 | 42.2703 | 90.3503% |
| `batch_global q0.1` | 70.7358 | 79.6345 | 42.2601 | 90.0000% |
| `off` | 70.3091 | 79.1179 | 42.1209 | - |

判定：

```text
best fixed = fixed0.6
best batch adaptive = batch_class q0.1
Phase B 必须至少超过 fixed0.6，才算自适应阈值重新成立。
```

## 3. 推荐方案

### 方案 A：继续扩大 batch quantile 网格

例如补 `q=0.05 / 0.15 / 0.2`。

优点：改动最少。

缺点：当前趋势已经明确，batch quantile 与 fixed0.6 的差距非常小，继续跑大概率只能得到微小扰动，论文价值弱。

结论：不推荐作为下一步主线。

### 方案 B：直接上 15-1 / 10-1 论文协议

优点：能更快接近论文主表。

缺点：当前 adaptive 还没有超过 fixed0.6。把未胜出的机制直接带到多 step 大实验，成本高且解释风险大。

结论：暂不推荐。

### 方案 C：Phase B artifact + audit

先用 teacher 离线扫训练集生成 per-class threshold artifact，再用 `artifact_class` 训练；同时用 audit 工具解释 fixed / batch / artifact 的 pseudo precision / recall。

优点：

```text
比 batch 内分位数更稳定；
能给出“为什么 adaptive 有效/无效”的证据；
更接近可写进论文的方法形态；
如果失败，也能清楚降级为消融。
```

结论：推荐执行。

## 4. Phase B 代码改动范围

### 4.1 扩展 grid runner

当前 `tools/run_pseudo_label_grid.sh` 的 TSV 字段不包含：

```text
threshold_artifact
threshold_max_batches
```

但 `tools/run_adaptive_pseudo_label.sh` 和训练代码已经能接收：

```text
PSEUDO_LABEL_THRESHOLD_ARTIFACT
PSEUDO_LABEL_THRESHOLD_MAX_BATCHES
```

需要改动：

```text
tools/run_pseudo_label_grid.sh
tools/summarize_pseudo_label_grid.py
tests/test_pseudo_label_grid_summary.py
```

设计要求：

1. 保持旧 TSV 兼容。
2. 支持新 TSV 在末尾增加可选字段：

```text
threshold_artifact
threshold_max_batches
```

3. dry-run 必须打印 artifact 环境变量。
4. summary 输出中记录 artifact path。
5. 如果 `strategy=artifact_class` 但 artifact path 为空，runner 直接报错。

### 4.2 新增 artifact calibration grid

新增：

```text
configs/pseudo_label_phaseB_artifact_calibration.tsv
```

建议字段：

```text
name
artifact_path
task
setting
teacher_ckpt
quantile
min_conf
max_conf
min_pixels
shrinkage
max_batches
batch_size
random_seed
```

建议先做 3 个 full artifacts：

| name | quantile | min_conf | shrinkage | 目的 |
| --- | ---: | ---: | ---: | --- |
| `artifact_q0p05` | 0.05 | 0.0 | 0 | 接受约 top 95%，贴近 `fixed0.6` 召回 |
| `artifact_q0p10` | 0.10 | 0.0 | 0 | 接受约 top 90%，贴近 `batch_class q0.1` |
| `artifact_q0p20` | 0.20 | 0.0 | 0 | 接受约 top 80%，观察更高精度是否更稳 |

### 4.3 新增 artifact calibration runner

新增：

```text
tools/run_pseudo_label_artifact_calibration_grid.sh
```

行为：

1. 读取 `configs/pseudo_label_phaseB_artifact_calibration.tsv`。
2. 调用 `tools/calibrate_pseudo_label_thresholds.py`。
3. 若 artifact 已存在且 JSON 可读，默认跳过。
4. 若 artifact 文件存在但 JSON 不合法，直接报错。
5. 支持 `--mode dry-run|run`。

### 4.4 新增 artifact train grid

新增：

```text
configs/pseudo_label_phaseB_artifact_train.tsv
```

建议 3 组：

| name | strategy | artifact |
| --- | --- | --- |
| `artifact_q0p05` | `artifact_class` | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p05.json` |
| `artifact_q0p10` | `artifact_class` | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p10.json` |
| `artifact_q0p20` | `artifact_class` | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p20.json` |

统一配置：

```text
TASK=15-5
SETTING=overlap
BASE_SUBPATH=20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32
SKIP_STEP0=1
BATCH_SIZE=32
STEP0_BATCH_SIZE=32
BUFFER=8196
GAMMA=1
RANDOM_SEED=1
MODEL=deeplabv3_resnet101
AIR_FEATURE_SOURCE=auto
```

### 4.5 raw-mask audit

如果时间允许，新增：

```text
tools/audit_pseudo_label_quality.py
```

最低功能：

1. 输入 teacher checkpoint、task、setting、strategy 参数。
2. 在 step1 train set 上重建 pseudo-label 候选。
3. 使用原始 VOC mask 作为参考，计算：

```text
candidate_count
accepted_count
accepted_ratio
old-class pseudo precision
old-class pseudo recall
per-class precision
per-class recall
top confused classes
```

4. 输出 JSON + Markdown。

如果 raw mask 读取路径或 label remap 关系不清楚，先只做 artifact 训练，不要把 audit 写成猜测逻辑。

## 5. 执行顺序

### Step 0：保护现场

```bash
cd /root/2TStorage/lyc/SegACIL
git status --short --branch
nvidia-smi
tmux ls
```

如果发现和本任务无关的未提交修改，不要覆盖。

### Step 1：实现 runner / summary 的 artifact 字段

改动文件：

```text
tools/run_pseudo_label_grid.sh
tools/summarize_pseudo_label_grid.py
tests/test_pseudo_label_grid_summary.py
```

验收：

```bash
bash -n tools/run_pseudo_label_grid.sh
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile tools/summarize_pseudo_label_grid.py tests/test_pseudo_label_grid_summary.py
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest tests.test_pseudo_label_grid_summary tests.test_pseudo_labeling tests.test_run_manifest -v
```

### Step 2：实现 calibration grid runner

新增：

```text
tools/run_pseudo_label_artifact_calibration_grid.sh
configs/pseudo_label_phaseB_artifact_calibration.tsv
```

先 dry-run：

```bash
bash tools/run_pseudo_label_artifact_calibration_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_calibration.tsv \
  --mode dry-run
```

再用 `max_batches=2` 的临时小 grid 做 smoke，不直接跑 full。

### Step 3：准备 artifact train grid

新增：

```text
configs/pseudo_label_phaseB_artifact_train.tsv
```

dry-run：

```bash
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_train.tsv \
  --mode dry-run
```

要求：

```text
dry-run 中必须能看到 PSEUDO_LABEL_THRESHOLD_ARTIFACT
strategy=artifact_class 时 artifact 路径不能为空
```

### Step 4：CodeRabbit 审查

```bash
coderabbit review --agent -t uncommitted --dir /root/2TStorage/lyc/SegACIL
```

修复所有 critical / warning / major，再复审。

### Step 5：是否启动训练

只有满足以下条件才启动：

```text
calibration dry-run 通过；
小样本 calibration smoke 通过；
artifact JSON schema 正确；
artifact train grid dry-run 能正确展开；
CodeRabbit 无 critical / warning / major；
GPU 显存足够。
```

启动顺序：

1. full calibration 生成 3 个 artifact；
2. `artifact_class` 训练 3 组；
3. 生成 summary。

tmux 建议：

```bash
tmux new-session -d -s apl_phaseB_artifact \
  "cd /root/2TStorage/lyc/SegACIL && \
   PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
   CUDA_VISIBLE_DEVICES=0 \
   CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass \
   TMPDIR=/root/2TStorage/tmp \
   bash tools/run_pseudo_label_artifact_calibration_grid.sh \
     --grid configs/pseudo_label_phaseB_artifact_calibration.tsv \
     --mode run && \
   bash tools/run_pseudo_label_grid.sh \
     --grid configs/pseudo_label_phaseB_artifact_train.tsv \
     --mode run 2>&1 | tee logs/pseudo_label/phaseB_artifact.log"
```

正式执行前要注意 shell 管道优先级；更稳妥的做法是写一个明确的 runner 脚本，避免 `tee` 只覆盖后半段命令。

## 6. 判定标准

当前强 baseline：

```text
fixed0.6 all mIoU = 70.7731
batch_class q0.1 all mIoU = 70.7694
```

判定：

```text
artifact_class > 70.7731：
    Phase B 成立，进入 15-5 disjoint 和 15-1 overlapped。
artifact_class 在 70.75-70.7731 附近：
    算贴近 fixed，需要 raw-mask audit 决定是否做 margin/cap/weighted。
artifact_class < 70.7383：
    artifact 没有超过 fixed0.7，暂不继续扩大同类训练，先 audit 或降级。
```

## 7. 不要做的事

- 不要直接启动 15-1 大实验。
- 不要继续无目的扩大 batch quantile 网格。
- 不要覆盖已有 `checkpoints`。
- 不要把 artifact calibration 输出放进 checkpoints 目录；建议放在 `artifacts/pseudo_label/`。
- 不要把 raw-mask audit 写成猜测逻辑；如果找不到原始 mask 对齐方式，先暂停并汇报。
