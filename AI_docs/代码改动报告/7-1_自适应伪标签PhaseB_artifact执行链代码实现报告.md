# 7-1 自适应伪标签 PhaseB artifact 执行链代码实现报告

生成时间：2026-07-01
分支：`feature/adaptive-pseudo-label`
执行范围：只完成 PhaseB artifact 执行链与启动前验证；未启动训练。

## 1. 本次目标

Phase A 已确认 `fixed0.6` 是当前最强 baseline，batch-level quantile adaptive 没有明显超过 fixed 阈值。因此 Phase B 的目标不是继续扩大 batch quantile，而是把更稳定的 offline threshold artifact 路线打通：

1. 先用 step0 teacher 离线扫描 step1 train loader，生成 per-class threshold artifact。
2. 再让 `artifact_class` 策略按行读取 artifact，启动 step1 对照训练。
3. summary 工具能记录 artifact path，保证实验结果可追溯。
4. 在真正启动训练前完成 dry-run、测试和代码审查。

## 2. 代码改动

### 2.1 `tools/run_pseudo_label_grid.sh`

改动点：

- 兼容旧 Phase A TSV header。
- 支持 Phase B 在 TSV 末尾追加：

```text
threshold_artifact
threshold_max_batches
```

- dry-run 和正式 run 都会传递：

```bash
PSEUDO_LABEL_THRESHOLD_ARTIFACT
PSEUDO_LABEL_THRESHOLD_MAX_BATCHES
```

- `strategy=artifact_class` 但 `threshold_artifact` 为空时直接退出，避免静默跑成错误配置。
- 增加 header 校验，防止 TSV 列错位。

### 2.2 `tools/summarize_pseudo_label_grid.py`

改动点：

- `read_grid()` 同时接受旧 header 与新 header。
- 旧 header 会自动补空：

```text
threshold_artifact = ""
threshold_max_batches = ""
```

- CSV / JSON / Markdown summary 增加 artifact 字段，便于后续确认结果来自哪个离线阈值文件。

### 2.3 `tools/run_pseudo_label_artifact_calibration_grid.sh`

新增 artifact calibration grid runner。

行为：

- 读取 `configs/pseudo_label_phaseB_artifact_calibration.tsv`。
- 调用 `tools/calibrate_pseudo_label_thresholds.py` 生成 threshold artifact。
- 支持 `--mode dry-run|run`。
- 如果 artifact 已存在且是合法 JSON，默认跳过。
- 如果 artifact 已存在但 JSON 不合法，直接报错。
- 校验 teacher checkpoint 是否存在。
- 校验 header、必填字段和重复 artifact path。

### 2.4 新增 PhaseB 配置

新增：

```text
configs/pseudo_label_phaseB_artifact_calibration.tsv
configs/pseudo_label_phaseB_artifact_train.tsv
```

当前准备的三组 artifact：

| name | quantile | artifact |
| --- | ---: | --- |
| `artifact_q0p05` | 0.05 | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p05.json` |
| `artifact_q0p10` | 0.10 | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p10.json` |
| `artifact_q0p20` | 0.20 | `artifacts/pseudo_label/phaseB_15-5_overlap_q0p20.json` |

共同配置：

```text
task = 15-5
setting = overlap
step0 teacher = checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
BASE_SUBPATH = 20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32
SKIP_STEP0 = 1
BATCH_SIZE = 32
STEP0_BATCH_SIZE = 32
BUFFER = 8196
GAMMA = 1
RANDOM_SEED = 1
MODEL = deeplabv3_resnet101
AIR_FEATURE_SOURCE = auto
```

### 2.5 测试

更新：

```text
tests/test_pseudo_label_grid_summary.py
```

新增覆盖：

- 旧 Phase A header 仍可读取，并补空 artifact 字段。
- 新 Phase B header 可读取 artifact 字段。
- `run_pseudo_label_grid.sh --mode dry-run` 能输出 artifact 环境变量。
- `artifact_class` 缺 `threshold_artifact` 时 runner 报错。

## 3. 未实现项

raw-mask audit 本次没有实现。

原因：计划文档明确要求，如果 raw mask 读取路径和 label remap 对齐关系没有充分确认，不要写猜测逻辑。当前本次任务的主要目标是“训练前打通 artifact 执行链”，audit 应作为下一步独立实现项，在确认 VOC raw mask 与 step1 train sample 的对齐方式后再做。

## 4. 验证记录

### 4.1 语法与空白

已通过：

```bash
bash -n tools/run_pseudo_label_grid.sh
bash -n tools/run_pseudo_label_artifact_calibration_grid.sh
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile tools/summarize_pseudo_label_grid.py tools/calibrate_pseudo_label_thresholds.py tests/test_pseudo_label_grid_summary.py
git diff --check
grep -n '[“”‘’]' <changed executable/config files>
```

`shellcheck` 本机未安装，因此未执行 shellcheck 静态检查。

### 4.2 单元测试

已通过：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest tests.test_pseudo_label_grid_summary tests.test_pseudo_labeling tests.test_run_manifest -v
```

结果：

```text
Ran 26 tests
OK
```

已通过全量测试：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

结果：

```text
Ran 44 tests
OK
```

### 4.3 dry-run

artifact calibration dry-run 已通过：

```bash
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
bash tools/run_pseudo_label_artifact_calibration_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_calibration.tsv \
  --mode dry-run
```

确认展开了 3 个 artifact 生成命令，且 teacher checkpoint 路径存在。

artifact train grid dry-run 已通过：

```bash
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_train.tsv \
  --mode dry-run
```

确认每行都传入：

```text
PSEUDO_LABEL_STRATEGY=artifact_class
PSEUDO_LABEL_THRESHOLD_ARTIFACT=artifacts/pseudo_label/phaseB_15-5_overlap_q*.json
PSEUDO_LABEL_THRESHOLD_MAX_BATCHES=0
SKIP_STEP0=1
BASE_SUBPATH=20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32
```

summary smoke 已通过：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_phaseB_artifact_train.tsv \
  --output-md /tmp/<tmpdir>/summary.md \
  --output-csv /tmp/<tmpdir>/summary.csv \
  --output-json /tmp/<tmpdir>/summary.json \
  --title 'PhaseB Artifact Grid Smoke'
```

输出中能看到 `threshold_artifact` 列，当前状态合理显示为 `missing_output`。

### 4.4 失败路径 smoke

已验证：当 artifact 文件存在但不是合法 JSON 时，calibration runner 会退出，不会静默跳过或覆盖。

观察到退出码：

```text
3
```

关键报错：

```text
artifact exists but is not valid JSON
```

### 4.5 GPU 与 tmux 状态

执行前检查：

```bash
nvidia-smi
tmux ls
```

观察：A100 GPU 当前有一个 Python 进程占用约 44.9GB 显存。本次没有启动训练或 calibration run，因此没有新增 GPU 任务。

## 5. Code Review 结果

### 5.1 CodeRabbit

尝试执行：

```bash
coderabbit review --agent -t uncommitted --dir /root/2TStorage/lyc/SegACIL
```

状态：

```text
CodeRabbit 进入 reviewing 后持续 heartbeat，数分钟未返回 findings。
已手动中断，避免留下悬挂审查进程。
```

因此本次 CodeRabbit 没有给出可操作 findings。

### 5.2 人工严格审查

已检查并修正：

- 新增 calibration runner 原本只跳过第一行，缺少 header 校验；已补 header 校验。
- `run_pseudo_label_grid.sh` 支持旧/新 TSV，但需要显式拒绝错位 header；已补 header 校验。
- `artifact_class` 缺 artifact 的情况不能等 Trainer 再报错；runner 已提前失败。
- `threshold_max_batches=0` 与 artifact 生成时 `max_batches=0` 保持一致，避免 Trainer 加载 artifact 时 metadata mismatch。
- summary 工具对旧 TSV 补空字段，不破坏 Phase A 已有 grid。
- 所有新增正式结果目录使用独立 `SUBPATH`，不会覆盖既有 checkpoint。

未发现会影响训练比较公平性的代码改动：本次只补执行链、配置和 summary，不改变 pseudo-label 生成算法、Trainer 损失、Dataset、模型结构或评估逻辑。

## 6. 下一步启动方式

正式启动前建议再次检查 GPU 显存：

```bash
nvidia-smi
```

然后先生成 full artifacts：

```bash
cd /root/2TStorage/lyc/SegACIL
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass \
TMPDIR=/root/2TStorage/tmp \
bash tools/run_pseudo_label_artifact_calibration_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_calibration.tsv \
  --mode run
```

确认三个 artifact JSON 生成且合法后，再启动训练：

```bash
cd /root/2TStorage/lyc/SegACIL
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass \
TMPDIR=/root/2TStorage/tmp \
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_phaseB_artifact_train.tsv \
  --mode run
```

若用 tmux 后台执行，建议会话名：

```text
apl_phaseB_artifact
```

## 7. 判定标准

当前 Phase B 必须对比的强 baseline：

| baseline | all mIoU |
| --- | ---: |
| `fixed0.6` | 0.707731 |
| `batch_class q0.1` | 0.707694 |
| `fixed0.7` | 0.707383 |
| `off` | 0.703091 |

判断：

- `artifact_class > 0.707731`：artifact 自适应阈值路线成立，进入 15-5 disjoint / 15-1 overlap。
- `artifact_class` 接近但未超过 `fixed0.6`：需要 raw-mask audit 判断原因，再决定 margin/cap/weighted。
- `artifact_class < 0.707383`：先不扩大同类训练，优先做质量审计或降级为消融。
