# SegACIL `run_manifest.json` / `args.yaml` 实验参数记录机制方案

更新日期：2026-06-26

## 1. 当前结论

主线在 DeepLabV3+ 融合后已经新增了一个和原 `args.yaml` 需求高度相近的机制：

```text
utils/run_manifest.py
trainer/trainer.py
tests/test_run_manifest.py
```

新实验会在 step 输出目录生成：

```text
checkpoints/<subpath>/<dataset>/<task>/<setting>/stepN/run_manifest.json
```

例如本次实验已经生成：

```text
checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/run_manifest.json
```

因此，不建议再并行新增一套功能重叠的 `args.yaml` 采集机制。后续应以 `run_manifest.json` 作为权威实验元数据 sidecar，并在现有实现上补齐原需求。

如果将来确实需要 OFQ 风格的 `args.yaml`，应从同一份 `run_manifest.json` 派生生成，不能另写一套参数采集逻辑，否则容易出现 `args.yaml` 和 `run_manifest.json` 不一致。

## 2. 代码出处

### 2.1 写入工具

`utils/run_manifest.py` 定义：

```python
def write_run_manifest(
    output_dir,
    opts,
    requested_air_feature_source,
    resolved_air_feature_source,
    base_checkpoint_path=None,
    git_commit=None,
):
```

它当前会：

1. 创建输出目录。
2. 计算 base checkpoint 的 SHA256。
3. 读取当前 Git commit。
4. 组装 manifest 字典。
5. 写入临时文件 `.run_manifest.json.tmp`。
6. 用 `replace()` 原子替换为 `run_manifest.json`。

### 2.2 调用位置

`trainer/trainer.py` 中从 `utils.run_manifest` 导入：

```python
from utils.run_manifest import write_run_manifest
```

当前主要调用点：

1. `Trainer.__init__` 中 `curr_step == 0`：初始化 step0 SGD 模型后写 step0 manifest。
2. `Trainer.train()` 中 `curr_step == 1`：加载 step0 checkpoint，解析 AIR feature source 后写 step1 manifest。
3. `Trainer.train()` 中 `curr_step > 1`：加载已有 AIR checkpoint，校验/解析 feature source 后写当前 step manifest。

### 2.3 测试覆盖

`tests/test_run_manifest.py` 已覆盖：

1. Git metadata 不可用时 `current_git_commit()` fallback 为 `unknown`。
2. `make_step0_loader_opts()` 深拷贝 step0 dataloader 参数，不污染原始 `opts.curr_step`。
3. manifest 记录 model、AIR source、RHL、base checkpoint hash 和若干关键训练参数。

对应提交来自 DeepLabV3+ 主线融合链：

```text
5d56d8c feat: record complete experiment manifests
85dc571 fix: validate AIR resume metadata and harden manifests
```

## 3. 当前实际文件内容

本次实验的 `run_manifest.json` 已记录：

```text
created_at_utc
git_commit
model
requested_air_feature_source
resolved_air_feature_source
data_root
dataset
task
setting
curr_step
num_classes
target_cls
subpath
base_subpath
base_checkpoint_path
base_checkpoint_sha256
batch_size
val_batch_size
crop_size
crop_val
output_stride
pretrained_backbone
bn_freeze
separable_conv
method
loss_type
lr
lr_policy
train_epoch
weight_decay
buffer
gamma
random_seed
rhl_norm
rhl_norm_eps
rhl_seed
rhl_stats
use_pseudo_label
pseudo_label_confidence
```

这已经覆盖实验复盘中最关键的路径、代码版本、base checkpoint、模型、AIR source、buffer、gamma、RHL 和训练配置。

## 4. 与原 `args.yaml` 需求的符合度

当前机制符合的部分：

1. 是旁路元数据文件，不参与训练恢复。
2. 不修改 `.pth` checkpoint schema。
3. 使用标准库 JSON，不依赖 PyYAML。
4. 写在 step 输出目录，便于和 `test_results_*.json` 一起同步。
5. 已记录 base checkpoint hash，能避免 step1 来源不清。
6. 已记录 requested/resolved AIR feature source，能区分 V3 和 V3+ 的 AIR 特征来源。
7. 已有单元测试。

当前机制尚未完全符合的部分：

| 原需求 | 当前状态 | 影响 |
| --- | --- | --- |
| 自动记录 `Config` 全部字段 | 当前手工逐项复制字段 | 新增参数后容易漏记 |
| 新增 Config 字段自动进入记录 | 不满足 | 需要同步维护 `run_manifest.py` |
| 记录真实启动命令 | 未记录 `sys.argv` | 不能完整还原 shell 覆盖方式 |
| 记录主机名 | 未记录 | 双服务器同步后来源机器不够直观 |
| 记录 Git dirty 状态 | 只记录 commit | 不能判断实验是否来自未提交代码 |
| 记录运行时环境 | 未记录 Python/PyTorch/CUDA_VISIBLE_DEVICES | 环境差异排查能力不足 |
| 记录显式 output_dir | 未记录，只能从文件位置推断 | 单独查看文件时上下文较弱 |
| 写入失败不影响训练 | 当前没有统一非致命 wrapper | 权限、磁盘或 hash 异常可能中断训练 |
| 重跑同一目录保留多份 manifest | 当前静默覆盖 | 同目录不同配置复跑时证据可能丢失 |

此外，当前 `utils/parser.py::Config` 中至少这些字段没有进入 manifest：

```text
test_only
curr_itrs
step_size
ckpt
unknown
print_interval
val_interval
gpu_id
local_rank
overlap
cil_step
initial
air_feature_source
```

其中 `air_feature_source` 的信息以 `requested_air_feature_source` 形式保存了，但如果目标是“完整保存 Config”，仍应进入 `args` 分区。

## 5. `.gitignore` 状态

当前 `.gitignore` 规则为：

```gitignore
checkpoints/**
!checkpoints/**/
!checkpoints/**/sequential/step*/**/*.json
```

这意味着：

1. `test_results_*.json` 会被 Git 发现。
2. `run_manifest.json` 也会被 Git 发现。
3. `.pth` 权重仍被忽略。
4. `events.out.tfevents.*` 仍被忽略。

本次实验目录中实际会进入 Git 的文件是：

```text
checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step0/test_results_20260625_210048.json
checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/run_manifest.json
checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/test_results_20260625_213931.json
```

同目录下的：

```text
final.pth
events.out.tfevents.*
```

不会进入 Git。这符合“双服务器轻量同步结果”的目标。

如果未来确实生成 YAML 派生文件，再追加：

```gitignore
!checkpoints/**/sequential/step*/args.yaml
!checkpoints/**/sequential/step*/args_*.yaml
!checkpoints/**/sequential/step*/args.reconstructed.yaml
```

## 6. 推荐目标结构

建议把当前扁平 JSON 升级为分区结构，同时保留向后可读性：

```json
{
  "schema_version": 2,
  "created_at_utc": "2026-06-26T10:05:10Z",
  "hostname": "master-192-168-8-48",
  "command": [
    "train.py",
    "--data_root",
    "/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
  ],
  "git": {
    "commit": "<git sha>",
    "dirty": false
  },
  "runtime": {
    "python": "3.x.x",
    "pytorch": "2.x.x",
    "cuda_available": true,
    "cuda_visible_devices": "2"
  },
  "resolved_paths": {
    "output_dir": "checkpoints/.../step1",
    "base_checkpoint_path": "checkpoints/.../step0/deeplabv3_resnet101_...pth",
    "base_checkpoint_sha256": "<sha256>"
  },
  "air": {
    "requested_feature_source": "auto",
    "resolved_feature_source": "decoder"
  },
  "args": {
    "dataset": "voc",
    "task": "15-5",
    "setting": "sequential",
    "curr_step": 1,
    "batch_size": 32,
    "buffer": 8208,
    "gamma": 1.0,
    "rhl_norm": "none",
    "rhl_norm_eps": 1e-6,
    "random_seed": 1
  }
}
```

关键要求：

```python
from dataclasses import asdict

manifest["args"] = asdict(opts)
```

以后只要新参数进入 `Config`，它就自动进入 `run_manifest.json`。

## 7. 推荐代码改动

当前不需要新增 `utils/experiment_args.py`。应增强现有 `utils/run_manifest.py`。

### 7.1 字段采集

将手工字段列表替换为：

```python
from dataclasses import asdict

args = normalize_for_json(asdict(opts))
```

保留现有关键字段，但移动到分区中：

1. `args`: 完整 Config。
2. `air`: requested/resolved feature source。
3. `resolved_paths`: output_dir、base_checkpoint_path、base_checkpoint_sha256。
4. `git`: commit、dirty。
5. `runtime`: Python、PyTorch、CUDA。
6. `command`: `sys.argv`。
7. `hostname`: `socket.gethostname()`。

### 7.2 非致命写入

新增安全 wrapper，例如：

```python
def safe_write_run_manifest(*args, **kwargs):
    try:
        return write_run_manifest(*args, **kwargs)
    except Exception as exc:
        print(f"[warning] failed to write run_manifest.json: {exc}")
        return None
```

训练主流程调用 wrapper，保证 manifest 失败不影响训练。

### 7.3 重跑同目录策略

同一个 step 目录重复运行时：

1. `run_manifest.json` 不存在：直接写。
2. 已存在且内容等价：保持原文件或覆盖为等价内容均可。
3. 已存在但配置不同：保留原文件，新增：

```text
run_manifest_YYYYMMDD_HHMMSS.json
```

这样可以保留同目录复跑证据。

### 7.4 可选 YAML 派生

如果为了和 OFQ 视觉习惯一致需要 `args.yaml`，建议仅做派生输出：

```text
run_manifest.json   # 权威来源
args.yaml           # 从 manifest["args"] 派生的人类阅读副本
```

不要让 `args.yaml` 和 `run_manifest.json` 分别采集参数。

## 8. 测试方案

更新 `tests/test_run_manifest.py`：

1. 构造 `Config`，断言所有 `dataclasses.fields(Config)` 都存在于 `manifest["args"]`。
2. 临时给测试 Config 增加或模拟一个字段，确认 writer 不需要手工更新字段名单。
3. 断言 `command`、`hostname`、`git.commit`、`git.dirty`、`runtime.python` 存在。
4. 断言 base checkpoint SHA256 仍正确。
5. mock 写入异常，确认 safe wrapper 返回 `None` 且不抛出。
6. mock dirty worktree，确认 dirty 状态记录正确。

集成 smoke test：

1. step0 run 生成 `step0/run_manifest.json`。
2. step1 run 生成 `step1/run_manifest.json`。
3. step1 manifest 中 `args.curr_step == 1`。
4. step1 manifest 中 base checkpoint path 和 SHA256 指向真实 step0 checkpoint。
5. `git status --short` 只出现结果 JSON/manifest JSON，不出现 `.pth` 或 event 文件。

## 9. 历史结果重建策略

不能把所有旧目录直接无损转换成权威 `run_manifest.json`。旧的结果 JSON、event 和 checkpoint 没有完整保存 `Config`、命令行、dirty 状态和运行时环境。

高可信历史目录可以生成：

```text
run_manifest.reconstructed.json
```

但必须包含：

```json
{
  "reconstructed": true,
  "confidence": "high",
  "evidence": [
    "historical shell",
    "log Config(...)",
    "checkpoint path"
  ],
  "unknown_fields": []
}
```

中低可信目录不能伪装成原始 `run_manifest.json`，只能生成带 `reconstructed: true` 的重建文件，并列出未知字段。

## 10. 实施顺序

1. 增强 `utils/run_manifest.py`，使用 `dataclasses.asdict(opts)` 作为完整 args 来源。
2. 补充 command、hostname、git dirty、runtime、output_dir。
3. 增加非致命写入 wrapper。
4. 增加同目录复跑冲突保留策略。
5. 更新 `tests/test_run_manifest.py`。
6. 做一次 step0/step1 smoke test。
7. 如果确实需要，再从 `run_manifest.json` 派生 `args.yaml`。

## 11. 接受标准

1. 新实验自动生成 `run_manifest.json`。
2. 新增 `Config` 字段无需修改 writer，也会自动进入 `manifest["args"]`。
3. manifest 记录失败不影响训练。
4. 不改 checkpoint schema。
5. 不改变模型、数据、随机种子、优化器、训练循环和评测结果。
6. Git 只同步结果 JSON 和 manifest JSON，不同步 `.pth`、events 或日志。
7. 历史重建文件明确标注证据、置信度和未知字段。
