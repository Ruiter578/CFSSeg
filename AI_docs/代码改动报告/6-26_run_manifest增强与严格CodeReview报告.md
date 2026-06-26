# 6-26 run_manifest 增强与严格 Code Review 报告

## 1. 背景

本次改动目标是增强 DeepLabV3+ 主线融合后已有的 `run_manifest.json` 机制，使其覆盖此前 `args.yaml` 方案中的核心需求：

1. 自动记录完整 `Config` 参数。
2. 记录启动命令、主机名、Git commit/dirty 状态和运行时环境。
3. 记录输出目录、base checkpoint 路径和 SHA256。
4. manifest 写入失败不影响训练。
5. 同一输出目录被不同配置复用时，不覆盖原始 manifest。
6. 保持 `.pth` 和 TensorBoard event 忽略，只同步轻量 JSON。

## 2. 代码改动范围

### 2.1 `utils/run_manifest.py`

主要增强：

1. 新增 `schema_version: 2`。
2. 新增 `args` 分区，来源为完整参数对象：

```python
args = normalize_for_json(options_to_dict(opts))
```

`Config` dataclass 使用 `dataclasses.asdict()`；普通对象或 `argparse.Namespace` 风格对象使用 `vars()`。

3. 新增 `command`，记录 `sys.argv`。
4. 新增 `hostname`，记录 `socket.gethostname()`。
5. 新增 `git` 分区：

```json
{
  "commit": "...",
  "dirty": true
}
```

`git status --short` 增加 `timeout=5`，避免 manifest 写入阶段被 Git 命令卡住。

6. 新增 `runtime` 分区，记录：

```text
python
pytorch
cuda_available
cuda_visible_devices
```

7. 新增 `resolved_paths` 分区，记录：

```text
output_dir
base_checkpoint_path
base_checkpoint_sha256
```

8. 新增 `air` 分区，记录 requested/resolved AIR feature source。
9. 保留旧版 flat keys，例如 `model`、`buffer`、`git_commit`、`base_checkpoint_sha256`，避免旧报告和临时脚本立刻失效。
10. 新增 `safe_write_run_manifest()` wrapper。训练流程调用 wrapper；写入失败只打印 warning 并返回 `None`。
11. 同目录不同配置复跑时，保留原 `run_manifest.json`，新文件写为：

```text
run_manifest_YYYYMMDD_HHMMSS_microseconds.json
```

12. 原子写入改为 `tempfile.NamedTemporaryFile(delete=False)`，每次写入使用唯一临时文件，避免并发 writer 争用同一个 `.run_manifest.json.tmp`。

### 2.2 `trainer/trainer.py`

将原来的：

```python
from utils.run_manifest import write_run_manifest
```

替换为：

```python
from utils.run_manifest import safe_write_run_manifest
```

并将 step0、step1、step2+ 三处 manifest 写入调用都切换为 `safe_write_run_manifest()`。

这保证 manifest 只是旁路元数据功能，不会因为磁盘、权限、Git metadata 或 checkpoint hash 异常中断训练。

### 2.3 `tests/test_run_manifest.py`

新增和扩展测试：

1. `current_git_dirty()` 使用 `timeout=5`。
2. Git dirty 探测 timeout 时返回 `None`。
3. `manifest["args"]` 包含 `Config` 的所有 dataclass 字段。
4. `command`、`hostname`、`git`、`runtime`、`resolved_paths`、`air` 分区存在且内容正确。
5. `CUDA_VISIBLE_DEVICES` 在测试中用 `patch.dict(os.environ, ...)` 固定，避免环境敏感。
6. 支持 plain namespace/options 对象。
7. `safe_write_run_manifest()` 写入失败时返回 `None`。
8. 同目录不同配置复跑时生成带时间戳的第二份 manifest，不覆盖原文件。

## 3. `.gitignore` 跟踪确认

当前 `.gitignore` 规则：

```gitignore
checkpoints/**
!checkpoints/**/
!checkpoints/**/sequential/step*/**/*.json
```

实际验证命令：

```bash
git check-ignore -v \
  checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/run_manifest.json \
  checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/final.pth \
  checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/events.out.tfevents.1782412307.master-192-168-8-48
```

验证结果：

```text
.gitignore:60:!checkpoints/**/sequential/step*/**/*.json    .../run_manifest.json
.gitignore:58:checkpoints/**                                .../final.pth
.gitignore:58:checkpoints/**                                .../events.out.tfevents.1782412307.master-192-168-8-48
```

结论：

1. `run_manifest.json` 能被 Git 发现并同步。
2. `test_results_*.json` 能被 Git 发现并同步。
3. `.pth` 权重仍被忽略。
4. TensorBoard event 文件仍被忽略。

## 4. Code Review 处理

### 4.1 第一轮 CodeRabbit

命令：

```bash
coderabbit review --agent -t uncommitted
```

第一轮返回 4 条 finding：

| 严重级别 | 文件 | 问题 | 处理 |
| --- | --- | --- | --- |
| major | `utils/run_manifest.py` | `git status --short` 无 timeout，可能卡住 manifest 写入 | 加 `timeout=5`，捕获 `subprocess.TimeoutExpired` |
| major | `utils/run_manifest.py` | `asdict(opts)` 假设 opts 必然是 dataclass | 新增 `options_to_dict()`，支持 dataclass 和 plain namespace |
| major | `utils/run_manifest.py` | 原子写入使用共享 temp 文件名，并发 writer 可能冲突 | 改为 `tempfile.NamedTemporaryFile(delete=False)` 唯一临时文件 |
| minor | `tests/test_run_manifest.py` | 测试依赖外部 `CUDA_VISIBLE_DEVICES` 环境 | 使用 `patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"})` 固定 |

### 4.2 第二轮 CodeRabbit

修复后复跑：

```bash
coderabbit review --agent -t uncommitted
```

结果：

```json
{"type":"complete","status":"review_completed","findings":0}
```

结论：第二轮未发现新的 Critical、Warning 或 Info finding。

## 5. 本地验证

### 5.1 单文件 manifest 测试

命令：

```bash
PYTHONPATH=. python3 tests/test_run_manifest.py
```

结果：

```text
Ran 8 tests in 0.488s
OK
```

### 5.2 全量测试发现

命令：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
```

结果：

```text
Ran 21 tests in 4.727s
OK
```

### 5.3 编译检查

命令：

```bash
PYTHONPATH=. python3 -m py_compile \
  utils/run_manifest.py \
  trainer/trainer.py \
  tests/test_run_manifest.py
```

结果：通过。

### 5.4 空白检查

命令：

```bash
git diff --check
```

结果：通过。

## 6. 行为边界

本次改动不改变：

1. checkpoint schema。
2. 模型结构。
3. dataloader。
4. optimizer/scheduler。
5. 随机种子设置顺序。
6. AIR/C-RLS 训练逻辑。
7. 评测逻辑。

本次改动只影响：

1. `run_manifest.json` 的内容结构和完整性。
2. manifest 写入失败时的容错行为。
3. 同目录复跑时 manifest 的保留策略。

## 7. 后续建议

1. 之后新实验优先查看 `run_manifest.json`，而不是另行维护 `args.yaml`。
2. 如果为了和 OFQ 展示风格统一需要 `args.yaml`，应从 `run_manifest.json` 派生生成，不能重复采集参数。
3. 若后续新增 `Config` 字段，只需要更新训练逻辑和 parser；manifest 会通过 `args` 自动记录新字段。
