# SegACIL `args.yaml` 实验参数记录机制实现方案

更新日期：2026-06-23

## 1. 结论

该功能可行，代码复杂度低，可以实现为纯旁路记录功能，不改变 SegACIL 的模型、数据加载、优化器、AIR/C-RLS、SEEQ/RHL 或评测机制。

推荐实现边界：

1. 每个 `checkpoints/<subpath>/<dataset>/<task>/<setting>/stepN/` 目录生成一份 `args.yaml`。
2. 自动记录 `Config` 中全部字段，不手写参数名单。
3. 同时记录启动命令、Git commit、工作区是否 dirty、主机名、时间和 Python/PyTorch 版本。
4. 记录失败只打印 warning，不阻断训练。
5. 不修改 checkpoint 格式，不把参数塞进 `.pth`，避免影响旧模型加载。

建议工作量：新增 1 个小工具模块、修改 `trainer/trainer.py` 1 个调用点、增加 1 组轻量测试。实现量约 80-140 行。

## 2. 为什么应该记录

当前项目只在 `train.py` 启动时执行：

```python
print(opts)
```

如果标准输出没有重定向到日志，这些参数不会进入结果目录。目前的：

```text
test_results_*.json
```

只包含评测指标；TensorBoard event 文件只包含 `loss`、`miou` 等 summary；checkpoint 只保存：

```text
model_state
model_architecture
optimizer_state（部分 checkpoint）
best_score（部分 checkpoint）
```

因此仅保留 JSON 和 `.pth` 不能可靠回答 batch size、seed、buffer、gamma、RHL 配置、启动命令和代码版本。

## 3. 文件位置和生命周期

现有结果路径由 `Trainer` 统一构造：

```python
self.root_path = (
    f"checkpoints/{opts.subpath}/{opts.dataset}/"
    f"{opts.task}/{opts.setting}/step{opts.curr_step}/"
)
```

推荐将参数文件写入：

```text
checkpoints/<subpath>/<dataset>/<task>/<setting>/stepN/args.yaml
```

每个 step 单独记录的理由：

1. `curr_step`、`batch_size`、`base_subpath` 可能不同。
2. step0 是 SGD 训练，step1+ 是 AIR 闭式更新，运行语义不同。
3. 一个 shell 循环会多次启动 `python train.py`，每次进程都应留下自己的已解析参数。

## 4. 推荐记录结构

```yaml
schema_version: 1
created_at_utc: "2026-06-23T20:05:10Z"
hostname: "master-192-168-8-48"
command:
  - "train.py"
  - "--data_root"
  - "/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
  - "--model"
  - "deeplabv3_resnet101"
git:
  commit: "<git sha>"
  dirty: false
runtime:
  python: "3.x.x"
  pytorch: "2.x.x"
  cuda_visible_devices: "2"
resolved_paths:
  output_dir: "checkpoints/.../step1"
  previous_checkpoint: "checkpoints/.../step0/deeplabv3_resnet101_...pth"
args:
  dataset: "voc"
  task: "15-5"
  setting: "sequential"
  curr_step: 1
  batch_size: 32
  buffer: 8216
  gamma: 1.0
  rhl_norm: "none"
  rhl_norm_eps: 1.0e-6
  random_seed: 1
```

`args` 节必须来自 `dataclasses.asdict(opts)`，不能手工逐项复制。以后只要新参数进入 `Config`，它就自动进入 `args.yaml`。

## 5. 不引入 PyYAML 依赖

当前 SegACIL conda 环境没有 PyYAML。为了不增加启动依赖，推荐使用标准库 `json` 生成 YAML 1.2 兼容内容：

```python
for key, value in payload.items():
    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
```

更完整的实现可以递归输出缩进结构，但核心原则是：

```text
只使用 Python 标准库
不因为缺少 yaml 包导致训练无法启动
```

JSON 本身也是 YAML 1.2 的兼容子集；也可以把完整 JSON 对象写入名为 `args.yaml` 的文件，TensorBoard/训练逻辑不会读取该文件。

如果后续明确愿意维护新依赖，再切换到 `yaml.safe_dump()`。

## 6. 推荐代码结构

新增：

```text
utils/experiment_args.py
```

建议接口：

```python
def write_experiment_args(
    output_dir: str,
    opts: Config,
    previous_checkpoint: str | None = None,
) -> str | None:
    """写入旁路实验元数据；I/O 失败时警告并返回 None。"""
```

内部步骤：

1. `dataclasses.asdict(opts)` 获取全部已解析参数。
2. 递归规范化 `Path`、tuple、NumPy 标量等潜在类型。
3. 用 `sys.argv` 保存真实启动参数。
4. 用 `socket.gethostname()`、`datetime` 保存来源和时间。
5. 用非致命 `subprocess.run()` 获取 Git SHA 和 dirty 状态。
6. 先写同目录临时文件，再 `os.replace()`，保证原子落盘。
7. 只捕获 `OSError` 和元数据采集异常，打印清楚 warning。

`trainer/trainer.py` 中只增加一次调用。推荐位置：

```python
mkdir(self.root_path)
# 完成 self.ckpt / previous checkpoint 解析后
write_experiment_args(
    output_dir=self.root_path,
    opts=self.opts,
    previous_checkpoint=getattr(self, "ckpt", None),
)
```

必须在 `Trainer.train()` 之前写入，因为 step1 训练逻辑会暂时把：

```python
self.opts.curr_step = 0
```

如果写得太晚，记录到的 step 会错误。

## 7. 重跑同一目录的处理策略

同一个 `subpath + step` 被重复运行时，不能静默覆盖不同配置。

推荐规则：

1. `args.yaml` 不存在：直接写入。
2. 已存在且配置内容一致：保持原文件，打印“配置一致”。
3. 已存在但配置不同：保留原 `args.yaml`，新增：

```text
args_20260623_200510.yaml
```

并打印 warning，提示同一结果目录发生不同配置的复用。

这个策略不会中止训练，但能保留证据。

## 8. 为什么不写入 checkpoint

不推荐修改 `utils/ckpt.py::save_ckpt()`，原因：

1. step0 和 AIR checkpoint 的结构已经被现有加载代码依赖。
2. `.pth` 很大且不纳入 Git，无法解决双服务器轻量同步问题。
3. 改 checkpoint schema 会扩大兼容性和测试范围。
4. 独立 `args.yaml` 更容易查看、diff 和 Git 管理。

因此 `args.yaml` 应是旁路元数据，不参与训练恢复。

## 9. `.gitignore` 配套规则

功能实现后，需要在当前 checkpoint 例外规则后增加：

```gitignore
!checkpoints/**/sequential/step*/args.yaml
!checkpoints/**/sequential/step*/args_*.yaml
```

模型权重、TensorBoard events 和其他大文件继续忽略。

## 10. 测试方案

### 10.1 单元测试

构造 `Config`，检查：

1. 所有 dataclass 字段都进入文件。
2. `None`、布尔、列表、字典、浮点值可正确输出。
3. 新增一个临时 Config 字段时无需修改 writer。
4. Git 命令失败时仍能生成基本 args。
5. 输出目录不可写时返回 warning，不影响调用方。

### 10.2 集成 smoke test

使用不会正式训练的短命令或 mock Trainer 路径，确认：

1. `step0/args.yaml` 中 `curr_step: 0`。
2. `step1/args.yaml` 中 `curr_step: 1`。
3. step1 记录正确的 `base_subpath` 和 resolved previous checkpoint。
4. `git status --short` 只显示 args/JSON，不出现 `.pth` 和 event 文件。

### 10.3 不变性检查

实现前后用同一参数运行 smoke test，确认：

1. dataloader 数量不变。
2. model state 初始化不变。
3. optimizer/scheduler 不变。
4. 随机种子设置顺序不变。
5. 训练和评测输出不读取 `args.yaml`。

## 11. 当前历史结果的可追溯性

不能把所有历史目录直接、无损地转换成权威 `args.yaml`。原因是结果 JSON、event 和 checkpoint 都没有完整保存 `Config`。

### 11.1 高可信，可生成重建文件

```text
20260613_v3plus_voc15-5_seq_bs16_trs
20260613_v3plus_voc15-5_seq_bs32_trs
```

对应日志包含完整 `Config(...)` 打印，可解析出当时参数。建议生成：

```text
args.reconstructed.yaml
```

并注明来源日志和 `reconstructed: true`。其中 bs32 日志显示该次运行 OOM，目录不应误标成成功实验。

```text
0615_step1_bs16_trs
```

Git 历史中的 `run_origin.sh` 与该 subpath 精确匹配，能高可信恢复启动脚本参数；但环境包版本和临时命令覆盖仍未被结果文件直接保存，也应标记为 reconstructed。

当前仍在运行的：

```text
20260622_buffer8216_step1_32_run2
```

可以从当前进程命令行保存本次真实 CLI；训练结束并重命名为 `_trs` 后，可生成高可信重建文件。

### 11.2 中低可信，只能部分推断

```text
1128_trs
20260621_baseline_bs16_16_trs
20260622_trs
20260622_buffer8216_step1_32_trs
```

可从目录结构、模型对象、JSON、当前/历史 shell 和文档推断 dataset、task、setting、step、部分 batch/buffer 信息，但不能证明所有参数和当时 parser 默认值。

这些目录不应直接生成看起来像原始事实的 `args.yaml`。如果确有需要，应生成：

```text
args.reconstructed.yaml
```

并包含：

```yaml
reconstructed: true
confidence: "partial"
evidence:
  - "checkpoint path"
  - "historical shell"
unknown_fields:
  - "..."
```

## 12. 实施顺序

1. 新增 `utils/experiment_args.py`。
2. 为 writer 添加单元测试。
3. 在 `Trainer.__init__` 完成路径解析后调用一次。
4. 做 step0/step1 smoke test。
5. 更新 `.gitignore` 放行 `args*.yaml`。
6. 对高可信历史结果生成 `args.reconstructed.yaml`，不伪装成原生记录。

## 13. 接受标准

1. 新实验每个 step 自动生成参数文件。
2. 新增 Config 字段自动进入记录。
3. 参数记录失败不影响训练。
4. 不改 checkpoint schema。
5. 不改变模型、数据、随机种子、优化器、训练循环和评测结果。
6. Git 只同步 JSON 和 args YAML，不同步 `.pth`、events 或日志。
7. 历史重建文件明确标注证据和置信度。
