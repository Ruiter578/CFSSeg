# SegACIL 当前代码库审查报告

> 日期：2026-06-17  
> 范围：当前 `feature/rhl-se-boa-p0-p1` 工作区，重点审查本轮 P0/P1 改动与原 SegACIL 训练、评估、脚本、checkpoint 链路。  
> 说明：以下“建议调整”未直接修改，按用户要求只在报告中阐述。

## 1. 总体结论

本轮新增的 RHL-SE 收尾和 BOA-RHL 实现边界清晰：新增参数进入 CLI、AIR、RandomBuffer、runner 和 checkpoint config；默认 `gaussian + legacy` 保持原 baseline 初始化行为；BOA 不改 C-RLS。当前代码可以进入实验启动阶段。

代码库整体仍有几个会影响实验可复现和报告可信度的风险点，主要集中在脚本路径、快速 smoke 能力、配置对象可变性和评估工具边界。

## 2. Findings

### Warning 1：`run_origin.sh` 仍硬写 baseline 输出目录，容易覆盖历史结果

位置：`run_origin.sh:14`

当前工作区中的 `run_origin.sh` 写死：

```text
SUBPATH="20260606"
```

这会把 step1 结果写回已知 baseline 树，和之前记忆中的风险一致。建议继续把 `run_origin.sh` 视为历史脚本，不用于新实验；新实验使用 `tools/run_boa_rhl.sh` 或专用 runner。

### Warning 2：当前工作区 `run.sh` / `run_origin.sh` 未提交且与本轮实验默认值不完全一致

位置：`run.sh:21`、`run_origin.sh:19`

当前脏工作区里两个脚本都是 `BUFFER=8192`，而 RHL-SE 文档和本轮 BOA runner 使用 `BUFFER=8196`。如果混用脚本，会造成 baseline 与 BOA/RHL-SE 不可比。建议后续明确统一 buffer，或在报告中标注每次实验的 buffer 来源。

### Warning 3：训练入口缺少快速 batch 限制参数，难做标准 train smoke

位置：`utils/parser.py`、`trainer/trainer.py:217-346`

AGENTS 规范建议训练改动后用 `--max-train-batches 1 --max-val-batches 1` 跑 1 epoch，但当前 CLI 没有这些参数。结果是每次训练验证都接近完整实验，只能用 AIR dummy smoke 替代。建议后续加：

```text
--max_train_batches
--max_val_batches
--max_test_batches
```

并在 train/eval 循环里统一生效。

### Warning 4：`Trainer` 仍依赖可变 `opts` 在 step1 内切换语义

位置：`trainer/trainer.py:289-300`

本轮已经修复 checkpoint config 被 `curr_step=0` 污染的问题，但 step1 分支仍通过临时改 `self.opts.curr_step` 来构造 step0 dataloader。这个模式可用但脆弱。建议后续改为复制一个 `realign_opts`，而不是直接修改共享 `self.opts`。

### Warning 5：未知 dataset 的 metrics 分支没有真正抛异常

位置：`metrics/stream_metrics.py:62-67`

当前代码：

```python
else:
    NotImplementedError
```

这不会 raise。若传入未知 dataset，后续可能在 `self.CLASSES` 缺失处才报错。建议改成：

```python
raise NotImplementedError(f"Unsupported dataset: {dataset}")
```

### Info 1：checkpoint 使用完整模型 pickle，适合本地可信产物，不适合加载外部不可信文件

位置：`utils/ckpt.py:4-17`

`save_ckpt` 保存 `model_architecture`，`load_ckpt` 直接 `torch.load()`。这符合当前本地实验复现方式，但不适合加载外部来源 checkpoint。建议报告和脚本中继续只使用本项目可信 checkpoint。

### Info 2：评估脚本已正确规避 test 调参，但默认会跑完整 split

位置：`tools/search_rhl_class_weights.py`、`tools/eval_rhl_ensemble.py`

两个工具已有 `--max_batches`，适合 debug；正式路径由 `tools/run_rhl_se_val_driven.sh` 固化为 val 搜权重、test 单次消费权重。建议后续实验报告必须保存 runner 生成的 `run_summary.md`，避免只凭命令历史复原。

## 3. 本轮实现准确性复核

- P0：没有新增新的调权逻辑，只把已有 val-driven 搜索和 test 评估串成可复现 runner，符合“收尾”定位。
- P1：BOA-RHL 只改 RHL 固定映射，未改 `RecursiveLinear`，符合方案要求。
- 参数链路：`utils/parser.py` -> `trainer/trainer.py::AIR` -> `network/Buffer.py::RandomBuffer` 已打通。
- 配置记录：`run_config.json` 和 checkpoint `training_config` 已包含新增 RHL 参数。
- 输出路径：BOA runner 每个 case 独立 `SUBPATH`，并用 batch size 写入路径，降低覆盖风险。

## 4. 后续建议

1. 先跑 BOA-0/1/2/3，保留 batch size、subpath、日志和最终 `test_results_*.json`。
2. 若 BOA 任一单模型达到 `+0.10 all` 或 `+0.30 new`，再考虑 BOA 多 seed 或 BOA + RHL-SE。
3. 若 BOA 只改善 sparsity / Gram 统计但 mIoU 不动，优先分析 gamma 相对尺度，不要立即扩散到更多 scale。
4. 在下一轮代码工作中补 train/eval 的 `max_*_batches` 调试参数，提高每次 agent 改动后的验收质量。

