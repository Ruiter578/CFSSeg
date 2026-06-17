# P0+P1 RHL-SE / BOA-RHL 代码实现与审查报告

> 日期：2026-06-17  
> 分支：`feature/rhl-se-boa-p0-p1`  
> 基线提交：`a2cbc4e`  
> 本轮提交：`e79dcd2`、`4437b3d`、`8e0861f`

## 1. 执行范围

本轮按 `AI_docs/idea构思与实验设计/RHL新方案/6-16_未实现三方案攻击审查与重构执行方案.md` 修订后的 5.2 执行 P0 和 P1。

P0 的目标是把 RHL-SE 的 val-driven class-wise 收尾固化成可复现入口，避免继续基于 test 手工调权重。P1 的目标是实现 BOA-RHL：新增 RHL 初始化族和 scale mode，在不改 C-RLS 公式的前提下改变固定随机特征构造。

## 2. 提交清单

| 提交 | 内容 |
|---|---|
| `e79dcd2 docs: finalize rhl se execution plan` | 修订执行方案文档；新增 `tools/run_rhl_se_val_driven.sh` |
| `4437b3d feat: add boa rhl initialization` | 新增 BOA-RHL 参数、`RandomBuffer` 初始化、RHL 统计、checkpoint config、测试和 runner |
| `8e0861f fix: preserve step config metadata` | 修复 step1 配置快照被 realignment 临时 `curr_step=0` 污染的问题 |

## 3. P0 实现审查

已更新 `6-16_未实现三方案攻击审查与重构执行方案.md`，删除“最小任务 / 前置条件”式串行防御规划，明确 P0 与 P1 同轮推进。新的 P0 定位是：RHL-SE 作为辅助型集成模块收尾，而不是继续包装为主要涨点方法。

新增 `tools/run_rhl_se_val_driven.sh`：

- 默认读取 seed1/2/3 的 step1 checkpoint。
- 先运行 `tools/search_rhl_class_weights.py`，只在 val split 选择 class-wise 权重。
- 再运行 `tools/eval_rhl_ensemble.py --mode test`，只消费 val 选出的 `class_weights.json`。
- 输出 `val_search.json`、`class_weights.json`、`test_results.json`、`test_diagnostics.json` 和 `run_summary.md`。
- 默认设置 `CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass`、`TMPDIR=/root/2TStorage/tmp`，符合共享 GPU 规范。

审查结论：P0 没有新增新的融合逻辑，而是复用已有搜索和评估工具，把实验协议固化为脚本。这个实现边界正确，避免了重复实现和 test-set tuning 风险。

## 4. P1 实现审查

新增 CLI 参数位于 `utils/parser.py`：

- `--rhl_init {gaussian,orthogonal,orthogonal_antithetic}`
- `--rhl_scale_mode {legacy,kaiming,unit}`

`network/Buffer.py::RandomBuffer` 新增三类初始化：

- `gaussian + legacy` 保持原始 `torch.nn.Linear.reset_parameters()` 路径，避免 baseline 漂移。
- `orthogonal` 使用分块 QR 正交方向，适配 `out_features >> in_features`。
- `orthogonal_antithetic` 使用 `[W, -W]`，总 buffer 维度不变。

scale 设计：

- `legacy`：orthogonal 行范数设为 `1/sqrt(3)`，匹配原 Linear uniform 初始化的期望行能量。
- `kaiming`：orthogonal 行范数设为 `sqrt(2)`，对应 ReLU/Kaiming 量级。
- `unit`：行向量单位范数，用于 scale audit。

`trainer/trainer.py::AIR` 已透传 `rhl_init` / `rhl_scale_mode`，并在 `rhl_stats` 输出中增加：

- `init`
- `scale`
- `sparsity`
- `gram_diag_mean`
- `trace_per_pixel`

`utils/ckpt.py::save_ckpt` 新增可选 `config`，checkpoint 保存 `training_config`；训练输出目录同时写 `run_config.json`。`8e0861f` 修复了 step1 中 realignment dataloader 临时改 `opts.curr_step` 导致 step1 checkpoint 记录错误 curr_step 的问题。

新增 `tools/run_boa_rhl.sh`：

- 默认执行 BOA-0/1/2/3：
  - BOA-0: `gaussian legacy`
  - BOA-1: `orthogonal legacy`
  - BOA-2: `orthogonal_antithetic legacy`
  - BOA-3: `orthogonal_antithetic kaiming`
- 默认 `DEFAULT_BATCH_SIZE=32`。
- 若日志命中 OOM，自动用 `FALLBACK_BATCH_SIZE=16` 重跑同一 case。
- 每个 case 独立 `SUBPATH` 和日志，避免覆盖正式 checkpoint。

审查结论：P1 没有改 `RecursiveLinear.fit()` / `update()`，保持闭式解主线；默认参数保持 baseline 行为，新增行为由显式 CLI 控制。实现与方案动机一致。

## 5. 验证记录

已执行：

```bash
python -m unittest tests.test_rhl_buffer -v
```

结果：8 项通过。

```bash
python -m py_compile network/Buffer.py utils/parser.py trainer/trainer.py utils/ckpt.py tests/test_rhl_buffer.py tools/eval_rhl_ensemble.py tools/search_rhl_class_weights.py
```

结果：退出码 0。

```bash
bash -n tools/run_rhl_se_val_driven.sh tools/run_boa_rhl.sh
```

结果：退出码 0。

```bash
git diff a2cbc4e..HEAD --check
```

结果：退出码 0。

```bash
python utils/parser.py --rhl_init orthogonal_antithetic --rhl_scale_mode kaiming --rhl_seed 2
```

结果：Config 正确显示 `rhl_seed=2`、`rhl_init='orthogonal_antithetic'`、`rhl_scale_mode='kaiming'`。

AIR CPU smoke 已执行：dummy backbone -> RHL feature expansion -> C-RLS fit/update -> forward，输出：

```text
AIR smoke OK (2, 16, 16) (2, 4, 4, 2)
```

CodeRabbit：未执行。原因是本机未安装 `coderabbit` CLI：

```text
coderabbit --version -> NOT_INSTALLED
coderabbit auth status -> command not found
```

## 6. 未直接修改但建议后续处理

1. `utils/parser.py` 中 `rhl_seed` 注释仍偏向 RHL-SE 语境；现在 `rhl_seed=-1` 的真实含义是“使用全局 RNG 状态”，不再意味着强制 baseline 初始化。建议后续把注释改成同时覆盖 BOA-RHL。
2. 当前 full train smoke 受限于项目没有 `--max-train-batches` / `--max-val-batches` 参数，无法按 AGENTS 规范做 1 epoch 小批量训练验证。建议后续新增调试型 batch 限制参数。
3. BOA 的 `legacy` scale 是按原 Linear uniform 初始化的期望行能量匹配，理论上合理；最终是否提升仍必须由 BOA-0/1/2/3 实验判断，不能只靠初始化统计定性。

