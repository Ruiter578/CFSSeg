# 20260719 clean replay 结果分析与后续安排

## 结论

本轮 `20260718_clean_e1_1_replay_from_main_gpu2_*` 八组实验已经完成，可以作为替代旧 `20260717_clean_e1_1_replay_v2_gpu2_*` 争议结果的正式证据。

核心选择结论：

- `air_feature_source=decoder`
- `buffer=8224`
- `gamma=1`
- `rhl_norm=none`
- 下一阶段优先搜索 `analytic_tail_epsilon`

## 可比性证据

本轮八组实验统一从 `main` 的干净提交启动：

- Git commit: `e81a5521075a574f10f2981b9081aa89bb97a9de`
- `run_manifest.json` 数量: 8
- `val_results_*.json` 数量: 8
- 所有 `run_manifest.json` 中 `git.dirty=false`
- 汇总文件: `Codex_Plans/20260718_clean_e1_1_replay_from_main_gpu2_summaries/final_validation_summary.tsv`

这些条件解决了上一轮 `20260717_clean_e1_1_replay_v2_gpu2_*` 的主要争议：上一轮虽然有结果，但启动时工作区包含未提交源码差异，manifest 记录为 dirty，不适合继续作为最终调参依据。

## 八组结果

| feature_source | buffer | rhl_seed | all_mIoU | old 0-15 mIoU | new 16-20 mIoU |
|---|---:|---:|---:|---:|---:|
| aspp | 8208 | 2 | 0.588612 | 0.666699 | 0.338734 |
| aspp | 8224 | 2 | 0.590638 | 0.666863 | 0.346716 |
| aspp | 8224 | 4 | 0.591275 | 0.666713 | 0.349873 |
| aspp | 8224 | 5 | 0.591770 | 0.668599 | 0.345918 |
| aspp | 8240 | 2 | 0.587907 | 0.665381 | 0.339991 |
| decoder | 8224 | 2 | 0.597310 | 0.676596 | 0.343595 |
| decoder | 8224 | 4 | 0.596562 | 0.675571 | 0.343734 |
| decoder | 8224 | 5 | 0.595364 | 0.675703 | 0.338279 |

## 对比判断

`buffer=8224` 且 seeds `2/4/5` 下：

| feature_source | all_mIoU mean | old mIoU mean | new mIoU mean |
|---|---:|---:|---:|
| decoder | 0.596412 | 0.675957 | 0.341869 |
| aspp | 0.591228 | 0.667392 | 0.347503 |

`decoder` 在主排序指标 `all_mIoU` 上领先 `aspp` 约 `0.00518`，差距超过当前调参阶段的噪声容忍范围。因此本阶段不再按 `new 16-20 mIoU` 单项反转结论，确定继续使用 `decoder`。

ASPP 局部 buffer 检查中，seed 2 下：

- `buffer=8208`: all mIoU `0.588612`
- `buffer=8224`: all mIoU `0.590638`
- `buffer=8240`: all mIoU `0.587907`

因此即使只看 ASPP 分支，`8224` 也是当前局部最优 buffer。

## 旧争议结果处理

建议删除并停止引用以下旧结果：

- `checkpoints/20260717_clean_e1_1_replay_v2_gpu2_*`
- `Codex_Plans/20260717_clean_e1_1_replay_v2_gpu2_summaries/`

原因是它们已经被本轮 `20260718_clean_e1_1_replay_from_main_gpu2_*` 干净 replay 覆盖。

早期为了隔离实验创建的 worktree 已不再需要继续参与实验：

- `SegACIL-clean-replay-main-20260718`: 本轮实验完成后可删除。
- `SegACIL-clean-validation`: 其中 clean-validation 相关源码、工具、测试和文档已经进入 `main`；旧 worktree 可删除。
- 其他早期 detached worktree: 只保留了已归档结果的工作副本，可删除。

## 当前进度

已完成：

1. buffer/gamma 基础搜索与最终确认。
2. clean validation holdout 协议落地。
3. clean step0 baseline。
4. E1.1 AIR feature source replay: `decoder` vs `aspp`，并确认 `decoder`。

下一步：

1. E1.2 搜索 `analytic_tail_epsilon`。
   - 固定 `air_feature_source=decoder`
   - 固定 `buffer=8224`
   - 固定 `gamma=1`
   - 固定 `rhl_norm=none`
   - 初筛建议: `analytic_tail_epsilon in {0, 1e-4, 1e-3}`，先用 `rhl_seed=2`
   - 当前 `1e-3` 的 seed 2/4/5 已由本轮 replay 覆盖，可复用作基线
2. E1.3 搜索 `gamma`。
   - 在 E1.2 最优 `analytic_tail_epsilon` 固定后进行。
   - 建议初筛 `{0.01, 0.1, 1, 10, 100}`。
3. E1.4 如有必要再做 batch size 或训练稳定性确认。
4. E2 再进入更重的 step0 相关搜索，例如学习率、loss、训练预算等。
