# 7-5 自适应伪标签 PhaseB artifact 结果与低分位后续实验

## 1. 完成性确认

本轮 PhaseB artifact 实验已正常完成。证据如下：

- tmux 会话 `apl_phaseB_artifact_705` 日志末尾出现 `[phaseB] done Sun Jul  5 05:50:07 UTC 2026`。
- `logs/pseudo_label/phaseB_artifact_summary.md`、`.csv`、`.json` 已生成。
- 三组实验均有 step1 `test_results_*.json` 和 `pseudo_label_stats.json`：
  - `20260701_pseudo_15-5_overlap_artifact_q0p05_seed1_bs32_reuse20260627step0`
  - `20260701_pseudo_15-5_overlap_artifact_q0p10_seed1_bs32_reuse20260627step0`
  - `20260701_pseudo_15-5_overlap_artifact_q0p20_seed1_bs32_reuse20260627step0`

本轮均使用同一协议：VOC `15-5`、`overlap`、step1、`deeplabv3_resnet101`、`batch_size=32`、`BASE_SUBPATH=20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32`，因此三组内部可比。

## 2. 结果表

| 方法 | all mIoU | old 0-15 mIoU | new 16-20 mIoU | 接收率 | 备注 |
|---|---:|---:|---:|---:|---|
| fixed 0.6 | 0.707731 | 0.796761 | 0.422836 | 0.949132 | 第一轮固定阈值最好结果 |
| artifact q0.05 | **0.707930** | **0.797027** | 0.422821 | 0.950256 | 本轮最好，略高于 fixed 0.6 |
| artifact q0.10 | 0.707716 | 0.796790 | 0.422680 | 0.901512 | 与 fixed 0.6 基本持平但略低 |
| artifact q0.20 | 0.707231 | 0.796218 | 0.422473 | 0.801851 | 明显变差 |
| fixed 0.7 | 0.707383 | 0.796346 | 0.422703 | 0.903503 | 原先常用固定阈值对照 |
| batch global q0.7 | 0.703257 | 0.791423 | 0.421124 | 0.300003 | 接收率过低，明显差 |
| batch class q0.7 | 0.704535 | 0.792941 | 0.421636 | 0.300008 | 接收率过低，明显差 |

## 3. 第一性原理分析

伪标签在这里解决的是 step1 训练时旧类监督不足的问题。模型需要在学习新类 16-20 的同时，尽量不把旧类 1-15 的像素遗忘成背景或新类。因此阈值不是越高越好：阈值太高会提高单个伪标签的置信度，但会丢掉大量旧类像素，旧类边界和小目标更容易被遗忘。

本轮数据支持这个判断：

- q0.05 接收率约 95.03%，all/old/new 都处在最高或近最高位置。
- q0.10 接收率降到 90.15%，all mIoU 降低 0.000214。
- q0.20 接收率降到 80.19%，all mIoU 继续降低。

artifact q0.05 的每类阈值中有 10/15 个低于 0.6，说明它不是简单复制 fixed 0.6，而是允许某些旧类使用更低阈值保留更多像素。它略高于 fixed 0.6，说明“按类自适应地放宽低置信旧类”的方向有价值，但当前提升只有 0.000199，属于弱信号，不能作为论文级强结论。

## 4. 对抗性审查

当前能确认：

- 结果来自真实 `test_results_*.json`，不是日志猜测。
- 三组 artifact 训练都使用同一个 teacher checkpoint SHA。
- 输出目录不同，没有覆盖历史结果。
- `artifact_class` 相比 batch quantile 显著更稳定。

当前还不能确认：

- 0.000199 的提升是否超过随机噪声。需要更多 q 值、必要时多 seed 验证。
- q0.05 的优势到底来自“类自适应”，还是只是接收率接近 fixed 0.6。需要 `q0.05 + min_conf=0.6` 反证。
- 低于 0.6 的类阈值是否真的帮助了弱类。如果 `q0.05 + min_conf=0.6` 变差，则说明低阈值对部分类有用；如果变好，则说明低阈值引入了噪声。

## 5. 初步结论

1. PhaseB artifact 流程是可运行、可复现、可汇总的，当前没有发现实验完成性问题。
2. `artifact_class q0.05` 是当前 15-5 overlap 伪标签线的最好单点，略高于 fixed 0.6。
3. 高阈值/高分位并不适合当前协议；q0.20 和 q0.7 batch 类方法都因为接收率降低而变差。
4. 当前结果只支持“artifact 低分位值得继续搜索”，还不足以支持“自适应伪标签已经形成稳定涨点方法”。

## 6. 已启动的下一组实验设计

下一组命名为 PhaseB low-q search，目标是围绕 q0.05 做低分位细搜索，并加入一个机制反证：

| 实验 | 目的 |
|---|---|
| artifact q0.01 | 测试更激进保留伪标签是否继续涨点或开始引入噪声 |
| artifact q0.03 | 补 q0.01 与 q0.05 中间点 |
| artifact q0.07 | 补 q0.05 与 q0.10 中间点 |
| artifact q0.05 + min_conf=0.6 | 反证低于 0.6 的自适应阈值是否有用 |

配置文件：

- `configs/pseudo_label_phaseB_artifact_lowq_calibration.tsv`
- `configs/pseudo_label_phaseB_artifact_lowq_train.tsv`

预期汇总输出：

- `logs/pseudo_label/phaseB_artifact_lowq_summary.md`
- `logs/pseudo_label/phaseB_artifact_lowq_summary.csv`
- `logs/pseudo_label/phaseB_artifact_lowq_summary.json`

## 7. tmux 连续训练机制

连续训练不是靠人工逐个启动，而是在同一个 tmux 会话中串行执行：

```bash
set -euo pipefail
bash tools/run_pseudo_label_artifact_calibration_grid.sh --grid configs/pseudo_label_phaseB_artifact_lowq_calibration.tsv --mode run
bash tools/run_pseudo_label_grid.sh --grid configs/pseudo_label_phaseB_artifact_lowq_train.tsv --mode run
python tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_phaseB_artifact_lowq_train.tsv \
  --output-md logs/pseudo_label/phaseB_artifact_lowq_summary.md \
  --output-csv logs/pseudo_label/phaseB_artifact_lowq_summary.csv \
  --output-json logs/pseudo_label/phaseB_artifact_lowq_summary.json \
  --title "PhaseB Artifact Low-Q Grid Summary"
```

`set -e` 保证任意一步失败时队列停止；runner 内部按 TSV 行顺序执行，因此 q0.01、q0.03、q0.07、q0.05_minconf0.6 会自动连续跑。

## 8. Low-Q 追加结果与下一组

`phaseB_artifact_lowq` 队列已于 2026-07-05 09:24 UTC 完成。结果如下：

| 方法 | all mIoU | old 0-15 mIoU | new 16-20 mIoU | 接收率 | 结论 |
|---|---:|---:|---:|---:|---|
| artifact q0.01 | **0.708082** | **0.797185** | **0.422949** | 0.989409 | 当前最好，仍未触及低分位左边界 |
| artifact q0.03 | 0.708019 | 0.797118 | 0.422902 | 0.970145 | 接近 q0.01 |
| artifact q0.07 | 0.707840 | 0.796929 | 0.422756 | 0.930301 | 低于 q0.03/q0.01 |
| artifact q0.05 + min_conf=0.6 | 0.707723 | 0.796772 | 0.422768 | 0.936502 | 强行抬高低阈值会变差 |

新增证据说明：q0.01 的所有旧类阈值都低于 0.6，而 `q0.05 + min_conf=0.6` 明显弱于 q0.05 原始版本。这支持一个更具体的机制判断：当前 15-5 overlap 下，低分位 artifact 的收益很大程度来自保留低置信旧类像素，而不是来自传统意义上“更干净、更高置信”的伪标签。

因此继续启动极低分位边界搜索：

| 实验 | 目的 |
|---|---|
| artifact q0.00 | 测试近似全接收旧类候选像素是否达到峰值或开始引入噪声 |
| artifact q0.005 | 补 q0.00 与 q0.01 中间点 |
| artifact q0.015 | 补 q0.01 与 q0.03 中间点 |
| artifact q0.01 + min_conf=0.6 | 反证 q0.01 的收益是否依赖低于 0.6 的阈值 |

配置文件：

- `configs/pseudo_label_phaseB_artifact_extreme_lowq_calibration.tsv`
- `configs/pseudo_label_phaseB_artifact_extreme_lowq_train.tsv`
