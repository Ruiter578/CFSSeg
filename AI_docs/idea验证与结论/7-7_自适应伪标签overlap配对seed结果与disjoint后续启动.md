# 7-7 自适应伪标签 overlap 配对 seed 结果与 disjoint 后续启动

## 1. 完成性确认

`15-5 overlap` paired-seed 队列已完成。证据：

- 主日志：`logs/pseudo_label/overlap_paired_seed_20260707_resume.log`
- 汇总文件：
  - `logs/pseudo_label/overlap_paired_seed_summary.md`
  - `logs/pseudo_label/overlap_paired_seed_summary.csv`
  - `logs/pseudo_label/overlap_paired_seed_summary.json`
- tmux 会话 `apl_overlap_paired_seed_707_resume` 已退出，无残留训练进程。

本轮协议：VOC `15-5`、`overlap`、step1、`deeplabv3_resnet101`、`batch_size=32`、`BUFFER=8196`、`GAMMA=1`，复用同一个 step0：

```text
checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
```

## 2. 结果表

| 实验 | all mIoU | old 0-15 mIoU | new 16-20 mIoU | 接收率 | 相对 fixed0.7 |
|---|---:|---:|---:|---:|---:|
| fixed0.6 seed2 | 0.707608 | 0.796257 | 0.423932 | 0.947389 | -0.000123 |
| artifact q0.01 seed2 | **0.708017** | **0.796752** | **0.424064** | 0.989750 | +0.000286 |
| fixed0.6 seed3 | 0.703632 | 0.796225 | 0.407333 | 0.947807 | -0.004099 |
| artifact q0.01 seed3 | **0.704058** | **0.796734** | **0.407493** | 0.991047 | -0.003673 |

同 seed 对比：

| seed | artifact q0.01 - fixed0.6 all mIoU | old mIoU | new mIoU |
|---:|---:|---:|---:|
| 2 | +0.000409 | +0.000495 | +0.000131 |
| 3 | +0.000426 | +0.000509 | +0.000160 |

## 3. 当前判断

已由日志和 JSON 确认的事实：

- `artifact_class q0.01` 在 seed2 和 seed3 上均高于同 seed 的 `fixed0.6`。
- 增益量级很小，约 `+0.0004` all mIoU。
- 改善主要来自 old mIoU 的极小提升，新类 mIoU 也略有上升但幅度更小。
- artifact 接收率约 `0.99`，显著高于 fixed0.6 的约 `0.947`。

由事实推出的判断：

- overlap 下的 artifact 低分位机制有稳定但很弱的正向信号。
- 当前结果不能支撑"强涨点方法"，更适合作为辅助消融或机制证据。
- 若要继续证明其价值，需要看 disjoint paired seed 是否也有同向结果，或者引入 raw-mask 质量审计解释为什么高召回未明显伤害新类。

若上述判断是错的，最可能错在：

- seed2/seed3 仍不足以估计随机噪声；
- 复用 seed1 step0 会让随机种子差异主要体现在 step1，而不是完整端到端多 seed；
- batch size 与原论文 step1 默认值不完全一致，因此只能与同 batch 的伪标签实验比较。

## 4. batch size 说明

原始 `run_origin.sh` 中 step1 的 `DEFAULT_BATCH_SIZE=64`，step0 的 `SPECIAL_BATCH_SIZE=32`。本轮伪标签实验不是用原始 `run_origin.sh` 直接跑，而是通过 `tools/run_adaptive_pseudo_label.sh` 和 grid TSV 明确指定：

```text
batch_size=32
step0_batch_size=32
```

因此上一轮回答中说的 `32` 是当前正在跑的伪标签实验的实际配置，不是原论文或原始脚本的 step1 默认配置。当前所有 PhaseB / low-q / paired-seed 伪标签结果都应在 `batch_size=32` 这一内部可比条件下解释，不应直接声称等同原论文 `batch_size=64` 协议。

## 5. 已启动的后续实验

根据 disjoint low-q 结果，`q0.00` 是当前 disjoint 单 seed 的最好 artifact 点。因此已启动 `15-5 disjoint` paired-seed 验证：

- tmux：`apl_disjoint_paired_seed_707`
- 日志：`logs/pseudo_label/disjoint_paired_seed_20260707.log`
- calibration：`configs/pseudo_label_disjoint_paired_seed_calibration.tsv`
- train grid：`configs/pseudo_label_disjoint_paired_seed_train.tsv`
- runner：`tools/run_disjoint_paired_seed_20260707.sh`

队列包含：

| 实验 | 目的 |
|---|---|
| fixed0.6 seed2 | disjoint seed2 fixed 对照 |
| artifact q0.00 seed2 | disjoint seed2 artifact 最佳点验证 |
| fixed0.6 seed3 | disjoint seed3 fixed 对照 |
| artifact q0.00 seed3 | disjoint seed3 artifact 最佳点验证 |

预期汇总输出：

- `logs/pseudo_label/disjoint_paired_seed_summary.md`
- `logs/pseudo_label/disjoint_paired_seed_summary.csv`
- `logs/pseudo_label/disjoint_paired_seed_summary.json`
