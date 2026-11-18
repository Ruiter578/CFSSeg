# 7-6 自适应伪标签 disjoint 复验结果与 low-q 后续实验

## 1. 完成性确认

上一组 `15-5 disjoint` 复验已完成。证据：

- 主日志 `logs/pseudo_label/disjoint_phaseB_20260705_150718.log` 末尾出现 `[disjoint-phaseB] done Mon Jul  6 10:40:09 UTC 2026`。
- 已生成 `logs/pseudo_label/disjoint_phaseB_base_fixed_summary.md`、`logs/pseudo_label/disjoint_phaseB_artifact_summary.md` 及对应 CSV/JSON。
- 三组 step1 均有 `test_results_*.json`：
  - `checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step1/test_results_20260706_095428.json`
  - `checkpoints/20260705_pseudo_15-5_disjoint_fixed0p6_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_101656.json`
  - `checkpoints/20260705_pseudo_15-5_disjoint_artifact_q0p005_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_104008.json`

本轮协议一致：VOC `15-5`、`disjoint`、step1、`deeplabv3_resnet101`、`batch_size=32`、`BUFFER=8196`、`GAMMA=1`、`random_seed=1`。`fixed0.6` 与 `artifact q0.005` 复用同一个 disjoint step0：

```text
checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step0/deeplabv3_resnet101_voc_15-5_step_0_disjoint.pth
```

## 2. 结果表

| 方法 | all mIoU | old 0-15 mIoU | new 16-20 mIoU | Overall Acc | Mean Acc | 接收率 |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.689438 | 0.770884 | **0.428810** | 0.924229 | 0.768682 | - |
| fixed0.6 | 0.694639 | 0.778582 | 0.426021 | **0.926505** | 0.782796 | 0.906862 |
| artifact q0.005 | **0.694852** | **0.778820** | 0.426152 | 0.926390 | **0.785290** | 0.994795 |

相对 `fixed0.6`：

| 对比 | all mIoU | old mIoU | new mIoU |
|---|---:|---:|---:|
| artifact q0.005 - fixed0.6 | +0.000212 | +0.000238 | +0.000131 |
| off - fixed0.6 | -0.005201 | -0.007698 | +0.002789 |

新类逐类对比 `artifact q0.005 - fixed0.6`：

| class id | fixed0.6 IoU | artifact q0.005 IoU | delta |
|---:|---:|---:|---:|
| 16 | 0.258374 | 0.257403 | -0.000971 |
| 17 | 0.571636 | 0.572222 | +0.000587 |
| 18 | 0.284963 | 0.285180 | +0.000217 |
| 19 | 0.704443 | 0.704516 | +0.000073 |
| 20 | 0.310691 | 0.311440 | +0.000750 |

## 3. 第一性原理分析

不可再分事实：

- `15-5 disjoint` step1 的训练数据不提供旧类 1-15 的真实标签，旧类保护依赖 step0 teacher 伪标签。
- `fixed0.6` 与 `artifact q0.005` 使用同一个 teacher checkpoint SHA：`040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962`。
- `artifact q0.005` 的平均旧类阈值显著低于 0.6，接收率为 `0.994795`；`fixed0.6` 接收率为 `0.906862`。
- 最终 all mIoU 由 21 类平均得到，old 0-15 占 16 类，new 16-20 占 5 类。

由事实推出的判断：

- 伪标签本身对 `disjoint` 有明显作用：`fixed0.6` 比 `off` 高 `+0.005201` all mIoU，主要来自 old mIoU `+0.007698`。
- `artifact q0.005` 相比 `fixed0.6` 仍为正，但只有 `+0.000212` all mIoU，属于弱信号。
- 该收益不是`新类显著变强`，而是 old/new 都有极小正增益；new 类内部还有 class 16 下降。

尚未验证的假设：

- disjoint 下最佳低分位是否也是 q0.005 附近，还是 q0.00/q0.01/q0.015 更强。
- `artifact q0.005` 的微弱收益是否超过随机种子噪声。
- 极高接收率是否主要补回正确旧类像素，还是引入了可被指标掩盖的噪声。

## 4. 对抗性审查

如果 `artifact q0.005 在 disjoint 上有效` 这个结论是错的，最可能错在：

1. **随机噪声**：当前 `+0.000212` 太小，单 seed 不能证明稳定涨点。最小反证是 seed2/seed3 paired run。
2. **阈值边界没摸清**：只测了 q0.005，不能确定它是 disjoint 最佳点。最小反证是 q0.00/q0.01/q0.015 low-q grid。
3. **接收率解释不完整**：artifact 接收率接近全接收，但缺少 raw-mask precision/recall audit。最小反证是对同一 train split 做旧类 raw mask 对齐审计。
4. **跨 setting 比较误读**：disjoint 绝对 mIoU 低于 overlap，不应和 overlap 直接当同一 baseline 比较；只能在同 setting 内比较 off/fixed/artifact。

因此当前只能说：`artifact q0.005` 在 `15-5 disjoint` 单 seed 上保持了极弱正向信号；不能说已经形成论文级稳定提升。

## 5. 自适应伪标签阈值当前情况

当前方法状态：

- `15-5 overlap` 已完成 fixed/batch/artifact/low-q/extreme-low-q 搜索，最佳区间是 `artifact q0.005 - q0.01`，比 `fixed0.6` 高约 `+0.00035` all mIoU。
- `15-5 disjoint` 已完成 off/fixed0.6/artifact q0.005 复验，artifact 比 fixed0.6 高 `+0.000212` all mIoU。
- `batch_global/batch_class q0.7` 已经基本排除为主方法，因为接收率过低。
- `artifact_class` 是当前唯一值得保留的自适应阈值路线，但收益很小，应定位为`有机制信号、需多协议/多 seed 放大或降级为辅助消融`。

## 6. 下一组实验

下一组启动 `15-5 disjoint` low-q 边界搜索，连续串行执行 calibration -> train grid -> summarize。

| 实验 | 目的 |
|---|---|
| artifact q0.00 | 测试近似全接收是否优于 q0.005 |
| artifact q0.01 | 检查 disjoint 峰值是否与 overlap 的 q0.01 对齐 |
| artifact q0.015 | 检查 q0.01 右侧是否开始下降 |
| artifact q0.005 + min_conf0.6 | 反证低于 0.6 的类阈值是否是收益来源 |

配置与 runner：

- `configs/pseudo_label_disjoint_artifact_lowq_calibration.tsv`
- `configs/pseudo_label_disjoint_artifact_lowq_train.tsv`
- `tools/run_disjoint_artifact_lowq_20260706.sh`

预期输出：

- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.md`
- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.csv`
- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.json`

这组实验回答的问题是：`disjoint` 下 artifact 的弱正向是否同样来自`极高召回 + 允许低阈值`，以及当前 q0.005 是否真是局部最佳。

## 7. 启动异常与处理

首次启动 `apl_disjoint_lowq_706` 时，calibration 在取第一批 train loader 时失败：

```text
RuntimeError: Caught RuntimeError in pin memory thread for device 0.
RuntimeError: CUDA error: invalid argument
```

定位结果：

- `torch.Tensor.pin_memory('cuda:0')` 单独测试正常；
- 同一 VOC disjoint train dataset 在 `pin_memory=False` 下首批正常；
- 同一 dataset 在 `pin_memory=True` 下稳定复现失败；
- calibration `max_batches=1` 在 `SEGACIL_PIN_MEMORY=0` 下可以生成临时 artifact。

处理方式：

- `datasets/init_dataset.py` 新增环境变量 `SEGACIL_PIN_MEMORY`，默认仍为 `1`，不改变既有训练默认行为；
- 本次连续 runner `tools/run_disjoint_artifact_lowq_20260706.sh` 显式设置 `SEGACIL_PIN_MEMORY=0`；
- 已执行 `py_compile`、`bash -n`、grid dry-run、DataLoader 首批检查、calibration `max_batches=1` smoke 和完整单元测试。

## 8. low-q 实验完成结果

`15-5 disjoint` low-q 边界搜索已完成。最终日志：

```text
logs/pseudo_label/disjoint_artifact_lowq_20260706_resume.log
```

汇总文件：

- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.md`
- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.csv`
- `logs/pseudo_label/disjoint_phaseB_artifact_lowq_summary.json`

结果表：

| 实验 | all mIoU | old 0-15 mIoU | new 16-20 mIoU | 接收率 | 相对 fixed0.6 |
|---|---:|---:|---:|---:|---:|
| artifact q0.00 | **0.694872** | **0.778822** | **0.426233** | 0.999936 | +0.000233 |
| artifact q0.01 | 0.694838 | 0.778810 | 0.426129 | 0.989413 | +0.000199 |
| artifact q0.015 | 0.694826 | 0.778801 | 0.426105 | 0.984050 | +0.000186 |
| artifact q0.005 + min_conf0.6 | 0.694639 | 0.778582 | 0.426021 | 0.906862 | +0.000000 |

完成性证据：

- q0.015 结果：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p015_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_224327.json`
- q0.005 + min_conf0.6 结果：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p005_minconf0p6_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_231315.json`
- summary CSV/JSON 共 4 行，全部 `status=done`。

中断原因与恢复：

- 首次恢复前，q0.015 在 step1 保存 `final.pth` 时失败，日志错误为 `PytorchStreamWriter failed writing file data/670: file write failed`。
- 根因是 `/root/2TStorage` 已满，`df -h` 显示可用空间为 0。
- 已将 `/root/2TStorage/lyc/2D_scannet_seg` 完整迁移到 `/data/lyc/2D_scannet_seg`，旧路径改为 symlink，释放约 100G；迁移校验使用 `rsync -n --itemize-changes`，差异为 0。
- 清理 q0.015 的失败半成品目录后，runner 自动跳过 q0.00/q0.01，补跑 q0.015 与 q0.005 + min_conf0.6，并完成 summarize。

结论更新：

- disjoint low-q 的最佳点目前是 `artifact q0.00`，不是先前的 `q0.005`。
- 但最强提升仍只有 `+0.000233` all mIoU，相对 fixed0.6 是弱正向信号，不足以单独作为主方法涨点。
- `q0.005 + min_conf0.6` 退回到 fixed0.6 水平，说明 disjoint 中这条弱收益主要来自允许低于 0.6 的类别阈值、提高旧类伪标签召回，而不是固定阈值本身。
- 后续若继续推进自适应阈值，应优先做 paired multi-seed 和 raw-mask 伪标签质量审计；若资源有限，应把它降级为辅助消融，把主线资源转向更强的 RHL / 集成机制。
