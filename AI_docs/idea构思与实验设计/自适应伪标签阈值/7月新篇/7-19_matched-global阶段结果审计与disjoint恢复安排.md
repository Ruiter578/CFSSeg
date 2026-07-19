# Matched-global 阶段结果审计与 disjoint 恢复安排

## 1. 审计结论

原定六组 matched-global fixed 实验并未全部完成：

| setting | seed | 状态 | 证据 |
| --- | ---: | --- | --- |
| overlap | 1 | 完成 | `test_results_20260718_155137.json` |
| overlap | 2 | 完成 | `test_results_20260718_161623.json` |
| overlap | 3 | 完成 | `test_results_20260718_164108.json` |
| disjoint | 1 | 保存失败 | 有 manifest 和 stats，无 test result，`final.pth` 仅 268 MB |
| disjoint | 2 | 未启动 | 无输出目录 |
| disjoint | 3 | 未启动 | 无输出目录 |

`apl_matched_global_718` 已退出，自动汇总未生成。日志显示 disjoint seed1 已完成特征与伪标签处理，但在 `torch.save` 写入 checkpoint 时发生：

```text
PytorchStreamWriter failed writing file data/670: file write failed
```

因此不能把该批次视为六组完成，也不能据此执行最终 stop-loss 判定。

## 2. 已完成 overlap 结果

定义：

\[
\Delta_s=\operatorname{mIoU}(\text{artifact}_s)
-\operatorname{mIoU}(\text{matched-global-fixed}_s)
\]

| seed | matched-global fixed | artifact q0.01 | Δ all mIoU | Δ（pp） | Δ old mIoU | Δ new mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.708038216 | 0.708081556 | +0.000043341 | +0.004334 | +0.000061694 | -0.000015391 |
| 2 | 0.707996605 | 0.708016739 | +0.000020134 | +0.002013 | +0.000025196 | +0.000003936 |
| 3 | 0.703999264 | 0.704057647 | +0.000058383 | +0.005838 | +0.000072562 | +0.000013011 |

三 seed 结果：

- artifact 胜出数：3/3；
- 平均差值：`+0.000040619`，即 `+0.004062 pp`；
- 差值样本标准差：`0.000019270`；
- artifact 和 matched-global fixed 的接收率均约为 99%，接收规模已基本匹配；
- 增益主要来自 old mIoU，new mIoU 在 seed1 反向、seed2/3 仅万分位变化。

该结果说明类别阈值相对同总体分位点全局阈值存在方向一致但极弱的正向信号；效应量只有预注册 `+0.1 pp` 门槛的约 4%。若六组总体均值最终达到 `+0.1 pp`，disjoint 三 seed 的平均差值需要达到约 `+0.195938 pp`。

现阶段证据等级为 `CAUTION`：不能把方向一致等同于具有实际意义，也不能在缺少 disjoint 结果时提前判定路线通过或失败。

## 3. 中断根因审查

已确认：

- 三个 overlap 运行使用同一保存链路，均成功写出约 791 MB 的 `final.pth`；
- disjoint seed1 只留下 268 MB 截断 checkpoint，错误发生在 checkpoint 写入；
- `/root/2TStorage` 当前使用率为 98%，但审计时仍有约 43 GB 可用；
- inode 使用率仅约 2%，文件大小限制为 unlimited；
- 内核日志没有同期块设备 I/O 错误；
- 当前在同一挂载点成功完成 1 GiB 写入、`fsync` 和删除探针。

最符合现有证据的解释是：失败时出现了瞬时可用空间不足或存储压力，之后其他任务释放了空间。由于原始异常没有保留底层 errno，不能把该解释表述为完全确认。

恢复时不得覆盖截断的 seed1 目录；使用新 `SUBPATH` 保留失败现场和谱系。

## 4. 恢复实验

配置：

```text
configs/pseudo_label_matched_global_disjoint_recovery_20260719.tsv
```

三组参数保持不变：

| seed | confidence | 新 SUBPATH |
| ---: | ---: | --- |
| 1 | 0.029296875 | `20260719_pseudo_15-5_disjoint_globalfixed0p029296875_seed1_bs32_recovery1_reuse20260705disjointstep0` |
| 2 | 0.048828125 | `20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed2_bs32_recovery1_reuse20260705disjointstep0` |
| 3 | 0.048828125 | `20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed3_bs32_recovery1_reuse20260705disjointstep0` |

共同控制变量：

- `task=15-5`
- `setting=disjoint`
- `strategy=fixed`
- `batch_size=32`
- `step0_batch_size=32`
- `buffer=8196`
- `gamma=1`
- `air_feature_source=auto -> decoder`
- `base_subpath=20260705_pseudo_15-5_disjoint_off_seed1_bs32`
- step0 SHA256：`040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962`

运行边界：

- tmux：`apl_matched_disjoint_recovery_719`
- 日志：`logs/pseudo_label/matched_global_disjoint_recovery_20260719.log`
- 汇总：`logs/pseudo_label/matched_global_disjoint_recovery_20260719_summary.{md,csv,json}`
- 启动前要求可用空间至少 10 GiB、GPU 0 显存充足、工作树干净、三个新目录均不存在。

## 5. 恢复完成后的唯一决策

1. 合并 overlap 三组与恢复后的 disjoint 三组，重新计算逐 seed、分 setting 和六组总体差值；
2. 按既定 `+0.1 pp` 门槛执行判定，不修改标准；
3. 若六组总体低于门槛，即使方向一致，也停止 hard classwise threshold 微调，将其降级为辅助实现并转入 reliability-weighted C-RLS；
4. 若 disjoint 出现反向或 setting 冲突，先分析标签可见性与候选像素构成，不增加 q 扫描。
