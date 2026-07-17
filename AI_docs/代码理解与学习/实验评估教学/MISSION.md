# Mission: SegACIL 实验评估判读

## Why

理解 SegACIL 的训练、增量验证与完整测试各自覆盖的数据和应承担的决策角色，能够独立判断一项超参数结论是否可解释、可复现、可报告。

## Success looks like

- 能从 dataloader 与结果 JSON 判断一项指标覆盖的是哪一批样本和类别。
- 能区分用于候选筛选的验证指标与用于最终报告的完整测试指标。
- 能发现并说明“缺失类别被按 0 计入 mIoU”带来的评价偏差。

## Constraints

- 以当前 SegACIL 的 VOC 15-5 sequential 代码和实验输出为准。
- 不把测试集分数用作持续调参的唯一依据。

## Out of scope

- 本课程暂不讨论网络结构、损失函数和 3D 数据集。
