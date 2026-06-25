# Review SegACIL Experiment

Use this prompt when an experiment has produced logs or results.

```text
$segacil-experiment-gate

请分析以下 SegACIL 实验结果。必须读取真实 test_results/run_manifest/log，不要凭记忆下结论。

实验目录：

需要回答：
1. 实验是否完成？
2. 使用的 MODEL、AIR_FEATURE_SOURCE、BASE_SUBPATH、BUFFER、GAMMA、RHL 配置是什么？
3. old/new/all mIoU 是多少？
4. 与指定 baseline 的差值是多少？
5. 结果是否符合预期，下一步是否继续？
```
