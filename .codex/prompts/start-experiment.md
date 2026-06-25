# Start SegACIL Experiment

Use this prompt when preparing a new SegACIL experiment.

```text
$segacil-experiment-gate

请检查并准备以下 SegACIL 实验。先只做协议、路径、显存和 checkpoint lineage 审查；如果可以启动，再给出 tmux 命令。

目标：
- 方法：
- 服务器：A100 / TRS
- MODEL：
- AIR_FEATURE_SOURCE：
- TASK / SETTING / STEP：
- SUBPATH：
- BASE_SUBPATH：
- BUFFER / GAMMA：
- RANDOM_SEED / RHL_SEED：
- batch size：
- 预期对照 baseline：
```
