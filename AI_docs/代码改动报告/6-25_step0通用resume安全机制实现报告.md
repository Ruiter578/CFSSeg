# 6-25 step0 通用 resume 安全机制实现报告

## 背景

历史分支 `feature/resume-step0-from-epoch39` 曾为一次 VOC step0 中断恢复加入本地专用逻辑，但该分支没有合入 `main`，且脚本绑定了 epoch39 场景。当前主线需要一个通用、安全、默认不触发的 step0 resume 机制。

## 设计结论

本次实现只在以下条件同时满足时启用恢复：

1. `curr_step == 0`；
2. 用户显式传入 `--ckpt`，或通过 `run.sh` 设置 `CKPT`；
3. 用户用 `--curr_itrs` / `CURR_ITRS` 明确说明此前已完成的 iteration 数。

默认 `run.sh` 不设置 `CKPT`，`CURR_ITRS=0`，所以普通 step0、step1、AIR、RHL 和 V3+ 路径不进入 resume 分支。

安全约束：

- `CURR_ITRS < 0` 直接报错；
- `curr_step == 0` 且 `CURR_ITRS > 0` 时必须提供 `CKPT`；
- scheduler 只有在 step0 checkpoint 已加载后才会按 `CURR_ITRS` 对齐；
- `run.sh` 只在 `CURR_STEP=0` 时向 `train.py` 传入 `--ckpt` 和 `--curr_itrs`。

## 代码改动

- `trainer/trainer.py`
  - 新增 `resume_step0_checkpoint()`；
  - 新增 `step0_scheduler_total_itrs()`；
  - 新增 `sync_step0_scheduler_for_resume()`；
  - step0 初始化时若 `opts.ckpt` 非空，则恢复 model、optimizer、best_score，并按 `curr_itrs` 对齐 scheduler。
- `run.sh`
  - 新增可选环境变量 `CKPT`；
  - 新增可选环境变量 `CURR_ITRS`；
  - 默认值不改变现有实验行为。
- `tools/run_step0_resume.sh`
  - 通用 step0 resume 入口；
  - 要求显式设置 `CKPT`、`SUBPATH`、`CURR_ITRS`；
  - 只跑 `START_STEP=0 END_STEP=0`。
- `tests/test_step0_resume.py`
  - 覆盖 checkpoint 恢复模型/optimizer/best_score；
  - 覆盖 scheduler 总 iteration 和当前位置对齐。
  - 覆盖负数 `curr_itrs` 和无 checkpoint 快进 scheduler 的错误配置。

## 使用方式

示例：

```bash
cd /root/2TStorage/lyc/SegACIL
nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass
export TMPDIR=/root/2TStorage/tmp

PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
CUDA_VISIBLE_DEVICES=0 \
MODEL=deeplabv3_resnet101 \
TASK=15-5 \
SETTING=sequential \
CKPT=checkpoints/<old_subpath>/voc/15-5/sequential/step0/final.pth \
SUBPATH=<new_resume_subpath> \
CURR_ITRS=<completed_iterations> \
TRAIN_EPOCH=<remaining_epochs> \
SPECIAL_BATCH_SIZE=32 \
bash tools/run_step0_resume.sh
```

## 边界

- 该机制从保存的 checkpoint 恢复，不恢复 dataloader RNG 和中断 batch 内部状态；
- 如果训练在某个 epoch 中途崩溃，应从上一次保存 checkpoint 的 epoch 边界继续；
- `CURR_ITRS` 用于学习率调度连续性，不会自动从 checkpoint 推断；
- step1 / AIR checkpoint 恢复不走该路径。

## 验证

已按 TDD 先写失败测试，再实现通过：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest tests.test_step0_resume -v
```

CodeRabbit review 发现过一个有效问题：`CURR_ITRS>0` 但未提供 checkpoint 时，scheduler 可能被快进。当前已修复为显式报错，并只在 checkpoint 实际加载后同步 scheduler。

后续提交前还需要运行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile trainer/trainer.py tests/test_step0_resume.py
bash -n run.sh tools/run_step0_resume.sh
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest discover -s tests -p 'test*.py' -v
```
