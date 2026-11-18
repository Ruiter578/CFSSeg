# SegACIL Step0 Resume / tmux 状态报告

生成时间：2026-06-04 15:55 UTC

## 结论

`tmux work` 会话中的实验已经从 `Epoch 39` 结束时保存的权重恢复并继续训练，不是完全卡死。

当前续跑命令使用的是：

```bash
cd /root/2TStorage/lyc/SegACIL && bash ./resume_epoch39_then_run.sh
```

关键参数已经确认：

```text
data_root=/root/2TStorage/lyc/SegACIL/data_root/VOC2012
ckpt=checkpoints/1128/voc/15-1/sequential/step0/final.pth
train_epoch=10
curr_itrs=10520
```

`10520 = 40 * 263`，表示原 step0 已完整完成 40 个 epoch，每个 epoch 263 个 iteration。崩溃发生在 `Epoch 40, Itrs 120/263`，但该部分没有保存 checkpoint，因此可靠恢复点是 `Epoch 39` 结束后的 `final.pth`。

## 现场状态

检查时进程仍在运行：

```text
/home/linyichen/miniconda3/envs/segacil/bin/python train.py \
  --data_root /root/2TStorage/lyc/SegACIL/data_root/VOC2012 \
  --curr_step 0 \
  --train_epoch 10 \
  --curr_itrs 10520 \
  --ckpt checkpoints/1128/voc/15-1/sequential/step0/final.pth
```

GPU 状态显示 `segacil` 进程仍在占用显存并计算：

```text
GPU: NVIDIA A100-SXM4-80GB
GPU util: 100%
segacil used memory: about 68.5GB
```

TensorBoard 事件文件也在更新，说明训练已经进入 iteration 循环。检查时最新 loss 事件为：

```text
event_file: checkpoints/1128/voc/15-1/sequential/step0/events.out.tfevents.1780585956.ubuntu20
latest loss step: 191
latest loss value: 0.13897642493247986
```

## 为什么看起来像卡住

续跑脚本使用了：

```bash
exec > >(tee -a "${LOG_FILE}") 2>&1
```

这会让 Python 的 stdout 变成管道。Python 在管道输出时通常会进行块缓冲，而不是像交互终端那样每行立刻刷新。因此 `print(...)` 输出可能不会实时出现在 tmux 或日志中。

当前训练确实较慢，主要因为同一张 A100 上还有其他训练进程同时占用显存和算力。事件文件大约每 10 个 iteration 写一次 loss，比 tmux 文本输出更适合作为当前进度判断依据。

## trainer.py 的改动

改动文件：

```text
/root/2TStorage/lyc/SegACIL/trainer/trainer.py
```

改动目的：让 `curr_step == 0` 时也可以通过 `--ckpt` 恢复训练。

新增行为：

```text
如果 opts.curr_step == 0 且传入 --ckpt：
1. 加载 checkpoint["model_state"]
2. 加载 checkpoint["optimizer_state"]，如果存在
3. 恢复 checkpoint["best_score"]，如果存在
4. 根据 --curr_itrs 调整 scheduler 进度
```

不传 `--ckpt` 时，原始 step0 从头训练逻辑保持不变。

## 为什么需要改代码

原代码中：

```text
curr_step == 0:
  初始化新模型
  初始化 optimizer
  初始化 scheduler
  直接从 epoch 0 开始训练
```

虽然参数解析器里有 `--ckpt`，但 step0 没有使用它。`init_ckpt()` 只服务于后续 step 的“加载前一步 checkpoint”，不能恢复 step0 中断训练。因此要想从 `step0/final.pth` 恢复，需要补充 step0 的 resume 逻辑。

## 原始错误原因

训练启动时使用的旧路径是：

```text
/root/2TStorage/lyc/CFSSeg/data_root/VOC2012
```

训练过程中目录被改名为：

```text
/root/2TStorage/lyc/SegACIL
```

DataLoader 在初始化时已经把旧路径拼进了每个样本的图片路径中。目录改名后，worker 后续读取图片时仍然访问旧路径，于是出现：

```text
FileNotFoundError:
/root/2TStorage/lyc/CFSSeg/data_root/VOC2012/JPEGImages/2008_004408.jpg
```

## 如何继续监控

查看 tmux：

```bash
tmux attach -t work
```

查看当前日志：

```bash
tail -f /root/2TStorage/lyc/SegACIL/logs/resume_epoch39_20260604_151229.log
```

注意：由于 stdout 缓冲，日志不实时刷新并不一定代表训练停止。

查看事件文件是否继续更新：

```bash
find /root/2TStorage/lyc/SegACIL/checkpoints/1128/voc/15-1/sequential/step0 \
  -maxdepth 1 -name 'events.out.tfevents.*' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail
```

查看 GPU：

```bash
nvidia-smi
```

## VSCode 中删除 tmux 终端会不会中断训练

如果训练是在 tmux 会话内部启动的，通常不会。

VSCode 里关闭/删除终端，相当于 detach 或关闭当前客户端连接；tmux server 和里面的 shell/训练进程会继续在后台运行。

会中断训练的情况包括：

```text
1. 在 tmux 内部按 Ctrl-C 杀掉训练
2. 执行 tmux kill-session -t work
3. kill 对应 python 进程
4. 服务器重启、显卡/驱动异常、进程 OOM
5. 训练其实不是在 tmux 内部启动，而是在普通 VSCode 终端直接启动
```

本次训练进程属于 `work` tmux 会话，因此单纯删除 VSCode 终端不会中断训练。
