# DeepLabV3+ 对照实验代码改动与结果报告

日期：2026-06-12  
工作区：`/root/2TStorage/lyc/SegACIL_deeplabv3plus`  
分支：`feature/deeplabv3plus-control`  
基线 commit：`68ef667`

> 2026-06-23 收尾：本报告记录早期接入过程；最终机制、代码接口、完整消融和结论以 `AI_docs/idea构思与实验设计/v3plus设计与验证（未完善收尾）/6-15_DeepLabV3Plus特征不匹配原因细化与验证方案.md` 为准。

---

## 1. 目标

为 SegACIL / CFSSeg 增加 `deeplabv3plus_resnet101` 对照模型，用于验证 DeepLabV3+ 强 decoder 对 VOC 15-5 sequential 的影响。

边界：

1. 不改原工作区 `/root/2TStorage/lyc/SegACIL` 的 `main`。
2. 不把 DeepLabV3+ 写成主方法贡献，只作为 architecture control。
3. `run.sh` 默认仍是 `deeplabv3_resnet101`。
4. DeepLabV3+ 不复用 DeepLabV3 step0 checkpoint，必须从 step0 重新训练。

---

## 2. 分支与工作区

创建了独立 worktree：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus
```

对应分支：

```text
feature/deeplabv3plus-control
```

原工作区仍保持：

```text
/root/2TStorage/lyc/SegACIL -> main
```

---

## 3. 代码改动

### 3.1 `network/modeling.py`

接通 ResNet 分支的 DeepLabV3+：

```python
elif name == 'deeplabv3plus':
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
```

保留原 DeepLabV3 路径：

```python
if name == 'deeplabv3':
    return_layers = {'layer4': 'out'}
    classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
```

### 3.2 `network/_deeplab.py`

重写 `DeepLabHeadV3Plus` 的 head 契约，使其兼容当前 step0 训练和 step1 AIR：

```text
project low-level feature
ASPP high-level feature
upsample + concat
decoder: 304 -> 256
head: Linear(256, sum(num_classes))
```

正常前向：

```text
model(images) -> logits, feat_dict
```

step1 中执行：

```python
self.model.classifier.head = nn.Identity()
```

之后：

```text
model(images) -> B x 256 x H x W dense feature, feat_dict
```

这与现有 `AIR(backbone_output=256, ...)` 对齐。

### 3.3 `run.sh`

将关键实验变量环境变量化：

```bash
MODEL="${MODEL:-deeplabv3_resnet101}"
TASK="${TASK:-15-5}"
SETTING="${SETTING:-sequential}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
GAMMA="${GAMMA:-1}"
```

默认仍不改变：

```text
MODEL=deeplabv3_resnet101
```

---

## 4. 环境修正

新 worktree 是 clean Git 状态，不包含原工作区本地未跟踪的数据列表：

```text
datasets/data/voc/train_cls.txt
datasets/data/voc/val_cls.txt
```

因此在新 worktree 中创建了本地 symlink：

```bash
datasets/data -> /root/2TStorage/lyc/SegACIL/datasets/data
```

该 symlink 是实验环境绑定，不应作为通用代码合并。

---

## 5. 验证结果

### 5.1 Python / PyTorch 环境

默认 `base` 环境没有 PyTorch，因此使用历史 SegACIL 实验环境：

```bash
source /home/linyichen/miniconda3/etc/profile.d/conda.sh
conda activate segacil
```

确认结果：

```text
torch 2.1.2+cu118
cuda_available True
GPU NVIDIA A100-SXM4-80GB
```

### 5.2 模型前向 smoke test

DeepLabV3+：

```text
normal logits: (1, 16, 32, 32)
feat keys: ['back_out', 'decoder_feature', 'feature', 'low_level']
air feature after Identity head: (1, 256, 32, 32)
```

DeepLabV3：

```text
normal logits: (1, 16, 16, 16)
feat keys: ['back_out', 'feature']
```

结论：

```text
deeplabv3plus_resnet101 构建和前向通过。
deeplabv3_resnet101 未被破坏。
V3+ 的 Identity head 契约能给 AIR 输出 256-d dense feature。
```

### 5.3 真实 dataloader + 单 batch 训练 smoke test

batch size = 2 通过：

```text
Dataset: voc, Train set: 8437, Val set: 1095, Test set: 1449
one_batch_ok
image (2, 3, 513, 513)
labels (2, 513, 513)
outputs (2, 16, 513, 513)
loss 11.846136093139648
feat_keys ['back_out', 'decoder_feature', 'feature', 'low_level']
```

batch size = 16 在当前 GPU 占用下 OOM。失败时 GPU 上已有多个其他训练进程，当前进程约占 22GB 后仍不足。

batch size = 8 通过：

```text
batch_probe_ok 8
loss 11.948945999145508
max_memory_gb 16.4095458984375
```

---

## 6. 完整实验启动状态

完整 VOC 15-5 sequential DeepLabV3+ 对照实验已启动在 tmux：

```text
tmux session: segacil_v3plus_control
```

运行命令：

```bash
source /home/linyichen/miniconda3/etc/profile.d/conda.sh
conda activate segacil
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
PYTHONUNBUFFERED=1 \
MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
SUBPATH=20260612_v3plus_voc15-5_seq_bs8_step1bs2 \
START_STEP=0 \
END_STEP=1 \
SPECIAL_BATCH_SIZE=8 \
DEFAULT_BATCH_SIZE=2 \
TRAIN_EPOCH=50 \
GAMMA=1 \
bash run.sh 2>&1 | tee logs/deeplabv3plus/20260612_v3plus_voc15-5_seq_bs8_step1bs2.log
```

日志路径：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/logs/deeplabv3plus/20260612_v3plus_voc15-5_seq_bs8_step1bs2.log
```

当前已确认进入训练：

```text
Running training for step 0 with batch size 8...
model='deeplabv3plus_resnet101'
task='15-5'
curr_step=0
train epoch : 50 , iterations : 52700 , val_interval : 527
[15-5 / step 0] Epoch 0, Itrs 0/1054, Loss=11.7200
```

---

## 7. 结果表

| 实验 | AIR feature | old 0-15 mIoU | new 16-20 mIoU | all mIoU | JSON |
|---|---|---:|---:|---:|---|
| V3+ 初始 control | `decoder` | 0.7815 | 0.3959 | 0.6897 | `checkpoints/20260622_v3plus_air_decoder/voc/15-5/sequential/step1/test_results_20260622_190708.json` |
| V3+ 修复后 control | `aspp_up` | 0.7793 | 0.4613 | **0.7036** | `checkpoints/20260622_v3plus_air_aspp_up/voc/15-5/sequential/step1/test_results_20260622_204635.json` |

`decoder` 结果复现了早期异常；`aspp_up` 在保持旧类的同时把新类提高约 6.54 点。问题来自 DeepLabV3+ low-level decoder feature 与 AIR 解析学习不匹配，不是 batch size。

---

## 8. 最终对照

| baseline | old 0-15 | new 16-20 | all |
|---|---:|---:|---:|
| DeepLabV3, 20260606 | 78.01 | 42.11 | 69.46 |
| DeepLabV3, 20260607 | 77.79 | 43.21 | 69.56 |
| DeepLabV3+ `decoder` | 78.15 | 39.59 | 68.97 |
| DeepLabV3+ `aspp_up` | 77.93 | **46.13** | **70.36** |

DeepLabV3+ 可以作为公平 architecture control，但 step1 AIR 必须读取 `aspp_up` 高层语义特征，不能沿用端到端 decoder 输出。

---

## 9. 合并建议

当前代码层面建议：

```text
保留在 feature/deeplabv3plus-control 分支作为 architecture control。
是否合并 main 由用户单独决定；本轮不自动合并或推送。
```

合并前必须确认：

1. DeepLabV3 默认行为不变。
2. V3+ 完整 step0+step1 通过。
3. `datasets/data` symlink、checkpoint、log、`.pth` 不进入 Git。
4. 默认 runner 使用 `aspp_up + pixel_balance=none + gamma=1 + buffer=8196`。
5. 如果 batch size 因显存限制改变，论文表中必须标注。
