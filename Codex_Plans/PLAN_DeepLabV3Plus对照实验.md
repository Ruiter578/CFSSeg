# PLAN: DeepLabV3+ 对照实验实现与验证

适用项目：`/root/2TStorage/lyc/SegACIL`  
生成日期：2026-06-12  
目标读者：下一轮执行代码改动和实验的 Codex  
边界：本计划只做 DeepLabV3+ + ResNet101 架构对照，不把它写成主方法贡献；默认 DeepLabV3 baseline 不变。

---

## 1. 最终方案

采用 **feature branch + git worktree + 自然代码接入 + 验证后再合并**。

不要在当前 `main` 工作区直接改 DeepLabV3+。当前 `main` 已有 RHL 相关 WIP 和文档改动，直接继续写会混淆实验变量。

执行原则：

1. 用 `git worktree` 创建干净并行工作区。
2. 在 `feature/deeplabv3plus-control` 中实现 `deeplabv3plus_resnet101`。
3. DeepLabV3+ 作为一等模型选项接入 `network/modeling.py`。
4. `run.sh` 默认仍为 `deeplabv3_resnet101`，但支持环境变量选择 V3+。
5. 先做无数据 smoke test，再跑 VOC `15-5 sequential` step0+step1。
6. 只有接口稳定且实验跑通后，才考虑把通用代码合并回 `main`。
7. 不合并 checkpoint、log、event file、`.pth` 等实验产物。

---

## 2. 预检

在原项目中只检查，不修改：

```bash
cd /root/2TStorage/lyc/SegACIL
git branch --show-current
git status --short
```

预期：

```text
branch: main
status: 可能有 RHL WIP 和文档改动
```

如果当前工作区有未提交改动，不要 stash，也不要 commit，优先用 worktree 隔离 DeepLabV3+。

---

## 3. 创建 V3+ 工作区

```bash
cd /root/2TStorage/lyc/SegACIL
git worktree add -b feature/deeplabv3plus-control ../SegACIL_deeplabv3plus HEAD
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
git branch --show-current
git status --short
```

要求：

```text
branch = feature/deeplabv3plus-control
status = clean
```

如果分支已存在：

```bash
cd /root/2TStorage/lyc/SegACIL
git worktree add ../SegACIL_deeplabv3plus feature/deeplabv3plus-control
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
```

---

## 4. 代码改动清单

### 4.1 `network/modeling.py`

目标：接通 ResNet101 的 DeepLabV3+ 路径。

在 `_segm_resnet()` 中支持：

```python
elif name == "deeplabv3plus":
    return_layers = {"layer4": "out", "layer1": "low_level"}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
```

保持原有 DeepLabV3 路径：

```python
if name == "deeplabv3":
    return_layers = {"layer4": "out"}
    classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
```

注意：

1. `deeplabv3_resnet101` 默认逻辑不得改变。
2. `deeplabv3plus_mobilenet` 不应被破坏。
3. `low_level_planes` 对 ResNet layer1 是 256。

### 4.2 `network/_deeplab.py`

目标：让 `DeepLabHeadV3Plus` 和当前训练/AIR 接口兼容。

当前阻塞点：

```python
DeepLabHeadV3Plus.forward() 只返回 heads
```

需要改成返回：

```python
return heads, feat_dict
```

推荐结构：

```python
self.project = ...
self.aspp = ASPP(...)
self.decoder = nn.Sequential(
    nn.Conv2d(304, 256, 3, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
)
self.head = nn.Sequential(
    nn.Linear(256, sum(num_classes)),
)
```

forward 逻辑应满足：

```text
normal:
    model(images) -> logits, feat_dict

after self.model.classifier.head = nn.Identity():
    model(images) -> B x 256 x H' x W', feat_dict
```

关键原因：`trainer/trainer.py` step1 会把最终分类层替换为 `Identity()`，然后把第一个返回值送入 AIR 的 RHL。

建议实现形态：

```python
def forward(self, feature):
    low_level_feature = self.project(feature["low_level"])
    output_feature = self.aspp(feature["out"])
    output_feature = F.interpolate(
        output_feature,
        size=low_level_feature.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    decoder_feature = self.decoder(torch.cat([low_level_feature, output_feature], dim=1))

    B, C, H, W = decoder_feature.shape
    flat_feature = decoder_feature.view(B, C, -1).permute(0, 2, 1)
    flat_output = self.head(flat_feature)
    heads = flat_output.permute(0, 2, 1).view(B, -1, H, W)

    return heads, {
        "feature": flat_output,
        "decoder_feature": decoder_feature,
        "back_out": feature["out"],
        "low_level": low_level_feature,
    }
```

当 `self.head = nn.Identity()` 后：

```text
flat_output = flat_feature
heads = B x 256 x H x W
```

这与 AIR 的 `backbone_output=256` 对齐。

### 4.3 `run.sh`

目标：让模型可选择，但默认不变。

把硬编码默认值改成环境变量默认：

```bash
MODEL="${MODEL:-deeplabv3_resnet101}"
TASK="${TASK:-15-5}"
SETTING="${SETTING:-sequential}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
SPECIAL_BATCH_SIZE="${SPECIAL_BATCH_SIZE:-32}"
```

保持默认行为：

```text
不传环境变量时仍跑当前默认 DeepLabV3 配置。
```

V3+ 全流程运行方式：

```bash
MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
SUBPATH=20260612_v3plus_voc15-5_seq \
START_STEP=0 \
END_STEP=1 \
bash run.sh
```

不要默认启用 RHL normalization。V3+ 第一轮是纯架构对照。

---

## 5. Smoke Test

### 5.1 模型构建与前向

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
python - <<'PY'
import torch
from network.modeling import DeepLabModelFactory

factory = DeepLabModelFactory()
model = factory.model_map["deeplabv3plus_resnet101"](
    num_classes=[1, 15],
    output_stride=8,
    pretrained_backbone=False,
    bn_freeze=False,
)
model.eval()

x = torch.randn(2, 3, 512, 512)
with torch.no_grad():
    y, feat = model(x)
print("normal logits:", tuple(y.shape))
print("feat keys:", sorted(feat.keys()))
assert y.shape[1] == 16, y.shape
assert "decoder_feature" in feat

model.classifier.head = torch.nn.Identity()
with torch.no_grad():
    z, feat2 = model(x)
print("air feature:", tuple(z.shape))
assert z.shape[1] == 256, z.shape
assert torch.isfinite(z).all()
PY
```

通过标准：

```text
normal logits channel = 16
Identity head output channel = 256
无异常、无 NaN、无 Inf
```

### 5.2 确认旧 DeepLabV3 仍可构建

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
python - <<'PY'
import torch
from network.modeling import DeepLabModelFactory

factory = DeepLabModelFactory()
model = factory.model_map["deeplabv3_resnet101"](
    num_classes=[1, 15],
    output_stride=8,
    pretrained_backbone=False,
    bn_freeze=False,
)
model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    y, feat = model(x)
print(tuple(y.shape), sorted(feat.keys()))
assert y.shape[1] == 16
PY
```

---

## 6. 完整实验

第一轮只跑 VOC `15-5 sequential`，从 step0 到 step1。

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
SUBPATH=20260612_v3plus_voc15-5_seq \
START_STEP=0 \
END_STEP=1 \
GAMMA=1 \
bash run.sh
```

如果显存不足，降低 batch size 并记录：

```bash
MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
SUBPATH=20260612_v3plus_voc15-5_seq_bs16 \
START_STEP=0 \
END_STEP=1 \
GAMMA=1 \
DEFAULT_BATCH_SIZE=16 \
SPECIAL_BATCH_SIZE=16 \
bash run.sh
```

结果提取：

```bash
find checkpoints/20260612_v3plus_voc15-5_seq -name 'test_results_*.json' -print
```

需要记录：

```text
Mean IoU
0 to 15 mIoU
16 to 20 mIoU
Class IoU
batch size
output_stride
train_epoch
SUBPATH
git branch and commit hash
```

---

## 7. 结果报告

实验结束后，在 V3+ worktree 中写：

```text
AI_docs/代码改动报告/DeepLabV3Plus对照实验代码改动与结果报告.md
```

报告必须包含：

1. 分支名和 commit hash。
2. 代码改动摘要。
3. smoke test 输出摘要。
4. 完整实验命令。
5. 结果 JSON 路径。
6. old/new/all mIoU 表格。
7. 和 DeepLabV3 baseline 的对比。
8. 是否建议合并回 main。
9. 如果失败，失败日志、原因判断、下一步修复建议。

---

## 8. 合并条件

只有全部满足才考虑合并：

1. `deeplabv3_resnet101` smoke test 通过。
2. `deeplabv3plus_resnet101` smoke test 通过。
3. V3+ 完成 `15-5 sequential` step0+step1。
4. `run.sh` 默认仍是 DeepLabV3。
5. V3+ 作为环境变量选择项可运行。
6. 没有 checkpoint、log、event file、`.pth` 进入 git。
7. 结果报告已写清楚。
8. 代码改动是通用模型支持，不是只针对一次实验的硬编码。

合并前检查：

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
git status --short
git diff --stat
git diff -- network/modeling.py network/_deeplab.py run.sh
```

如果需要合并回原项目，先回到原工作区，处理当前 `main` WIP：

```bash
cd /root/2TStorage/lyc/SegACIL
git status --short
```

原工作区不干净时，不要强行 merge。先由用户决定是提交、stash、还是继续保留 WIP。

---

## 9. 不做的事

本计划明确不做：

1. 不把 `run.sh` 默认模型改成 DeepLabV3+。
2. 不复用 DeepLabV3 的 step0 checkpoint 跑 V3+。
3. 不同时引入新的 RHL normalization 变量。
4. 不改 ResNet101 backbone 类型。
5. 不把 V3+ 实验结果写成核心方法贡献。
6. 不 commit / push，除非用户明确要求。
7. 不覆盖当前原项目 `main` 工作区的 RHL WIP。

---

## 10. 一句话任务

```text
用 git worktree 在 /root/2TStorage/lyc/SegACIL_deeplabv3plus 创建 feature/deeplabv3plus-control，接通 deeplabv3plus_resnet101，使其和 AIR 的 256-d feature 契约兼容；run.sh 默认 DeepLabV3 不变但支持 MODEL 环境变量；完成 DeepLabV3 和 DeepLabV3+ smoke test 后，从 step0 到 step1 跑 VOC 15-5 sequential，并写结果报告。不要污染当前 main，不合并实验产物。
```
