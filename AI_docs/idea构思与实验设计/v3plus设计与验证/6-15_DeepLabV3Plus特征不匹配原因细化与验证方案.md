# DeepLabV3+ 特征不匹配原因细化与验证方案

日期：2026-06-15

适用代码目录：

- 主分析文档所在仓库：`/root/2TStorage/lyc/SegACIL`
- DeepLabV3+ 实验分支 worktree：`/root/2TStorage/lyc/SegACIL_deeplabv3plus`
- 建议执行分支：`feature/deeplabv3plus-control`

前置结论：

目前已经在另一台服务器确认 **step1 batch size 大小基本不会影响精度**。因此，DeepLabV3+ step1 新类 `16-20` 明显低于原 DeepLabV3 的主要原因，基本可以从 batch size 转向 **DeepLabV3+ 提供给 AIR / RecursiveLinear 的特征不匹配**。

本文件面向下一步 Codex 或其他 agent 执行，目标是把现象拆成 3 个最优先原因，并给出对应代码改动和实验方案。

## 1. 当前问题简述

已完成实验：

```text
DeepLabV3+:
  subpath = 20260614_v3plus_voc15-5_seq_bs32-16
  step0 batch size = 32
  step1 batch size = 16
  task = VOC 15-5 sequential
```

结果概要：

| 模型 | step0 0-15 mIoU | step1 0-15 mIoU | step1 16-20 mIoU |
|---|---:|---:|---:|
| DeepLabV3 baseline | 约 0.747-0.748 | 约 0.778-0.780 | 约 0.421-0.432 |
| DeepLabV3+ bs32/16 | 0.7511 | 0.7815 | 0.3959 |

关键现象：

1. DeepLabV3+ 的 step0 旧类略高。
2. DeepLabV3+ 的 step1 旧类也略高。
3. DeepLabV3+ 的 step1 新类明显低。
4. 掉点主要集中在 `pottedplant`、`tvmonitor` 等小物体/背景混淆类。

这说明模型本身没有整体失败，问题集中在 **冻结 DeepLabV3+ 特征后，AIR 闭式学习新类的适配性**。

## 2. 需要先理解的代码机制

### 2.1 step0 是正常监督训练

step0 训练 DeepLabV3+ 时，走正常 segmentation head：

```text
ResNet backbone
  -> layer4 high-level feature
  -> ASPP
  -> upsample to low-level size
  -> concat low-level projected feature
  -> decoder
  -> Linear head
  -> logits
```

代码位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/network/_deeplab.py
DeepLabHeadV3Plus.forward()
```

当前关键代码：

```python
back_out = feature['out']
low_level_feature = self.project(feature['low_level'])
output_feature = self.aspp(back_out)
output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)

decoder_feature = torch.cat([low_level_feature, output_feature], dim=1)
decoder_feature = self.decoder(decoder_feature)

B, C, H, W = decoder_feature.shape
flat_feature = decoder_feature.view(B, C, -1).permute(0, 2, 1)
flat_output = self.head(flat_feature)
heads = flat_output.permute(0, 2, 1).view(B, -1, H, W)
```

### 2.2 step1 不是 SGD，而是 AIR 解析学习

step1 里，代码会把 step0 的普通分类头替换成 `Identity()`：

```python
self.model.classifier.head = nn.Identity()
backbone = self.model
self.model = AIR(...)
```

然后先用 base 数据 realign，再用 step1 新类数据 fit：

```python
for seq, (X, y, _) in enumerate(self.train_loader0):
    self.model.fit(X, y)
self.model.update()

for _, (X, y, _) in enumerate(self.train_loader):
    self.model.fit(X, y)
self.model.update()
```

代码位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/trainer/trainer.py
Trainer.train(), curr_step == 1
```

### 2.3 AIR 实际拿到的是“head 被 Identity 后的特征图”

AIR 里：

```python
X, _ = self.backbone(X)
self.B, self.channle, self.H, self.W = X.shape
X = X.view(self.B, self.channle, -1).permute(0, 2, 1)
return self.buffer(X)
```

也就是说，`self.backbone(X)` 返回什么特征图，AIR 就用什么特征图做随机映射和闭式分类。

对于当前 DeepLabV3+，`classifier.head = Identity()` 后，AIR 拿到的是：

```text
decoder_feature: B x 256 x H/4 x W/4
```

这就是最关键的变化。

## 3. 原因一：DeepLabV3+ decoder feature 与原 AIR 特征分布不一致

### 3.1 面向小白的解释

可以把 step1 的 AIR 理解成一个“冻结特征 + 数学公式分类器”：

```text
图片 -> 冻结的 DeepLab 特征提取器 -> 每个像素的特征向量 -> AIR 直接算出分类器
```

原来的 DeepLabV3 给 AIR 的特征比较“语义化”：

```text
这是车、这是狗、这是人、这是背景
```

DeepLabV3+ 给 AIR 的 decoder feature 混入了更多低层细节：

```text
边缘、纹理、局部形状、颜色变化、细碎边界
```

这些低层细节对 step0 的端到端训练是好事，因为模型可以通过反向传播自己学会如何利用它们。但 step1 的 AIR 不会反向传播训练 decoder，只是在冻结特征上用一个闭式公式直接算分类头。

所以会出现一个反直觉现象：

> step0 更强的 DeepLabV3+ 特征，不一定是 step1 AIR 最适合的特征。

尤其是 `pottedplant`、`tvmonitor` 这类小物体，低层边缘和背景纹理很多，AIR 可能更难把它们和背景/旧类分开。

### 3.2 代码改动方案 A：增加 AIR feature source 开关

目标：让 DeepLabV3+ 在 step0 仍保持原始 decoder head，但 step1 进入 AIR 时，可以选择不同特征源。

建议新增参数：

```text
--air_feature_source
```

可选值：

| 值 | 含义 | 目的 |
|---|---|---|
| `decoder` | 当前默认，使用 stride-4 decoder feature | 复现实验基线 |
| `decoder_stride8` | 使用 decoder feature，但降采样到 stride-8 | 测试是否高分辨率导致问题 |
| `aspp` | 使用 raw ASPP output，接近 DeepLabV3 的 stride-8 语义特征 | 测试 decoder 是否不适配 AIR |
| `aspp_up` | 使用 ASPP feature 上采样到 stride-4，不拼 low-level | 分离 low-level 影响和空间分辨率影响 |

推荐实现位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/network/_deeplab.py
/root/2TStorage/lyc/SegACIL_deeplabv3plus/trainer/trainer.py
/root/2TStorage/lyc/SegACIL_deeplabv3plus/utils/parser.py
```

#### 3.2.1 修改 parser

在 `utils/parser.py` 增加：

```python
air_feature_source: str = "decoder"
```

并加入参数：

```python
parser.add_argument(
    "--air_feature_source",
    type=str,
    default=Config.air_feature_source,
    choices=["decoder", "decoder_stride8", "aspp", "aspp_up"],
    help="feature source used by AIR when DeepLabV3+ classifier head is Identity",
)
```

#### 3.2.2 修改 DeepLabHeadV3Plus

在 `DeepLabHeadV3Plus.__init__()` 中增加：

```python
self.air_feature_source = "decoder"
```

在 `forward()` 中保留 raw ASPP：

```python
raw_aspp_feature = self.aspp(back_out)
output_feature = F.interpolate(
    raw_aspp_feature,
    size=low_level_feature.shape[2:],
    mode="bilinear",
    align_corners=False,
)
```

然后在 `head` 是 `Identity()` 时单独返回 AIR 特征：

```python
if isinstance(self.head, nn.Identity):
    if self.air_feature_source == "decoder":
        air_feature = decoder_feature
    elif self.air_feature_source == "decoder_stride8":
        air_feature = F.avg_pool2d(decoder_feature, kernel_size=2, stride=2)
    elif self.air_feature_source == "aspp":
        air_feature = raw_aspp_feature
    elif self.air_feature_source == "aspp_up":
        air_feature = output_feature
    else:
        raise ValueError(f"Unknown air_feature_source: {self.air_feature_source}")

    return air_feature, {
        "air_feature": air_feature,
        "decoder_feature": decoder_feature,
        "raw_aspp": raw_aspp_feature,
        "aspp_up": output_feature,
        "low_level": low_level_feature,
    }
```

正常 step0 不变，继续走：

```python
flat_feature = decoder_feature.view(B, C, -1).permute(0, 2, 1)
flat_output = self.head(flat_feature)
heads = flat_output.permute(0, 2, 1).view(B, -1, H, W)
```

#### 3.2.3 修改 trainer

在 step1 设置 `Identity()` 后，把参数写入 classifier：

```python
self.model.classifier.head = nn.Identity()
if hasattr(self.model.classifier, "air_feature_source"):
    self.model.classifier.air_feature_source = self.opts.air_feature_source
```

建议同时打印：

```python
print(f"AIR feature source: {self.opts.air_feature_source}")
```

### 3.3 对应实验 A：特征源对照实验

前提：复用现有 DeepLabV3+ step0 checkpoint，不需要重新训练 step0。

基础 step0：

```text
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16
```

每次新实验只跑 step1：

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus

MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
START_STEP=1 \
END_STEP=1 \
SUBPATH=20260615_v3plus_air_aspp \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
DEFAULT_BATCH_SIZE=16 \
AIR_FEATURE_SOURCE=aspp \
bash run_v3plus_air.sh
```

需要新增或改造脚本 `run_v3plus_air.sh`，使其支持：

```bash
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-decoder}"
...
--air_feature_source "$AIR_FEATURE_SOURCE"
```

实验矩阵：

| 实验名 | `air_feature_source` | 目的 |
|---|---|---|
| `v3plus_air_decoder` | `decoder` | 复现当前低结果 |
| `v3plus_air_decoder_stride8` | `decoder_stride8` | 只改变空间分辨率 |
| `v3plus_air_aspp` | `aspp` | 改为 stride-8 高层语义特征 |
| `v3plus_air_aspp_up` | `aspp_up` | 保持 stride-4，但不混 low-level |

判定标准：

| 现象 | 解释 |
|---|---|
| `aspp` 明显高于 `decoder` | decoder/low-level 特征不适合 AIR |
| `decoder_stride8` 明显高于 `decoder` | 高分辨率 dense fitting 是主因 |
| `aspp_up` 低但 `aspp` 高 | 空间分辨率比语义源更关键 |
| 四者都差不多低 | 特征源不是唯一问题，继续看原因二/三 |

成功目标：

```text
16-20 mIoU >= 0.421
0-15 mIoU 不低于 0.775
```

## 4. 原因二：stride-4 dense pixel 放大背景/旧类像素主导

### 4.1 面向小白的解释

AIR 在 step1 做的事情不是“每张图片算一次”，而是“每个像素都算一次”。

一张 VOC 图片里，通常背景像素非常多，新类物体像素很少。例如一张有 `pottedplant` 的图片，可能大部分都是背景、桌子、墙、旧类物体，真正属于 `pottedplant` 的像素只占一小块。

DeepLabV3 的特征图大约是 stride-8：

```text
513x513 图片 -> 约 65x65 特征点 -> 约 4k 个像素特征
```

DeepLabV3+ 当前 AIR 用的是 stride-4 decoder feature：

```text
513x513 图片 -> 约 129x129 特征点 -> 约 16k 个像素特征
```

也就是说，DeepLabV3+ 每张图送给 AIR 的像素特征数量大约是 DeepLabV3 的 4 倍。

这听起来像是“信息更多”，但对类别不平衡任务未必是好事。因为多出来的像素里，绝大部分也可能是背景或旧类。结果是：

```text
背景和旧类声音变大
小新类声音被淹没
```

这正好解释当前现象：

- 旧类 `0-15` 没掉，甚至略涨；
- 新类 `16-20` 掉；
- 小物体 `pottedplant`、`tvmonitor` 掉得最明显；
- 大物体 `train` 没掉。

### 4.2 代码改动方案 B1：AIR feature downsample

这是原因二的最小验证方案，和原因一的 `decoder_stride8` 可以合并实现。

最小实现：

```python
elif self.air_feature_source == "decoder_stride8":
    air_feature = F.avg_pool2d(decoder_feature, kernel_size=2, stride=2)
```

优点：

- 改动小；
- 不改变 step0；
- 不改变 AIR 数学；
- 直接验证“stride-4 太密”这个假设。

缺点：

- 只减少像素数量，不解决类别不均衡；
- 如果低层 decoder 特征本身不适合 AIR，降采样也未必足够。

### 4.3 代码改动方案 B2：像素级类别均衡采样

目标：在 `RecursiveLinear.fit()` 中，不让背景/旧类像素数量远远超过小新类。

建议新增参数：

```text
--air_pixel_balance none | class_cap | fg_bg
--air_max_pixels_per_class 4096
--air_bg_max_pixels 8192
```

推荐先实现 `class_cap`，最容易验证。

#### 4.3.1 修改 parser

```python
air_pixel_balance: str = "none"
air_max_pixels_per_class: int = 0
air_bg_max_pixels: int = 0
```

参数：

```python
parser.add_argument(
    "--air_pixel_balance",
    type=str,
    default=Config.air_pixel_balance,
    choices=["none", "class_cap", "fg_bg"],
)
parser.add_argument("--air_max_pixels_per_class", type=int, default=Config.air_max_pixels_per_class)
parser.add_argument("--air_bg_max_pixels", type=int, default=Config.air_bg_max_pixels)
```

#### 4.3.2 让 AIR 把参数传给 RecursiveLinear

当前：

```python
self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
```

可以改成：

```python
self.analytic_linear = linear(
    buffer_size,
    gamma,
    pixel_balance=pixel_balance,
    max_pixels_per_class=max_pixels_per_class,
    bg_max_pixels=bg_max_pixels,
    **factory_kwargs,
)
```

注意：这会修改 `RecursiveLinear.__init__()` 签名。为了不影响旧路径，默认值必须保持关闭：

```python
pixel_balance="none"
max_pixels_per_class=0
bg_max_pixels=0
```

#### 4.3.3 在 RecursiveLinear.fit() 中采样

位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/network/AnalyticLinear.py
RecursiveLinear.fit()
```

当前代码是：

```python
mask = y != 255
X = X[mask]
y = y[mask]
```

建议在这之后增加：

```python
if self.pixel_balance == "class_cap" and self.max_pixels_per_class > 0:
    keep_indices = []
    for cls in torch.unique(y):
        cls_idx = torch.nonzero(y == cls, as_tuple=False).flatten()
        if cls_idx.numel() > self.max_pixels_per_class:
            perm = torch.randperm(cls_idx.numel(), device=cls_idx.device)
            cls_idx = cls_idx[perm[:self.max_pixels_per_class]]
        keep_indices.append(cls_idx)
    keep_indices = torch.cat(keep_indices, dim=0)
    X = X[keep_indices]
    y = y[keep_indices]
```

进一步版本 `fg_bg`：

```python
background = 0
foreground_count = (y != background).sum()
background_cap = min(self.bg_max_pixels, int(foreground_count * bg_ratio))
```

但第一轮不建议做太复杂，先做 `class_cap`。

### 4.4 对应实验 B：像素均衡实验

只跑 step1，复用 DeepLabV3+ step0。

实验矩阵：

| 实验名 | feature source | pixel balance | max pixels/class | 目的 |
|---|---|---|---:|---|
| `decoder_none` | `decoder` | `none` | 0 | 当前基线 |
| `decoder_cap4096` | `decoder` | `class_cap` | 4096 | 看类别均衡是否救新类 |
| `decoder_cap8192` | `decoder` | `class_cap` | 8192 | 看 cap 强度 |
| `decoder_stride8_none` | `decoder_stride8` | `none` | 0 | 只降采样 |
| `decoder_stride8_cap4096` | `decoder_stride8` | `class_cap` | 4096 | 降采样 + 均衡 |
| `aspp_cap4096` | `aspp` | `class_cap` | 4096 | 语义特征 + 均衡 |

命令模板：

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus

MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
START_STEP=1 \
END_STEP=1 \
SUBPATH=20260615_v3plus_decoder_cap4096 \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
DEFAULT_BATCH_SIZE=16 \
AIR_FEATURE_SOURCE=decoder \
AIR_PIXEL_BALANCE=class_cap \
AIR_MAX_PIXELS_PER_CLASS=4096 \
bash run_v3plus_air.sh
```

判定标准：

| 现象 | 解释 |
|---|---|
| class cap 后 `pottedplant/tvmonitor` 明显涨 | 背景/旧类像素主导是主因 |
| 新类涨但旧类掉很多 | 均衡太强，需要提高 cap 或只限制背景 |
| 无明显变化 | 类别像素数量不是主要矛盾，继续看特征源和 gamma |

建议记录：

```text
0-15 mIoU
16-20 mIoU
pottedplant IoU/Acc
tvmonitor IoU/Acc
train IoU/Acc
```

不要只看 Mean IoU，因为 Mean IoU 可能掩盖新类变化。

## 5. 原因三：gamma / buffer 对 DeepLabV3+ 特征不再合适

### 5.1 面向小白的解释

AIR 的闭式学习可以粗略理解成：

```text
找到一个分类器，让所有像素特征尽量分到正确类别
```

但如果特征数量非常多、特征之间高度相似、背景像素很多，数学公式可能会“太相信训练像素”，导致分类边界偏向大类/旧类。

`gamma` 可以理解成一个“刹车”：

```text
gamma 小：更努力拟合已有像素，可能过拟合大类/背景
gamma 大：更保守，分类器不会被某些像素模式带偏太多
```

`buffer=8196` 是随机特征维度，可以理解成把 256 维特征随机扩展到 8196 维。维度越高，表达能力越强，但也可能更容易让解析解受数据分布和数值条件影响。

原 DeepLabV3 用 `gamma=1, buffer=8196` 可行，不代表 DeepLabV3+ 也最合适。因为 DeepLabV3+ 的特征：

- 分辨率更高；
- 混入低层纹理；
- 像素相关性更强；
- 小类被背景包围更多。

因此，DeepLabV3+ 可能需要重新扫描 `gamma`，甚至调整 `buffer`。

### 5.2 代码改动方案 C1：先不改代码，做 gamma 扫描

当前训练入口已经支持：

```text
--gamma
--buffer
```

如果 `run_v3plus_air.sh` 还没有 env 化，需要加：

```bash
GAMMA="${GAMMA:-1}"
BUFFER="${BUFFER:-8196}"
...
--gamma "$GAMMA"
--buffer "$BUFFER"
```

第一轮只扫 gamma，不动 buffer：

| 实验名 | gamma | buffer |
|---|---:|---:|
| `g0p1` | 0.1 | 8196 |
| `g1` | 1 | 8196 |
| `g10` | 10 | 8196 |
| `g100` | 100 | 8196 |

命令模板：

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus

MODEL=deeplabv3plus_resnet101 \
TASK=15-5 \
SETTING=sequential \
START_STEP=1 \
END_STEP=1 \
SUBPATH=20260615_v3plus_aspp_g10 \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
DEFAULT_BATCH_SIZE=16 \
AIR_FEATURE_SOURCE=aspp \
GAMMA=10 \
BUFFER=8196 \
bash run_v3plus_air.sh
```

推荐不要在最开始对所有 feature source 都扫 gamma。先选两个最有代表性的：

```text
decoder
aspp
```

如果 `aspp` 已经明显恢复，就优先对 `aspp` 扫 gamma。

### 5.3 代码改动方案 C2：增加 feature normalization

如果 gamma 扫描显示某些 gamma 能改善，但不稳定，说明特征尺度/分布也有问题。

建议增加：

```text
--air_feature_norm none | l2 | channel
```

#### 5.3.1 在 AIR.feature_expansion() 里实现

位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/trainer/trainer.py
AIR.feature_expansion()
```

当前：

```python
X, _ = self.backbone(X)
self.B, self.channle, self.H, self.W = X.shape
X = X.view(self.B, self.channle, -1).permute(0, 2, 1)
return self.buffer(X)
```

建议改成：

```python
X, _ = self.backbone(X)

if self.feature_norm == "l2":
    X = F.normalize(X, p=2, dim=1)
elif self.feature_norm == "channel":
    mean = X.mean(dim=(2, 3), keepdim=True)
    std = X.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    X = (X - mean) / std

self.B, self.channle, self.H, self.W = X.shape
X = X.view(self.B, self.channle, -1).permute(0, 2, 1)
return self.buffer(X)
```

同时修改 `AIR.__init__()`：

```python
def __init__(..., feature_norm="none"):
    self.feature_norm = feature_norm
```

以及 trainer step1：

```python
self.model = AIR(
    ...,
    feature_norm=self.opts.air_feature_norm,
)
```

parser 增加：

```python
parser.add_argument(
    "--air_feature_norm",
    type=str,
    default="none",
    choices=["none", "l2", "channel"],
)
```

### 5.4 对应实验 C：gamma + norm

第一轮：

| 实验名 | feature source | norm | gamma |
|---|---|---|---:|
| `decoder_g1` | decoder | none | 1 |
| `decoder_g10` | decoder | none | 10 |
| `decoder_g100` | decoder | none | 100 |
| `aspp_g1` | aspp | none | 1 |
| `aspp_g10` | aspp | none | 10 |
| `aspp_g100` | aspp | none | 100 |

第二轮，如果 gamma 有改善：

| 实验名 | feature source | norm | gamma |
|---|---|---|---:|
| `aspp_l2_g10` | aspp | l2 | 10 |
| `aspp_channel_g10` | aspp | channel | 10 |
| `decoder_stride8_l2_g10` | decoder_stride8 | l2 | 10 |

判定标准：

| 现象 | 解释 |
|---|---|
| gamma 变大后新类涨 | 原 `gamma=1` 正则不足 |
| norm 后新类涨且更稳定 | 特征尺度/分布是重要问题 |
| gamma/norm 只涨新类但旧类大掉 | 需要折中或类别均衡 |
| 全部无效 | 主因更可能是 feature source 或像素不均衡 |

## 6. 推荐执行顺序

不要一次性把所有改动叠在一起，否则无法判断到底是哪一项起作用。

推荐顺序如下：

### 阶段 0：只加诊断与开关，不改变默认行为

代码目标：

1. 增加 `--air_feature_source`，默认 `decoder`。
2. 增加 `--air_feature_norm`，默认 `none`。
3. 增加 `--air_pixel_balance`，默认 `none`。
4. 新增 `run_v3plus_air.sh`，用于只跑 step1 并复用 step0。
5. 默认参数必须复现当前 `0.3959` 左右结果。

验证：

```text
air_feature_source=decoder
air_feature_norm=none
air_pixel_balance=none
gamma=1
```

如果这个结果不能复现当前实验，说明改动破坏了默认路径，必须先修。

### 阶段 1：验证 feature source

只跑：

```text
decoder
decoder_stride8
aspp
aspp_up
```

最关键观察：

```text
16-20 mIoU 是否从 0.3959 回到 0.421+
```

### 阶段 2：验证 dense pixel / 类别不均衡

在阶段 1 的最佳 feature source 上做：

```text
none
class_cap=4096
class_cap=8192
```

重点观察：

```text
pottedplant
tvmonitor
0-15 mIoU 是否保持
```

### 阶段 3：验证 gamma / norm

在阶段 1 或 2 的最佳方案上扫：

```text
gamma = 0.1, 1, 10, 100
norm = none, l2, channel
```

### 阶段 4：组合最佳方案

例如可能的最终候选：

```text
air_feature_source=aspp
air_pixel_balance=class_cap
air_max_pixels_per_class=4096
air_feature_norm=l2
gamma=10
buffer=8196
```

最后与原 DeepLabV3 baseline 对照：

| 指标 | 目标 |
|---|---:|
| 0-15 mIoU | >= 0.778 |
| 16-20 mIoU | >= 0.421 |
| Mean IoU | >= 0.694 |
| pottedplant IoU | 明显高于 0.174 |
| tvmonitor IoU | 明显高于 0.249 |

## 7. 建议新脚本结构

建议新增：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/run_v3plus_air.sh
```

脚本核心：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA_ROOT="${DATA_ROOT:-/root/2TStorage/lyc/SegACIL/data_root/VOC2012}"
MODEL="${MODEL:-deeplabv3plus_resnet101}"
TASK="${TASK:-15-5}"
SETTING="${SETTING:-sequential}"
SUBPATH="${SUBPATH:-$(date +%Y%m%d)_v3plus_air}"
BASE_SUBPATH="${BASE_SUBPATH:-20260614_v3plus_voc15-5_seq_bs32-16}"

START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-1}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-16}"
TRAIN_EPOCH="${TRAIN_EPOCH:-50}"

GAMMA="${GAMMA:-1}"
BUFFER="${BUFFER:-8196}"
OUTPUT_STRIDE="${OUTPUT_STRIDE:-8}"

AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-decoder}"
AIR_FEATURE_NORM="${AIR_FEATURE_NORM:-none}"
AIR_PIXEL_BALANCE="${AIR_PIXEL_BALANCE:-none}"
AIR_MAX_PIXELS_PER_CLASS="${AIR_MAX_PIXELS_PER_CLASS:-0}"
AIR_BG_MAX_PIXELS="${AIR_BG_MAX_PIXELS:-0}"

python train.py \
  --data_root "$DATA_ROOT" \
  --model "$MODEL" \
  --lr 0.01 \
  --batch_size "$DEFAULT_BATCH_SIZE" \
  --loss_type bce_loss \
  --dataset voc \
  --task "$TASK" \
  --lr_policy poly \
  --curr_step 1 \
  --subpath "$SUBPATH" \
  --base_subpath "$BASE_SUBPATH" \
  --method acil \
  --setting "$SETTING" \
  --pretrained_backbone \
  --crop_val \
  --train_epoch "$TRAIN_EPOCH" \
  --gamma "$GAMMA" \
  --buffer "$BUFFER" \
  --output_stride "$OUTPUT_STRIDE" \
  --air_feature_source "$AIR_FEATURE_SOURCE" \
  --air_feature_norm "$AIR_FEATURE_NORM" \
  --air_pixel_balance "$AIR_PIXEL_BALANCE" \
  --air_max_pixels_per_class "$AIR_MAX_PIXELS_PER_CLASS" \
  --air_bg_max_pixels "$AIR_BG_MAX_PIXELS"
```

如果 parser 暂时还没实现后几个参数，脚本也要等代码改完再启用。

## 8. 建议给下一步 Codex 的执行任务

可以直接把下面任务交给下一步 agent：

```text
请在 /root/2TStorage/lyc/SegACIL_deeplabv3plus 的 feature/deeplabv3plus-control 分支上实现以下最小改动：

1. 增加 --air_feature_source，支持 decoder / decoder_stride8 / aspp / aspp_up。
2. 修改 DeepLabHeadV3Plus.forward，使 head 为 nn.Identity 时按 air_feature_source 返回 AIR 特征；默认 decoder 行为必须复现当前结果。
3. 增加 --air_feature_norm，支持 none / l2 / channel，在 AIR.feature_expansion 中实现。
4. 增加 --air_pixel_balance、--air_max_pixels_per_class、--air_bg_max_pixels，默认 none/0，不改变旧行为。
5. 在 RecursiveLinear.fit 中实现 class_cap 采样，默认关闭。
6. 新增 run_v3plus_air.sh，支持复用 BASE_SUBPATH 的 step0，只跑 step1。
7. 做 smoke test：DeepLabV3+ step0 normal logits shape 正常；head=Identity 后 decoder/aspp/decoder_stride8 的 AIR feature shape 正常。
8. 跑一个默认 decoder/none/gamma=1 的 step1 复现实验，确认 16-20 mIoU 接近 0.3959；然后再启动 feature_source 对照实验。
```

## 9. 风险与注意事项

1. **不要改 step0 默认训练路径。** DeepLabV3+ step0 已经略优，当前问题在 step1 AIR。
2. **所有新参数默认必须保持旧行为。** 这样可以安全比较。
3. **不要一开始叠加多个改动。** 先 feature source，再 pixel balance，再 gamma/norm。
4. **复用 step0 checkpoint 时必须用 `BASE_SUBPATH`。** 避免覆盖已有 step0。
5. **重点看新类和类别级指标。** Mean IoU 不足以判断方案好坏。
6. **显存仍然要保守。** 当前目标不是跑更大 batch，而是让 AIR 特征更合适。

## 10. 预期最可能成功的方向

按目前证据，最可能有效的方向排序：

1. `air_feature_source=aspp`：让 AIR 使用更接近原 DeepLabV3 的 stride-8 高层语义特征。
2. `air_feature_source=decoder_stride8`：保留 decoder 语义，但减少 stride-4 dense pixel 造成的不均衡。
3. 在最佳 feature source 上加 `class_cap`：缓解背景/旧类像素主导。
4. 在最佳 feature source 上扫 `gamma`：寻找适合 DeepLabV3+ 特征的正则强度。
5. 如仍不稳定，再加 `l2/channel` feature normalization。

最理想结果是：

```text
DeepLabV3+ step0 仍保留略高的 0-15 mIoU；
step1 旧类不下降；
step1 新类 16-20 回到或超过 DeepLabV3 baseline；
最终说明 DeepLabV3+ 可以作为公平增强 backbone，但必须调整 AIR 使用的特征接口。
```

