# DeepLabV3+ 特征不匹配原因细化与验证方案

> 状态（2026-06-23）：阶段 0-2 已执行并完成代码审查；主因已确认，最终 DeepLabV3+ AIR 配置已收敛。阶段 3 的 gamma/norm 扫描因缺少触发证据而停止。

日期：2026-06-15

适用代码目录：

- 主分析文档所在仓库：`/root/2TStorage/lyc/SegACIL`
- DeepLabV3+ 实验分支 worktree：`/root/2TStorage/lyc/SegACIL_deeplabv3plus`
- 建议执行分支：`feature/deeplabv3plus-control`

前置结论：

目前已经在另一台服务器确认 **step1 batch size 大小基本不会影响精度**。因此，DeepLabV3+ step1 新类 `16-20` 明显低于原 DeepLabV3 的主要原因，基本可以从 batch size 转向 **DeepLabV3+ 提供给 AIR / RecursiveLinear 的特征不匹配**。

本文件面向下一步 Codex 或其他 agent 执行，目标是把现象拆成 3 个最优先原因，并给出对应代码改动和实验方案。

## 0. 2026-06-22 审查结论与方案修订

### 0.1 原因判断是否成立

截至 2026-06-22，另一台服务器已经完成同模型不同 batch size 的对照，确认 step1 batch size 不会造成这次数个点的精度差异。因此本文的主判断成立：问题应优先定位到 **DeepLabV3+ 给 AIR 的特征语义、空间密度和随机特征统计**，而不是继续追求 bs32。

三个原因的修订后优先级如下：

| 优先级 | 原因 | 当前证据 | 验证方式 |
|---:|---|---|---|
| 1 | AIR 隐式取得 stride-4 decoder feature，而原 DeepLabV3 使用 stride-8 高层语义特征 | 旧类略涨、新类小物体明显下降；代码确认特征源发生变化 | 同一 step0 checkpoint 对比 `decoder/aspp/decoder_stride8/aspp_up` |
| 2 | stride-4 dense pixel 使背景和大类贡献进一步占优 | `pottedplant/tvmonitor` 掉点最大，`train` 不降 | 精确降采样；必要时在 RandomBuffer 前做确定性像素采样 |
| 3 | decoder 特征尺度、相关性与随机映射/岭正则不匹配 | 仍是合理假设，但已有 RHL 实验表明小范围 gamma 扫描常常几乎不改变结果 | 先记录特征统计；只在特征源确定后再做 norm/gamma/buffer 实验 |

### 0.2 “最小改动”审查

原第 8 节要求“实现以下最小改动”，其中把 `classifier.head` 替换为 `nn.Identity`，再在 `DeepLabHeadV3Plus.forward()` 里通过 `isinstance(self.head, nn.Identity)` 判断是否进入 AIR 特征模式。这个做法虽然代码行数少，但属于 AI Agent 常见的防御性补丁：

1. 分类头类型被偷偷用作运行模式开关，接口语义不明确。
2. 普通分割前向和 AIR 特征抽取耦合在同一个条件分支中。
3. 后续更换 head、导出模型或做测试时，很难知道 `Identity` 代表“不要分类”还是“进入 AIR”。
4. 每增加一种特征源，`forward()` 的隐式状态分支会继续膨胀。

因此，**当前情况不适合以最少代码行数为目标**。适合的是“最小完整设计”：只建立本问题确实需要的显式特征接口，不提前实现所有调参选项。

推荐接口：

```text
DeepLabHeadV3Plus.extract_features(backbone_features)
    -> {decoder, decoder_stride8, aspp, aspp_up, low_level}

DeepLabHeadV3Plus.select_air_feature(feature_dict, source)
    -> B x 256 x H x W

_SimpleSegmentationModel.forward_air_features(image, source)
    -> B x 256 x H x W
```

普通 `forward()` 继续只负责 logits；AIR 显式调用 `forward_air_features()`。这样改动比 `Identity` 判断多一些，但它修复的是实际接口缺陷，测试边界也更清楚，不属于过度设计。

### 0.3 原方案中需要纠正的实现细节

1. `decoder_stride8` 不应固定 `avg_pool2d(kernel_size=2)`。513 输入时 stride-4/stride-8 特征尺寸可能是 129 和 65，固定池化会得到 64。应使用 `adaptive_avg_pool2d(decoder, aspp.shape[-2:])` 精确对齐。
2. 类别采样不应放在 `RecursiveLinear.fit()` 且发生在 RandomBuffer 之后。当前大显存来自 `B*H*W*8196` 的展开；晚采样不能降低峰值显存，也把分割数据策略塞进了通用线性层。采样应在 `AIR.fit()` 中、256 维特征进入 RandomBuffer 之前完成。
3. 不应在阶段 0 一次性实现 feature source、normalization、pixel balance 三套功能。先完成显式特征接口并得到阶段 1 结果，后续代码依赖真实结果决定。
4. `gamma=0.1/1/10` 在已有 RHL 实验中几乎等价，说明 `E^T E` 可能已经压过 `gamma I`。gamma 只应在特征源改变后、且统计显示条件数/尺度有问题时再扫描。
5. 所有对照必须复用同一个 step0 checkpoint，并保存到独立 `SUBPATH`；否则不能把变化归因于 AIR 特征源。

### 0.4 修订后的阶段门禁

```text
阶段 0：显式 AIR feature API + shape/stat smoke tests；parser 兼容默认 `decoder` 与旧行为等价。
阶段 1：只实现并运行 feature source 对照；先拿到结果。
阶段 2：若 stride8/aspp 改善，再围绕最佳源做像素采样；若都无改善，优先检查语义可分性而不是盲加采样。
阶段 3：仅在统计证据支持时实现 normalization / gamma / buffer 实验。
阶段 4：组合经过单因素验证的有效项，做最终复验。
```

每个阶段必须独立提交、完成代码审查并记录实验配置。不得为了“先把接口留好”提前加入未验证的分支。

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
| `decoder` | parser 兼容默认，使用 stride-4 decoder feature | 复现实验基线 |
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

原草案曾建议在 `head` 是 `Identity()` 时单独返回 AIR 特征。审查后不再采用该方案，改为显式特征接口：

```python
def extract_features(self, feature):
    back_out = feature["out"]
    low_level = self.project(feature["low_level"])
    aspp = self.aspp(back_out)
    aspp_up = F.interpolate(aspp, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
    decoder = self.decoder(torch.cat([low_level, aspp_up], dim=1))
    decoder_stride8 = F.adaptive_avg_pool2d(decoder, aspp.shape[-2:])
    return {
        "decoder": decoder,
        "decoder_stride8": decoder_stride8,
        "aspp": aspp,
        "aspp_up": aspp_up,
        "low_level": low_level,
    }

def select_air_feature(self, features, source):
    if source not in {"decoder", "decoder_stride8", "aspp", "aspp_up"}:
        raise ValueError(f"Unknown AIR feature source: {source}")
    return features[source]
```

正常 step0 不变，继续走：

```python
flat_feature = decoder_feature.view(B, C, -1).permute(0, 2, 1)
flat_output = self.head(flat_feature)
heads = flat_output.permute(0, 2, 1).view(B, -1, H, W)
```

#### 3.2.3 修改 trainer

在模型层增加显式 AIR 特征前向：

```python
def forward_air_features(self, x, source):
    backbone_features = self.backbone(x)
    features = self.classifier.extract_features(backbone_features)
    return self.classifier.select_air_feature(features, source)
```

AIR 保存完整模型和 `feature_source`，并显式调用：

```python
X = self.backbone.forward_air_features(X, self.feature_source)
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
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-aspp_up}"
...
--air_feature_source "$AIR_FEATURE_SOURCE"
```

阶段 0 为复现旧行为时应显式设置 `AIR_FEATURE_SOURCE=decoder`；完成阶段 2 后，专用脚本默认已收敛为 `aspp_up`，parser 本身仍默认 `decoder`。

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

最小完整实现：

```python
decoder_stride8 = F.adaptive_avg_pool2d(decoder_feature, raw_aspp_feature.shape[-2:])
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

目标：在 `AIR.fit()` 中、进入 RandomBuffer 之前，不让背景/旧类像素数量远远超过小新类，同时真正降低随机扩展的峰值显存。

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

#### 4.3.2 让 AIR 在低维特征上采样

当前流程：

```python
256-d feature map -> RandomBuffer 8196-d -> RecursiveLinear.fit()
```

应改成：

```python
256-d feature map -> resize labels -> flatten/mask -> deterministic pixel sampling
                  -> RandomBuffer 8196-d -> RecursiveLinear.fit()
```

采样策略属于分割任务的 AIR，不修改通用 `RecursiveLinear.__init__()`。为了不影响旧路径，默认值必须保持关闭：

```python
pixel_balance="none"
max_pixels_per_class=0
bg_max_pixels=0
```

#### 4.3.3 在 AIR.fit() 中采样

位置：

```text
/root/2TStorage/lyc/SegACIL_deeplabv3plus/trainer/trainer.py
AIR.fit()
```

建议在 256 维特征展平后采样，再调用 RandomBuffer：

```python
features = self.extract_backbone_features(X)
labels = self.resize_and_flatten_labels(y, features.shape[-2:])
features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
mask = labels != 255
features, labels = features[mask], labels[mask]
```

建议在这之后增加：

```python
if self.pixel_balance == "class_cap" and self.max_pixels_per_class > 0:
    keep_indices = []
    for cls in torch.unique(y):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        if cls_idx.numel() > self.max_pixels_per_class:
            perm = torch.randperm(cls_idx.numel(), device=cls_idx.device)
            cls_idx = cls_idx[perm[:self.max_pixels_per_class]]
        keep_indices.append(cls_idx)
    keep_indices = torch.cat(keep_indices, dim=0)
    features = features[keep_indices]
    labels = labels[keep_indices]

expanded = self.buffer(features)
self.analytic_linear.fit(expanded, labels)
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

### 阶段 0：建立显式 AIR 特征接口，不改变默认行为

代码目标：

1. 增加 `--air_feature_source`，parser 默认 `decoder`。
2. 为 DeepLabV3+ 建立 `extract_features/select_air_feature/forward_air_features` 显式接口。
3. AIR 改为显式调用 `forward_air_features()`，不再依赖 `head=Identity` 传递运行模式。
4. 增加 shape、默认等价性和非法 source 测试。
5. 新增 `run_v3plus_air.sh`，用于只跑 step1 并复用 step0。
6. 默认参数必须复现当前 `0.3959` 左右结果。

验证：

```text
air_feature_source=decoder
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

### 阶段 2：根据阶段 1 结果验证 dense pixel / 类别不均衡

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

### 阶段 3：有统计证据时再验证 gamma / norm

在阶段 1 或 2 的最佳方案上扫：

```text
gamma = 0.1, 1, 10, 100
norm = none, l2, channel
```

### 阶段 4：组合最佳方案

阶段 1、2 完成后的实际最终候选为：

```text
air_feature_source=aspp_up
air_pixel_balance=none
air_max_pixels_per_class=0
gamma=1
buffer=8196
```

这是已经跑出结果的配置，不再是假设组合。`class_cap` 会在提升部分稀有类召回的同时明显损伤旧类 IoU，因此不进入最终默认；gamma/norm 没有满足阶段 3 门禁，也不进入组合。

最后与原 DeepLabV3 baseline 对照：

| 指标 | 目标 |
|---|---:|
| 0-15 mIoU | >= 0.778 |
| 16-20 mIoU | >= 0.421 |
| Mean IoU | >= 0.694 |
| pottedplant IoU | 明显高于 0.174 |
| tvmonitor IoU | 明显高于 0.249 |

## 7. 已实现的脚本结构

| 脚本 | 用途 |
|---|---|
| `run_v3plus_air.sh` | 单组 step1 实验；复用 `BASE_SUBPATH`，要求显式设置唯一 `SUBPATH` |
| `run_v3plus_air_sources.sh` | 顺序运行四种 feature source 单因素对照 |
| `run_v3plus_air_pixel_caps.sh` | 顺序复现 `aspp_up + cap4096/cap8192` |

最终推荐配置可直接运行：

```bash
SUBPATH=20260623_v3plus_air_aspp_up_none_bs8 \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
BATCH_SIZE=8 \
bash run_v3plus_air.sh
```

`run_v3plus_air.sh` 当前默认：

```text
AIR_FEATURE_SOURCE=aspp_up
AIR_PIXEL_BALANCE=none
AIR_MAX_PIXELS_PER_CLASS=0
GAMMA=1
BUFFER=8196
```

脚本拒绝省略 `SUBPATH`，各 sweep 脚本还会在 checkpoint 目录已存在时退出，防止覆盖正式实验产物。

## 8. 原执行清单与完成状态

下面是本轮使用的原执行清单，保留用于追溯，不应再交给下一步 agent 重复执行：

```text
请在 /root/2TStorage/lyc/SegACIL_deeplabv3plus 的 feature/deeplabv3plus-control 分支上按阶段实现。这里的目标是最小完整设计，不是最少代码行数：

阶段 0：
1. 增加 --air_feature_source，支持 decoder / decoder_stride8 / aspp / aspp_up，parser 默认 decoder。
2. 在 DeepLabHeadV3Plus 中建立 extract_features() 和 select_air_feature()；在模型层建立 forward_air_features()。不要用 head 是否为 nn.Identity 判断 AIR 模式。
3. decoder_stride8 使用 adaptive_avg_pool2d 精确匹配 raw ASPP 尺寸。
4. AIR 显式调用 forward_air_features()，普通 step0 forward 行为不变。
5. 新增 run_v3plus_air.sh，支持 BASE_SUBPATH 复用 step0、只跑 step1。
6. 添加自动测试：四个 source 的 shape、普通 logits 不变、parser 默认 decoder 与旧 decoder feature 数值等价、非法 source 报错。
7. 提交并代码审查。

阶段 1：
8. 先运行 decoder/gamma=1 复现实验，再运行 decoder_stride8、aspp、aspp_up 单因素对照；每个实验使用相同 step0 checkpoint 和独立 SUBPATH。
9. 汇总 0-15、16-20、pottedplant、tvmonitor、train 指标，并根据结果决定阶段 2。

阶段 2（只有阶段 1 结果支持时执行）：
10. 在 AIR.fit() 的 256 维特征进入 RandomBuffer 前实现确定性 class_cap 采样；不要把分割采样逻辑放入 RecursiveLinear。
11. 默认关闭，添加采样数量、确定性和旧行为等价测试；提交并代码审查。

阶段 3（只有特征统计支持时执行）：
12. 再实现 feature norm 或有针对性的 gamma/buffer 扫描，不做无证据的参数穷举。
```

截至 2026-06-23：阶段 0、1、2 已完成并分别提交、测试和审查；阶段 3 因门禁未触发而有依据地跳过。后续只执行第 12.7 节列出的论文级复验，不重新展开本清单。

## 9. 风险与注意事项

1. **不要改 step0 默认训练路径。** DeepLabV3+ step0 已经略优，当前问题在 step1 AIR。
2. **所有新参数默认必须保持旧行为。** 这样可以安全比较。
3. **不要一开始叠加多个改动。** 先 feature source，再 pixel balance，再 gamma/norm。
4. **复用 step0 checkpoint 时必须用 `BASE_SUBPATH`。** 避免覆盖已有 step0。
5. **重点看新类和类别级指标。** Mean IoU 不足以判断方案好坏。
6. **显存仍然要保守。** 当前目标不是跑更大 batch，而是让 AIR 特征更合适。

## 10. 预期最可能成功的方向

验证后的方向排序：

1. `air_feature_source=aspp_up`：已验证为最佳，兼顾高层语义与 stride-4 标签对齐。
2. `air_feature_source=aspp`：显存和运行时间更低，精度略低于 `aspp_up`，可作为资源受限备选。
3. `class_cap`：只保留为实验性显存/召回方案，不能作为最高精度配置。
4. gamma/norm：当前没有统计或结果证据支持继续扫描，停止而不是继续堆参数。

最理想结果是：

```text
DeepLabV3+ step0 仍保留略高的 0-15 mIoU；
step1 旧类不下降；
step1 新类 16-20 回到或超过 DeepLabV3 baseline；
最终说明 DeepLabV3+ 可以作为公平增强 backbone，但必须调整 AIR 使用的特征接口。
```

## 11. 2026-06-22 阶段 1 实验结果

所有实验复用同一 step0 checkpoint：

```text
20260614_v3plus_voc15-5_seq_bs32-16
```

固定参数：

```text
batch_size=16
buffer=8196
gamma=1
random_seed=1
```

结果：

| AIR feature source | Mean IoU | 0-15 mIoU | 16-20 mIoU | pottedplant | tvmonitor | train |
|---|---:|---:|---:|---:|---:|---:|
| `decoder` | 0.6897 | 0.7815 | 0.3959 | 0.1742 | 0.2491 | 0.7079 |
| `decoder_stride8` | 0.6860 | 0.7750 | 0.4011 | 0.1896 | 0.2565 | 0.7101 |
| `aspp` | 0.6995 | 0.7771 | 0.4510 | 0.1962 | 0.3528 | 0.7506 |
| `aspp_up` | **0.7036** | **0.7793** | **0.4613** | **0.2131** | **0.3757** | **0.7534** |

### 11.1 接口等价性验证通过

新显式接口的兼容模式 `decoder` 得到：

```text
Mean IoU=0.6897
0-15=0.7815
16-20=0.3959
```

它与改造前实验完全一致，证明阶段 0 没有改变默认数学路径，差异可以归因于 feature source。

### 11.2 主因已经确认

从 `decoder` 切换到 `aspp`，新类提升约 5.51 个点；使用 `aspp_up` 后提升约 6.54 个点。旧类仍保持在 0.7793，最终 Mean IoU 比原 DeepLabV3 baseline 约高 0.8-0.9 个点。

因此原因一得到强证据支持：

> DeepLabV3+ decoder 融合 low-level texture/boundary 后的特征适合端到端 segmentation head，但不适合当前冻结特征 + RandomBuffer + RecursiveLinear 的 AIR 路径。ASPP 高层语义特征才是正确接口。

### 11.3 对原因二的修正

`decoder_stride8` 仅从 0.3959 提升到 0.4011，说明单纯减少空间点数不能解决问题。更关键的是，`aspp_up` 保持 stride-4 空间尺寸却取得最佳结果，证明：

```text
高分辨率本身不是精度下降主因；
low-level decoder 语义污染才是主因；
高层 ASPP 语义在更细标签对齐尺度上反而更有利。
```

原因二从“高分辨率必然放大背景从而伤害精度”修正为更窄的待验证假设：在最佳 `aspp_up` 上，类别 cap 是否还能改善残余较低的 `pottedplant`，并同时降低 stride-4 解析计算成本。

### 11.4 运行成本证据

| source | 大致总耗时 |
|---|---:|
| `decoder` | 约 58 分钟 |
| `decoder_stride8` | 约 22 分钟 |
| `aspp` | 约 22 分钟 |
| `aspp_up` | 约 55 分钟 |

stride-4 source 的耗时显著更高，符合 RandomBuffer 前 dense pixel 数量约增加 4 倍的预期。即使 class-cap 不继续涨点，只要不损失精度，它也可能是必要的工程优化。

### 11.5 阶段门禁决定

1. 阶段 1 已成功修复精度异常，当前最佳默认候选为 `aspp_up`。
2. 进入阶段 2，但目标收窄为：在 `aspp_up` 上测试 `class_cap=4096/8192`，同时观察精度与耗时。
3. 暂不进入 gamma/norm 扫描。当前特征源切换已经超过目标，且已有证据表明小范围 gamma 扫描价值低。

## 12. 2026-06-23 阶段 2 实验结果与最终结论

### 12.1 实验目的与固定变量

阶段 2 只回答一个问题：在阶段 1 最好的 `aspp_up` 特征上，按类别限制每个 batch 的像素数，能否继续提高稀有新类，同时保住旧类？

两组实验都复用：

```text
step0 checkpoint: 20260614_v3plus_voc15-5_seq_bs32-16
feature source: aspp_up
batch_size: 16
buffer: 8196
gamma: 1
random_seed: 1
setting: VOC 15-5 sequential step1
```

唯一变量是：

```text
air_pixel_balance=class_cap
air_max_pixels_per_class=4096 或 8192
```

运行入口：

```bash
bash run_v3plus_air_pixel_caps.sh
```

### 12.2 结果总表

| 配置 | Mean IoU | 0-15 mIoU | 16-20 mIoU | pottedplant | tvmonitor | train |
|---|---:|---:|---:|---:|---:|---:|
| `aspp_up + none` | **0.7036** | **0.7793** | 0.4613 | 0.2131 | 0.3757 | **0.7534** |
| `aspp_up + cap4096` | 0.6376 | 0.6946 | 0.4552 | 0.2732 | 0.3787 | 0.7118 |
| `aspp_up + cap8192` | 0.6537 | 0.7109 | **0.4707** | **0.3117** | **0.3988** | 0.7157 |

相对无采样 `aspp_up`：

| 配置 | Mean 变化 | 旧类变化 | 新类变化 |
|---|---:|---:|---:|
| `cap4096` | -0.0660 | -0.0847 | -0.0061 |
| `cap8192` | -0.0499 | -0.0684 | +0.0094 |

结论非常明确：`cap8192` 确实让新类再涨约 0.94 点，但代价是旧类下降约 6.84 点，最终 Mean 下降约 4.99 点。它不是总体精度改进。

### 12.3 面向小白：为什么新类涨了，整体反而跌了

可以把解析分类器想成一次性解一道“哪些像素应该影响答案更多”的方程。

- 不采样时，常见像素出现得多，对方程影响也大，这接近分割数据的真实像素分布。
- `class_cap` 会把大类截短，让少数类在方程中的相对音量变大。
- 少数类因此更容易被检出，召回率会上升；但分类器也更容易把别的像素误报成这些类，假阳性增多。
- IoU 同时惩罚漏检和误检，所以“检出了更多”不等于“IoU 一定更高”。

实验证据与这个机制完全一致。`cap8192` 相比无采样时：

| 类别 | IoU 变化 | class accuracy 变化 |
|---|---:|---:|
| pottedplant | +0.0986 | +0.4630 |
| tvmonitor | +0.0230 | +0.4743 |
| sofa | +0.0250 | +0.2424 |
| aeroplane | -0.1818 | +0.0332 |
| bird | -0.1125 | +0.0349 |
| boat | -0.1158 | +0.0871 |

很多类别的 accuracy（这里主要反映召回）上升，但 IoU 下降，说明主要副作用是过度预测和假阳性，而不是“模型没看到这些类”。

### 12.4 为什么旧类损失最大

当前 `class_cap` 同时作用于两个阶段：

1. 用 step0 数据重建 AIR 解析分类器；
2. 用 step1 数据增量更新解析分类器。

在第 1 阶段就对每个 batch 的 0-15 类做硬截断，会改变整个旧类基座的二阶统计量 `X^T X` 和交叉项 `X^T Y`。AIR 没有后续 SGD epoch 去慢慢纠正这个偏移，因此旧类损失会一直保留到最终结果。`cap4096` 截断更强，旧类下降也比 `cap8192` 更大，进一步支持这一解释。

所以阶段 2 不是“采样代码没生效”，恰恰相反，是采样生效后改变了学习目标。这个结果否定的是“所有类别统一硬截断能无代价改善不均衡”这一假设。

### 12.5 显存与速度价值

首个 batch 的日志与 `nvidia-smi` 点观测为：

| 配置 | 有效像素 -> 选中像素 | 观测显存 |
|---|---:|---:|
| `cap4096` | 263,304 -> 43,182 | 约 14.1 GiB |
| `cap8192` | 263,304 -> 71,602 | 约 33.6 GiB |

两组从 2026-06-23 07:12:11 到 07:48:52 顺序完成，总计约 36 分 41 秒。采样显著降低了进入 8196 维 RandomBuffer 后的张量大小，因此工程上有效；但最高精度配置仍应优先保持 `none`。

用户已经在另一台服务器验证 batch size 本身不会造成这次精度异常。因此，在 48 GiB 显存限制下，优先顺序应是：

1. 保持 `aspp_up + none`，先降低 batch size；从 8 开始实测显存。
2. 若仍受限，使用 stride-8 的 `aspp + none`，它的 Mean=0.6995、新类=0.4510，精度损失远小于 class-cap。
3. 只有在更重视稀有新类召回、且接受总体/旧类明显下降时，才使用 `cap8192`。

不要为了维持 batch size 而优先选择 `class_cap`。在已经确认 batch size 不影响精度的前提下，这会用真实精度换取一个没有必要固定的 batch 数字。

### 12.6 最终配置与代码默认值

最终 DeepLabV3+ AIR 对照配置：

```text
air_feature_source=aspp_up
air_pixel_balance=none
air_max_pixels_per_class=0
gamma=1
buffer=8196
```

默认值分两层处理：

- `utils/parser.py` 继续默认 `decoder`，保证旧命令、旧实验和历史 checkpoint 行为兼容。
- 专用 `run_v3plus_air.sh` 默认 `aspp_up`，代表已经验证的 DeepLabV3+ AIR 运行方案。
- `class_cap` 代码保留且默认关闭，用于复现实验或显存/召回研究。

48 GiB 服务器建议命令：

```bash
SUBPATH=20260623_v3plus_air_aspp_up_none_bs8 \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
BATCH_SIZE=8 \
bash run_v3plus_air.sh
```

该命令中的 `bs8` 显存仍需在目标服务器以 `nvidia-smi` 实测；精度判断依据是用户已完成的 batch-size 对照，不把显存估算写成已验证结果。

### 12.7 阶段门禁最终决定

1. 原异常主因已由 feature source 单因素实验确认并修复。
2. class-cap 只作为显存/召回折中，不进入最高精度默认。
3. 不进入 gamma/norm 扫描：主问题已解决，且当前没有特征尺度/数值不稳定证据；已有 RHL 结果也显示 `gamma=0.1/1/10` 近乎等价。
4. 如果要形成论文级结论，下一轮只需做不超过三项工作：
   - 固定 step0，复验 `aspp_up + none` 的 bs8 显存和指标；
   - 对 `decoder` 与 `aspp_up` 做 3 个 RandomBuffer seed，报告均值和标准差；
   - 若研究目标转向稀有类召回，再单独设计“仅 step1 软加权”，不要继续使用 base 和 step1 共用的硬截断。

### 12.8 产物位置

```text
代码提交：9631848 feat: balance AIR fit pixels before expansion
cap4096: checkpoints/20260623_v3plus_air_aspp_up_cap4096
cap8192: checkpoints/20260623_v3plus_air_aspp_up_cap8192
单实验日志：logs/deeplabv3plus_air/20260623_v3plus_air_aspp_up_cap*.log
控制日志：logs/deeplabv3plus_air/20260623_v3plus_air_aspp_up_caps_controller.log
汇总工具：tools/summarize_v3plus_air_results.py
```
