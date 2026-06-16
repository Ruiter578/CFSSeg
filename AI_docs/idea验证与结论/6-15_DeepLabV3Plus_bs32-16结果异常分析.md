# DeepLabV3+ bs32/16 对照实验结果异常分析

日期：2026-06-15

实验目录：

- DeepLabV3+ 结果：`/root/2TStorage/lyc/SegACIL_deeplabv3plus/checkpoints/20260614_v3plus_voc15-5_seq_bs32-16`
- DeepLabV3+ 日志：`/root/2TStorage/lyc/SegACIL_deeplabv3plus/logs/deeplabv3plus/20260614_v3plus_voc15-5_seq_bs32-16_step0-bs32_step1-bs16.log`
- 对照 DeepLabV3 结果主要参考：
  - `/root/2TStorage/lyc/SegACIL/checkpoints/20260606`
  - `/root/2TStorage/lyc/SegACIL/checkpoints/20260607`
  - `/root/2TStorage/lyc/SegACIL/checkpoints/20260610_rhl_none_g1`

## 1. 结论先行

这次结果“不符合直觉”的地方是真实存在的，但**主要不应归因于 step1 batch size 只有 16**。

更合理的判断是：

1. **DeepLabV3+ 的 step0 表现略高是合理的**：它利用 low-level feature 和 decoder，边界与细节更强，`0-15 mIoU` 比 DeepLabV3 高约 `0.3-0.4` 个点。
2. **step1 新类 `16-20` 低不是因为少看了数据**：ACIL/AIR 的 step1 是解析式 `fit/update`，不是 SGD 训练；batch size 只影响递推分块方式、显存、数值路径，不影响使用的数据总量。
3. **DeepLabV3+ 的 step1 新类下降更像是 AIR 特征接口与 DeepLabV3+ decoder feature 不完全匹配**：当前 V3+ 的 AIR 输入是 stride-4 decoder feature，空间点数约为原 DeepLabV3 stride-8 feature 的 4 倍，新增类尤其是小物体类会受到更强的背景/旧类像素主导和数值条件问题影响。
4. **下降集中在新增类中的小物体/细碎类**：`pottedplant` 和 `tvmonitor` 掉得最明显；`train` 反而略涨。这支持“高分辨率 dense pixel analytic fitting 下，类别/像素分布失衡被放大”的解释。

因此，当前最优先验证的问题不是“能不能跑 bs32 step1”，而是：

> DeepLabV3+ 在 step1 给 AIR 的特征，是否应该直接使用 stride-4 decoder feature？还是应该降采样、归一化、重加权，或改用更接近 DeepLabV3 的 stride-8 ASPP feature？

## 2. 实验事实

### 2.1 DeepLabV3+ bs32/16 配置

日志确认：

```text
step0 batch size = 32
step1 batch size = 16
model = deeplabv3plus_resnet101
task = voc 15-5
setting = sequential
loss_type = bce_loss
buffer = 8196
gamma = 1.0
output_stride = 8
```

step0：

```text
Train set: 8437
batch_size: 32
iters/epoch: 263
```

step1：

```text
Current step train set: 2145
Base realign train set: 8437
batch_size: 16
```

### 2.2 总体指标对照

| 实验 | step | Mean IoU | 0-15 mIoU | 16-20 mIoU | 0-15 mAcc | 16-20 mAcc |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3 `20260606` | 0 | 0.5700 | 0.7481 | - | 0.8914 | - |
| DeepLabV3 `20260606` | 1 | 0.6946 | 0.7801 | 0.4211 | 0.8776 | 0.4818 |
| DeepLabV3 `20260607` | 0 | 0.5693 | 0.7472 | - | 0.8910 | - |
| DeepLabV3 `20260607` | 1 | 0.6956 | 0.7779 | 0.4321 | 0.8743 | 0.5000 |
| DeepLabV3 `20260610_rhl_none_g1` | 0 | 0.5700 | 0.7481 | - | 0.8914 | - |
| DeepLabV3 `20260610_rhl_none_g1` | 1 | 0.6946 | 0.7801 | 0.4211 | 0.8776 | 0.4818 |
| DeepLabV3+ `20260614_v3plus_voc15-5_seq_bs32-16` | 0 | 0.5722 | 0.7511 | - | 0.8939 | - |
| DeepLabV3+ `20260614_v3plus_voc15-5_seq_bs32-16` | 1 | 0.6897 | 0.7815 | 0.3959 | 0.8749 | 0.4833 |

关键差异：

| 对比项 | DeepLabV3+ 相对 DeepLabV3 `20260607` |
|---|---:|
| step0 `0-15 mIoU` | `+0.0039` |
| step1 `0-15 mIoU` | `+0.0035` |
| step1 `16-20 mIoU` | `-0.0362` |
| step1 Mean IoU | `-0.0059` |

也就是说，**DeepLabV3+ 并不是整体崩了**。旧类甚至略高，主要问题集中在新类。

## 3. 类别级差异

以 DeepLabV3 `20260607 step1` 为主要对照：

| class | DeepLabV3 IoU | DeepLabV3+ IoU | delta | DeepLabV3 Acc | DeepLabV3+ Acc | delta Acc |
|---|---:|---:|---:|---:|---:|---:|
| 16 pottedplant | 0.2699 | 0.1742 | -0.0957 | 0.2928 | 0.1893 | -0.1035 |
| 17 sheep | 0.5779 | 0.5558 | -0.0222 | 0.7772 | 0.8141 | +0.0369 |
| 18 sofa | 0.2987 | 0.2927 | -0.0059 | 0.3294 | 0.3548 | +0.0254 |
| 19 train | 0.6989 | 0.7079 | +0.0090 | 0.7684 | 0.7852 | +0.0168 |
| 20 tvmonitor | 0.3151 | 0.2491 | -0.0661 | 0.3324 | 0.2733 | -0.0591 |

新增类平均变化：

```text
16-20 mIoU: -0.0362
0-15 mIoU: +0.0035
```

最主要的负贡献来自：

```text
pottedplant: -9.57 points
tvmonitor:   -6.61 points
sheep:       -2.22 points
```

这很重要：`train` 没掉，`sheep` 的 Acc 还涨了，`sofa` 的 Acc 也涨了。问题不是“新类全都学不会”，而是若干对像素分布、边界、背景混淆很敏感的类被显著压低。

## 4. 为什么不应首先归因于 batch size 16

### 4.1 step1 不是 SGD 训练

代码路径在 `trainer/trainer.py`：

```python
self.model.classifier.head = nn.Identity()
backbone = self.model
self.model = AIR(...)

for seq, (X, y, _) in enumerate(self.train_loader0):
    self.model.fit(X, y)
self.model.update()

for _, (X, y, _) in enumerate(self.train_loader):
    self.model.fit(X, y)
self.model.update()
```

step1 的核心是 `AIR.fit()` 调用 `RecursiveLinear.fit()`，不是反向传播训练。batch size 控制的是每次递推解析更新看到多少图像，而不是像 SGD 那样改变梯度估计噪声。

在精确数学下，递推最小二乘类方法对同一批数据的分块方式应基本等价；现实中会因为浮点误差、矩阵条件数、更新顺序出现差异，但这通常是**数值稳定性问题**，不是“bs16 训练不充分”。

### 4.2 DeepLabV3+ bs16 并不等于有效像素样本更少

当前 DeepLabV3 的 AIR 特征来自 stride-8 语义特征，DeepLabV3+ 当前实现的 AIR 特征来自 decoder 后的 stride-4 feature。

粗略估算，对于 `crop_size=513`：

| 模型 | AIR 特征空间分辨率 | 每图像像素点数 | batch | 每次 fit 约像素点 |
|---|---:|---:|---:|---:|
| DeepLabV3 | stride-8，约 `65x65` | 约 `4.2k` | 32 | 约 `135k` |
| DeepLabV3+ | stride-4，约 `129x129` | 约 `16.6k` | 16 | 约 `266k` |

也就是说，DeepLabV3+ step1 虽然 batch 是 16，但每次解析更新的 dense pixel 样本量反而可能比 DeepLabV3 bs32 更大。

所以“DeepLabV3+ step1 低是因为 batch size 16 小于原配置 32”不是最有力解释。

### 4.3 batch size 仍可能有二阶影响

不能完全排除 batch size：

1. `RecursiveLinear.fit()` 每个 batch 都会做一次矩阵更新。
2. bs16 比 bs32 更新次数更多。
3. 每次更新都涉及 `X.T @ X` 和矩阵逆。
4. 在 `buffer=8196`、dense pixel 非常多、特征相关性强时，递推数值路径可能发生可观差异。

但这属于数值路径/条件数影响。它应通过“同模型换 batch size”验证，而不是直接拿 DeepLabV3 bs32 与 DeepLabV3+ bs16 比较下结论。

## 5. 更可能的机制原因

### 5.1 DeepLabV3+ 的 AIR 输入特征不再是原论文/原实现的特征分布

DeepLabV3 原路径：

```python
back_out -> ASPP -> head_pre -> 256-d feature -> Linear head
```

DeepLabV3+ 当前路径：

```python
back_out -> ASPP
low_level -> project
concat -> decoder -> 256-d decoder_feature -> Linear head
```

step1 时：

```python
self.model.classifier.head = nn.Identity()
```

于是 DeepLabV3+ 传给 AIR 的是：

```text
decoder_feature: B x 256 x H/4 x W/4
```

这和 DeepLabV3 的 stride-8 ASPP/head_pre feature 不同。DeepLabV3+ 的 decoder feature 混入了低层纹理和边界信息，对 step0 监督训练可能有利，但对 AIR 的随机映射 + 闭式分类头不一定更友好。

### 5.2 高分辨率 dense feature 会放大背景/旧类主导

AIR 当前对所有有效像素做解析拟合：

```python
X = X.view(B * HW, C)
y = y.view(-1)
mask = y != 255
X = X[mask]
y = y[mask]
Y = one_hot(y)
```

没有类别均衡采样，也没有前景/背景重加权。

DeepLabV3+ stride-4 feature 的像素点数量约为 DeepLabV3 stride-8 的 4 倍。对于 VOC 这类数据，背景像素和旧类像素本来就占大多数；空间分辨率提高后，解析拟合里被“计数”的背景/旧类像素更多。

这会导致：

1. base realign 阶段旧类拟合很强；
2. step1 新类更新时，新类小物体像素相对更弱；
3. pottedplant、tvmonitor 这类小目标/边界复杂类更容易被背景或旧类吞掉；
4. 旧类 `0-15 mIoU` 可以保持甚至略涨，但新类下降。

这与当前指标高度吻合。

### 5.3 `gamma=1, buffer=8196` 可能是为 DeepLabV3 特征调过的，不一定适合 V3+

当前随机特征映射：

```python
RandomBuffer(256 -> 8196) + ReLU
RecursiveLinear(buffer=8196, gamma=1)
```

`gamma` 在 `RecursiveLinear` 中相当于正则项强度：

```python
R = I / gamma
S = gamma * I + X.T @ X
```

DeepLabV3+ decoder feature 的分布、空间相关性、低层纹理成分都不同。继续沿用 `gamma=1` 和 `buffer=8196`，可能会造成：

- 对旧类/背景高频像素拟合过强；
- 小新类的分类边界被压缩；
- `X.T @ X` 条件数更差；
- 递推更新对 batch/order 更敏感。

### 5.4 新类下降并不均匀，支持“特征/类别分布”解释

如果只是 batch size 变小，通常会预期五个新类一起变差，或者整体 Acc 明显下降。但实际：

```text
pottedplant: IoU -0.0957, Acc -0.1035
tvmonitor:   IoU -0.0661, Acc -0.0591
train:       IoU +0.0090, Acc +0.0168
sofa:        IoU -0.0059, Acc +0.0254
```

大目标 `train` 没有变差，小目标/背景混淆类掉得更明显。这更像特征分辨率、像素采样分布和类别不均衡问题。

### 5.5 DeepLabV3+ decoder 对 SGD 有利，不代表对闭式 AIR 有利

DeepLabV3+ 的优势来自 decoder 把低层空间细节与高层语义融合。这个设计服务于端到端训练的 segmentation head。

但 step1 不再训练 decoder，只是冻结它，然后在其输出上做随机映射和解析线性分类。低层细节可能带来更多纹理/边缘变化，反而降低新类线性可分性。也就是说：

> 对 step0 监督学习有利的 feature，不一定对 step1 的冻结特征 + 随机映射 + 闭式分类最优。

## 6. 应优先做的验证实验

### 实验 A：DeepLabV3 原配置 step1 改成 bs16

目的：直接验证 batch size 是否是主因。

做法：

- 使用原 DeepLabV3 step0 checkpoint。
- 只重跑 step1。
- 将 step1 batch size 从 32 改为 16。
- 其他参数保持一致。

预期解释：

| 结果 | 说明 |
|---|---|
| `16-20 mIoU` 仍在 `0.421-0.432` 附近 | batch size 不是主因 |
| `16-20 mIoU` 明显掉到 `0.396` 附近 | batch size/递推分块是主要因素 |

这个实验成本最低，应该第一个做。

### 实验 B：DeepLabV3+ step1 bs8 / bs4 敏感性

目的：看 V3+ 的解析路径是否对 batch size 极敏感。

做法：

- 复用同一个 DeepLabV3+ step0 checkpoint。
- 分别用 step1 bs16、bs8、bs4 重跑。

预期解释：

| 现象 | 说明 |
|---|---|
| bs16、bs8、bs4 差异很小 | batch size 不是主因，特征/AIR 适配更可疑 |
| batch 越小新类越低 | V3+ 的递推数值路径对分块敏感，需要稳定化 |

### 实验 C：DeepLabV3+ AIR 特征降采样到 stride-8

目的：验证 stride-4 decoder feature 是否引入过多 dense pixel / 背景主导。

做法：

- step0 仍使用 DeepLabV3+ 正常训练。
- step1 时，在 AIR 的 `feature_expansion()` 里对 `X` 做 `avg_pool2d` 或插值降采样，使其接近 DeepLabV3 的 stride-8 分辨率。
- 其他参数不变。

预期：

- 如果 `16-20 mIoU` 明显回升，说明当前 V3+ step1 低主要来自高分辨率 dense fitting。
- 如果不回升，再看特征分布/正则化。

### 实验 D：DeepLabV3+ AIR 使用 ASPP stride-8 feature

目的：区分“DeepLabV3+ decoder feature 问题”和“DeepLabV3+ backbone 本身问题”。

做法：

- step0 正常用 DeepLabV3+ decoder head。
- step1 AIR 不使用 `decoder_feature`，改用更接近 DeepLabV3 的 ASPP output / stride-8 256-d feature。

预期：

- 如果新类回升，说明 decoder/low-level feature 不适合当前 AIR。
- 如果仍低，说明 V3+ step0 学到的整体表示或训练过程与 AIR 不匹配。

### 实验 E：gamma 扫描

目的：验证 V3+ 特征是否需要更强正则。

建议先做：

```text
gamma = 0.1, 1, 10, 100
```

重点看：

- `16-20 mIoU`
- `pottedplant`
- `tvmonitor`
- `0-15 mIoU` 是否明显掉

如果增大 gamma 后新类上涨，说明原 `gamma=1` 对 V3+ 的高相关 dense feature 正则不足。

### 实验 F：像素级类别均衡/前景采样

目的：解决高分辨率 feature 下背景/旧类像素主导。

可以先做最小版本：

- 在 `RecursiveLinear.fit()` 前对像素做采样；
- 限制背景像素数量；
- 或按类别最多采样固定数量像素；
- 保持每张图都有前景类参与。

这个实验对机制改动更大，建议排在降采样和 gamma 扫描之后。

## 7. 当前判断

这次结果的合理解释排序如下：

| 优先级 | 可能原因 | 可信度 | 说明 |
|---:|---|---:|---|
| 1 | DeepLabV3+ stride-4 decoder feature 与 AIR/RecursiveLinear 不匹配 | 高 | step0/旧类涨，新类尤其小物体掉，符合该机制 |
| 2 | 高分辨率 dense pixel 放大背景/旧类像素主导 | 高 | pottedplant、tvmonitor 大幅下降，train 不降 |
| 3 | gamma/buffer 对 V3+ 特征不再合适 | 中高 | V3+ 特征分布与 V3 不同，解析解正则可能需重调 |
| 4 | step1 bs16 导致数值路径变化 | 中 | 可能有影响，但不是“少训练”的意义，需要单独验证 |
| 5 | 代码接入错误 | 中低 | 形状契约成立，step0/旧类正常；但仍建议做 feature shape/norm 诊断 |
| 6 | 单纯随机波动 | 低 | 新类平均掉 2.5-3.6 点，且集中在特定类，不像普通波动 |

## 8. 下一步建议

推荐按下面顺序推进：

1. **先跑 DeepLabV3 baseline step1 bs16**：这是验证 batch size 假设的最小实验。
2. **再跑 DeepLabV3+ step1 bs8/bs4 敏感性**：判断 V3+ 解析路径是否对分块敏感。
3. **做 DeepLabV3+ AIR feature 降采样到 stride-8**：直接验证高分辨率 dense fitting 是否是主因。
4. **做 gamma 扫描**：`0.1/1/10/100`，先不动网络结构。
5. 如果以上支持特征不匹配，再考虑：
   - AIR 使用 ASPP stride-8 feature；
   - 像素级类别均衡；
   - RHL/feature normalization 专门针对 V3+ 重跑。

当前不建议为了“公平对齐 batch size”强行追求 DeepLabV3+ step1 bs32。原因是：

- 显存大概率超过 80GB；
- 即使跑成，bs32 也只能回答分块问题；
- 更关键的机制疑点是 V3+ 给 AIR 的特征分辨率和分布已经变了。

更高性价比的验证路径是：**先证明 DeepLabV3 step1 bs16 是否掉点，再证明 DeepLabV3+ stride-4 feature 是否是问题源头。**

