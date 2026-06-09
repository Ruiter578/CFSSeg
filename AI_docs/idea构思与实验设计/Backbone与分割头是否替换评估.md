# Backbone 与分割头是否替换评估

> 日期：2026-06-09  
> 结论类型：当前 30 天论文冲刺阶段的工程与论文策略结论。后续如果主线实验稳定，可以再作为扩展实验重新评估。

## 1. 先给结论

当前阶段的确定性结论是：

1. **主实验不要替换 DeepLabV3 + ResNet101。** 它应继续作为 CFSSeg 复现、RHL 归一化、自适应伪标签、轻量集成的公平底座。
2. **DeepLabV3 可以升级到 DeepLabV3+，但只能作为后续“强 decoder / 架构鲁棒性”实验，不应作为当前论文核心贡献。**
3. **ResNet101 不建议在当前 30 天主线中替换。** 换 backbone 会重训 step0、导致 checkpoint 不兼容，并把审稿焦点从解析持续学习模块转移到强表征模型。
4. **当前仓库不能只改 `MODEL=deeplabv3plus_resnet101` 就跑通。** 代码里虽然有这个 model name，但 ResNet 分支没有真正接通 V3+ head，且 V3+ head 返回值与训练代码接口不一致。

一句话决策：

```text
短期论文主线：固定 DeepLabV3 + ResNet101，先做 RHL normalization、adaptive pseudo-label、lightweight analytic ensemble。
后续补充实验：在主线稳定后，再补一个 DeepLabV3+ + ResNet101 作为 architecture robustness，不把它写成核心贡献。
```

## 2. 术语纠正

用户问题里提到“backbone（deeplab v3）和分割头 resnet101”。更准确地说：

| 术语 | 当前项目对应 |
|---|---|
| backbone / encoder | ResNet101 |
| segmentation model / decoder | DeepLabV3，主要包含 ASPP 和 head_pre/classifier |
| classifier head | step0 的 `classifier.head`，step>0 被替换为 RHL + C-RLS |
| analytic adapter/head | RHL `RandomBuffer` + `RecursiveLinear` |

所以 ResNet101 不是分割头，而是 backbone；DeepLabV3 才是当前分割网络框架和 decoder/head 结构。

## 3. 为什么主线不换

### 3.1 论文公平性

CFSSeg 论文 2D 实验明确使用 DeepLabV3 + ImageNet-1K 预训练 ResNet101，训练 50 epoch，BCE loss，`d_E=8192`、`gamma=1`。如果本课题直接换成更强 backbone 或 decoder，提升很容易被解释为“强网络带来的收益”，而不是解析持续学习模块的贡献。

对于一篇 follow-up 论文，更稳的比较方式是：

```text
同一 DeepLabV3 + ResNet101 底座
  baseline CFSSeg
  + RHL normalization
  + adaptive pseudo-label
  + lightweight analytic ensemble
```

这样每个增益都能归因到本课题的新模块。

### 3.2 结项指标已经不是瓶颈

当前已有 `15-5 sequential` 结果：

| 实验 | old mIoU | new mIoU | all mIoU |
|---|---:|---:|---:|
| `20260606` | 78.01 | 42.11 | 69.46 |
| `20260607` | 77.79 | 43.21 | 69.56 |

结项图里的目标是单模型 65.9%、集成系统 67.0%。因此当前问题不是“分割底座不够强导致达不到指标”，而是“需要形成可写论文的方法创新、消融和复盘”。

### 3.3 换 backbone 会重置实验链

换 ResNet101 意味着：

1. step0 必须重训，已有 step0 checkpoint 不能复用。
2. step1 realign 后的 AIR checkpoint 不能复用。
3. RHL 输入特征分布变化，`gamma`、`buffer`、归一化策略都要重新调。
4. 旧实验结果与新结果可比性下降。
5. 30 天冲刺阶段会被大量工程迁移、显存和数据复现实验占用。

这不适合作为当前第一步。

## 4. 当前代码事实

### 4.1 model name 有，但 ResNet V3+ 未接通

`network/modeling.py` 的 `model_map` 中包含：

```python
'deeplabv3plus_resnet50': self.deeplabv3plus_resnet50,
'deeplabv3plus_resnet101': self.deeplabv3plus_resnet101,
```

但 `_segm_resnet()` 只支持：

```python
if name == 'deeplabv3':
    classifier = DeepLabHead(...)
elif name == 'deeplabv3_bga':
    classifier = DeepLabHeadBgA(...)
else:
    raise ValueError(...)
```

因此 `deeplabv3plus_resnet101` 会进入 `else` 并报错。

### 4.2 V3+ head 返回值不符合当前训练接口

`_SimpleSegmentationModel.forward()` 写死了：

```python
x, feat = self.classifier(features)
return x, feat
```

当前 `DeepLabHead.forward()` 返回：

```python
return heads, {"feature": feature, "back_out": back_out}
```

但 `DeepLabHeadV3Plus.forward()` 当前只返回：

```python
return heads
```

所以即使补上 ResNet 低层特征返回，V3+ head 仍会在 step0 训练时因为解包失败而报错。它还需要返回 `feat_dict`，并保证 step1 去掉 classifier head 后仍能提供稳定的 256-d dense feature 给 RHL。

### 4.3 AIR 默认假设 RHL 输入是 256 维

`trainer/trainer.py` 中 step1 构造 AIR 时写死：

```python
backbone_output=256
```

这与当前 DeepLabV3 的 `head_pre` 输出 256-d feature 对齐。如果换成 UPerNet、HRNet-OCR、ConvNeXt decoder 或其他网络，必须明确“哪一层 256-d feature 送入 RHL”。直接把 1024/2048 维 backbone feature 送进 RHL，会增加显存和矩阵计算压力，也会改变方法含义。

## 5. 可选方案评估

| 方案 | 当前建议 | 原因 |
|---|---|---|
| 保持 DeepLabV3 + ResNet101 | 主线采用 | 最公平、最省时间、与 CFSSeg 和已有 checkpoint 对齐 |
| DeepLabV3+ + ResNet101 | 后续补充实验 | 改动中等，能验证 decoder 边界和低层特征是否帮助 RHL，但当前代码需修 |
| ResNet50 / MobileNet | 不建议 | 更轻但大概率不是性能提升，论文价值弱 |
| ConvNeXt + UPerNet | 中长期候选 | 强 CNN 表征，但迁移成本高，容易偏离解析学习主线 |
| HRNet-W48 + OCR | 中长期候选 | 高分辨率分割强，但显存和接口成本高 |
| SegNeXt / MSCAN | 中长期候选 | 非 ViT、分割专用，但需要重新定义 RHL feature interface |
| InternImage / 大模型 backbone | 当前不建议 | 性能上限高但工程复杂，审稿解释风险大 |

## 6. 如果后续要做 DeepLabV3+，执行边界如下

DeepLabV3+ 可以做，但应作为独立分支实验，不要混入 RHL 归一化第一阶段。最小实现要求：

1. 在 ResNet 分支返回低层特征：

```python
return_layers = {"layer4": "out", "layer1": "low_level"}
```

2. 对 ResNet101 使用：

```python
DeepLabHeadV3Plus(
    in_channels=2048,
    low_level_channels=256,
    num_classes=opts.num_classes,
    aspp_dilate=aspp_dilate,
)
```

3. 修改 `DeepLabHeadV3Plus.forward()`，让它返回：

```python
return heads, {
    "feature": output_feature_before_final_classifier,
    "back_out": feature["out"],
    "low_level": low_level_feature,
}
```

4. 确认 step1 中：

```python
self.model.classifier.head = nn.Identity()
```

对 V3+ 仍然合理。如果 V3+ 的分类层结构不同，必须改成只移除最终分类层，保留 256-d decoder feature。

5. 单独跑：

```text
Baseline-A: DeepLabV3 + ResNet101
Baseline-B: DeepLabV3+ + ResNet101
```

只比较架构影响，不与 RHL normalization 同时改变。

## 7. 推荐实验顺序

### 当前阶段

```text
Exp0: CFSSeg baseline, DeepLabV3 + ResNet101, 15-5 sequential
Exp1: + RHL normalization
Exp2: + gamma sweep for normalized RHL
Exp3: + adaptive pseudo-label on 15-5 disjoint/overlap
Exp4: lightweight analytic ensemble
```

### 主线稳定后

```text
Exp5: DeepLabV3+ + ResNet101 baseline
Exp6: DeepLabV3+ + ResNet101 + best RHL normalization
```

Exp5/Exp6 的论文定位应是：

```text
architecture robustness / stronger decoder compatibility
```

不要写成核心方法贡献。

## 8. 最终建议

当前 30 天冲刺阶段，backbone 和分割框架应当被视作“实验底座”，而不是“创新主体”。真正值得写进论文贡献的，是：

1. 稳定化 RHL 特征接口。
2. 自适应伪标签阈值。
3. 共享 encoder 的轻量解析集成。

DeepLabV3+ 或更强 CNN 可以做，但只能在主线结果稳定后作为补充证据。否则工程成本和审稿风险都不划算。

