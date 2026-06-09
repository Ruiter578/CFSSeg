问题：
还有一个问题是step0输出的到底是特征还是别的什么，形状是什么？如果除了backbone还有deeplabv3这个decoder，那么step0到底输出的是啥，后续步骤的输入输出的训练逻辑是啥，帮我理顺一下

## 一句话理顺

**step0 正常训练时，模型对外输出的是 segmentation logits（每个像素对每个类别的未归一化分数），不是最终特征；但它内部会产生一个 256 通道的 dense feature map，后续增量步骤会把原来的分类器去掉，把这个 256 维像素特征送入 RHL + C-RLS 闭式解头。**

所以你要把它分成两套逻辑看：

```text
step0：DeepLabV3/ResNet101 正常 BP 训练，输出 logits，用 loss.backward()
step>0：冻结 DeepLab 特征提取部分，输出 dense feature，用 RHL + RecursiveLinear 闭式拟合，不再 BP
```

代码报告也明确把流程分成：第 0 步用 SGD 训练 DeepLab encoder/head，之后增量步骤冻结特征提取网络，用解析闭式解更新像素级分类头。

---

# 1. 先区分三种“输出”

你现在困惑的核心是：**模型里有 backbone，又有 DeepLabV3 head，那输出到底算 feature 还是 prediction？**

在语义分割模型里至少有三层东西：

```text
image
  ↓
1. backbone / encoder 输出：深层视觉特征
  ↓
2. decoder / segmentation head 输出：dense feature
  ↓
3. classifier 输出：logits
```

以当前 SegACIL 的 DeepLabV3 + ResNet101 为例，可以近似理解为：

```text
输入图像 x: [B, 3, H, W]

ResNet101 backbone:
  F = backbone(x)
  F: [B, 2048, Hf, Wf]   # 高层特征，通道多，分辨率低

DeepLabV3 ASPP/head_pre:
  Z = decoder(F)
  Z: [B, 256, Hf, Wf]    # dense feature，后续增量阶段主要用它

Pixel classifier:
  O = classifier(Z)
  O: [B, K, Hf, Wf]      # logits，每个像素每个类别一个分数
```

其中：

- (B)：batch size；
    
- (H,W)：输入图像高宽，比如 crop 后 `513 × 513`；
    
- (H_f,W_f)：特征图高宽，和 `output_stride` 有关；
    
- (K)：当前 step 已见类别数，VOC 里通常包含 background；
    
- `O` 是 logits，不是 softmax/sigmoid 后概率。
    

如果 `output_stride=8` 且输入是 `513 × 513`，通常 (H_f,W_f) 约为 `65 × 65`；如果 `output_stride=16`，通常约为 `33 × 33`。具体数值会受 padding / crop / 实现细节影响，但大致就是输入分辨率除以 output stride。

---

# 2. step0 到底输出什么？

## 2.1 step0 对外输出的是 logits

step0 是普通 segmentation 训练。流程可以写成：

```text
image, label
  ↓
DeepLabV3_ResNet101
  ↓
logits
  ↓
interpolate 到 label 大小
  ↓
BCE / CE / Focal loss
  ↓
loss.backward()
  ↓
optimizer.step()
```

代码报告里 step0 流程写得很直接：

```text
image, mask
  -> transforms
  -> DeepLab backbone
  -> ASPP + head_pre + linear pixel classifier
  -> logits
  -> interpolate to mask size
  -> BCE/CE/Focal loss
  -> SGD backward + scheduler
  -> validation + checkpoint
```

并且 step0 / 常规 DeepLab 评估时使用：

```python
outputs, _ = self.model(images)
outputs = torch.sigmoid(outputs)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
preds = outputs.detach().max(dim=1)[1].cpu().numpy()
```

这说明 step0 的 `outputs` 是按 `[B, C, H, W]` 组织的类别 logits 或概率图，后面会插值到标签大小再取 `argmax`。

所以答案是：

```text
step0 正常训练输出 = logits
不是 feature
不是 mask
不是 one-hot
不是最终类别 ID
```

最终类别 ID 是：

```python
pred = outputs.argmax(dim=1)
```

从 logits 后处理得到的。

---

## 2.2 step0 内部确实有 feature，但不是 loss 直接监督的最终输出

虽然 step0 对外输出 logits，但在 logits 前面有一层 dense feature：

```text
Z: [B, 256, Hf, Wf]
```

这个 `Z` 很关键。因为后续增量阶段会把原来的 pixel classifier 去掉，转而用这个 `Z` 作为解析学习的输入特征。

所以：

```text
step0 训练时：
  Z 只是中间特征
  O = classifier(Z) 才是 loss 用的输出

step>0 增量时：
  原 classifier 被移除或替换
  Z 变成 RHL + C-RLS 的输入
```

这就是你困惑的根源：**同一个 DeepLabV3，在 step0 是完整分割网络；到了 step>0，它被拆成 frozen feature extractor。**

---

# 3. step0 的类别维度是什么？

以 VOC `15-1` 为例：

```text
step0：background + 15 个 foreground 类
```

所以 step0 logits 通道数大概率是：

```text
K0 = 16
```

形状大致是：

```python
images:  [B, 3, 513, 513]
labels:  [B, 513, 513]

logits:  [B, 16, Hf, Wf]      # 可能是 [B, 16, 65, 65]
logits_up:
         [B, 16, 513, 513]
```

然后和 label 做 loss。

如果是 `15-1` 的后续 step：

```text
step1：background + 16 foreground = 17 类
step2：background + 17 foreground = 18 类
...
step5：background + 20 foreground = 21 类
```

所以解析头的输出类别维度会逐步扩展。

---

# 4. 后续步骤的训练逻辑是什么？

## 4.1 step1 不是简单接着训练 DeepLab

step1 的逻辑非常关键。它不是：

```text
加载 step0 checkpoint → 继续 BP 学新类
```

而是：

```text
加载 step0 checkpoint
  ↓
移除 / 替换原来的 classifier
  ↓
冻结 backbone + DeepLab feature extractor
  ↓
先用 step0 数据做 analytic realignment
  ↓
再用 step1 数据做 C-RLS 更新
```

代码报告里也明确写到，`curr_step >= 1` 时不做反向传播，而是闭式更新；流程包括加载 previous checkpoint、移除 original classifier head、用 backbone 提取 256-channel dense feature map、RandomBuffer 映射 `256 -> buffer_size`、再用 `RecursiveLinear.fit()` 累积 (R) 和权重。

---

## 4.2 为什么 step1 要 realign？

因为 step0 的分类器是 SGD 训练出来的，不是闭式解训练出来的。CFSSeg 后续要用 C-RLS（递归最小二乘）更新头部，它要求历史头部权重也处在同一个解析学习范式里。

所以 step1 一开始要做一件事：

```text
把 step0 的 SGD classifier 替换成 analytic classifier
```

这一步可以叫：

```text
analytic realignment / AIR
```

直觉上就是：

```text
原来：
  Z -> SGD classifier -> logits

替换为：
  Z -> RHL -> RecursiveLinear -> logits
```

这样后续 step1、step2、... 才能继续用递归闭式公式更新。

---

# 5. step>0 的输入输出形状

后续增量阶段的核心数据流是：

```text
image
  ↓
frozen DeepLab feature extractor
  ↓
dense feature Z
  ↓
flatten pixels
  ↓
RHL / RandomBuffer
  ↓
RecursiveLinear
  ↓
logits
```

用形状写就是：

## 5.1 输入图像

```python
images: [B, 3, H, W]
```

例如：

```python
[B, 3, 513, 513]
```

---

## 5.2 冻结 feature extractor 输出

```python
Z: [B, 256, Hf, Wf]
```

例如：

```python
[B, 256, 65, 65]    # output_stride=8 时的大致情况
```

这里的 `Z` 是：

```text
backbone + DeepLabV3 ASPP/head_pre 的输出
```

不是最终分类 logits。

---

## 5.3 展平成像素样本

为了做岭回归，每个像素位置都被看成一个训练样本。

通常会变成：

```python
Z_flat: [B, Hf * Wf, 256]
```

例如：

```python
[B, 4225, 256]
```

如果进一步合并 batch：

```python
X_pixels: [B * Hf * Wf, 256]
```

---

## 5.4 RHL 高维映射

RHL 公式是：

[  
E = \operatorname{ReLU}(Z \Phi_E)  
]

其中随机映射矩阵 (\Phi_E) 固定，不通过梯度训练。代码报告里 `RandomBuffer` 使用 `register_buffer("weight", W)` 保存随机权重，并返回 `activation(super().forward(X))`，也就是随机线性映射后接激活。

形状是：

```python
Z_flat: [B, Hf * Wf, 256]
E:      [B, Hf * Wf, buffer_size]
```

如果你的脚本里：

```bash
--buffer 8196
```

那么：

```python
E: [B, Hf * Wf, 8196]
```

---

## 5.5 标签下采样并展平

原始标签是：

```python
labels: [B, H, W]
```

但特征图是：

```python
Hf × Wf
```

所以标签需要被对齐到特征图大小：

```python
labels_down: [B, Hf, Wf]
labels_flat: [B * Hf * Wf]
```

然后过滤掉 ignore label：

```python
mask = y != 255
```

代码报告中 `RecursiveLinear.fit()` 的输入形状就是 `B, HW, C`，随后展平为 `B * HW, C`，过滤 `255`，再 one-hot 成标签矩阵。

---

## 5.6 one-hot 标签矩阵

假设当前 step 已见类别数是 (K_t)，那么：

```python
Y: [N_valid, K_t]
```

其中：

```text
N_valid <= B * Hf * Wf
```

因为 ignore 像素会被过滤掉。

---

## 5.7 RecursiveLinear 更新

解析头有两个关键变量：

```python
R:      [buffer_size, buffer_size]
weight: [buffer_size, K_t]
```

例如：

```python
R:      [8196, 8196]
weight: [8196, 17]    # VOC 15-1 step1
```

更新公式可以简化理解为：

[  
\Phi = (E^\top E + \gamma I)^{-1}E^\top Y  
]

递归实现不是每次存全部历史 (E)，而是保存：

```text
R      ≈ 历史特征自相关矩阵逆
weight ≈ 当前解析分类头权重
```

代码报告中对应关系是：

|论文符号|代码变量|含义|
|---|---|---|
|(E_t)|`X`|经 backbone + RHL 后的像素特征|
|(Y_t)|`Y`|one-hot 标签矩阵|
|(\Psi_t)|`self.R`|正则化特征自相关矩阵的逆|
|(\Phi_t)|`self.weight`|解析分类头权重|
|(\gamma)|`self.gamma`|ridge 正则项|

并且代码核心是：

```python
R_inv = torch.inverse(self.R)
S = R_inv + X.T @ X
S_inv = torch.inverse(S)
self.R = S_inv
self.weight += self.R @ X.T @ (Y - X @ self.weight)
```

---

# 6. step>0 推理时输出什么？

增量阶段的 analytic model 输出和 step0 DeepLab 输出的维度顺序不一样，这一点很容易踩坑。

代码报告里说：

```python
outputs = self.model(images)
outputs = torch.sigmoid(outputs)
outputs = outputs.permute(0,3,1,2)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
preds = outputs.detach().max(dim=1)[1].cpu().numpy()
```

也就是说，`AnalyticLinear.forward()` 输出是：

```python
outputs: [B, Hf, Wf, K_t]
```

而普通 DeepLab 输出是：

```python
outputs: [B, K_t, Hf, Wf]
```

所以 analytic model 评估时要先：

```python
outputs = outputs.permute(0, 3, 1, 2)
```

变成：

```python
[B, K_t, Hf, Wf]
```

再插值到标签大小。

---

# 7. 用一张总流程图总结

## step0：普通 BP 训练

```text
images: [B, 3, H, W]
labels: [B, H, W]

images
  ↓
ResNet101 backbone
  ↓
DeepLabV3 ASPP / head_pre
  ↓
dense feature Z: [B, 256, Hf, Wf]
  ↓
pixel classifier
  ↓
logits: [B, K0, Hf, Wf]
  ↓
interpolate
  ↓
logits_up: [B, K0, H, W]
  ↓
BCE / CE loss
  ↓
loss.backward()
  ↓
更新 backbone + DeepLab head + classifier
```

其中 `K0` 是 step0 的类别数。例如 VOC `15-1` 中：

```text
K0 = background + 15 = 16
```

---

## step1：把 SGD head 转成 analytic head，再学新类

```text
加载 step0 checkpoint
  ↓
移除原 pixel classifier
  ↓
冻结 ResNet101 + DeepLabV3 feature extractor
  ↓
用 step0 数据 realign：
    Z: [B, 256, Hf, Wf]
    Z_flat: [B, Hf*Wf, 256]
    E: [B, Hf*Wf, buffer]
    Y: [N_valid, K0]
    fit RecursiveLinear
  ↓
用 step1 数据递归更新：
    类别维度 K0 -> K1
    weight: [buffer, K0] -> [buffer, K1]
    R: [buffer, buffer] 继续更新
```

---

## step2 及之后：继续 C-RLS 更新

```text
加载上一步 analytic model
  ↓
冻结 feature extractor 和 RandomBuffer
  ↓
当前 step 图像
  ↓
Z = frozen DeepLab feature
  ↓
E = RHL(Z)
  ↓
Y = 当前标签 / pseudo-label
  ↓
RecursiveLinear.fit(E, Y)
  ↓
保存 R 和 weight
```

这里没有：

```text
loss.backward()
optimizer.step()
```

所以它不是普通 fine-tuning。

---

# 8. 你可以这样记

最简单的记忆方式：

```text
step0 输出 logits，训练整个 DeepLab。
step1 以后输出 feature 给 RHL，再由解析头输出 logits。
```

更精确一点：

```text
step0:
  model = backbone + DeepLab decoder + SGD classifier
  output = logits [B, K, Hf, Wf]

step>0:
  model = frozen(backbone + DeepLab decoder_without_classifier)
          + RandomBuffer
          + RecursiveLinear
  intermediate feature = [B, 256, Hf, Wf]
  RHL feature = [B, Hf*Wf, buffer]
  output = analytic logits [B, Hf, Wf, K]
```

---

# 9. 对你改代码最重要的接口

如果你后面要换 backbone / decoder，务必保证最终喂给 RHL 的东西满足：

```python
feat: [B, 256, Hf, Wf]
```

或者至少经过一个 projection 变成：

```python
feat = proj(feat)  # [B, C_in, Hf, Wf] -> [B, 256, Hf, Wf]
```

然后：

```python
feat = feat.flatten(2).transpose(1, 2)
# [B, 256, Hf, Wf] -> [B, Hf*Wf, 256]

feat = random_buffer(feat)
# [B, Hf*Wf, 256] -> [B, Hf*Wf, buffer]

logits = recursive_linear(feat)
# [B, Hf*Wf, K_t] -> reshape -> [B, Hf, Wf, K_t]
```

所以你替换网络时，不要只关心最终 logits。对 CFSSeg 来说，最关键的是：

```text
分类器前一层 dense feature 的形状、通道数、分辨率、数值尺度。
```

这也是为什么我之前建议：如果换更强 encoder/decoder，最好明确设计一个稳定的 `feature interface`，而不是直接把整个分割模型黑盒替换掉。