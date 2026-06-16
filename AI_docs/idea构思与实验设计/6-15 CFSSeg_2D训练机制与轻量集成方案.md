# CFSSeg / SegACIL 2D 训练机制与轻量集成系统方案

> 面向当前课题：在 **PASCAL VOC 2012，15-5，0-20 共 21 类 mIoU** 评价协议下，以原论文 **CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds** 和原代码库 **SegACIL** 为基础，解释 2D 场景训练逻辑，并设计“持续学习为主、集成系统为辅”的简单可落地方案。  
> 关键词：**CISS（Class-Incremental Semantic Segmentation，类别增量语义分割）**、**RHL（Randomly-initialized Hidden Layer，随机初始化隐藏层）**、**C-RLS（Concatenated Recursive Least Squares，拼接式递归最小二乘）**、**Snapshot Ensemble（快照集成）**、**Multi-seed Ensemble（多随机种子集成）**。

---

## 0. 一句话总览

CFSSeg 在 2D 语义分割中的训练机制可以概括为：

```text
step0：用 BP/SGD 正常训练 DeepLabV3 + ResNet101，得到一个基础分割模型
  ↓
step1 开始：去掉原来的梯度训练分类器，把 DeepLab 的 dense feature 送入 RHL
  ↓
RHL：固定随机高维映射，把 256 维像素特征扩展到更高维
  ↓
C-RLS：用闭式递归最小二乘更新分类头，不再通过 loss.backward() 训练增量头
  ↓
可选伪标签：在 disjoint / overlapped 设置下修复旧类被标成 background 的问题
```

你的项目中，“集成系统”不应喧宾夺主。最稳妥的做法是：

```text
主线：CFSSeg 的类别增量语义分割
辅线：在推理阶段加入轻量 ensemble
推荐：RHL 多随机种子集成 > step0 快照集成 > TTA 集成
```

其中最推荐的方案是 **方案 A：RHL 多随机种子集成**。它与 CFSSeg 的 RHL 机制天然结合，工程改动小，成本低，容易解释为“随机高维子空间集成”。

---

## 1. 先把任务背景讲清楚

### 1.1 普通语义分割在做什么？

**Semantic Segmentation（语义分割）** 是对图像中每一个像素预测类别。

例如输入图像：

```text
[B, 3, H, W]
```

模型输出 logits：

```text
[B, C, H, W]
```

其中：

| 符号 | 含义 |
|---|---|
| `B` | batch size，批大小 |
| `3` | RGB 三通道 |
| `H, W` | 图像高宽 |
| `C` | 类别数，VOC 2012 中通常是 21，包括 background |
| `logits` | 未经过 sigmoid / softmax 的类别分数 |

最终对每个像素取类别分数最大的类别：

```python
pred = logits.argmax(dim=1)  # [B, H, W]
```

### 1.2 类别增量语义分割在做什么？

**Class-Incremental Semantic Segmentation，CISS，类别增量语义分割** 不是一次性学习全部类别，而是分阶段学习。

以 VOC 2012 的 `15-5` 为例：

```text
step0：学习 0-15
       其中 0 是 background，1-15 是前 15 个前景类

step1：学习 16-20
       一次性新增 5 个前景类

最终评估：0-20 共 21 类 mIoU
```

你的导师确认的评价协议是：

```text
0-20 共 21 类都参与 mIoU 计算
15-5 使用 0-15 作为初始类，16-20 作为增量类
```

这和 SegACIL 代码中 `utils/tasks.py` 的 `15-5` 配置一致：

```python
"15-5": {
    0: [0, 1, 2, ..., 15],
    1: [16, 17, 18, 19, 20]
}
```

---

## 2. 原论文 2D 场景的整体训练机制

CFSSeg 的 2D 实验使用 **PASCAL VOC2012**，模型主体是 **DeepLabV3 + ResNet-101**。论文中说明，2D 实验使用的训练集规模是 10,582 张，验证集是 1,449 张，也就是常见的 VOC trainaug 协议。

### 2.1 它不是“整个网络都闭式训练”

这是初学者最容易误解的点。

CFSSeg 的 **Closed-Form Solution（闭式解）** 主要作用在 **增量阶段的分类头更新**，不是把整个 DeepLabV3 + ResNet101 都用闭式解训练。

更准确地说：

```text
step0：仍然使用 BP/SGD 训练 DeepLabV3 + ResNet101
step>0：冻结 DeepLab 特征提取部分，只用 RHL + C-RLS 更新解析分类头
```

也就是说，CFSSeg 是一个混合式框架：

| 阶段 | 训练方式 | 是否 BP/SGD | 是否闭式解 |
|---|---|---:|---:|
| step0 基础训练 | 普通 DeepLab 训练 | 是 | 否 |
| step1 analytic realignment | 用 step0 数据重建解析头 | 否 | 是 |
| step1 增量学习 | 学习 16-20 | 否 | 是 |
| step2 及以后 | 继续学习新类 | 否 | 是 |

对于你的 `15-5`，只有：

```text
step0 + step1
```

没有 step2 之后的多轮增量。

---

## 3. step0：普通 DeepLabV3 训练阶段

### 3.1 step0 的输入输出

step0 是标准监督语义分割训练。

数据流可以写成：

```text
image: [B, 3, H, W]
  ↓
ResNet101 backbone
  ↓
DeepLabV3 ASPP / classifier 前半部分
  ↓
dense feature: [B, 256, Hf, Wf]
  ↓
pixel classifier
  ↓
logits: [B, C0, Hf, Wf]
  ↓
插值到标签大小
  ↓
loss + backward + optimizer.step()
```

其中：

```text
C0 = 16  # 对 15-5 的 step0 来说，类别是 0-15
```

如果 `output_stride=16` 且 `crop_size=513`，则中间特征图大小通常接近：

```text
Hf, Wf ≈ 33, 33
```

如果 `output_stride=8`，则大约是：

```text
Hf, Wf ≈ 65, 65
```

具体大小以代码实际打印为准。

### 3.2 step0 在代码里如何体现？

SegACIL / CFSSeg 的 `train.py` 会根据当前 `curr_step` 计算类别列表和类别数量：

```python
opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step))
                    for step in range(opts.curr_step + 1)]
opts.target_cls = [get_tasks(opts.dataset, opts.task, step)
                   for step in range(opts.curr_step + 1)]
opts.num_classes = [1, opts.num_classes[0] - 1] + opts.num_classes[1:]
```

这段逻辑的含义是：

```text
先把 background 单独拆成 1 类
再把初始前景类作为另一组
后续每个增量 step 再追加一组新类
```

对 `15-5, curr_step=0`：

```text
target_cls = [[0, 1, ..., 15]]
num_classes = [1, 15]
```

对 `15-5, curr_step=1`：

```text
target_cls = [[0, 1, ..., 15], [16, 17, 18, 19, 20]]
num_classes = [1, 15, 5]
```

step0 训练代码核心是：

```python
outputs, _ = self.model(images)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear', align_corners=False)
loss = self.criterion(outputs, labels)
loss.backward()
self.optimizer.step()
self.scheduler.step()
```

这说明：

```text
step0 输出是 logits
step0 使用 loss.backward()
step0 是普通深度学习训练
```

### 3.3 step0 为什么重要？

因为 step0 得到的 DeepLab 特征提取器后续会被冻结。也就是说，后续所有增量学习都依赖 step0 学到的特征。

如果 step0 特征质量差，后面的 RHL 和 C-RLS 再优雅也很难补救。

所以 step0 决定了：

```text
旧类基础能力
dense feature 的可分性
后续 RHL 输入质量
伪标签置信度质量
```

---

## 4. step1：解析重对齐与增量闭式更新

对于 `15-5`，step1 是唯一的增量阶段。它包括两件事：

```text
A. Analytic realignment：把 step0 的 SGD 分类头替换成解析分类头
B. Incremental update：用 16-20 的数据更新解析分类头
```

### 4.1 为什么需要 analytic realignment？

因为 step0 训练出来的 pixel classifier 是通过 SGD 学到的，不是闭式解得到的。

而后续 C-RLS 的递归更新要求：

```text
历史分类头也必须处在同一个解析学习框架里
```

因此 step1 一开始要重新用 step0 数据拟合一个解析头。

这一步可以理解成：

```text
原来：dense feature -> SGD classifier -> logits
现在：dense feature -> RHL -> RecursiveLinear -> logits
```

代码中体现为：

```python
self.model.classifier.head = nn.Identity()
backbone = self.model
self.model = AIR(
    backbone_output=256,
    backbone=backbone,
    buffer_size=self.opts.buffer,
    gamma=self.opts.gamma,
    linear=RecursiveLinear,
)
```

其中 `classifier.head = nn.Identity()` 的作用是：

```text
去掉原来的最后分类层
保留 DeepLab 的特征提取部分
让模型输出 256 通道 dense feature，而不是最终类别 logits
```

### 4.2 AIR 模块是什么？

代码中的 `AIR` 可以理解为 CFSSeg 2D 增量阶段的核心封装：

```text
AIR = frozen DeepLab feature extractor + RandomBuffer/RHL + RecursiveLinear
```

它有三个关键方法：

```python
feature_expansion(X)
forward(X)
fit(X, y)
```

数据流为：

```text
image X
  ↓
self.backbone(X)
  ↓
feature map: [B, 256, Hf, Wf]
  ↓
view + permute
  ↓
pixel feature: [B, Hf*Wf, 256]
  ↓
RandomBuffer / RHL
  ↓
high-dimensional feature: [B, Hf*Wf, buffer_size]
  ↓
RecursiveLinear
  ↓
logits: [B, Hf, Wf, C]
```

注意代码中的解析头输出维度顺序是：

```text
[B, Hf, Wf, C]
```

而普通 DeepLab 输出通常是：

```text
[B, C, Hf, Wf]
```

所以评估时需要：

```python
outputs = outputs.permute(0, 3, 1, 2)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
```

---

## 5. RHL：CFSSeg 中最核心、最值得讲清楚的组件

### 5.1 RHL 是什么？

**RHL，Randomly-initialized Hidden Layer，随机初始化隐藏层**，在代码里对应 `RandomBuffer`。

它做的事是：

```text
把 DeepLab 输出的 256 维像素特征，随机映射到更高维空间
```

数学形式是：

\[
E = \operatorname{ReLU}(F \Phi_E)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(F\) | DeepLab 输出的像素特征，形状约为 `[B*Hf*Wf, 256]` |
| \(\Phi_E\) | 随机初始化并固定的映射矩阵 |
| \(E\) | RHL 输出的高维特征 |
| `ReLU` | 非线性激活函数 |

代码中的 `RandomBuffer`：

```python
W = torch.empty((out_features, in_features))
self.register_buffer("weight", W)
self.reset_parameters()

@torch.no_grad()
def forward(self, X):
    X = X.to(self.weight)
    return self.activation(super().forward(X))
```

重点是：

```text
weight 是 buffer，不是 parameter
不会被 optimizer 更新
不会参与 loss.backward()
```

也就是说，RHL 是一个 **固定随机特征映射**。

### 5.2 RHL 为什么有用？直觉解释

后续的 C-RLS 分类头本质上是一个线性模型。

线性模型直接作用在 256 维 DeepLab feature 上，可能表达能力不够。RHL 的作用是把特征投影到一个更高维的非线性空间，让原本不容易线性分开的像素类别变得更容易被线性分类器分开。

直觉类比：

```text
原始二维平面上，红点和蓝点混在一起，一条直线分不开。
把点通过非线性映射抬到三维空间后，可能一个平面就能分开。
```

在 CFSSeg 里，这个思想对应：

```text
冻结 backbone 提供稳定性
RHL 高维映射提供可塑性
C-RLS 线性闭式头提供快速增量更新
```

### 5.3 RHL 为什么能补偿冻结 backbone 的缺点？

冻结 backbone 的好处是：

```text
旧知识不被梯度更新破坏
```

坏处是：

```text
模型对新类的适应能力下降
```

RHL 的补偿方式是：

```text
不改 backbone，但改变后续分类头看到的特征空间
```

也就是：

```text
不是让 ResNet101 继续学习新类
而是把 ResNet101 的固定特征投影到更高维，让解析头更容易拟合新类
```

这就是 CFSSeg 中 **stability-plasticity dilemma（稳定性-可塑性两难）** 的解决方式：

| 问题 | 解决模块 |
|---|---|
| 旧知识容易被梯度破坏 | 冻结 encoder |
| 冻结后新类学习能力不足 | RHL 高维映射 |
| 增量训练慢且易遗忘 | C-RLS 闭式递归更新 |
| 旧类被标成 background | 伪标签修复 |

### 5.4 RHL 的形状流转

假设：

```text
B = 16
Hf = Wf = 33
DeepLab dense feature channel = 256
buffer_size = 8192
```

那么：

```text
DeepLab feature:
[B, 256, 33, 33]

展平空间维度：
[B, 256, 1089]

转置为像素样本：
[B, 1089, 256]

RHL 输出：
[B, 1089, 8192]
```

从机器学习视角看：

```text
每个像素位置 = 一个训练样本
每个样本有 8192 维特征
分类目标是该像素的 one-hot 类别标签
```

### 5.5 RHL 与你当前 CFSSeg 新库的 RHL 归一化

你在 CFSSeg 新库中已经加入了 RHL 输出归一化选项：

```python
rhl_norm: none / l2 / l2_sqrt / layernorm
```

设计逻辑是：

```text
先做固定随机映射 + ReLU
再对每个像素的高维特征向量做可选归一化
最后送入 RecursiveLinear
```

这不改变 C-RLS 的核心公式，因为它仍然是：

```text
固定特征映射 E + 闭式线性头
```

只是把原来的：

\[
E = \operatorname{ReLU}(F\Phi_E)
\]

换成：

\[
\tilde{E} = \operatorname{Normalize}(\operatorname{ReLU}(F\Phi_E))
\]

只要训练和测试、step0 realignment 和 step1 增量阶段都使用同一个映射，闭式递归学习的逻辑仍然成立。

---

## 6. C-RLS：解析分类头如何更新？

### 6.1 普通岭回归

CFSSeg 把像素级分类头学习写成 ridge regression，岭回归：

\[
\hat{\Phi} = (E^\top E + \gamma I)^{-1}E^\top Y
\]

其中：

| 符号 | 含义 |
|---|---|
| \(E\) | RHL 输出的像素特征矩阵 |
| \(Y\) | one-hot 标签矩阵 |
| \(\Phi\) | 解析分类头权重 |
| \(\gamma\) | ridge 正则系数 |
| \(I\) | 单位矩阵 |

这相当于用矩阵公式一次性求出分类头，不需要多轮 `loss.backward()`。

### 6.2 为什么要递归？

如果每次都把历史数据和当前数据拼起来求闭式解，那就需要保存所有旧数据：

```text
E_all = [E_step0; E_step1; ...]
Y_all = [Y_step0; Y_step1; ...]
```

但持续学习通常要求：

```text
不能访问历史训练数据
不能保存旧样本
```

所以需要递归形式。

递归更新的核心思想是：

```text
不保存旧样本
只保存一个历史相关矩阵 R / Ψ 和旧分类头权重
新数据到来时，用新数据修正这两个量
```

代码中的 `RecursiveLinear` 保存了：

```python
self.R      # regularized feature autocorrelation matrix
self.weight # analytic classifier weight
```

每次 `fit(X, y)` 时：

```python
X = X.view(B * HW, C)
y = y.view(-1)
mask = y != 255
X = X[mask]
y = y[mask]
Y = F.one_hot(y)
```

即：

```text
把每个像素当一个样本
忽略标签 255 的 void / ignore 区域
把类别标签变成 one-hot 矩阵
```

然后更新：

```text
R = (R^{-1} + X^T X)^{-1}
weight = weight + R X^T (Y - X weight)
```

这就是代码层面的 C-RLS / RLS 闭式递归更新。

### 6.3 为什么它能减少遗忘？

普通 SGD 增量训练：

```text
只看新类数据
通过梯度更新共享参数
旧类决策边界可能被破坏
```

C-RLS 递归更新：

```text
用 R 保存历史特征相关信息
用 weight 保存历史分类头
新数据到来时，更新结果等价于历史数据 + 新数据一起做闭式求解
```

所以它不是“靠经验减少遗忘”，而是基于解析学习的 **weight-invariant property（权重不变性）**：

```text
递归增量学习的分类头 ≈ 联合训练的分类头
```

这里的“不忘”成立在一个重要前提下：

```text
特征提取器和 RHL 映射固定不变
```

如果你在后续 step 中继续用 SGD 改 backbone，这个等价性就不再成立。

---

## 7. 伪标签：为什么 sequential 下不重要，disjoint / overlapped 下重要？

在 **sequential setting（顺序完整标注设置）** 中，当前 step 的训练标签同时包含旧类和新类。因此旧类不会被错误标成 background。

但在 **disjoint / overlapped setting** 中，旧类像素可能被标为 background。

例子：

```text
step0 学过 person
step1 学 car
step1 图像里有 person 和 car
但标签只标 car，person 被标成 background
```

如果直接训练，模型会学到：

```text
person ≈ background
```

这就是 **semantic drift（语义漂移）**。

CFSSeg 的 2D 伪标签逻辑是：

```text
如果当前标签是 background
并且上一步模型高置信预测它是旧类
就把这个 background 像素替换为旧类伪标签
```

代码中对应：

```python
outputs = self.model_prev(images)
outputs = torch.sigmoid(outputs)  # bce_loss 时
pred_scores, pred_labels = torch.max(outputs, dim=1)

pseudo_labels = torch.where(
    (labels == 0) & (pred_labels > 0) &
    (pred_scores >= self.opts.pseudo_label_confidence),
    pred_labels,
    labels
)
```

对你的项目，如果只做 **sequential 15-5**，伪标签不是核心，因为 sequential 下旧类标签已经可见。若导师最终要求 disjoint / overlapped，则伪标签和你做的自适应阈值才是重点。

---

## 8. 快照集成：Snapshot Ensemble 的机制与作用

### 8.1 它是什么？

**Snapshot Ensemble（快照集成）** 来自论文 *Snapshot Ensembles: Train 1, get M for free*。

它的基本思想是：

```text
不从头训练 M 个模型
而是在一次训练过程中保存 M 个不同时间点的模型快照
测试时把这些快照模型的预测平均
```

普通 ensemble：

```text
训练模型 A
训练模型 B
训练模型 C
测试时融合 A/B/C
```

Snapshot ensemble：

```text
只训练一个模型
在不同 epoch / 不同学习率周期末尾保存 checkpoint 1/2/3
测试时融合 checkpoint 1/2/3
```

### 8.2 为什么它能产生多个“不同模型”？

经典 snapshot ensemble 会使用 **cyclic learning rate，循环学习率**。

学习率周期性地从大变小：

```text
cycle 1: high lr -> low lr -> 保存 snapshot 1
cycle 2: high lr -> low lr -> 保存 snapshot 2
cycle 3: high lr -> low lr -> 保存 snapshot 3
```

较大的学习率帮助模型跳出当前局部区域，较小的学习率帮助模型收敛到一个局部最小值。每个周期末尾的 checkpoint 可以看作一个不同的 ensemble member。

### 8.3 在分割任务中怎么融合？

对每个 snapshot 模型，输入同一张图，得到 logits：

```text
logits_1: [B, C, H, W]
logits_2: [B, C, H, W]
logits_3: [B, C, H, W]
```

推荐融合概率而不是融合最终类别 ID：

```python
prob_1 = softmax_or_sigmoid(logits_1)
prob_2 = softmax_or_sigmoid(logits_2)
prob_3 = softmax_or_sigmoid(logits_3)

prob_ens = (prob_1 + prob_2 + prob_3) / 3
pred = prob_ens.argmax(dim=1)
```

对于 SegACIL / CFSSeg 当前 `bce_loss`，代码评估时使用 sigmoid：

```python
outputs = torch.sigmoid(outputs)
```

所以 ensemble 时也应保持一致。

### 8.4 它的作用

Snapshot ensemble 的作用主要是：

| 作用 | 解释 |
|---|---|
| 提升稳定性 | 多个 checkpoint 的平均能降低单模型偶然误差 |
| 提升泛化 | 不同局部最优点可能捕获略有差异的决策边界 |
| 成本低于普通 ensemble | 不需要从头训练多个完整模型 |
| 适合做“集成系统”标签 | 符合导师项目中的 ensemble system 口径 |

但要注意：

```text
CFSSeg 的增量阶段不是多 epoch SGD，所以 snapshot ensemble 更适合放在 step0 DeepLab 训练阶段。
```

也就是说，快照应该来自：

```text
step0 的 DeepLabV3 + ResNet101 训练过程
```

而不是来自 C-RLS 的增量更新过程。

---

## 9. 多随机种子集成：Multi-seed Ensemble 的机制与作用

### 9.1 它是什么？

**Multi-seed Ensemble（多随机种子集成）** 是最常见、最直接的深度集成方式之一。

做法是：

```text
使用相同代码、相同数据、相同超参数
只改变 random seed
训练出多个模型
推理时平均它们的预测概率
```

例如：

```text
seed 0 -> model_0
seed 1 -> model_1
seed 2 -> model_2
```

推理时：

```python
prob_ens = (prob_0 + prob_1 + prob_2) / 3
pred = prob_ens.argmax(dim=1)
```

### 9.2 在 CFSSeg 中，多随机种子可以作用在哪些地方？

CFSSeg 中的随机性来源主要有：

| 随机来源 | 是否建议用于 ensemble | 说明 |
|---|---:|---|
| step0 DeepLab 初始化 / dataloader 顺序 | 可以 | 会影响基础特征提取器 |
| RHL 随机矩阵 \(\Phi_E\) | 强烈推荐 | 成本低，和方法天然相关 |
| dropout / 数据增强随机性 | 可以 | 但解释性不如 RHL seed |
| 伪标签阈值随机扰动 | 不建议第一版做 | 不稳定，容易污染解释 |

其中最适合你项目的是：

```text
固定同一个 step0 backbone
只改变 RHL random seed
得到多个解析分类头
推理时融合
```

原因是：

```text
RHL 本来就是随机高维映射
不同随机映射相当于不同随机子空间
C-RLS 增量阶段很快，训练多个 RHL head 成本远低于训练多个完整 DeepLab
```

### 9.3 多随机种子集成的作用

它的核心作用是降低随机映射导致的方差。

单个 RHL：

```text
随机投影矩阵可能刚好对某些类别不友好
```

多个 RHL：

```text
不同随机投影从不同角度观察同一 dense feature
错误不完全一致
平均后更稳定
```

这特别适合你的项目，因为你不需要把 ensemble 做成复杂主线，只需要让系统中确实存在合理的 ensemble 模块。

---

## 10. 方案 A：RHL 多随机种子集成

### 10.1 设计目标

方案 A 的目标是：

```text
在不改变 CFSSeg 主线的前提下，构建一个轻量、低成本、可解释的集成系统。
```

它的最终表述可以是：

> 我们基于 CFSSeg 的随机高维映射机制，构建 RHL Subspace Ensemble。具体地，在相同冻结 DeepLab 特征提取器上，使用多个不同随机种子的 RHL 生成多个高维随机子空间，并分别通过 C-RLS 得到解析分类头。推理阶段对多个解析头的像素级预测概率进行平均，得到最终集成结果。

### 10.2 为什么这样设计？

因为它同时满足 4 个条件：

| 条件 | 是否满足 | 说明 |
|---|---:|---|
| 不改变持续学习主线 | 是 | 主体仍是 CFSSeg |
| 是真正的 ensemble | 是 | 多个 RHL head 产生多个预测 |
| 工程成本低 | 是 | 不需要训练多个完整 backbone |
| 论文故事自然 | 是 | RHL 本身就是核心组件 |

它比“强行加一个复杂 ensemble 模块”更合理。

### 10.3 具体实现路线

#### Step 1：固定 step0 checkpoint

先跑标准 step0：

```bash
python train.py \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 0 \
  --model deeplabv3_resnet101 \
  --train_epoch 50 \
  --batch_size 32 \
  --loss_type bce_loss \
  --subpath baseline_step0_seed1 \
  --random_seed 1
```

得到：

```text
checkpoints/baseline_step0_seed1/voc/15-5/sequential/step0/final.pth
```

#### Step 2：用不同 RHL seed 跑 step1

保持 step0 checkpoint 不变，只改变 RHL seed 和输出目录。

```bash
for seed in 1 2 3; do
  python train.py \
    --dataset voc \
    --task 15-5 \
    --setting sequential \
    --curr_step 1 \
    --model deeplabv3_resnet101 \
    --buffer 8192 \
    --gamma 1 \
    --loss_type bce_loss \
    --base_subpath baseline_step0_seed1 \
    --subpath rhl_ens_seed${seed} \
    --random_seed ${seed}
done
```

如果你使用 CFSSeg 新库中的 RHL 归一化，可以加：

```bash
--rhl_norm l2_sqrt --rhl_stats
```

建议第一轮不要同时改变太多变量。实验顺序建议：

```text
A0：baseline RHL，seed=1
A1：baseline RHL，seed=1/2/3 ensemble
A2：l2_sqrt RHL，seed=1
A3：l2_sqrt RHL，seed=1/2/3 ensemble
```

#### Step 3：实现 ensemble 推理脚本

新增一个脚本，例如：

```text
tools/eval_ensemble.py
```

核心伪代码：

```python
models = [load_ckpt(path)[0].eval().cuda() for path in ckpt_paths]
metrics.reset()

for images, labels, _ in loader:
    images = images.cuda()
    labels = labels.cuda()

    probs = []
    for model in models:
        logits = model(images)              # [B, Hf, Wf, C]
        prob = torch.sigmoid(logits)        # bce_loss 协议
        prob = prob.permute(0, 3, 1, 2)     # [B, C, Hf, Wf]
        prob = F.interpolate(prob, labels.shape[-2:], mode='bilinear')
        probs.append(prob)

    prob_ens = torch.stack(probs, dim=0).mean(dim=0)
    preds = prob_ens.argmax(dim=1)
    metrics.update(labels.cpu().numpy(), preds.cpu().numpy())

print(metrics.get_results())
```

#### Step 4：输出单模型与集成系统结果

最终表格建议这样组织：

| 方法 | 集成方式 | 0-15 mIoU | 16-20 mIoU | all mIoU |
|---|---|---:|---:|---:|
| CFSSeg single | 无 | x.xx | x.xx | x.xx |
| CFSSeg + RHL-SE-3 | 3 个 RHL seed 概率平均 | x.xx | x.xx | x.xx |
| CFSSeg + RHL-SE-5 | 5 个 RHL seed 概率平均 | x.xx | x.xx | x.xx |

`RHL-SE` 可以解释为：

```text
Random Hidden Layer Subspace Ensemble
随机隐藏层子空间集成
```

### 10.4 预期效果

合理预期是：

```text
提升幅度：0.1 ~ 1.0 mIoU 不等
主要收益：结果更稳定，方差更小
最差情况：基本持平，不应明显下降
```

由于你的单模型指标已经超过项目要求，即使 ensemble 只持平，也能作为“集成系统”成立。

### 10.5 风险与注意事项

| 风险 | 说明 | 处理 |
|---|---|---|
| 多个 RHL 结果太相似 | ensemble 提升有限 | 增加 seed 数到 5，或加入 RHL norm 差异 |
| 单个 seed 异常差 | 平均后拖累结果 | 报告 mean/std，剔除异常必须有规则 |
| 显存压力 | 同时加载多个模型可能爆显存 | 串行推理，累加概率到 CPU 或逐模型 forward |
| 结果不可复现 | seed 设置不全 | 同步设置 torch / cuda / numpy / random seed |

---

## 11. 方案 B：step0 快照集成 + 解析增量头

### 11.1 设计目标

方案 B 是把经典 Snapshot Ensemble 放到 CFSSeg 的 step0 基础训练阶段。

它的逻辑是：

```text
step0 训练 DeepLab 时保存多个 checkpoint
每个 checkpoint 都作为一个 frozen feature extractor
分别接 RHL + C-RLS 完成 step1 增量学习
最终融合多个完整 CFSSeg 模型的预测
```

### 11.2 为什么这样设计？

因为 snapshot ensemble 需要一个多 epoch 梯度训练过程，而 CFSSeg 的增量阶段不是这种训练过程。

所以：

```text
不建议在 C-RLS 阶段硬做 snapshot
建议在 step0 DeepLab 训练阶段做 snapshot
```

这和经典 snapshot ensemble 的思想更一致。

### 11.3 具体实现路线

#### Step 1：修改 step0 训练保存策略

在 step0 训练时保存多个 checkpoint，例如：

```text
epoch 30 -> snapshot_30.pth
epoch 40 -> snapshot_40.pth
epoch 50 -> snapshot_50.pth
```

最小代码改动可以在 step0 每个 epoch 末尾加入：

```python
snapshot_epochs = [30, 40, 50]
if (epoch + 1) in snapshot_epochs:
    save_ckpt(
        self.root_path + f"snapshot_epoch_{epoch+1}.pth",
        self.model,
        self.optimizer,
        curr_score,
    )
```

更接近原始 snapshot ensemble 的做法是加入 cyclic learning rate，但第一版不建议上来就改学习率策略。你的时间紧，建议先做：

```text
last-k checkpoint ensemble
```

它比完整 cyclic snapshot 简单，仍然可以作为“快照集成”的工程变体。

#### Step 2：每个 snapshot 分别跑 step1

假设有三个 step0 snapshot：

```text
snapshot_30.pth
snapshot_40.pth
snapshot_50.pth
```

你需要让 step1 能指定加载某个 step0 checkpoint。可以通过以下两种方式：

| 方式 | 工程复杂度 | 推荐度 |
|---|---:|---:|
| 临时复制 snapshot 为 step0/final.pth | 低 | 快速验证推荐 |
| 增加 `--base_ckpt` 参数 | 中 | 正式代码推荐 |

快速验证方式：

```bash
for e in 30 40 50; do
  cp checkpoints/snapshot_exp/voc/15-5/sequential/step0/snapshot_epoch_${e}.pth \
     checkpoints/snapshot_exp/voc/15-5/sequential/step0/final.pth

  python train.py \
    --dataset voc \
    --task 15-5 \
    --setting sequential \
    --curr_step 1 \
    --model deeplabv3_resnet101 \
    --buffer 8192 \
    --gamma 1 \
    --subpath snapshot_step1_epoch${e} \
    --random_seed 1
done
```

正式代码方式：增加：

```python
parser.add_argument("--base_ckpt", type=str, default=None)
```

在 `curr_step == 1` 加载 step0 checkpoint 时优先使用：

```python
if self.opts.base_ckpt is not None:
    self.ckpt = self.opts.base_ckpt
```

#### Step 3：融合多个 snapshot 分支

每个 snapshot 分支最终都会得到一个完整模型：

```text
snapshot_epoch_30 -> CFSSeg model A
snapshot_epoch_40 -> CFSSeg model B
snapshot_epoch_50 -> CFSSeg model C
```

推理融合和方案 A 一样：

```python
prob_ens = mean([prob_A, prob_B, prob_C])
pred = prob_ens.argmax(dim=1)
```

### 11.4 预期效果

合理预期：

```text
如果 snapshot 之间差异足够，可能提升 0.2 ~ 1.0 mIoU
如果 step0 已经充分收敛且 snapshot 很接近，提升可能很小
```

它的主要价值是：

```text
更符合传统 snapshot ensemble 定义
更容易向导师解释“snapshot”这个词
```

但它的工程成本高于方案 A，因为每个 snapshot 都要重新做一次 step1 realignment + incremental update。

### 11.5 风险与注意事项

| 风险 | 说明 | 处理 |
|---|---|---|
| snapshot 之间太接近 | ensemble 提升有限 | 选择间隔更远的 epoch 或使用 cyclic LR |
| 早期 snapshot 单模型弱 | 拖累平均结果 | 只选验证集表现较好的 snapshot |
| 改 LR 影响 baseline | 不利于公平比较 | 第一版只做 last-k checkpoint |
| 多次 step1 耗时 | 比方案 A 成本高 | 只做 3 个 snapshot |

---

## 12. 方案 A 和方案 B 对比

| 维度 | 方案 A：RHL 多随机种子集成 | 方案 B：step0 快照集成 |
|---|---|---|
| 核心差异来源 | RHL 随机映射不同 | step0 backbone checkpoint 不同 |
| 是否贴合 CFSSeg 核心 | 很贴合 | 中等贴合 |
| 是否贴合 snapshot 术语 | 一般 | 很贴合 |
| 工程复杂度 | 低 | 中 |
| 训练成本 | 低 | 中等 |
| 推理成本 | 多模型推理 | 多模型推理 |
| 论文故事自然度 | 高 | 中高 |
| 推荐优先级 | 第一优先 | 第二优先 |

建议采用：

```text
先做方案 A，保证项目指标与论文标签
再做方案 B，作为补充实验或导师提到 snapshot 时的回应
```

---

## 13. 推荐实验矩阵

### 13.1 最小可交付实验

| 编号 | 实验 | 目的 |
|---|---|---|
| E0 | CFSSeg baseline single, 15-5 sequential | 对齐原论文和项目指标 |
| E1 | CFSSeg + RHL seed 1/2/3 single | 观察 seed 方差 |
| E2 | CFSSeg + RHL 3-seed ensemble | 构建集成系统 |
| E3 | CFSSeg + RHL norm single | 验证你已有改动 |
| E4 | CFSSeg + RHL norm 3-seed ensemble | 最终推荐系统 |

### 13.2 可选扩展实验

| 编号 | 实验 | 目的 |
|---|---|---|
| S1 | step0 snapshot 30/40/50 | 验证 snapshot 差异 |
| S2 | snapshot branches + C-RLS | 构建 snapshot ensemble |
| S3 | snapshot + RHL seed hybrid | 上限实验，不建议第一轮做 |

---

## 14. 论文写作中可以怎么表述？

### 14.1 方法名建议

可选名称：

```text
CFSSeg-RSE: CFSSeg with Random Subspace Ensemble
CFSSeg-RHL-E: CFSSeg with Random Hidden Layer Ensemble
CFE-CISS: Closed-Form Ensemble for Class-Incremental Semantic Segmentation
```

最稳妥的是：

```text
CFSSeg-RHL-E
```

因为它直接指出 ensemble 来自 RHL。

### 14.2 方法段落模板

> To satisfy the system-level ensemble requirement without changing the core continual learning objective, we introduce a lightweight Random Hidden Layer Ensemble on top of the closed-form CISS framework. Given a frozen DeepLab feature extractor, we instantiate multiple independently initialized RHL mappings and train their analytic classifiers using the same C-RLS update. During inference, the pixel-wise probabilities of all analytic heads are averaged to produce the final segmentation map. This design preserves the exemplar-free and gradient-free incremental update property of CFSSeg while reducing the variance induced by a single random projection.

中文含义：

> 为满足系统级集成要求，同时不改变持续学习主任务，我们在闭式解 CISS 框架上引入轻量随机隐藏层集成。给定冻结的 DeepLab 特征提取器，我们构造多个独立初始化的 RHL 映射，并使用相同的 C-RLS 更新训练其解析分类头。推理阶段，对所有解析头的像素级概率进行平均，得到最终分割图。该设计保留了 CFSSeg 无样本回放、无梯度增量更新的特性，同时降低了单个随机投影带来的方差。

---

## 15. 最终建议

你现在最应该做的是：

```text
1. 先固定论文配置，复现 15-5 sequential 单模型结果。
2. 在 CFSSeg 新库中保留 RHL norm 改动，但不要和太多变量混在一起。
3. 优先实现 RHL 多随机种子集成，作为项目“集成系统”。
4. 如果导师强调 snapshot，再补做 step0 last-k snapshot ensemble。
5. 论文主线仍然写持续学习 / 解析增量语义分割，不要把集成学习写成核心创新。
```

最推荐的最终系统结构是：

```text
DeepLabV3 + ResNet101 step0 training
  ↓
Frozen dense feature extractor
  ↓
RHL seed 1 -> C-RLS head 1
RHL seed 2 -> C-RLS head 2
RHL seed 3 -> C-RLS head 3
  ↓
probability averaging
  ↓
final segmentation map
  ↓
0-20 mIoU
```

这套方案的优势是：

```text
简单
合理
和原论文机制贴合
能满足“集成系统”标签
不破坏持续学习主线
容易在短时间内完成实验和写作
```

---

## 参考依据

1. CFSSeg 原论文：*CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds*。
2. SegACIL 原代码库：`Ruiter578/SegACIL`，重点文件包括 `train.py`、`trainer/trainer.py`、`network/Buffer.py`、`network/AnalyticLinear.py`、`utils/tasks.py`。
3. CFSSeg 新代码库：`Ruiter578/CFSSeg`，重点新增内容包括 RHL 输出归一化参数 `rhl_norm`、`rhl_norm_eps`、`rhl_stats`，以及 `RandomBuffer` 中的 `l2`、`l2_sqrt`、`layernorm` 选项。
4. Snapshot Ensemble 经典论文：Gao Huang et al., *Snapshot Ensembles: Train 1, get M for Free*, 2017。
5. Deep Ensemble 经典论文：Balaji Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, 2017。
