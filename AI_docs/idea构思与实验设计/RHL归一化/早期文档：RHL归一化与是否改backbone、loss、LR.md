# 早期-方案一：RHL归一化与是否改backbone、loss、LR.md

早期方案，仅作参考。

## 核心判断

CFSSeg 的主线不是反复调 loss 或 LR，而是 **冻结 backbone + RHL 高维映射 + C-RLS 闭式更新分类头 + 伪标签修复 semantic drift**。代码报告也明确指出，当前 SegACIL 的核心改动入口应优先放在 `network/Buffer.py` 的 RHL / Buffer 和 `trainer/trainer.py:get_pseudo_labels` 的伪标签逻辑上，而不是先动 C-RLS 主公式。

---

# 1. 改 backbone、loss、学习率能不能提升性能？

可以，但要分清它们作用在哪一段。

## 1.1 换 backbone：不只是 step0 起作用，但训练通常只在 step0 起作用

CFSSeg 的机制是：

```text
step0：用 SGD/BP 训练 backbone + segmentation head
step>0：冻结 encoder/backbone，用 RHL + C-RLS 闭式更新分类头
```

所以 backbone 的参数主要在 step0 训练或加载，但它的影响会贯穿所有增量阶段。因为后续每个 step 的新图像、新点云都要先经过同一个冻结 backbone 得到特征：

$$
F_t = f_\theta(X_t)
$$

然后再进入 RHL：

$$
E_t = \operatorname{ReLU}(F_t \Phi_E)
$$

再进入闭式解分类头。换句话说，backbone 是在 step0 决定的，但它影响 step1、step2、... 的所有特征质量。不是“只在 step0 有作用”。

### ResNet101 能不能换成 DINOv3？

原则上可以，但不是简单把 `resnet101.py` 改成 `dinov3.py` 就完事。

DINOv3 是 Meta 发布的通用视觉 backbone，官方说明强调它能提供强 dense image features，并支持 segmentation 等 dense prediction 任务；官方 GitHub 也已经释放了 semantic segmentation 线性探测相关代码。([Meta AI](https://ai.meta.com/research/dinov3/?utm_source=chatgpt.com "DINOv3 - Meta AI"))

但在 SegACIL 里，原模型是：

```text
DeepLabV3 + ResNet101
```

DeepLabV3 期望 CNN feature map，例如：

```text
B × C × H' × W'
```

而 ViT 型 DINOv3 通常输出 patch tokens，例如：

```text
B × N × C
```

所以你至少需要解决：

1. **token 到 feature map 的 reshape**：把 (N) 个 patch token reshape 成 (H_p \times W_p)；
2. **多尺度特征问题**：DeepLab/ASPP 原本依赖 CNN 空间特征，ViT 单尺度 token 不一定直接兼容；
3. **通道数适配**：DINOv3 hidden dim 和 DeepLab classifier 输入通道不同，需要 `1×1 conv` 或 linear projection；
4. **checkpoint 不兼容**：换 backbone 后，原来的 RHL 统计矩阵 (R)、解析头权重、checkpoint 基本都不能复用；
5. **论文公平性问题**：如果你只换更强 backbone，审稿人会认为提升来自 foundation model，而不是你的持续学习方法。

更稳的路线是：

```text
第一阶段：不要换 backbone，先做 RHL / pseudo-label 小改动
第二阶段：尝试 ConvNeXt-DINOv3 或 ViT-DINOv3 frozen feature extractor
第三阶段：把故事写成 “foundation dense features + analytic continual segmentation”
```

如果用 DINOv3，建议优先考虑 **ConvNeXt 版本 DINOv3**，因为它更接近 CNN feature map，接入 DeepLab 类 decoder 的工程阻力小于 ViT tokens。

---

## 1.2 BCE 换 Dice：可能提升 step0，但不会直接改变 C-RLS 增量闭式解

你当前脚本里是：

```bash
LOSS_TYPE="bce_loss"
```

代码报告也记录当前配置中 `loss_type=bce_loss` 表示多通道 BCE-with-ignore；而增量阶段 `curr_step > 0` 主要通过 AIR / C-RLS 做闭式更新，不再走常规 `loss.backward()`。

所以：

| 改动                   | 主要影响                                                           |
| ---------------------- | ------------------------------------------------------------------ |
| `BCE -> Dice`        | 主要影响 step0 的 DeepLab 训练质量、logit calibration、base 类特征 |
| 学习率 / LR policy     | 主要影响 step0 backbone/head 的 BP 训练                            |
| `gamma`              | 直接影响闭式解 ridge regression 正则强度                           |
| `buffer`             | 直接影响 RHL 高维特征维度和线性可分性                              |
| RHL normalization      | 直接影响 (E^\top E) 条件数和闭式解稳定性                           |
| pseudo-label threshold | 直接影响增量阶段旧类像素是否被错误当成 background                  |

Dice loss（Dice 损失）对类别不均衡更敏感，可能改善小目标或 rare class，但纯 Dice 往往会损害概率校准。伪标签依赖旧模型置信度，如果 step0 训练后输出概率不可靠，后面 pseudo-labeling 会受影响。因此更建议先做组合损失：

$$
\mathcal{L} = \mathcal{L}_{BCE} + \lambda \mathcal{L}_{Dice}
$$

而不是直接把 BCE 完全替换成 Dice。

推荐先试：

```text
BCE baseline
BCE + 0.5 Dice
BCE + 1.0 Dice
```

不要一开始同时改 backbone、loss、LR、RHL、pseudo-label，否则结果不可解释。

---

# 2. RHL 输出归一化：原理、作用、代码方案

## 2.1 RHL 在 CFSSeg 里做什么？

RHL（Random Hidden Layer，随机高维隐层）把冻结 encoder 输出的特征映射到更高维空间：

$$
E = \operatorname{ReLU}(F \Phi_E)
$$

其中：

- (F)：backbone/encoder 输出的像素或点特征；
- (\Phi_E)：随机初始化并固定的高维映射矩阵；
- (E)：进入闭式解分类头的高维特征。

然后解析分类头通过 ridge regression（岭回归）得到：

$$
\hat{\Phi} = (E^\top E + \gamma I)^{-1}E^\top Y
$$

代码报告显示，当前 `RandomBuffer` 的核心逻辑基本是随机权重注册为 buffer，然后返回 `activation(super().forward(X))`；也就是 **随机线性映射 + 激活函数**，没有额外归一化。

---

## 2.2 为什么要归一化 RHL 输出？

RHL 输出没有归一化时，可能出现三个问题。

### 问题 1：特征尺度不稳定

不同 batch、不同 step、不同类别的 (E_i) 范数可能差很多。岭回归最小化的是：

$$
|Y - E\Phi|_F^2 + \gamma |\Phi|_F^2
$$

如果某些样本的 (|E_i|) 特别大，它们会在 (E^\top E) 和 (E^\top Y) 中占更大权重，相当于被隐式加权。

### 问题 2：矩阵条件数变差

C-RLS 要更新类似：

$$
\Psi_t = (\Psi_{t-1}^{-1} + E_t^\top E_t)^{-1}
$$

如果 (E_t^\top E_t) 条件数很差，就容易出现：

- `NaN`
- `cusolver error`
- `torch.inverse` / `torch.linalg.inv` 不稳定
- 某些类别权重异常大

### 问题 3：跨 step 的特征尺度漂移

增量阶段每个 step 的图像类别分布不同。即使 backbone 冻结，输入分布变了，RHL 输出的尺度也可能变。这样同一个 `gamma=1` 在不同 step 的有效正则强度不一致。

---

## 2.3 最小可行改动：对 RHL 输出做 row-wise L2 normalization

建议先做最简单、最不破坏理论主线的版本：

$$
\tilde{E}_i = \frac{E_i}{|E_i|_2 + \epsilon}
$$

其中：

- ($E_i$)：第 (i) 个 pixel/point 的 RHL 特征；
- ($\epsilon$)：防止除零的小常数，例如 `1e-6`；
- ($\tilde{E}_i$)：归一化后的特征。

这样做的直觉是：让闭式解主要利用 **特征方向**，减少 **特征范数大小** 对最小二乘的隐式加权。

这个改动不会破坏 CFSSeg 的闭式解结构。因为你只是把原来的固定特征映射：

$$
x \mapsto E
$$

换成了另一个固定特征映射：

$$
x \mapsto \tilde{E}
$$

只要这个映射在 joint learning 和 incremental learning 中一致，C-RLS 的递归等价逻辑仍然成立。

---

## 2.4 不建议一开始做的归一化

暂时不要先上：

```text
BatchNorm
可训练 LayerNorm
用全训练集均值方差做 whitening
step-wise 学习一个缩放参数
```

原因是这些方法可能引入额外可训练参数或依赖数据分布统计，会把故事从“固定随机映射 + 闭式解”变复杂。

第一版建议只做：

```text
none
l2
l2_sqrt
layernorm_no_affine
```

其中 `layernorm_no_affine` 指 `elementwise_affine=False`，不引入可训练参数。

---

## 2.5 代码改动方案

### 第一步：在 `utils/parser.py` 增加参数

```python
parser.add_argument(
    "--rhl_norm",
    type=str,
    default="none",
    choices=["none", "l2", "l2_sqrt", "layernorm"],
    help="Normalization applied to RHL/Buffer outputs before analytic fitting."
)

parser.add_argument(
    "--rhl_norm_eps",
    type=float,
    default=1e-6,
    help="Epsilon for RHL output normalization."
)
```

---

### 第二步：修改 `network/Buffer.py`

假设你当前 `RandomBuffer.forward()` 类似：

```python
return self.activation(super().forward(X))
```

可以改成：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomBuffer(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=nn.ReLU(),
        rhl_norm="none",
        rhl_norm_eps=1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.activation = activation
        self.rhl_norm = rhl_norm
        self.rhl_norm_eps = rhl_norm_eps

        if self.rhl_norm == "layernorm":
            self.rhl_ln = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.rhl_ln = None

        # 如果原代码用 register_buffer 管理 weight，请保留原实现。
        # 不要把随机映射变成可训练参数。
        self.reset_parameters()

    def forward(self, X):
        # X: [..., in_features]
        Z = super().forward(X)
        Z = self.activation(Z)

        if self.rhl_norm == "none":
            return Z

        if self.rhl_norm == "l2":
            # 每个 pixel/point 的高维特征向量做 L2 归一化
            return F.normalize(Z, p=2, dim=-1, eps=self.rhl_norm_eps)

        if self.rhl_norm == "l2_sqrt":
            # 保持方向，同时把每个样本的总能量约束到 sqrt(D)
            D = Z.shape[-1]
            return F.normalize(Z, p=2, dim=-1, eps=self.rhl_norm_eps) * math.sqrt(D)

        if self.rhl_norm == "layernorm":
            return self.rhl_ln(Z)

        raise ValueError(f"Unknown rhl_norm: {self.rhl_norm}")
```

注意：如果你的原始 `RandomBuffer` 不是标准 `nn.Linear`，而是自己用 `register_buffer("weight", W)` 管理随机矩阵，就不要照抄整个类，只把 `forward()` 里的归一化逻辑加进去即可。

---

### 第三步：在 `trainer/trainer.py` 创建 Buffer 的地方传参

找到类似：

```python
self.buffer = RandomBuffer(
    in_features=...,
    out_features=opts.buffer,
    activation=nn.ReLU()
)
```

改成：

```python
self.buffer = RandomBuffer(
    in_features=...,
    out_features=opts.buffer,
    activation=nn.ReLU(),
    rhl_norm=opts.rhl_norm,
    rhl_norm_eps=opts.rhl_norm_eps,
)
```

如果代码里叫 `self.AIR`、`self.rhl` 或其他名字，就在对应创建 `RandomBuffer` 的地方传入。

---

### 第四步：运行命令增加参数

```bash
python train.py \
  ... \
  --buffer 8196 \
  --gamma 1 \
  --rhl_norm l2 \
  --rhl_norm_eps 1e-6
```

建议实验表：

| 实验                 | `rhl_norm` | `gamma` |
| -------------------- | ------------ | --------: |
| baseline             | none         |         1 |
| RHL-L2               | l2           |         1 |
| RHL-L2 + gamma small | l2           |       0.1 |
| RHL-L2 + gamma large | l2           |        10 |
| RHL-LayerNorm        | layernorm    |         1 |

RHL normalization 改了 (E^\top E) 的尺度，所以必须同时扫一下 `gamma`，否则容易误判。

---

## 2.6 怎么验证 RHL 归一化有没有用？

至少记录这些量：

```python
with torch.no_grad():
    row_norm = E.norm(dim=-1)
    print("RHL row norm mean:", row_norm.mean().item())
    print("RHL row norm std:", row_norm.std().item())
    print("RHL row norm max:", row_norm.max().item())
    print("RHL has nan:", torch.isnan(E).any().item())
```

不要每个 batch 都算完整 condition number，`buffer=8196` 时很贵。可以先看：

```text
row_norm mean/std
E 是否 NaN
R 是否 NaN
解析头 weight 范数
old/new/all mIoU
```

如果归一化有效，通常会看到：

```text
RHL row norm 更稳定
矩阵求逆错误减少
不同 step 的权重范数不再剧烈波动
old mIoU 更稳
new mIoU 不明显下降，甚至略升
```

---

原因：

1. **RHL normalization** 改动小、理论故事干净、不依赖 setting，sequential / disjoint / overlap 都能测。
2. **adaptive pseudo-label** 对 disjoint / overlap 更有意义；如果你只跑 sequential，它基本不是核心矛盾。
3. **换 backbone** 可能提升最大，但工程成本高、审稿解释难、容易变成“强 backbone 论文”。
4. **loss / LR** 主要作用 step0，不是 CFSSeg 的闭式解创新核心。

---
