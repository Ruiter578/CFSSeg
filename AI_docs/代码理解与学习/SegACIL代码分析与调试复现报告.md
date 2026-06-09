# SegACIL 代码分析与调试复现报告

> 项目：SegACIL / CFSSeg 2D+3D 语义分割实现  
> 当前报告范围：代码结构、方法映射、数据与训练流程、关键实现细节、初步调试观察。  
> 说明：复现实验数值结果将在当前报错修复并完成 baseline 后补充，本版重点作为代码理解与调试依据。

---

## 1. 项目实现范围

SegACIL 是 CFSSeg 中 2D 类增量语义分割流程的工程实现。代码以 Pascal VOC2012 为主要运行对象，围绕 DeepLabV3 / ResNet-101、随机高维映射层和递归解析分类头构建。仓库中也包含 ADE20K 与 Cityscapes domain 相关文件，但从当前代码分析看，VOC2012 是最完整、最适合优先复现的主流程。论文中的 3D S3DIS / ScanNet / DGCNN 分支在当前仓库材料中未观察到完整实现。

项目结构可分为五层：

| 层级 | 主要文件 | 作用 |
|---|---|---|
| 入口与配置 | `train.py`, `run.sh`, `utils/parser.py` | 接收命令行参数、设置任务、启动训练 |
| 数据协议 | `datasets/voc.py`, `datasets/init_dataset.py`, `utils/tasks.py` | 构造 VOC 数据集、增量类别划分、标签重映射 |
| 分割网络 | `network/modeling.py`, `network/_deeplab.py`, `network/backbone/resnet.py` | DeepLabV3 / ResNet-101 主体 |
| 解析学习模块 | `network/Buffer.py`, `network/AnalyticLinear.py` | RHL 高维映射与递归闭式分类头 |
| 训练与评估 | `trainer/trainer.py`, `metrics/stream_metrics.py`, `utils/ckpt.py` | step0 训练、step>0 解析更新、mIoU 评估、checkpoint |

该代码结构与 CFSSeg 的方法拆分较一致：初始阶段训练 encoder，增量阶段冻结 encoder，通过 RHL + C-RLS 更新 dense classifier。

---

## 2. 方法与代码的对应关系

| 论文概念 | 数学符号 | 代码位置 | 实现含义 |
|---|---|---|---|
| Base encoder training | $q_{\theta_0}$ | `trainer/trainer.py`, `network/_deeplab.py` | step0 使用 BP 训练 DeepLabV3 |
| Frozen encoder | fixed encoder | `trainer/trainer.py` 中 step>0 分支 | 加载旧模型并移除原 classifier head |
| RHL | $E=\mathrm{ReLU}(X^{encoder}\Phi_E)$ | `network/Buffer.py:RandomBuffer` | 将 256 维 dense feature 映射到 `buffer` 维 |
| Ridge classifier | $\hat{\Phi}$ | `network/AnalyticLinear.py` | 保存解析分类头权重 `weight` |
| AutoCor memory | $\Psi_t$ / $R_t$ | `RecursiveLinear.R` | 保存正则化特征自相关矩阵的逆 |
| C-RLS update | Eq. C-RLS | `RecursiveLinear.fit()` | 根据当前数据更新 `R` 与 `weight` |
| Pseudo-labeling | $U_i,\tau$ | `trainer.get_pseudo_labels()` | 用旧模型高置信预测替换部分背景标签 |
| CSS protocol | $S_t,\mathcal{C}_t$ | `utils/tasks.py`, `datasets/voc.py` | 15-1 / 10-1 等类别划分与标签映射 |

需要特别指出，`--method acil` 在当前代码中更多是脚本参数，并未观察到对 `opts.method` 的实际分支控制。真正决定训练逻辑的是 `curr_step`、`setting`、`model`、`loss_type` 与 `use_pseudo_label`。

---

## 3. 数据流：从 VOC2012 到增量标签

### 3.1 数据目录

当前运行脚本使用：

```bash
DATA_ROOT="/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
```

`VOCSegmentation` 主要依赖以下目录：

```text
VOC2012/
├── JPEGImages/
├── SegmentationClassAug/
├── SegmentationClass/
└── ImageSets/Segmentation/val.txt
```

同时，`utils/tasks.py` 还会读取：

```text
datasets/data/voc/train_cls.txt
datasets/data/voc/val_cls.txt
```

这两个文件记录每张图中包含的前景类别，用于按当前 step 筛选样本。若缺失，会导致数据集初始化失败。

### 3.2 类别划分

`train.py` 根据当前 `curr_step` 调用 `get_tasks()`，构造 `opts.num_classes` 与 `opts.target_cls`。以 VOC `15-1` 为例：

| step | 学习类别 | `opts.num_classes` |
|---:|---|---|
| 0 | 背景 + 1-15 类 | `[1, 15]` |
| 1 | 新增第 16 类 | `[1, 15, 1]` |
| 2 | 新增第 17 类 | `[1, 15, 1, 1]` |
| 5 | 新增第 20 类 | `[1, 15, 1, 1, 1, 1, 1]` |

其中背景类被单独计为 1 类，前景类别按增量阶段扩展。

### 3.3 setting 对样本与标签的影响

| setting | 样本选择 | 标签处理 | 难度 |
|---|---|---|---|
| `sequential` | 当前数据中旧类与新类标注均保留 | 基本不将旧类压成背景 | 较易 |
| `disjoint` | 当前数据包含新类，不包含未来类 | 非目标类映射为背景 | 中等 |
| `overlap` | 当前数据只需出现新类，可混有旧类和未来类 | 非目标类映射为背景 | 较难 |

`sequential` 更适合作为第一条复现链路；`disjoint` 和 `overlap` 更能检验 pseudo-labeling 对 semantic drift 的作用。

---

## 4. 训练流程分析

### 4.1 step0：常规 DeepLabV3 训练

第 0 步是标准分割模型训练：

```text
image -> DeepLabV3 backbone -> ASPP/classifier -> logits
      -> interpolate to label size -> BCE loss
      -> loss.backward() -> optimizer.step()
```

关键特征：

- 使用 SGD 优化 backbone 和 classifier；
- backbone 学习率为 0.001，classifier 学习率为 0.01；
- 使用 `bce_loss`，忽略标签 255；
- 每轮验证并根据当前类别 mIoU 保存 checkpoint。

这一步没有使用解析头，也不涉及 `Buffer` 或 `RecursiveLinear`。因此 step0 的 OOM 属于常规分割训练显存问题。

### 4.2 step1：解析重对齐与第一轮增量

第 1 步是最关键的转换阶段。流程为：

```text
load step0 DeepLab checkpoint
  -> remove classifier head / set Identity
  -> construct AIR(backbone, RandomBuffer, RecursiveLinear)
  -> temporarily use step0 data to fit analytic head
  -> save step0/final.pth
  -> use step1 data to fit new class
  -> save step1/final.pth
```

这相当于把原始 BP 训练得到的分割模型转入解析学习范式。若 step0 checkpoint 不存在，step1 必然失败。

### 4.3 step>1：递归闭式更新

第 2 步及之后不再做反向传播：

```text
load previous AIR final.pth
  -> images through frozen backbone
  -> feature_expansion
  -> RandomBuffer
  -> RecursiveLinear.fit(X, y)
  -> update R and weight
  -> save final.pth
```

此时模型的“训练”本质是矩阵运算。`RandomBuffer.weight` 是 buffer 而非 parameter；`RecursiveLinear.R` 和 `RecursiveLinear.weight` 由闭式更新写入，不经 optimizer。

---

## 5. 解析学习模块细读

### 5.1 `RandomBuffer`

`network/Buffer.py` 中 `RandomBuffer` 继承 `torch.nn.Linear`，但将 `weight` 注册为 buffer：

```python
self.register_buffer("weight", W)
```

这意味着它会随模型保存和迁移到 GPU，但不会参与梯度训练。其 forward 将输入转换到 buffer 权重 dtype/device，并执行线性映射与激活：

```text
B, H*W, C -> B, H*W, buffer_size
```

当前代码中的 RHL 与论文公式对应：

$$
E_t=\operatorname{ReLU}(X_t^{encoder}\Phi_E).
$$

该模块是后续改动最自然的位置，因为它影响解析头输入特征空间，但不破坏 C-RLS 主公式。

### 5.2 `RecursiveLinear`

`RecursiveLinear` 保存两个核心对象：

| 对象 | 形状 | 作用 |
|---|---|---|
| `self.R` | `buffer_size × buffer_size` | 正则化特征自相关矩阵的逆 |
| `self.weight` | `buffer_size × num_classes` | 解析分类头权重 |

`fit()` 首先把像素级特征展平：

```text
X: B, H*W, C -> B*H*W, C
label: B, 1, H, W -> B*H*W
```

再过滤掉 ignore label 255，并构造 one-hot 标签矩阵。若当前类别数超过已有输出列数，则扩展 `self.weight` 的列。

核心更新为：

```python
R_inv = torch.inverse(self.R)
S = R_inv + X.T @ X
S_inv = torch.inverse(S)
self.R = S_inv
self.weight += self.R @ X.T @ (Y - X @ self.weight)
```

这与论文中的 C-RLS 等价，只是代码采用 `R_inv + X.T @ X` 的形式更新，而不是直接使用 Woodbury 展开式。该实现可支持 mini-batch 式拟合，但矩阵求逆规模与 `buffer_size` 强相关。

### 5.3 `AIR` 包装器

`trainer/trainer.py` 中定义的 `AIR` 将 backbone、RHL 与解析头包装为一个模块：

```text
AIR.backbone -> AIR.buffer -> AIR.analytic_linear
```

`feature_expansion()` 会调用 backbone 获得 dense feature，然后 reshape 为像素样本矩阵。这里代码假设输出 feature map 可用 `H*W` 表示，并在 `AnalyticLinear.forward()` 中通过 `int(HW**0.5)` 恢复空间尺寸。因此，输入裁剪策略应尽量保持输出特征图为近似方形，否则可能出现 reshape 风险。

---

## 6. 伪标签实现观察

论文 2D 伪标签依据不确定度：

$$
U_i=1-\sigma(\max_c q_{\theta_{t-1}}(i,c)).
$$

若背景像素处旧模型足够确定，则以旧模型预测替代 background 标签。

代码中的实现更直接：

```python
outputs = self.model_prev(images)
outputs = torch.sigmoid(outputs)
pred_scores, pred_labels = torch.max(outputs, dim=1)
pseudo_labels = torch.where(
    (labels == 0) & (pred_labels > 0) &
    (pred_scores >= self.opts.pseudo_label_confidence),
    pred_labels,
    labels)
```

两点值得注意：

1. 当前 `run.sh` 未传入 `--use_pseudo_label`，所以默认 sequential 复现不会启用伪标签。
2. 论文使用不确定度阈值 $\tau$，代码使用 sigmoid score 阈值 `pseudo_label_confidence`。若 `score=1-U`，二者存在换算关系，但默认值并不直接等价。论文 2D 使用 $\tau=0.4$，代码默认 confidence 为 0.7，后续 disjoint / overlap 实验需要单独说明。

---


## 7. 总结

SegACIL 的代码实现与 CFSSeg 的核心方法基本对应：step0 通过常规 BP 获得分割表征，step1 进入解析重对齐，step2 以后通过 RHL 与 RecursiveLinear 递归更新分割分类头。代码中最重要的研究入口是 `network/Buffer.py` 和 `network/AnalyticLinear.py`，最重要的工程入口是 `trainer/trainer.py` 与 `run.sh`。

当前问题主要是运行脚本与资源配置问题，而非方法公式本身。修复 GPU 环境变量导出、失败即停和 batch 设置后，应优先获得一条 VOC 15-1 sequential 的完整 baseline。随后再进行 RHL 层的最小改动与消融，形成可解释、可对比、可写入论文的实验链条。

---


