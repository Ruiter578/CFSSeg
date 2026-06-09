# SegACIL / CFSSeg 代码报告

> 项目路径：`/TRS-SAS/linwei/SegACIL`  
> 论文：`CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds`  
> 本报告基于当前仓库源码与仓库内论文 PDF。目录树已排除 `.git`；`__pycache__` 保留但不展开；`VOC2012` 只展开到一级目录。

## 1. 项目目录树

```text
SegACIL/
├── (庄)CFSSeg_Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds.pdf
├── .gitignore
├── README.md
├── checkpoints/
│   └── 1128/
│       └── voc/
│           └── 15-1/
│               └── sequential/
│                   ├── step0/
│                   │   └── events.out.tfevents.1777304848.master-192-168-8-48
│                   ├── step1/
│                   │   └── events.out.tfevents.1777304857.master-192-168-8-48
│                   ├── step2/
│                   │   └── events.out.tfevents.1777304860.master-192-168-8-48
│                   └── step3/
│                       └── events.out.tfevents.1777304863.master-192-168-8-48
├── data_root/
│   ├── SegmentationClassAug.zip
│   ├── VOC2012/
│   │   ├── Annotations/
│   │   ├── ImageSets/
│   │   ├── JPEGImages/
│   │   ├── SegmentationClass/
│   │   ├── SegmentationClassAug/
│   │   ├── SegmentationObject/
│   │   └── __MACOSX/
│   └── VOCtrainval_11-May-2012.tar
├── datasets/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── ade.py
│   ├── cityscapes_domain.py
│   ├── data/
│   │   └── voc/
│   │       ├── train_cls.txt
│   │       └── val_cls.txt
│   ├── init_dataset.py
│   ├── utils.py
│   └── voc.py
├── environment.from-history.yml
├── metrics/
│   ├── __init__.py
│   ├── __pycache__/
│   └── stream_metrics.py
├── network/
│   ├── AnalyticLinear.py
│   ├── Buffer.py
│   ├── __init__.py
│   ├── __pycache__/
│   ├── _deeplab.py
│   ├── backbone/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── mobilenetv2.py
│   │   └── resnet.py
│   ├── modeling.py
│   └── utils.py
├── run.sh
├── train.py
├── trainer/
│   ├── __pycache__/
│   └── trainer.py
└── utils/
    ├── __init__.py
    ├── __pycache__/
    ├── ckpt.py
    ├── ext_transforms.py
    ├── logger.py
    ├── loss.py
    ├── misc.py
    ├── parser.py
    ├── scheduler.py
    ├── tasks.py
    └── utils.py
```

## 2. 项目定位

SegACIL 是 CFSSeg 思路在语义分割上的实现：第 0 步先用 SGD 训练一个 DeepLab 编码器和分类头；之后的增量步骤冻结特征提取网络，用解析闭式解更新像素级分类头，从而避免对旧知识权重反复做梯度更新。

当前仓库主要覆盖 2D 图像语义分割：

| 范围 | 当前代码状态 |
|---|---|
| VOC2012 | 完整主流程，`run.sh` 默认使用 |
| ADE20K | 数据集类与任务划分存在 |
| Cityscapes domain | 代码存在，但与 `init_dataset.py` 的统一调用接口不完全匹配 |
| 3D S3DIS / ScanNet / DGCNN | 论文中有，当前仓库未看到对应实现 |

需要特别注意：`utils/parser.py` 中有 `--method` 参数，`run.sh` 传入 `METHOD="acil"`，但当前源码没有使用 `opts.method` 做真正分支。实际方法分支由 `curr_step`、`model`、`setting`、`loss_type`、`use_pseudo_label` 决定。

## 3. CFSSeg 主要方法

CFSSeg 延续了庄辉平副教授团队 ACIL / G-ACIL 一类解析持续学习思想：把增量学习中最容易遗忘的分类器更新改写成带正则的最小二乘问题，用闭式解或递归闭式解直接得到分类头参数。当前代码的 `network/AnalyticLinear.py` 和 `network/Buffer.py` 文件头部也明确引用了 ACIL、GKEAL、DS-AL、G-ACIL 等解析学习工作。

### 3.1 任务背景

类别增量语义分割需要在第 `t` 步加入新类别集合 `S_t`，同时保留旧类别集合 `C_{t-1}` 的分割能力：

$$
C_t = C_{t-1} \cup S_t,\qquad S_i \cap S_j = \varnothing\ (i \ne j)
$$

论文把分割输入统一表示为像素或点云元素：

$$
X \in \mathbb{R}^{N \times C_{in}},\qquad
q_\theta: X \to \mathbb{R}^{N \times |\mathcal{C}|}
$$

输出掩码由逐元素最大类别得到：

$$
\hat{y}_i = \arg\max_{c \in \mathcal{C}} q_\theta(x)[i,c]
$$

### 3.2 稳定性与可塑性

CFSSeg 的设计可以概括为两句话：

| 目标 | 做法 | 代码对应 |
|---|---|---|
| 稳定性 | 增量阶段冻结 encoder/backbone，不再反向传播更新旧特征 | `trainer/trainer.py:224-239`, `trainer/trainer.py:261-269` |
| 可塑性 | 在冻结特征后插入随机高维隐层 RHL，让特征更线性可分 | `network/Buffer.py:37-69`, `trainer/trainer.py:34-39` |

论文中的随机高维隐层 RHL：

$$
E_1 = \operatorname{ReLU}(X^{encoder}_1 \Phi_E)
$$

其中 `Phi_E` 是随机初始化、固定不通过梯度学习的高维映射矩阵。代码中对应 `RandomBuffer`：

```python
# network/Buffer.py:55-69
W = torch.empty((out_features, in_features), **factory_kwargs)
self.register_buffer("weight", W)
self.reset_parameters()
return self.activation(super().forward(X))
```

这里 `register_buffer` 表示它随模型保存和迁移设备，但不是优化器训练参数。

### 3.3 岭回归闭式解

对高维特征矩阵 `E_1` 和 one-hot 标签矩阵 `Y_1^{train}`，分类头 `Phi_1` 通过岭回归得到：

$$
\arg\min_{\Phi_1}
\left(
\left\|Y^{train}_1 - E_1\Phi_1\right\|_F^2
+ \gamma \left\|\Phi_1\right\|_F^2
\right)
$$

闭式解为：

$$
\hat{\Phi}_1 =
\left(E_1^\top E_1 + \gamma I\right)^{-1}
E_1^\top Y^{train}_1
$$

代码没有直接一次性存储全量 `E_1` 再求解，而是用递归形式累积自相关逆矩阵 `R` 和权重 `weight`。

### 3.4 C-RLS 递归闭式更新

论文把第 `t` 步的累计标签和特征写成：

$$
Y^{train}_{1:t} =
\begin{bmatrix}
Y^{train}_{1:t-1} & 0 \\
\bar{Y}^{train}_t & \tilde{Y}^{train}_t
\end{bmatrix},
\qquad
E_{1:t} =
\begin{bmatrix}
E_{1:t-1} \\
E_t
\end{bmatrix}
$$

旧步的解析解：

$$
\hat{\Phi}_{t-1} =
\left(E_{1:t-1}^\top E_{1:t-1} + \gamma I\right)^{-1}
E_{1:t-1}^\top Y^{train}_{1:t-1}
$$

定义倒置自相关矩阵：

$$
\Psi_{t-1} =
\left(E_{1:t-1}^\top E_{1:t-1} + \gamma I\right)^{-1}
$$

C-RLS 只用当前步数据递归更新：

$$
\Psi_t =
\left(\Psi_{t-1}^{-1} + E_t^\top E_t\right)^{-1}
$$

$$
\hat{\Phi}_t =
\left[
\hat{\Phi}_{t-1}
- \Psi_t E_t^\top E_t \hat{\Phi}_{t-1}
+ \Psi_t E_t^\top \bar{Y}^{train}_t
\quad
\Psi_t E_t^\top \tilde{Y}^{train}_t
\right]
$$

代码核心在 `network/AnalyticLinear.py:107-165`：

```python
# network/AnalyticLinear.py:157-165
R_inv = torch.inverse(self.R)
S = R_inv + X.T @ X
S_inv = torch.inverse(S)
self.R = S_inv
self.weight += self.R @ X.T @ (Y - X @ self.weight)
```

变量对应关系：

| 论文符号 | 代码变量 | 说明 |
|---|---|---|
| `E_t` | `X` | 经 backbone + RHL 后的像素特征 |
| `Y_t` | `Y` | 标签 one-hot 矩阵 |
| `Psi_t` | `self.R` | 正则化特征自相关矩阵的逆 |
| `Phi_t` | `self.weight` | 解析分类头权重 |
| `gamma` | `self.gamma` | 岭回归正则项，初始化 `R = I / gamma` |

`RecursiveLinear.fit()` 先把 `B, H*W, C` 展平为像素样本，过滤 `255` ignore label，再 one-hot 成标签矩阵：

```python
# network/AnalyticLinear.py:113-128
B, HW, C = X.shape
X = X.view(B * HW, C)
y = y.view(-1)
mask = y != 255
X = X[mask]
y = y[mask]
Y = F.one_hot(y, num_classes=num_targets).to(self.weight)
```

当新类别出现时，分类头列数扩展：

```python
# network/AnalyticLinear.py:136-143
if num_targets > self.out_features:
    increment_size = num_targets - self.out_features
    tail = torch.randn((self.weight.shape[0], increment_size)).to(self.weight) * epsilon
    self.weight = torch.cat((self.weight, tail), dim=1)
elif num_targets < self.out_features:
    tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
    Y = torch.cat((Y, tail), dim=1)
```

这对应论文中 `Phi_t` 随类别增量扩展列维度的过程。

### 3.5 伪标签处理语义漂移

论文指出，在 disjoint 和 overlapped setting 中，旧类像素会被标成背景，导致 semantic drift。2D 伪标签策略可以写成：

$$
U_i = 1 - \sigma\left(\max_c q_{\theta_{t-1}}(i,c)\right)
$$

$$
\tilde{y}^i_t =
\begin{cases}
y^i_t, & y^i_t \in S_t \\
y^i_t, & (y^i_t = c_b) \land (U_i > \tau) \\
\hat{y}^i_{t-1}, & (y^i_t = c_b) \land (U_i \le \tau)
\end{cases}
$$

代码对应 `trainer/trainer.py:403-423`。它直接用旧模型输出的最大 sigmoid 分数做置信度判断：

```python
# trainer/trainer.py:409-422
outputs = self.model_prev(images)
outputs = torch.sigmoid(outputs)
pred_scores, pred_labels = torch.max(outputs, dim=1)
pseudo_labels = torch.where(
    (labels == 0) & (pred_labels > 0) &
    (pred_scores >= self.opts.pseudo_label_confidence),
    pred_labels,
    labels)
```

注意两点：

1. `run.sh` 没有传 `--use_pseudo_label`，所以默认实验脚本不启用伪标签。
2. 论文 2D 设置里 `tau=0.4`；代码默认 `pseudo_label_confidence=0.7`，而且判断方式是 `score >= confidence`，不是直接用不确定性 `U_i <= tau`。二者等价关系取决于 `score = 1 - U_i`，但阈值数值并不相同。

## 4. 核心模块与作用

| 模块 | 作用 | 关键位置 |
|---|---|---|
| `train.py` | 项目入口，组装类别数、设置随机种子、启动 Trainer | `train.py:14-39` |
| `run.sh` | VOC 15-1 顺序增量实验脚本 | `run.sh:4-56` |
| `utils/parser.py` | 命令行参数和默认配置 | `utils/parser.py:5-126` |
| `utils/tasks.py` | VOC/ADE/Cityscapes 的增量类别划分与样本列表过滤 | `utils/tasks.py:1-312` |
| `datasets/init_dataset.py` | 数据增强与 DataLoader 初始化 | `datasets/init_dataset.py:11-78` |
| `datasets/voc.py` | VOC 图像、mask、增量标签重映射 | `datasets/voc.py:34-123` |
| `network/modeling.py` | DeepLab 模型工厂，选择 ResNet/MobileNet backbone | `network/modeling.py:8-207` |
| `network/_deeplab.py` | DeepLabV3/V3+、ASPP、分割头 | `network/_deeplab.py:76-120`, `network/_deeplab.py:256-285` |
| `network/Buffer.py` | RHL 随机高维映射层 | `network/Buffer.py:37-69` |
| `network/AnalyticLinear.py` | 解析线性层、递归闭式更新 | `network/AnalyticLinear.py:29-165` |
| `trainer/trainer.py` | 训练、增量更新、评估、伪标签 | `trainer/trainer.py:18-423` |
| `utils/loss.py` | CE、Focal、BCE-with-ignore | `utils/loss.py:5-68` |
| `utils/scheduler.py` | Poly / Step / Warmup Poly 学习率 | `utils/scheduler.py:7-98` |
| `metrics/stream_metrics.py` | 混淆矩阵、mIoU、mAcc、整体准确率 | `metrics/stream_metrics.py:53-132` |
| `utils/ckpt.py` | 保存和加载模型对象、state_dict、optimizer | `utils/ckpt.py:4-27` |

## 5. 数据从输入到输出的 Pipeline

### 5.1 数据准备

VOC 数据默认路径来自 `run.sh`：

```bash
DATA_ROOT="/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
```

`VOCSegmentation` 读取：

| 数据 | 代码 |
|---|---|
| 图像 | `data_root/VOC2012/JPEGImages/*.jpg` |
| mask | `data_root/VOC2012/SegmentationClassAug/*.png` |
| train/val 样本列表 | `datasets/data/voc/train_cls.txt`, `datasets/data/voc/val_cls.txt` |
| test 样本列表 | `data_root/VOC2012/ImageSets/Segmentation/val.txt` |

### 5.2 增量类别组织

入口 `train.py:16-20` 先读取当前 step 前所有类别：

```python
opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step))
                    for step in range(opts.curr_step+1)]
opts.target_cls = [get_tasks(opts.dataset, opts.task, step)
                   for step in range(opts.curr_step+1)]
opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]
```

以 VOC `15-1` 为例：

| step | 原始类别 | `opts.num_classes` 含义 |
|---|---|---|
| 0 | `[0..15]` | `[1, 15]`，背景 1 类 + 初始前景 15 类 |
| 1 | `[16]` | `[1, 15, 1]` |
| 2 | `[17]` | `[1, 15, 1, 1]` |
| 5 | `[20]` | `[1, 15, 1, 1, 1, 1, 1]` |

最终分类通道数由 `sum(opts.num_classes)` 给出。

### 5.3 样本筛选与标签重映射

`get_dataset_list()` 按 setting 过滤样本：

| setting | 样本选择逻辑 | 标签逻辑 |
|---|---|---|
| `overlap` | 只要图像中出现当前新类，就进入当前 step 数据 | 非目标类在 `gt_label_mapping()` 中映射为背景 |
| `disjoint` | 图像必须包含当前新类，且不含未来类 | 非目标类映射为背景 |
| `sequential` | 代码里样本筛选走 `else` 分支，标签映射不改旧类 | `gt_label_mapping()` 中 `pass`，保留旧类和新类标签 |

`VOCSegmentation.gt_label_mapping()` 先根据 setting 处理背景，再用 `ordering_map` 把原始类别 id 重排到连续增量 id：

```python
# datasets/voc.py:109-118
if self.image_set != 'test':
    if self.setting == 'sequential':
        pass
    else:
        gt = np.where(np.isin(gt, self.target_cls), gt, 0)
gt = self.ordering_map[gt]
```

### 5.4 第 0 步训练流

第 0 步是常规梯度训练：

```text
image, mask
  -> transforms: random scale / crop / flip / normalize
  -> DeepLab backbone
  -> ASPP + head_pre + linear pixel classifier
  -> logits
  -> interpolate to mask size
  -> BCE/CE/Focal loss
  -> SGD backward + scheduler
  -> validation + checkpoint
```

关键代码：

| 环节 | 位置 |
|---|---|
| 模型构造 | `trainer/trainer.py:110-130` |
| 优化器 | `trainer/trainer.py:158-166` |
| forward/loss/backward | `trainer/trainer.py:171-188` |
| 验证与保存 | `trainer/trainer.py:203-219` |

### 5.5 增量步骤训练流

第 `curr_step >= 1` 步不做反向传播，而是闭式更新：

```text
load previous checkpoint
  -> remove original classifier head by Identity()
  -> backbone extracts 256-channel dense feature map
  -> flatten each pixel to a sample
  -> RandomBuffer maps 256 -> buffer_size
  -> downsample mask to feature map size
  -> one-hot labels
  -> RecursiveLinear.fit() accumulates R and weight
  -> RecursiveLinear.update() checks numerical stability
  -> save final.pth
  -> evaluate
```

第 1 步有一个特殊 realign 过程：

1. 先把 `opts.curr_step` 临时设回 0，加载 step0 数据。
2. 用 step0 数据重新拟合一个 AIR 解析头并保存到 `step0/final.pth`。
3. 再用原本 step1 的 `train_loader` 拟合新类别，保存到 `step1/final.pth`。

关键代码：

| 分支 | 位置 | 含义 |
|---|---|---|
| `curr_step == 0` | `trainer/trainer.py:171-219` | 常规 DeepLab 训练 |
| `curr_step == 1` | `trainer/trainer.py:220-259` | 构造 AIR，先 realign step0，再 fit step1 |
| `curr_step > 1` | `trainer/trainer.py:260-272` | 加载上一步 AIR，继续递归 fit 当前数据 |

### 5.6 推理与评估

常规 DeepLab 评估走 `validate()`，输出是 `B,C,H,W`：

```python
# trainer/trainer.py:387-398
outputs, _ = self.model(images)
outputs = torch.sigmoid(outputs)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
preds = outputs.detach().max(dim=1)[1].cpu().numpy()
```

AIR 评估走 `do_evaluate()`，`AnalyticLinear.forward()` 输出 `B,H,W,C`，所以先 `permute(0,3,1,2)`：

```python
# trainer/trainer.py:287-296
outputs = self.model(images)
outputs = torch.sigmoid(outputs)
outputs = outputs.permute(0,3,1,2)
outputs = F.interpolate(outputs, labels.shape[-2:], mode='bilinear')
preds = outputs.detach().max(dim=1)[1].cpu().numpy()
```

指标来自混淆矩阵：

$$
\operatorname{IoU}_c =
\frac{TP_c}{TP_c + FP_c + FN_c}
$$

代码位置：`metrics/stream_metrics.py:93-128`。

## 6. 完整训练过程

### 6.1 项目入口

推荐从 `run.sh` 启动 VOC 15-1：

```bash
bash run.sh
```

核心命令最终进入：

```bash
python train.py \
  --data_root /TRS-SAS/linwei/SegACIL/data_root/VOC2012 \
  --model deeplabv3_resnet101 \
  --lr 0.01 \
  --batch_size <32 or 64> \
  --loss_type bce_loss \
  --dataset voc \
  --task 15-1 \
  --lr_policy poly \
  --curr_step <0..5> \
  --subpath 1128 \
  --method acil \
  --setting sequential \
  --pretrained_backbone \
  --crop_val \
  --train_epoch 50 \
  --gamma 1 \
  --buffer 8196 \
  --output_stride 8
```

### 6.2 当前脚本的增量步骤

`run.sh:24-27` 设置：

```bash
START_STEP=0
END_STEP=5
STEP_INCREMENT=1
```

因此 VOC `15-1` 会执行 6 个步骤：

| step | 学习类别 |
|---|---|
| 0 | 背景 + `1..15` |
| 1 | 新增 `16` |
| 2 | 新增 `17` |
| 3 | 新增 `18` |
| 4 | 新增 `19` |
| 5 | 新增 `20` |

checkpoint 路径格式：

```text
checkpoints/{subpath}/{dataset}/{task}/{setting}/step{curr_step}/
```

当前脚本对应：

```text
checkpoints/1128/voc/15-1/sequential/step0/
checkpoints/1128/voc/15-1/sequential/step1/
...
```

### 6.3 方法分支

| 参数/条件 | 分支 |
|---|---|
| `curr_step == 0` | SGD 训练 DeepLab |
| `curr_step == 1` | 从 step0 checkpoint 构造 AIR，执行 base realign + 新类 fit |
| `curr_step > 1` | 加载上一步 AIR，递归 fit 当前 step |
| `model=deeplabv3_resnet101` | ResNet-101 + DeepLabV3 |
| `model=deeplabv3bga_resnet101` | 可选 BGA head，但 `run.sh` 未使用 |
| `loss_type=bce_loss` | 多通道 BCE-with-ignore |
| `setting=sequential/disjoint/overlap` | 控制样本过滤、标签 remap、伪标签是否有意义 |
| `--use_pseudo_label` | 仅在 `curr_step > 1` 且非 sequential 时尝试启用 |
| `--method acil` | 当前源码未使用 |

### 6.4 相关超参数

| 超参数 | `run.sh` 当前值 | 默认值/代码位置 | 说明 |
|---|---:|---:|---|
| `dataset` | `voc` | `voc` | 数据集 |
| `task` | `15-1` | `15-1` | 类别增量划分 |
| `setting` | `sequential` | `overlap` | 分割增量 setting |
| `model` | `deeplabv3_resnet101` | 必填 | 模型结构 |
| `output_stride` | `8` | `16` | DeepLab 输出步幅 |
| `pretrained_backbone` | true | false | ImageNet 预训练 |
| `train_epoch` | `50` | `50` | step0 训练 epoch |
| `batch_size` | step0 为 `32`，增量为 `64` | `32` | 训练 batch |
| `val_batch_size` | 未传 | `1` | 验证 batch |
| `crop_size` | 未传 | `513` | crop 尺寸 |
| `lr` | `0.01` | `0.01` | 基础学习率 |
| `lr_policy` | `poly` | `warm_poly` | 学习率策略 |
| `weight_decay` | 未传 | `1e-4` | SGD 正则 |
| `loss_type` | `bce_loss` | `bce_loss` | 损失函数 |
| `buffer` | `8196` | `16384` | RHL 高维维度 `d_E` |
| `gamma` | `1` | `10.0` | 岭回归正则 |
| `pseudo_label_confidence` | 未传 | `0.7` | 伪标签置信度 |
| `random_seed` | 未传 | `1` | 随机种子 |

论文 2D 实验设置为 `d_E=8192`、`gamma=1`、`tau=0.4`。当前脚本 `BUFFER=8196` 与论文 `8192` 有 4 维差异，若要严格复现实验，建议确认是否是手误。

### 6.5 可训练参数

第 0 步：

| 参数组 | 代码 | 学习率 |
|---|---|---:|
| backbone | `self.model.backbone.parameters()` | `0.001` |
| classifier | `self.model.classifier.parameters()` | `0.01` |

代码位置：`trainer/trainer.py:158-166`。如果传 `--bn_freeze`，BatchNorm 会设为 eval 且不更新 affine 参数，位置是 `network/utils.py:20-31`。

增量步骤：

| 组件 | 是否 optimizer 训练 | 更新方式 |
|---|---|---|
| frozen backbone / encoder | 否 | 只做特征提取 |
| `RandomBuffer.weight` | 否 | 初始化后固定，`register_buffer` |
| `RecursiveLinear.R` | 否 | 解析递归更新 |
| `RecursiveLinear.weight` | 否 | 闭式解赋值更新 |

也就是说，增量阶段没有 `loss.backward()`、没有 optimizer，也没有常规意义上的 trainable parameters；模型参数通过矩阵运算被直接写入 buffer。

## 7. 代码实现观察与复现注意点

1. `--method` 当前未接入实际逻辑。若要比较 `ft/acil/...`，需要在 `Trainer.train()` 或入口处补充分支。
2. `run.sh` 默认 `SETTING="sequential"`，且没有启用 `--use_pseudo_label`；这与论文中 disjoint/overlapped 需要伪标签缓解 semantic drift 的实验不完全一致。
3. `RecursiveLinear.forward()` 使用 `int(HW**0.5)` 恢复特征图高宽，隐含假设输出特征图是正方形。当前 `crop_size=513` 基本满足，但非方形输入可能出错或 reshape 不正确。
4. `CityscapesSegmentationIncrementalDomain` 的构造函数不是 `opts=...` 风格，和 `init_dataset.py` 当前统一调用不匹配，可能不能直接跑。
5. `utils/tasks.py` 中 `get_dataset_list()` 的 lambda 形参 `c` 与外部变量 `classes` 混用，可运行但可读性弱，后续维护建议重写为显式函数。
6. checkpoint 会保存完整 `model_architecture` 对象，跨代码版本加载时更脆弱；只保存结构配置 + `state_dict` 会更稳。

## 8. 与论文方法的对应程度

| 论文组件 | 当前代码是否实现 | 说明 |
|---|---|---|
| 第一步 SGD 训练 encoder | 是 | DeepLabV3/ResNet-101 |
| 冻结 encoder | 是 | 增量阶段加载旧模型并移除 head |
| RHL 随机高维映射 | 是 | `RandomBuffer` |
| 岭回归闭式分类头 | 是 | `RecursiveLinear` |
| C-RLS 递归更新 | 是 | `self.R` 和 `self.weight` |
| 2D 伪标签 | 部分 | 有代码，但默认脚本未启用，阈值与论文不同 |
| 3D 伪标签 / DGCNN | 否 | 当前仓库未看到 3D 实现 |
| VOC sequential/disjoint/overlap | 是 | 由 `setting` 控制 |
| ADE | 部分 | 数据类和任务存在，但未提供完整数据树 |

## 9. 你的两个附加问题

### 9.1 Markdown 公式能否正确可视化渲染，只看打开软件是否支持？

基本是的。Markdown 本身没有统一强制的数学公式标准；公式是否能渲染，主要取决于打开它的软件或平台是否支持 LaTeX 数学扩展，以及支持哪种语法。

为了提高兼容性，报告里的公式采用常见写法：

| 场景 | 写法 |
|---|---|
| 行内公式 | `$E_t^\top E_t$` |
| 独立块公式 | `$$ ... $$` |
| 矩阵/分段函数 | LaTeX `bmatrix`、`cases` |

常见支持情况：

| 软件/平台 | 支持情况 |
|---|---|
| GitHub Markdown | 支持大多数 `$...$` 与 `$$...$$` 数学公式 |
| VS Code 内置 Markdown Preview | 取决于版本和扩展，通常建议装数学预览扩展 |
| Typora / Obsidian | 一般支持较好 |
| 普通文本编辑器 | 不渲染，只显示源码 |
| Word | 不直接按 Markdown 渲染，需要 Pandoc 等转换 |

如果公式不渲染，不代表公式写错，通常只是查看器没有启用 MathJax/KaTeX/LaTeX math 支持。

### 9.2 我能否调用容器中安装的 VSCode 插件、PDF 插件、docx editor？

当前我可以通过命令行和 Python 访问文件，但不能直接调用 VSCode 插件的交互式 UI 或扩展 API。这个容器里能看到 VSCode Server 扩展目录，包含 `vscode-office`、`vscode-docx-editor` 等扩展；但这些扩展主要服务 VSCode 前端界面，不等同于我在终端里可调用的命令行工具。

具体能力边界：

| 功能 | 当前可做 | 说明 |
|---|---|---|
| 查看 PDF 内容 | 可以 | 本报告使用 Python `pypdf` 提取了论文文本 |
| 调用 VSCode PDF 插件直接看 PDF | 不可以直接调用 | 插件是 VSCode UI 能力，不是稳定 CLI |
| 生成 Markdown 报告 | 可以 | 当前报告即 Markdown |
| 直接用 VSCode docx editor 编辑 Word | 不可以直接调用 | 需要 VSCode 前端打开插件 |
| 生成 `.docx` | 取决于 CLI 工具 | 当前环境未检测到 `pandoc`、`libreoffice`、`python-docx` |

如果要稳定实现 “Markdown/PDF/Word 自动生成和编辑”，建议安装命令行工具，而不是依赖 VSCode 插件：

| 目标 | 推荐工具 |
|---|---|
| Markdown 转 Word | `pandoc` |
| Word 后处理/批量转换 | `libreoffice --headless` |
| Python 生成 docx | `python-docx` |
| PDF 文本提取 | `pypdf`、`PyMuPDF`、`poppler-utils` 的 `pdftotext` |
| Markdown 数学预览 | VSCode `Markdown Preview Enhanced` 或 KaTeX/MathJax 支持插件 |
| PDF 预览 | VSCode `vscode-pdf` |
| Office 预览 | VSCode `vscode-office` |
| docx 简单编辑 | VSCode `vscode-docx-editor` |

最稳的工作流是：先维护 `segacil_code_report.md`，再用 `pandoc` 转成 `.docx`；需要 PDF 时再由 `pandoc`/LaTeX 或浏览器打印生成。这样公式、目录、表格更可控，也更适合自动化。
