# CFSSeg 与 SegACIL 实验配置对照报告

> 生成时间：2026-06-16  
> 关注范围：两篇论文的实验部分，以及当前 `/TRS-SAS/linwei/SegACIL` 代码库中的训练设置与参数实现。  
> 对照论文：  
> - `/TRS-SAS/linwei/OFQ/All_papers/庄-核心-SegACIL.pdf`  
> - `/TRS-SAS/linwei/OFQ/All_papers/庄-核心-CFSSeg_Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds.pdf`

## 0. 核心结论

1. **CFSSeg 论文的 2D 主实验结果与 SegACIL 论文的 VOC2012 结果表基本完全一致**：sequential、disjoint、overlapped 三类设置下的 `15-1`、`15-5`、`10-1` 结果数值相同或沿用同一组结果。CFSSeg 后文扩展了 3D 点云部分，但当前 `SegACIL` 代码库没有完整的 S3DIS / ScanNet / DGCNN 3D 训练分支。

2. **当前代码库与 CFSSeg 论文 2D 配置“部分吻合，但不是逐字完全吻合”**。吻合项包括 DeepLabv3 + ResNet-101、ImageNet 预训练、VOC2012、50 epochs、SGD、momentum 0.9、weight decay `1e-4`、polynomial LR、BCE、`gamma=1`、RHL + 解析线性分类器。主要不一致点是：
   - 论文写 `d_E=8192`，当前 `run.sh` / README 写 `BUFFER=8196`。
   - CFSSeg 论文 2D 初始训练 batch size 写 `32`，当前 README / `run.sh` 也是 step0 `32`，但 SegACIL 论文写 batch size `16`。
   - CFSSeg 论文写 `tau=0.4` 是“不确定性阈值”；当前代码实现的是 `pseudo_label_confidence` 置信度阈值，默认 `0.7`。若按 CFSSeg 公式 `U=1-confidence` 理解，论文 `tau=0.4` 对应代码置信度阈值约 `0.6`，不是直接填 `0.4`。
   - 当前 `run.sh` 默认 `SETTING="sequential"` 且没有传 `--use_pseudo_label`，所以当前脚本路径下 `tau` 实际不参与训练。
   - step0 optimizer 在代码里额外启用了 `nesterov=True`，论文实验描述没有明确写这一点。
   - 代码 step0 对 backbone 和 classifier 使用了不同学习率：backbone `0.001`，classifier `0.01`；论文只概括写学习率 `1e-2`。

3. **如果调 momentum、weight decay、step0 LR schedule、BCE/CE、batch size 等会改变 step0 表征的参数，通常必须重新训练 step0 才能验证真实性能**。因为后续 AIR/RHL/RecursiveLinear 都依赖 step0 学到的 encoder/backbone 特征。

4. **如果只调 `d_E`、`gamma`、`tau` 这类闭式增量阶段参数，原则上可以复用同一个 step0 checkpoint，从 step1 或后续 step 重新跑，不必完整重训 step0**。但这仍然需要运行增量阶段评估，不能纯靠静态分析得出真实性能。

5. **Codex / agent 不能“无需训练”完成真实性能意义上的超参数搜索**。它可以做静态代码审计、配置生成、实验矩阵编排、日志解析、已有 checkpoint 的增量阶段扫参、早停策略和结果汇总；但不能在完全不训练/不评估的情况下可靠判断哪个超参组合 mIoU 更高。

## 1. 论文实验配置摘要

### 1.1 SegACIL 论文

SegACIL 论文实验部分集中在 Pascal VOC2012 的 2D 类增量语义分割。论文页 4-5 给出的关键信息如下：

| 项目 | SegACIL 论文实验描述 |
|---|---|
| 数据集 | Pascal VOC2012，21 类，10,582 train，1,449 val |
| 设置 | sequential: `15-1`, `15-5`；disjoint/overlap: `15-1`, `10-1` |
| 指标 | mIoU，分别报告初始类、增量类、全部类 |
| 模型 | DeepLabv3 decoder + ResNet101 encoder，ImageNet-1K 预训练 |
| step0 初始训练 | 50 epochs，batch size 16 |
| optimizer | SGD，lr `1e-2`，momentum `0.9`，weight decay `1e-4` |
| LR schedule | polynomial learning rate schedule |
| loss | BCE loss |
| 增量阶段 | 冻结 encoder，插入 RHL，闭式/递归解析更新分类头 |
| buffer / `d_E` | buffer size `8192` |
| `gamma` | `1` |
| 伪标签阈值 | disjoint 和 overlap 中使用阈值 `0.6` |
| 图像尺寸/增强 | resize/crop 到 `513 x 513`，random scaling / flipping / cropping |

SegACIL 的结果表中，sequential `15-1` 和 `15-5` 都报告 `0-15=78.1`, `16-20=42.0`, `all=70.0`，论文用这点强调闭式解对类别顺序的稳定性。

### 1.2 CFSSeg 论文

CFSSeg 论文实验部分在 SegACIL 2D VOC 基础上增加了 3D 点云 S3DIS / ScanNet。页 6-9 给出的 2D 配置如下：

| 项目 | CFSSeg 论文 2D 实验描述 |
|---|---|
| 数据集 | Pascal VOC2012，21 类，10,582 train，1,449 val |
| 设置 | sequential、disjoint、overlapped |
| 模型 | DeepLabv3 + ResNet-101，ImageNet-1K 预训练 |
| step0 初始训练 | 50 epochs，batch size 32 |
| optimizer | SGD，lr `1e-2`，momentum `0.9`，weight decay `1e-4` |
| LR schedule | polynomial learning rate scheduler |
| loss | BCE |
| 增量阶段 | 冻结 encoder，插入 RHL |
| 2D `d_E` | `8192` |
| 2D `gamma` | `1` |
| 2D `tau` | `0.4` |

注意：CFSSeg 页 6 的伪标签公式把 `tau` 定义在不确定性 `U_i = 1 - sigmoid(max logit)` 上，满足 `U_i <= tau` 时采用旧模型伪标签。因此 `tau=0.4` 等价于 `sigmoid(max logit) >= 0.6`。这和 SegACIL 论文直接用置信度阈值 `0.6` 在数值意义上可以对应起来。

CFSSeg 的 2D 主结果表与 SegACIL 的 VOC 表一致：

| 设置 | 论文结果摘要 |
|---|---|
| sequential `15-1` | `0-15=78.1`, `16-20=42.0`, `all=70.0` |
| sequential `15-5` | `0-15=78.1`, `16-20=42.0`, `all=70.0` |
| disjoint `15-1` | `0-15=77.66`, `16-20=40.33`, `all=68.77` |
| overlap `15-1` | `0-15=79.16`, `16-20=38.00`, `all=69.36` |
| overlap `10-1` | `0-10=75.02`, `11-20=41.20`, `all=58.91` |

## 2. 代码库逐项验证

### 2.1 入口脚本与当前工作树状态

当前 `run.sh` 的实际配置如下：

| 项目 | 当前代码位置 | 当前值 | 与 CFSSeg 论文关系 |
|---|---|---|---|
| GPU | `run.sh:4` | `CUDA_VISIBLE_DEVICES=0` | 运行环境项，论文未限定 |
| 数据路径 | `run.sh:8` | `/root/2TStorage/lyc/SegACIL/data_root/VOC2012` | 指向 VOC2012，但当前检查该路径不存在；实际运行时需修正 |
| 模型 | `run.sh:9` | `deeplabv3_resnet101` | 吻合 |
| LR | `run.sh:10` | `0.01` | 概括上吻合，但 optimizer 内部 backbone/classifier 分组 LR 不同 |
| loss | `run.sh:11` | `bce_loss` | 吻合 |
| dataset | `run.sh:12` | `voc` | 吻合 |
| task | `run.sh:13` | `15-5` | 属于论文 sequential 设置之一；若复现 `15-1` 需改 |
| LR policy | `run.sh:14` | `poly` | 吻合 |
| setting | `run.sh:18` | `sequential` | 吻合 sequential 表 |
| epochs | `run.sh:19` | `50` | 吻合 |
| ImageNet 预训练 | `run.sh:20` | `--pretrained_backbone` | 吻合 |
| buffer / `d_E` | `run.sh:21` | `8196` | 与论文 `8192` 不吻合 |
| gamma | `run.sh:23` | 默认 `1` | 吻合 |
| RHL norm | `run.sh:24-25` | 默认 `none` | baseline 吻合；其他值属于本地扩展 |
| step0 batch | `run.sh:33-34` | step0 `32`，step>0 `64` | step0 与 CFSSeg 吻合；与 SegACIL 论文 `16` 不吻合 |
| 当前起止 step | `run.sh:38-39` | `START_STEP=1`, `END_STEP=1` | 当前脚本只跑 step1，不是完整论文流程 |
| pseudo label | `run.sh:64-87` | 未传 `--use_pseudo_label` | 当前 sequential 不用；disjoint/overlap 复现需显式打开 |

README 示例与当前 `run.sh` 基本一致：`README.md:53-65` 写 `deeplabv3_resnet101`、`LR=0.01`、`bce_loss`、`TASK="15-1"`、`LR_POLICY="poly"`、`TRAIN_EPOCH=50`、`BUFFER=8196`；`README.md:68-69` 写 step>0 batch `64`、step0 batch `32`；`README.md:101-103` 写 `--gamma 1`、`--buffer $BUFFER`。

当前工作树需要特别注意：`run.sh` 现在是 `TASK="15-5"` 且 `START_STEP=1 END_STEP=1`，并不是从 step0 到最终 step 的完整复现实验脚本。如果目标是复现论文 `15-1` 全流程，需要改回对应 task 和 step 范围。

### 2.2 数据集、任务划分与评价指标

| 论文要求 | 代码证据 | 判断 |
|---|---|---|
| VOC2012 | `datasets/init_dataset.py:38-43` 根据 `opts.dataset == 'voc'` 选择 `VOCSegmentation`；`datasets/voc.py:43-58` 检查 VOC 根目录和 `SegmentationClassAug` | 实现存在 |
| `15-1` / `15-5` / `10-1` | `utils/tasks.py:9-21` 定义 VOC `15-5` 和 `15-1`；`utils/tasks.py:22-35` 定义 `10-1` | 吻合论文协议 |
| sequential / disjoint / overlap | `utils/tasks.py:283-311` 用 `setting` 控制样本过滤；`datasets/voc.py:109-117` 在非 sequential 下把非当前目标类映射为 background | 实现了三种数据设置 |
| 513 crop | `utils/parser.py:28` 默认 `crop_size=513`；`datasets/init_dataset.py:14-20` random scale/crop/flip；`datasets/init_dataset.py:23-30` val resize/center crop | 吻合 |
| mIoU | `metrics/stream_metrics.py:101-128` 计算 Overall Acc、Mean Acc、FreqW Acc、Mean IoU、Class IoU | 吻合 |
| 初始类/增量类/全部类 | `trainer/trainer.py:359-371` 额外写 `0 to ... mIoU` 与 `... to ... mIoU`；`metrics/stream_metrics.py:115-128` 的 `Mean IoU` 是全类平均 | 吻合 |

当前本地已有 `15-1 sequential` 完整结果：

```text
checkpoints/1128/voc/15-1/sequential/step5/test_results_20260509_083832.json
Mean IoU: 0.6950792628591231
0 to 15 mIoU: 0.7825233890631731
16 to 20 mIoU: 0.4152580590061633
```

这与论文 sequential `15-1` 的 `all=70.0`、`0-15=78.1`、`16-20=42.0` 接近，属于同一量级结果。这里 `Mean IoU` 对应论文表里的 `all`。

### 2.3 模型结构与 step0 SGD

| 论文要求 | 代码证据 | 判断 |
|---|---|---|
| DeepLabv3 | `run.sh:9` 使用 `deeplabv3_resnet101`；`network/modeling.py:20-24` 注册 DeepLabv3/DeepLabv3+；`network/modeling.py:137-147` 构建 `deeplabv3_resnet101` | 吻合 |
| ResNet-101 | `network/modeling.py:137-147` 中 `backbone='resnet101'` | 吻合 |
| ImageNet-1K 预训练 | `run.sh:20` 传 `--pretrained_backbone`；`network/modeling.py:45-48` 把 `pretrained_backbone` 传入 ResNet 构造 | 吻合 |
| step0 常规训练 | `trainer/trainer.py:217-265` 中 `curr_step == 0` 分支执行 `loss.backward()`、`optimizer.step()`、`scheduler.step()` | 吻合 |
| 冻结 encoder 后闭式更新 | `trainer/trainer.py:281-293` 构造 AIR；`trainer/trainer.py:294-312` 调 `fit()` 和 `update()`；`network/AnalyticLinear.py:155-165` 递归更新 `R` 和 `weight` | 吻合 |

代码中 step0 还有一个论文未显式强调的实现细节：`trainer/trainer.py:174` 设置 backbone BatchNorm momentum 为 `0.01`。这不是 SGD optimizer 的 momentum，而是 BN running statistics 更新系数，不能和论文中的 SGD momentum `0.9` 混为一谈。

### 2.4 optimizer、LR、loss、batch size

| 参数 | 代码证据 | 论文配置 | 当前代码判断 |
|---|---|---|---|
| SGD | `trainer/trainer.py:204-212` | SGD | 吻合 |
| momentum | `trainer/trainer.py:207-211` | `0.9` | 吻合 |
| Nesterov | `trainer/trainer.py:211` | 未明确写 | 代码额外启用 |
| weight decay | `utils/parser.py:35` 默认 `1e-4`；`trainer/trainer.py:210` 传给 SGD | `1e-4` | 吻合 |
| LR | `run.sh:10` 为 `0.01`；但 `trainer/trainer.py:205-206` 分组为 backbone `0.001`、classifier `0.01` | 论文概括写 `1e-2` | classifier 吻合，backbone 实际低 10 倍 |
| poly LR | `run.sh:14` 传 `poly`；`utils/scheduler.py:90-93` 构造 `PolyLR(power=0.9)`；`utils/scheduler.py:14-16` 公式为 `base_lr * (1 - iter/max_iter)^0.9` | polynomial scheduler | 吻合 |
| BCE | `run.sh:11` 为 `bce_loss`；`utils/loss.py:24-56` 实现 one-hot 后 BCEWithLogits；`utils/loss.py:65-66` 选择该 loss | BCE | 吻合 |
| batch size | `run.sh:33-34` step0 `32`、step>0 `64` | CFSSeg step0 `32`；SegACIL step0 `16` | 与 CFSSeg 吻合，与 SegACIL 论文文本不吻合 |

### 2.5 `d_E`、`gamma`、`tau`

| 参数 | 论文含义 | 代码位置 | 当前判断 |
|---|---|---|---|
| `d_E` / buffer size | RHL 输出维度，高维随机特征空间维度 | `run.sh:21` 为 `8196`；`trainer/trainer.py:281-285` 传入 AIR；`network/Buffer.py:39-68` 中 `RandomBuffer(out_features)` 注册固定随机权重 | 论文写 `8192`，代码是 `8196`，不严格吻合 |
| `gamma` | 岭回归正则项，稳定逆矩阵并控制权重范数 | `run.sh:23` 默认 `1`；`trainer/trainer.py:284-285` 传入；`network/AnalyticLinear.py:102-105` 初始化 `R=I/gamma`；`network/AnalyticLinear.py:155-165` 递归更新 | 吻合 |
| `tau` | 伪标签阈值，用于 disjoint/overlap 的 background 语义漂移修正 | `utils/parser.py:66-67` 默认 `use_pseudo_label=False`、`pseudo_label_confidence=0.7`；`trainer/trainer.py:473-476` 用 `pred_scores >= pseudo_label_confidence` 采纳伪标签 | 当前默认不吻合 CFSSeg；若按 CFSSeg 不确定性 `tau=0.4`，代码应使用置信度阈值约 `0.6` |

`tau` 是最容易误读的参数。CFSSeg 论文定义：

```text
U_i = 1 - sigmoid(max logit)
若 U_i <= tau，则采用旧模型伪标签
```

因此 CFSSeg 的 `tau=0.4` 等价于 `sigmoid(max logit) >= 0.6`。当前代码没有显式计算 `U_i`，而是在 `trainer/trainer.py:465-476` 中先对输出做 sigmoid/softmax，再直接比较 `pred_scores >= pseudo_label_confidence`。所以代码里的 `pseudo_label_confidence` 是“置信度阈值”，不是论文 CFSSeg 里的“不确定性阈值”。

如果要按 CFSSeg 论文 2D 伪标签设置运行 disjoint/overlap，更合理的命令行是：

```bash
--use_pseudo_label --pseudo_label_confidence 0.6
```

而不是：

```bash
--pseudo_label_confidence 0.4
```

但当前 `run.sh` 默认 `SETTING="sequential"`，且不传 `--use_pseudo_label`，所以 `tau` 对当前 sequential 脚本没有作用。

## 3. step0 中各参数的作用、必要性与取值原因

### 3.1 SGD momentum = 0.9

**代码位置**：`trainer/trainer.py:207-211`

SGD momentum 的作用是把前几次梯度方向以指数滑动的形式累积到当前更新中。语义分割 step0 是一个正常的 DeepLabv3 监督训练过程，输入是 513x513 的 dense prediction 任务，梯度噪声较大；momentum 可以让更新方向更平滑，减少 batch 间噪声造成的震荡。

为什么常用 `0.9`：

- `0.9` 是 SGD 训练 CNN/ResNet/DeepLab 类模型的经典经验值。
- 太低时接近普通 SGD，收敛慢、受 batch 噪声影响更大。
- 太高时历史梯度惯性太强，可能在后期或 LR 衰减阶段不够灵活。
- 代码还启用了 `nesterov=True`，会使用 Nesterov momentum，通常能让带 momentum 的 SGD 更新更“前瞻”一点；但这点论文没有明确写。

对 CFSSeg/SegACIL 来说，step0 的 backbone 表征质量决定后续 RHL + 闭式分类器的上限。因此 momentum 的主要作用不是直接解决 continual learning，而是让初始 DeepLab 表征训练得更稳。

### 3.2 weight decay = 1e-4

**代码位置**：`utils/parser.py:35`，`trainer/trainer.py:207-211`

weight decay 是 L2 参数正则，作用是抑制 step0 中 backbone 和 classifier 权重过大，降低过拟合风险。VOC2012 train augmented 虽然有一万多张，但相对 DeepLabv3/ResNet101 的参数量仍不算大；`1e-4` 是语义分割和 ResNet 微调里很常见的默认正则强度。

为什么选 `1e-4`：

- 对 CNN/ResNet 是常用经验值。
- 太小会削弱正则，可能让 step0 对初始类过拟合。
- 太大可能压制模型容量，导致初始类和后续冻结特征都变弱。

由于后续 closed-form 阶段冻结 encoder，step0 中 weight decay 对最终持续学习结果有间接但很关键的影响：它决定了 frozen encoder 的泛化质量。

### 3.3 polynomial learning rate schedule

**代码位置**：`run.sh:14`，`utils/scheduler.py:7-16`，`utils/scheduler.py:90-93`，`trainer/trainer.py:233-234`

代码里的 PolyLR 公式为：

```text
lr_t = max(base_lr * (1 - t / max_iters)^0.9, min_lr)
```

并且 `scheduler.step()` 在每个 iteration 后调用，而不是每个 epoch 后调用。

作用：

- 训练初期保持较大学习率，快速适配 VOC 初始类。
- 训练后期平滑降低学习率，让分割边界和分类头权重收敛得更稳定。
- DeepLab 系列语义分割中 poly schedule 是非常经典的配置，很多 VOC/ADE/Cityscapes 实验都沿用。

为什么使用 `power=0.9`：

- `0.9` 是 DeepLab 训练中常见默认值。
- 衰减比线性稍缓，避免学习率过早降得太低。
- 配合 50 epochs 和 batch-size 设定，能够在有限训练轮数内兼顾收敛速度与后期微调。

代码细节上，虽然 `run.sh` 写了 `LR=0.01`，但 optimizer 实际分组为：

```text
backbone lr = 0.001
classifier lr = 0.01
```

所以 poly schedule 会分别衰减这两个 base LR。这个分组策略在迁移学习中常见：预训练 backbone 用较小 LR，随机初始化或任务相关的 classifier 用较大 LR。

### 3.4 BCE loss

**代码位置**：`utils/loss.py:24-56`，`utils/loss.py:65-66`

代码实现不是直接 `nn.BCEWithLogitsLoss` 对原标签使用，而是：

1. 将 `targets` 转成 one-hot。
2. 将 ignore label `255` 临时转成额外类别，再切掉额外类别。
3. 对 `B x C x H x W` logits 做 binary cross entropy。
4. 对类别维求和。
5. 对非 ignore 像素求 mean。

为什么语义分割增量学习里会用 BCE：

- 类增量语义分割中，当前 step 可能只知道部分标签，背景类也可能混入旧类对象。
- BCE 把每个类别看作独立二分类信号，相比强制 softmax 互斥的 CE，和增量/未知/背景建模更兼容。
- 后续 `bce_loss` 推理时，代码在 `trainer/trainer.py:343-350`、`trainer/trainer.py:443-450` 用 sigmoid 再 argmax 得到预测。

为什么论文选择 BCE：

- 它延续了很多 CSS 方法对背景/未知类问题的处理习惯。
- 它和伪标签机制更自然：旧类预测可以通过 sigmoid 置信度筛选后填回 background 区域。

需要注意：如果把 BCE 换成 CE，不只是换一个 loss 名称。它会改变 step0 logits 的标定、旧类/背景关系、伪标签分数解释，通常必须重新训练 step0 并完整评估后续 steps。

### 3.5 `d_E = 8192`

**代码位置**：`run.sh:21` 当前为 `8196`，`trainer/trainer.py:281-285`，`network/Buffer.py:39-68`

`d_E` 是 RHL 随机高维映射后的维度。原始 encoder 输出的像素特征先经过固定随机线性映射和 ReLU，再交给解析线性分类器。理论直觉来自高维随机特征：映射到更高维空间后，样本更可能线性可分，从而提升解析线性头的可塑性。

为什么需要它：

- 只用冻结 encoder 的原始特征做线性分类，增量类可能不可分。
- RHL 增加特征维度，为闭式线性分类器提供更强表达能力。
- 论文消融显示去掉 RHL 会明显损害新类性能，CFSSeg Table 5 中 overlap `10-1` 的 new-class mIoU 从 `41.20` 降到 `9.36`。

为什么选 `8192`：

- 维度越大，随机特征表达能力通常越强，但矩阵逆和显存开销也变大。
- SegACIL 论文的 buffer size 消融显示 mIoU 随 buffer size 增大而上升，但存在计算成本折中。
- `8192` 是 2 的幂，工程上对矩阵/显存布局也比较自然。

当前代码的 `8196` 与论文 `8192` 不一致。这个差异只有 4 维，通常不会导致方法逻辑变化，但严格复现实验时应改成：

```bash
BUFFER=8192
```

### 3.6 `gamma = 1`

**代码位置**：`run.sh:23`，`trainer/trainer.py:284-285`，`network/AnalyticLinear.py:102-105`，`network/AnalyticLinear.py:155-165`

`gamma` 是闭式岭回归中的正则项：

```text
min ||Y - XW||^2 + gamma ||W||^2
```

在代码中，`RecursiveLinear` 初始化：

```text
R = I / gamma
```

后续每个 batch 用：

```text
R = inverse(inverse(R) + X^T X)
weight += R X^T (Y - X weight)
```

作用：

- 避免 `X^T X` 奇异或病态，提升矩阵求逆稳定性。
- 控制解析分类头权重范数，降低过拟合。
- 在 RHL 维度很高时尤其重要，因为高维随机特征可能造成相关矩阵数值条件变差。

为什么选 `1`：

- 论文消融显示 `gamma` 从 `0.01` 到 `100` 结果变化很小，说明方法对该参数较鲁棒。
- `gamma=1` 是一个中等强度的默认值，既不过度放大 `R=I/gamma`，也不过度收缩分类头。
- 代码里的数值稳定性提示也说明，遇到不稳定时可以增大 `gamma`。

因此，如果目标是调参提升性能，`gamma` 可以扫，但它不太可能是最大收益来源；更大的收益通常来自 step0 表征、RHL 维度/归一化、伪标签和数据设置。

### 3.7 `tau = 0.4`

**代码位置**：`utils/parser.py:66-67`，`trainer/trainer.py:457-477`

CFSSeg 的 `tau=0.4` 是不确定性阈值；代码里的 `pseudo_label_confidence` 是置信度阈值。两者关系为：

```text
uncertainty = 1 - confidence
uncertainty <= 0.4  <=>  confidence >= 0.6
```

作用：

- disjoint / overlap 设置中，新 step 的标注会把旧类像素当 background。
- 如果直接用这些标签训练，旧类区域会被错误推向 background，产生 semantic drift。
- 伪标签用旧模型预测把高置信度旧类像素填回来，缓解旧类遗忘。

为什么不能设太高或太低：

- 置信度阈值太低：会引入很多错误旧类伪标签，污染新 step 训练。
- 置信度阈值太高：保留下来的伪标签太少，接近不用伪标签。
- SegACIL 论文 Table IV 显示 confidence threshold `0.6` 比 `0.7/0.8/0.9` 更好。

当前代码默认 `pseudo_label_confidence=0.7`，如果目标是按 CFSSeg 论文的 2D `tau=0.4` 复现，建议显式改为 `0.6` 并打开 `--use_pseudo_label`，同时只在 disjoint/overlap 中使用。

## 4. 调参是否必须完整重训 step0

### 4.1 必须重训 step0 的参数

以下参数会改变 step0 学到的 DeepLabv3/ResNet101 表征，真实验证通常必须从 step0 重训：

| 参数 | 是否必须重训 step0 | 原因 |
|---|---:|---|
| SGD momentum | 是 | 改变每一步参数更新轨迹 |
| weight decay | 是 | 改变 backbone/classifier 正则与泛化 |
| step0 LR / poly schedule / power / warmup | 是 | 直接改变收敛轨迹和最终 frozen encoder |
| BCE vs CE/Focal | 是 | 改变 logits、类别竞争关系和背景建模 |
| batch size | 是 | 改变梯度噪声、BN 统计、scheduler iteration 数和 drop_last 样本 |
| crop size / augmentation | 是 | 改变训练分布和特征 |
| output stride | 是 | 改变特征图分辨率和 DeepLab 结构行为 |

原因很直接：CFSSeg/SegACIL 的闭式部分不是从原图重新学习特征，而是依赖 step0 训练好的 encoder/backbone。step0 一变，后续所有 RHL 输入特征都变了。

### 4.2 不一定需要重训 step0 的参数

以下参数主要发生在冻结 encoder 之后，可以复用一个固定 step0 checkpoint：

| 参数 | 是否必须重训 step0 | 推荐验证方式 |
|---|---:|---|
| `d_E` / buffer size | 否 | 复用 step0 checkpoint，从 step1 重建 AIR 并跑后续 steps |
| `gamma` | 否 | 复用 step0 checkpoint，扫 `gamma` 后比较最终 mIoU |
| `tau` / pseudo-label threshold | 否 | 只在 disjoint/overlap 复用 step0，跑 step1+ |
| RHL seed | 否 | 固定 step0，改变 RHL 随机映射，多 seed 评估均值/方差 |
| 当前本地扩展 `rhl_norm` | 否 | 固定 step0，跑增量阶段，比较 mIoU 和数值稳定性 |

不过“不重训 step0”不等于“不训练”。这些参数仍然需要至少跑闭式增量阶段和测试评估。好处是增量阶段比 step0 SGD 快得多，尤其 `gamma`、`d_E`、`tau` 可以用已有 step0 权重做较便宜的 sweep。

## 5. Codex / agent 能不能做无需训练的超参数搜索

严格说，不能。原因是 mIoU 是训练和数据共同作用后的经验结果，静态代码或论文文字无法推出某个新超参组合的真实性能。

Codex / agent 可以做的事情：

1. **静态配置审计**：发现 `8196` vs `8192`、`tau` 不确定性阈值 vs 置信度阈值、batch size 差异、脚本当前只跑 step1 等问题。
2. **设计实验矩阵**：例如固定 step0 checkpoint，扫 `gamma in {0.1, 1, 10}`、`BUFFER in {4096, 8192, 16384}`、`pseudo_label_confidence in {0.5, 0.6, 0.7}`。
3. **自动生成脚本**：为每组参数生成不同 `SUBPATH`，避免覆盖结果。
4. **利用已有 checkpoint 做低成本增量阶段搜索**：这不是“无需训练”，但可以避免重复 step0。
5. **解析日志和 JSON**：自动汇总 `Mean IoU`、`0-15 mIoU`、`16-20 mIoU`，生成表格和推荐。
6. **早停或失败检测**：监控 NaN、矩阵求逆错误、显存、异常 loss。

Codex / agent 不能可靠做的事情：

1. 不运行训练/评估就断言哪个超参数 mIoU 更高。
2. 不看实际数据和 checkpoint 就判断 `gamma=0.1` 一定优于 `gamma=1`。
3. 用静态推理替代 step0 SGD 重训后的真实性能验证。

因此更实际的策略是：

```text
先固定论文配置复现 baseline
-> 再只扫不需要重训 step0 的闭式阶段参数
-> 如果确认收益明显，再考虑昂贵的 step0 参数搜索
```

## 6. SegACIL 与 CFSSeg 的 2D 实验配置是否完全一致

结论：**结果表基本一致，但实验配置文字并非完全一致。**

### 6.1 一致的地方

| 项目 | SegACIL | CFSSeg 2D | 判断 |
|---|---|---|---|
| 数据集 | VOC2012 | VOC2012 | 一致 |
| 类别数 | 21 | 21 | 一致 |
| 协议 | sequential / disjoint / overlap | sequential / disjoint / overlapped | 一致 |
| 任务 | sequential `15-1`, `15-5`; disjoint/overlap `15-1`, `10-1` | 2D 表中相同 | 一致 |
| 模型 | DeepLabv3 + ResNet101 | DeepLabv3 + ResNet101 | 一致 |
| 预训练 | ImageNet-1K | ImageNet-1K | 一致 |
| epochs | 50 | 50 | 一致 |
| optimizer | SGD | SGD | 一致 |
| lr | `1e-2` | `1e-2` | 一致 |
| momentum | `0.9` | `0.9` | 一致 |
| weight decay | `1e-4` | `1e-4` | 一致 |
| LR schedule | polynomial | polynomial | 一致 |
| loss | BCE | BCE | 一致 |
| RHL / closed-form | 使用 | 使用 | 一致 |
| `d_E` / buffer | `8192` | `8192` | 一致 |
| `gamma` | `1` | `1` | 一致 |
| 2D 主结果 | VOC 表结果 | VOC 表结果 | 基本完全一致 |

### 6.2 不一致或表述变化的地方

| 项目 | SegACIL 论文 | CFSSeg 论文 | 分析 |
|---|---|---|---|
| step0 batch size | `16` | `32` | 明确不一致。当前 README / `run.sh` 更接近 CFSSeg 的 `32`。 |
| 伪标签阈值写法 | 直接写 confidence threshold `0.6` | 写 uncertainty threshold `tau=0.4` | 表面不一致，但若 `confidence = 1 - uncertainty`，二者等价于 confidence `0.6`。 |
| 3D 实验 | 无 | 增加 S3DIS / ScanNet / DGCNN | CFSSeg 扩展了 3D，当前 SegACIL 代码库源代码未发现对应完整实现。 |
| 效率实验 | SegACIL 论文未突出 3D 效率表 | CFSSeg Table 6 报 Ours batch `64`、FT batch `32` | CFSSeg 更强调闭式阶段效率。 |
| 论文命名 | SegACIL | CFSSeg | 方法核心相同，CFSSeg 是更宽的 2D+3D 命名。 |

最重要的判断是：**两篇论文的 2D 结果完全像同一套结果，但配置描述经过了修改，尤其 batch size 和 `tau` 的符号定义发生变化。**

如果要在当前代码库中做“最接近 CFSSeg 2D 论文”的配置，建议：

```bash
MODEL="deeplabv3_resnet101"
LR=0.01
LOSS_TYPE="bce_loss"
DATASET="voc"
TASK="15-1"            # 或 15-5 / 10-1
LR_POLICY="poly"
SETTING="sequential"   # 或 disjoint / overlap
TRAIN_EPOCH=50
SPECIAL_BATCH_SIZE=32
DEFAULT_BATCH_SIZE=64
BUFFER=8192            # 改正当前 8196
GAMMA=1
RHL_NORM=none
RHL_SEED=-1
```

若跑 disjoint / overlap 并要匹配 CFSSeg `tau=0.4` 的不确定性阈值，则应加：

```bash
--use_pseudo_label --pseudo_label_confidence 0.6
```

## 7. 推荐的后续实验优先级

### 7.1 先修正可疑复现实验偏差

1. 把 `BUFFER=8196` 改为 `BUFFER=8192`，先跑一次与论文严格一致的 baseline。
2. 确认 `DATA_ROOT` 指向实际存在的 VOC2012 路径；当前 `run.sh` 中 `/root/2TStorage/lyc/SegACIL/data_root/VOC2012` 在本次检查时不存在。
3. 如果目标是完整复现，不要让 `START_STEP=1 END_STEP=1` 停在单步；需要根据 `TASK` 设置完整 step 范围。
4. sequential 不需要 pseudo label；disjoint/overlap 再打开。

### 7.2 低成本调参顺序

优先不重训 step0：

1. `gamma`: `{0.1, 1, 10}`，预期变化不大，但能验证数值稳定性。
2. `BUFFER`: `{4096, 8192, 16384}`，观察新类 mIoU 与显存/时间折中。
3. `pseudo_label_confidence`: `{0.5, 0.6, 0.7}`，只用于 disjoint/overlap。
4. `rhl_seed`: 多 seed 看方差，判断单次随机 RHL 是否稳定。
5. 本地扩展 `rhl_norm`: `none` vs `l2_sqrt` 等，但这属于改方法，不再是纯论文复现。

再考虑昂贵 step0：

1. batch size 16 vs 32。
2. backbone LR `0.001` vs `0.01` 或其他分组策略。
3. BCE vs CE/Focal。
4. poly power / warmup。
5. weight decay。

### 7.3 最小化实验覆盖建议

如果资源有限，建议不要一开始全量扫 step0 参数。更稳的顺序是：

```text
VOC 15-1 sequential baseline
-> 固定 step0，扫 BUFFER/gamma/RHL seed
-> VOC 10-1 overlap，扫 pseudo_label_confidence
-> 只有当增量阶段收益明确时，再做 step0 batch/LR/loss 搜索
```

这样能把昂贵的 SGD 搜索压到最后，避免把 GPU 时间花在静态审计已经能排除的配置上。

## 8. 总结

CFSSeg 论文的 2D 部分与 SegACIL 论文在方法和结果上高度一致，核心都是：先用 DeepLabv3/ResNet101 训练 step0，再冻结 encoder，通过 RHL 高维随机映射和 RecursiveLinear 闭式更新持续学习新类。当前代码库实现了这条 2D 主线，但不是所有默认配置都与论文逐项一致。

最需要你注意的三点是：

1. 当前代码的 `BUFFER=8196` 与论文 `d_E=8192` 不一致。
2. CFSSeg 的 `tau=0.4` 是不确定性阈值，映射到当前代码应接近 `pseudo_label_confidence=0.6`。
3. step0 相关超参必须重训 step0 才能验证，闭式阶段超参可以复用 step0 checkpoint 做更便宜的搜索。

因此，后续如果要做严谨复现，应先把 baseline 配置与论文对齐，再把调参分成“需要重训 step0”和“只需重跑增量阶段”两类管理。
