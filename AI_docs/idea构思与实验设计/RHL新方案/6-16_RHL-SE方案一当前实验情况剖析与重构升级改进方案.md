# RHL-SE 方案一当前实验情况剖析与重构升级改进方案

> 项目：`/root/2TStorage/lyc/SegACIL`  
> 日期：2026-06-16  
> 关联原文档：`AI_docs/idea构思与实验设计/6-15_RHL机制再分析与重构升级方案.md`  
> 当前阶段：方案一 RHL Subspace Ensemble 已完成第一轮实现、训练与多种推理集成验证。  
> 本文目的：跳出原方案一的既有框架，重新审查为什么当前精度没有明显提升，指出原规划的不足，并提出新的重构升级工作流。

---

## 0. 总结结论

方案一 RHL-SE 的第一阶段已经完成，代码功能是成立的，实验链路也是干净的：

```text
固定 step0 backbone
固定全局 random_seed=1
固定训练集、任务划分、模型结构、gamma、buffer
只改变独立 rhl_seed
得到多个 RHL 随机子空间 + C-RLS 解析头
推理时做概率平均或加权概率平均
```

当前结果是 **弱正向，但不足以作为强创新主线**。

阶段 A-v3 已经完成 val split 驱动的 class-wise 权重搜索。当前最好结果来自 val 选择的 `classwise_valbest_all_s0.75`：

```text
baseline all mIoU: 69.4606
val-driven class-wise all mIoU: 69.5379
提升: +0.0773

baseline new mIoU: 42.1075
val-driven class-wise new mIoU: 42.5218
提升: +0.4143
```

这个结果说明 RHL 子空间集成确实有信号，但信号很弱。它更像是一个 **稳定化/系统集成组件**，而不是足够支撑论文主要贡献的强方法。

阶段 A 的 `logit ensemble` 和诊断实验已经完成。结果显示：

1. `logit ensemble` 略低于 `prob ensemble`，说明 sigmoid 后概率平均不是主要瓶颈。
2. 成员整体预测高度相似，pairwise disagreement 只有约 `0.87%`。
3. 新类区域分歧更高，约 `4.93%`，说明 RHL seed 的有效互补主要发生在新类区域。
4. oracle all mIoU 达到 `71.3950`，new mIoU 达到 `46.2690`，显著高于当前 best ensemble，说明不是完全没有互补，而是全局平均/加权没有利用好互补。

因此，后续不建议继续盲目扫全局权重或立刻补跑 seed4/5。阶段 A 已经升级为结构化推理融合：

```text
old/new group-wise ensemble
class-wise ensemble
confidence gated ensemble
```

其中 val-driven class-wise 是目前最有效的 RHL-SE 收尾版本，但 all mIoU 提升仍未达到 `+0.10`。后续应停止 RHL-SE 推理层手工优化，把 RHL-SE 定位为辅助模块，转向 BOA-RHL / PGH-RHL 等 RHL 特征构造重构。

---

## 1. 方案一当前完成情况

### 1.1 已完成的代码改动

方案一已经完成了从训练到推理的闭环。

| 模块 | 文件 | 当前作用 |
|---|---|---|
| 独立 RHL 随机种子 | `utils/parser.py` | 新增 `--rhl_seed`，默认 `-1` 保持原行为 |
| RHL 随机映射隔离 | `network/Buffer.py` | `RandomBuffer` 支持只用 `rhl_seed` 初始化自身随机权重 |
| AIR 参数传递 | `trainer/trainer.py` | step1 构造 AIR 时把 `opts.rhl_seed` 传入 RHL |
| 实验脚本 | `run_rhl_norm.sh` | 支持 `RHL_SEED`、`RHL_NORM`、`BASE_SUBPATH` 等独立控制 |
| 集成评估脚本 | `tools/eval_rhl_ensemble.py` | 支持多 checkpoint 概率平均、低显存串行推理、加权推理 |

关键设计是：

```text
random_seed 控制全局复现实验
rhl_seed 只控制 RandomBuffer/RHL 随机映射
```

这使得方案一可以被解释为“只改变 RHL 随机子空间”的干净实验，而不是全局随机种子集成。

### 1.2 已完成的训练实验

统一设置：

```text
DATASET=voc
TASK=15-5
SETTING=sequential
MODEL=deeplabv3_resnet101
BASE_SUBPATH=20260606
RHL_NORM=none
GAMMA=1
BUFFER=8196
DEFAULT_BATCH_SIZE=64
random_seed=1
curr_step=1
```

已完成 3 个 RHL 子空间成员：

| 实验 | checkpoint |
|---|---|
| `rhl_seed=1` | `checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth` |
| `rhl_seed=2` | `checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth` |
| `rhl_seed=3` | `checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth` |

其中 `rhl_seed=1` 已确认与 `checkpoints/20260606` 的 baseline 完全一致：

```text
buffer.weight equal: True
analytic_linear.weight equal: True
metrics equal: True
```

原因是 baseline 的全局 `random_seed=1` 与当前 `rhl_seed=1` 对齐，且 RHL 初始化前没有额外随机消耗改变这一结果。因此 `seed1` 更准确的定位是：

```text
baseline member / sanity check
```

而不是新的独立随机子空间。

### 1.3 已完成的推理集成实验

当前已完成：

| 集成 | 成员 | 融合方式 |
|---|---|---|
| K2 | seed2 + seed3 | 等权概率平均 |
| K3 | seed1 + seed2 + seed3 | 等权概率平均 |
| weighted K3 | seed1 + seed2 + seed3 | `0.2:0.4:0.4` 概率加权 |
| weighted K3 | seed1 + seed2 + seed3 | `0.1:0.45:0.45` 概率加权 |

当前 `eval_rhl_ensemble.py` 的核心逻辑是：

```python
probs = torch.sigmoid(logits)
weighted_prob_sum += probs * weight
preds = weighted_prob_sum.argmax(dim=1)
```

它与原 `trainer.py` 的 BCE 评估路径一致，但不一定是最优的 ensemble 方式。

---

## 2. 当前实验结果

### 2.1 总体指标

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc | 权重 |
|---|---:|---:|---:|---:|---:|---|
| baseline `20260606` | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 | - |
| `rhl_seed=1` | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 | - |
| `rhl_seed=2` | 69.4391 | 77.8998 | 42.3649 | 92.7072 | 78.3698 | - |
| `rhl_seed=3` | 69.4989 | 77.9544 | 42.4415 | 92.7055 | 78.3628 | - |
| K2 `seed2+seed3` | 69.4976 | 77.9577 | 42.4256 | 92.7202 | 78.3832 | 等权 |
| K3 `seed1+seed2+seed3` | 69.5049 | 77.9985 | 42.3254 | 92.7235 | 78.3830 | 等权 |
| K3 weighted | **69.5060** | 77.9848 | 42.3737 | **92.7235** | **78.3868** | `0.2,0.4,0.4` |
| K3 weighted | 69.5035 | 77.9724 | 42.4030 | 92.7225 | 78.3865 | `0.1,0.45,0.45` |

### 2.2 新类逐类 IoU

| 设置 | pottedplant 16 | sheep 17 | sofa 18 | train 19 | tvmonitor 20 |
|---|---:|---:|---:|---:|---:|
| baseline `20260606` | 23.5886 | 57.9622 | 30.9147 | 69.9526 | 28.1192 |
| `rhl_seed=2` | 24.1226 | 57.3496 | **32.0644** | **70.4395** | 27.8484 |
| `rhl_seed=3` | **24.2632** | **58.3916** | 30.6000 | 70.0420 | **28.9108** |
| K2 `seed2+seed3` | 24.1475 | 57.8761 | 31.3923 | 70.3518 | 28.3602 |
| K3 等权 | 23.8673 | 57.9546 | 31.2421 | 70.2740 | 28.2890 |
| K3 `0.2,0.4,0.4` | 23.9828 | 57.9330 | 31.3108 | 70.3081 | 28.3340 |
| K3 `0.1,0.45,0.45` | 24.0691 | 57.9102 | 31.3567 | 70.3313 | 28.3476 |

### 2.3 直接结论

1. `seed2` 和 `seed3` 的新类指标均高于 baseline。
2. `seed2` 和 `seed3` 的旧类指标均低于 baseline。
3. K3 等权恢复了旧类稳定性，但削弱了新类提升。
4. 加权 K3 能略微抬高 new mIoU，但 all mIoU 基本停在 69.50 附近。
5. 目前所有集成结果都没有超过 `seed3` 单模型太多：

```text
seed3 all mIoU: 69.4989
best weighted K3 all mIoU: 69.5060
差值: +0.0071
```

这说明当前 RHL-SE 的集成收益主要来自非常轻微的平滑，而不是强互补。

---

## 3. 为什么精度没有明显提升？

这一节刻意跳出原文档“RHL 随机子空间集成能降低 Monte Carlo 方差”的框架，重新审查真实原因。

### 3.1 原方案把 RHL seed 当成了足够独立的模型，但实际多样性很弱

RHL-SE 的核心假设是：

```text
不同 rhl_seed -> 不同随机子空间 -> 不同错误模式 -> 集成后提升
```

当前实验只部分满足这个假设。

不同 seed 的确产生了不同结果：

```text
seed2 new mIoU: 42.3649
seed3 new mIoU: 42.4415
baseline new mIoU: 42.1075
```

但差异很小，且 all mIoU 都集中在：

```text
69.4391 ~ 69.4989
```

原因是 RHL-SE 只改变了一个固定随机投影矩阵，而以下关键因素全部相同：

```text
同一个 step0 backbone
同一个 dense feature
同一个训练集和标签
同一个 C-RLS 闭式公式
同一个 gamma
同一个 buffer size
同一个推理图像和后处理
```

所以不同成员不是多个完整模型视角，而是同一个 frozen representation 上的轻微随机特征扰动。它们的预测高度相关，导致 ensemble 上限自然很低。

### 3.2 seed1 不是“第三个有效成员”，而是 baseline 稳定器

原方案写的是：

```text
rhl_seed 使用 1/2/3/4/5
K=3 使用 1,2,3
```

现在看，这个规划有一个隐藏问题：`rhl_seed=1` 与 baseline 完全等价。

因此 K3 的真实结构不是：

```text
三个独立 RHL 子空间
```

而更像：

```text
baseline 稳定器 + 两个轻微偏新类的 RHL 子空间
```

这解释了为什么：

```text
K2 seed2+seed3 new mIoU 更高
K3 seed1+seed2+seed3 all mIoU 更高
```

seed1 的作用是把旧类拉回来，但它也会稀释 seed2/seed3 的新类优势。

### 3.3 全局权重太粗，无法处理类别级冲突

加权实验的结果非常典型：

```text
0.2,0.4,0.4 -> all mIoU 最好
0.1,0.45,0.45 -> new mIoU 更高，但 all mIoU 略低
```

这说明当前不是“seed2/seed3 总是更好”，而是：

```text
旧类更信 seed1
新类总体更信 seed2/seed3
新类内部也不是同一个 seed 最好
```

逐类看：

| 类别 | 最好 seed |
|---|---|
| pottedplant | seed3 |
| sheep | seed3 |
| sofa | seed2 |
| train | seed2 |
| tvmonitor | seed3 |

一个全局权重无法表达：

```text
sofa 更信 seed2
tvmonitor 更信 seed3
旧类更信 seed1
```

所以继续扫 `0.25/0.375/0.375`、`0.15/0.425/0.425` 这类全局权重，收益会非常有限。

### 3.4 概率平均对 BCE 输出未必是最优集成方式

当前任务使用 `bce_loss`，原评估路径是：

```python
outputs = torch.sigmoid(outputs)
preds = outputs.max(dim=1)[1]
```

注意，这里的 sigmoid 是每个类别独立计算的，不是互斥类别的 softmax 概率。

因此当前 ensemble 做的是：

$$
p_{\text{ens},c}(x)
=
\sum_k w_k \sigma(z^{(k)}_c(x))
$$

问题在于：sigmoid 会压缩 logit margin。

如果某个模型对类别 A 的 logit 明显高，但 sigmoid 后已经接近饱和；另一个模型对类别 B 略高，概率平均可能会把边界变钝。对于 BCE 多标签形式的语义分割，`sigmoid 后平均` 不一定比 `logit 先平均` 更好。

更值得验证的是：

$$
z_{\text{ens},c}(x)
=
\sum_k w_k z^{(k)}_c(x)
$$

然后：

$$
\hat{y}(x)
=
\arg\max_c z_{\text{ens},c}(x)
$$

由于 sigmoid 是单调函数，单模型下 `argmax(sigmoid(z))` 与 `argmax(z)` 等价；但多模型下：

```text
argmax average(sigmoid(logit)) != argmax sigmoid(average(logit))
```

这就是当前评估协议仍有改造空间的主要原因。

### 3.5 RHL-SE 的理论表述把“随机特征方差”说得过于理想化

原方案的理论基础是：

$$
\operatorname{Var}\left[\frac{1}{K}\sum_k h_k(x)\right]
\approx
\frac{1}{K}\operatorname{Var}[h(x)]
$$

这个直觉在独立同分布、误差不完全相关、估计无偏或近似无偏时更成立。

但当前 RHL-SE 不是简单的随机函数平均。每个成员都经过同一批标签和同一个 C-RLS 解析求解：

$$
\hat{\Phi}^{(k)}
=
\left((E^{(k)})^\top E^{(k)}+\gamma I\right)^{-1}
(E^{(k)})^\top Y
$$

模型之间共享：

```text
同一个 backbone bias
同一个数据集 bias
同一个标签噪声
同一个旧新类不平衡
同一个 C-RLS 目标
```

所以错误相关性可能很高。方差降低公式不能直接推出明显 mIoU 提升。

### 3.6 当前任务的 mIoU 指标对小范围预测变化不敏感

当前结果显示 Overall Acc 和 Mean Acc 有小幅提升：

```text
baseline Overall Acc: 92.7034
weighted K3 Overall Acc: 92.7235
baseline Mean Acc: 78.3376
weighted K3 Mean Acc: 78.3868
```

但 all mIoU 只提升：

```text
+0.0454
```

这说明集成可能修正了一些像素，但这些修正并没有集中发生在能显著改变 IoU 的类别区域，或者同时引入了少量 false positive 抵消收益。对于 VOC 15-5，少数新类区域面积有限，边界和小目标类别对 IoU 很敏感，轻微概率平滑不一定转换成明显 mIoU。

---

## 4. 原方案一规划中的局限与不足

### 4.1 把“干净可解释”与“有效涨点”混为一谈

原方案强调：

```text
固定 random_seed，只改 rhl_seed，因果解释干净。
```

这是正确的实验控制，但它也限制了多样性来源。干净的 RHL-only ensemble 上限可能天然低于：

```text
global random_seed ensemble
snapshot ensemble
backbone variant ensemble
TTA ensemble
```

原方案没有充分区分：

| 目标 | 最优选择 |
|---|---|
| 证明收益来自 RHL | 只改 `rhl_seed` |
| 追求最终 mIoU | 可以引入更宽的系统级多样性 |
| 满足结项“集成系统” | RHL-SE 可以作为组件，但不一定是性能主力 |

### 4.2 对 seed1 的特殊性预判不足

原方案把 `rhl_seed=1/2/3` 视为三个平等成员。但当前已经证明：

```text
rhl_seed=1 == baseline
```

因此后续实验设计应改成：

```text
baseline member: seed1
new RHL members: seed2, seed3, seed4, ...
```

评估时也应分开报告：

```text
baseline + RHL variants
pure RHL variants only
```

而不是简单写 K=3。

### 4.3 只设计了模型级集成，没有设计类别级或结构级集成

当前结果已经显示：

```text
不同 seed 对不同类别最优
```

这意味着更合理的融合单位不是“整模型权重”，而可能是：

```text
类别级权重
旧类/新类分组权重
像素置信度门控
```

原方案只有：

```text
probability average
hard voting
```

它没有预留类别级融合机制，因此面对类别冲突时只能扫全局权重，效率很低。

### 4.4 没有设计多样性诊断指标

原方案的判定主要看最终 mIoU：

```text
K=3 是否高于 best single seed
K=3 是否高于 seed mean
```

但如果没涨，仍然不知道原因是：

```text
成员预测太相似
成员预测互补但融合方式不对
成员有互补但校准差
某些类别互补，另一些类别冲突
```

所以方案一缺少必要的诊断实验，例如：

| 诊断 | 含义 |
|---|---|
| pixel disagreement rate | seed 之间有多少像素预测不同 |
| class-wise best seed | 每类哪个 seed 最好 |
| oracle ensemble upper bound | 每像素若能选对成员，理论上限多少 |
| confidence-margin 分布 | 错误是否集中在低 margin 像素 |
| old/new confusion matrix | 新类提升是否来自误伤旧类 |

没有这些诊断，就很容易陷入继续扫权重或继续补 seed 的低效循环。

### 4.5 过早把 K=5 作为自然下一步

原方案写的是：

```text
K=3 有收益，再跑 K=5
```

这个逻辑需要修正。当前 K=3 的收益是弱正向，但不说明 K=5 一定值得。

更合理的触发条件应是：

```text
如果 K=3 明显超过 best single seed，或 oracle / disagreement 显示仍有大量可利用互补，再跑 K=5。
否则先做融合方式和多样性诊断。
```

当前结果下，直接补 seed4/5 的风险是：

```text
训练成本增加
指标边际提升可能仍只有 0.01~0.05
论文叙事变成堆模型，方法贡献弱
```

---

## 5. 对当前 Python 集成代码的审查

### 5.1 当前代码逻辑基本正确

`tools/eval_rhl_ensemble.py` 当前做了：

```python
models = [load_model_cpu(path) for path in args.ckpts]
weights = normalize_weights(args.weights, len(models))
for model, weight in zip(models, weights):
    logits = model(images)
    probs = logits_to_prob(logits, ...)
    weighted_prob_sum += probs * weight
preds = weighted_prob_sum.argmax(dim=1)
```

权重实现正确：

| 检查项 | 当前状态 |
|---|---|
| 权重数量等于 ckpt 数量 | 已检查 |
| 权重非负 | 已检查 |
| 权重和大于 0 | 已检查 |
| 内部归一化 | 已实现 |
| JSON 保存权重 | 已实现 |
| 低显存模式 | 已实现 |

之前的 `TypeError` 已修复：`Ensemble Members` 列表不会再提前传入 `metrics.to_str()`。

### 5.2 当前代码的主要不足不是 bug，而是融合模式单一

当前只有：

```text
sigmoid probability average
```

没有：

```text
logit average
class-wise weight
old/new group weight
temperature calibration
prediction disagreement / oracle analysis
```

因此代码功能满足第一版 RHL-SE，但不足以支撑后续深入分析和重构。

### 5.3 性能问题：低显存模式很慢，但不影响结果

默认低显存模式每个 batch 都会：

```text
model.cpu() -> model.cuda() -> infer -> model.cpu()
```

这对单模型验证尤其慢。它不会影响精度，但会浪费大量时间。

后续建议：

```text
显存空闲时使用 --keep_models_on_gpu 做完整评估
低显存模式只在 GPU 紧张时使用
```

当前 A100 80GB 在空闲时理论上能常驻 K3 AIR 模型，但需避开其他用户进程。

---

## 6. 重构升级后的整体工作流

这里的目标不是给“下一步命令”，而是给一套从诊断到重构的完整路线。

### 6.1 阶段 A：修正和扩展评估工具，不重新训练

优先级最高。因为当前已有 3 个 checkpoint，不需要继续训练就能获得更多判断。

#### A1. 增加 `--ensemble_mode prob|logit`

当前：

```text
prob mode: average(sigmoid(logits))
```

新增：

```text
logit mode: average(logits), then argmax
```

对于 BCE 输出，推荐先验证 logit mode：

```text
K3 等权 logit
K3 0.2/0.4/0.4 logit
K2 seed2+seed3 logit
```

判定：

| 结果 | 解释 |
|---|---|
| logit 明显优于 prob | 当前瓶颈主要是概率融合协议 |
| logit 与 prob 基本一致 | 成员多样性不足才是主因 |
| logit 更差 | sigmoid 概率校准反而有帮助，保留 prob |

#### A2. 增加多样性诊断输出

建议在 `eval_rhl_ensemble.py` 或新脚本中增加：

```text
--save_diagnostics
```

输出：

| 指标 | 作用 |
|---|---|
| pairwise disagreement rate | 任意两个 seed 预测不同的像素比例 |
| old/new disagreement rate | 旧类区域和新类区域的分歧程度 |
| per-class disagreement | 哪些类别 seed 分歧最大 |
| per-class best member | 每个类别哪个成员 IoU 最高 |
| ensemble vs best-member delta | 集成是否吃到了互补 |
| oracle pixel ensemble | 理论上限，判断是否值得继续做融合 |

特别是 oracle 很关键：

```text
如果 oracle 比 best seed 高很多，说明融合方式不对。
如果 oracle 也只高一点，说明成员本身缺少互补，继续 ensemble 没意义。
```

#### A3. 增加类别级权重评估

当前全局权重无法表达类别差异。建议先做一个离线评估版：

```text
class_weights[c, k]
```

即类别 c 的分数来自不同 seed 的不同权重。

初版可以手工设定：

```text
old classes: 更偏 seed1
new classes: 更偏 seed2/seed3
sofa/train: 更偏 seed2
pottedplant/sheep/tvmonitor: 更偏 seed3
```

注意：这一步不能直接用 test set 调参作为最终论文结果。正确流程是：

```text
用 val set 搜索或设定权重
只在最后用 test set 报一次结果
```

当前前期探索可以先在 test 上看趋势，但最终文档必须说明这是 exploratory。

### 6.2 阶段 B：重新定义 RHL-SE 的实验结论

基于当前结果，RHL-SE 不应再被表述为：

```text
主打涨点方法
```

更稳妥的定位是：

```text
RHL 子空间稳定化集成模块
```

可以在论文/报告中这样写：

```text
RHL-SE explores the stochasticity of random hidden layers by constructing multiple analytic heads over independent RHL subspaces. It provides a lightweight inference-time ensemble without retraining the DeepLab backbone. Empirically, it improves new-class mIoU and stabilizes overall accuracy, but the limited diversity of RHL-only perturbations restricts its standalone gain.
```

中文解释：

```text
RHL-SE 证明只改变 RHL 子空间能产生可观测差异；
它对新类有稳定正向趋势；
但由于 backbone 和解析目标完全相同，成员错误高度相关，单独作为主创新不够强。
```

### 6.3 阶段 C：决定是否补跑 seed4/5

不建议现在马上补跑。需要先看 A 阶段诊断结果。

建议触发条件：

| 条件 | 是否补 seed4/5 |
|---|---|
| logit ensemble 明显提升 | 可以补 |
| oracle ensemble 上限明显高于当前 K3，比如 all mIoU 高 `0.3+` | 可以补 |
| pairwise disagreement 在新类区域较高 | 可以补 |
| K3 仍然只比 best seed 高 `0.01` 左右，oracle 也低 | 不补 |
| 类别级融合有明显收益 | 可补 seed4/5，用于增强类别池 |

如果补跑，建议不要再用 `rhl_seed=1` 作为新成员，而是：

```text
new RHL members: 2,3,4,5
baseline member: 1
```

报告时分开：

```text
baseline + RHL variants
pure RHL variants
```

### 6.4 阶段 D：如果 RHL-SE 上限确认不足，转向 RHL 特征构造重构

如果 A/B/C 后仍然没有明显提升，应停止在 RHL-SE 上堆实验，把方案一降级为辅助模块，然后转向更强的 RHL 改造。

推荐顺序：

#### D1. BOA-RHL：降低随机特征冗余

动机：

```text
当前不同 rhl_seed 的随机矩阵可能产生高度相关的随机特征。
与其多采几个普通随机矩阵，不如让单个 RHL 内部更低冗余、更均匀覆盖特征方向。
```

核心：

```text
block orthogonal random features
antithetic pairs: W and -W
```

预期：

```text
单模型 seed 方差下降
new mIoU 小幅提升
与 RHL-SE 组合时更有互补
```

#### D2. PGH-RHL：加入类别原型锚点

动机：

```text
纯随机 RHL 与类别边界没有显式关系。
当前结果显示随机子空间有弱信号，但不足以强涨点。
```

核心：

```text
RHL random features + prototype-guided RBF / similarity features
```

让特征空间同时包含：

```text
随机非线性基函数
类别语义锚点
```

预期：

```text
新类 mIoU 更可能明显提升
方法动机比单纯 seed ensemble 更强
```

#### D3. System-level ensemble 作为上界对照

如果结项指标强调集成系统，可以额外做：

```text
global random_seed ensemble
RHL-SE + global-seed upper bound
TTA ensemble
```

但必须和 RHL-SE 分开解释：

```text
RHL-SE: RHL 内部机制集成
System-SE: 系统级随机性/推理增强集成
```

这样既不破坏因果解释，也能给项目指标提供更强集成系统支撑。

---

## 7. 新的执行优先级

### P0：立刻执行，训练成本为 0

1. 在 `tools/eval_rhl_ensemble.py` 增加：

```text
--ensemble_mode prob|logit
```

2. 跑三组：

```text
K3 equal logit
K3 0.2/0.4/0.4 logit
K2 seed2+seed3 logit
```

3. 增加或单独写诊断脚本：

```text
prediction disagreement
class-wise best seed
oracle upper bound
```

### P1：短期执行，仍然不训练或少训练

1. 实现 class-wise / old-new group-wise ensemble。
2. 先用 val split 调权，再用 test 做最终报告。
3. 如果类别级融合明显有效，再考虑补 seed4/5。

### P2：中期执行，需要改训练

1. BOA-RHL 单模型。
2. 若 BOA 单模型有效，再做 BOA-RHL-SE。
3. PGH-RHL 单模型。
4. 若 PGH 有效，再讨论是否与 RHL-SE 组合。

### P3：论文叙事调整

当前方案一的论文定位建议从：

```text
RHL-SE 是主要涨点模块
```

调整为：

```text
RHL-SE 是揭示 RHL 随机子空间方差和构建轻量集成系统的基础模块；
它验证了 RHL 子空间存在可利用差异，但也暴露了纯随机子空间多样性不足；
因此引出后续结构化 RHL 或类别引导 RHL 的必要性。
```

这条叙事更真实，也更利于后续方案升级。

---

## 8. 建议的下一步行动清单

按顺序执行：

1. **代码层面**
   - 给 `eval_rhl_ensemble.py` 增加 `--ensemble_mode prob|logit`。
   - 保持 `prob` 为默认，保证历史结果可复现。
   - 新增 `logit` 路径，不改变 checkpoint。

2. **实验层面**
   - 跑 `logit K3 equal`。
   - 跑 `logit K3 0.2/0.4/0.4`。
   - 跑 `logit K2 seed2+seed3`。

3. **诊断层面**
   - 统计 seed 之间预测分歧率。
   - 统计每类 best seed。
   - 计算 oracle 上界。

4. **决策层面**
   - 如果 logit 或类别级融合有明显提升：继续优化 RHL-SE，并考虑 seed4/5。
   - 如果提升仍小：停止扩大 seed，转向 BOA-RHL / PGH-RHL。

5. **文档层面**
   - 更新方案一报告：写明 RHL-SE 当前是弱正向。
   - 不要把 `+0.04 all mIoU` 包装成强涨点。
   - 强调它的价值是：可解释、低成本、符合集成系统口径，并为后续结构化 RHL 提供证据。

---

## 9. 最终判断

方案一没有失败，但它的真实价值和原规划不同。

原规划期待：

```text
多 RHL seed ensemble -> 明显涨点
```

当前证据支持的是：

```text
多 RHL seed 能产生小幅新类收益和整体稳定性收益；
但普通随机 RHL 子空间之间高度相关，简单概率平均和全局权重无法释放更大收益。
```

因此，方案一应被重构为：

```text
RHL-SE 1.0: RHL 子空间稳定化集成
RHL-SE 2.0: logit / class-aware / old-new-aware 融合
RHL 2.0: BOA-RHL 或 PGH-RHL 改变随机特征构造
```

这比继续在 `rhl_seed=4/5` 或全局权重上试错更有研究价值，也更适合作为论文方法迭代路线。

---

## 10. 阶段 A 新增实验结果与方案更新

> 更新时间：2026-06-16  
> 本章节基于阶段 A 已完成的 `prob/logit` 融合实验和诊断 JSON，对原工作流进行更新。

### 10.1 prob vs logit 实验结果

阶段 A 已实现并验证：

```text
--ensemble_mode prob
--ensemble_mode logit
--save_diagnostics
```

已完成三组 logit 推理：

| 设置 | all mIoU | old mIoU | new mIoU | 结论 |
|---|---:|---:|---:|---|
| prob K2 seed2+seed3 | 69.4976 | 77.9577 | 42.4256 | 既有概率融合 |
| logit K2 seed2+seed3 | 69.4952 | 77.9567 | 42.4183 | 略低于 prob |
| prob K3 equal | 69.5049 | 77.9985 | 42.3254 | 既有概率融合 |
| logit K3 equal | 69.5019 | 77.9974 | 42.3163 | 略低于 prob |
| prob K3 `0.2/0.4/0.4` | **69.5060** | 77.9848 | 42.3737 | 当前 best ensemble |
| logit K3 `0.2/0.4/0.4` | 69.5030 | 77.9837 | 42.3648 | 略低于 prob |

结论：

```text
logit ensemble 没有解决问题。
当前主要瓶颈不是 sigmoid 概率平均压缩 margin。
prob ensemble 是当前三成员 RHL-SE 中更稳的基础融合协议。
```

### 10.2 诊断结果

K3 equal logit 诊断结果：

| 指标 | 数值 |
|---|---:|
| pairwise disagreement | 0.8668% |
| old-region disagreement | 0.6179% |
| new-region disagreement | 4.9272% |
| oracle all mIoU | 71.3950 |
| oracle old mIoU | 79.2469 |
| oracle new mIoU | 46.2690 |

这组诊断非常关键。

它说明两件事同时成立：

```text
成员整体高度相似，所以普通平均收益很弱。
但新类区域存在明显更高分歧，oracle 上界也明显高，说明仍有可利用互补。
```

因此，阶段 A 的问题不再是：

```text
prob 还是 logit？
全局权重是多少？
```

而是：

```text
如何在旧类保守、新类激进、不同新类信任不同 seed 的条件下做结构化融合？
```

---

## 11. 阶段 A-v2：结构化推理融合方案

阶段 A-v2 不重新训练，只利用已有三个 checkpoint：

```text
seed1: baseline / 旧类稳定器
seed2: 新类偏强，sofa/train 更好
seed3: 新类偏强，pottedplant/sheep/tvmonitor 更好
```

目标是把 oracle 显示出的互补转化为实际 mIoU。

### 11.1 方案 A：old/new group-wise ensemble

#### 动机

当前结果显示：

```text
seed1 old mIoU 最好
seed2/seed3 new mIoU 更好
```

所以不应对所有类别使用同一组权重。更合理的是：

```text
旧类类别 0-15：偏向 seed1
新类类别 16-20：偏向 seed2/seed3
```

#### 数学表达

对类别 c：

$$
s_c(x)=\sum_k w_{c,k}s^{(k)}_c(x)
$$

其中：

$$
w_{c,k} =
\begin{cases}
w^{old}_k, & c \in old \\
w^{new}_k, & c \in new
\end{cases}
$$

当前第一版使用：

```text
old classes: [0.7, 0.15, 0.15]
new classes: [0.1, 0.45, 0.45]
```

含义：

```text
旧类尽量保留 baseline 稳定性；
新类更多相信 seed2/seed3 的 RHL 子空间。
```

#### 预期

如果 old/new 冲突是主要瓶颈，应看到：

```text
old mIoU 接近 baseline/K3 equal
new mIoU 接近 K2 seed2+seed3
all mIoU 高于全局加权 K3
```

### 11.2 方案 B：class-wise ensemble

#### 动机

新类内部最优 seed 不一致：

| 类别 | 当前最好成员 |
|---|---|
| pottedplant 16 | seed3 |
| sheep 17 | seed3 |
| sofa 18 | seed2 |
| train 19 | seed2 |
| tvmonitor 20 | seed3 |

old/new 分组仍然太粗。class-wise ensemble 进一步细化到每个类别。

#### 第一版探索权重

为了验证机制，第一版使用探索性权重：

```json
{
  "16": [0.1, 0.3, 0.6],
  "17": [0.1, 0.2, 0.7],
  "18": [0.1, 0.7, 0.2],
  "19": [0.1, 0.65, 0.25],
  "20": [0.1, 0.2, 0.7]
}
```

同时旧类使用：

```text
old classes: [0.7, 0.15, 0.15]
```

注意：这组权重来自 test 观察，只用于探索上限，不能直接作为最终论文调参结果。若效果明显，后续必须在 val split 上重新确定权重，再用 test 报最终结果。

#### 预期

如果类别级冲突是主要瓶颈，应看到：

```text
new mIoU 高于 old/new group-wise
sofa/train 接近 seed2
pottedplant/sheep/tvmonitor 接近 seed3
old mIoU 不明显掉
```

### 11.3 方案 C：confidence gated ensemble

#### 动机

全局加权和类别级权重仍然是静态规则。真正理想的是：

```text
seed1 很自信时，保留 seed1
seed1 不确定时，切换到 RHL-SE ensemble
```

这样可以保留 baseline 的旧类稳定性，同时在边界、小目标、新类区域利用 ensemble 互补。

#### 机制

定义某个像素的 top1-top2 margin：

$$
m(x)=s_{top1}(x)-s_{top2}(x)
$$

使用 seed1 作为保守 base，使用 `0.2/0.4/0.4` prob ensemble 作为 alternate：

```text
if margin(seed1) < threshold:
    use ensemble scores
else:
    use seed1 scores
```

可选约束：

```text
只有 ensemble margin > seed1 margin 时才切换。
```

当前第一版参数：

```text
gate_base_index=0
gate_margin_threshold=0.10
gate_require_ensemble_better=True
```

#### 预期

如果 seed1 的错误主要集中在低置信区域，门控应提高 new mIoU，同时尽量少伤 old mIoU。

### 11.4 阶段 A-v2 实验矩阵

| 实验 | 融合方式 | 参数 | 目的 |
|---|---|---|---|
| A-v2-1 | old/new group-wise | old `[0.7,0.15,0.15]`, new `[0.1,0.45,0.45]` | 验证旧新类分组能否解决 trade-off |
| A-v2-2 | class-wise | 旧类偏 seed1，新类按 best seed 分配 | 验证类别级互补能否转化为 mIoU |
| A-v2-3 | confidence gate | seed1 base, weighted ensemble alternate, margin 0.10 | 验证像素级不确定性门控 |

### 11.5 阶段 A-v2 判定标准

| 结果 | 下一步 |
|---|---|
| 任一结构化融合 all mIoU 明显超过 69.55 | 继续优化该融合，并考虑 val split 调参 |
| new mIoU 明显超过 42.5 且 old 不明显下降 | 作为 RHL-SE 2.0 候选 |
| 结构化融合只带来 0.01 级别变化 | 停止推理融合扫参，转向 BOA-RHL / PGH-RHL |
| confidence gate 切换比例很低 | 降低 threshold 或说明 seed1 margin 过高，门控难生效 |
| confidence gate 切换比例高但指标下降 | margin 不是可靠选择准则，需要学习式/val 驱动门控 |

### 11.6 阶段 A-v2 工作流

1. 实现代码接口：
   - `--old_class_weights`
   - `--new_class_weights`
   - `--class_weights_json`
   - `--gating_mode margin`
   - `--gate_base_index`
   - `--gate_margin_threshold`
   - `--gate_require_ensemble_better`

2. 用 `--max_batches 1` 做 smoke test。

3. 用 tmux 同时跑三组完整 test 推理。

4. 汇总与以下基线对比：
   - baseline
   - prob K3 `0.2/0.4/0.4`
   - prob K2 seed2+seed3
   - oracle upper bound

5. 根据结果决定：
   - 若结构化融合有效，转入 val split 正式调参；
   - 若无效，停止 RHL-SE 推理层优化，转入 RHL 特征构造升级。

### 11.7 阶段 A-v2 实验结果

三组结构化融合已完成：

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 | 原始基线 |
| prob K3 `0.2/0.4/0.4` | 69.5060 | 77.9848 | 42.3737 | 92.7235 | 78.3868 | 阶段 A 旧 best |
| old/new group-wise | 69.5033 | 78.0029 | 42.3043 | 92.7215 | 78.3795 | old 保住了，但 new 被压低 |
| class-wise v1 | **69.5229** | 78.0022 | 42.3893 | **92.7251** | **78.4009** | 当前 best |
| confidence gate | 69.4977 | **78.0191** | 42.2294 | 92.7192 | 78.3642 | old 更稳，但 new 损失 |

新类逐类 IoU：

| 设置 | pottedplant 16 | sheep 17 | sofa 18 | train 19 | tvmonitor 20 |
|---|---:|---:|---:|---:|---:|
| baseline | 23.5886 | 57.9622 | 30.9147 | 69.9526 | 28.1192 |
| prob K3 `0.2/0.4/0.4` | 23.9828 | 57.9330 | 31.3108 | 70.3081 | 28.3340 |
| old/new group-wise | 23.6347 | 58.1589 | 31.2776 | 70.2630 | 28.1875 |
| class-wise v1 | 23.6018 | **58.3234** | **31.4027** | 70.2956 | 28.3228 |
| confidence gate | 23.7015 | 57.9839 | 31.0954 | 70.1771 | 28.1892 |

#### 结果解释

1. **class-wise v1 是当前最优结果**  
   相比 baseline：

   ```text
   all mIoU: +0.0623
   old mIoU: -0.0063
   new mIoU: +0.2818
   Mean Acc: +0.0633
   ```

   相比阶段 A 旧 best `prob K3 0.2/0.4/0.4`：

   ```text
   all mIoU: +0.0169
   old mIoU: +0.0174
   new mIoU: +0.0156
   ```

   提升仍然不大，但方向是正确的：类别级融合比全局权重更能利用 RHL seed 的局部互补。

2. **old/new group-wise 不够细**  
   它让 old mIoU 回到 `78.0029`，接近 baseline，但 new mIoU 从全局加权的 `42.3737` 降到 `42.3043`。  
   说明“所有新类一起偏 seed2/3”仍然太粗，因为新类内部最好 seed 不一致。

3. **confidence gate 第一版失败**  
   gate 切换比例约 `7.27%`，old mIoU 达到 `78.0191`，但 new mIoU 只有 `42.2294`。  
   说明当前 margin gate 更像是在保护 baseline，而不是有效捕获新类互补。可能原因：

   ```text
   seed1 在很多错误新类像素上仍然有较高 margin；
   ensemble margin > base margin 的约束过保守；
   top1-top2 margin 不是可靠的“该不该切换”指标。
   ```

4. **oracle 空间仍然没有被充分释放**  
   oracle all mIoU 为 `71.3950`，new mIoU 为 `46.2690`。当前 best class-wise 仍远低于 oracle。  
   这说明还有融合策略空间，但简单手工规则无法充分利用。

#### 阶段 A-v2 更新结论

阶段 A-v2 的结论应改为：

```text
结构化融合 > 全局权重，但提升仍小。
class-wise 是当前最值得保留的 RHL-SE 2.0 方向。
confidence gate 第一版不成立，不建议继续直接调 threshold；除非改成 val 驱动或学习式门控。
```

后续优先级：

1. 用 val split 正式搜索 class-wise / old-new 权重，避免 test-set tuning。
2. 如果 val 驱动 class-wise 仍只提升 0.05 左右，就停止 RHL-SE 推理层优化。
3. 把 RHL-SE 作为辅助集成模块，转入 BOA-RHL / PGH-RHL 改变 RHL 特征构造。

---

## 12. 阶段 A-v3：val split 驱动 class-wise 权重搜索

### 12.1 为什么必须转向 val 驱动

阶段 A-v2 的 `class-wise v1` 是手工基于 test 结果构造的权重。它能说明“类别级融合方向有信号”，但不能作为正式方法结论，因为它存在 test-set tuning 风险。

阶段 A-v3 的目标是把 class-wise 权重选择改成：

```text
只在 val split 上选择权重
只把最终选中的 class-wise 权重拿到 test split 评估一次
```

这样可以回答一个关键问题：

```text
RHL-SE 的类别级互补是否能在独立验证集上被稳定发现？
```

如果 val 选出的权重在 test 上仍然提升，说明 RHL-SE 作为辅助模块是成立的；如果不能提升，说明 A-v2 的 class-wise 收益主要是 test 调参偶然性。

### 12.2 实现方式

新增脚本：

```text
tools/search_rhl_class_weights.py
```

核心流程：

```text
1. 加载 seed1/seed2/seed3 三个 step1 checkpoint
2. 在 val split 上分别评估每个成员的 per-class IoU
3. 构造一组候选 class-wise 权重矩阵
4. 在 val split 上评估每个候选融合策略
5. 选择 val objective 最优候选
6. 保存 class_weights_json
7. 用 eval_rhl_ensemble.py 在 test split 上做最终评估
```

本次 objective 使用：

```text
--objective all_miou
```

这意味着权重选择优先服务最终主指标 all mIoU，而不是只追新类。

候选集合包括：

| 候选类型 | 含义 |
|---|---|
| `equal_all_classes` | 所有类别等权 K3 |
| `global_0.2_0.4_0.4` | 阶段 A 旧 best 全局权重 |
| `oldnew_*` | 旧类一套权重，新类一套权重 |
| `classwise_valbest_all_s*` | 每个类别选择 val 上最优成员，并用强度 `s` 加权 |
| `classwise_valbest_new_s*_oldstable` | 旧类稳定，只有新类做 class-wise |

最终被 val 选中的候选：

```text
classwise_valbest_all_s0.75
```

它的含义是：

```text
对每个类别，找 val split 上该类别 IoU 最高的成员；
该成员权重设为 0.75；
另外两个成员各给 0.125；
每个类别单独使用自己的权重行。
```

### 12.3 val split 选择结果

三成员在 val 上的指标：

| 成员 | val all mIoU | val old mIoU | val new mIoU |
|---|---:|---:|---:|
| `rhl_seed=1` | 43.5849 | 42.6350 | 46.6243 |
| `rhl_seed=2` | 43.8629 | 42.8776 | 47.0160 |
| `rhl_seed=3` | 43.8033 | 42.7724 | 47.1021 |
| val-selected class-wise | **43.9438** | **42.9179** | **47.2266** |

val 上最优候选：

```text
Best candidate: classwise_valbest_all_s0.75
Best score, all_mIoU: 43.9438
```

新类 val IoU：

| 类别 | val-selected IoU |
|---|---:|
| 16 pottedplant | 24.2556 |
| 17 sheep | 75.2405 |
| 18 sofa | 33.4099 |
| 19 train | 74.8973 |
| 20 tvmonitor | 28.3295 |

val 上选出的逐类最优成员为：

| 类别范围 | 选择规律 |
|---|---|
| old classes 0-15 | seed1/seed2/seed3 均有被选中，说明旧类也存在局部差异 |
| new classes 16-20 | 16/17/20 选 seed3，18/19 选 seed2 |

这与 A-v2 的 test 诊断一致：新类内部并不是统一偏向一个 seed，因此 old/new group-wise 太粗。

### 12.4 test split 最终结果

使用 val 选出的 class-wise 权重，在 test split 上评估结果如下：

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc |
|---|---:|---:|---:|---:|---:|
| baseline | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 |
| prob K3 `0.2/0.4/0.4` | 69.5060 | 77.9848 | 42.3737 | 92.7235 | 78.3868 |
| class-wise v1, test 手工 | 69.5229 | 78.0022 | 42.3893 | 92.7251 | 78.4009 |
| **val-driven class-wise** | **69.5379** | 77.9804 | **42.5218** | **92.7283** | **78.4391** |

相对 baseline：

```text
all mIoU: +0.0773
old mIoU: -0.0281
new mIoU: +0.4143
Mean Acc: +0.1015
```

相对阶段 A 旧 best `prob K3 0.2/0.4/0.4`：

```text
all mIoU: +0.0319
old mIoU: -0.0044
new mIoU: +0.1481
Mean Acc: +0.0523
```

相对 test 手工 `class-wise v1`：

```text
all mIoU: +0.0150
old mIoU: -0.0218
new mIoU: +0.1325
Mean Acc: +0.0382
```

新类 test IoU：

| 设置 | pottedplant 16 | sheep 17 | sofa 18 | train 19 | tvmonitor 20 |
|---|---:|---:|---:|---:|---:|
| baseline | 23.5886 | 57.9622 | 30.9147 | 69.9526 | 28.1192 |
| class-wise v1, test 手工 | 23.6018 | 58.3234 | 31.4027 | 70.2956 | 28.3228 |
| **val-driven class-wise** | **24.0709** | **58.0609** | **31.7575** | **70.4123** | **28.3077** |

### 12.5 结果解释

1. **val 驱动 class-wise 不是偶然无效，方向成立**  
   它在没有使用 test 选权重的情况下超过了 A-v2 的手工 class-wise：

   ```text
   69.5379 > 69.5229
   ```

   这说明类别级 RHL 互补不是纯 test 调参假象。

2. **提升主要来自新类，不来自旧类**  
   new mIoU 相对 baseline 提升 `+0.4143`，但 old mIoU 下降 `-0.0281`。  
   这与之前成员结果一致：seed2/3 更偏新类，seed1/baseline 更稳旧类。

3. **all mIoU 仍然没有达到强方法标准**  
   all mIoU 只提升 `+0.0773`。在 VOC 15-5 中，旧类占 16/21，新类占 5/21，所以新类 `+0.4143` 被旧类轻微下降稀释后，最终 all mIoU 仍然较小。

4. **RHL-SE 的真实定位应收敛**  
   现在已有充分证据说明：

   ```text
   RHL-SE 有稳定弱收益；
   val-driven class-wise 是更正式、更干净的版本；
   但仅靠推理融合很难释放 oracle 上限；
   继续手工扫权重的边际价值很低。
   ```

### 12.6 阶段 A 最终结论

RHL-SE 方案一可以保留，但定位应从“主要创新”调整为：

```text
辅助型 RHL 子空间集成模块
用于结项中的集成系统支撑
用于论文消融中的轻量增强项
```

不建议继续做：

```text
继续手工扫 test 权重
继续补 seed4/5
继续调 margin gate threshold
```

建议转入：

```text
BOA-RHL：改变 RHL 随机特征构造
PGH-RHL：给 RHL 加类别原型语义锚点
```

如果后续 BOA-RHL 或 PGH-RHL 单模型有效，RHL-SE 可以作为最后一层辅助集成再叠加，而不是继续单独挖推理权重。
