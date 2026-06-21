# 自适应 pseudo-label 阈值：原理、作用、代码方案

## 1.1 伪标签在 CFSSeg 里解决什么问题？

在 disjoint / overlapped 设置中，当前 step 的训练标签只标新类，旧类像素经常被标成 background。这个现象叫 semantic drift（语义漂移）。

例如 step t 的图像里有旧类 “person”，但当前标注只关心新类 “car”，于是 person 像素被标成 background。模型如果直接用这些标签更新，会把旧类往 background 推。

CFSSeg 的做法是：

```text
用上一步模型 q_{t-1} 预测当前图像；
如果某个 background 像素被旧模型高置信度预测为旧类；
则把它从 background 改成旧类 pseudo-label。
```

代码报告中当前实现大致是：对旧模型输出做 sigmoid，取最大分数和类别，然后用 `pred_scores >= pseudo_label_confidence` 判断是否替换 background label。

当前逻辑类似：

```python
outputs = self.model_prev(images)
outputs = torch.sigmoid(outputs)
pred_scores, pred_labels = torch.max(outputs, dim=1)

pseudo_labels = torch.where(
    (labels == 0) & (pred_labels > 0) &
    (pred_scores >= self.opts.pseudo_label_confidence),
    pred_labels,
    labels
)
```

---

## 1.2 固定阈值的问题

固定阈值比如 `0.7` 有明显局限：

|情况|固定阈值太高|固定阈值太低|
|---|---|---|
|旧类模型不自信|几乎没有 pseudo-label，旧类仍被当 background|可能还能救回一些旧类|
|旧类模型过度自信|影响不大|大量错误 pseudo-label 污染闭式解|
|类别难度不同|难类召回很低|易类噪声增加|
|step 越往后|置信度分布可能漂移|同一个阈值不再合适|

对 CFSSeg 尤其敏感，因为增量阶段不是多轮 SGD 慢慢纠错，而是闭式解一次性吸收标签。如果 pseudo-label 噪声很大，错误会直接进入 (E^\top Y)。

---

## 1.3 自适应阈值的核心思路

把固定阈值：

$$ 
s_i \ge \tau  
$$

改成动态阈值：

$$  
s_i \ge \tau_{t,c}  
$$

其中：

- (s_i)：旧模型对第 (i) 个像素/点的置信度；
    
- (t)：当前增量 step；
    
- (c)：旧模型预测类别；
    
- (\tau_{t,c})：当前 step、当前类别的自适应阈值。
    

推荐第一版使用 **class-wise quantile threshold（按类别分位数阈值）**：

 $$
\tau_{t,c}

\operatorname{clip}  
\left(  
Q_q({s_i \mid \hat{y}_i=c,\ y_i=c_b}),  
\tau_{\min},  
\tau_{\max}  
\right)  
$$

解释：

- 只看当前被标成 background 的像素；
    
- 只看旧模型预测为旧类 (c) 的候选像素；
    
- 取这些置信度的 (q) 分位数；
    
- 例如 (q=0.7)，表示阈值取 70% 分位点，最终保留 top 30% 的高置信 pseudo-label；
    
- 用 `clip` 限制阈值范围，避免太低或太高。
    

然后：

$$  
\tilde{y}_i =  
\begin{cases}  
\hat{y}^{prev}_i, & y_i=c_b,\ \hat{y}^{prev}_i=c,\ s_i \ge \tau_{t,c} \  
y_i, & \text{otherwise}  
\end{cases}  
$$

这比固定 `0.7` 更合理，因为不同类别有不同置信度分布。

---

## 1.4 最小代码方案：batch 内 class-wise adaptive threshold

这是最容易接入的版本，不需要额外跑一遍数据。

### 第一步：在 `utils/parser.py` 增加参数

```python
parser.add_argument(
    "--pseudo_label_strategy",
    type=str,
    default="fixed",
    choices=["fixed", "batch_quantile", "class_quantile"],
    help="Pseudo-label threshold strategy."
)

parser.add_argument(
    "--pseudo_label_quantile",
    type=float,
    default=0.7,
    help="Quantile for adaptive pseudo-label threshold. 0.7 keeps roughly top 30% candidates."
)

parser.add_argument(
    "--pseudo_label_min_conf",
    type=float,
    default=0.5,
    help="Lower bound of adaptive pseudo-label threshold."
)

parser.add_argument(
    "--pseudo_label_max_conf",
    type=float,
    default=0.95,
    help="Upper bound of adaptive pseudo-label threshold."
)

parser.add_argument(
    "--pseudo_label_min_pixels",
    type=int,
    default=64,
    help="Minimum candidate pixels for computing adaptive threshold."
)
```

---

### 第二步：在 `trainer/trainer.py` 增加阈值函数

```python
def _get_adaptive_threshold_map(self, pred_scores, pred_labels, labels):
    """
    pred_scores: [B, H, W], old model max confidence
    pred_labels: [B, H, W], old model predicted class id
    labels:      [B, H, W], current ground-truth labels

    return:
        threshold_map: [B, H, W]
    """
    opts = self.opts

    threshold_map = torch.full_like(
        pred_scores,
        fill_value=float(opts.pseudo_label_confidence),
    )

    candidate = (labels == 0) & (pred_labels > 0)

    if opts.pseudo_label_strategy == "fixed":
        return threshold_map

    if opts.pseudo_label_strategy == "batch_quantile":
        vals = pred_scores[candidate]
        if vals.numel() >= opts.pseudo_label_min_pixels:
            tau = torch.quantile(vals.float(), opts.pseudo_label_quantile)
            tau = tau.clamp(
                min=opts.pseudo_label_min_conf,
                max=opts.pseudo_label_max_conf,
            )
            threshold_map[candidate] = tau
        return threshold_map

    if opts.pseudo_label_strategy == "class_quantile":
        old_classes = torch.unique(pred_labels[candidate])

        for c in old_classes:
            if int(c.item()) <= 0:
                continue

            class_mask = candidate & (pred_labels == c)
            vals = pred_scores[class_mask]

            if vals.numel() < opts.pseudo_label_min_pixels:
                continue

            tau_c = torch.quantile(vals.float(), opts.pseudo_label_quantile)
            tau_c = tau_c.clamp(
                min=opts.pseudo_label_min_conf,
                max=opts.pseudo_label_max_conf,
            )

            threshold_map[class_mask] = tau_c

        return threshold_map

    raise ValueError(f"Unknown pseudo_label_strategy: {opts.pseudo_label_strategy}")
```

---

### 第三步：替换 `get_pseudo_labels`

把原来的：

```python
pseudo_labels = torch.where(
    (labels == 0) & (pred_labels > 0) &
    (pred_scores >= self.opts.pseudo_label_confidence),
    pred_labels,
    labels
)
```

改成：

```python
threshold_map = self._get_adaptive_threshold_map(
    pred_scores=pred_scores,
    pred_labels=pred_labels,
    labels=labels,
)

pseudo_mask = (
    (labels == 0) &
    (pred_labels > 0) &
    (pred_scores >= threshold_map)
)

pseudo_labels = torch.where(
    pseudo_mask,
    pred_labels,
    labels
)
```

建议同时加日志：

```python
if self.opts.local_rank == 0:
    ratio = pseudo_mask.float().mean().item()
    mean_conf = pred_scores[pseudo_mask].mean().item() if pseudo_mask.any() else 0.0
    print(f"[PseudoLabel] ratio={ratio:.6f}, mean_conf={mean_conf:.4f}")
```

---

## 1.5 运行命令

固定阈值 baseline：

```bash
python train.py \
  ... \
  --setting overlap \
  --use_pseudo_label \
  --pseudo_label_confidence 0.7 \
  --pseudo_label_strategy fixed
```

自适应阈值：

```bash
python train.py \
  ... \
  --setting overlap \
  --use_pseudo_label \
  --pseudo_label_strategy class_quantile \
  --pseudo_label_quantile 0.7 \
  --pseudo_label_min_conf 0.5 \
  --pseudo_label_max_conf 0.95 \
  --pseudo_label_min_pixels 64
```

建议网格：

|实验|strategy|quantile|min/max|
|---|---|--:|---|
|fixed-0.6|fixed|-|0.6|
|fixed-0.7|fixed|-|0.7|
|fixed-0.8|fixed|-|0.8|
|global-adaptive|batch_quantile|0.7|0.5 / 0.95|
|class-adaptive|class_quantile|0.7|0.5 / 0.95|
|class-adaptive-strict|class_quantile|0.8|0.5 / 0.95|

注意：`q=0.8` 更保守，保留 top 20%；`q=0.6` 更激进，保留 top 40%。

# RHL归一化和自适应伪标签阈值：哪个更适合先做？

优先级建议：

```
第一优先：RHL output normalization 第二优先：adaptive pseudo-label threshold第三优先：gamma / buffer 联合搜索第四优先：BCE + Dice第五优先：换 backbone，比如 DINOv3
```

原因：

1. **RHL normalization** 改动小、理论故事干净、不依赖 setting，sequential / disjoint / overlap 都能测。
2. **adaptive pseudo-label** 对 disjoint / overlap 更有意义；如果你只跑 sequential，它基本不是核心矛盾。
3. **换 backbone** 可能提升最大，但工程成本高、审稿解释难、容易变成“强 backbone 论文”。
4. **loss / LR** 主要作用 step0，不是 CFSSeg 的闭式解创新核心。

---

## 推荐实验路线

你可以按这个顺序推进：

### 阶段 A：确认 baseline

```
VOC 15-1 sequentialResNet101 + DeepLabV3bce_lossrhl_norm=nonepseudo-label off
```

记录：

```
old mIoUnew mIoUall mIoU每个 step 的日志是否 NaN运行时间
```

### 阶段 B：RHL normalization

```
VOC 15-1 sequentialrhl_norm = none / l2 / l2_sqrt / layernormgamma = 0.1 / 1 / 10
```

看：

```
old 是否更稳new 是否下降all 是否提升矩阵数值错误是否减少
```

### 阶段 C：pseudo-label adaptive threshold

```
VOC 10-1 overlap 或 disjointfixed threshold: 0.6 / 0.7 / 0.8adaptive: batch_quantile / class_quantile
```

看：

```
pseudo-label ratiopseudo-label mean confidenceold mIoUnew mIoUall mIoU可视化 pseudo-mask
```

### 阶段 D：再考虑 BCE + Dice

```
BCEBCE + 0.5 DiceBCE + 1.0 Dice
```

只要记住：这主要是 step0 表征质量优化，不是 C-RLS 的核心贡献。

### 阶段 E：最后考虑 DINOv3

这个阶段应该作为一个更大的方法故事：

```
frozen foundation dense representation+ analytic random feature lift+ recursive closed-form segmentation head
```

不要和 RHL / pseudo-label 小实验混在一起。否则结果虽然可能好，但论文贡献会变得不清楚。