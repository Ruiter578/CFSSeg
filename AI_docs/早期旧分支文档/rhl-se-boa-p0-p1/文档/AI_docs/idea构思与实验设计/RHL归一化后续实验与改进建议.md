# RHL 归一化后续实验与改进建议

> 项目：`/root/2TStorage/lyc/SegACIL`  
> 日期：2026-06-11  
> 依据：`RHL归一化专题报告.md`、`Codex_Plans/PLAN.md`、当前 VOC `15-5` sequential step1 实验结果  
> 目标：判断 RHL 归一化是否还值得继续训练，以及如果继续，应如何做才有信息增量。

## 1. 当前判断

结论先行：

```text
不建议继续做同样的 l2_sqrt gamma=0.1/1/10 扩展训练。
```

原因：

1. 第二组 `l2_sqrt` 的 `gamma=0.1/1/10` 已全部完成。
2. 三个 gamma 的结果几乎完全相同。
3. `l2_sqrt_bs64` 低于 `none_bs64` baseline。
4. 继续在 `0.1-10` 附近加密扫描，大概率只会重复相同结论。

当前更合理的策略是：

```text
RHL 归一化不作为主贡献继续堆实验；
只做少量“收尾验证”或“改造成新方法”的定向实验。
```

---

## 2. 已完成实验给出的信号

### 2.1 batch size 64 对齐对照

| 设置 | all mIoU | old mIoU | new mIoU | 相对 none_bs64 |
|---|---:|---:|---:|---|
| `none, gamma=1, bs64` | 69.461 | 78.008 | 42.107 | baseline |
| `l2, gamma=1, bs64` | 69.451 | 77.939 | 42.287 | all 略低，新类小涨 |
| `layernorm, gamma=1, bs64` | 69.415 | 77.957 | 42.079 | 略低 |
| `l2_sqrt, gamma=1, bs64` | 69.304 | 77.943 | 41.657 | 明确低于 baseline |

### 2.2 `l2_sqrt` gamma sweep

| 设置 | all mIoU | old mIoU | new mIoU |
|---|---:|---:|---:|
| `l2_sqrt, gamma=0.1, bs64` | 69.304 | 77.943 | 41.657 |
| `l2_sqrt, gamma=1, bs64` | 69.304 | 77.943 | 41.657 |
| `l2_sqrt, gamma=10, bs64` | 69.304 | 77.943 | 41.657 |

这说明：在当前 `l2_sqrt` 特征尺度下，`gamma=0.1/1/10` 对最终解几乎没有影响。

### 2.3 一个未完全解释的现象

`l2_sqrt, gamma=1, bs32` 的一次 retry 结果较好：

```text
all mIoU: 69.515
new mIoU: 43.080
```

但它和主要 baseline `none_bs64` 的 batch size 不一致。因此它只能作为“有现象值得解释”，不能直接作为“RHL 归一化有效”的证据。

---

## 3. 是否还需要进一步训练？

按优先级分三类。

### 3.1 不建议继续做的实验

| 实验 | 不建议原因 |
|---|---|
| `l2_sqrt gamma=0.3/3/5` | 已有 `0.1/1/10` 几乎完全一样，加密扫描信息量低 |
| 继续只换 `rhl_norm_eps` | 当前没有除零、NaN、Inf，eps 不是主要矛盾 |
| 继续跑更多 `layernorm` | 第一组已经低于 baseline，没有明确线索 |
| 直接把 `l2_sqrt` 写成主模块 | batch size 64 对照不支持这个结论 |

### 3.2 可选的最小收尾实验

如果需要把 RHL 归一化这条线彻底收口，建议只补一个最小控制实验：

```text
none, gamma=1, batch size=32
```

目的：判断 `l2_sqrt_bs32_retry` 的新类提升到底来自 `l2_sqrt`，还是来自 batch size / 数值路径。

推荐命令：

```bash
cd /root/2TStorage/lyc/SegACIL
tmux new-session -d -s rhl_bs32_closeout \
  'source /home/linyichen/miniconda3/etc/profile.d/conda.sh && \
   conda activate segacil && \
   DEFAULT_BATCH_SIZE=32 \
   SUBPATH=20260611_rhl_none_g1_bs32_closeout \
   BASE_SUBPATH=20260606 \
   RHL_NORM=none \
   GAMMA=1 \
   RHL_STATS=1 \
   bash ./run_rhl_norm.sh 2>&1 | tee -a logs/rhl_norm/20260611_rhl_none_g1_bs32_closeout.log'
```

判定：

| 结果 | 解释 |
|---|---|
| `none_bs32` 也接近或超过 43.0 new mIoU | `l2_sqrt_bs32` 的提升主要不是归一化贡献 |
| `none_bs32` 明显低于 `l2_sqrt_bs32` | 可以考虑再重复一次 `l2_sqrt_bs32` 做稳定性验证 |

是否必须做：**不是必须**。如果时间紧，建议直接转向方案二/伪标签方向。

### 3.3 如果要把 RHL 继续改成一个新方法

当前“强制固定范数”的版本没有收益，但 RHL 方向仍有改造空间。更有潜力的不是继续调 `gamma`，而是让归一化保留一部分幅值信息，或者把解析更新改成类别感知。

---

## 4. 可改进方向一：部分归一化，而不是完全抹掉范数

### 4.1 动机

`none` 的 RHL row norm 有明显波动：

```text
mean 约 28-30，std 约 5-6，max 可到 55-67
```

这说明原始 RHL 特征幅值差异确实存在。但实验表明，把所有像素强行压到同一个范数并不一定更好。可能原因是：范数本身包含“这个像素有多像目标/特征有多强”的信息。

因此更合理的是做“部分归一化”：

```text
不要完全删除幅值信息，只削弱过大的尺度波动。
```

### 4.2 数学形式

设原始 RHL 输出为 $e_i$，维度为 $d$。可以定义：

$$
\tilde{e}_i =
e_i \cdot
\left(
\frac{\sqrt{d}}{\|e_i\|_2+\epsilon}
\right)^\beta
$$

其中：

- $\beta=0$：等价于 `none`；
- $\beta=1$：等价于 `l2_sqrt`；
- $0<\beta<1$：部分归一化。

推荐先试：

```text
beta = 0.25
beta = 0.50
```

直觉：

```text
none       : 完全保留范数，可能有尺度噪声
l2_sqrt    : 完全抹平范数，可能丢失强弱信号
partial    : 保留一部分强弱信号，同时减轻极端尺度
```

### 4.3 实验价值

如果 partial norm 提升，论文故事会比当前 `l2_sqrt` 更合理：

```text
不是简单做归一化，而是为解析持续分割设计“幅值保持的随机特征尺度校准”。
```

优先级：中等。需要新增代码参数，例如：

```text
--rhl_norm power
--rhl_norm_power 0.5
```

---

## 5. 可改进方向二：类别感知的解析更新

### 5.1 动机

VOC `15-5` 的难点不只是数值尺度，而是新旧类样本不平衡、旧类语义漂移和背景混淆。单纯 row-wise normalization 不知道“这个像素属于哪个类”，所以它解决不了类别层面的偏置。

更贴近问题的是 weighted ridge regression：

$$
\min_{\Phi} \|\Omega^{1/2}(E\Phi-Y)\|_F^2 + \gamma\|\Phi\|_F^2
$$

闭式解：

$$
\hat{\Phi}
=
(E^\top \Omega E+\gamma I)^{-1}E^\top \Omega Y
$$

其中 $\Omega$ 是像素权重矩阵，可以按类别频率设置。

### 5.2 直觉解释

普通解析更新把每个像素近似等权处理。分割任务中背景像素、旧类像素、新类像素数量差异很大，等权可能让多数类支配闭式解。

类别感知解析更新可以让少数新类像素在闭式解里有更大权重。

### 5.3 实验建议

第一版不要复杂，先做 class-frequency inverse weighting：

```text
pixel_weight = 1 / sqrt(class_pixel_count + eps)
```

优先只在 `curr_step=1` 的新类训练集上启用，对 old/new mIoU 做对照。

优先级：较高。它比 RHL row norm 更直接对应 CFSSeg 的增量分割痛点，也更容易形成方法贡献。

---

## 6. 可改进方向三：数值稳定性优化，不作为主精度模块

当前日志中曾出现过一次 `cusolver error` 和一次受外部进程影响的 OOM，但正式完成结果没有 NaN/Inf。

如果目标是提高工程稳定性，可以考虑：

1. 用 `torch.linalg.solve` 替代显式 `torch.inverse`。
2. 对矩阵 $S$ 做轻微 diagonal jitter：

   $$
   S = R^{-1} + X^\top X + \lambda I
   $$

3. 在 `torch.inverse` 失败时 fallback 到 CPU 或更稳定的 linalg backend。

但这类改动主要解决稳定性，不一定提升 mIoU，不适合作为当前论文主贡献。

优先级：低到中。除非后续大规模实验频繁出现 inverse / cusolver 报错。

---

## 7. 可改进方向四：与伪标签方案联动

当前 RHL 归一化没有直接解决旧类漂移。CFSSeg 的关键问题之一是增量阶段旧类像素可能被标成 background，导致旧类知识被冲淡。

因此，下一步更建议推进：

```text
自适应伪标签阈值 / 类别感知伪标签筛选
```

RHL 归一化可以作为辅助：

1. 保留 `--rhl_norm` 参数，默认 `none`。
2. 在伪标签实验中只跑一个安全对照，例如 `none` vs `l2`。
3. 不要把 RHL 与伪标签同时大范围网格搜索，避免实验爆炸。

优先级：高。更符合 CFSSeg 原始方法痛点和论文增量空间。

---

## 8. 推荐后续路线

### 路线 A：时间紧，优先产出论文可用方法

推荐：

```text
停止 RHL 归一化主线训练
转向方案二：自适应伪标签阈值
RHL 归一化作为消融和失败分析保留
```

理由：

1. 当前 RHL 归一化没有可靠提升。
2. 伪标签更贴近旧类遗忘/语义漂移问题。
3. 时间有限，不应在低收益方向继续消耗 GPU。

### 路线 B：想把 RHL 线条收口得更严谨

只补一个实验：

```text
none, gamma=1, batch size=32
```

目的：解释 `l2_sqrt_bs32_retry` 的正向现象。

如果 `none_bs32` 也高，那么 RHL 归一化线可以明确停止。

### 路线 C：想把 RHL 改造成真正新模块

按顺序做：

1. 实现 `power_norm`，测试 `beta=0.25/0.5`。
2. 如果仍无收益，停止 RHL normalization。
3. 转向 weighted C-RLS / class-balanced analytic update。

---

## 9. 建议实验矩阵

### 9.1 最小收尾矩阵

| 优先级 | 实验 | 目的 |
|---:|---|---|
| P0 | `none_g1_bs32_closeout` | 判断 bs32 正向现象是否来自 batch size |

### 9.2 RHL 改进矩阵

只有在决定继续 RHL 方向时才做：

| 优先级 | 实验 | 参数 | 目的 |
|---:|---|---|---|
| P1 | `power_norm_beta0p25` | `beta=0.25, gamma=1, bs64` | 弱化尺度波动，保留大部分幅值 |
| P1 | `power_norm_beta0p5` | `beta=0.5, gamma=1, bs64` | 折中版部分归一化 |
| P2 | `l2_gamma_large` | `l2, gamma=100/1000` | 只在想理解 gamma 尺度时做，不优先 |

### 9.3 更有论文价值的矩阵

| 优先级 | 实验 | 目的 |
|---:|---|---|
| P0 | 自适应伪标签阈值 baseline 对照 | 验证是否改善旧类保持和新类学习 |
| P1 | 类别感知伪标签阈值 | 解决不同类别置信度分布不同的问题 |
| P1 | weighted C-RLS | 解析更新中处理类别不平衡 |
| P2 | RHL partial norm + 伪标签 | 只在单独模块有效后组合 |

---

## 10. 最终建议

当前 RHL 归一化的最可靠结论是：

```text
RHL 特征尺度控制是可实现的，但不是当前 VOC 15-5 性能瓶颈。
```

因此建议：

1. 不再继续 `l2_sqrt` 常规 gamma sweep。
2. 如需严谨收尾，只补 `none_bs32` 一个控制实验。
3. 不把 RHL normalization 作为主线贡献。
4. 将主要精力转向伪标签、自适应阈值、类别不平衡解析更新。
5. 如果继续 RHL，只尝试 partial norm 或 weighted C-RLS 这类有明确机制变化的版本。

