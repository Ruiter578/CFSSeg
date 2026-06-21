# RHL 归一化新方案：幅值保留的部分归一化与类别感知解析更新

> 日期：2026-06-20
> 状态：现有静态归一化已验证；新方案待实现
> 方法定位：控制随机/原型隐空间的尺度与解析更新中的类别不平衡

## 0. 核心结论

现有 `none/l2/l2_sqrt/layernorm` 已完成代码和实验。结果表明：完全 L2、LayerNorm 和 `l2_sqrt` 没有稳定提升，`l2_sqrt` 的 gamma `0.1/1/10` 也几乎不改变结论。该结果否定的是“用固定强归一化直接替换原始幅值”的做法，不等于尺度控制和类别感知解析学习没有价值。

新方案由两个可独立启用、可组合验证的轴组成：

1. **幅值保留的部分归一化（PowerNorm-RHL）**：连续控制保留多少原始范数，而不是在 `none` 和完全归一化之间二选一；
2. **类别感知加权 C-RLS（CA-C-RLS）**：在解析更新目标中提高稀缺新类的有效权重，它属于目标函数改造，不冒充“归一化”。

两者共同解决原方案暴露出的两个问题：强归一化可能抹掉有判别价值的幅值信息，而普通 C-RLS 又可能被背景和旧类像素主导。

## 1. 已有实现与实验结论

### 1.1 当前开关

当前代码支持：

```text
--rhl_norm none|l2|l2_sqrt|layernorm
--rhl_norm_eps FLOAT
```

主要实现位于 `network/Buffer.py`，参数由 `utils/parser.py` 进入 `trainer/trainer.py`，checkpoint metadata 已补充相关配置。`run_rhl_norm.sh` 是现有专用 runner。

### 1.2 已有结果

Batch64 对齐实验：

| norm | all mIoU | new mIoU | 判断 |
|---|---:|---:|---|
| none | 69.461 | 42.107 | 对齐基线 |
| l2 | 69.451 | 42.287 | new 微升、all 基本持平 |
| layernorm | 69.415 | 42.079 | 无收益 |
| l2_sqrt | 69.304 | 41.657 | 退化 |

`l2_sqrt` 的 gamma `0.1/1/10` 基本不变。另一个 Batch32 重跑得到 all `69.515`、new `43.080`，但它同时改变了训练 Batch Size，不能归因为 normalization。

因此，新方案不再重复静态 `l2_sqrt + gamma` 大扫参，而是显式分离“特征尺度”“有效正则”“类别权重”。

## 2. 为什么固定归一化没有解决问题

设随机隐特征为 `h(x)`。其范数可能同时包含：

- 输入 feature 强度；
- ReLU 激活稀疏度；
- 随机矩阵行范数；
- 类别/边界难度；
- 空间位置和 backbone 响应幅值。

完全 L2 把所有像素投影到单位球面，LayerNorm 则消除通道均值并固定方差。这些变换提升数值尺度一致性，但也可能删除解析头本来可以利用的样本难度和激活幅值。

另一方面，即使每个像素的隐特征被良好归一化，训练样本中背景/旧类数量远多于新类时，普通最小二乘目标仍由多数类主导。该问题不能只靠 feature normalization 解决。

## 3. 轴一：PowerNorm-RHL

### 3.1 数学形式

令隐特征维度为 `d`，定义：

```text
scale(x) = sqrt(d) / (||h(x)||_2 + eps)
h_tilde(x) = h(x) * scale(x)^beta
```

其中：

- `beta=0`：严格等价于 `none`；
- `beta=1`：等价于 `l2_sqrt`；
- `0 < beta < 1`：部分压缩范数差异，同时保留幅值排序。

为了避免低范数像素被过度放大：

```text
scale_clipped = clip(scale, scale_min, scale_max)
```

第一轮建议：

```text
beta = 0.25, 0.5, 0.75
scale_min = 0.25
scale_max = 4.0
```

`beta=0/1` 作为两端锚点一并报告，而不是重新搜索已知无效的 gamma 组合。

### 3.2 归一化作用域

未来 BOA+PGH 会形成多个 feature branch。必须支持：

- `random_only`：只对随机分支归一化；
- `prototype_only`：只对原型分支归一化；
- `per_branch`：各分支独立归一化后拼接；
- `joint`：拼接后统一归一化。

默认 `per_branch`，因为 random 与 prototype 的维度、分布和语义不同。每个作用域都写入配置，禁止在组合实验中用同一个 `rhl_norm=power` 含糊表示。

### 3.3 trace 与 gamma 匹配

特征尺度变化会改变 Ridge 的有效正则。对每个配置记录：

```text
trace_mean = mean(||h(x)||_2^2)
```

给定 baseline `trace_base` 与新配置 `trace_new`，加入匹配对照：

```text
gamma_matched = gamma_base * trace_new / trace_base
```

每个 beta 至少比较固定 gamma 与 trace-matched gamma。这样可以区分收益来自范数分布重塑，还是仅来自等价的正则缩放。

### 3.4 必须落盘的统计

- pre/post norm mean/std；
- `||h||` 的均值、标准差、P1/P50/P99；
- scale factor 的 P1/P50/P99 与 clipping ratio；
- feature trace；
- old/new/class-wise norm 分布；
- NaN/Inf count；
- RecursiveLinear condition proxy。

## 4. 轴二：CA-C-RLS

### 4.1 目标函数

普通 Ridge：

```text
min_W sum_i ||h_i W - y_i||^2 + gamma ||W||^2
```

类别感知加权版本：

```text
min_W sum_i a_i ||h_i W - y_i||^2 + gamma ||W||^2
```

通过：

```text
H_w = sqrt(a) * H
Y_w = sqrt(a) * Y
```

继续使用同一解析求解，不引入反向传播头。该变换必须同时作用于 `H` 和 `Y`，只缩放输入会改变问题定义并产生错误实现。

### 4.2 权重定义

建议支持三类策略：

1. `inverse_sqrt`：`w_c = 1 / sqrt(n_c + eps)`；
2. `effective_num`：`w_c = (1-rho) / (1-rho^n_c)`；
3. `old_new`：old/new 两组独立权重，先验证组级假设。

权重归一化为训练像素上的均值 1，并限制：

```text
w_min <= w_c <= w_max
```

第一轮用 `w_max=4`，避免稀有类或标签噪声控制解析更新。background 是否加权必须是显式参数。

### 4.3 类别计数协议

类别计数只能来自相应 step 的 train split，并使用与训练标签一致的 mapping/ignore_index。建议构建独立 counts artifact：

```json
{
  "dataset": "voc",
  "task": "15-5",
  "step": 1,
  "split": "train",
  "counts": {"0": 0, "1": 0},
  "ignore_index": 255,
  "mapping_hash": "...",
  "transform": "label-only deterministic scan"
}
```

不能用随机 crop 后的单次观察计数作为全局权重，也不能使用 val/test 计数。

### 4.4 多标签 BCE 语义

当前像素标签最终转为 one-hot/BCE target。样本权重 `a_i` 应由像素真实类别产生，并扩展到该像素的全部通道 loss/解析目标。ignore 像素权重为 0。实现测试应验证：

- `all weights=1` 逐元素复现现有 C-RLS；
- 同时缩放 H/Y；
- ignore 不参与统计和更新；
- 权重归一化后平均值为 1；
- 分 batch 更新与全 batch 解的一致性误差在容忍范围内。

## 5. 代码实现设计

### 5.1 文件与职责

| 文件 | 修改/新增职责 |
|---|---|
| `network/Buffer.py` | `power` normalization、scope、scale clipping 与统计 |
| `network/AnalyticLinear.py` | 接收 sample weights，正确构造 weighted update |
| `utils/parser.py` | PowerNorm 与 CA-C-RLS 参数 |
| `trainer/trainer.py` | 读取 counts、生成像素权重、持久化完整配置 |
| `utils/ckpt.py` | 保存 norm/weight 配置、artifact path/hash |
| `tools/build_rhl_class_counts.py` | 生成确定性 train class counts artifact |
| `tools/run_rhl_norm_v2.sh` | 独立轴与组合实验 runner |
| `tools/summarize_rhl_norm_v2.py` | 输出指标、trace、权重和稳定性对照 |
| `tests/test_rhl_normalization.py` | 端点兼容、clipping、scope、统计测试 |
| `tests/test_weighted_recursive_linear.py` | weighted C-RLS 数学正确性测试 |

### 5.2 CLI

```text
--rhl_norm none|l2|l2_sqrt|layernorm|power
--rhl_norm_power FLOAT
--rhl_norm_target sqrt_dim|unit|FLOAT
--rhl_norm_scope random_only|prototype_only|per_branch|joint
--rhl_norm_scale_clip_min FLOAT
--rhl_norm_scale_clip_max FLOAT

--rhl_sample_weight_mode none|inverse_sqrt|effective_num|old_new
--rhl_class_counts PATH
--rhl_weight_rho FLOAT
--rhl_weight_min FLOAT
--rhl_weight_max FLOAT
--rhl_weight_background FLOAT
```

所有默认值必须保持旧行为：`rhl_norm=none`、`rhl_sample_weight_mode=none`。

## 6. 实验矩阵

### 6.1 Axis A：PowerNorm 独立实验

| Case | beta | clipping | gamma |
|---|---:|---|---|
| A0 | 0 | off | 1 |
| A1 | 0.25 | `[0.25,4]` | 1 / trace-matched |
| A2 | 0.50 | `[0.25,4]` | 1 / trace-matched |
| A3 | 0.75 | `[0.25,4]` | 1 / trace-matched |
| A4 | 1.00 | `[0.25,4]` | 1 / trace-matched |

在最佳 beta 上再做 clipping `2/4/8` 上限消融，不与初始 beta 网格一次性笛卡尔积。

### 6.2 Axis B：CA-C-RLS 独立实验

| Case | weight mode | cap | 目的 |
|---|---|---:|---|
| B0 | none | 1 | baseline |
| B1 | old_new | 2/4 | 验证组级新类欠权重 |
| B2 | inverse_sqrt | 4 | 类别级权重 |
| B3 | effective_num | 4 | 平滑类别级权重 |
| B4 | 最佳模式 | 2/4/8 | cap 敏感性 |

### 6.3 两轴组合

在各自独立确认后执行 2x2：

| | CA-C-RLS off | CA-C-RLS on |
|---|---|---|
| PowerNorm off | baseline | weighted only |
| PowerNorm on | norm only | norm + weighted |

组合结果必须报告 interaction，不把两种机制合并成一个无法归因的“新归一化”。

### 6.4 与 BOA/PGH 的后续组合

- BOA 改随机基，PowerNorm 改随机基输出尺度，存在直接交互；先固定 BOA 最佳配置，再做 norm on/off。
- PGH 有随机/原型双分支，优先使用 `per_branch`，并报告每个 branch trace。
- CA-C-RLS 位于拼接后的解析求解，对 BOA/PGH 都可用，但必须在单方法已验证后组合。
- RHL-SE 2.0 位于输出端，可集成这些成员，不参与本方案的单成员因果判断。

## 7. 评价与判定

除 all/old/new/per-class mIoU 外，必须报告：

- 三个 RHL seed 的 mean/std；
- feature norm/trace/condition proxy；
- 每类权重与实际参与像素数；
- old/new 增益是否以旧类显著退化为代价；
- 训练时延、峰值显存、解析矩阵稳定性。

PowerNorm 的有效证据是：在匹配 gamma 后仍优于两端锚点，并且对 seed 稳定。CA-C-RLS 的有效证据是：new mIoU 稳定提升、old 类退化受控，且权重效应不依赖偶然的类别 crop 分布。

## 8. 执行顺序

1. 修正 runner 的默认对齐，建立 Batch32/Buffer8196/norm none 的冻结基线。
2. 实现 PowerNorm、端点兼容测试和 trace 统计。
3. 完成 beta、clipping、gamma-match 实验。
4. 构建 train class counts artifact，实现 weighted C-RLS 数学测试。
5. 完成 old/new、inverse-sqrt、effective-number 实验。
6. 执行两轴 2x2 组合并计算 interaction。
7. 在 BOA 最佳配置上验证 random branch PowerNorm。
8. 在 PGH 最佳配置上验证 per-branch PowerNorm 与 CA-C-RLS。

本方案的目标不是为旧 `l2_sqrt` 找一个更幸运的 gamma，而是把“保留多少幅值”和“哪些类别应在解析更新中获得更高权重”拆成两个可检验的机制，并通过严格的尺度匹配和因子实验获得可归因结论。
