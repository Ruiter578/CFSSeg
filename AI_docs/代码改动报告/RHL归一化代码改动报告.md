# RHL 归一化代码改动报告

> 日期：2026-06-09  
> 改动范围：`network/Buffer.py`、`trainer/trainer.py`、`utils/parser.py`、`run.sh`  
> 目标：落地“方案一：RHL 归一化”，在不改变 DeepLabV3 + ResNet101、C-RLS 主公式和默认 baseline 行为的前提下，增加可控的 RHL 输出归一化实验入口。

## 1. 改动结论

本次改动已经将 RHL 输出归一化接入到 SegACIL 的解析增量训练链路：

```text
DeepLabV3 frozen feature
  -> RandomBuffer random projection + ReLU
  -> optional RHL output normalization
  -> RecursiveLinear / C-RLS closed-form fitting
```

默认参数为：

```text
--rhl_norm none
--rhl_norm_eps 1e-6
```

因此不传新参数时，代码行为保持原始 CFSSeg / SegACIL baseline。归一化只改变进入解析头的固定特征表示，不引入可训练参数，不改变 C-RLS 更新公式。

## 2. 方法原理与代码对应

CFSSeg 原始 RHL 写法为：

$$
E = \operatorname{ReLU}(F\Phi_E)
$$

其中：

- $F$ 是冻结 encoder / DeepLab head_pre 输出的像素特征；
- $\Phi_E$ 是随机初始化且固定的高维映射矩阵；
- $E$ 是进入解析分类头的高维特征。

解析分类头的岭回归形式为：

$$
\hat{\Phi} = (E^\top E + \gamma I)^{-1}E^\top Y
$$

如果不同像素、batch 或 step 的 $E_i$ 范数波动很大，那么 $E^\top E$ 会隐式放大大范数样本的权重，也会改变 `gamma` 的有效正则强度。本次改动在 $E$ 进入 C-RLS 前增加：

$$
\tilde{E}_i = \operatorname{Norm}(E_i)
$$

代码落点是 `network/Buffer.py:RandomBuffer.forward()`。也就是说，归一化发生在随机映射和 ReLU 之后，发生在 `RecursiveLinear.fit()` 之前。

## 3. 新增归一化模式

| 参数值 | 数学含义 | 作用 |
|---|---|---|
| `none` | $\tilde{E}=E$ | baseline，完全保持原始行为 |
| `l2` | $\tilde{E}_i=E_i/(||E_i||_2+\epsilon)$ | 强制每个 pixel 的 RHL 特征为单位范数 |
| `l2_sqrt` | $\tilde{E}_i=\sqrt{d_E}E_i/(||E_i||_2+\epsilon)$ | 保持方向，同时使 row norm 接近 `sqrt(buffer)`，主候选 |
| `layernorm` | no-affine per-pixel layer norm | 不引入参数的中心化和尺度归一对照 |

主实验优先使用 `l2_sqrt`。原因是纯 `l2` 会把 row norm 压到 1，在 `buffer=8196` 时会显著缩小 $E^\top E$ 的尺度，使 `gamma=1` 变成相对更强的正则；`l2_sqrt` 则既消除样本间范数差异，又保留高维特征的总能量量级。

## 4. 具体代码改动

### 4.1 `utils/parser.py`

新增 3 个命令行参数：

```python
rhl_norm: str = 'none'
rhl_norm_eps: float = 1e-6
rhl_stats: bool = False
```

命令行入口：

```bash
--rhl_norm {none,l2,l2_sqrt,layernorm}
--rhl_norm_eps 1e-6
--rhl_stats
```

作用：

1. `rhl_norm` 控制是否启用 RHL 输出归一化。
2. `rhl_norm_eps` 防止除零或极小范数导致数值不稳定。
3. `rhl_stats` 只在前几个 batch 打印 row norm 和 NaN/Inf 状态，方便实验日志证明归一化确实生效。

### 4.2 `network/Buffer.py`

`RandomBuffer` 新增参数：

```python
rhl_norm: str = "none"
rhl_norm_eps: float = 1e-6
```

核心逻辑：

```python
Z = self.activation(super().forward(X))

if norm == "none":
    return Z
if norm == "l2":
    return F.normalize(Z, p=2, dim=-1, eps=eps)
if norm == "l2_sqrt":
    return F.normalize(Z, p=2, dim=-1, eps=eps) * math.sqrt(Z.shape[-1])
if norm == "layernorm":
    return F.layer_norm(Z, (Z.shape[-1],), weight=None, bias=None, eps=eps)
```

实现要点：

1. 保留 `register_buffer("weight", W)`，随机映射仍不是可训练参数。
2. 使用 `getattr(self, "rhl_norm", "none")` 和 `getattr(self, "rhl_norm_eps", 1e-6)`，兼容旧 AIR checkpoint。
3. `layernorm` 使用 functional no-affine 版本，不创建 `nn.LayerNorm` 参数，避免改变闭式解主线。

### 4.3 `trainer/trainer.py`

`AIR.__init__()` 新增：

```python
rhl_norm="none"
rhl_norm_eps=1e-6
rhl_stats=False
```

创建 RHL 时传入：

```python
self.buffer = RandomBuffer(
    backbone_output,
    buffer_size,
    rhl_norm=rhl_norm,
    rhl_norm_eps=rhl_norm_eps,
    **factory_kwargs,
)
```

在 step1 构造 AIR 时，从 `opts` 传入：

```python
rhl_norm=self.opts.rhl_norm
rhl_norm_eps=self.opts.rhl_norm_eps
rhl_stats=self.opts.rhl_stats
```

作用：使 `run.sh` 或命令行中的 RHL 参数真正进入 step1 的 analytic realignment 和 step1 closed-form fitting。

### 4.4 RHL stats 日志

`AIR.feature_expansion()` 中增加轻量统计：

```text
[RHL stats] mode=... eps=... mean=... std=... min=... max=... nan=False inf=False
```

只打印前 3 次，避免每个 batch 输出过多。统计的是归一化后的 RHL 输出 row norm，即实际进入 `RecursiveLinear.fit()` 的特征尺度。

### 4.5 `run.sh`

新增可由环境变量覆盖的实验参数：

```bash
GAMMA="${GAMMA:-1}"
RHL_NORM="${RHL_NORM:-none}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_STATS="${RHL_STATS:-0}"
```

训练命令追加：

```bash
--gamma "$GAMMA"
--rhl_norm "$RHL_NORM"
--rhl_norm_eps "$RHL_NORM_EPS"
```

当 `RHL_STATS=1` 时追加 `--rhl_stats`。这样后续实验可以直接用：

```bash
SUBPATH=20260609_rhl_l2sqrt_g1 BASE_SUBPATH=20260607 RHL_NORM=l2_sqrt RHL_STATS=1 bash run.sh
```

## 5. 验证结果

已完成验证：

1. `bash -n run.sh` 通过。
2. `python -m py_compile` 编译 `parser.py`、`Buffer.py`、`trainer.py` 通过。
3. `utils/parser.py --rhl_norm l2_sqrt --rhl_stats` 参数解析通过。
4. `trainer.trainer.AIR` 在 `segacil` 环境中导入通过。
5. `RandomBuffer` CPU/GPU smoke test 通过。

RHL smoke test 摘要：

```text
cpu none      mean_norm=37.535744 std_norm=1.573727
cpu l2        mean_norm=1.000000  std_norm=0.000000
cpu l2_sqrt   mean_norm=90.531754 std_norm=0.000014
cpu layernorm mean_norm=90.531349 std_norm=0.000199
gpu l2_sqrt   mean_norm=90.531762
```

其中 `sqrt(8196) ≈ 90.5318`，说明 `l2_sqrt` 实现符合预期。

## 6. CUDA / MPS 运行注意

当前机器上 `/tmp/nvidia-mps` 有其他用户启动的 MPS daemon，直接初始化 CUDA 会出现：

```text
MPS client failed to connect to the MPS control daemon
```

本次已单独启动当前用户的 MPS 目录：

```bash
mkdir -p /tmp/nvidia-mps-linyichen /tmp/nvidia-mps-log-linyichen
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-linyichen \
CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-linyichen \
nvidia-cuda-mps-control -d
```

后续 tmux 实验命令需要显式带上：

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-linyichen
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-linyichen
```

带上后 `torch.cuda.is_available()` 为 `True`。

## 7. 后续实验建议

当前 GPU 是 A100 80GB，启动前显存基本空闲。建议先同时跑：

| 实验 | 目的 |
|---|---|
| `20260609_rhl_none` | 新代码 baseline 等价性 |
| `20260609_rhl_l2sqrt_g1` | RHL 归一化主候选 |

如果同时运行后显存或算力压力过大，优先保留 `l2_sqrt` 主候选。

实验完成后，应记录：

1. old mIoU: `0 to 15 mIoU`
2. new mIoU: `16 to 20 mIoU`
3. all mIoU: `Mean IoU`
4. RHL stats 前几条日志
5. 是否出现 NaN、Inf、`torch.inverse` 或 CUDA solver 错误

## 8. 本次实验启动记录

本次实际运行时，`20260607` 的 step0 目录只有 `final.pth`，没有当前代码默认读取的命名 checkpoint：

```text
deeplabv3_resnet101_voc_15-5_step_0_sequential.pth
```

因此实验使用 `BASE_SUBPATH=20260606` 作为 step0 读取来源，新的 RHL 实验仍写入各自的 `SUBPATH`，避免覆盖旧结果。

并发尝试过 `none` 与 `l2_sqrt` 两种设置。两路同时运行时 A100 80GB 接近满显存，主实验的 C-RLS 求逆阶段曾出现：

```text
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
```

该错误发生在 `network/AnalyticLinear.py` 的 `torch.inverse(S)`，更像是大矩阵求逆在高显存压力下的 CUDA solver 稳定性问题，而不是 RHL 特征中出现 NaN/Inf；日志中的 RHL stats 均显示 `nan=False inf=False`。

按“显存不够则先跑主要设置”的原则，已停止并发 baseline，保留主候选单独运行：

```bash
SUBPATH=20260609_rhl_l2sqrt_g1_retry \
BASE_SUBPATH=20260606 \
RHL_NORM=l2_sqrt \
RHL_STATS=1 \
GAMMA=1 \
bash ./run.sh
```

日志位置：

```text
logs/rhl_norm/20260609_rhl_l2sqrt_g1_retry.log
```

tmux 会话：

```text
rhl_l2sqrt_g1
```

如果该单跑仍在同一求逆点失败，下一步优先启动 `l2_sqrt + gamma=10` 作为数值更稳的主候选备选；若通过，再补跑 `none + gamma=1` 做新代码 baseline 等价性对照。
