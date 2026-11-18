# PLAN: RHL 归一化落地、实验与验证

> 适用项目：`/root/2TStorage/lyc/SegACIL`  
> 生成日期：2026-06-09  
> 目标读者：下一轮执行代码改动的 Codex / Claude Code  
> 当前计划边界：只规划 RHL normalization，不实现 adaptive pseudo-label，不替换 backbone，不 commit，不 push。

## 1. 目标

在 CFSSeg / SegACIL 的 RHL 输出后加入无可训练参数的归一化选项，使进入 `RecursiveLinear.fit()` 的高维特征尺度更稳定，从而改善闭式解数值稳定性和 `15-5` 最终 mIoU。

必须满足：

1. 默认行为完全保持 baseline：`--rhl_norm none`。
2. 不改变 step0 DeepLabV3 + ResNet101 训练逻辑。
3. 不改变 C-RLS 主公式。
4. 不引入可训练参数。
5. 兼容旧 checkpoint，尤其是旧 AIR pickle 中没有新属性的情况。

## 2. 推荐设计

在 `RandomBuffer.forward()` 中做：

```text
input feature -> random linear -> ReLU -> normalization -> RecursiveLinear
```

支持模式：

| 模式 | 公式 | 用途 |
|---|---|---|
| `none` | `Z` | baseline，默认值 |
| `l2` | `Z / (||Z||_2 + eps)` | 强约束每个像素特征范数为 1 |
| `l2_sqrt` | `sqrt(D) * Z / (||Z||_2 + eps)` | 保持方向，同时让每个像素特征范数约为 `sqrt(buffer)`，建议作为主实验 |
| `layernorm` | per-pixel no-affine layer norm | 不引入参数的尺度与中心化对照 |

首选主实验不是纯 `l2`，而是 `l2_sqrt`。原因：原始 RHL 的 row norm 通常随 `buffer` 维度增长，纯 `l2` 会显著缩小 `E^T E` 的尺度，使 `gamma=1` 的有效正则变强；`l2_sqrt` 更接近保持原始能量级，同时消除样本间范数波动。

## 3. 代码改动清单

### 3.1 `utils/parser.py`

在 `Config` 中增加：

```python
rhl_norm: str = "none"
rhl_norm_eps: float = 1e-6
rhl_stats: bool = False
```

在 argparse 中增加：

```python
parser.add_argument(
    "--rhl_norm",
    type=str,
    default=Config.rhl_norm,
    choices=["none", "l2", "l2_sqrt", "layernorm"],
    help="Normalization applied to RHL outputs before analytic fitting."
)
parser.add_argument(
    "--rhl_norm_eps",
    type=float,
    default=Config.rhl_norm_eps,
    help="Epsilon for RHL output normalization."
)
parser.add_argument(
    "--rhl_stats",
    action="store_true",
    default=Config.rhl_stats,
    help="Print lightweight RHL output statistics for the first few fit batches."
)
```

### 3.2 `network/Buffer.py`

增加 `math` 和 `torch.nn.functional as F` 导入。

在 `RandomBuffer.__init__()` 增加参数：

```python
rhl_norm: str = "none"
rhl_norm_eps: float = 1e-6
```

保存属性：

```python
self.rhl_norm = rhl_norm
self.rhl_norm_eps = rhl_norm_eps
```

`forward()` 建议写成：

```python
@torch.no_grad()
def forward(self, X: torch.Tensor) -> torch.Tensor:
    X = X.to(self.weight)
    Z = self.activation(super().forward(X))
    norm = getattr(self, "rhl_norm", "none")
    eps = getattr(self, "rhl_norm_eps", 1e-6)

    if norm == "none":
        return Z
    if norm == "l2":
        return F.normalize(Z, p=2, dim=-1, eps=eps)
    if norm == "l2_sqrt":
        return F.normalize(Z, p=2, dim=-1, eps=eps) * math.sqrt(Z.shape[-1])
    if norm == "layernorm":
        return F.layer_norm(Z, (Z.shape[-1],), weight=None, bias=None, eps=eps)
    raise ValueError(f"Unknown rhl_norm: {norm}")
```

关键点：

1. 用 `getattr` 兼容旧 checkpoint。
2. 不要把 `weight` 改成 `Parameter`。
3. 不要引入 `nn.LayerNorm` 模块，使用 functional no-affine 版本即可，减少 checkpoint 状态变化。

### 3.3 `trainer/trainer.py`

修改 `AIR.__init__()` 签名：

```python
def __init__(
    self,
    backbone_output,
    backbone,
    buffer_size,
    gamma,
    device=None,
    dtype=torch.double,
    linear=RecursiveLinear,
    learned_classes=None,
    rhl_norm="none",
    rhl_norm_eps=1e-6,
    rhl_stats=False,
):
```

创建 `RandomBuffer` 时传入：

```python
self.buffer = RandomBuffer(
    backbone_output,
    buffer_size,
    rhl_norm=rhl_norm,
    rhl_norm_eps=rhl_norm_eps,
    **factory_kwargs,
)
```

在 step1 构造 AIR 时传入：

```python
rhl_norm=self.opts.rhl_norm,
rhl_norm_eps=self.opts.rhl_norm_eps,
rhl_stats=self.opts.rhl_stats,
```

旧 AIR checkpoint 加载后，如果继续 step2，`RandomBuffer.forward()` 的 `getattr` 会让旧模型默认为 `none`。

### 3.4 RHL stats

优先做轻量统计，不要每个 batch 求 condition number。建议在 `AIR.feature_expansion()` 里只在 `rhl_stats=True` 时打印前 3 次：

```text
RHL norm mode
row_norm mean/std/min/max
has_nan / has_inf
```

不要默认打开，避免日志过多和影响速度。

### 3.5 `run.sh`

增加环境变量：

```bash
RHL_NORM="${RHL_NORM:-none}"
RHL_NORM_EPS="${RHL_NORM_EPS:-1e-6}"
RHL_STATS="${RHL_STATS:-0}"
```

追加命令参数：

```bash
--rhl_norm "$RHL_NORM" \
--rhl_norm_eps "$RHL_NORM_EPS" \
```

如果 `RHL_STATS=1`，再追加 `--rhl_stats`。建议用数组实现，保持 `set -euo pipefail` 下安全。

## 4. 验证命令

实现后先做不依赖数据集的 smoke test：

```bash
cd /root/2TStorage/lyc/SegACIL
python - <<'PY'
import torch
from network.Buffer import RandomBuffer

X = torch.randn(2, 9, 256)
for norm in ["none", "l2", "l2_sqrt", "layernorm"]:
    rb = RandomBuffer(256, 8196, rhl_norm=norm)
    Y = rb(X)
    print(norm, Y.shape, torch.isfinite(Y).all().item(), Y.norm(dim=-1).mean().item())
PY
```

再验证 CLI：

```bash
python utils/parser.py --rhl_norm l2_sqrt --rhl_norm_eps 1e-6 --rhl_stats
```

再做 baseline 等价性检查：

```bash
SUBPATH=20260609_rhl_none BASE_SUBPATH=20260607 RHL_NORM=none bash run.sh
```

预期：结果应接近已有 `20260607` 的 `15-5 sequential step1`，即 all mIoU 约 69.56%。允许小幅随机波动，但不能出现明显崩塌或 NaN。

## 5. 实验矩阵

当前 `run.sh` 已是 `TASK=15-5`、`SETTING=sequential`、`START_STEP=1`、`END_STEP=1`，适合复用 step0 快速跑 RHL 对照。

统一使用：

```bash
BASE_SUBPATH=20260607
TASK=15-5
SETTING=sequential
MODEL=deeplabv3_resnet101
BUFFER=8196
```

第一组，归一化方式：

| 实验名 | RHL_NORM | gamma | 目的 |
|---|---|---:|---|
| `20260609_rhl_none` | none | 1 | 新代码 baseline 等价性 |
| `20260609_rhl_l2sqrt_g1` | l2_sqrt | 1 | 主候选 |
| `20260609_rhl_l2_g1` | l2 | 1 | 强归一化对照 |
| `20260609_rhl_ln_g1` | layernorm | 1 | 无参数 LN 对照 |

第二组，只对第一组最好的 normalization 做 gamma sweep：

| 实验名 | RHL_NORM | gamma |
|---|---|---:|
| `*_g0p1` | best norm | 0.1 |
| `*_g1` | best norm | 1 |
| `*_g10` | best norm | 10 |

如果 `l2` 在 gamma=1 明显下降，不要直接否定它，补跑 `gamma=0.01` 或 `0.1`。纯 `l2` 缩小特征能量后需要更弱的 ridge 正则。

## 6. 记录指标

每个实验必须记录：

1. `subpath`
2. `base_subpath`
3. `rhl_norm`
4. `rhl_norm_eps`
5. `gamma`
6. old mIoU: `0 to 15 mIoU`
7. new mIoU: `16 to 20 mIoU`
8. all mIoU: `Mean IoU`
9. 是否 NaN / inverse error / CUDA solver error
10. RHL row norm 摘要，如果打开 `--rhl_stats`

建议把结果追加到：

```text
AI_docs/实验记录_RHL归一化.md
```

如果该文件不存在，未来执行者可以新建。

## 7. 判定标准

RHL normalization 可以进入论文主线的最低标准：

1. `15-5 sequential` all mIoU 不低于 baseline 0.2 个百分点以上。
2. old/new mIoU 至少一个有稳定提升，且另一个不明显下降。
3. RHL row norm 的 std/max 明显更受控。
4. 无 NaN、无 `torch.inverse` / `cusolver` 报错。
5. 在 `disjoint` 或 `overlap` 中仍不破坏伪标签实验。

如果只提升数值稳定性但 mIoU 不提升，可以作为辅助模块写入，但不宜作为唯一贡献。

## 8. 风险与处理

| 风险 | 处理 |
|---|---|
| 旧 AIR checkpoint 缺少 `rhl_norm` 属性 | `RandomBuffer.forward()` 用 `getattr(..., "none")` |
| `l2` 导致 mIoU 下降 | 补跑更小 gamma；优先看 `l2_sqrt` |
| `layernorm` 改变特征分布过强 | 作为对照，不优先写主线 |
| 日志过多 | `rhl_stats` 默认关闭，只打印前几次 |
| 误把 RHL 改成可训练层 | 明确保留 `register_buffer("weight", W)` |
| 与 backbone 替换混杂 | 当前计划禁止同时换 backbone |

## 9. 完成后交付

执行完本计划后，应交付：

1. 代码 diff：`utils/parser.py`、`network/Buffer.py`、`trainer/trainer.py`、`run.sh`。
2. smoke test 输出摘要。
3. 至少 2 个实验结果：`none` 与 `l2_sqrt`。
4. 一份实验记录 Markdown。
5. 明确结论：RHL normalization 是否进入最终方法组合。

