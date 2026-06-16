# RHL-SE 方案一代码改动与功能实现报告

> 日期：2026-06-16  
> 方案来源：`AI_docs/idea构思与实验设计/6-15_RHL机制再分析与重构升级方案.md` 的方案一  
> 方案名称：RHL Subspace Ensemble, RHL-SE  
> 改动目标：固定全局训练流程，只改变 RHL 随机子空间，训练多个解析头，并在推理阶段做像素级概率平均。

---

## 1. 本次实现结论

本次已经完成 RHL-SE 的核心代码落地：

```text
独立 rhl_seed 参数
  -> 只控制 RandomBuffer / RHL 随机矩阵初始化
  -> 不污染全局 random_seed / DataLoader / 数据增强随机状态
  -> 支持多个 RHL seed 训练多个 step1 final.pth
  -> 新增 ensemble 评估脚本做概率平均
```

实现后，默认行为保持兼容：

```text
--rhl_seed -1
```

表示沿用原有全局随机状态，不改变旧实验逻辑。

---

## 2. 为什么要新增独立 `rhl_seed`

如果直接改全局 `--random_seed`，会同时影响：

1. `RandomBuffer` 的 RHL 随机矩阵；
2. DataLoader shuffle；
3. 随机裁剪、随机翻转等数据增强；
4. `RecursiveLinear` 新类别 tail 的随机初始化；
5. 其他 PyTorch / NumPy / Python 随机过程。

这会让 ensemble 多样性更大，但结果不可解释：如果涨点，无法判断是 RHL 子空间贡献，还是数据顺序、增强或其他随机过程贡献。

因此 RHL-SE 第一版采用：

```text
random_seed 固定
rhl_seed 单独变化
```

这样才能把实验解释为：

```text
同一个冻结 DeepLab 特征提取器 + 同一个训练流程 + 多个不同 RHL 随机子空间
```

---

## 3. 代码改动清单

### 3.1 `utils/parser.py`

新增配置字段：

```python
rhl_seed: int = -1
```

新增命令行参数：

```bash
--rhl_seed
```

含义：

| 值 | 行为 |
|---:|---|
| `-1` | 默认，沿用原全局随机状态 |
| `0,1,2,...` | 只用该 seed 初始化 RHL 随机矩阵 |

### 3.2 `network/Buffer.py`

`RandomBuffer` 新增参数：

```python
rhl_seed: int = -1
```

核心实现：

```python
self._reset_parameters_with_optional_seed(rhl_seed)
```

其中 `_reset_parameters_with_optional_seed()` 使用：

```python
torch.random.fork_rng(...)
```

作用是：

1. 进入临时 RNG 环境；
2. 用 `rhl_seed` 初始化 `RandomBuffer.weight`；
3. 退出时恢复外部 RNG 状态。

这保证了：

```text
rhl_seed 只改变 RHL 权重
不改变后续 DataLoader、augmentation、其他 torch.rand 调用
```

### 3.3 `trainer/trainer.py`

`AIR.__init__()` 新增：

```python
rhl_seed=-1
```

构造 `RandomBuffer` 时传入：

```python
self.buffer = RandomBuffer(
    backbone_output,
    buffer_size,
    rhl_norm=rhl_norm,
    rhl_norm_eps=rhl_norm_eps,
    rhl_seed=rhl_seed,
    **factory_kwargs,
)
```

step1 创建 AIR 时传入：

```python
rhl_seed=self.opts.rhl_seed
```

这样 `train.py --rhl_seed 2` 会真正进入 RHL 初始化。

### 3.4 `run_rhl_norm.sh`

新增环境变量：

```bash
RHL_SEED="${RHL_SEED:--1}"
```

训练命令追加：

```bash
--rhl_seed "$RHL_SEED"
```

日志会打印：

```text
rhl_norm=..., rhl_norm_eps=..., rhl_seed=..., rhl_stats=...
```

推荐 RHL-SE 训练命令示例：

```bash
DEFAULT_BATCH_SIZE=64 \
SUBPATH=20260616_rhl_se_seed1 \
BASE_SUBPATH=20260606 \
RHL_NORM=none \
RHL_SEED=1 \
GAMMA=1 \
bash ./run_rhl_norm.sh
```

### 3.5 `run.sh`

同样新增：

```bash
RHL_SEED="${RHL_SEED:--1}"
```

并透传：

```bash
--rhl_seed "$RHL_SEED"
```

默认 `-1`，不改变普通 `run.sh` 的原有行为。

### 3.6 `tools/eval_rhl_ensemble.py`

新增 RHL-SE 推理评估脚本，支持多个 checkpoint：

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --save_json logs/rhl_ensemble/20260616_rhl_se_k3.json
```

核心逻辑：

```python
prob_ens = (prob_1 + prob_2 + ... + prob_K) / K
pred = prob_ens.argmax(dim=1)
```

实现细节：

1. 默认将 checkpoint 加载到 CPU。
2. 默认逐个模型放到 GPU 做 forward，避免多个 AIR 模型同时占满显存。
3. 如果显存足够，可加 `--keep_models_on_gpu` 提速。
4. 支持 AIR 输出 `[B, H, W, C]`，也兼容普通分割模型输出 `[B, C, H, W]`。
5. 如果模型 forward 返回 `(logits, features)`，会自动取 `logits`。

---

## 4. 方案一实验流程

### 4.1 单 seed 训练

建议先跑 3 个 RHL seed：

```bash
cd /root/2TStorage/lyc/SegACIL

for seed in 1 2 3; do
  DEFAULT_BATCH_SIZE=64 \
  SUBPATH=20260616_rhl_se_seed${seed} \
  BASE_SUBPATH=20260606 \
  RHL_NORM=none \
  RHL_SEED=${seed} \
  GAMMA=1 \
  RHL_STATS=1 \
  bash ./run_rhl_norm.sh 2>&1 | tee -a logs/rhl_norm/20260616_rhl_se_seed${seed}.log
done
```

### 4.2 K=3 ensemble 评估

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --save_json logs/rhl_ensemble/20260616_rhl_se_k3.json
```

### 4.3 推荐结果表

| 方法 | seed / K | old mIoU | new mIoU | all mIoU |
|---|---|---:|---:|---:|
| CFSSeg single | seed 1 | 待填 | 待填 | 待填 |
| CFSSeg single | seed 2 | 待填 | 待填 | 待填 |
| CFSSeg single | seed 3 | 待填 | 待填 | 待填 |
| RHL-SE | K=3 | 待填 | 待填 | 待填 |

必须同时报告：

```text
single seed mean/std
ensemble vs mean(single)
ensemble vs best(single)
```

避免只挑一个 seed 作为对照。

---

## 5. 验证结果

已完成静态与轻量验证：

### 5.1 Python 编译检查

```bash
python -m py_compile \
  network/Buffer.py \
  utils/parser.py \
  trainer/trainer.py \
  tools/eval_rhl_ensemble.py
```

结果：通过。

### 5.2 参数解析检查

```bash
python utils/parser.py --rhl_seed 3 --rhl_norm none --rhl_stats
```

输出中包含：

```text
rhl_seed=3
rhl_norm='none'
rhl_stats=True
```

### 5.3 RHL seed 确定性检查

在 `segacil` 环境中验证：

```text
same_seed_weight_equal True
different_seed_weight_equal False
same_seed_output_equal True
```

说明：

1. 相同 `rhl_seed` 得到相同 RHL 权重；
2. 不同 `rhl_seed` 得到不同 RHL 权重；
3. 相同 `rhl_seed` 对同一输入得到相同输出。

### 5.4 全局 RNG 不污染检查

验证结果：

```text
global_rng_preserved True
```

说明 `rhl_seed` 初始化 RHL 后，外部 torch RNG 状态被恢复，后续随机过程不会被 RHL seed 改写。

### 5.5 Ensemble 脚本参数检查

```bash
python tools/eval_rhl_ensemble.py --help
```

结果：正常显示参数，包括 `--ckpts`、`--save_json`、`--keep_models_on_gpu`、`--max_batches`。

---

## 6. 注意事项

1. 本次只实现方案一 RHL-SE，没有启动新训练。
2. `RHL_SEED=-1` 是兼容模式，等价于不单独控制 RHL seed。
3. RHL-SE 第一轮建议使用 `RHL_NORM=none`，不要和此前效果不佳的 `l2_sqrt` 混合。
4. 如果后续想比较“全局 seed ensemble”，应另起实验名，例如 `global_seed_ensemble`，不要和 `RHL-SE` 混在一个表述中。
5. 当前工作区已有其他文档和脚本改动，本报告只覆盖 RHL-SE 相关实现。

