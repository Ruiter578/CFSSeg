# 自适应伪标签阈值 Codex 实现与独立实验执行方案

> 日期：2026-06-21
> 面向对象：后续直接执行代码、测试、实验和结果分析的 Codex
> 方法边界：non-sequential 伪监督支线；不作为 VOC 15-5 sequential 五方法主线的默认开关

## 0. 执行结论

本方案可以和五方法主线 **独立完成代码实现和单方法实验**，因为它改变的是 `AIR.fit()` 前的标签 target，而 BOA/PGH/PowerNorm/Snapshot/RHL-SE 主要改变 feature、backbone 或输出融合。

但“独立”有两个限制：

1. 它与五方法共享 `trainer/trainer.py`、`utils/parser.py`、checkpoint/config 和 runner 基础设施，不能在两个分支各自复制一套不兼容实现后直接合并。
2. 它只应在 overlap/disjoint 评估；五方法当前主表是 15-5 sequential。两者不能因为使用同一代码库就放进同一结果表直接比较。

推荐组织方式：

- 五方法先完成 Phase 0 的 manifest、配置恢复、batch-limit 和 neutral baseline 基础设施；
- 从该 Phase 0 commit 创建 `feature/adaptive-pseudo-label` 独立 worktree；
- 伪标签在该分支完成 helper、Trainer 接入、runner 和 overlap/disjoint 实验；
- standalone 有正向证据后，再在 non-sequential 协议内与 BOA/PGH/PowerNorm 做 2x2；
- 不把伪标签默认并入 sequential Phase 4/6。

## 1. 开始前的硬约束

### 1.1 不在当前 baseline checkout 改代码

当前 `/root/2TStorage/lyc/SegACIL` 的 `run_origin.sh` 正执行 `20260621_baseline_bs64_32`。step0 刚完成，shell 会为 step1 重新启动 Python。当前 checkout 的代码修改会被 step1 读取。

文档可在当前目录编辑；代码实现必须等待 baseline 完成，或在 sibling worktree 中进行。即使使用 worktree，同一 GPU 上的 smoke/训练也等待 baseline 完成。

### 1.2 协议 guard

默认规则：

```text
setting == sequential and pseudo_label_strategy != off -> 明确报错
setting in {overlap, disjoint} -> 允许
```

不要让 sequential 悄悄 no-op，也不要在真实 background 上误开伪标签。

### 1.3 兼容基线

以下路径必须保留：

- `pseudo_label_strategy=off`：逐元素复现无伪标签 baseline；
- `strategy=fixed` 且 AIR teacher：复现当前 step2+ 的固定阈值语义；
- 旧的 `--use_pseudo_label --pseudo_label_confidence 0.7` 命令仍可映射到 fixed；
- 所有新参数进入 run manifest/checkpoint sidecar。

## 2. 推荐模块边界

不要继续把所有逻辑堆进 `Trainer.get_pseudo_labels()`。新增一个聚焦模块：

```text
utils/pseudo_label.py
```

它只处理 tensor、阈值和统计，不加载 Dataset/Model，也不依赖完整 Trainer。

### 2.1 `extract_teacher_probabilities`

接口建议：

```python
def extract_teacher_probabilities(
    model_output,
    *,
    loss_type: str,
    expected_classes: int,
) -> torch.Tensor:
    """Return probabilities in NCHW layout."""
```

职责：

- DeepLab `(logits, feature_dict)`：取第一个元素；
- AIR tensor：识别 NHWC 并转 NCHW；
- 已是 NCHW 时保持；
- BCE 使用 sigmoid，CE 使用 softmax；
- 类别维无法唯一判断时直接报错，不猜测。

### 2.2 `compute_pseudo_label_candidates`

```python
def compute_pseudo_label_candidates(
    probabilities,
    labels,
    old_class_ids,
):
    """Return top1 score, top1 label, margin and candidate mask."""
```

候选必须同时满足：

- `labels == 0`；
- teacher top1 在显式 `old_class_ids` 中；
- ignore 255 永不进入；
- 不使用 `pred_labels > 0` 代替旧类集合。

### 2.3 `resolve_class_thresholds`

支持：

```text
fixed
batch_global
batch_class
artifact_class
```

`artifact_class` 是推荐正式模式。阈值函数接收 class counts、raw quantile、global fallback、shrinkage lambda 和上下界，返回 `{class_id: threshold}` 与 fallback 原因。

### 2.4 `apply_pseudo_labels`

输入原 label、candidate stats、thresholds、`margin_min`，输出：

```text
pseudo_labels
pseudo_mask
PseudoLabelBatchStats
```

必须保证所有非 background label 和 255 原样不变。

## 3. 参数设计

在 `Config` 与 parser 中新增：

```text
--pseudo_label_strategy off|fixed|batch_global|batch_class|artifact_class
--pseudo_label_confidence FLOAT
--pseudo_label_quantile FLOAT
--pseudo_label_min_conf FLOAT
--pseudo_label_max_conf FLOAT
--pseudo_label_min_pixels INT
--pseudo_label_shrinkage FLOAT
--pseudo_label_margin_min FLOAT
--pseudo_label_threshold_artifact PATH
--pseudo_label_calibration_bins INT
--pseudo_label_stats
```

兼容规则：

- 未给 `--use_pseudo_label` 时强制 `strategy=off`；
- 给了 `--use_pseudo_label` 但未给 strategy 时使用 `fixed`；
- `artifact_class` 没有 artifact 或 schema/hash 不匹配时退出，不回退为 fixed；
- quantile 必须在 `[0,1]`，min/max/margin 做启动时校验。

## 4. Threshold artifact

新增工具：

```text
tools/calibrate_pseudo_label_thresholds.py
```

### 4.1 输入

- dataset/task/step/setting；
- teacher checkpoint；
- deterministic calibration transform；
- quantile/min/max/min_pixels/shrinkage；
- histogram bins；
- output JSON。

### 4.2 统计方式

逐 batch teacher inference，按旧类别累计 `[0,1]` histogram，不保存全部像素。候选只使用当前 protocol label 为 background 的位置。

### 4.3 Artifact schema

```json
{
  "schema_version": 1,
  "dataset": "voc",
  "task": "15-5",
  "step": 1,
  "setting": "overlap",
  "teacher_checkpoint": "...",
  "teacher_sha256": "...",
  "split": "train",
  "transform": {"type": "deterministic", "crop_size": 512},
  "old_class_ids": [1, 2, 3],
  "quantile": 0.7,
  "min_conf": 0.5,
  "max_conf": 0.95,
  "min_pixels": 64,
  "shrinkage": 256.0,
  "global_threshold": 0.0,
  "classes": {
    "1": {
      "candidate_count": 0,
      "raw_threshold": 0.0,
      "final_threshold": 0.0,
      "fallback": "none"
    }
  }
}
```

Trainer 加载时校验 dataset/task/step/setting、teacher hash、old class IDs 和阈值参数。

## 5. Trainer 接入

### 5.1 不再依赖共享 opts 的临时 step

当前 step1 路径会临时把 `self.opts.curr_step` 改成 0 以构造 step0 loader。应使用局部配置副本：

```text
step0_opts = copy.deepcopy(self.opts)
step0_opts.curr_step = 0
```

`self.opts.curr_step` 始终保持真实当前 step。该修复属于 Phase 0 共享基础设施；若伪标签分支先实现，提交必须足够独立，便于之后 rebase/cherry-pick。

### 5.2 Teacher 生命周期

- step1：teacher 是 step0 DeepLab checkpoint；
- step2+：teacher 是上一步 AIR checkpoint；
- teacher 始终 `eval()`、`no_grad()`；
- 统一 adapter 处理 tuple/tensor 和 NCHW/NHWC；
- 不从 realigned step0 AIR 误取当前正在训练的 student 作为 teacher。

### 5.3 调用位置

在每个 incremental `self.model.fit(X, y)` 前：

```python
if self.pseudo_labeler.enabled:
    y, batch_stats = self.pseudo_labeler(images=X, labels=y)
self.model.fit(X, y)
```

不要在 Dataset 中永久改写 mask，也不要在 eval/test 使用伪标签。

## 6. 配置与结果持久化

每个 run 必须记录：

```text
strategy/fixed confidence/quantile/min/max
min_pixels/shrinkage/margin
threshold artifact path + sha256
teacher path + sha256
old class IDs
setting/task/step
per-class candidate/accepted/threshold/fallback
```

建议输出：

```text
checkpoints/<run>/.../pseudo_label_config.json
checkpoints/<run>/.../pseudo_label_stats.json
logs/pseudo_label/<run>/run_summary.md
```

## 7. 测试计划

新增 `tests/test_pseudo_labeling.py`。

### 7.1 纯函数测试

1. fixed 阈值输出与当前规则一致；
2. batch-global quantile 数值正确；
3. batch-class 各类阈值不同；
4. 低候选类走 global/fixed fallback；
5. shrinkage 在 `n` 增大时趋近 class threshold；
6. min/max clipping 正确；
7. margin 拒绝 top1/top2 接近的像素；
8. 非 background 与 255 永不改变；
9. 非 old class 永不作为伪标签。

### 7.2 输出适配测试

1. DeepLab tuple + NCHW；
2. AIR tensor + NHWC；
3. CE softmax 与 BCE sigmoid；
4. 类别维歧义时报错；
5. 两种布局构造相同 logits 时，top1/score/margin 完全一致。

### 7.3 Artifact 测试

1. JSON round-trip；
2. checkpoint hash 不一致拒绝；
3. task/step/setting 不一致拒绝；
4. histogram quantile 与直接 `torch.quantile` 在 bin 容差内一致；
5. 空类和全空候选可序列化。

### 7.4 Trainer smoke

- `setting=sequential + strategy on` 明确失败；
- `setting=overlap + strategy off` 一批 fit；
- `setting=overlap + fixed` 一批 fit；
- `setting=overlap + artifact_class` 一批 fit；
- checkpoint 和 stats 正常落盘。

## 8. Runner

新增：

```text
tools/run_adaptive_pseudo_label.sh
tools/summarize_adaptive_pseudo_label.py
```

runner 默认：

```text
TASK=15-5
SETTING=overlap
BATCH_SIZE=32
BUFFER=8196
GAMMA=1
RHL_NORM=none
PSEUDO_LABEL_STRATEGY=off
```

run id 必须编码 setting、task、strategy、q/fixed、margin、seed 和 Batch。输出目录不能与 sequential 五方法实验复用。

## 9. 实验矩阵

### PL-0：协议与实现正确性

| Case | task/setting | 预期 |
|---|---|---|
| S0 | 15-5 sequential/off | 正常 baseline |
| S1 | 15-5 sequential/on | 启动时明确拒绝 |
| S2 | synthetic DeepLab teacher | tuple/NCHW 正确 |
| S3 | synthetic AIR teacher | tensor/NHWC 正确 |

### PL-1：单步 overlap

| Case | strategy | 参数 |
|---|---|---|
| P0 | off | baseline |
| P1 | fixed | 0.6 |
| P2 | fixed | 0.7 |
| P3 | fixed | 0.8 |
| P4 | batch-global | q=0.7 |
| P5 | batch-class | q=0.7 |
| P6 | artifact-class | q=0.7 + shrinkage |
| P7 | artifact-class | P6 + margin |

所有 case 固定 base checkpoint、global/RHL seed、Batch、Buffer、gamma 和 feature method。

### PL-2：disjoint 复验

选择 off、fixed0.7、PL-1 最佳 adaptive 三项，在 15-5 disjoint 复验。

### PL-3：多 step

在 15-1 或 10-1 overlap 上比较 off/fixed0.7/最佳 adaptive，至少完成 step1-5；再在 disjoint 复验最佳两项。

### PL-4：组件消融

顺序：

1. artifact vs batch；
2. shrinkage on/off；
3. margin on/off；
4. q 0.6/0.7/0.8；
5. `tau_max` on/off。

先完成组件主效应，再做 q 敏感性，不执行全笛卡尔积。

## 10. 分析与判定

主指标：old/new/all/per-class mIoU、multi-step forgetting。

机制指标：candidate/accepted ratio、per-class threshold/coverage、confidence/margin 分布、fallback count、raw-mask audit precision/recall。

有效证据：

- 相对 off/fixed0.7，old mIoU 或 forgetting 稳定改善；
- new/all 没有由错误旧类伪标签造成明显退化；
- 多 seed 或多 step 方向一致；
- 类别覆盖改善不是靠接受大量低置信像素；
- overlap 与 disjoint 的差异能由未来类/背景噪声解释。

## 11. 与五方法的独立性和耦合

| 五方法模块 | 作用位置 | 与伪标签关系 | standalone 后怎样组合 |
|---|---|---|---|
| BOA-RHL | 随机 feature | 原理独立 | non-sequential 做 BOA on/off x pseudo on/off |
| PGH-RHL | prototype feature | 默认独立 | prototype bank 只用原 GT；禁止默认用 pseudo 构建 bank |
| PowerNorm | feature scale | 独立 | 固定 pseudo 最佳配置后 norm on/off |
| CA-C-RLS | target/sample weight | 强耦合 | pseudo 改类别，CA 改权重；必须 2x2 并说明 pseudo 像素权重 |
| Snapshot | backbone/teacher 来源 | 间接耦合 | Snapshot 会改变 teacher 校准，threshold artifact 必须按 teacher 重建 |
| RHL-SE 2.0 | 输出融合 | 下游独立 | 先比较单成员 pseudo 效果，再做 ensemble |

最强耦合是 CA-C-RLS：若伪标签像素直接获得与真实稀有类相同的大权重，错误标签影响会被放大。组合时建议增加：

```text
pseudo_sample_weight <= 1
```

并单独消融，不把它藏在 class weight 中。

## 12. 如何嵌入总体工作流

新增一条旁路线，不改原 Phase 编号：

```text
Phase 0 shared foundation
  -> PL-A helper/adapter/artifact
  -> PL-B 15-5 overlap/disjoint standalone
  -> PL-C 15-1/10-1 multi-step
  -> PL-D non-sequential 2x2 combinations
```

- PL-A 可与 BOA/SE2/PGH 实现并行；
- PL-B/C 使用独立 setting、runner、目录和结果表；
- PL-D 等待伪标签和待组合 RHL 方法各自 standalone 有证据；
- 原 Phase 6 sequential 最终表不混入伪标签；另建 non-sequential 结果表。

## 13. Git 提交边界

建议三个提交：

1. `pseudo: add teacher-output adapter and threshold unit tests`
2. `pseudo: add calibration artifact and trainer integration`
3. `pseudo: add runners summaries and experiment documentation`

每个提交都保持 strategy off 可运行。不要在同一提交中顺手加入 BOA/PGH/CA-C-RLS。

## 14. Codex 执行检查清单

1. 确认 baseline 已结束，或工作目录是新 worktree。
2. 确认分支基于统一 Phase 0 commit。
3. 先写 helper/adapter 测试，再接 Trainer。
4. 修复真实 step 的局部配置，不修改共享 opts。
5. 实现 calibration artifact 和 hash/schema 校验。
6. 完成 parser/config/checkpoint/summary 一致性。
7. 执行 `py_compile`、单元测试、`bash -n`。
8. 执行 sequential guard 和 overlap 单 batch smoke。
9. 启动 PL-1 矩阵，结果独立目录落盘。
10. 生成机制统计和结果报告后，再决定 PL-2/PL-3 配置。

该执行方案允许伪标签支线独立推进，同时通过共享 Phase 0、独立 worktree、协议分表和组合 2x2 避免与五方法主线发生不可归因耦合。
