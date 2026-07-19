# 可靠性加权 C-RLS 升级方案与 Codex 执行指南

> **状态：设计与实施规格已完成，代码尚未修改，实验尚未启动。**
>
> 本文是后续 Codex 的唯一执行规格。用户确认前，只允许读代码、核验路径和维护本文；不得实现、不得启动 W0/W1。

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Mode：`plan`
- 生成日期：2026-07-19
- 设计依据：
  - `7-19_自适应伪标签阈值全阶段总结与小白版复盘.md`
  - `7-19_matched-global六组正式对比结论与后续安排.md`
- Verification Status：`PLANNED`
- Version：`reliability_weighted_crls_v1`
- 当前推荐：`PROCEED_WITH_GATES`

---

## 1. Goal

在不改 backbone、不重训 step0、不引入 SGD、不改变伪标签 hard class ID 的前提下，把 teacher 的连续可靠性作为 `sample_weight` 传入 C-RLS：

\[
\min_W \sum_i w_i\lVert\phi_iW-y_i\rVert_2^2+\gamma\lVert W\rVert_2^2.
\]

第一轮只回答一个问题：

> 在候选集合和 hard pseudo-label 完全一致时，连续 confidence 或 confidence×margin 权重是否稳定优于未加权 matched-global baseline？

---

## 2. 为什么必须升级

matched-global 六组实验已经完成 hard classwise threshold 的 stop-loss：

```text
artifact_class - matched-global:
overlap 三 seed = +0.004062 pp
disjoint 三 seed = -0.000051 pp
六组平均 = +0.002005 pp
预注册实际意义门槛 = +0.1 pp
```

继续微调 q、`min_conf` 或 hard margin 只会改变极少边界像素，不能利用候选内部的连续可靠性。

当前底层矛盾是：

```text
teacher 提供 confidence/margin
  ↓
hard threshold 压缩成 accept/reject
  ↓
所有 accepted pseudo pixels 在 C-RLS 中权重完全相同
```

升级后：

```text
teacher confidence/margin
  ↓
相同 matched-global 候选与标签
  ↓
每个 accepted pseudo pixel 带连续 sample_weight
  ↓
进入 XᵀDX 与 XᵀDY
```

---

## 3. Global Constraints

### 3.1 研究边界

- 不预设该方法必须成为论文级核心创新；
- W0/W1/W2 均使用预先定义的通过门槛；
- 未通过当前 gate 就停止，不用更多小网格为弱收益续命；
- 不在 `sequential` 上启用伪标签；
- raw GT 只用于 audit，不进入训练、不拟合测试专用参数；
- W1 不同时改变候选集合、阈值和权重函数；
- W1 不引入 alpha、temperature、floor 等超参数网格；
- soft target 不与本方案同时实现。

### 3.2 实验控制变量

必须保持：

```text
dataset=voc
task=15-5
curr_step=1
model=deeplabv3_resnet101
air_feature_source=auto -> decoder
batch_size=32
step0_batch_size=32
buffer=8196
gamma=1
rhl_norm=none
rhl_seed=-1
min_conf=0
max_conf=1
min_pixels=1
shrinkage=0
margin_min=0
SEGACIL_PIN_MEMORY=0
```

### 3.3 文件与谱系

- 所有新实验使用独立 `SUBPATH`；
- 不覆盖历史 checkpoint、artifact、JSON、stats 或日志；
- step0 checkpoint 必须做 SHA256 校验；
- W1 允许在未提交的实现状态上做机制筛选，但 runner 必须保存：
  - `git status --short`
  - `git rev-parse HEAD`
  - 与实现相关文件的 `git diff --binary`
- W2/W3 必须固定 clean commit；
- 未经用户明确要求不 commit、不 push。

### 3.4 资源

每次启动前：

```bash
df -h /root/2TStorage
nvidia-smi
ps -eo pid,user,etimes,cmd | rg 'train.py|run_pseudo_label|calibrate_pseudo'
```

W1 四个 step1 预计每组约 1.5 GB，启动前至少保留 20 GiB 可用空间。当前快照为约 99 GB 可用、GPU 空闲，但必须以启动时检查为准。

---

## 4. Non-Goals

第一版明确不做：

- 新的类别阈值；
- q/min_conf/margin hard-filter sweep；
- class-balanced weight；
- learnable calibration head；
- temperature scaling；
- confidence power \(\alpha\) 网格；
- weight floor 网格；
- teacher soft target；
- DeepLabV3+ 组合；
- 15-1 多 seed；
- sequential 实验。

这些项目只有在前一 gate 通过、且能够回答新的机制问题时才重新评估。

---

## 5. 候选路线与选择

| 路线 | 内容 | 决策 |
| --- | --- | --- |
| A | 继续 hard threshold 扫参 | 拒绝，matched-global 已否定 |
| B | hard label + continuous sample weight | 当前推荐 |
| C | teacher soft target C-RLS | B 失败后的二级升级 |

路线 B 的优势是模块边界窄：

```text
PseudoLabelResult
  → Trainer.apply_pseudo_labels_if_enabled
  → AIR.fit
  → RecursiveLinear.fit
```

不需要改变 backbone、RHL、checkpoint 拓扑或评估流程。

---

## 6. Domain Model

### 6.1 术语

| 术语 | 精确定义 |
| --- | --- |
| candidate | 当前增量标签为 background、非 ignore、teacher top1 为旧类的像素 |
| accepted | candidate 中通过 matched-global fixed threshold 和现有 margin guard 的像素 |
| visible GT | 当前增量标签中直接可见、未被伪标签替换的监督 |
| pseudo weight | accepted 伪标签进入 C-RLS 的连续权重 |
| audit raw GT | VOC 完整 mask 经过 task ordering map 后的评价标签，只用于诊断 |
| matched-global | 每个 seed 使用对应 artifact 的 `global_threshold` 作为单一 fixed threshold |

### 6.2 不变量

1. 伪标签只能覆盖 background candidate；
2. 可见 GT 不被 teacher 改写；
3. ignore 始终不进入解析更新；
4. `weighting=none` 返回 `sample_weight=None`，走历史未加权路径；
5. weighted candidate 与 matched-global baseline 的 accepted mask 完全一致；
6. `sample_weight` 取值必须有限且在 `[0,1]`；
7. 真标签权重为 1；
8. `sample_weight=None` 的数值结果必须与改动前等价。

---

## 7. W0：实现前 raw-mask 只读审计

W0 是进入 weighted code 的前置 gate，不训练模型。

### 7.1 审计问题

对 overlap 和 disjoint 分别回答：

1. confidence 越高，伪标签正确率是否越高？
2. `confidence × margin` 是否提供更强或至少不更差的排序？
3. 类别之间主要是校准偏移，还是 teacher 根本无法排序正确/错误候选？
4. 低 confidence 候选主要补回 hidden old，还是制造 false old？
5. 两个 setting 的可靠性关系是否方向一致？

### 7.2 数据与 checkpoint

overlap teacher：

```text
checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/
  voc/15-5/overlap/step0/
  deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
SHA256=6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913
```

disjoint teacher：

```text
checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/
  voc/15-5/disjoint/step0/
  deeplabv3_resnet101_voc_15-5_step_0_disjoint.pth
SHA256=040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962
```

### 7.3 数据对齐设计

新增：

```text
tools/audit_pseudo_label_reliability.py
tests/test_pseudo_label_reliability_audit.py
```

工具复用 `VOCSegmentation` 的：

- file list；
- incremental `gt_label_mapping`；
- `ordering_map`；
- image/mask 路径。

审计使用确定性原图：

```text
ExtToTensor
ExtNormalize
batch_size=1
无 random scale/crop/flip
```

包装 Dataset 的 `__getitem__`：

1. 调用基础 VOC dataset 得到归一化 image、incremental label、file_name；
2. 从同一个 `dataset.masks[index]` 读取完整 raw mask；
3. 使用同一个 `ordering_map` 把 raw VOC ID 转成模型类别 ID；
4. 断言 raw mask 与 incremental label 空间尺寸一致；
5. 返回 `(image, incremental_label, raw_ordered_label, file_name)`。

选择确定性原图而不是复制随机训练增强，原因：

- W0 只判断 reliability 排序，不拟合训练参数；
- 原图审计完全可复现；
- 避免 raw/incremental mask 在随机几何变换中错位；
- 限制是与随机 crop 训练分布不完全一致，报告必须明确记录。

### 7.4 正确性定义

对每个 candidate：

```text
correct = candidate_predicted_class == raw_ordered_label
```

还要分解错误：

```text
false_old_on_background:
  raw_ordered_label == 0

wrong_old_class:
  raw_ordered_label in old_class_ids
  and raw_ordered_label != predicted_old_class

false_old_on_current_or_future:
  raw_ordered_label not in {0, 255, old_class_ids}
```

hidden-old recall 的分母：

```text
incremental_label == 0
and raw_ordered_label in old_class_ids
```

### 7.5 审计指标

全局与 per-class 均输出：

- candidate count；
- exact-class precision；
- hidden-old recall；
- confidence mean/std/min/P10/P25/P50/P75/P90/max；
- margin 同类统计；
- 10 个等宽 reliability bins；
- 10 个等量 reliability bins；
- ECE；
- top quartile precision；
- bottom quartile precision；
- decile mean-score 与 precision 的 Spearman rho；
- 三类错误占比；
- matched-global threshold 下的 precision/recall/coverage。

评估三个排序信号：

```text
confidence = top1 score
margin = top1 - top2
confidence_margin = confidence × margin
```

### 7.6 W0 产物

```text
artifacts/pseudo_label/reliability_audit_w0_20260719/
  overlap/
    audit_manifest.json
    reliability_summary.json
    reliability_bins.csv
    per_class.csv
    report.md
  disjoint/
    audit_manifest.json
    reliability_summary.json
    reliability_bins.csv
    per_class.csv
    report.md
logs/pseudo_label/reliability_audit_w0_20260719.log
```

manifest 必须记录：

- `audit_only=true`；
- dataset/task/step/setting；
- file list 数量；
- teacher path/hash；
- git commit/dirty；
- deterministic transform；
- loss type；
- expected teacher classes；
- old class IDs；
- matched-global threshold；
- max samples；
- seed；
- timestamp。

### 7.7 W0 单元测试

`tests/test_pseudo_label_reliability_audit.py` 至少包含：

1. synthetic incremental/raw mask 像素级对齐；
2. VOC ordering map 正确；
3. exact-class correct 判定；
4. 三类错误分解互斥且计数守恒；
5. hidden-old recall 分母正确；
6. 等宽 bin 边界包含 0 和 1；
7. 等量 bin 总计数等于 candidate count；
8. ECE 与手工小例子一致；
9. 空类别不产生 NaN 污染；
10. manifest 标记 `audit_only=true`；
11. checkpoint SHA 不匹配时拒绝执行；
12. `--max-samples 1` smoke 能生成五个正式产物。

### 7.8 W0 进入条件

每个 candidate signal 独立判定。

某个 signal 进入 W1，必须同时满足：

- overlap 和 disjoint 都至少有 8 个非空等量 bins；
- 两个 setting 的 decile Spearman rho 均不低于 `0.6`；
- 两个 setting 的 top-quartile precision 均至少比 bottom-quartile 高 `5 pp`；
- 没有出现 setting 间方向相反；
- 权重分布不是超过 95% 像素集中在 `[0,0.05]` 或 `[0.95,1]`。

决策：

| W0 结果 | 后续 |
| --- | --- |
| confidence 通过，confidence_margin 通过 | 实现并运行 W1 两个 candidate |
| 仅 confidence 通过 | 只实现/运行 confidence candidate，W1 减为 2 个训练 |
| 仅 confidence_margin 通过 | 只实现/运行 confidence_margin candidate，W1 减为 2 个训练 |
| 两者均未通过 | 停止 weighted hard-label，转入 soft-target 设计审查 |

这些是机制筛选 gate，不是像素独立性统计显著性检验。像素高度相关，因此不报告伪精确 p 值。

---

## 8. Reliability Weight 规格

W0 通过后才实施本节。

### 8.1 CLI

新增唯一参数：

```text
--pseudo_label_weighting
```

取值：

```text
none
confidence
confidence_margin
```

默认：

```text
none
```

不新增 power、floor、temperature 等参数。

### 8.2 权重函数

`none`：

```text
sample_weight = None
```

`confidence`：

\[
w_i=p_i.
\]

`confidence_margin`：

\[
w_i=p_i(p_i-p_i^{(2)}).
\]

其中 \(p_i^{(2)}\) 为 top-2 score。

统一裁剪：

\[
w_i\leftarrow\operatorname{clip}(w_i,0,1).
\]

### 8.3 完整像素权重图

当 weighting 非 none 时：

```text
初始化所有像素 weight=1
labels==255 → weight=0
accepted pseudo pixels → confidence 或 confidence_margin
rejected candidates → 保持原标签与 weight=1
```

W1 中 rejected candidates 极少；保留其原始行为可以保证相对 matched-global baseline 只改变 accepted pseudo pixels 的解析权重。

### 8.4 为什么不做类归一化

第一轮不按类别归一化或 rank：

- matched-global 已经否定了类别阈值的主要价值；
- 类归一化会再次引入 classwise 机制，破坏清晰归因；
- 原始 confidence 能直接检验 teacher reliability 是否可用。

若 W0 显示明显类别校准偏移但全局 reliability 仍单调，必须先写新的可证伪方案，不能在 W1 临时加入类归一化。

---

## 9. 模块设计

### 9.1 模块与接口

| Module | 当前接口 | 新接口 | 责任 |
| --- | --- | --- | --- |
| `utils/pseudo_label.py` | `PseudoLabelResult(labels, mask, stats)` | 增加 `sample_weight` | 生成标签、accepted mask 和可靠性权重 |
| `Trainer` | 返回 `labels` | 返回 `(labels, sample_weight)` | 连接 teacher 与 AIR |
| `AIR.fit` | `fit(X, y)` | `fit(X, y, sample_weight=None)` | 将 label/weight 对齐到 feature 空间 |
| `RecursiveLinear.fit` | `fit(X, y)` | `fit(X, y, sample_weight=None)` | 执行加权解析更新 |

这是一个窄 seam：

```text
PseudoLabeler 隐藏可靠性细节
Trainer 只传两个监督张量
AIR 只负责空间对齐
RecursiveLinear 只负责加权线性代数
```

### 9.2 `utils/pseudo_label.py`

修改：

```text
VALID_WEIGHTING_STRATEGIES
PseudoLabelConfig.weighting
PseudoLabelResult.sample_weight
PseudoLabelBatchStats 权重统计字段
validate_pseudo_label_config
config_from_opts
apply_pseudo_labels
PseudoLabeler.apply
```

新增纯函数：

```text
build_pseudo_label_sample_weight(
    labels,
    candidates,
    accepted_mask,
    weighting,
) -> Optional[torch.Tensor]
```

返回：

- `weighting=none`：`None`；
- 其他：与 labels 同空间的 `float32 [N,H,W]`。

必须检查：

- shape 一致；
- finite；
- `[0,1]`；
- accepted mask 是 candidate mask 子集；
- visible GT 恒为 1；
- ignore 恒为 0。

### 9.3 `trainer/trainer.py`

将：

```python
y = self.apply_pseudo_labels_if_enabled(X, y)
self.model.fit(X, y)
```

改成：

```python
y, sample_weight = self.apply_pseudo_labels_if_enabled(X, y)
self.model.fit(X, y, sample_weight=sample_weight)
```

off 或 weighting none：

```text
sample_weight=None
```

两条增量路径都必须改：

- `curr_step == 1`；
- `curr_step > 1`。

step0 realign 保持：

```python
self.model.fit(X, y)
```

因为 step0 是可见真值，不需要权重图。

### 9.4 `AIR.fit`

新签名：

```python
def fit(self, X, y, sample_weight=None):
```

处理：

1. `X` 通过 `feature_expansion` 得到 `[B,HW,C]`；
2. label 使用 nearest 对齐到 `(H,W)`；
3. weight 若存在，也使用 nearest 对齐到同一个 `(H,W)`；
4. nearest 的理由是 weight 与被 nearest 采样的离散监督位置一一对应，避免在真标签/伪标签/ignore 边界混合；
5. 对齐后 clamp `[0,1]` 并检查 finite；
6. 调用：

```python
self.analytic_linear.fit(X, y, sample_weight=sample_weight)
```

### 9.5 `network/AnalyticLinear.py`

抽象接口：

```python
def fit(self, X, y, sample_weight=None):
```

`RecursiveLinear.fit`：

1. 展平 `X/y/weight`；
2. 过滤 `y==255`；
3. 若有 weight，检查 shape、finite、非负；
4. 构造 one-hot `Y`；
5. 扩展输出类别数的历史逻辑保持不变；
6. 未加权分支保留现有代码，保护数值等价；
7. 加权分支：

\[
X_w=\sqrt{w}\odot X,\quad Y_w=\sqrt{w}\odot Y;
\]

\[
S=R^{-1}+X_w^\top X_w;
\]

\[
R=S^{-1};
\]

\[
W\leftarrow W+RX_w^\top(Y_w-X_wW).
\]

零权重行可以过滤；若过滤后无有效样本，直接返回且不得修改 `R/weight`。

### 9.6 checkpoint 兼容性

本方案不新增模型参数或 buffer，不改变 checkpoint 拓扑：

- 历史 checkpoint 可以继续加载；
- 新 checkpoint 仍保存同一 AIR/RecursiveLinear 状态；
- 变化只在 fit 时的更新数据；
- weighting 参数必须写入 manifest/stats，不能只存在 CLI。

---

## 10. 统计与可复现性

### 10.1 `pseudo_label_stats.json`

schema 升级为 2，并增加：

```text
weighting
weighted_pixel_count
weight_sum
weight_mean
weight_std
weight_min
weight_p10
weight_p25
weight_p50
weight_p75
weight_p90
weight_max
weight_histogram
per_class_weight_mean
per_class_weight_sum
```

只对 accepted pseudo pixels 统计上述权重分布。真标签权重 1 不混入，否则会掩盖 pseudo weight 是否退化。

`weighting=none`：

```text
weighted_pixel_count=0
weight_* = null 或空字典
```

不得伪造为全 1 权重分布。

### 10.2 `run_manifest.json`

增加平铺字段和 `args` 字段：

```text
pseudo_label_weighting
```

runner 额外保存：

```text
source_commit
source_dirty
source_status_path
source_patch_path
teacher_sha256
baseline_result_path
baseline_result_sha256
```

### 10.3 summary

W1 汇总必须报告：

- baseline/candidate all、old、new；
- Δ all/old/new，单位同时给 fraction 和 pp；
- candidate/accepted count 与 ratio；
- pseudo weight mean/std/percentiles；
- per-class mIoU 差；
- W1 gate 每一条是否通过；
- 输出 `recommendation=continue|stop|review`。

---

## 11. 精确文件改动清单

W0：

```text
新增 tools/audit_pseudo_label_reliability.py
新增 tests/test_pseudo_label_reliability_audit.py
```

Weighted core：

```text
修改 utils/pseudo_label.py
修改 utils/parser.py
修改 utils/run_manifest.py
修改 trainer/trainer.py
修改 network/AnalyticLinear.py
修改 run.sh
修改 tools/run_adaptive_pseudo_label.sh
修改 tools/run_pseudo_label_grid.sh
修改 tools/summarize_adaptive_pseudo_label.py
修改 tools/summarize_pseudo_label_grid.py
修改 tests/test_pseudo_labeling.py
修改 tests/test_run_manifest.py
修改 tests/test_pseudo_label_grid_summary.py
修改 tests/test_air_feature_integration.py
新增 tests/test_weighted_recursive_linear.py
```

W1：

```text
新增 configs/pseudo_label_weighted_w1_20260719.tsv
新增 configs/pseudo_label_weighted_w1_baselines_20260719.json
新增 tools/summarize_pseudo_label_weighted_w1.py
新增 tools/run_pseudo_label_weighted_w1_20260719.sh
新增 tests/test_pseudo_label_weighted_w1_runner.py
```

若实际实现发现某个文件无需修改，必须在实施报告中说明原因；不得为了对齐清单制造空改动。

---

## 12. TDD 实施任务

### Task 1：W0 audit 数据与指标

**Files**

- Create: `tools/audit_pseudo_label_reliability.py`
- Test: `tests/test_pseudo_label_reliability_audit.py`

**Steps**

1. 写 raw/incremental alignment、correctness、error decomposition 的失败测试；
2. 运行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest \
  tests.test_pseudo_label_reliability_audit -v
```

3. 实现最小 Dataset wrapper 和纯统计函数；
4. 让单元测试通过；
5. 用 `--max-samples 1` 分别跑 overlap/disjoint smoke；
6. 检查五类正式产物；
7. 再运行完整 W0 audit；
8. 按第 7.8 节做 gate 判定；
9. 如果没有 signal 通过，停止，不执行 Task 2。

### Task 2：伪标签权重纯函数

**Files**

- Modify: `utils/pseudo_label.py`
- Modify: `utils/parser.py`
- Test: `tests/test_pseudo_labeling.py`

**Tests first**

1. parser 默认 `none`；
2. 非法 weighting 拒绝；
3. none 返回 `None`；
4. confidence 权重等于 top1 score；
5. confidence_margin 等于 score×margin；
6. visible GT=1；
7. ignore=0；
8. rejected candidate 保持 1；
9. shape/finite/range guard；
10. accepted mask 不变。

### Task 3：加权 RecursiveLinear

**Files**

- Modify: `network/AnalyticLinear.py`
- Create: `tests/test_weighted_recursive_linear.py`

**Tests first**

1. `sample_weight=None` 与旧路径完全一致；
2. 全 1 权重与未加权 close；
3. 0 权重行等价于删除该样本；
4. 分数权重与显式 batch weighted ridge direct solve close；
5. 多 batch recursive 与一次性 weighted sufficient statistics close；
6. ignore 与 weight 同时生效；
7. 非 finite、负权重、shape 不匹配报错；
8. 全 0 有效权重不修改状态；
9. bias=False 当前主路径通过；
10. 类别扩展逻辑不回归。

容差根据 double precision 设置，不得用宽松容差掩盖公式错误。

### Task 4：Trainer/AIR 接口

**Files**

- Modify: `trainer/trainer.py`
- Modify: `tests/test_air_feature_integration.py`
- Modify: `tests/test_pseudo_labeling.py`

**Tests first**

1. off 返回 `(labels, None)`；
2. weighting none 返回 `(pseudo_labels, None)`；
3. weighted 返回正确 NHW 权重；
4. AIR nearest 对齐 label/weight；
5. step1 调用传 weight；
6. step2+ 调用传 weight；
7. step0 realign 不传 weight；
8. 现有伪标签测试不回归。

### Task 5：manifest/stats/CLI 链

**Files**

- Modify: `utils/run_manifest.py`
- Modify: `run.sh`
- Modify: `tools/run_adaptive_pseudo_label.sh`
- Modify: `tools/run_pseudo_label_grid.sh`
- Modify: `tools/summarize_adaptive_pseudo_label.py`
- Modify: `tools/summarize_pseudo_label_grid.py`
- Modify corresponding tests

**要求**

1. `PSEUDO_LABEL_WEIGHTING` 从 shell 进入 CLI；
2. 进入 Config；
3. 进入 PseudoLabelConfig；
4. 进入 manifest；
5. 进入 stats；
6. 进入 summary；
7. 老 grid header 继续可用；
8. artifact 旧 grid 继续可用；
9. 新 weighted grid 使用 `weighting` 可选列；
10. dry-run 命令必须打印 weighting。

### Task 6：W1 配置、runner 与专用汇总

**Files**

- Create W1 TSV/baseline JSON/runner/summarizer/tests

**runner preflight**

1. 检查两个 step0 checkpoint 及 SHA；
2. 检查四个新输出目录不存在；
3. 检查至少 20 GiB 空间；
4. 打印 GPU 与占用进程；
5. 保存 source status/patch；
6. 先 dry-run；
7. 校验四行命令只有 setting、threshold、weighting、SUBPATH 差异；
8. 正式运行；
9. 所有训练完成后汇总；
10. 如果只失败在 summarize，不重训。

### Task 7：全量验证

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile \
  utils/pseudo_label.py \
  utils/parser.py \
  utils/run_manifest.py \
  trainer/trainer.py \
  network/AnalyticLinear.py \
  tools/audit_pseudo_label_reliability.py \
  tools/summarize_adaptive_pseudo_label.py \
  tools/summarize_pseudo_label_grid.py \
  tools/summarize_pseudo_label_weighted_w1.py
```

```bash
rg -n '[“”‘’]' \
  utils/pseudo_label.py \
  utils/parser.py \
  utils/run_manifest.py \
  trainer/trainer.py \
  network/AnalyticLinear.py \
  tools tests
```

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest discover -s tests -v
```

```bash
bash -n \
  run.sh \
  tools/run_adaptive_pseudo_label.sh \
  tools/run_pseudo_label_grid.sh \
  tools/run_pseudo_label_weighted_w1_20260719.sh
```

```bash
git diff --check
```

然后执行：

- audit `--max-samples 1`；
- weighted C-RLS synthetic smoke；
- W1 runner `DRY_RUN=1`；
- 一个真实 batch 的 Trainer smoke；
- 检查 manifest/stats/summary 字段。

---

## 13. W1 精确实验矩阵

只有 W0 对应 signal 通过时才保留该行。

| name | setting | threshold | weighting | seed | BASE_SUBPATH |
| --- | --- | ---: | --- | ---: | --- |
| overlap_confweight_seed1 | overlap | 0.447265625 | confidence | 1 | `20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32` |
| overlap_confmarginweight_seed1 | overlap | 0.447265625 | confidence_margin | 1 | `20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32` |
| disjoint_confweight_seed1 | disjoint | 0.029296875 | confidence | 1 | `20260705_pseudo_15-5_disjoint_off_seed1_bs32` |
| disjoint_confmarginweight_seed1 | disjoint | 0.029296875 | confidence_margin | 1 | `20260705_pseudo_15-5_disjoint_off_seed1_bs32` |

计划 `SUBPATH`：

```text
20260719_pseudo_15-5_overlap_globalfixed0p447265625_confweight_seed1_bs32_reuse20260627step0
20260719_pseudo_15-5_overlap_globalfixed0p447265625_confmarginweight_seed1_bs32_reuse20260627step0
20260719_pseudo_15-5_disjoint_globalfixed0p029296875_confweight_seed1_bs32_reuse20260705disjointstep0
20260719_pseudo_15-5_disjoint_globalfixed0p029296875_confmarginweight_seed1_bs32_reuse20260705disjointstep0
```

复用 baseline：

overlap：

```text
checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p447265625_seed1_bs32_reuse20260627step0/
  voc/15-5/overlap/step1/test_results_20260718_155137.json
```

启动前以目录内唯一/最新正式 `test_results_*.json` 再核验；baseline all mIoU 应为：

```text
0.708038216
```

disjoint：

```text
checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p029296875_seed1_bs32_recovery1_reuse20260705disjointstep0/
  voc/15-5/disjoint/step1/test_results_20260719_045822.json
```

baseline all mIoU：

```text
0.694873145
```

后续执行时仍须从目录重新读取正式 JSON，并核对 all mIoU 与 manifest，不得只依赖本文抄录值。

---

## 14. W1 判定

对每种 weighting 分别计算：

```text
Δ_overlap = candidate_overlap - matched_global_overlap
Δ_disjoint = candidate_disjoint - matched_global_disjoint
mean_Δ = (Δ_overlap + Δ_disjoint) / 2
```

通过条件全部满足：

1. `mean_Δ all mIoU >= +0.1 pp`；
2. 任一 setting `Δ all mIoU >= -0.05 pp`；
3. old mIoU 改善不以 new mIoU 明显下降换取；
4. pseudo weight 分布不是几乎全 0 或全 1；
5. 与 W0 reliability 方向一致；
6. 没有代码/manifest/accepted-mask 对照异常。

选择规则：

- 两个 candidate 都通过：选平均 all 提升更大的一个；
- 差异低于 `0.02 pp`：优先更简单的 `confidence`；
- 只有一个通过：选通过者；
- 都不通过：停止，不启动 W2。

---

## 15. W2 与 W3

### 15.1 W2

仅扩展 W1 最优 candidate：

```text
overlap seed2
overlap seed3
disjoint seed2
disjoint seed3
```

阈值继续匹配已有 global baseline：

```text
overlap seed2 = 0.419921875
overlap seed3 = 0.443359375
disjoint seed2 = 0.048828125
disjoint seed3 = 0.048828125
```

通过条件：

- 六组平均 `>= +0.1 pp`；
- 至少 5/6 为正；
- overlap/disjoint 不方向冲突；
- new mIoU 无系统性损害；
- clean commit 与完整 source provenance。

### 15.2 W3

仅 W2 通过：

- clean replay：off、fixed0.6、matched-global、weighted；
- `15-1 overlap` step1-step5；
- 再评估 DeepLabV3+；
- 多 seed 与成本报告；
- 正式论文消融。

---

## 16. 失败分支

### 16.1 W0 失败

结论：

```text
teacher confidence/margin 不能稳定排序 pseudo-label correctness
```

动作：

- 不实现/不训练简单 sample weight；
- 分析 BCE sigmoid 校准、背景通道和 teacher 错误类型；
- 单独设计 soft target C-RLS；
- 或将伪标签保留为 matched-global 辅助模块，资源转向 RHL/集成主线。

### 16.2 W1 失败

结论：

```text
可靠性可排序，但简单连续重加权没有转化为足够 mIoU 收益
```

动作：

- 不做 seed2/3；
- 检查权重是否过度削弱 hidden-old 召回贡献；
- 检查 feature-space 下采样后的 weight 分布；
- 决定 soft target 或停止该研究线；
- 不追加 alpha/floor 小网格。

### 16.3 W2 失败

结论：

```text
单 seed 信号无法跨 seed/setting 稳定复现
```

动作：

- 方法降级为负结果或辅助模块；
- 不进入 15-1/V3+；
- 主资源转向更有潜力路线。

---

## 17. 计划启动命令

以下命令只在用户确认、代码和 gate 全部通过后执行。

环境：

```bash
cd /root/2TStorage/lyc/SegACIL
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass
export TMPDIR=/root/2TStorage/tmp
export SEGACIL_PIN_MEMORY=0
export PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python
```

W0 smoke：

```bash
$PYTHON tools/audit_pseudo_label_reliability.py \
  --task 15-5 \
  --setting overlap \
  --curr-step 1 \
  --teacher-checkpoint checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth \
  --expected-teacher-sha256 6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913 \
  --matched-global-threshold 0.447265625 \
  --max-samples 1 \
  --output-dir artifacts/pseudo_label/reliability_audit_w0_20260719/smoke_overlap
```

W0 正式审计应由工具 runner 串行执行 overlap/disjoint，并在结束后生成 gate 报告。不要在两个进程中同时加载 teacher。

W1 dry-run：

```bash
DRY_RUN=1 bash tools/run_pseudo_label_weighted_w1_20260719.sh
```

W1 后台启动：

```bash
tmux new-session -d -s apl_weighted_w1_719 \
  "bash -lc 'cd /root/2TStorage/lyc/SegACIL && export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass TMPDIR=/root/2TStorage/tmp SEGACIL_PIN_MEMORY=0 PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python && bash tools/run_pseudo_label_weighted_w1_20260719.sh'"
```

查看：

```bash
tmux attach -t apl_weighted_w1_719
tail -f logs/pseudo_label/weighted_w1_20260719.log
```

---

## 18. 启动前最终验收

- [ ] 用户已明确确认开始实施；
- [ ] W0 工具单元测试通过；
- [ ] W0 overlap/disjoint 正式 audit 完成；
- [ ] 至少一个 signal 通过 W0 gate；
- [ ] weighted core 单元测试通过；
- [ ] `sample_weight=None` 与历史路径数值等价；
- [ ] accepted mask 与 matched-global baseline 语义一致；
- [ ] parser → config → manifest → stats → summary 字段贯通；
- [ ] Python compile 通过；
- [ ] shell `bash -n` 通过；
- [ ] 全测试通过；
- [ ] `git diff --check` 通过；
- [ ] W1 dry-run 只有预期变量变化；
- [ ] 两个 step0 SHA 正确；
- [ ] baseline JSON 指标正确；
- [ ] 四个新输出目录不存在；
- [ ] 磁盘至少 20 GiB；
- [ ] GPU 显存足够；
- [ ] 没有需要避让的他人训练；
- [ ] tmux 名称、日志、summary 路径明确。

---

## 19. 当前准备状态

已完成：

- [x] hard-threshold stop-loss 结论；
- [x] 最新方法目标；
- [x] W0 audit 口径与 gate；
- [x] 权重函数；
- [x] 模块 seam；
- [x] 加权 C-RLS 公式；
- [x] 兼容性边界；
- [x] 文件级改动清单；
- [x] TDD 测试清单；
- [x] W1 四组矩阵；
- [x] baseline/checkpoint/hash；
- [x] SUBPATH；
- [x] W1/W2/W3 停止条件；
- [x] tmux、日志和资源边界。

等待用户确认后执行：

- [ ] 实现并运行 W0；
- [ ] 根据 W0 gate 决定实现的 candidate；
- [ ] 实现 weighted C-RLS；
- [ ] 完成测试与 smoke；
- [ ] dry-run；
- [ ] 启动 W1；
- [ ] 汇总并严格判定。

---

## 20. 给后续 Codex 的最终指令

1. 先读本文和 `7-19_matched-global六组正式对比结论与后续安排.md`；
2. 检查 `git status`，保留用户已有改动；
3. 没有用户确认时不得改训练代码或启动任务；
4. 确认后先实现 W0，不要先写 weighted core；
5. W0 不通过就停止，不能绕过 gate；
6. W0 通过后严格按 TDD 顺序实施；
7. 不新增本文未定义的调参项；
8. W1 只复用 matched-global baseline，不重跑 baseline；
9. 参数尽量与历史实验完全一致；
10. 训练完成后以真实 JSON/stats/manifest 汇报；
11. 只有 summarize 失败时只修 summarize，不重训；
12. W1 未通过就停止，不做 seed2/3；
13. 不因沉没成本反复验证弱收益；
14. 不因方法暂时不能成为论文核心而忽略真实、稳定的正向证据。
