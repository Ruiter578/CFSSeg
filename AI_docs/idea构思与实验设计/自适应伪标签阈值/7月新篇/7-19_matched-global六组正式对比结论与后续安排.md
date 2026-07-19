# Matched-global 六组正式对比结论与后续安排

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `validate`
- Origin Date: `2026-07-19`
- Verification Status: `ANALYZED`
- Version Label: `matched_global_validation_v1`
- Evidence Source: `test_results_*.json`、`pseudo_label_stats.json`、`run_manifest.json`、训练日志
- Overall Confidence: `CAUTION`

> `ANALYZED` 表示已逐项核验现有原始结果和运行谱系，但没有重新运行历史 artifact 组。本文用于机制筛选和路线决策，不将历史 dirty-worktree 结果包装成论文级最终复现。

---

## 1. 结论先行

### 1.1 Stop-loss 决策

**当前 `artifact_class` 硬类别阈值没有达到预注册机制门槛，应停止继续扫描 `quantile`、`min_conf`、`margin` 或相邻全局阈值。**

六组 matched-global 配对实验中：

- `artifact_class - matched-global fixed` 的 all mIoU 平均差为 `+0.000020055`，即 **`+0.002005 pp`**；
- 该效应只有预注册 `+0.1 pp` 门槛的约 **2.0%**；
- overlap 三组为方向一致但极弱的正差；
- disjoint 三组全部为极微弱负差；
- new mIoU 六组平均差为 `-0.000000454`，即 **`-0.000045 pp`**，没有新类收益；
- disjoint 的接收率已达到 `99.9936%–99.9994%`（artifact）与 `99.9982%–100%`（matched-global），阈值筛选实际上已经退化为近似全接收。

因此，之前 `artifact_class` 相对 fixed 0.6 的稳定正差，主要应归因于**降低总体阈值、提高旧类伪标签覆盖率**，而不是类别特异阈值本身。类别阈值最多存在一个只在 overlap 可见、量级远低于实际价值门槛的弱信号。

### 1.2 保留什么，停止什么

| 对象 | 决策 | 理由 |
| --- | --- | --- |
| 伪标签机制本身 | 保留并继续研究 | seed1 的 pseudo-on 相对 off 在 15-5 overlap/disjoint 均约有 `+0.5 pp` 方向性收益；15-1 overlap 也有更大链路收益的历史信号 |
| 较低总体阈值 / 高覆盖率 | 保留为强控制基线 | matched-global 与 artifact 几乎等效，说明覆盖率是当前主要有效因素 |
| `artifact_class` 类别硬阈值 | 降级为辅助实现 | 六组机制对照未达到门槛，不能再作为当前主攻方向 |
| 相邻低 q、`min_conf`、margin 微调 | 立即终止 | 已进入近全接收区，新增实验只能制造多重比较，不能回答机制问题 |
| 底层设计 | 需要升级 | 从二值接收转为连续可靠性进入 C-RLS 目标，避免把大量置信度信息压缩成 0/1 |

### 1.3 是否立即启动新实验

**本次不启动新的 hard-threshold 训练。**

这不是实验不足，而是已有六组结果已经触发 stop-loss。当前仓库尚无可靠性加权 C-RLS 的可执行配置；在没有完成接口设计、单元测试和 smoke test 前直接启动训练，会把代码正确性与方法效果混在一起。

另外，审计时 A100 80 GB 仅余约 `2.2 GB` 显存，GPU 利用率 `99%`，当前也不具备安全启动新训练的资源条件。磁盘剩余约 `99 GB`，后续正式矩阵开始前还需要设置空间门槛并控制 checkpoint 数量。

---

## 2. 比较问题与控制变量

本轮唯一研究问题为：

> 在 teacher checkpoint、总体分位点、seed 语义、模型、batch size、RHL 和 C-RLS 参数一致时，类别特异阈值是否稳定优于单一全局阈值？

定义：

\[
\Delta_s =
\operatorname{mIoU}(\text{artifact\_class}_s)
-\operatorname{mIoU}(\text{matched-global fixed}_s).
\]

共同参数：

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
```

step0 谱系：

| setting | BASE_SUBPATH | step0 SHA256 |
| --- | --- | --- |
| overlap | `20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32` | `6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913` |
| disjoint | `20260705_pseudo_15-5_disjoint_off_seed1_bs32` | `040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962` |

matched-global 的 `confidence` 逐 seed 直接取对应 artifact 的 `global_threshold`：

- overlap：`0.447265625 / 0.419921875 / 0.443359375`；
- disjoint：`0.029296875 / 0.048828125 / 0.048828125`。

artifact 历史组与 matched-global 组的相关训练代码差异审查结果：`trainer/trainer.py`、`utils/pseudo_label.py`、`network/Buffer.py` 和 `network/AnalyticLinear.py` 无相关差异；可执行链上只新增了 DataLoader `pin_memory` 环境开关。该差异不改变样本、标签、特征或解析更新公式，但历史 artifact manifest 为 dirty worktree，因此证据等级仍保持 `CAUTION`。

---

## 3. 六组原始结果

### 3.1 all / old / new mIoU

| setting | seed | artifact all | global all | Δ all（pp） | Δ old（pp） | Δ new（pp） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap | 1 | 0.708081556 | 0.708038216 | +0.004334 | +0.006169 | -0.001539 |
| overlap | 2 | 0.708016739 | 0.707996605 | +0.002013 | +0.002520 | +0.000394 |
| overlap | 3 | 0.704057647 | 0.703999264 | +0.005838 | +0.007256 | +0.001301 |
| disjoint | 1 | 0.694872400 | 0.694873145 | -0.000075 | +0.000055 | -0.000490 |
| disjoint | 2 | 0.694181001 | 0.694181540 | -0.000054 | -0.000093 | +0.000070 |
| disjoint | 3 | 0.692857843 | 0.692858088 | -0.000024 | -0.000030 | -0.000008 |

分 setting 汇总：

| setting | Δ all 均值（pp） | 样本标准差（pp） | artifact 胜出 | 判断 |
| --- | ---: | ---: | ---: | --- |
| overlap | +0.004062 | 0.001927 | 3/3 | 方向一致但实际量级极小 |
| disjoint | -0.000051 | 0.000025 | 0/3 | 与全局阈值数值等效且轻微反向 |
| 六组描述性汇总 | +0.002005 | 0.002561 | 3/6 | 远低于 `+0.1 pp` 门槛 |

六组 old / new 汇总：

| 指标 | artifact - global 均值（pp） | 结论 |
| --- | ---: | --- |
| old mIoU | +0.002646 | 只存在极弱旧类方向性信号 |
| new mIoU | -0.000045 | 新类没有受益 |
| all mIoU | +0.002005 | 不具备实际意义 |

### 3.2 接收率

| setting | seed | artifact 接收率 | global 接收率 | artifact - global（pp） |
| --- | ---: | ---: | ---: | ---: |
| overlap | 1 | 98.940897% | 98.998912% | -0.058015 |
| overlap | 2 | 98.975006% | 99.290226% | -0.315220 |
| overlap | 3 | 99.104727% | 99.119837% | -0.015110 |
| disjoint | 1 | 99.993586% | 100.000000% | -0.006414 |
| disjoint | 2 | 99.995426% | 99.998166% | -0.002740 |
| disjoint | 3 | 99.999369% | 99.999919% | -0.000550 |

这组结果还说明 matched-global 并非逐像素接收集合完全相同的严格对照，但总体覆盖已经非常接近。即使 overlap seed2 的接收率相差 `0.315 pp`，最终 mIoU 也只差 `0.002 pp`，进一步说明当前 C-RLS 对这部分边界像素的 hard selection 不敏感。

### 3.3 相对 fixed 0.6 的结果

| setting | artifact - fixed 0.6 all mIoU（pp） | 胜出 |
| --- | ---: | ---: |
| overlap 三 seed 均值 | +0.039487 | 3/3 |
| disjoint 三 seed 均值 | +0.032961 | 3/3 |
| 六组均值 | +0.036224 | 6/6 |

`artifact_class` 相对 fixed 0.6 的弱正向信号是真实存在的描述性事实，不能因为类别机制失败就忽略。但 matched-global 对照表明，这个收益主要来自**总体阈值从 0.6 下移并接收更多候选像素**。若类别差异是主要原因，则 artifact 相对 matched-global 不应在 disjoint 三组全部归零并反向。

### 3.4 相对 pseudo-off 的方向性证据

当前 off 只具备 seed1 可直接比较结果，不能冒充三 seed 结论：

| setting | pseudo off | artifact | artifact - off（pp） |
| --- | ---: | ---: | ---: |
| 15-5 overlap seed1 | 0.703091067 | 0.708081556 | +0.499049 |
| 15-5 disjoint seed1 | 0.689438156 | 0.694872400 | +0.543424 |

这两组结果支持保留伪标签研究线，但不能用于证明哪一种阈值分配方式更优。

### 3.5 Per-class 审查

- overlap 三 seed 的 artifact-global 平均差中，最大正差为 class 9：`+0.035625 pp`，其次为 class 11：`+0.020466 pp`；
- overlap 最大负差为 class 16：`-0.006040 pp`；
- disjoint 全部 21 类的平均绝对差均低于 `0.00055 pp`，基本达到数值等效；
- old/new 分组和单类结果均没有显示一个跨 setting 稳定的受益类别群。

因此，不能用个别 overlap 类别的微小正差为类别阈值机制续命。若未来可靠性审计发现 class 9/11 确有独特校准误差，应重新提出可证伪假设，而不是继续围绕现有 q 值扫参。

---

## 4. 第一性原理解释

### 4.1 当前硬阈值真正优化的对象

当前方法只改变：

\[
m_i=\mathbb{1}[p_i\ge \tau_{c_i}],
\]

然后把接受像素的类别写回标签，送入未加权的 C-RLS。它没有：

- 提高 teacher 的预测正确率；
- 校准 teacher confidence；
- 区分同一阈值以上像素的可靠性；
- 将置信度或 margin 传入解析目标；
- 对错误伪标签降低影响。

当接收率接近 100% 时，绝大多数 \(m_i=1\)，不同阈值策略只改变极少边界像素。此时类别阈值对充分统计量的影响必然很弱，实验结果与这一机制推导一致。

### 4.2 为什么 fixed 0.6 仍然略差

fixed 0.6 的接收率约为：

- overlap：`94.74%–94.78%`；
- disjoint：`90.31%–90.92%`。

artifact/matched-global 则接收约 `99%–100%`。当前结果说明被 fixed 0.6 丢弃的低置信度旧类候选中，整体仍含有对 C-RLS 有净正贡献的信息；但这不意味着它们都同等可靠，也不意味着类别分位阈值是利用这些信息的最佳方式。

### 4.3 底层矛盾

当前任务的矛盾不是“阈值小数点不够准确”，而是：

> 二值 hard selection 把 teacher 的 confidence、margin 和潜在校准差异压缩为接收/拒绝，随后所有被接收伪标签在 C-RLS 中权重完全相同。

因此，正确升级方向是让可靠性直接进入解析学习充分统计量，而不是继续精炼 hard threshold。

---

## 5. 统计与方法学审查

### 5.1 Statistical Findings

| Metric | Test | Value | Effect Size | Confidence |
| --- | --- | --- | --- | --- |
| artifact-global all mIoU | 六组配对描述性比较 | `+0.002005 pp` | 仅为预注册门槛的 2.0% | `RED_FLAG`（相对机制通过声明） |
| overlap artifact-global | 三 seed 配对描述性比较 | `+0.004062 pp` | 实际量级极小 | `CAUTION` |
| disjoint artifact-global | 三 seed 配对描述性比较 | `-0.000051 pp` | 数值等效 | `CAUTION` |
| artifact-fixed0.6 | 六组配对描述性比较 | `+0.036224 pp` | 稳定弱正向 | `CAUTION` |

未计算或汇报 p 值：每个 setting 只有三个 seed，且两个 setting 不是可随意合并的独立同分布重复。对这种样本量做显著性包装会产生虚假精确性。决策依据是预注册实际意义门槛、配对方向、跨 setting 一致性和机制可解释性。

### 5.2 Fallacy Scan

- Coverage: **11/11 checked**

| Fallacy | Severity | 本轮判断 |
| --- | --- | --- |
| Simpson's paradox | `CAUTION` | 六组总体为正，但 disjoint 子组为负；必须分 setting 报告，不能只报总均值 |
| Ecological fallacy | `NOTE` | 未从 setting 均值推断单类都受益；已检查 per-class |
| Berkson's paradox | `NOTE` | 候选集合受“background 且 teacher 预测旧类”筛选；结论只适用于该候选集合 |
| Collider bias | `NOTE` | 未通过额外控制变量回归推断因果 |
| Base rate neglect | `CAUTION` | 类别候选数高度不均衡；all mIoU 与像素接收率必须联合解释 |
| Regression to the mean | `NOTE` | 没有按极端 seed 选择重复；三 seed 均纳入 |
| Survivorship bias | `CAUTION` | 原 disjoint seed1 保存失败后已用独立目录恢复，未只报告成功 overlap |
| Look-elsewhere effect | `CAUTION` | 历史已做多轮 q 扫描；本轮使用预注册 matched-global 问题和门槛止损 |
| Garden of forking paths | `CAUTION` | 历史探索路径多；本轮未在看到结果后修改 `+0.1 pp` 门槛 |
| Correlation != causation | `NOTE` | matched-global 是受控实验，但 dirty-worktree 历史组限制最终因果强度 |
| Reverse causality | `NOTE` | 不适用于此受控算法比较 |

### 5.3 Reproducibility

- Method: 原始产物、manifest、teacher hash、参数和相关代码差异核验；disjoint matched-global 三组完成恢复
- Verdict: `PARTIALLY_REPRODUCIBLE`
- 原因：
  - matched-global 六组均有正式 JSON、stats 和 manifest；
  - 两个 setting 的 teacher SHA256 一致且已核验；
  - artifact 历史组 manifest 标记 dirty，未在当前 clean commit 上重放；
  - 由于机制效应远低于门槛，不再为证明失败结论耗费六组 clean replay；若未来方法升级通过筛选，正式论文矩阵必须在固定 clean commit 上重放所有基线。

---

## 6. 后续技术路线

### 6.1 三条候选路线

#### 路线 A：继续 hard threshold 扫参

- 内容：继续尝试 q、min confidence、margin 或更细全局阈值；
- 优点：无需改代码；
- 缺点：已被 matched-global 结果否定，无法增加机制信息；
- 决策：**拒绝**。

#### 路线 B：可靠性加权 C-RLS

- 内容：当前标签继续保持 hard class ID，但对伪标签像素使用连续 \(w_i\in[0,1]\)，可见真标签权重保持 1；
- 目标：

\[
\min_W \sum_i w_i\lVert\phi_iW-y_i\rVert_2^2+\gamma\lVert W\rVert_2^2;
\]

- 可用信息：teacher top-1 confidence、top1-top2 margin，不使用增量训练阶段不可见的 raw GT；
- 优点：直接修复二值阈值的信息损失，改动仍局限于伪标签结果、AIR.fit 和 RecursiveLinear.fit；
- 风险：confidence 未校准，简单 \(p_i^\alpha\) 可能只是另一种超参数化；
- 决策：**推荐作为下一条实现路线**。

#### 路线 C：teacher soft target C-RLS

- 内容：旧类伪标签不再转成 one-hot，而把 teacher 类别概率作为 soft target；
- 优点：保留最多 teacher 信息；
- 风险：需要明确背景、新类可见标签与 teacher 旧类分布的拼接语义，修改目标矩阵接口更深；
- 决策：**作为路线 B 失败后的二级升级，不与 B 同时铺开**。

### 6.2 推荐设计边界

第一版可靠性加权只做一个清晰机制，不引入类别阈值：

1. 候选集合仍定义为“当前标签为背景、teacher top1 为旧类、非 ignore”；
2. 低全局门槛只承担数值安全下界，不再承担主要建模职责；
3. 被接受伪标签权重由 confidence 与 margin 的预先定义函数给出；
4. 当前数据中真实可见标签权重恒为 1，ignore 权重为 0；
5. `RecursiveLinear.fit(X, y, sample_weight)` 使用 \(\sqrt{w_i}X_i\) 和 \(\sqrt{w_i}Y_i\) 实现加权最小二乘，保持原 Woodbury/充分统计量语义；
6. checkpoint、manifest、stats 必须记录 weighting strategy、参数和权重分布；
7. `sample_weight=None` 必须与现有实现数值等价，保护历史 checkpoint 和 baseline。

在实现前应先完成 raw-mask **只读审计**，输出 confidence/margin 分箱的 precision、coverage、classwise ECE 或可靠性曲线。raw mask 只能用于诊断和评价权重函数，不得进入增量训练标签或直接拟合测试协议专用参数，避免标签泄漏。

---

## 7. 后续实验安排与停止条件

### 阶段 W0：实现前诊断

目标：回答 confidence/margin 是否与伪标签正确率单调相关，以及类别间主要差异是校准偏移还是难度差异。

产物：

- 样本索引与 VOC raw mask 对齐审计；
- overlap/disjoint 各自的 reliability bins；
- per-class precision/coverage/ECE；
- 明确训练时不可使用的信息边界。

若 confidence 和 margin 与正确率都无稳定单调关系，则不进入简单权重实验，直接转向 soft target 或其他主线。

### 阶段 W1：单 seed 机制筛选

在代码、单元测试和 smoke test 通过后，仅新增下列训练：

| setting | baseline | candidate 1 | candidate 2 |
| --- | --- | --- | --- |
| 15-5 overlap seed1 | 复用 matched-global | confidence-weighted | confidence × margin-weighted |
| 15-5 disjoint seed1 | 复用 matched-global | confidence-weighted | confidence × margin-weighted |

不重跑已有 baseline，不增加 alpha 大网格。若需要权重形状参数，只允许从诊断结果预先选定至多两个值。

W1 通过标准：

- 两个 setting 平均 all mIoU 相对 matched-global 至少 `+0.1 pp`；
- 任一 setting 不得低于 baseline 超过 `-0.05 pp`；
- old mIoU 改善不能以 new mIoU 明显下降换取；
- stats 证明权重不是几乎全 0 或全 1。

若 W1 不通过：停止 reliability-weighted hard-label 路线，不做 seed2/3。

### 阶段 W2：三 seed 复核

只对 W1 最优且通过门槛的一个候选扩展 seed2/3：

- 2 settings × 2 additional seeds，共 4 个新训练；
- 与 matched-global、fixed 0.6、pseudo-off 分层比较；
- 固定 clean commit、独立 SUBPATH、完整 manifest。

W2 通过标准：

- 六组配对平均提升至少 `+0.1 pp`；
- 至少 5/6 方向为正；
- overlap/disjoint 不得出现机制方向冲突；
- 报告 old/new/per-class、权重分布和 raw-mask reliability audit。

若 W2 不通过：该方法只保留为负结果和工程模块，伪标签研究线转向 soft targets 或其他更有潜力路线。

### 阶段 W3：论文级重放

仅在 W2 通过后执行：

- 最新主线 clean replay；
- 15-1 overlap 长链路验证；
- 与 DeepLabV3+ 主线或最终 backbone 组合验证；
- 正式消融：off、fixed 0.6、matched-global、weighted；
- 报告多 seed 均值、标准差、原始 JSON 路径和运行成本。

---

## 8. 当前正式实验产物

matched-global overlap：

```text
checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p447265625_seed1_bs32_reuse20260627step0/
checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p419921875_seed2_bs32_reuse20260627step0/
checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p443359375_seed3_bs32_reuse20260627step0/
```

matched-global disjoint recovery：

```text
checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p029296875_seed1_bs32_recovery1_reuse20260705disjointstep0/
checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed2_bs32_recovery1_reuse20260705disjointstep0/
checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed3_bs32_recovery1_reuse20260705disjointstep0/
```

运行与汇总：

```text
configs/pseudo_label_matched_global_fixed_20260718.tsv
configs/pseudo_label_matched_global_disjoint_recovery_20260719.tsv
logs/pseudo_label/matched_global_20260718.log
logs/pseudo_label/matched_global_disjoint_recovery_20260719.log
logs/pseudo_label/matched_global_disjoint_recovery_20260719_summary.{md,csv,json}
```

所有 step1 都是解析式 C-RLS 更新，不存在传统 SGD 意义上的 best epoch、epochs_run 或 early stop。manifest 中的 `train_epoch=50` 属于统一 CLI 配置，不应误解为 step1 实际进行了 50 个 epoch 的梯度训练。

---

## 9. 最终研究表述

当前证据支持以下严谨表述：

> 在 VOC 15-5 overlap 与 disjoint 的三 seed 配对实验中，低总体阈值带来的高覆盖伪标签相对 fixed 0.6 呈现稳定但较小的正向收益；然而，在控制 seed-specific 总体分位阈值后，类别特异 hard threshold 相对 matched-global fixed 的平均提升仅为 0.0020 个百分点，并在 disjoint 中完全消失。因此，现有收益主要来自伪标签覆盖率，而非类别阈值分配。后续停止 hard-threshold 微调，保留伪标签机制，并转向将连续可靠性直接引入 C-RLS 解析目标。

不能表述为：

- “类别自适应阈值已稳定提升性能”；
- “六组方向一致”；
- “disjoint 也验证了类别阈值有效”；
- “只需继续微调 q 即可获得更大提升”。
