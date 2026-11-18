# W1 可靠性加权 C-RLS 正式结果与路线决策

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Mode：`validate`
- 生成日期：2026-07-19
- Verification Status：`ANALYZED`
- Version：`weighted_crls_w1_final_v1`
- 正式汇总：
  - `logs/pseudo_label/weighted_w1_20260719_summary.json`
  - SHA256：`782d2e05aa2787e9044762bacb09d318bc40735cc61787c72ed5b78b63e2cf90`
- 总结论：`STOP`

---

## 1. 一句话结论

W0 已证明 teacher confidence / margin 能排序伪标签正确性，但 W1 证明这种排序能力不能通过简单连续样本加权稳定转化为足够的 all mIoU 收益。两种 weighting 都未达到预注册的 `+0.1 pp` 平均提升门槛，因此停止该路线，不启动 W2/W3，也不追加 seed2/3 或小超参数网格。

这不是代码错误、实验中断或协议不一致造成的阴性结果。四组实验均有唯一正式 JSON，accepted mask、W0 方向、manifest、teacher checkpoint hash、权重统计和结果完整性检查全部通过。

---

## 2. 实验完成情况

W1 固定候选集合、hard pseudo-label、teacher、threshold、seed、batch size、BUFFER 和 GAMMA，只改变 C-RLS 中伪标签样本的权重。

| setting | weighting | baseline all / old / new | W1 all / old / new | Δall (pp) | Δold (pp) | Δnew (pp) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| overlap | confidence | 0.708038 / 0.797124 / 0.422965 | 0.707979 / 0.796944 / 0.423291 | -0.0059 | -0.0179 | +0.0327 |
| overlap | confidence×margin | 0.708038 / 0.797124 / 0.422965 | 0.707879 / 0.796653 / 0.423804 | -0.0159 | -0.0471 | +0.0840 |
| disjoint | confidence | 0.694873 / 0.778822 / 0.426237 | 0.695312 / 0.778750 / 0.428312 | +0.0439 | -0.0072 | +0.2074 |
| disjoint | confidence×margin | 0.694873 / 0.778822 / 0.426237 | 0.695711 / 0.778475 / 0.430866 | +0.0838 | -0.0347 | +0.4629 |

两种 weighting 的跨 setting 平均 Δall：

| weighting | overlap Δall | disjoint Δall | 平均 Δall | `>= +0.1 pp` | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| confidence | -0.0059 pp | +0.0439 pp | +0.0190 pp | 否 | FAIL |
| confidence×margin | -0.0159 pp | +0.0838 pp | +0.0339 pp | 否 | FAIL |

除平均收益不足外，其余预注册检查均通过：

- 任一 setting 的 Δall 均未低于 `-0.05 pp`；
- 任一 setting 的 Δnew 均未低于 `-0.10 pp`；
- 权重未退化为超过 95% 的单端点质量；
- W0 reliability 方向一致；
- accepted counts 与逐类 counts 和 baseline 一致；
- manifest、result、teacher SHA256 与 source provenance 均正常。

因此自动汇总的唯一失败项是 `mean_all_delta`，正式 recommendation 为 `stop`。

---

## 3. 为什么 W0 正相关却没有转化为足够收益

### 3.1 “能判断对错”不等于“能判断训练价值”

W0 测量的是 confidence / margin 与伪标签 correctness 的关系。W1 优化的却是下式中的样本影响：

\[
\min_W\sum_i w_i\lVert\phi_iW-y_i\rVert_2^2+\gamma\lVert W\rVert_2^2.
\]

低 confidence 像素即使更容易出错，也可能来自边界、遮挡、小目标或旧类的困难区域。这些像素的标签可靠性较低，但对维持困难旧类的召回和决策边界仍可能有较高价值。按 correctness 排序后直接降低其梯度等价贡献，并不能保证 mIoU 最优。

### 3.2 overlap 中 confidence 权重接近“几乎不加权”

overlap 的 accepted ratio 为 `0.989989`。confidence 权重均值为 `0.9260`，中位数和 P90 均约为 `0.975`，约 68.7% 的权重质量集中在最高端区间。它与 matched-global 未加权解非常接近，因此 Δall 只有 `-0.0059 pp`，基本属于可忽略变化。

confidence×margin 提高了动态范围，权重均值降至 `0.8147`，但它主要放大了 old/new 权衡，而不是带来净收益。

### 3.3 disjoint 中出现真实但不够好的 old/new 权衡

disjoint 接受了全部候选像素。confidence×margin 的权重均值为 `0.7332`，明显低于 confidence 的 `0.8831`，因此机制确实改变了解析解，而非静默失效。

它对新类改善明显，尤其：

- class 18（sofa）：`+1.4836 pp`；
- class 16（pottedplant）：`+0.6090 pp`。

但同时削弱部分困难旧类：

- class 9（chair）：`-0.8534 pp`；
- class 5（bottle）：`-0.0578 pp`；
- class 10（cow）：`-0.0471 pp`。

all mIoU 是 21 类平均，old 类共有 16 类而 new 类只有 5 类。新类的大幅局部改善被旧类的小幅、广泛下降抵消，所以最终 Δall 只有 `+0.0838 pp`，跨 setting 平均只有 `+0.0339 pp`。

### 3.4 已确认事实与尚未确认假设

已确认：

- 权重只作用于 accepted pseudo-label 像素；
- visible GT 权重保持 1，ignore 像素权重为 0；
- C-RLS 使用 `sqrt(weight)` 对 X/Y 做等价 weighted ridge 更新；
- sample weight 与 label 一样用 nearest 对齐到 feature space；
- 四组 accepted mask 与对应 matched-global baseline 一致；
- 更强的 margin 加权同时带来更大的新类收益和更明显的旧类损失。

尚未通过专门审计确认：

- nearest 下采样后，各旧类在 feature space 的有效权重质量是否进一步失衡；
- 困难旧类的低权重像素是否主要集中在边界、小目标或少样本区域。

这些问题可以解释失败，但不改变 W1 已失败、不得进入 W2 的决策。

---

## 4. 严谨性与谬误扫描

- Overall Confidence：`CAUTION`
- 原因：四组协议和机制验证完整，但性能结论仍是 seed1 的筛选结果；可以否决进入 W2，不能声称该机制在所有 seed 上必然无效。
- Fallacy Scan：`11/11 checked`

| 类型 | 状态 | 说明 |
| --- | --- | --- |
| Simpson's paradox | NOTE | 已分别报告 overlap/disjoint，方向差异没有被总体平均隐藏。 |
| Ecological fallacy | N/A | 指标与推断单位均为当前数据集/协议实验。 |
| Berkson's paradox | N/A | 不涉及条件抽样相关性推断。 |
| Collider bias | N/A | 未通过控制变量回归推断因果。 |
| Base-rate neglect | NOTE | 同时报告 old/new/all，未用单一新类改善代替总体判断。 |
| Regression to mean | N/A | 不是按极端 seed 选择后的前后测设计。 |
| Survivorship bias | PASS | 预注册四组全部完成，没有只汇报成功行。 |
| Look-elsewhere effect | PASS | 两种 weighting 和 gate 均预注册，未在结果后追加小网格。 |
| Garden of forking paths | PASS | threshold、mask、seed、协议和判据均锁定。 |
| Correlation != causation | NOTE | W0 相关性没有被误写成 W1 性能因果保证。 |
| Reverse causality | N/A | 不涉及观察性因果方向。 |

---

## 5. 正式路线决策

### 5.1 立即停止

- 不启动 W2 的 overlap/disjoint seed2/3；
- 不启动 W3、15-1、DeepLabV3+ 或论文级扩展；
- 不做 alpha、floor、temperature、margin threshold、q 或 min_conf 小网格；
- 不把 `+0.0190 pp` 或 `+0.0339 pp` 描述为稳定涨点；
- 不因为 disjoint 新类改善而忽略 overlap 无收益和 old 类损失。

### 5.2 保留什么

- 保留 weighted C-RLS 实现及测试，作为已验证可用的基础能力；
- 保留 W0/W1 artifact、正式 JSON、manifest、summary 和 negative evidence；
- matched-global 可作为辅助伪标签模块保留，但不再作为当前核心研究线；
- 当前结论可用于论文中的设计选择、失败分析或补充实验，但不作为核心创新。

### 5.3 下一步优先级

主资源转向已有更高潜力的 RHL / 集成学习路线。自适应阈值研究线进入冻结状态。

soft-target C-RLS 不立即实现。它虽然是与 hard weight 不同的底层机制，但会引入 BCE sigmoid 非归一化、背景语义、visible GT 与 teacher target 融合等新的设计自由度。若未来重新开启，必须满足：

1. 形成独立的一页 first-principles 假设和 target 数学定义；
2. 明确它如何保留困难旧类的有效样本质量，而不是继续削弱样本；
3. 只做 overlap/disjoint seed1 的一次预注册 S0；
4. 仍以跨 setting 平均 `+0.1 pp` 为最低收益门槛；
5. S0 不通过即永久停止，不扩 seed、不扫参。

在此之前，不再启动新的伪标签训练实验。

---

## 6. 可复核路径

- 自动总汇：
  - `logs/pseudo_label/weighted_w1_20260719_summary.md`
  - `logs/pseudo_label/weighted_w1_20260719_summary.csv`
  - `logs/pseudo_label/weighted_w1_20260719_summary.json`
- 总日志：`logs/pseudo_label/weighted_w1_20260719.log`
- 四组正式结果目录：
  - `checkpoints/20260719_pseudo_15-5_overlap_globalfixed0p447265625_confweight_seed1_bs32_reuse20260627step0/`
  - `checkpoints/20260719_pseudo_15-5_overlap_globalfixed0p447265625_confmarginweight_seed1_bs32_reuse20260627step0/`
  - `checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p029296875_confweight_seed1_bs32_reuse20260705disjointstep0/`
  - `checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p029296875_confmarginweight_seed1_bs32_reuse20260705disjointstep0/`

