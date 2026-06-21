# RHL-SE 2.0：像素级可靠性感知的随机解析头集成方案

> 日期：2026-06-20
> 状态：方案定稿，待代码实现与实验
> 作用位置：多个已训练解析头的推理融合层，不改变单个解析头的训练目标

## 0. 核心结论

RHL-SE 的前一轮改进已经完成代码实现和实验验证。现有实现不是停留在“多随机种子取平均”，而是已经覆盖：

- 独立 `rhl_seed` 与全局随机种子解耦；
- probability/logit 两种融合空间；
- global、old/new group-wise、class-wise 权重；
- margin confidence gate；
- member、disagreement、oracle 等诊断；
- val-only 类别权重搜索，以及 test 只消费冻结权重的 P0 流程。

P0 在旧 RHL-SE seed1 参照上有小幅正收益，但没有超过当前更强的 Batch32/Buffer8196 单模型基线。其关键瓶颈不是“权重搜索还不够大”，而是类别级静态权重无法回答同一类别内部“当前像素应该信哪个成员”。与此同时，oracle 与 disagreement 证明多个成员之间仍有可利用的像素级互补性。

因此，RHL-SE 2.0 不应被降级为仅做收尾的轻量模块。它应继续作为独立研究支线，核心由“静态类别权重”升级为“像素级可靠性估计与路由”，但优先级低于直接改造单模型表示能力的 BOA-RHL 与 PGH-RHL。它与这些方法的边界是：BOA/PGH/归一化改变单成员，Snapshot 改变成员来源，RHL-SE 2.0 只负责在输出端判断每个像素应信任谁。

## 1. 前一阶段是否已经实现并验证

### 1.1 已实现能力

| 能力 | 当前实现位置 | 状态 |
|---|---|---|
| 独立 RHL 随机种子 | `network/Buffer.py`、`utils/parser.py` | 已实现 |
| probability/logit 融合 | `tools/eval_rhl_ensemble.py` | 已实现 |
| 全局权重 | `tools/eval_rhl_ensemble.py` | 已实现 |
| old/new 分组权重 | `tools/eval_rhl_ensemble.py` | 已实现 |
| class-wise 权重 | `tools/eval_rhl_ensemble.py` | 已实现 |
| margin confidence gate | `tools/eval_rhl_ensemble.py` | 已实现 |
| disagreement/oracle/member 诊断 | `tools/eval_rhl_ensemble.py` | 已实现 |
| val 权重搜索 | `tools/search_rhl_class_weights.py` | 已实现 |
| val 搜索、test 冻结评估 runner | `tools/run_rhl_se_val_driven.sh` | 已实现 |

### 1.2 已验证结果

VOC 15-5 step1 的 P0 结果为：

| 方法 | all mIoU | old mIoU | new mIoU |
|---|---:|---:|---:|
| 旧 RHL-SE seed1 参照 | 69.4606 | - | 42.1075 |
| P0 val-driven class-wise | 69.5382 | 77.9803 | 42.5236 |
| 当前更强 Batch32/Buffer8196 单模型 | 69.5598 | - | 43.2099 |

对照关系必须准确解释：

- 相对旧 seed1，P0 的 all/new 分别约 `+0.0777/+0.4161`；
- 相对 Batch32/Buffer8196 强基线，P0 的 all/new 分别约 `-0.0216/-0.6863`；
- P0 使用 Batch32 或 Batch16 评估结果基本一致，说明该 Batch Size 只是评估吞吐参数，不是成员训练 Batch Size，也不是性能差异来源。

成员分歧像素约占全部像素 `0.8668%`，但新类区域约为 `4.9279%`；oracle 达到 all `71.4056`、new `46.2872`。这说明可利用空间集中在少量困难像素和新类区域，静态全局/类别权重无法把它转化为实际收益。

## 2. RHL-SE 2.0 要解决的问题

### 2.1 静态类别权重的结构性限制

现有 class-wise 权重是 `w[k, c]`：对成员 `k`、类别 `c` 固定。它可以表达“类 17 总体更信 seed2”，但不能表达：

- 类 17 的内部区域信 seed2，边界区域信 seed3；
- 某像素上 seed2 虽然通常更强，但当前预测熵很高；
- 多数成员一致时不应切换，只有少数高置信成员反对时才切换；
- 新类与旧类预测的校准误差不同。

RHL-SE 2.0 的目标权重是 `w[k, c, h, w]`，并且只能由推理时可见的信息和 validation 校准结果产生。

### 2.2 BCE 输出不能直接被当成标准多类概率

当前训练使用逐通道 BCE。每个类别的 sigmoid 分数彼此独立，通道和不等于 1。若直接把 sigmoid 熵当作多类熵，会制造错误的“置信度”。RHL-SE 2.0 必须明确区分：

- 最终预测仍保持与原实现一致的 logit/score 语义；
- 可靠性特征可使用温度缩放后的 softmax logits，或明确归一化后的 sigmoid score；
- 两者只用于 gate，不应偷偷改变单成员输出和 baseline 定义。

### 2.3 可靠性学习容易发生 validation 过拟合

如果同一批 val 像素同时用于拟合 gate、选择超参和汇报结果，像素数量虽大，但图像相关性和类别不平衡会造成严重乐观偏差。数据划分必须以图像为单位，而不是随机打散像素。

## 3. 方法定义

设有 `K` 个成员，第 `k` 个成员对像素 `p` 输出 logits `z_k(p)`，类别为 `c`。RHL-SE 2.0 分为“成员校准、可靠性特征、像素路由、保守回退”四层。

### 3.1 成员校准

对每个成员学习 temperature `T_k`，可扩展为 old/new 两组温度：

```text
z_cal_k(p, c) = z_k(p, c) / T_k[group(c)]
```

第一版采用成员级或 old/new 组级 temperature，避免每类独立温度在小样本类别上过拟合。校准仅在 validation calibration split 上拟合。

### 3.2 像素级可靠性特征

对每个成员和像素构造：

1. `margin`：top1 与 top2 校准 logit 的差；
2. `entropy`：对校准 logits 做 softmax 后的归一化熵；
3. `class_reliability`：该成员在预测类别上的 validation 可靠性表；
4. `consensus`：该成员预测是否与多数成员一致，以及 top1 类上的跨成员方差；
5. `old_new_prior`：根据预测类别属于 old/new、成员在 old/new 上的 validation 表现形成先验；
6. `boundary_proxy`：当前像素与邻域预测的一致性，仅使用预测，不使用测试标签；
7. `member_identity`：成员固定偏置，用于表达总体能力差异。

`class_reliability` 需要平滑，不能直接使用单类 IoU 作为裸权重。建议以正确像素率、校准误差或 Brier 分量构造，并采用经验贝叶斯收缩：

```text
r_kc = (n_kc * observed_kc + lambda * global_k) / (n_kc + lambda)
```

### 3.3 两级 gate

#### 3.3.1 可解释规则 gate

先实现可审计的线性可靠性分数：

```text
s_k(p) = b_k
       + a_m * margin_k(p)
       - a_e * entropy_k(p)
       + a_r * class_reliability_k(pred_k(p))
       + a_c * consensus_k(p)
       + a_g * old_new_prior_k(p)

w_k(p) = softmax_k(s_k(p) / tau_gate)
```

融合 logits：

```text
z_ens(p, c) = sum_k w_k(p) * z_cal_k(p, c)
```

规则 gate 的系数通过 validation selection split 搜索，搜索对象是少量、可解释的系数组合，不做无边界随机搜索。

#### 3.3.2 轻量学习 gate

规则 gate 验证流程正确后，实现一个线性层或 `1x1` 小型 MLP。输入仅为上述可靠性特征，不读取 backbone feature，避免将推理融合变成新的大型分割模型。

训练目标由两部分组成：

```text
L_gate = BCE(z_ens, y) + lambda_switch * L_switch
```

`L_switch` 约束 gate 在没有可靠证据时接近强 baseline 成员，抑制无意义切换。学习 gate 必须使用按图像划分的交叉验证或固定三段式划分：

- calibration split：拟合 temperature 和可靠性表；
- gate-train split：拟合 gate；
- gate-select split：选择 gate 超参并报告 validation 指标；
- test：方案冻结后只执行一次。

数据不足时优先采用 5-fold image-level out-of-fold 预测，最终再用全部 val 拟合冻结参数。

### 3.4 保守回退机制

选定当前最强单成员作为 anchor。只有在以下条件成立时才允许 gate 改变 anchor 预测：

- 成员发生分歧；
- 非 anchor 成员相对 anchor 的可靠性优势超过阈值；
- gate 最大权重和次大权重差超过阈值；
- 目标类别校准样本数达到最低要求。

其目的不是人为保证不掉点，而是把可解释的“是否切换”变成可测对象。报告必须同时给出：

- switch ratio；
- beneficial switch ratio；
- harmful switch ratio；
- oracle switch capture；
- old/new/class-wise switch 分布。

## 4. 代码实现设计

### 4.1 文件与职责

| 文件 | 修改/新增职责 |
|---|---|
| `tools/eval_rhl_ensemble.py` | 增加校准 logits、pixel reliability feature、rule/linear gate 与切换诊断 |
| `tools/calibrate_rhl_se_reliability.py` | 从 val 成员输出拟合温度、可靠性表和 gate，生成冻结 JSON/PT |
| `tools/run_rhl_se_2.sh` | 串联 calibration、selection、test，输出独立日志与目录 |
| `tests/test_rhl_ensemble.py` | 覆盖权重归一化、无分歧回退、BCE 可靠性特征和配置校验 |
| `tests/test_rhl_se_calibration.py` | 覆盖 split 隔离、temperature、可靠性平滑和 artifact schema |

### 4.2 CLI 与配置

新增参数统一使用以下命名：

```text
--reliability_mode none|rule|linear
--calibration_artifact PATH
--reliability_features margin,entropy,class,consensus,old_new,boundary
--gate_temperature FLOAT
--gate_anchor_member INT
--gate_switch_threshold FLOAT
--gate_min_class_pixels INT
--gate_split_manifest PATH
--gate_cv_folds INT
--gate_seed INT
```

`none` 必须逐元素复现当前 class-wise/global baseline。所有参数及成员 checkpoint 列表写入结果 JSON，不允许只存在于 shell 环境变量。

### 4.3 成员清单与兼容性校验

新增 member manifest，至少记录：

```json
{
  "dataset": "voc",
  "task": "15-5",
  "step": 1,
  "method": "rhl",
  "members": [
    {
      "checkpoint": ".../final.pth",
      "rhl_seed": 1,
      "buffer": 8192,
      "gamma": 1.0,
      "rhl_init": "gaussian",
      "rhl_scale_mode": "legacy",
      "rhl_norm": "none"
    }
  ]
}
```

评估前必须比较 dataset、task、step、class count、model architecture、buffer、训练 batch、base checkpoint 和 feature method。默认只允许同构成员；BOA/PGH/Snapshot 异构成员需显式 `--allow_heterogeneous_members`，并在结果中标记。

### 4.4 中间产物

校准 artifact 必须包含：

- schema version；
- val split manifest 与图像级划分；
- member manifest/hash；
- temperature；
- class reliability 与样本数；
- gate 特征标准化参数；
- gate 参数；
- selection metric；
- 生成命令与 git commit。

不应默认长期保存所有全分辨率 logits。第一版可以流式累积校准统计；学习 gate 需要缓存时，使用分图像压缩文件并记录 dtype/shape/checksum。

## 5. 实验设计

### 5.1 固定基线

所有实验必须同时报告：

1. 当前最强单成员；
2. uniform logit ensemble；
3. 现有 val-driven class-wise P0；
4. 现有 margin gate；
5. RHL-SE 2.0 新 gate。

成员首先使用已有 seeds 1/2/3，避免把“重新训练得到更强成员”误归因于 gate。之后再用 BOA 或 PGH 成员验证泛化。

### 5.2 消融矩阵

| 编号 | temperature | margin/entropy | class reliability | consensus | old/new prior | gate |
|---|---|---|---|---|---|---|
| SE2-0 | 否 | 否 | 否 | 否 | 否 | uniform |
| SE2-1 | 是 | 否 | 否 | 否 | 否 | uniform |
| SE2-2 | 是 | 是 | 否 | 否 | 否 | rule |
| SE2-3 | 是 | 是 | 是 | 否 | 是 | rule |
| SE2-4 | 是 | 是 | 是 | 是 | 是 | rule |
| SE2-5 | 是 | 是 | 是 | 是 | 是 | linear |

边界代理在主特征确认有效后单独加入，不与全部特征一次性绑定。

### 5.3 评价指标

主指标：

- all/old/new mIoU；
- 21 类 per-class IoU；
- 三个随机重复的 mean/std。

可靠性指标：

- NLL、Brier score、ECE；
- disagreement 区域 mIoU；
- switch/beneficial/harmful ratio；
- oracle 与实际 gate 的差距；
- oracle gain capture ratio：`actual_gain / oracle_gain`。

效率指标：

- 单成员与 K 成员推理时延；
- gate 额外显存；
- calibration 时间与 artifact 大小。

### 5.4 判定规则

RHL-SE 2.0 被认为形成有效支线，需要同时满足：

- 相对强单成员 all mIoU 不下降，new mIoU 有稳定提升；
- 相对现有 class-wise P0，在至少两个重复上改善；
- harmful switch ratio 明显低于 beneficial switch ratio；
- validation 选择方向与 test 一致；
- 收益不是由成员配置不一致造成。

若 mean 不提升但校准、方差或困难区域稳定改善，应如实定位为可靠性/系统增强结果，不包装为单模型精度创新。

## 6. 与其他方案的边界和组合

| 方案 | 作用位置 | 与 RHL-SE 2.0 的关系 |
|---|---|---|
| BOA-RHL | 随机映射矩阵构造 | 先单独验证 BOA，再把不同 seed/初始化成员交给 gate |
| PGH-RHL | 随机特征后的语义原型分支 | 可产生更互补成员，但先验证单成员增益 |
| RHL 归一化新方案 | 隐空间缩放、解析目标加权 | 改变成员本身，不能在 gate 实验中无记录启用 |
| Snapshot | backbone checkpoint 来源 | 天然增加结构级成员差异，是 RHL-SE 的上游 |

组合的基本纪律是“成员方法先独立、路由方法后组合”。同一次实验中如果 BOA、PGH、归一化或 Snapshot 与 RHL-SE 共同启用，结果名称、manifest 和表格必须明确列出每一项开关。

## 7. 执行顺序

1. 为现有 seeds 1/2/3 建立 member manifest 和图像级 val split manifest。
2. 实现 temperature、margin、entropy、class reliability、consensus 的流式统计与单元测试。
3. 完成 rule gate，验证 `none`/uniform 对旧实现的逐元素兼容。
4. 在 val 内完成 calibration/gate-train/gate-select 隔离，冻结 artifact 后执行 test。
5. 增加 linear gate，并与 rule gate 做相同协议的消融。
6. 用 BOA 最佳成员或 PGH 最佳成员复验，判断 gate 是否能泛化到异构成员。
7. 在 Snapshot 成员可用后进行系统级最终组合。

RHL-SE 2.0 的研究价值不来自“集成三个模型”本身，而来自：在解析增量分割中，对随机头的像素级可靠性、校准误差和成员互补性进行显式建模，并证明该路由能从 oracle 空间中稳定回收真实增益。
