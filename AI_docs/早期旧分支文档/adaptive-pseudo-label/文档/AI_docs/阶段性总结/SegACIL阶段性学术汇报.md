# SegACIL / CFSSeg 阶段性学术汇报

## 摘要

本阶段工作围绕 CFSSeg 在 SegACIL 代码库中的 2D PASCAL VOC 类增量语义分割流程展开。当前课题目标不是简单替换更强分割网络，而是在原有“冻结表征 + 随机高维映射 + 递归闭式解分类头 + 伪标签修复语义漂移”的解析持续学习框架上，寻找可解释、可复现、能支撑论文叙事的改进点，并满足导师项目中关于集成学习系统的结项要求。

到目前为止，结论可以概括为四点。

第一，CFSSeg 原框架仍然是本课题最稳固的主干。它把增量阶段最容易遗忘的 dense classifier 从梯度训练改写为 C-RLS 递归闭式更新，用历史二阶统计量替代旧样本保存；RHL 则在冻结 encoder 之后补偿新类可分性。该主干在当前代码中的对应关系清晰，后续改进应优先围绕 RHL 表示空间、C-RLS 目标权重和伪标签监督质量展开。

第二，已完成的 RHL 输出归一化、多随机 RHL seed 集成、BOA-RHL、DeepLabV3+ control、自适应伪标签阈值等路线给出了清晰边界：RHL 归一化不适合作为主贡献；RHL-SE 有弱正向但成员多样性不足；BOA-RHL 的 orthogonal 分支有很小正信号但 antithetic 明显失败；DeepLabV3+ 的 `aspp_up` AIR feature 已可作为更强可选 base；伪标签 artifact 路线存在稳定但很小的正向信号，仍缺少跨 setting、多 seed 和论文主协议证据。

第三，当前最接近可保留的方法性进展是“offline artifact 自适应伪标签阈值”和“RHL 表示空间重构”。前者已经跑通完整工具链，但涨幅太小，暂时更适合作为辅助消融或标签修复模块；后者仍是更可能形成论文主线的方向，尤其是 prototype-guided RHL、orthogonal RHL 复核、类别感知 C-RLS 和 snapshot/heterogeneous ensemble。

第四，集成学习不宜目前直接作为主故事线中心。已实验的 RHL 多随机种子集成证明“只改变 RHL 随机子空间”的成员差异过弱，静态概率平均和 class-wise 权重都无法充分利用 oracle 上界。更合理的定位是：短期作为系统级涨点和诊断模块；只有在 snapshot、PGH-RHL、DeepLabV3+ 等异构成员引入后，才考虑把集成学习提升为主要贡献之一。

## 原工作剖析：CFSSeg 的问题线索、动机与方法架构

### 任务与核心矛盾

CFSSeg 面向 class-incremental semantic segmentation。与图像分类不同，语义分割是 dense prediction：每张图像会产生大量像素级样本，模型不仅要学习新类别，还要在后续测试中保持旧类别的像素级判别能力。

在该任务中，遗忘问题由三类因素叠加形成。

一是稳定性与可塑性矛盾。继续用 SGD/BP 更新大规模 encoder 会使旧类表征漂移；完全冻结 encoder 又会降低新类适应能力。

二是 dense prediction 的样本结构。分割头不是对整张图输出一个标签，而是对每个像素位置做分类。错误标签和类别不平衡会以更细粒度进入目标函数。

三是 semantic drift。在 `disjoint` 和 `overlap` 设置下，当前 step 的标注经常把旧类像素写成 background。如果直接用这些标签训练增量头，旧类会被系统性压向背景。

CFSSeg 的核心问题意识可以概括为：能否在不保存旧样本、不过多轮梯度优化增量头的条件下，用解析方式持续学习新的分割类别，并尽量避免旧类被背景污染破坏。

### 故事线与关键 insight

原工作的故事线不是“设计一个更复杂的分割网络”，而是把 ACIL 系列解析持续学习思想迁移到 2D/3D 密集分割任务。

关键 insight 是：在 encoder 冻结、RHL 映射固定、标签构造一致的条件下，分割头可以被看作大量像素特征到 one-hot 标签的线性岭回归问题。此时增量更新不必依赖 SGD，也不必显式保存旧样本；只要递归维护历史特征相关矩阵，就可以得到与联合闭式求解一致的解析头。

这一 insight 把持续学习中的“旧知识保存”从参数微调问题转化为二阶统计记忆问题。RHL 的作用则是在冻结表征之后提供高维非线性随机特征，使线性解析头仍有足够可塑性。

### 方法 pipeline 与代码映射

当前 SegACIL 代码中的主流程可对应为：

```text
step0:
  image
    -> DeepLabV3 / ResNet101
    -> logits
    -> BCE loss + SGD/BP
    -> 保存基础分割模型

step1:
  加载 step0 checkpoint
    -> 去掉原 classifier head
    -> 冻结 dense feature extractor
    -> dense feature
    -> RandomBuffer / RHL
    -> RecursiveLinear / C-RLS
    -> 先 realign 旧类解析头，再 fit 新类

disjoint / overlap:
  teacher model 预测旧类
    -> 从 background 候选中恢复高置信旧类伪标签
    -> 修改 C-RLS 的 label target matrix
```

关键代码位置如下。

| 论文机制 | 代码位置 | 作用 |
|---|---|---|
| VOC 任务划分 | `utils/tasks.py`, `datasets/voc.py` | 定义 `15-5`、`15-1`、`sequential/disjoint/overlap` 标签可见性 |
| step0 BP 训练 | `trainer/trainer.py` | 训练基础 DeepLab 模型 |
| AIR 包装 | `trainer/trainer.py` | 将 backbone、RHL、RecursiveLinear 串成解析学习模型 |
| RHL | `network/Buffer.py` | 固定随机高维映射，`weight` 为 buffer，不参与梯度训练 |
| C-RLS | `network/AnalyticLinear.py` | 保存 `R` 与 `weight`，递归闭式更新分类头 |
| 伪标签 | `trainer/trainer.py`, `utils/pseudo_label.py` | 用 teacher 输出修复 background 中的旧类监督 |

因此，后续方法设计的有效切入点主要有三类：改变 RHL 特征空间、改变 C-RLS 目标或样本权重、改变伪标签的阈值和质量控制。

## 第一性原理与对抗性审查

### 已由代码和结果文件确认的事实

- 当前最重要的本地协议是 VOC `15-5`：step0 学习 0-15，step1 新增 16-20，最终在 0-20 共 21 类上计算 mean IoU。
- `sequential` 设置下旧类标签可见，伪标签不是主要矛盾；`disjoint/overlap` 设置下旧类可能被映射为 background，伪标签才直接作用于 semantic drift。
- 当前伪标签改动不改变 backbone、teacher 或 test label，只改变 AIR fit 阶段的 label target。
- RHL-SE 已将全局 `random_seed` 与 `rhl_seed` 分离，因此 RHL-SE 的第一阶段实验可解释为“只改变 RHL 随机子空间”。
- DeepLabV3+ 的有效配置不是默认 decoder feature，而是 `aspp_up` 作为 AIR feature source。

### 由事实推出的判断

- RHL 输出归一化失败不说明 RHL 不重要，只说明“调尺度”没有触及 RHL 的核心能力，即高维非线性基函数和类别可分性。
- RHL-SE 静态融合收益小，主要不是代码没跑通，而是同构成员差异不足；pairwise disagreement 很低，但 oracle 上界显示新类区域仍存在可利用互补。
- 自适应伪标签的 q0.7 batch quantile 失败，主要因为过度保守；在当前 `15-5 overlap` step1 中，高召回恢复旧类 supervision 比只保留极高置信像素更重要。
- offline artifact 低分位策略略强于 fixed baseline，说明按类别候选分布放宽阈值有价值，但目前收益幅度不足以单独支撑论文主贡献。

### 尚未验证的假设

- artifact 伪标签的弱正向是否能跨 `overlap -> disjoint` 保持。
- artifact 伪标签在 `15-1` 或 `10-1` 多 step 论文主协议中是否会累计放大。
- BOA-RHL orthogonal 的弱正向是否能跨 seed、buffer 和 gamma 稳定存在。
- PGH-RHL 或类别感知 C-RLS 是否能给出比阈值搜索更强的单模型收益。
- 异构成员集成能否把当前 RHL-SE 的 oracle 上界转化为真实 test 提升。

### 如果当前结论是错的，最可能错在哪里

| 结论 | 主要风险 | 最小反证检查 |
|---|---|---|
| artifact 伪标签有弱正向 | 单 seed 偶然性、`15-5 overlap` 特殊性 | paired seed，对 `off/fixed/artifact` 同协议复验 |
| q0.7 batch adaptive 失败 | 只验证了一个过保守分位点 | 已补低 q 搜索；下一步看 raw-mask precision/recall |
| RHL-SE 不是主贡献 | 成员过同质，融合方式过弱 | 用 snapshot、PGH、V3+ 等异构成员重测 |
| BOA-1 值得复核 | 弱涨幅可能是噪声 | seed1/2/3 paired + rank/coherence 诊断 |
| V3+ 可作为更强 base | 当前只在 `15-5 sequential` control 上充分验证 | 方法迁移到伪标签/RHL 主线前做 golden replay |

## 阶段性进展与处理方式

### 基线复现与代码理解

已确认 SegACIL 当前实现与 CFSSeg 主机制基本一致。`15-1 sequential` 多 step 链路完整跑通，`15-5 sequential` 也形成了可复用的 DeepLabV3 对照。该阶段的主要价值是建立可比较的本地 protocol、确认代码中的训练分支和 checkpoint 语义，并明确 `sequential` 与 `overlap/disjoint` 的伪标签适用边界。

处理方式：作为后续所有方法实验的 protocol 基座保留。

### RHL 输出归一化

已实现 `none/l2/l2_sqrt/layernorm` 等 RHL 输出归一化方式，并完成 `gamma` 扫描。结果显示，归一化代码确实生效，但主对照下没有提升；`l2_sqrt` 反而降低 new mIoU，且 `gamma=0.1/1/10` 基本无差异。

机制解释是：原始 RHL 的 row norm 可能包含 objectness、响应强度和隐式置信度；强制归一化会把这些幅值信息抹掉。归一化只改变尺度，不增加新的随机基函数，也不引入类别结构。

处理方式：暂不作为论文主贡献，仅作为失败消融和后续 PowerNorm/trace-matched gamma 的背景证据。

### RHL-SE 多随机子空间集成

已实现独立 `rhl_seed`，固定全局 `random_seed`，只改变 RandomBuffer 的随机子空间，并扩展了 ensemble evaluation 工具。单成员结果显示 seed2/seed3 在新类上略强，但旧类下降；等权、加权和 class-wise val-driven 集成均只有弱正向。

关键诊断是：全局 pairwise disagreement 约 0.87%，新类区域约 4.93%；oracle all mIoU 可到约 71.4%，明显高于真实融合结果。这说明成员互补并非不存在，而是目前的成员差异太小、静态融合规则太弱。

处理方式：保留为集成学习辅助模块和诊断工具，不再把“只改 RHL seed 的静态集成”包装成主贡献。

### BOA-RHL

BOA-RHL 第一轮实验中，`orthogonal + legacy` 相对 batch32 baseline 有很小正向信号，new mIoU 也略有提升；但 `orthogonal + antithetic` 在不扩总 buffer 的设置下明显掉点。

机制解释是：正交初始化可能减少随机方向冗余，因此值得有限复核；antithetic 在总维度不变时牺牲了独立方向数，带来的表达力损失大于符号补偿收益。

处理方式：orthogonal 分支暂缓复核；antithetic 分支止损搁置。

### DeepLabV3+ control

DeepLabV3+ 路线定位为 stronger architecture control，而不是课题主方法。已定位早期 V3+ 掉点的核心原因：默认 decoder feature 更偏低层细节，不适合作为冻结表征后的 AIR 输入；改用高层 ASPP 语义特征，尤其 `aspp_up` 后，新类 mIoU 明显提升。

该路线已完成主线融合和 golden replay，可作为后续方法迁移的更强 base。但若直接把 V3+ 作为主贡献，会偏离“解析持续学习机制改造”的课题主线。

处理方式：已采纳为可选 base 和 stronger control，不作为核心创新点。

### 自适应伪标签阈值

伪标签路线已从 fixed baseline、batch quantile、自适应 low-q 搜索推进到 offline artifact。结果显示：

- 无伪标签 baseline 已建立。
- fixed confidence 0.6/0.7 明显强于无伪标签，说明恢复 background 中旧类监督是有效的。
- batch quantile q0.7 过保守，接收率约 30%，效果弱。
- batch_class q0.1 接近 fixed0.6，但没有超过。
- artifact_class q0.005/q0.01 达到当前最佳区间，略高于 fixed0.6。
- 强制 `min_conf=0.6` 会抹掉 artifact 的低阈值优势，结果退回 fixed0.6 附近。

机制解释是：当前 `15-5 overlap step1` 的主要矛盾不是旧类伪标签太脏，而是旧类 supervision 被 background 隐藏后召回不足。因此极低分位 artifact 的收益来自高召回和按类别放宽阈值，而不是更严格过滤。

处理方式：保留为弱正向辅助方法；必须通过 raw-mask audit、`disjoint` 复验、多 seed 和 `15-1 overlap` 论文协议后，才能决定是否进入论文主线。

## 关键实验结果总表

下表统一使用百分制 mIoU。详细参数、结果路径和小数原值见同目录下的 `SegACIL阶段性实验结果汇总.csv`。

| 阶段 | 关键设置 | step0 来源 | step0 结果 | buffer / batch | final all | old 0-15 | new 16-20 | 对照差距 | 处理方式 |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| DeepLabV3 baseline | `15-5 sequential` | 自训 step0 | 56.93 | 8196 / 32->32 | 69.56 | 77.79 | 43.21 | - | 当前强对照 |
| RHL norm | `l2_sqrt, gamma=1` | DeepLabV3 baseline | 未记录 | 8196 / 32->64 | 69.30 | 77.94 | 41.66 | -0.16 vs none | 搁置 |
| RHL-SE | class-wise val-driven | DeepLabV3 baseline | 未记录 | 8196 / 32->64 | 69.54 | 77.98 | 42.52 | +0.08 vs seed1 | 辅助模块 |
| BOA-RHL | orthogonal legacy | DeepLabV3 baseline | 未记录 | 8196 / 32->32 | 69.64 | 77.82 | 43.45 | +0.08 vs BOA-0 | 有限复核 |
| BOA-RHL | antithetic legacy | DeepLabV3 baseline | 未记录 | 8196 / 32->32 | 68.34 | 77.71 | 38.38 | -1.22 vs BOA-0 | 搁置 |
| DeepLabV3+ | `aspp_up` AIR feature | V3+ self step0 | 未记录 | 8196 / 32->16 | 70.36 | 77.93 | 46.13 | +0.80 vs DeepLabV3 | 已采纳为可选 base |
| Pseudo-label | off | overlap step0 | 60.51 | 8196 / 32->32 | 70.31 | 79.12 | 42.12 | - | overlap baseline |
| Pseudo-label | fixed confidence 0.6 | overlap step0 | 60.51 | 8196 / 32->32 | 70.77 | 79.68 | 42.28 | +0.46 vs off | 强 baseline |
| Pseudo-label | batch_class q0.7 | overlap step0 | 60.51 | 8196 / 32->32 | 70.45 | 79.29 | 42.16 | +0.14 vs off | 搁置 |
| Pseudo-label | batch_class q0.1 | overlap step0 | 60.51 | 8196 / 32->32 | 70.77 | 79.68 | 42.27 | +0.46 vs off | 暂缓 |
| Pseudo-label | artifact_class q0.01 | overlap step0 | 60.51 | 8196 / 32->32 | 70.81 | 79.72 | 42.29 | +0.50 vs off | 保留/需复验 |
| Pseudo-label | artifact_class q0.005 | overlap step0 | 60.51 | 8196 / 32->32 | 70.81 | 79.72 | 42.29 | +0.50 vs off | 保留/需复验 |

需要强调的是，不同 setting、不同 step0 来源、不同 batch size、不同 base architecture 的结果不能直接合并为同一 baseline 比较。上表中的差距只在相同或明确指定的对照口径内成立。

## 已遇到的问题、原因与可讨论解法

| 问题 | 当前原因判断 | 可能解法 |
|---|---|---|
| RHL 归一化没有涨点 | 只改变尺度，未改变高维基函数；可能抹掉有用幅值 | 转向 BOA/PGH/类别权重，而不是继续扫静态 norm |
| RHL-SE 涨幅太小 | 只改 RHL seed，成员错误模式高度相似 | 引入 snapshot、PGH、V3+ 等异构成员，再做 RHL-SE 2.0 |
| BOA antithetic 大幅掉点 | 总 buffer 不变时独立方向数减少 | 止损；如复验仅在扩容 buffer 后作为边界实验 |
| 伪标签 q0.7 失败 | top 30% 接收过保守，旧类召回不足 | 使用 artifact 低分位、fixed0.6、class-wise threshold |
| artifact 涨幅太小 | 当前只在单 seed、单 setting、单 step 验证 | raw-mask audit、`disjoint` 复验、`15-1 overlap` 多 step |
| V3+ decoder 掉点 | decoder feature 低层细节过强，不适合 AIR | 使用 `aspp_up`，并在迁移主方法前做 golden replay |
| 论文证据口径不足 | `15-5 overlap` 不能直接对比论文 `15-1/10-1 overlap` | 补 full protocol、multi-seed 和主表/消融 |

## 后续方向：从原论文模块改进到更深层机制

### 保守验证线

保守线的目标是排除实现错误和 protocol mismatch，不追求立刻造新方法。

1. 完成 raw-mask pseudo-label quality audit，统计 artifact q0.005/q0.01 的 precision、recall、confusion 和 per-class 贡献。
2. 完成 `15-5 disjoint` 的 off / fixed0.6 / artifact q0.005 三组复验，判断 artifact 信号能否跨 setting。
3. 对 fixed0.6 与 artifact q0.005/q0.01 做 paired seed 复验，避免把单 seed 小涨幅写成稳定结论。
4. 对 BOA orthogonal 做小矩阵复核，同时记录 random matrix rank、coherence、trace 和 condition 诊断。

### 进取探索线

进取线的目标是主动寻找更强机制，而不是继续围绕已摸清边界的 q 值微调。

1. PGH-RHL：在随机 RHL 外加入 train-only 类别原型相似度分支，使 RHL 从纯随机 feature lift 变成“随机基函数 + 语义锚点”的 hybrid feature。
2. 类别感知 C-RLS：在解析目标中引入 train-only class counts、inverse-sqrt/effective-number 权重，修正背景和旧类像素过多导致的新类拟合弱问题。
3. RHL-SE 2.0：不再只做静态概率平均，而是利用 entropy、margin、成员一致性、old/new prior 和 class reliability 做像素级 routing。

### 论文级证据线

论文线的目标是让结果能进入主表、消融和审稿问答。

1. 在 `15-1 overlap` 上补 off / fixed0.6 / artifact / 最佳 RHL 方法的完整多 step 结果。
2. 在 `10-1 overlap` 上做压力测试，验证方法在更长增量链路中是否仍成立。
3. 对最终候选做 multi-seed mean/std，并报告 old/new/all、per-class、accepted ratio、oracle/diagnostic。
4. 保留负结果：RHL norm、antithetic、V3+ cap 都可以作为“为什么不走这些路”的消融边界。

### 跳出既有框架的更深方向

从解析持续学习用于 2D/3D 分割的角度看，后续不应只局限于调 RHL 初始化或伪标签阈值。更深层的问题是：dense prediction 中哪些信息应进入“可递归闭式保存的统计记忆”，哪些信息应通过 teacher/pseudo label 修复，哪些信息必须来自可塑性更强的 feature space。

可讨论方向包括：

1. 结构化解析记忆：在 C-RLS 的像素级二阶统计之外，加入类别原型、边界区域统计或难例区域统计，使解析记忆不只是全局 `E^T E`。
2. 2D/3D 领域迁移：CFSSeg 原论文覆盖 2D 图像和 3D 点云，但当前仓库 3D 分支不完整。论文后续可以把 2D 中验证有效的“原型引导 RHL / 伪标签质量审计 / 解析集成”作为迁移到 3D 的方法论，而不是立即承诺完整 3D 实验。
3. 标签质量驱动的闭式更新：让 pseudo label 不只是硬替换 background，而是以 confidence/margin 进入加权 C-RLS 或 soft target matrix，从监督质量层面改造解析头训练目标。

## 集成学习专题

### SegACIL 上可用的集成方式

| 集成方式 | 多样性来源 | 优点 | 风险 | 当前定位 |
|---|---|---|---|---|
| RHL seed ensemble | 只改变 RandomBuffer 子空间 | 因果干净，符合 RHL 机制 | 成员过同质 | 已实验，辅助模块 |
| global seed ensemble | 数据顺序、增强、RHL、训练随机性 | 多样性可能更强 | 无法归因到 RHL | 系统上界对照 |
| snapshot ensemble | 不同 step0 backbone checkpoint | 表征差异更大 | 成本高，需保存 snapshot | 未来重点 |
| method ensemble | RHL/PGH/V3+/pseudo 多方法成员 | 多样性强 | 归因复杂 | 后期系统集成 |
| RHL-SE 2.0 routing | 像素级选择成员 | 有机会接近 oracle | 易过拟合 val | 等异构成员后推进 |
| TTA ensemble | 输入变换 | 实现简单 | 论文贡献弱 | 只作最终系统补强 |

### 多随机种子集成失败原因

已实验的 RHL-SE 并非完全失败，而是“信号存在但强度不足”。其原因有三层。

第一，成员差异来源太窄。固定 backbone、固定全局 seed、固定训练数据与解析流程，只改变 RHL seed，导致成员预测高度相似。

第二，静态融合规则无法利用局部互补。new-region disagreement 明显高于全局 disagreement，说明新类区域确有互补；但等权概率平均、logit 平均和 class-wise 权重都无法判断具体像素该信哪个成员。

第三，当前成员质量没有超过强单模型太多。即使融合能从 seed2/seed3 中提取一点新类收益，也会被旧类下降抵消。

后续措施不应是盲目增加 seed 数，而应提高成员多样性，并让融合规则具备像素级可靠性判断。

### 集成学习贡献应如何处理

当前建议是：短期不要把集成学习作为主故事线中心，而是作为系统级贡献和结项指标模块处理。

可行写法有两种。

第一种是“主方法 + 集成补强”。主贡献聚焦 RHL 表示空间重构和伪标签质量控制；集成学习作为最终系统模块，用于把多个可解释成员组合成结项系统。这样风险较低，也符合当前证据强度。

第二种是“解析学习中的随机子空间集成”。只有在 snapshot/PGH/V3+ 异构成员引入后，RHL-SE 2.0 能稳定超过强单模型，并能解释 oracle capture，才把集成学习写进主故事线。

目前更推荐第一种。现有 RHL-SE 结果不足以单独支撑主要创新，但足以说明课题已经围绕集成学习做了干净实现、失败诊断和后续升级路径。

## 当前建议

下一阶段应避免继续在已摸清边界的单点参数上细磨。建议优先执行三项工作。

1. 完成伪标签 raw-mask audit 与 `disjoint` 复验，决定 artifact 阈值是否保留为论文方法或降级为消融模块。
2. 将 RHL 方向从归一化转向 PGH-RHL、BOA-RHL 复核和类别感知 C-RLS，寻找更强单模型机制。
3. 将集成学习从同质 RHL seed 扩展到 snapshot/heterogeneous members，再评估 RHL-SE 2.0 是否有资格成为主贡献。

若短期要向导师汇报，一个稳妥表述是：

> 本阶段已完成 CFSSeg 原框架复现、关键模块定位、多个改进方向实现与边界验证。当前最有希望的主线不是简单阈值搜索或同质 seed ensemble，而是围绕解析分割的表示空间重构、标签质量控制和异构成员集成，形成“单模型机制提升 + 系统级集成补强”的论文与结项方案。
