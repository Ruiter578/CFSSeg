# SegACIL 项目改进全景总结（新人版）

> 范围：SegACIL 的二维 PASCAL VOC 实验与方法改进。三维点云部分不在本文范围内。  
> 数据依据：CFSSeg 原论文、当前代码、`AI_docs`、`Codex_Plans`、完整测试结果 JSON、运行 manifest 和集成评估日志。  
> 配套明细：[SegACIL实验结果与方法对应表.csv](./SegACIL实验结果与方法对应表.csv)

## 0. 先看结论

SegACIL 已完成“复现基线、工程超参数摸底、RHL 第一轮改造、随机子空间集成、BOA-RHL、DeepLabV3+ 升级、伪标签阈值两阶段验证”这一整轮研究筛选。项目的工程指标已经达到：`15-5 sequential` 的 DeepLabV3+ 最佳完整测试集结果为 **70.3565% all mIoU**，高于 65.9% 的单模型目标和 67.0% 的系统目标。真正尚未完成的是论文方法目标：目前还没有一个“机制新颖、同协议严格配对、跨随机种子稳定、能在强基线上复现”的新增解析学习方法。

现阶段最可靠的判断是：

- **已经验证有效并进入主线**：DeepLabV3+ 的 `aspp_up` AIR 特征接口；固定阈值伪标签在 `overlap/disjoint` 和 `15-1 overlap` 协议中的整体作用。
- **完成工程摸底，但不能当作方法创新**：Batch Size、RHL 维度（`buffer`）与基础复现实验。
- **弱正向，证据不足**：BOA-RHL 的正交初始化；RHL 多随机子空间的静态集成。
- **已经验证无效并形成升级方案**：静态 RHL 归一化、在线 batch 分位伪标签、硬类别自适应阈值、RHL-SE 静态融合、DeepLabV3+ 默认 decoder 特征。
- **已经验证无效并止损**：BOA 的 antithetic 分支；DeepLabV3+ 的类别采样上限方案。
- **已有完整设计、尚未实验**：PowerNorm、CA-C-RLS、PGH-RHL-lite、RHL-SE 2.0、Snapshot Analytic Ensemble、可靠性加权 C-RLS。

因此，项目下一阶段不应继续细扫硬阈值或静态权重。最值得优先验证的是：把伪标签置信度作为连续权重写入 C-RLS 充分统计量；用配对多 seed 复核 BOA 正交初始化；再推进具有显式类别语义锚点的 PGH-RHL-lite。

## 1. 阅读本文前需要知道的四件事

### 1.1 `15-5` 与 `15-1` 是类别增量协议

PASCAL VOC 包含背景类和 20 个前景类。`15-5` 表示初始阶段学习 15 个前景类，后续一次加入 5 个新类；`15-1` 表示初始阶段学习 15 个前景类，随后分五次、每次加入 1 个新类。本文的 `step1`、`step5` 分别表示对应协议的最终增量阶段。

### 1.2 三种 setting 不能混为一谈

- `sequential`：当前阶段仍能看到旧类真值标签，不需要用伪标签纠正旧类被写成背景的问题。
- `overlap`：当前图像可能同时含旧类和新类，但旧类区域可能被标为背景。
- `disjoint`：当前任务的标注更严格地只保留新类，旧类更容易被当成背景。

当前代码在 `trainer/trainer.py:127-135` 显式禁止 `sequential` 启用伪标签。因此，伪标签实验不能被描述成 `15-5 sequential` 主线上的验证。

### 1.3 三个 mIoU 分别回答不同问题

- `旧类 mIoU`：模型保住旧知识的能力，即稳定性。
- `新类 mIoU`：模型学习新增类别的能力，即可塑性。
- `全类 mIoU`：背景、旧类和新类的总体平均，是项目主指标。

只看全类 mIoU 容易掩盖“旧类升、新类降”或相反的情况，所以本文同时报告三者。

### 1.4 “完整实验”的计数规则

本次审计共纳入 **110 项完整实验或完整测试集评估**：

- 一个独立 `SUBPATH` 只取最终增量步骤的最后一份合法 `test_results_*.json`，同一训练链的中间步骤不重复计数；
- `15-1` 从 step0 到 step5 视为一条完整多步实验链；
- 集成方法只有在完整测试集上输出 old/new/all mIoU 才计入；
- one-batch debug、只完成 step0 的任务、仅有权重搜索中间文件、重复诊断输出均不计；
- 一次 disjoint matched-global 运行因 checkpoint 写盘失败并留下截断文件，不是完整实验，已排除；其三个独立恢复任务均已完成并纳入；
- 审计时没有仍在运行的自适应伪标签任务。原先待完成的 matched-global 队列已经结束，因此只纳入其完整落盘结果。

原始指标优先取 `test_results_*.json`，参数优先取 `run_manifest.json`。旧实验缺少 manifest 时，仅使用能从 runner、目录名和阶段报告共同确认的参数，并在表中标为“历史参数记录不完整”。

## 2. CFSSeg 原论文究竟创新在哪里

### 2.1 它解决的核心矛盾

普通增量分割继续用反向传播更新整个网络时，新任务梯度可能覆盖旧知识。CFSSeg 将问题拆成两部分：

1. 初始阶段仍用梯度下降训练分割网络，获得有辨别力的 encoder；
2. 增量阶段冻结 encoder，不再反向传播，只更新一个解析分类头。

冻结 encoder 提供稳定性，但也限制学习新类的可塑性。RHL 用固定随机非线性高维映射补偿可塑性，C-RLS 用递归闭式更新保存历史统计，伪标签则处理 overlap/disjoint 中旧类被标成背景造成的 semantic drift。论文的核心流程可写成：

```text
初始阶段：图像 -> 分割网络常规训练 -> 保存 encoder

增量阶段：图像 -> 冻结 encoder -> RHL 固定高维映射
                            -> C-RLS 解析头递归更新 -> 所有已见类预测

overlap/disjoint：旧模型预测背景区域 -> 高置信旧类伪标签
                                      -> 与当前真值混合后更新 C-RLS
```

### 2.2 RHL：用固定随机高维映射恢复可塑性

论文将 encoder 特征 \(X\) 送入随机初始化且不训练的线性层，再经过 ReLU：

\[
E=\operatorname{ReLU}(X\Phi_E).
\]

直觉上，非线性高维展开使类别更可能在线性空间中分离。当前代码的对应实现为：

- `network/Buffer.py:39-75`：`RandomBuffer` 注册固定随机权重；
- `network/Buffer.py:96-122`：线性映射、ReLU 和可选静态归一化；
- `trainer/trainer.py:25-60`：`AIR` 把 `RandomBuffer` 与解析分类器串起来；
- `trainer/trainer.py:70-103`：分割特征展平后进入 RHL，再交给 C-RLS。

这里的“随机”不等于每个 batch 都重采样。权重初始化后固定，并随 checkpoint 保存；它不属于 optimizer 参数，也不接受梯度更新。

### 2.3 Ridge Regression 与 C-RLS：增量阶段不反向传播

初始解析头求解岭回归：

\[
\hat{\Phi}=(E^\top E+\gamma I)^{-1}E^\top Y.
\]

如果每个增量阶段都重新拼接全部历史特征，就违背无样本回放的目标。CFSSeg 因而维护历史自相关逆矩阵，并用 C-RLS 只结合上一阶段统计和当前特征递归更新。当前 `network/AnalyticLinear.py:90-165` 的 `RecursiveLinear.fit()` 实现了这一过程：过滤 `ignore_index=255`，构造 one-hot 目标，更新 \(R\) 和解析权重。

这也是 CFSSeg 的“闭式解”边界：闭式更新主要发生在增量分类头，并不意味着初始 encoder 也无需梯度训练。

### 2.4 Pseudo-labeling：纠正旧类被写成背景

在 overlap/disjoint 中，当前标注里的背景可能包含真正背景，也可能包含旧类。论文让上一阶段模型对这些背景像素预测；若置信度足够高且预测为旧类，就用旧类预测替换背景标签。当前实现进一步提供五种策略：

- `off`：关闭；
- `fixed`：所有旧类共用固定阈值；
- `batch_global`：每个 batch 用全局分位点；
- `batch_class`：每个 batch 分类别估计分位点；
- `artifact_class`：训练前离线统计类别阈值，固化为可审计 artifact。

核心代码位于 `utils/pseudo_label.py:10-66`、`215-308` 和 `311-404`；`trainer/trainer.py:615-648` 证明伪标签确实进入每个增量阶段的 `model.fit()` 之前。

### 2.5 原论文对核心模块的直接证据

论文在 PASCAL VOC overlapped `10-1` 上给出如下消融：

| 模型 | 旧类 mIoU(%) | 新类 mIoU(%) | 全类 mIoU(%) | 能说明什么 |
|---|---:|---:|---:|---|
| 完整 CFSSeg | 75.02 | 41.20 | 58.91 | RHL、C-RLS 与伪标签共同工作 |
| 去掉 RHL | 63.91 | 9.36 | 37.94 | RHL 对新类可塑性非常关键 |
| 去掉伪标签 | 71.83 | 36.19 | 54.86 | 伪标签缓解 semantic drift |

论文还报告，解析更新单轮耗时 43.25 秒，而 10 个 epoch 的 fine-tuning 总耗时 651.46 秒；对应 GPU 显存为 51.61 GB 与 59.55 GB。以上数字来自原论文 Table 5、Table 6，不与本仓库实验混用。

## 3. 当前 SegACIL 与原论文的关系

### 3.1 已保留的原始主干

当前项目仍以“冻结分割表征 + RHL + RecursiveLinear”为增量学习主干，保留 `gamma`、`buffer`、固定随机映射和解析递推。二维基线通常使用 DeepLabV3-ResNet101；项目历史默认 `buffer` 多为 8196，而原论文二维设置写的是 8192。这是需要在论文复现说明中明确记录的工程差异。

### 3.2 已扩展的能力

项目围绕三个层级开展改进：

- **单成员表示层**：RHL 归一化、BOA-RHL、PGH-RHL、DeepLabV3+ 特征接口；
- **解析目标层**：CA-C-RLS、可靠性加权 C-RLS；
- **系统融合层**：RHL-SE、RHL-SE 2.0、Snapshot 异构成员。

伪标签路线则横跨“标签构造”和“解析目标”两层：早期只决定某个背景像素是否被硬替换，下一版计划让可靠性权重直接进入 C-RLS。

### 3.3 DeepLabV3+ 升级不只是换一个模型名

DeepLabV3+ 同时包含高层 ASPP 特征与低层细节分支。实验表明，直接把融合后的 `decoder` 特征送入 AIR 会损害新类学习；最佳接口是 decoder 之前、已经上采样对齐的高层语义特征 `aspp_up`。当前代码在 `network/_deeplab.py:33-92` 暴露 `decoder/decoder_stride8/aspp/aspp_up`，并把 V3+ 的 `auto` 默认解析为 `aspp_up`。

因此，严谨表述应是 **“DeepLabV3+-ResNet101 + `aspp_up` AIR feature interface”**。它是已验证的更强架构控制变量，不应包装成解析持续学习的新理论贡献。

## 4. 所有改进方法与当前状态

| 方法或方案 | 作用位置 | 主要动机 | 当前证据 | 状态归类 | 下一步 |
|---|---|---|---|---|---|
| Batch Size 与 `buffer` 搜索 | 工程配置 | 找到稳定、可运行的基线区间 | 14 项完整敏感性实验；最优点 69.7343%，但 step0 谱系不完全一致 | 工程有效，不可作为方法因果结论 | 后续统一固定强基线，不再无边界扫参 |
| RHL 静态归一化 | RHL 输出 | 控制高维特征尺度 | `none/l2/l2_sqrt/layernorm` 与三档 gamma 均无稳定收益 | 已验证无效，已重构方案 | PowerNorm + trace-matched gamma |
| PowerNorm-RHL | RHL 输出 | 连续控制保留多少幅值 | 方案与实验矩阵已完成，代码未实现 | 尚未验证 | 与 `none`、完全归一化及 gamma 匹配对照 |
| CA-C-RLS | 解析目标 | 缓解背景/旧类多数像素主导最小二乘 | 方案已完成，代码未实现 | 尚未验证 | old/new、inverse-sqrt、effective-number 权重 |
| RHL-SE 1.x | 输出融合 | 利用不同 RHL 随机子空间的互补性 | 3 个成员、16 个完整融合评估；仅弱正向，未超强单模型 | 弱辅助，旧方案已完成升级设计 | 不再扫静态权重 |
| RHL-SE 2.0 | 输出融合 | 用像素级可靠性而非类别静态权重路由 | 方案已定稿，代码未实现 | 尚未验证 | 等更异构成员后验证 reliability gate |
| BOA-RHL 正交初始化 | 随机基几何 | 减少随机方向冗余 | seed1 相对 BOA-0 全类 +0.0789、新类 +0.2408 个百分点 | 弱正向，需多 seed | 完成 BOA-1.5 配对复核与尺度诊断 |
| BOA antithetic | 随机基几何 | 通过正负成对构造特征 | 两种尺度均显著退化 | 已验证无效，已止损 | 不再作为主线；大 `buffer` 只保留诊断价值 |
| PGH-RHL-lite | RHL 语义分支 | 给纯随机特征加入类别原型锚点 | 单/多原型、k-means、cosine/RBF 方案已完成 | 尚未验证 | 优先做 K=1 cosine，再扩展多原型 |
| Snapshot Analytic Ensemble | 上游 backbone | 制造比 RHL seed 更强的成员差异 | 方案已完成，尚无真实 snapshot 实验 | 尚未验证 | 建立中间 checkpoint 与显式 base checkpoint |
| DeepLabV3+ decoder | AIR 特征接口 | 利用低层细节增强分割 | 完整实验下降 | 已验证无效并完成升级 | 已被 `aspp_up` 替代 |
| DeepLabV3+ `aspp_up` | 架构与 AIR 接口 | 保留高层语义并对齐空间尺寸 | 70.3565%，黄金重放精确复现 | 已验证有效并采纳 | 作为 stronger architecture control |
| V3+ 类别采样上限 | 解析训练采样 | 降低显存开销 | 4096/8192 均明显降低总体精度 | 已验证无效 | 仅保留资源诊断用途 |
| 固定阈值伪标签 | 标签构造 | 恢复被标为背景的旧类像素 | 三类协议均优于关闭伪标签 | 已验证有效（伪标签整体） | 作为强基线保留 |
| 在线 batch 分位阈值 | 标签构造 | 随 batch 自适应置信度分布 | `batch_global/batch_class` 未超过 fixed 0.6 | 已验证无明显优势，已升级 | 已转为离线 artifact |
| `artifact_class` 硬类别阈值 | 标签构造 | 用离线、可复现的类别阈值取代在线波动 | 相对 fixed 仅约 +0.03–0.04 点；matched-global 后机制不成立 | 弱正向但核心主张被否定 | 停止 q/min_conf 微调 |
| 可靠性加权 C-RLS | 解析目标 | 保留连续置信度，让权重进入充分统计量 | 数学与接口方案已完成，代码未实现 | 尚未验证，当前最高优先级 | overlap/disjoint 各 3 seed 严格配对 |
| ConvNeXt+UPerNet、HRNet+OCR、SegNeXt | 更强二维底座 | 检验架构鲁棒性与性能上限 | 仅完成方案评估 | 长期未验证 | 等主方法成立后再做，避免偏离解析主线 |

## 5. 分方法实验结论

### 5.1 复现与工程超参数：完成，但要控制因果表述

二维复现结果分布在 68.9464%–69.6015% all mIoU。Batch Size 与 `buffer` 搜索的最高观测值为 69.7343%，对应 TRS 的 `buffer=8208` 复跑。由于这些历史 run 的 step0 来源、Batch Size 或参数持久化并非全部严格一致，能得出的结论是“8192–8216 附近均可稳定工作，32 左右的 step1 batch 可作为强基线”，不能声称 `buffer=8208` 本身带来确定增益。

推荐后续对照固定为：同一 step0 checkpoint、`batch=32`、`buffer=8196`、`gamma=1`、`rhl_norm=none`，并为每个方法保存完整 manifest。

### 5.2 RHL 静态归一化：否定的是强归一化，不是 RHL

同谱系主对照 `norm=none` 为 69.4606%。纯 L2 为 69.4505%，LayerNorm 为 69.4150%；`l2_sqrt` 在 `gamma=0.1/1/10` 下均约 69.3036%。一次 `l2_sqrt` 早期重跑达到 69.5146%，但其 batch 与主对照不同，不能归因给归一化。

失败原因有两层：

1. 完全归一化只改变尺度，没有增加新的随机基方向或类别语义；
2. 像素范数可能携带置信度、边界或可分性信息，强制统一范数会抹掉有用幅值。

已完成的重构把问题拆成两个可检验机制：PowerNorm 连续控制幅值压缩；CA-C-RLS 改变解析目标中不同类别样本的权重。二者不能再混成一次“换 norm”实验。

### 5.3 RHL-SE：存在 oracle 空间，但静态融合没有吃到

三个只改变 `rhl_seed` 的单成员分别为 69.4606%、69.4391% 和 69.4989%。16 项完整测试集融合评估中，最优约为 69.5386%，相对 seed1 约 +0.078 点，但仍低于 Batch32 强单模型 69.5598%。

诊断显示成员总体预测分歧约 0.87%，新类区域约 4.93%；逐像素 oracle 可达到约 71.41%。这说明“成员之间有少量互补性”成立，但“静态平均或 class-wise 权重足以回收互补性”不成立。对应原始诊断为 `logs/rhl_se_val_driven/20260618_p0_rhl_bs32_se_val_driven/test_diagnostics.json`。RHL-SE 2.0 已把方案升级为温度校准、margin、entropy、成员一致性、old/new prior 和像素级 reliability routing；在引入 PGH、Snapshot 或 V3+ 等异构成员之前，它应保持次优先级。

### 5.4 BOA-RHL：正交分支值得复验，antithetic 已止损

| 配置 | 旧类 mIoU(%) | 新类 mIoU(%) | 全类 mIoU(%) | 相对 BOA-0 |
|---|---:|---:|---:|---:|
| BOA-0 Gaussian legacy | 77.7942 | 43.2099 | 69.5598 | 基线 |
| BOA-1 Orthogonal legacy | 77.8225 | 43.4507 | 69.6387 | +0.0789 |
| BOA-2 Orthogonal antithetic | 77.9542 | 38.3768 | 68.3424 | -1.2174 |
| BOA-3 Antithetic + Kaiming | 77.6565 | 37.7973 | 68.1668 | -1.3930 |

BOA-1 的方向为正，但未达到预设的 all +0.10、新类 +0.30 个百分点门槛，且只有一个 seed。BOA-2/3 的失败不是 NaN、OOM 或程序崩溃，而更可能来自正负成对后独立方向数减半；尺度调整没有挽救结果。因此，BOA-1.5 只需对 orthogonal 做 paired seeds 2/3 与 trace/gamma 诊断，antithetic 不再扩成大网格。

### 5.5 DeepLabV3+：当前最明确的有效升级

| AIR 特征或变体 | 旧类 mIoU(%) | 新类 mIoU(%) | 全类 mIoU(%) | 结论 |
|---|---:|---:|---:|---|
| DeepLabV3 Batch32 强基线 | 77.7942 | 43.2099 | 69.5598 | 架构对照 |
| V3+ decoder | 78.1453 | 39.5933 | 68.9663 | 低层融合特征损害新类 |
| V3+ decoder_stride8 | 77.5043 | 40.1131 | 68.5985 | 降采样未解决语义失配 |
| V3+ aspp | 77.7115 | 45.0984 | 69.9464 | 高层语义特征有效 |
| V3+ aspp_up / 黄金重放 | 77.9276 | 46.1287 | 70.3565 | 已采纳 |

`aspp_up` 相对 V3 强基线提高 **0.7966 个百分点**，其中旧类 +0.1335、新类 +2.9188。收益主要来自新类可塑性。黄金重放使用 `auto -> aspp_up`、`buffer=8196`、`gamma=1`，精确得到 `Mean IoU=0.7035645831`。

新类逐类结果进一步说明提升并非所有类别一致：

| 新类 | DeepLabV3 IoU(%) | DeepLabV3+ `aspp_up` IoU(%) | 差值 |
|---|---:|---:|---:|
| 盆栽（pottedplant） | 26.9873 | 21.3104 | -5.6769 |
| 羊（sheep） | 57.7928 | 61.3111 | +3.5183 |
| 沙发（sofa） | 29.8677 | 35.1109 | +5.2432 |
| 火车（train） | 69.8898 | 75.3376 | +5.4478 |
| 电视/显示器（tvmonitor） | 31.5120 | 37.5736 | +6.0615 |

因此，V3+ 是可靠的 stronger base，但仍需承认盆栽类退化，并在主方法迁移时检查收益是否跨架构保持。

### 5.6 伪标签：整体有效，硬类别自适应机制无效

固定阈值 `0.6` 的严格对照结果如下：

| 协议与 setting | 关闭时全类 mIoU(%) | fixed 0.6 全类 mIoU(%) | 全类差值 | 旧类差值 | 新类差值 |
|---|---:|---:|---:|---:|---:|
| `15-5 overlap` seed1 | 70.3091 | 70.7731 | +0.4640 | +0.5582 | +0.1627 |
| `15-5 disjoint` seed1 | 68.9438 | 69.4639 | +0.5201 | +0.7698 | -0.2789 |
| `15-1 overlap` 完整链 | 69.7703 | 70.4120 | +0.6417 | +0.6645 | +0.5688 |

这证明的是“用旧模型恢复部分背景中的旧类像素”整体有效。disjoint 中总体增益来自旧类，新类略降，说明伪标签仍存在稳定性与可塑性的权衡。

关键新类逐类对照如下：

| setting | 策略 | 盆栽 | 羊 | 沙发 | 火车 | 电视/显示器 |
|---|---|---:|---:|---:|---:|---:|
| overlap | off | 18.7071 | 71.6179 | 17.4228 | 72.0788 | 30.7782 |
| overlap | fixed 0.6 | 18.7497 | 71.8222 | 17.6273 | 72.1970 | 31.0217 |
| disjoint | off | 26.3902 | 57.1104 | 29.5718 | 70.4051 | 30.9277 |
| disjoint | fixed 0.6 | 25.8374 | 57.1636 | 28.4963 | 70.4443 | 31.0691 |

在线 `batch_global/batch_class q=0.7` 只接收约 30% 候选像素，明显过于保守；降低 q 后接收率提高，但没有稳定超过 fixed 0.6。路线随后升级为离线 `artifact_class`，它相对同 seed fixed 0.6 的三 seed 平均差值为：

- overlap：约 **+0.0395 个百分点**；
- disjoint：约 **+0.0330 个百分点**。

这个弱正向仍不能证明“分类别阈值”本身有效，因为 artifact 同时改变了总体接收规模。最终 matched-global 对照把每个 seed 的全局固定阈值设为 artifact 的总体分位点：

- overlap 的 artifact 相对 matched-global 三 seed 平均为 **+0.004062 个百分点**；
- disjoint 平均为 **-0.000051 个百分点**；
- 六组总体平均仅 **+0.002005 个百分点**，artifact 只赢 3/6。

该结果远低于预设 +0.1 个百分点门槛，且两个 setting 方向不一致。结论已经足够明确：**伪标签整体有效，但 hard classwise adaptive threshold 的核心机制未成立。** 后续应停止 q/min_conf 微调，转向可靠性加权 C-RLS。

## 6. 110 项完整实验总表

下表按“方法大类 → 实验类型 → 实验名称”排序。路径中的日历信息不放入表格，而以证据编号连接到附录，满足无日期表格与结果可追溯两个要求。CSV 保留更多字段，适合筛选和二次统计。

| 证据 | 方法大类 | 实验类型 | 实验名称 | 协议/setting/最终步骤 | 模型 | 关键参数 | 旧类 | 新类 | 全类 | 相对差值 | 验证状态 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | 复现与基线 | 完整链路复现 | 15-1 TRS完整复现链 | 15-1 / sequential / step5 | DeepLabV3 ResNet101 | 最终step=5；buffer=8196；gamma=1；历史参数记录不完整 | 78.2523 | 41.5258 | 69.5079 | — | 完成 |
| E002 | 复现与基线 | 完整链路复现 | 15-1本地完整复现链 | 15-1 / sequential / step5 | DeepLabV3 ResNet101 | 最终step=5；buffer=8196；gamma=1；历史参数记录不完整 | 77.9050 | 40.2789 | 68.9464 | — | 完成 |
| E003 | 复现与基线 | 完整链路复现 | 15-5 TRS step1 batch16复现 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=16；历史参数记录不完整 | 78.4678 | 41.2291 | 69.6015 | — | 完成 |
| E004 | 复现与基线 | 完整链路复现 | 15-5 batch32强基线 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=32→32；buffer=8196；gamma=1；seed=1 | 77.7942 | 43.2099 | 69.5598 | — | 完成 |
| E005 | 复现与基线 | 完整链路复现 | 15-5 canonical基线A | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=32→64；buffer=8196；gamma=1；seed=1 | 78.0085 | 42.1075 | 69.4606 | — | 完成 |
| E006 | 复现与基线 | 完整链路复现 | 15-5基线复跑B | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=48；buffer信息未固化 | 77.8337 | 41.1692 | 69.1040 | — | 完成 |
| E007 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS batch16基线 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=16→16；buffer为当时默认 | 78.3916 | 40.9181 | 69.4693 | — | 完成的超参数实验 |
| E008 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS buffer8196复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8196；run2 | 78.2851 | 41.7986 | 69.5978 | — | 完成的超参数实验 |
| E009 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS buffer8200复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8200；run2 | 78.4416 | 41.3724 | 69.6156 | — | 完成的超参数实验 |
| E010 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS buffer8204复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8204；run2 | 78.4531 | 41.4769 | 69.6493 | — | 完成的超参数实验 |
| E011 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS buffer8208复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8208；run2 | 78.4826 | 41.7398 | 69.7343 | — | 完成的超参数实验 |
| E012 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS buffer8216 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8216 | 78.3479 | 40.9395 | 69.4411 | — | 完成的超参数实验 |
| E013 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | TRS基线复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer为当时默认 | 78.4416 | 41.3724 | 69.6156 | — | 完成的超参数实验 |
| E014 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | batch32与buffer8192 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=32；buffer=8192；gamma=1 | 77.7810 | 42.5245 | 69.3866 | — | 完成的超参数实验 |
| E015 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | batch32与buffer8196 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=32；buffer=8196；gamma=1 | 77.7942 | 43.2099 | 69.5598 | — | 完成的超参数实验 |
| E016 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | batch64到32与buffer8200 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=64→32；buffer=8200 | 77.8853 | 42.3663 | 69.4284 | — | 完成的超参数实验 |
| E017 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | batch64到32基线 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | batch=64→32；buffer为当时默认 | 78.2152 | 41.8541 | 69.5578 | — | 完成的超参数实验 |
| E018 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | buffer8216本地复跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=32；buffer=8216；run2 | 78.3479 | 40.9395 | 69.4411 | — | 完成的超参数实验 |
| E019 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | step1 batch48与buffer8196 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=48；buffer=8196；gamma=1 | 77.8857 | 41.8764 | 69.3121 | — | 完成的超参数实验 |
| E020 | 批量大小与RHL维度搜索 | 工程超参数敏感性 | step1 batch48旧buffer | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | step1 batch=48；buffer未固化 | 77.8337 | 41.1692 | 69.1040 | — | 完成的超参数实验 |
| E021 | RHL静态归一化 | 归一化与gamma消融 | RHL l2_sqrt gamma=0.1 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=l2_sqrt；gamma=0.1；batch=64；buffer=8196 | 77.9433 | 41.6567 | 69.3036 | -0.1570 | 已验证无效并完成重构方案 |
| E022 | RHL静态归一化 | 归一化与gamma消融 | RHL l2_sqrt gamma=1 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=l2_sqrt；gamma=1；batch=64；buffer=8196 | 77.9433 | 41.6567 | 69.3036 | -0.1570 | 已验证无效并完成重构方案 |
| E023 | RHL静态归一化 | 归一化与gamma消融 | RHL l2_sqrt gamma=10 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=l2_sqrt；gamma=10；batch=64；buffer=8196 | 77.9433 | 41.6566 | 69.3036 | -0.1570 | 已验证无效并完成重构方案 |
| E024 | RHL静态归一化 | 归一化与gamma消融 | RHL l2_sqrt早期重跑 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=l2_sqrt；gamma=1；batch=32；buffer=8196 | 77.7754 | 43.0802 | 69.5146 | +0.0540 | 不可归因 |
| E025 | RHL静态归一化 | 归一化与gamma消融 | RHL不归一化基线 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=none；gamma=1；batch=64；buffer=8196 | 78.0085 | 42.1075 | 69.4606 | +0.0000 | 基线 |
| E026 | RHL静态归一化 | 归一化与gamma消融 | RHL无仿射LayerNorm | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=layernorm；gamma=1；batch=64；buffer=8196 | 77.9574 | 42.0792 | 69.4150 | -0.0457 | 已验证无效 |
| E027 | RHL静态归一化 | 归一化与gamma消融 | RHL纯L2归一化 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | norm=l2；gamma=1；batch=64；buffer=8196 | 77.9390 | 42.2873 | 69.4505 | -0.0101 | 已验证无明显收益 |
| E028 | RHL随机子空间集成 | 单成员训练 | RHL子空间成员 seed1 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | random_seed=1；rhl_seed=1；batch=64；buffer=8196；gamma=1 | 78.0085 | 42.1075 | 69.4606 | +0.0000 | 完成的集成成员 |
| E029 | RHL随机子空间集成 | 单成员训练 | RHL子空间成员 seed2 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | random_seed=1；rhl_seed=2；batch=64；buffer=8196；gamma=1 | 77.8998 | 42.3649 | 69.4391 | -0.0215 | 完成的集成成员 |
| E030 | RHL随机子空间集成 | 单成员训练 | RHL子空间成员 seed3 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | random_seed=1；rhl_seed=3；batch=64；buffer=8196；gamma=1 | 77.9544 | 42.4415 | 69.4989 | +0.0383 | 完成的集成成员 |
| E031 | RHL随机子空间集成 | 完整测试集融合评估 | K2等权logit融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed2+seed3；logit；等权 | 77.9567 | 42.4183 | 69.4952 | +0.0346 | 弱正向辅助；未超batch32强基线 |
| E032 | RHL随机子空间集成 | 完整测试集融合评估 | K2等权概率融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed2+seed3；prob；等权 | 77.9577 | 42.4256 | 69.4976 | +0.0370 | 弱正向辅助；未超batch32强基线 |
| E033 | RHL随机子空间集成 | 完整测试集融合评估 | K2等权概率融合CPU复核 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed2+seed3；prob；CPU | 77.9577 | 42.4256 | 69.4976 | +0.0370 | 弱正向辅助；未超batch32强基线 |
| E034 | RHL随机子空间集成 | 完整测试集融合评估 | K3加权logit融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed1/2/3；logit；权重0.2/0.4/0.4 | 77.9837 | 42.3648 | 69.5030 | +0.0424 | 弱正向辅助；未超batch32强基线 |
| E035 | RHL随机子空间集成 | 完整测试集融合评估 | K3加权概率融合A | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed1/2/3；prob；权重0.1/0.45/0.45 | 77.9724 | 42.4030 | 69.5035 | +0.0428 | 弱正向辅助；未超batch32强基线 |
| E036 | RHL随机子空间集成 | 完整测试集融合评估 | K3加权概率融合B | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed1/2/3；prob；权重0.2/0.4/0.4 | 77.9848 | 42.3737 | 69.5060 | +0.0454 | 弱正向辅助；未超batch32强基线 |
| E037 | RHL随机子空间集成 | 完整测试集融合评估 | K3等权logit融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed1/2/3；logit；等权 | 77.9974 | 42.3163 | 69.5019 | +0.0413 | 弱正向辅助；未超batch32强基线 |
| E038 | RHL随机子空间集成 | 完整测试集融合评估 | K3等权概率融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 成员seed1/2/3；prob；等权 | 77.9985 | 42.3254 | 69.5049 | +0.0443 | 弱正向辅助；未超batch32强基线 |
| E039 | RHL随机子空间集成 | 完整测试集融合评估 | old/new分组概率融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | 旧类权重偏seed1；新类偏seed2/3 | 78.0029 | 42.3043 | 69.5033 | +0.0426 | 弱正向辅助；未超batch32强基线 |
| E040 | RHL随机子空间集成 | 完整测试集融合评估 | 手工class-wise概率融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | class-wise；基于test诊断权重 | 78.0022 | 42.3893 | 69.5229 | +0.0623 | 弱正向辅助；未超batch32强基线 |
| E041 | RHL随机子空间集成 | 完整测试集融合评估 | 置信度门控概率融合 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | seed1主模型；门限0.10；权重2/4/4 | 78.0191 | 42.2294 | 69.4977 | +0.0371 | 弱正向辅助；未超batch32强基线 |
| E042 | RHL随机子空间集成 | 完整测试集融合评估 | 验证集驱动class-wise完整评估 batch1 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | val_batch=1；val选权重；test冻结评估 | 77.9804 | 42.5218 | 69.5379 | +0.0773 | 弱正向辅助；未超batch32强基线 |
| E043 | RHL随机子空间集成 | 完整测试集融合评估 | 验证集驱动class-wise完整评估 batch1 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | val_batch=1；val选权重；test冻结评估 | 77.9804 | 42.5218 | 69.5379 | +0.0773 | 弱正向辅助；未超batch32强基线 |
| E044 | RHL随机子空间集成 | 完整测试集融合评估 | 验证集驱动class-wise完整评估 batch16 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | val_batch=16；val选权重；test冻结评估 | 77.9811 | 42.5228 | 69.5386 | +0.0780 | 弱正向辅助；未超batch32强基线 |
| E045 | RHL随机子空间集成 | 完整测试集融合评估 | 验证集驱动class-wise完整评估 batch32 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | val_batch=32；val选权重；test冻结评估 | 77.9803 | 42.5236 | 69.5382 | +0.0776 | 弱正向辅助；未超batch32强基线 |
| E046 | RHL随机子空间集成 | 完整测试集融合评估 | 验证集驱动class-wise测试 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | val选权重；test冻结评估 | 77.9804 | 42.5218 | 69.5379 | +0.0773 | 弱正向辅助；未超batch32强基线 |
| E047 | BOA-RHL | RHL初始化消融 | BOA-0 高斯基线 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | init=gaussian；scale=legacy；batch=32；buffer=8196；gamma=1；seed=1 | 77.7942 | 43.2099 | 69.5598 | +0.0000 | 基线 |
| E048 | BOA-RHL | RHL初始化消融 | BOA-1 正交初始化 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | init=orthogonal；scale=legacy；batch=32；buffer=8196；gamma=1；seed=1 | 77.8225 | 43.4507 | 69.6387 | +0.0789 | 弱正向，需复验 |
| E049 | BOA-RHL | RHL初始化消融 | BOA-2 正交反向成对 | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | init=orthogonal_antithetic；scale=legacy；batch=32；buffer=8196 | 77.7067 | 38.3768 | 68.3424 | -1.2174 | 已验证无效，已止损 |
| E050 | BOA-RHL | RHL初始化消融 | BOA-3 正交反向成对Kaiming | 15-5 / sequential / step1 | DeepLabV3 ResNet101 | init=orthogonal_antithetic；scale=kaiming；batch=32；buffer=8196 | 77.6573 | 37.7973 | 68.1668 | -1.3930 | 已验证无效，已止损 |
| E051 | DeepLabV3+升级 | 架构与特征源对照 | V3+ ASPP上采样特征 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=aspp_up；batch=32→16；buffer=8196 | 77.9276 | 46.1287 | 70.3565 | +0.7966 | 已验证有效并采纳 |
| E052 | DeepLabV3+升级 | 架构与特征源对照 | V3+ ASPP特征 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=aspp；batch=32→16；buffer=8196 | 77.7115 | 45.0984 | 69.9464 | +0.3866 | 已验证有效 |
| E053 | DeepLabV3+升级 | 架构与特征源对照 | V3+ buffer8200敏感性 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder；batch=32→16；buffer=8200 | 78.1346 | 39.3059 | 68.8897 | -0.6702 | 完成的敏感性实验 |
| E054 | DeepLabV3+升级 | 架构与特征源对照 | V3+ buffer8216敏感性 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder；batch=32→16；buffer=8216 | 78.1784 | 40.2370 | 69.1447 | -0.4151 | 完成的敏感性实验 |
| E055 | DeepLabV3+升级 | 架构与特征源对照 | V3+ decoder特征 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder；batch=32→16；buffer=8196 | 78.1453 | 39.5933 | 68.9663 | -0.5936 | 已验证无效并完成升级 |
| E056 | DeepLabV3+升级 | 架构与特征源对照 | V3+ decoder降采样特征 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder_stride8；batch=32→16；buffer=8196 | 77.5002 | 40.1131 | 68.5985 | -0.9614 | 已验证无效并完成升级 |
| E057 | DeepLabV3+升级 | 架构与特征源对照 | V3+ step1 batch32敏感性 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder；batch=32→32；buffer=8196 | 78.1453 | 39.5933 | 68.9663 | -0.5936 | 完成的敏感性实验 |
| E058 | DeepLabV3+升级 | 架构与特征源对照 | V3+ 类别采样上限4096 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=aspp_up；class_cap=4096；batch=32→16 | 69.4642 | 45.5214 | 63.7635 | -5.7963 | 已验证无效 |
| E059 | DeepLabV3+升级 | 架构与特征源对照 | V3+ 类别采样上限8192 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=aspp_up；class_cap=8192；batch=32→16 | 71.0924 | 47.0660 | 65.3718 | -4.1880 | 已验证无效 |
| E060 | DeepLabV3+升级 | 架构与特征源对照 | V3+主线黄金重放 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=auto→aspp_up；batch=32→16；buffer=8196；gamma=1；seed=1 | 77.9276 | 46.1287 | 70.3565 | +0.7966 | 已验证有效并采纳 |
| E061 | DeepLabV3+升级 | 架构与特征源对照 | V3+早期基线 | 15-5 / sequential / step1 | DeepLabV3+ ResNet101 | feature=decoder；batch=32→16；buffer=8196 | 78.1453 | 39.5933 | 68.9663 | -0.5936 | 已解释的负结果 |
| E062 | 伪标签与阈值 | 多步增量链 | 15-1 overlap 伪标签关闭 seed1 | 15-1 / overlap / step5 | DeepLabV3 ResNet101 | strategy=off；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1 | 78.9526 | 40.3870 | 69.7703 | — | 基线 |
| E063 | 伪标签与阈值 | 多步增量链 | 15-1 overlap 固定阈值0.6 seed1 | 15-1 / overlap / step5 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=98.54% | 79.6171 | 40.9558 | 70.4120 | +0.6417 | 已验证有效（伪标签整体） |
| E064 | 伪标签与阈值 | 多步增量链 | 15-1 overlap 固定阈值0.7 seed1 | 15-1 / overlap / step5 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=61.41% | 79.4150 | 40.5465 | 70.1606 | +0.3903 | 有效但非最优 |
| E065 | 伪标签与阈值 | 机制判别 | 15-5 disjoint matched-global固定阈值 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.029296875；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=100.00% | 77.8822 | 42.6237 | 69.4873 | +0.0001 | 机制判别完成 |
| E066 | 伪标签与阈值 | 机制判别 | 15-5 disjoint matched-global固定阈值 seed2 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.048828125；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=100.00% | 77.9783 | 42.0257 | 69.4182 | +0.0001 | 机制判别完成 |
| E067 | 伪标签与阈值 | 机制判别 | 15-5 disjoint matched-global固定阈值 seed3 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.048828125；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=100.00% | 77.8567 | 41.8591 | 69.2858 | +0.0000 | 机制判别完成 |
| E068 | 伪标签与阈值 | 机制判别 | 15-5 overlap matched-global固定阈值 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.447265625；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=99.00% | 79.7124 | 42.2965 | 70.8038 | -0.0043 | 机制判别完成 |
| E069 | 伪标签与阈值 | 机制判别 | 15-5 overlap matched-global固定阈值 seed2 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.419921875；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=99.29% | 79.6727 | 42.4060 | 70.7997 | -0.0020 | 机制判别完成 |
| E070 | 伪标签与阈值 | 机制判别 | 15-5 overlap matched-global固定阈值 seed3 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.443359375；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=99.12% | 79.6662 | 40.7480 | 70.3999 | -0.0058 | 机制判别完成 |
| E071 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.0 min_conf=0.0 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=99.99% | 77.8822 | 42.6233 | 69.4872 | +0.0233 | 弱正向但机制未成立，触发重构 |
| E072 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.0 min_conf=0.0 seed2 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=100.00% | 77.9782 | 42.0257 | 69.4181 | +0.0339 | 弱正向但机制未成立，触发重构 |
| E073 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.0 min_conf=0.0 seed3 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=100.00% | 77.8566 | 41.8591 | 69.2858 | +0.0417 | 弱正向但机制未成立，触发重构 |
| E074 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.005 min_conf=0.0 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.005；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=99.48% | 77.8820 | 42.6152 | 69.4852 | +0.0212 | 弱正向但机制未成立，触发重构 |
| E075 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.005 min_conf=0.6 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.005；min_conf=0.6；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.69% | 77.8582 | 42.6021 | 69.4639 | +0.0000 | 弱正向但机制未成立，触发重构 |
| E076 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.01 min_conf=0.0 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=98.94% | 77.8810 | 42.6129 | 69.4838 | +0.0199 | 弱正向但机制未成立，触发重构 |
| E077 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint artifact_class q=0.015 min_conf=0.0 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.015；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=98.40% | 77.8801 | 42.6105 | 69.4826 | +0.0186 | 弱正向但机制未成立，触发重构 |
| E078 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint 伪标签关闭 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=off；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1 | 77.0884 | 42.8810 | 68.9438 | — | 基线 |
| E079 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint 固定阈值0.6 seed1 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.69% | 77.8582 | 42.6021 | 69.4639 | +0.5201 | 已验证有效（伪标签整体） |
| E080 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint 固定阈值0.6 seed2 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=90.31% | 77.9620 | 41.9354 | 69.3842 | +0.4404 | 已验证有效（伪标签整体） |
| E081 | 伪标签与阈值 | 阈值策略与消融 | 15-5 disjoint 固定阈值0.6 seed3 | 15-5 / disjoint / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=90.92% | 77.8167 | 41.8117 | 69.2441 | +0.3003 | 已验证有效（伪标签整体） |
| E082 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.0 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.0；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=99.98% | 79.7154 | 42.2948 | 70.8057 | +0.0326 | 弱正向但机制未成立，触发重构 |
| E083 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.005 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.005；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=99.44% | 79.7185 | 42.2949 | 70.8081 | +0.0350 | 弱正向但机制未成立，触发重构 |
| E084 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.01 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=98.94% | 79.7185 | 42.2949 | 70.8082 | +0.0350 | 弱正向但机制未成立，触发重构 |
| E085 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.01 min_conf=0.0 seed2 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=98.98% | 79.6752 | 42.4064 | 70.8017 | +0.0409 | 弱正向但机制未成立，触发重构 |
| E086 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.01 min_conf=0.0 seed3 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.01；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=99.10% | 79.6734 | 40.7493 | 70.4058 | +0.0426 | 弱正向但机制未成立，触发重构 |
| E087 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.01 min_conf=0.6 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.01；min_conf=0.6；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=94.91% | 79.6761 | 42.2836 | 70.7731 | +0.0000 | 弱正向但机制未成立，触发重构 |
| E088 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.015 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.015；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=98.46% | 79.7174 | 42.2939 | 70.8070 | +0.0339 | 弱正向但机制未成立，触发重构 |
| E089 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.03 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.03；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=97.01% | 79.7118 | 42.2902 | 70.8019 | +0.0288 | 弱正向但机制未成立，触发重构 |
| E090 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.05 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.05；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=95.03% | 79.7027 | 42.2821 | 70.7930 | +0.0199 | 弱正向但机制未成立，触发重构 |
| E091 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.05 min_conf=0.6 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.05；min_conf=0.6；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=93.65% | 79.6772 | 42.2768 | 70.7723 | -0.0008 | 弱正向但机制未成立，触发重构 |
| E092 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.07 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.07；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=93.03% | 79.6929 | 42.2756 | 70.7840 | +0.0109 | 弱正向但机制未成立，触发重构 |
| E093 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.1 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.1；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.15% | 79.6790 | 42.2680 | 70.7716 | -0.0015 | 弱正向但机制未成立，触发重构 |
| E094 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap artifact_class q=0.2 min_conf=0.0 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=artifact_class；confidence=0.7；q=0.2；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=80.19% | 79.6218 | 42.2473 | 70.7231 | -0.0500 | 弱正向但机制未成立，触发重构 |
| E095 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_class q=0.1 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_class；confidence=0.7；q=0.1；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.00% | 79.6758 | 42.2691 | 70.7694 | +0.4603 | 已验证无明显优势，已升级路线 |
| E096 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_class q=0.3 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_class；confidence=0.7；q=0.3；min_conf=0.6；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=69.73% | 79.5595 | 42.2512 | 70.6766 | +0.3675 | 已验证无明显优势，已升级路线 |
| E097 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_class q=0.3 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_class；confidence=0.7；q=0.3；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=70.00% | 79.5596 | 42.2540 | 70.6773 | +0.3682 | 已验证无明显优势，已升级路线 |
| E098 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_class q=0.5 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_class；confidence=0.7；q=0.5；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=50.00% | 79.4257 | 42.2217 | 70.5676 | +0.2585 | 已验证无明显优势，已升级路线 |
| E099 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_class q=0.7 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_class；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=30.00% | 79.2941 | 42.1636 | 70.4535 | +0.1444 | 已验证无明显优势，已升级路线 |
| E100 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_global q=0.1 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_global；confidence=0.7；q=0.1；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.00% | 79.6345 | 42.2601 | 70.7358 | +0.4267 | 已验证无明显优势，已升级路线 |
| E101 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_global q=0.3 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_global；confidence=0.7；q=0.3；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=70.00% | 79.4469 | 42.2107 | 70.5811 | +0.2720 | 已验证无明显优势，已升级路线 |
| E102 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_global q=0.5 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_global；confidence=0.7；q=0.5；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=50.00% | 79.2672 | 42.1487 | 70.4294 | +0.1203 | 已验证无明显优势，已升级路线 |
| E103 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap batch_global q=0.7 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=batch_global；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=30.00% | 79.1423 | 42.1124 | 70.3257 | +0.0166 | 已验证无明显优势，已升级路线 |
| E104 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 伪标签关闭 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=off；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1 | 79.1179 | 42.1209 | 70.3091 | — | 基线 |
| E105 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.6 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=94.91% | 79.6761 | 42.2836 | 70.7731 | +0.4640 | 已验证有效（伪标签整体） |
| E106 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.6 seed2 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=2；接收率=94.74% | 79.6257 | 42.3932 | 70.7608 | +0.4517 | 已验证有效（伪标签整体） |
| E107 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.6 seed3 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.6；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=3；接收率=94.78% | 79.6225 | 40.7333 | 70.3632 | +0.0541 | 已验证有效（伪标签整体） |
| E108 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.7 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.7；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=90.35% | 79.6346 | 42.2703 | 70.7383 | +0.4292 | 有效但非最优 |
| E109 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.8 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.8；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=84.70% | 79.5830 | 42.2592 | 70.6964 | +0.3873 | 有效但非最优 |
| E110 | 伪标签与阈值 | 阈值策略与消融 | 15-5 overlap 固定阈值0.9 seed1 | 15-5 / overlap / step1 | DeepLabV3 ResNet101 | strategy=fixed；confidence=0.9；q=0.7；min_conf=0.0；batch=32；buffer=8196；gamma=1.0；seed=1；接收率=76.18% | 79.5024 | 42.2393 | 70.6302 | +0.3211 | 有效但非最优 |

## 7. 方法与结果的一一对应判定

| 判定类别 | 对应方法 | 为什么这样判定 | 是否进入论文主结论 |
|---|---|---|---|
| 已验证有效并采纳 | V3+ `aspp_up` | 完整特征源消融、主线融合和黄金重放一致；相对强 V3 +0.7966 点 | 可作为架构控制与有效工程成果，不作为解析方法主创新 |
| 已验证有效 | 固定阈值伪标签整体 | overlap/disjoint/15-1 均优于 off | 可作为 semantic drift 基础机制 |
| 工程有效 | Batch Size、`buffer` 稳定区间 | 多次完整运行建立稳定基线，但历史谱系未完全对齐 | 只写实验配置依据 |
| 弱正向待复验 | BOA orthogonal | 单 seed all/new 为正，但低于门槛 | 暂不可写成有效方法 |
| 弱辅助 | RHL-SE 1.x | 最优静态融合略高于 seed1，但未超 Batch32 强单模型 | 作为负结果、oracle 诊断和 SE2 动机 |
| 已验证无效并完成升级 | RHL 静态归一化 | 同谱系多 norm、多 gamma 均无收益 | 旧方案写负结果；升级为 PowerNorm/CA-C-RLS |
| 已验证无效并完成升级 | batch 分位与 hard classwise threshold | matched-global 机制判别未达门槛且 setting 冲突 | 停止硬阈值主张；升级为可靠性加权 C-RLS |
| 已验证无效并完成升级 | V3+ decoder feature | decoder/stride8 均不佳，aspp/aspp_up 解释并修复 | 作为接口消融 |
| 已验证无效并止损 | BOA antithetic | 两种尺度均大幅退化 | 负结果，不继续主线 |
| 已验证无效，暂无必要重构 | V3+ class cap | 节省资源但显著损害精度 | 仅保留资源诊断 |
| 已设计未验证 | PGH-RHL-lite | 缺少 prototype bank 实现与正式结果 | 候选单模型主方法 |
| 已设计未验证 | PowerNorm、CA-C-RLS | 尚无代码与实验 | 候选尺度/目标函数方法 |
| 已设计未验证 | 可靠性加权 C-RLS | matched-global 已触发升级条件 | 当前最高优先级 |
| 已设计未验证 | RHL-SE 2.0、Snapshot | 当前成员多样性不足，基础设施未实现 | 后期系统线 |
| 长期未验证 | 其他 CNN 分割底座 | 成本高且易偏离解析学习主线 | 等主方法成立后补充 |

## 8. 总任务推进情况

### 8.1 已完成

- CFSSeg 二维闭式增量链路可以稳定运行，`15-5` 与 `15-1` 均有完整结果；
- RHL、C-RLS、伪标签在当前代码中的实际执行路径已明确；
- DeepLabV3+ 已成为一等可选模型，`auto -> aspp_up` 已通过黄金重放；
- 静态 RHL 归一化、RHL-SE 1.x、BOA 第一轮、伪标签 Phase A/Phase B 与机制判别均已完成；
- run manifest、伪标签 artifact、独立 `SUBPATH` 与汇总工具已显著改善可复现性；
- 项目结项指标已超过：当前最佳 `15-5 sequential` 单模型 70.3565% 高于 65.9%，也高于 67.0% 的系统目标。

### 8.2 尚未完成

科研层面的总目标不是“再找到一个最高点”，而是得到一个能解释、能复现、能跨 seed 的新机制。目前尚缺：

- 至少一个新增解析学习机制在强基线上稳定提高 new/all mIoU；
- 严格同 checkpoint、同协议、同训练预算的多 seed mean/std；
- 新方法的机制诊断与指标变化方向一致；
- 方法独立有效后再做组合，避免无法归因；
- 将最终有效方法迁移到 V3+，验证它不是只对某个 feature distribution 有效。

因此，若按工程验收，项目已达标；若按论文创新验收，项目仍处在“完成第一轮筛选，进入第二轮机制重构”的阶段。

## 9. 未来工作与实验量

### 优先级 A：可靠性加权 C-RLS

硬阈值只决定“收或不收”，会丢掉连续置信度。第一版应令每个伪标签像素得到 \(w_i\in[0,1]\)，并让 \(\sqrt{w_i}\phi_i\) 与 \(\sqrt{w_i}y_i\) 进入 C-RLS 的 \(A\) 与 \(B\) 充分统计量。

建议实验量：

- 筛选：overlap/disjoint × 3 seeds，共 6 个方法 run；
- 若两个 setting 方向一致，再补同 seed fixed/off 或 clean replay，最多约 6 个；
- 必须报告 all/old/new、逐类、接受率、权重分布、有效样本量与矩阵稳定性。

### 优先级 B：BOA-1.5 有界复验

现有 seed1 BOA-0/1 可复用，只需新增 seeds 2/3 的 Gaussian 与 orthogonal 配对，共 4 个主要 run。先冻结 `buffer=8196`、`gamma=1`、同一 step0；若均值仍为正，再做 trace-matched gamma。antithetic 不进入主网格。

### 优先级 C：PGH-RHL-lite

先完成 K=1 cosine 的 prototype bank、无标签泄漏测试和约 4 个筛选 run；只有单原型显示正向，再推进 K=2/4 k-means 与 RBF sigma。它比继续扩静态 ensemble 更接近“单模型解析表示改进”。

### 优先级 D：PowerNorm 与 CA-C-RLS

建议先拆开验证，再做 2×2：

- PowerNorm：`beta` 端点与中间点、固定 gamma 与 trace-matched gamma；
- CA-C-RLS：old/new、inverse-sqrt、effective-number 三类权重；
- 初步约 6–10 个筛选 run，正向后再多 seed。

### 优先级 E：异构成员与系统融合

只有当 BOA、PGH 或加权 C-RLS 产生质量合格且错误互补的成员时，再推进：

- Snapshot：约 3 个 step0 候选 × plain step1；
- RHL-SE 2.0：uniform、class-wise、rule gate、linear gate 四类冻结测试；
- 最终系统结果必须与最佳单成员分开报告。

### 优先级 F：迁移到 V3+

选出一个有效方法后，执行小规模架构鲁棒性验证：

- V3 方法 off/on；
- V3+ `aspp_up` 方法 off/on；
- 必要时补一个 seed。

这一步回答“方法是否依赖 V3 特征分布”，而不是重新开展无边界架构搜索。

## 10. 代表性复现入口与产物

### 10.1 DeepLabV3+ 黄金配置

代表性 runner：

```bash
SUBPATH=<新的独立目录> \
BASE_SUBPATH=<V3Plus-step0目录> \
MODEL=deeplabv3plus_resnet101 \
AIR_FEATURE_SOURCE=aspp_up \
BATCH_SIZE=16 \
BUFFER=8196 \
GAMMA=1 \
bash ../SegACIL_deeplabv3plus/run_v3plus_air.sh
```

已验证结果：

- 输出目录：`../SegACIL_deeplabv3plus/checkpoints/20260624_v3plus_integration_golden_replay_14fc116/voc/15-5/sequential/step1/`
- checkpoint：同目录 `final.pth`
- 测试结果：同目录 `test_results_20260624_153910.json`
- manifest 记录：`requested_air_feature_source=auto`，`resolved_air_feature_source=aspp_up`

### 10.2 RHL 对照

```bash
SUBPATH=<新的独立目录> \
BASE_SUBPATH=<固定step0目录> \
DEFAULT_BATCH_SIZE=32 \
BUFFER=8196 \
GAMMA=1 \
RHL_NORM=none \
RHL_SEED=1 \
bash run_rhl_norm.sh
```

正式新实验应优先使用当前 `run.sh` 的 manifest 能力，并显式记录 `MODEL/AIR_FEATURE_SOURCE/BASE_SUBPATH/SUBPATH`。历史 RHL 结果路径通过附录证据编号追溯。

### 10.3 伪标签机制判别

批量入口：

```bash
DRY_RUN=0 bash tools/run_pseudo_label_matched_global_20260718.sh
```

该 runner 会校验 step0 checkpoint SHA256、使用独立 grid、逐项分配 `SUBPATH`，并在结束后生成 Markdown/CSV/JSON 汇总。完整恢复结果为 E065–E067；overlap 结果为 E068–E070。

## 11. 证据索引

CSV 与正文实验总表只写证据编号；下列索引给出每项结果的真实相对路径。路径名保留历史命名，可能包含运行时标识。所有指标均可直接从对应 JSON 或完整测试集日志复核。

- `E001`：`checkpoints/1128_trs/voc/15-1/sequential/step5/test_results_20260509_083832.json`
- `E002`：`checkpoints/1128/voc/15-1/sequential/step5/test_results_20260604_224745.json`
- `E003`：`checkpoints/0615_step1_bs16_trs/voc/15-5/sequential/step1/test_results_20260616_234601(tmux 615-segacil-bs16).json`
- `E004`：`checkpoints/20260607/voc/15-5/sequential/step1/test_results_20260607_203548.json`
- `E005`：`checkpoints/20260606/voc/15-5/sequential/step1/test_results_20260607_104912.json`
- `E006`：`checkpoints/20260616/voc/15-5/sequential/step1/test_results_20260616_223811.json`
- `E007`：`checkpoints/20260621_baseline_bs16_16_trs/voc/15-5/sequential/step1/test_results_20260622_083613.json`
- `E008`：`checkpoints/20260626_buffer8196_step1_32_run2_trs/voc/15-5/sequential/step1/test_results_20260626_214010.json`
- `E009`：`checkpoints/20260624_buffer8200_step1_32_run2_trs/voc/15-5/sequential/step1/test_results_20260624_172136.json`
- `E010`：`checkpoints/20260622_buffer8204_step1_32_run2_trs/voc/15-5/sequential/step1/test_results_20260624_104111.json`
- `E011`：`checkpoints/20260625_buffer8208_step1_32_run2_trs/voc/15-5/sequential/step1/test_results_20260625_213931.json`
- `E012`：`checkpoints/20260622_buffer8216_step1_32_trs/voc/15-5/sequential/step1/test_results_20260623_102140.json`
- `E013`：`checkpoints/20260622_trs/voc/15-5/sequential/step1/test_results_20260622_211224.json`
- `E014`：`checkpoints/20260617_bs32_8192/voc/15-5/sequential/step1/test_results_20260617_142131.json`
- `E015`：`checkpoints/20260617_bs32_8196/voc/15-5/sequential/step1/test_results_20260617_121741.json`
- `E016`：`checkpoints/20260625_baseline_bs64_32_8200/voc/15-5/sequential/step1/test_results_20260626_163608.json`
- `E017`：`checkpoints/20260621_baseline_bs64_32/voc/15-5/sequential/step1/test_results_20260622_105925.json`
- `E018`：`checkpoints/20260622_buffer8216_step1_32_run2/voc/15-5/sequential/step1/test_results_20260623_231319.json`
- `E019`：`checkpoints/20260617_bs48_step1_8196/voc/15-5/sequential/step1/test_results_20260617_093250.json`
- `E020`：`checkpoints/20260617_bs48_step1/voc/15-5/sequential/step1/test_results_20260617_080533.json`
- `E021`：`checkpoints/20260610_rhl_l2sqrt_g0p1_bs64/voc/15-5/sequential/step1/test_results_20260610_204628.json`
- `E022`：`checkpoints/20260610_rhl_l2sqrt_g1_bs64/voc/15-5/sequential/step1/test_results_20260610_221231.json`
- `E023`：`checkpoints/20260610_rhl_l2sqrt_g10_bs64/voc/15-5/sequential/step1/test_results_20260610_233856.json`
- `E024`：`checkpoints/20260609_rhl_l2sqrt_g1_retry/voc/15-5/sequential/step1/test_results_20260609_192910.json`
- `E025`：`checkpoints/20260610_rhl_none_g1/voc/15-5/sequential/step1/test_results_20260610_115359.json`
- `E026`：`checkpoints/20260610_rhl_ln_g1_bs64/voc/15-5/sequential/step1/test_results_20260610_174503.json`
- `E027`：`checkpoints/20260610_rhl_l2_g1/voc/15-5/sequential/step1/test_results_20260610_125752.json`
- `E028`：`checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/test_results_20260616_120655.json`
- `E029`：`checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/test_results_20260616_125047.json`
- `E030`：`checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/test_results_20260616_133547.json`
- `E031`：`logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_logit.json`
- `E032`：`logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3.json`
- `E033`：`logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_cpu.json`
- `E034`：`logs/rhl_ensemble/20260616_rhl_se_k3_logit_w020_040_040.json`
- `E035`：`logs/rhl_ensemble/20260616_rhl_se_k3_w010_045_045.json`
- `E036`：`logs/rhl_ensemble/20260616_rhl_se_k3_w020_040_040.json`
- `E037`：`logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal.json`
- `E038`：`logs/rhl_ensemble/20260616_rhl_se_k3.json`
- `E039`：`logs/rhl_ensemble/20260616_rhl_se_oldnew_prob_o70_n14545.json`
- `E040`：`logs/rhl_ensemble/20260616_rhl_se_classwise_prob_v1.json`
- `E041`：`logs/rhl_ensemble/20260616_rhl_se_gate_prob_seed1_w244_m010.json`
- `E042`：`logs/rhl_se_val_driven/20260617_p0_rhl_se_val_driven/test_results.json`
- `E043`：`logs/rhl_se_val_driven/20260618_p0_rhl_se_val_driven/test_results.json`
- `E044`：`logs/rhl_se_val_driven/20260618_p0_rhl_bs16_se_val_driven/test_results.json`
- `E045`：`logs/rhl_se_val_driven/20260618_p0_rhl_bs32_se_val_driven/test_results.json`
- `E046`：`logs/rhl_ensemble/20260616_rhl_se_val_classwise_test.json`
- `E047`：`checkpoints/20260617_p1_boa_rhl_BOA-0_gaussian_legacy_seed1_bs32/voc/15-5/sequential/step1/test_results_20260617_223901.json`
- `E048`：`checkpoints/20260617_p1_boa_rhl_BOA-1_orthogonal_legacy_seed1_bs32/voc/15-5/sequential/step1/test_results_20260617_233020.json`
- `E049`：`checkpoints/20260617_p1_boa_rhl_BOA-2_orthogonal_antithetic_legacy_seed1_bs32/voc/15-5/sequential/step1/test_results_20260618_001844.json`
- `E050`：`checkpoints/20260617_p1_boa_rhl_BOA-3_orthogonal_antithetic_kaiming_seed1_bs32/voc/15-5/sequential/step1/test_results_20260618_010704.json`
- `E051`：`../SegACIL_deeplabv3plus/checkpoints/20260622_v3plus_air_aspp_up/voc/15-5/sequential/step1/test_results_20260622_204635.json`
- `E052`：`../SegACIL_deeplabv3plus/checkpoints/20260622_v3plus_air_aspp/voc/15-5/sequential/step1/test_results_20260622_195132.json`
- `E053`：`../SegACIL_deeplabv3plus/checkpoints/20260623_v3plus_voc15-5_seq_bs32-16_8200/voc/15-5/sequential/step1/test_results_20260623_235108.json`
- `E054`：`../SegACIL_deeplabv3plus/checkpoints/20260623_v3plus_voc15-5_seq_bs32-16_8216/voc/15-5/sequential/step1/test_results_20260623_113300.json`
- `E055`：`../SegACIL_deeplabv3plus/checkpoints/20260622_v3plus_air_decoder/voc/15-5/sequential/step1/test_results_20260622_190708.json`
- `E056`：`../SegACIL_deeplabv3plus/checkpoints/20260622_v3plus_air_decoder_stride8/voc/15-5/sequential/step1/test_results_20260622_192921.json`
- `E057`：`../SegACIL_deeplabv3plus/checkpoints/20260623_v3plus_voc15-5_seq_bs32-32/voc/15-5/sequential/step1/test_results_20260623_100801.json`
- `E058`：`../SegACIL_deeplabv3plus/checkpoints/20260623_v3plus_air_aspp_up_cap4096/voc/15-5/sequential/step1/test_results_20260623_072849.json`
- `E059`：`../SegACIL_deeplabv3plus/checkpoints/20260623_v3plus_air_aspp_up_cap8192/voc/15-5/sequential/step1/test_results_20260623_074851.json`
- `E060`：`../SegACIL_deeplabv3plus/checkpoints/20260624_v3plus_integration_golden_replay_14fc116/voc/15-5/sequential/step1/test_results_20260624_153910.json`
- `E061`：`../SegACIL_deeplabv3plus/checkpoints/20260614_v3plus_voc15-5_seq_bs32-16/voc/15-5/sequential/step1/test_results_20260615_153928.json`
- `E062`：`checkpoints/20260707_pseudo_15-1_overlap_off_seed1_bs32_phaseD/voc/15-1/overlap/step5/test_results_20260708_235809.json`
- `E063`：`checkpoints/20260707_pseudo_15-1_overlap_fixed0p6_seed1_bs32_phaseD_reuse_offstep0/voc/15-1/overlap/step5/test_results_20260709_004125.json`
- `E064`：`checkpoints/20260707_pseudo_15-1_overlap_fixed0p7_seed1_bs32_phaseD_reuse_offstep0/voc/15-1/overlap/step5/test_results_20260709_012457.json`
- `E065`：`checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p029296875_seed1_bs32_recovery1_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260719_045822.json`
- `E066`：`checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed2_bs32_recovery1_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260719_052100.json`
- `E067`：`checkpoints/20260719_pseudo_15-5_disjoint_globalfixed0p048828125_seed3_bs32_recovery1_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260719_054339.json`
- `E068`：`checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p447265625_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260718_155137.json`
- `E069`：`checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p419921875_seed2_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260718_161623.json`
- `E070`：`checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p443359375_seed3_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260718_164108.json`
- `E071`：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p00_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_181314.json`
- `E072`：`checkpoints/20260707_pseudo_15-5_disjoint_artifact_q0p00_seed2_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260707_084421.json`
- `E073`：`checkpoints/20260707_pseudo_15-5_disjoint_artifact_q0p00_seed3_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260707_095029.json`
- `E074`：`checkpoints/20260705_pseudo_15-5_disjoint_artifact_q0p005_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_104008.json`
- `E075`：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p005_minconf0p6_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_231315.json`
- `E076`：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p01_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_184312.json`
- `E077`：`checkpoints/20260706_pseudo_15-5_disjoint_artifact_q0p015_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_224327.json`
- `E078`：`checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step1/test_results_20260706_095428.json`
- `E079`：`checkpoints/20260705_pseudo_15-5_disjoint_fixed0p6_seed1_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260706_101656.json`
- `E080`：`checkpoints/20260707_pseudo_15-5_disjoint_fixed0p6_seed2_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260707_082040.json`
- `E081`：`checkpoints/20260707_pseudo_15-5_disjoint_fixed0p6_seed3_bs32_reuse20260705disjointstep0/voc/15-5/disjoint/step1/test_results_20260707_091720.json`
- `E082`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p00_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_123227.json`
- `E083`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p005_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_130848.json`
- `E084`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p01_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_073518.json`
- `E085`：`checkpoints/20260707_pseudo_15-5_overlap_artifact_q0p01_seed2_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260707_065850.json`
- `E086`：`checkpoints/20260707_pseudo_15-5_overlap_artifact_q0p01_seed3_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260707_074834.json`
- `E087`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p01_minconf0p6_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_140856.json`
- `E088`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p015_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_134417.json`
- `E089`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p03_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_081139.json`
- `E090`：`checkpoints/20260701_pseudo_15-5_overlap_artifact_q0p05_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_043705.json`
- `E091`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p05_minconf0p6_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_092401.json`
- `E092`：`checkpoints/20260705_pseudo_15-5_overlap_artifact_q0p07_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_084743.json`
- `E093`：`checkpoints/20260701_pseudo_15-5_overlap_artifact_q0p10_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_051335.json`
- `E094`：`checkpoints/20260701_pseudo_15-5_overlap_artifact_q0p20_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260705_055006.json`
- `E095`：`checkpoints/20260630_pseudo_15-5_overlap_batchclass_q0p1_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_164038.json`
- `E096`：`checkpoints/20260630_pseudo_15-5_overlap_batchclass_q0p3_minconf0p6_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_175447.json`
- `E097`：`checkpoints/20260630_pseudo_15-5_overlap_batchclass_q0p3_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_170522.json`
- `E098`：`checkpoints/20260630_pseudo_15-5_overlap_batchclass_q0p5_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_173005.json`
- `E099`：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step1/test_results_20260628_071622.json`
- `E100`：`checkpoints/20260630_pseudo_15-5_overlap_batchglobal_q0p1_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_152641.json`
- `E101`：`checkpoints/20260630_pseudo_15-5_overlap_batchglobal_q0p3_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_155118.json`
- `E102`：`checkpoints/20260630_pseudo_15-5_overlap_batchglobal_q0p5_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_161555.json`
- `E103`：`checkpoints/20260630_pseudo_15-5_overlap_batchglobal_q0p7_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_090055.json`
- `E104`：`checkpoints/20260630_pseudo_15-5_overlap_off_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_065032.json`
- `E105`：`checkpoints/20260630_pseudo_15-5_overlap_fixed0p6_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_121818.json`
- `E106`：`checkpoints/20260707_pseudo_15-5_overlap_fixed0p6_seed2_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260707_045107.json`
- `E107`：`checkpoints/20260707_pseudo_15-5_overlap_fixed0p6_seed3_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260707_072343.json`
- `E108`：`checkpoints/20260630_pseudo_15-5_overlap_fixed0p7_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_073707.json`
- `E109`：`checkpoints/20260630_pseudo_15-5_overlap_fixed0p8_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_124305.json`
- `E110`：`checkpoints/20260630_pseudo_15-5_overlap_fixed0p9_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_130756.json`

## 12. 主要资料

- Li, J. 等，*CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds*，ACM Multimedia，DOI: `10.1145/3746027.3755023`。本地文件：[CFSSeg 原论文](<../../All_papers/庄-核心-CFSSeg_Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds.pdf>)。
- [CFSSeg 论文精读与思考分析报告](../论文精读/CFSSeg论文精读与思考分析报告.md)
- [五方法完整工作流](../../Codex_Plans/5方法原理动机与基于优先级排序的完整工作流行动路线.md)
- [RHL 归一化新方案](<../idea构思与实验设计/RHL归一化/6-20 RHL归一化新方案.md>)
- [RHL-SE 2.0 新方案](../idea构思与实验设计/RHL新方案/6-20_RHL-SE_2.0新方案.md)
- [P1–P3 具体方案与实验计划](../idea构思与实验设计/RHL新方案/6-20_P1-P3具体方案与实验计划.md)
- [DeepLabV3+ 收尾评审与集成决策](../idea验证与结论/6-23_DeepLabV3Plus大型任务收尾评审与主线集成决策.md)
- [自适应伪标签路线再审与机制判别方案](../idea构思与实验设计/自适应伪标签阈值/7月新篇/7-18_自适应伪标签阈值路线再审与机制判别实验方案.md)
