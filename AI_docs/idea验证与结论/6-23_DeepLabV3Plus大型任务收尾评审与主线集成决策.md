# DeepLabV3+ 大型任务收尾评审、实验结论与主线集成决策

> 日期：2026-06-23
> 审查目录：`/root/2TStorage/lyc/SegACIL_deeplabv3plus`
> feature 分支：`feature/deeplabv3plus-control`
> main 工作区：`/root/2TStorage/lyc/SegACIL`
> 结论性质：代码审查 + 实验结果审查 + 非正式 ARA 六维语义严谨性审查
> 注意：当前资料不是通过 ARA Level 1 的标准 ARA 目录，因此本文不声称获得正式 ARA Seal，也不生成 `level2_report.json`。

## 0. 最终结论

本次大型任务已经完成其原始目标：

1. DeepLabV3+ step1 新类掉点的主要原因已经定位并通过单因素实验修复。
2. 最终有效配置是 `DeepLabV3+ + aspp_up + pixel_balance=none + gamma=1 + buffer=8196`。
3. VOC 15-5 sequential step1 的最终结果为：

| 配置 | 0-15 mIoU | 16-20 mIoU | All mIoU |
|---|---:|---:|---:|
| DeepLabV3 20260606 | 0.7801 | 0.4211 | 0.6946 |
| DeepLabV3 20260607 | 0.7779 | 0.4321 | 0.6956 |
| V3+ `decoder` | 0.7815 | 0.3959 | 0.6897 |
| V3+ `aspp` | 0.7771 | 0.4510 | 0.6995 |
| V3+ `aspp_up` | **0.7793** | **0.4613** | **0.7036** |

`aspp_up` 相比当前最好的 DeepLabV3 结果：

```text
old: +0.0014
new: +0.0292
all: +0.0080
```

因此，V3+ 已经证明自己是一个有价值的 stronger architecture control，也具备成为可选实验 base 的条件。

但本文不建议立即把 V3+ 设为唯一默认 base，也不建议直接把当前 feature 分支整体 merge 到 main。正确动作是：

```text
保留 DeepLabV3 为 canonical baseline
        +
把 V3+ 作为一等可选模型集成进最新 main
        +
在 integration 分支完成冲突整合、paired multi-seed 和方法迁移小实验
        +
验证后再合并 main
```

## 1. 与原工作流的逐项对照

原工作流见：

```text
AI_docs/idea构思与实验设计/v3plus设计与验证（未完善收尾）/
6-15_DeepLabV3Plus特征不匹配原因细化与验证方案.md
```

### 1.1 总体完成表

| 原阶段 | 原目标 | 完成情况 | 证据 | 最终判断 |
|---|---|---|---|---|
| 前置审查 | 重新审查三个原因及“最小改动”提示词 | 已完成 | `e020447` | 原 Identity 模式判断属于防御性补丁，改成显式 AIR feature API |
| Stage 0 | 建立显式 feature source，保持默认等价 | 已完成 | `5c45590` + 单测 + 真实 checkpoint smoke | `decoder` 逐元素/指标复现旧路径 |
| Stage 1 | 对比 decoder、decoder_stride8、aspp、aspp_up | 已完成 | 四组完整 step1 JSON | `aspp_up` 最佳，主因得到强证据支持 |
| Stage 2 | 在最佳 source 上验证 class-cap | 已完成 | cap4096、cap8192 完整实验 | 降显存、涨稀有类召回，但严重伤旧类，不进入默认 |
| Stage 3 | 有证据时再做 gamma/norm | 有依据地跳过 | source 修复已超过目标；既有 gamma 证据弱 | 不继续无证据扫参 |
| Stage 4 | 组合有效项并收敛默认 | 已完成 | `23a97a9` | `aspp_up + none + gamma1 + buffer8196` |
| 最终验收 | 测试、真实 forward、审查、文档 | 已完成 | `675cc2a`、`a475e8d` | 本次异常修复任务完成 |

### 1.2 原因一：feature 语义不匹配

原假设：DeepLabV3+ 的 low-level decoder feature 更偏纹理、边界和局部细节，不一定适合冻结特征后的 RandomBuffer + RecursiveLinear。

结果：

```text
decoder new mIoU = 0.3959
aspp new mIoU    = 0.4510
aspp_up new mIoU = 0.4613
```

在同一个 step0 checkpoint、同一个 seed、同一个 buffer/gamma 下，仅切换 AIR feature source，新类提高 6.54 个点，旧类基本保持。

结论：在当前 VOC 15-5 sequential、当前 checkpoint 和 AIR 实现范围内，原因一得到强证据支持。

更严谨的表述应是：

> 当前 DeepLabV3+ 的 low-level fused decoder feature 不适合本项目的冻结特征解析学习路径；高层 ASPP feature 更适合作为 AIR 接口。

不应扩大成：

> 所有 DeepLabV3+ decoder feature 都不适合持续学习。

后者超出了现有数据集、协议和实现证据范围。

### 1.3 原因二：高分辨率和像素密度

关键对照：

| source | 空间尺度 | new mIoU |
|---|---|---:|
| decoder | stride-4 | 0.3959 |
| decoder_stride8 | stride-8 | 0.4011 |
| aspp | stride-8 | 0.4510 |
| aspp_up | stride-4 | 0.4613 |

`decoder_stride8` 只提高 0.52 个点，而保持 stride-4 的 `aspp_up` 反而最好。

因此原始“高分辨率是主要原因”被证伪并修正：

```text
主要矛盾是 feature 语义；
分辨率影响计算成本和标签对齐，但不是 decoder 掉点的根因。
```

### 1.4 原因三：gamma、尺度和随机映射统计

当前没有必要继续做 gamma/norm 搜索，理由是：

1. feature source 单因素已经把新类提高 6.54 个点。
2. 当前没有 NaN、Inf、求逆失败或明显数值不稳定证据。
3. 既有 RHL 实验中 `gamma=0.1/1/10` 变化很小。
4. 继续扫描会把已清楚的 feature-source 结论重新混入参数搜索。

跳过 Stage 3 不是“任务没做完”，而是遵守预先设定的实验门禁。

### 1.5 class-cap 负结果是否有价值

有价值，而且应保留为明确的失败分支：

| 配置 | 0-15 | 16-20 | All | 观测显存 |
|---|---:|---:|---:|---:|
| `aspp_up + none` | 0.7793 | 0.4613 | 0.7036 | 先前约 56 GiB |
| `cap4096` | 0.6946 | 0.4552 | 0.6376 | 约 14.1 GiB |
| `cap8192` | 0.7109 | 0.4707 | 0.6537 | 约 33.6 GiB |

它证明：

1. 低维采样确实能明显降低 RandomBuffer 前后的 dense pixel 显存。
2. 强行均衡类别会提高 pottedplant、tvmonitor 等稀有类召回。
3. 但统一硬截断改变了 base replay 的 `X^T X` / `X^T Y`，导致大量旧类假阳性和总体退化。

因此 class-cap 是有效的显存/召回实验手段，不是最终精度方案。

## 2. 代码实现审查总结

### 2.1 审查范围

本轮重新审查：

```text
merge-base: 68ef667
feature HEAD: a475e8d
```

覆盖 DeepLabV3+ 模型工厂和 head、显式 AIR feature API、Trainer/AIR、parser、runner、sweep、summarizer、resume、测试与文档。

CodeRabbit CLI 为 `0.6.1`，认证正常。完整范围审查返回 6 条建议；其中 1 条有效，其余需要结合项目协议校准。

### 2.2 已确认正确的核心实现

#### A. 显式 feature API 边界正确

当前结构：

```text
DeepLabHead(V3/V3+).extract_features()
DeepLabHead(V3/V3+).select_air_feature()
_SimpleSegmentationModel.forward_air_features()
AIR._extract_feature_map()
```

它将普通 segmentation forward 与 AIR feature extraction 分离，不再依赖 `classifier.head = Identity` 作为隐式模式开关，非法 source 也会显式报错。

#### B. `decoder_stride8` 实现正确

使用 `adaptive_avg_pool2d(..., aspp.shape[-2:])`，避免 129 经固定 pool 得到 64、无法与实际 65 对齐。

#### C. class-cap 采样位置正确

采样发生在 256 维 feature 进入 8196 维 RandomBuffer 之前，真正降低峰值显存；没有把分割采样策略塞进通用 `RecursiveLinear`。

#### D. 自动测试覆盖关键合同

测试覆盖普通 logits、四种 feature shape、decoder 默认等价、非法 source、class-cap、ignore、旧 fit 等价和结果汇总器。

最终 12/12 单元测试通过，真实 513x513 checkpoint forward 也通过。

### 2.3 需要在主线集成前处理的问题

#### Major 1：不能直接 merge 当前 feature 到 main

当前拓扑：

```text
main 相对共同祖先：11 个独立提交
feature 相对共同祖先：13 个独立提交
feature 相对 origin/feature：领先 9 个提交
```

`git merge-tree` 已确认至少四个实际冲突：

```text
6-15_DeepLabV3Plus特征不匹配原因细化与验证方案.md
run.sh
trainer/trainer.py
utils/parser.py
```

最关键的代码冲突是：

```text
main:    RHL norm / RHL seed / RHL stats / 五方法基础设施
feature: AIR feature source / pixel balance / V3+ 显式接口
```

任何“直接接受 ours/theirs”都会丢功能。必须在最新 main 上做 integration 分支，手工组合两侧参数和训练路径。

#### Major 2：当前 `run_v3plus.sh` 不是最终有效 runner

当前工作区中的 `run_v3plus.sh`：

```text
没有传 --air_feature_source aspp_up
未提交改动中 BUFFER=8216
```

如果直接运行，它会使用 parser 默认 `decoder`，回到已经验证较差的 `new=0.3959` 路径；`BUFFER=8216` 也不属于任何已验证协议。

当前可复现的有效入口是：

```text
run_v3plus_air.sh
AIR_FEATURE_SOURCE=aspp_up
BUFFER=8196
AIR_PIXEL_BALANCE=none
```

因此 `run_v3plus.sh` 在决定合并前必须由用户确认、修正或丢弃，不能带着 dirty 状态进入 merge。

#### Major 3：未来 step1 伪标签仍被共享 `curr_step` 状态阻断

feature 和 main 的 step1 都存在：

```python
self.opts.curr_step = 0
```

该操作用于构建 step0 replay loader，但没有恢复真实 step。随后伪标签判断又读取 `self.opts.curr_step > 1`。

结果是：

1. VOC 15-5 step1 原本就不会进入旧伪标签路径。
2. 即使简单改成 `>=1`，DeepLab teacher 返回 tuple/NCHW，当前函数仍不兼容。
3. run config/checkpoint 记录也可能写入错误 step 语义。

这不会影响已经完成的 sequential V3+ 实验，但在实现自适应伪标签前必须使用局部 `step0_opts` 副本，并增加统一 teacher-output adapter。

### 2.4 Warning 与维护项

| 问题 | 影响 | 建议 |
|---|---|---|
| feature `run.sh` 硬编码 `CUDA_VISIBLE_DEVICES=2` | runner 不可移植 | 改为环境变量可覆盖 |
| main `run.sh` 使用 Buffer8192，计划和 V3+ 使用8196 | paired 对照协议不一致 | Phase 0 明确唯一基线值并写入 manifest |
| parser 默认 decoder，V3+ 最佳需要 aspp_up | 只改 MODEL 会静默跑错 | 增加 model-aware `auto` source 或显式校验 |
| step1 没有 checkpoint 模型拓扑校验 | V3+ 可能误读 V3 step0 路径 | 校验 model/config/hash 后再加载 |
| `run_v3plus.sh` 与 `run_v3plus_air.sh` 重复 | 默认值漂移 | 通用能力进 `run.sh`，专用脚本只保留复现用途 |
| class-cap 已证伪为默认方案 | 增加主线配置面 | 可留在实验分支；主线集成时不必默认吸收 |

### 2.5 对 CodeRabbit 六条 finding 的裁决

| finding | 裁决 | 理由 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=2` 硬编码 | 有效 | 应允许环境变量覆盖 |
| 8196 不是 2 的幂，应改 8192（两条） | 驳回 | 8196 是项目历史实验协议；问题是协议不一致，不是必须为 2 的幂 |
| resume optimizer state 仍在 CPU | 驳回 | 模型参数在 optimizer 创建前已到目标设备；当前 PyTorch 会按参数设备 cast optimizer state |
| result 文件字典序会把 10 排在 2 前 | 当前生产协议不成立 | 生产文件名是固定宽度 `YYYYMMDD_HHMMSS`；可改用 mtime 增强鲁棒性 |
| `Class IoU` 应使用 `.get()` | 驳回 | 缺失核心指标应 fail fast，静默空值会掩盖损坏结果 |

### 2.6 Code review 总评

```text
核心 V3+ feature-source 修复：通过
已完成实验的可复现性：通过
直接合并当前分支：不通过
作为未来 pseudo-label/五方法统一底座：需先做 integration hardening
```

没有发现会推翻已经产出的 V3+ source-sweep 结果的 Critical bug。

## 3. 实验结果分析与讨论

### 3.1 step0 结果

| 模型 | step0 0-15 mIoU |
|---|---:|
| V3 20260606 | 0.7481 |
| V3 20260607 | 0.7472 |
| V3+ 20260614 | **0.7511** |

V3+ step0 比 V3 高约 0.3-0.4 个点，方向符合“低层细节融合有利于监督分割”的预期，但差距不大。

### 3.2 初始 step1 异常为什么不合理

V3+ decoder 配置为 old=0.7815、new=0.3959。旧类略高而新类明显低，说明不是整个 V3+ 模型训练失败，而是 step1 冻结特征后，新类线性可分性或标签对齐出了问题。

### 3.3 source sweep 为什么能定位原因

四个 source 分别隔离“是否包含 low-level fusion”和“是否保持 stride-4”：

- `decoder -> decoder_stride8` 几乎无改善：仅降分辨率不能修复。
- `decoder_stride8 -> aspp` 大幅改善：高层语义是关键。
- `aspp -> aspp_up` 继续改善：无参数上采样提高了标签空间对齐。

这组实验比直接扫 gamma 更有解释力，因为它逐步分离了语义和分辨率。

### 3.4 结果是否符合预期

符合，而且略好于最低预期：

1. 原目标是把 V3+ 新类从 0.3959 恢复到 V3 的 0.421-0.432。
2. `aspp=0.4510` 已超过目标。
3. `aspp_up=0.4613` 进一步提高。
4. All=0.7036 超过当前 V3 最好结果 0.6956。

因此“异常修复”目标已经完成，不需要继续围绕 V3+ 盲目调参。

### 3.5 当前证据还不能证明什么

当前不能直接证明：

1. V3+ 在所有 seed 下都稳定优于 V3。
2. V3+ 在 overlap/disjoint 或 15-1 多 step 中仍优于 V3。
3. 五方法在 V3 上的收益会无条件迁移到 V3+。
4. `aspp_up=0.7036` 是“纯模型架构替换”的全部收益。

原因是 `aspp_up` 除了更换 V3+，还选择了不同的 AIR feature tap 和空间对齐方式。这个改动合理、无可训练参数、经过消融验证，但论文中应称为：

```text
CFSSeg-compatible DeepLabV3+ feature interface
```

而不是把 0.7036 全部描述为“只把 DeepLabV3 换成 DeepLabV3+”。

严格 architecture control 应同时报告 V3 baseline、V3+ aspp 和 V3+ aspp_up。

## 4. 问题一：是否还需要实验、模型在 step1 的角色与公平性

### 4.1 当前是否还有必须做的实验

分两种目标回答。

#### 目标 A：完成本次异常定位和修复

不需要继续实验。本任务已经完成：

```text
异常复现 -> 原因消融 -> 修复 -> 负结果验证 -> 最终配置
```

#### 目标 B：把 V3+ 升格为统一默认 base

还需要三个验收实验，控制在以下范围：

1. **paired multi-seed**：在同一个 integration commit、同一个 Buffer/gamma/batch 下，对 V3 与 V3+ aspp/aspp_up 跑 RandomBuffer seed 1/2/3，报告 mean/std。
2. **non-sequential compatibility**：至少跑 15-5 overlap 的 pseudo off baseline，并完成 DeepLab tuple/NCHW teacher adapter smoke；阈值 artifact 按 V3/V3+ teacher 分别生成。
3. **方法迁移 2x2 pilot**：选择 BOA-RHL 或 PGH-RHL，做 `backbone(V3/V3+) x method(off/on)`，确认方法收益没有因 feature distribution 改变而反转。

bs8 的显存实测可并入第 1 项，不单独扩成新的精度搜索。

### 4.2 DeepLabV3/V3+ 是否只参与 step0

不是。

准确流程：

```text
step0:
  ResNet101 + ASPP + V3/V3+ segmentation head
  全部参与监督训练

step1:
  加载完整 step0 模型
  冻结后作为 feature extractor
  V3: 取 decoder/head_pre feature
  V3+: 最终取 aspp_up feature
  -> RandomBuffer
  -> RecursiveLinear
  -> 最终 step1 logits
```

因此：

- V3/V3+ 的原分类器不再直接决定 step1 最终类别预测；
- 但 backbone、ASPP 和所选 feature path 仍直接决定 AIR 输入；
- V3+ 最佳方案中 low-level project/decoder 不进入 AIR，但它们参与过 step0 监督训练并影响 ASPP 表示。

### 4.3 除了升级模型还改了什么

有三类改动。

#### 必需的兼容性改动

- V3+ head 改为 CFSSeg 可用的共享 256 维 decoder + per-pixel Linear head；
- 普通 forward 返回 `(logits, feature_dict)`；
- 增加显式 `forward_air_features()`。

这些是让 V3+ 满足现有 CFSSeg/AIR contract 的必要工程改动。

#### 最终生效的方法改动

- AIR feature source 从 V3+ `decoder` 改为 `aspp_up`。

它没有新增可训练参数，但改变了 step1 解析分类器看到的 feature 语义和空间密度。

#### 只用于诊断、未进入最终默认的改动

- `decoder_stride8/aspp` source；
- `class_cap=4096/8192`；
- sweep 和 summarizer。

最终配置中 `pixel_balance=none`，所以 class-cap 不污染最终结果。

### 4.4 是否公平

在以下表述下是公平的：

> 比较同一 CFSSeg 解析学习框架下，DeepLabV3 与经过显式 feature-interface 适配的 DeepLabV3+ control。

依据：

- 数据、task、setting、step0 checkpoint、gamma、buffer、seed 固定；
- source sweep 只改变一个 feature source；
- aspp_up 是无参数上采样，不引入额外监督或训练数据；
- decoder 复现旧结果，证明接口重构没有偷改默认数学路径。

但如果论文声称“唯一变量只是模型名”，则不够严格。因为最终同时改变了 feature tap。应把 `aspp` 和 `aspp_up` 消融一起报告，并明确 interface adaptation。

## 5. 问题二：能否替代原 base、后续方法是否使用 V3+、是否合并

### 5.1 V3+ 能否替代 DeepLabV3

当前结论：

```text
作为可选/推荐工程 base：可以
作为唯一论文 canonical baseline：暂时不建议
```

理由：

1. V3+ 最佳结果已经超过 V3，具备使用价值。
2. 但领先幅度 All 只有约 0.8 个点，尚无 multi-seed 统计。
3. 原论文和现有五方法 checkpoint/结果以 V3 为基准。
4. 直接替换会让已经完成的 V3 方法实验失去 paired baseline。

最稳妥的论文结构：

```text
主表/canonical baseline：DeepLabV3-ResNet101
architecture robustness / stronger base：DeepLabV3+-ResNet101 + aspp_up
```

三 seed 和方法迁移 pilot 通过后，可把 V3+ 提升为后续大规模工程实验的 preferred base；仍保留 V3 结果用于论文公平对比。

### 5.2 五方法行动路线是否可以改用 V3+

可以使用，但不能把文档中的 V3 主基线直接静默替换。

推荐方式：

1. Phase 0 把 `MODEL`、`AIR_FEATURE_SOURCE`、step0 checkpoint hash 写入 manifest。
2. 五方法首先在既定 V3 baseline 上完成 standalone 归因。
3. 每个有稳定正向证据的方法，再在 V3+ 上做 paired replication。
4. 至少对优先级最高的方法做 `V3/V3+ x method off/on` 2x2。
5. 不把“更强 base 的主效应”误写成“新方法收益”。

V3+ 不会破坏 BOA/PGH/PowerNorm/CA-C-RLS 的理论接口，因为这些方法都位于 feature 之后；但它会改变 feature 分布、尺度、相关性和类别可分性，所以不能假设超参数和增益自动迁移。

### 5.3 自适应伪标签是否可以使用 V3+

可以，而且现有 6-21 实现计划已经按模型无关方式设计：

```text
DeepLab teacher -> tuple/NCHW adapter
AIR teacher     -> tensor/NHWC adapter
```

需要注意：

1. 自适应伪标签只应在 overlap/disjoint 验证，不进入当前 15-5 sequential 主表。
2. V3+ teacher 的概率校准可能与 V3 不同，不能复用同一个 threshold artifact。
3. step1 teacher 应是完整 step0 V3+ segmentation model；student AIR 使用 aspp_up feature。
4. 先修复 `self.opts.curr_step=0` 的共享状态问题。
5. pseudo off 必须逐元素复现 V3+ non-sequential baseline。

推荐先在 V3 上开发和验证伪标签 helper/adapter，减少同时变化因素；然后在 V3+ 上做独立 replication。若更关注工程最高结果，也可以以 V3+ 为第二条正式实验线，但必须单独生成 teacher artifact 和 baseline。

### 5.4 当前 feature 是否应该直接 merge main

不应该直接 merge。

不是因为 V3+ 不值得合并，而是当前状态不满足安全集成条件：

- feature 尚有 9 个本地提交未 push；
- feature 有用户 dirty `run_v3plus.sh`；
- feature 有未跟踪策略文档；
- main 和 feature 已显著分叉；
- `trainer.py/parser.py/run.sh` 存在真实冲突；
- main 的 RHL 功能不能被 feature 覆盖；
- class-cap 负结果代码不一定需要进入主线。

最终推荐是：

> 在最新 main 上创建 integration 分支，选择性移植稳定的 V3+ 通用能力，完成回归后再合并 main。

### 5.5 是否应把 V3+ 做成 run.sh 的 MODEL 可选项

应该，而且是正式代码实现项。

main 的 `network/modeling.py` 已有 V3+ model name，但 main 的 V3+ head 和 step1 AIR contract 仍不完整；main 的 `run.sh` 也把 MODEL 硬编码为 V3。因此不是简单改一行变量。

完整实现应包含：

```text
MODEL="${MODEL:-deeplabv3_resnet101}"
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-auto}"
```

`auto` 规则：

```text
deeplabv3_*     -> decoder
deeplabv3plus_* -> aspp_up
```

并增加：

1. run id/manifest 记录 model 和 source；
2. step1 checkpoint 模型类型与 source 校验；
3. V3+ step1 禁止静默使用 decoder，除非用户显式做消融；
4. default 仍是 `deeplabv3_resnet101`；
5. 每个 model 使用自己的 `BASE_SUBPATH`；
6. 统一 Buffer 8192/8196 协议后再跑 paired 对照。

长期结构：

```text
run.sh                    通用入口，模型和方法均可配置
run_v3plus_air.sh         保留为本次实验复现脚本
run_v3plus.sh             修正后保留或删除，避免两个 V3+ 默认漂移
```

## 6. 推荐主线集成工作流

### Step 0：先处理当前 dirty 状态

1. 审查 `run_v3plus.sh` 的用户改动，确认 `BUFFER=8216` 是笔误还是新实验意图。
2. 决定未跟踪策略文档是否需要提交。
3. 不覆盖用户文件。
4. 把当前 9 个已验证提交 push 到远端 feature 分支。

### Step 1：从最新 main 创建 integration worktree

```bash
cd /root/2TStorage/lyc/SegACIL
git switch main
git pull --ff-only origin main
git worktree add -b feature/integrate-deeplabv3plus \
  ../SegACIL_integrate_v3plus main
```

不要在当前两个 worktree 中直接解决所有冲突。

### Step 2：选择性集成

移植：

1. V3+ shared decoder + Linear head contract；
2. `extract_features/select_air_feature/forward_air_features`；
3. AIR `feature_source`，同时保留 main 的 RHL norm/seed/stats；
4. parser 中 model/source 配置；
5. feature-source 单测；
6. generic `run.sh` model-aware source；
7. summarizer 和最终文档。

默认不优先移植：

- class-cap 代码及 sweep，除非明确保留为显存研究工具；
- 当前 dirty `run_v3plus.sh`；
- 与 main 已有文档冲突的历史重复内容。

### Step 3：同步修复共享基础设施

1. 使用 `step0_opts = copy.deepcopy(self.opts)`，不再修改真实 `self.opts.curr_step`。
2. checkpoint/config/manifest 记录 `model`、`air_feature_source`、RHL 配置和 base checkpoint hash。

teacher-output adapter 属于 pseudo-label 独立提交，不应与 V3+ 主集成提交混在一起。

### Step 4：代码验收

必须执行：

```text
Python py_compile
bash -n
V3 普通 forward + AIR feature smoke
V3+ 普通 forward + 四 source smoke
V3 默认路径回归
V3+ aspp_up step1 one-batch smoke
错误 model/source/checkpoint 组合显式失败
```

### Step 5：实验验收

只跑第 4.1 节的三项实验，不继续扩张 V3+ 调参空间。

### Step 6：合并决策

满足以下条件后合并 main：

```text
V3 baseline 无回归
V3+ aspp_up 结果可复现
三 seed 方向稳定
run.sh 可安全切换 MODEL
main RHL 功能未丢失
pseudo off/non-sequential smoke 通过
工作区 clean
```

## 7. 非正式 ARA 六维严谨性审查

### 7.1 适用性说明

正式 `ara-rigor-reviewer` 要求 `PAPER.md`、`logic/`、`trace/exploration_tree.yaml`、`evidence/` 和 Level 1 passed。当前项目没有这套结构，所以以下评分是适配审查，不是正式 ARA Seal。

### 7.2 六维评分

| 维度 | 分数 | 评价 |
|---|---:|---|
| D1 Evidence Relevance | 4/5 | source 消融直接对应 feature 不匹配主张；class-cap 也能解释均衡副作用 |
| D2 Falsifiability | 4/5 | 预先设置等价性、阈值和阶段门禁；部分判断尚未写成正式 falsification 条目 |
| D3 Scope Calibration | 4/5 | 主要结论限定在当前 VOC 15-5；“主因确认”仍应保留实现/协议限定词 |
| D4 Argument Coherence | 5/5 | 异常 -> 三原因 -> source 消融 -> 负结果 -> 收敛配置，逻辑链完整 |
| D5 Exploration Integrity | 5/5 | 真实记录 decoder_stride8 弱结果和 class-cap 失败，没有只保留最好点 |
| D6 Methodological Rigor | 3/5 | 单 step0、单 RandomBuffer seed，V3 对照来自历史 run；缺同 commit paired multi-seed |

平均分：

```text
(4 + 4 + 4 + 5 + 5 + 3) / 6 = 4.17
```

按 ARA 映射可视为：

```text
Accept（非正式适配评价）
```

### 7.3 主要严谨性 finding

#### Major：默认 base 替代结论缺少 multi-seed

现有证据足以支持“异常已修复”和“V3+ 是强 control”，不足以支持“V3+ 无条件取代 V3”。

修复：执行 paired seed 1/2/3，报告 mean/std 和 paired delta。

#### Minor：batch-size 结论依赖另一台服务器结果

主文档记录 batch size 已确认不影响精度，但当前 feature 目录没有把该对照的完整命令、JSON 和表格统一归档。

修复：在最终论文证据表补充另一服务器的 run id、commit、JSON 路径和指标，不需要重跑。

#### Suggestion：区分 architecture control 与 compatible pipeline

论文建议使用：

```text
V3+ aspp: strict high-level architecture control
V3+ aspp_up: best CFSSeg-compatible V3+ pipeline
```

避免把 feature-interface 适配收益全部归给模型升级。

## 8. 最终行动决策

### 8.1 现在停止什么

- 停止继续扫描 V3+ gamma/norm。
- 停止继续尝试统一 class-cap。
- 停止在当前 feature 分支叠加五方法和 pseudo-label。
- 不直接把当前 feature merge main。

### 8.2 现在推进什么

1. 清理并 push 当前 feature 的已验证提交。
2. 在最新 main 上建立 V3+ integration worktree。
3. 把 V3+ 做成 `run.sh` 的一等可选 MODEL，并实现 model-aware source。
4. 完成三项晋级实验。
5. 通过后合并 main，但 DeepLabV3 继续作为默认和 canonical baseline。

### 8.3 一句话结论

> DeepLabV3+ 异常已经修复，最佳 V3+ pipeline 已超过现有 V3 baseline；它现在应该进入主代码库成为可选 stronger base，但应通过最新 main 上的选择性集成和 multi-seed/迁移验收，而不是把当前已分叉、带 dirty 文件的 feature 分支直接整体合并或立即取代 DeepLabV3 canonical baseline。
