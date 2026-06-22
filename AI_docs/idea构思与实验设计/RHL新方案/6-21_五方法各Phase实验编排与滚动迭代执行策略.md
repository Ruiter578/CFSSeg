# 五方法各 Phase 实验编排与滚动迭代执行策略

> 日期：2026-06-21
> 主任务：VOC 15-5 sequential step1
> 依据：`Codex_Plans/5方法原理动机与基于优先级排序的完整工作流行动路线.md`
> 本文只提炼代码完成后要执行的实验，以及实验之间真正的先后依赖

## 0. 先给结论

五方法工作流不应被理解成“Phase 0 到 Phase 6 全部严格串行”，也不适合采用以下两个极端：

- 先把五种方法的全部代码一次性写完，再开始看任何结果；
- 每完成一个很小的实验就停止后续开发，反复修改到它涨点为止。

推荐采用 **滚动波次 + 双轨推进**：

1. Phase 0 的实验协议和配置基础设施先完成并冻结；
2. 每个方法形成一个完整、可独立开关、可独立验证的纵向块；
3. 实验轨验证当前块时，实现轨可以推进下一个不存在结果依赖的块；
4. 只在预先定义的“证据门”处暂停组合实验或下游配置选择；
5. 每个单方法允许一次基于机制证据的定向修订，不进行无边界扫参；
6. Phase 4 的组合、Phase 5 的最佳方法复用和 Phase 6 的最终系统必须等待上游证据。

这不是“方法 1 的全部代码先写完”，因为后续组合不会在上游结果未知时固化；也不是“方法 2 的每步都停”，因为互不依赖的实现和实验可以重叠推进。

## 1. 统一术语

- **实现完成**：代码、neutral off 开关、配置持久化、单元测试、runner 和 smoke test 齐全。
- **实验准入**：该实现的 off-path 能复现统一 baseline，可以开始正式训练。
- **单方法证据门**：相对同配置 baseline 的多 seed/成对结果已经完成，能够判断机制方向。
- **组合准入**：参与组合的两个模块都已通过各自的单方法证据门。
- **定向迭代**：实验诊断明确指向某个机制变量后，只修改该变量并复验。
- **探索性结果**：用于理解机制，但不进入最终结论表。
- **冻结结果**：配置由 val 决定，test 只评估一次，可进入最终结论表。

## 2. 当前执行前的现实前提

### 2.1 当前 baseline 处于 step0/step1 交界

`run_origin.sh` 当前配置为：

```text
SUBPATH=20260621_baseline_bs64_32
TASK=15-5
SETTING=sequential
START_STEP=0
END_STEP=1
step0 batch=32
step1 batch=64
BUFFER=8196
```

2026-06-21 17:42 UTC 已生成 step0 `final.pth`，当时 step1 目录尚未出现。该脚本每个 step 都重新执行一次 `python train.py`。因此，在原 checkout 修改 Python 或 shell 文件，可能让紧接着启动的 step1 使用新代码，破坏 baseline 的单一版本身份。

### 2.2 当前 main 不是五方法代码的完整起点

当前 `main` 已包含五方法文档，但 BOA/P0 的完整实现仍位于 `feature/rhl-se-boa-p0-p1`。正式 Phase 0 应先在新的 integration worktree 中审查并整合该分支，不能直接假定 main 已具备 P1.5 的全部开关、runner 和测试。

## 3. Phase 0：统一基线与实验协议

Phase 0 是所有正式结果的共同前置。它不是精度方法实验，而是证明后续结果可归因。

### E0-1：分支整合回归

目的：确认 P0/P1 实现进入新的执行分支后，没有覆盖 main 的路径与配置改动。

检查：

- neutral RHL forward/fit；
- checkpoint save/load；
- `BASE_SUBPATH` 只读加载与 `SUBPATH` 独立写入；
- P0/BOA runner 的 shell 语法；
- `trainer/trainer.py` 中 `curr_step`、step0 loader 和 step1 配置恢复。

### E0-2：三入口 neutral 等价实验

使用同一 base checkpoint、global seed、RHL seed、Batch32、Buffer8196、gamma1：

| 入口 | 方法开关 |
|---|---|
| 通用 runner | legacy、norm none、proto none、weight none |
| BOA runner | legacy baseline、legacy scale、norm none |
| normalization runner | norm none、sample weight none |

比较 checkpoint config、feature trace、all/old/new/per-class。三者必须复现同一 neutral baseline；否则先修 runner，不进入方法实验。

### E0-3：配置 round-trip

训练一个单 batch smoke，保存 checkpoint/run manifest，再加载评估。逐字段核对：

```text
base checkpoint hash
global/RHL/kmeans/gate seed
train/eval batch
buffer/gamma/init/scale/norm
prototype/sample weight/snapshot/ensemble
```

### E0-4：资源 smoke

对 Buffer8196、12288、16392 分别执行初始化、forward、单 batch fit，记录峰值显存和时延。16392 未通过 smoke 时不启动完整训练，但要把“资源不可行”和“方法无效”分开记录。

### Phase 0 顺序

`E0-1 -> E0-2 -> E0-3` 必须串行；`E0-4` 可在 E0-1 后独立执行。所有正式方法实验等待 E0-2/E0-3 通过。

## 4. Phase 1A：BOA-RHL 1.5

### E1A-1：seed 成对复核

| 方法 | init/scale | Buffer | RHL seed |
|---|---|---:|---|
| A0 | legacy baseline/legacy | 8196 | 1,2,3 |
| A1 | orthogonal/legacy | 8196 | 1,2,3 |

固定 global seed 和其他配置。主统计量是每个 seed 的 `A1-A0` paired delta、mean/std，不采用最好单点作为结论。

### E1A-2：方向与尺度解耦

| Case | init | scale | gamma |
|---|---|---|---:|
| B0 | orthogonal | legacy | 1 |
| B1 | orthogonal | unit | 1 |
| B2 | orthogonal | unit | 3 |
| B3 | orthogonal | kaiming | 1 |
| B4 | orthogonal | kaiming | 6 |

同时报告 row norm、feature trace、effective rank、coherence、condition proxy。B2/B4 用于判断收益来自方向几何还是有效正则变化。

### E1A-3：容量与 antithetic 独立方向

| Case | init | Buffer | 独立方向 |
|---|---|---:|---:|
| C0 | legacy baseline | 8196 | 8196 |
| C1 | orthogonal | 8196 | 8196 |
| C2 | legacy baseline | 12288 | 12288 |
| C3 | orthogonal | 12288 | 12288 |
| C4 | legacy baseline | 16392 | 16392 |
| C5 | orthogonal | 16392 | 16392 |
| C6 | antithetic | 16392 | 8196 |

只有资源 smoke 通过的容量才进入完整训练。比较同总维度和同独立方向两种公平性。

### Phase 1A 顺序

`E1A-1` 先确认 orthogonal 信号；`E1A-2` 可与 `E1A-1` 后两组 seed 并行；`E1A-3` 依赖资源 smoke，但不必等所有 scale 实验。BOA 最终配置必须等三组分析完成。

## 5. Phase 1B：RHL-SE 2.0

RHL-SE 2.0 首轮直接消费已有 seeds 1/2/3，不重新训练成员，因此可与 Phase 1A、Phase 2 的训练并行。

### E1B-0：成员与 split 审计

- 生成 member manifest 和 checkpoint hash；
- 验证成员除 RHL seed 外配置一致；
- 以图像为单位划分 calibration/gate-train/gate-select；
- 明确 BCE sigmoid 只作为独立通道分数，可靠性熵使用校准后的归一化分布。

### E1B-1 至 E1B-5：可靠性消融

| 编号 | temperature | margin/entropy | class reliability | consensus | old/new prior | gate |
|---|---|---|---|---|---|---|
| SE2-0 | 否 | 否 | 否 | 否 | 否 | uniform |
| SE2-1 | 是 | 否 | 否 | 否 | 否 | uniform |
| SE2-2 | 是 | 是 | 否 | 否 | 否 | rule |
| SE2-3 | 是 | 是 | 是 | 否 | 是 | rule |
| SE2-4 | 是 | 是 | 是 | 是 | 是 | rule |
| SE2-5 | 是 | 是 | 是 | 是 | 是 | linear |

固定对照包括：强单成员、uniform logit、P0 class-wise、现有 margin gate。报告 ECE/NLL/Brier、switch ratio、beneficial/harmful switch、oracle capture。

### Phase 1B 顺序

`E1B-0 -> SE2-0/1 -> SE2-2/3/4 -> SE2-5`。规则 gate 必须先证明数据隔离和诊断正确，再训练 linear gate。异构 BOA/PGH/Snapshot 成员实验等待相应成员产生，但同构成员实验不等待其他 Phase。

## 6. Phase 2：PGH-RHL-lite

### E2-A：单原型 cosine 与容量公平

| Case | K | prototype | budget | scale |
|---|---:|---|---|---|
| P2-A0 | 0 | none | 8196 random | - |
| P2-A1 | 1 | cosine | additive | trace-match |
| P2-A2 | 1 | cosine | fixed | trace-match |
| P2-A3 | 1 | cosine | additive | 0.5x/1x/2x trace |

先验证 prototype bank 的 train-only 来源、label mapping、count、hash、cache round-trip，再做正式训练。

### E2-B：多原型 k-means

| Case | K | prototype | budget |
|---|---:|---|---|
| P2-B1 | 2 | cosine | additive/fixed |
| P2-B2 | 4 | cosine | additive/fixed |

报告空簇、重复簇、prototype utilization、类内最近原型距离和 per-class 增益。

### E2-C：RBF sigma 搜索

对 K=1/2 的最佳结构执行：

```text
sigma mode = global_median / class_median / prototype_local
multiplier = 0.25 / 0.5 / 1 / 2 / 4
```

val 选择每类模式的最佳配置，冻结后执行 test。K=4 RBF 在 K=4 cosine 确有额外信息时继续执行，而不是从方案中删除。

### E2-D：PGH 与 BOA 组合

在 BOA 和 PGH 均完成单方法证据后做 2x2：baseline、BOA、PGH、BOA+PGH，并计算 interaction。

### Phase 2 顺序

`E2-A -> E2-B -> E2-C` 是表示复杂度递进；原型库实现可与 BOA 实验并行。`E2-D` 严格等待 BOA 与 PGH 单方法结果。

## 7. Phase 3：RHL 归一化 v2

### E3-A：PowerNorm

| Case | beta | clipping | gamma |
|---|---:|---|---|
| A0 | 0 | off | 1 |
| A1 | 0.25 | `[0.25,4]` | 1 / trace-match |
| A2 | 0.50 | `[0.25,4]` | 1 / trace-match |
| A3 | 0.75 | `[0.25,4]` | 1 / trace-match |
| A4 | 1.00 | `[0.25,4]` | 1 / trace-match |

最佳 beta 再比较 clipping 上限 2/4/8。先在 plain RHL 上完成，避免与 BOA/PGH 同时引入。

### E3-B：CA-C-RLS

| Case | weight mode | cap |
|---|---|---:|
| B0 | none | 1 |
| B1 | old_new | 2/4 |
| B2 | inverse_sqrt | 4 |
| B3 | effective_num | 4 |
| B4 | 最佳模式 | 2/4/8 |

类别计数只能来自 train split。`all weights=1` 必须复现普通 RecursiveLinear。

### E3-C：两轴组合

做 PowerNorm on/off x CA-C-RLS on/off 的 2x2，报告 interaction。

### E3-D：作用域迁移

- BOA 最佳配置：random branch PowerNorm on/off；
- PGH 最佳配置：random-only/prototype-only/per-branch/joint；
- 最佳 hybrid feature：CA-C-RLS on/off。

### Phase 3 顺序

`E3-A` 与 `E3-B` 在数学单测完成后可并行；`E3-C` 等待两轴结果；`E3-D` 的 plain 部分不等待 PGH，PGH scope 部分等待 E2。

## 8. Phase 4：单模型组合与交互

只组合已经完成单方法证据的配置：

1. BOA x PGH；
2. 最佳 hidden feature x PowerNorm；
3. 最佳 hidden feature x CA-C-RLS；
4. 最多保留一个由前述 interaction 支持的三模块组合。

每组都保留四个角并固定 base checkpoint、seed、Batch、gamma 和总维度。Phase 4 必须等待 Phase 1A、2、3 的相关证据，不能提前用“全部开关 on”的结果替代。

## 9. Phase 5：Snapshot 系统线

### E5-A：Snapshot 候选生成

新 step0 run 保存指定 epoch/interval 的 checkpoint，记录 manifest、val 指标和 hash。现有 step0 `final.pth` 或 realigned AIR 产物不作为 Snapshot 冒充使用。

### E5-B：选择准则

比较质量 top3 与质量+prediction diversity top3，固定 probe set，报告 pairwise disagreement、logit distance 和单成员质量。

### E5-C：plain RHL 隔离实验

每个 Snapshot 使用同一 RHL seed/config 完成 step1，唯一变量是 `base_ckpt`。

### E5-D：融合策略

比较强单成员、uniform、class-wise 和 RHL-SE 2.0。

### E5-E：最佳 step1 方法

把 Phase 4 的最佳单模型配置应用到同一组 Snapshot，再比较单成员和融合结果。

### Phase 5 顺序

Snapshot 保存基础设施可以提前实现，`E5-A/B/C` 也可以在 Phase 4 前进行；`E5-E` 必须等待 Phase 4。RHL-SE 2.0 的 Snapshot 复验等待 E5-C 成员。

## 10. Phase 6：最终证据

Phase 6 不新增方法开关，只冻结并汇总：

- 单机制表：BOA、PGH、PowerNorm、CA-C-RLS；
- 单模型组合表：通过交互实验的最佳组合；
- 系统表：Snapshot 成员 + RHL-SE 2.0；
- all/old/new/per-class、mean/std、资源、配置 hash；
- val 选择与 test 结果来源。

Phase 6 严格最后执行。

## 11. 哪些实验必须按顺序

| 前置实验 | 后续实验 | 是否必须等待 | 原因 |
|---|---|---|---|
| E0-2/E0-3 | 所有正式方法实验 | 是 | baseline 与配置身份未锁定时不可归因 |
| BOA/PGH 单方法 | BOA x PGH | 是 | 需要四个角计算 interaction |
| PowerNorm/CA 单轴 | E3-C | 是 | 先知道各自主效应 |
| PGH | PGH branch scope normalization | 是 | 没有 PGH 分支无法定义 scope |
| rule gate | linear gate | 是 | 先验证 split、特征和诊断链路 |
| plain Snapshot | Snapshot + 最佳方法 | 是 | 先隔离 backbone 多样性 |
| Phase 1-5 冻结 | Phase 6 | 是 | 最终表不能包含未决配置 |
| BOA seed | RHL-SE 同构成员 | 否 | 使用已有 checkpoint 即可 |
| BOA scale | PGH prototype bank | 否 | 不共享结果依赖 |
| PGH 训练 | plain PowerNorm/CA | 否 | 可先在 plain RHL 上验证 |
| Phase 4 | Snapshot 候选保存 | 否 | 基础设施与候选生成可提前 |

## 12. 三种执行方式比较

### 方式 A：所有代码完成后统一实验

优点：连续编码、接口一次铺开。缺点：后续方法可能建立在错误的接口假设上；大量尚未被证据需要的组合代码会同时进入 Trainer；最终 debug 和归因成本最高。

### 方式 B：每个小实验完成后停止、分析、改完再继续

优点：反馈最及时。缺点：容易把随机波动当成设计信号，反复切换上下文；GPU 训练期间实现工作停滞；整体进度被最慢实验锁死。

### 方式 C：滚动波次 + 双轨推进（推荐）

按以下块组推进：

| 波次 | 实现轨 | 实验轨 | 必须等待的证据门 |
|---|---|---|---|
| W0 | Phase 0 统一基础设施 | neutral 等价/round-trip/resource smoke | 所有正式实验准入 |
| W1 | BOA 诊断 + SE2 同构 gate | BOA seed/scale；SE2 rule | BOA 最终配置、linear gate |
| W2 | PGH 完整分支 + PowerNorm/CA | PGH A/B/C；plain norm/weight | PGH 与两归一化轴单方法证据 |
| W3 | 组合支持 + Snapshot 基础设施 | 2x2 交互；plain Snapshot | 最佳单模型配置 |
| W4 | Snapshot 最佳方法 + 系统汇总 | Snapshot/SE2 最终实验 | Phase 6 冻结 |

当 W1 实验运行时可以实现 W2 中不依赖 W1 结果的 prototype bank、PowerNorm 数学路径和测试；但不能提前固化“BOA+PGH 最佳组合”。这保持开发速度，也保留结果驱动的正确决策。

## 13. 每个波次怎样处理“实验—分析—迭代”

每个单方法使用固定闭环：

1. 在启动前写明假设、唯一变量、矩阵、主指标和诊断指标；
2. 完成预定首轮矩阵，不因第一个坏点立即改设计；
3. 对照强 baseline，区分代码错误、资源失败、机制失败和随机方差；
4. 若诊断明确指向 scale、阈值、容量或校准问题，允许一次定向迭代；
5. 迭代后冻结该方法当前结论，继续下一波次；
6. 未通过单方法证据门的方法仍保留完整实现和报告，但不进入 Phase 4 默认组合。

这样不会因为每个实验都无限返工拖慢进度，也不会因为完全忽略结果而退化成“所有代码一次写完”。

## 14. 当前 baseline 期间的执行边界

当前 baseline 未结束时：

- 可以继续读代码、写文档、创建 sibling worktree、在 worktree 写代码和跑纯 CPU/轻量单元测试；
- 不应在 `/root/2TStorage/lyc/SegACIL` 原 checkout 修改训练代码或 runner；
- 不应在同一 GPU 启动 Buffer/Trainer smoke 或正式实验；
- worktree 输出目录不得与 `20260621_baseline_bs64_32` 或已有实验复用；
- baseline 完成后再在本服务器执行 GPU smoke/正式实验，或把这些实验交给环境已对齐的第二台服务器。

推荐的总体策略是：**现在建立独立 worktree 并推进 W0/W1 的代码与 CPU 测试；GPU 实验等待本机 baseline 结束，或在第二台已同步并完成环境/数据/checkpoint 校验的服务器运行。**
