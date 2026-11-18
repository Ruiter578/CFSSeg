# SegACIL / CFSSeg 课题 Home

> 最后更新：2026-06-09  
> 用途：本文件作为本课题给 Claude Code / Codex / 后续人工接手时的优先入口文档。做实验、改代码、写论文前，先读这里，再读对应的 `Codex_Plans/PLAN.md` 和 `AI_docs/idea构思与实验设计`。

## 0. 协作与维护规则

1. 本课题的主线不是“单纯换更强分割网络”，而是在 CFSSeg 的解析持续学习框架上做可解释、可复现、能写论文的轻量改进。
2. 任何代码改动都要优先保持 CFSSeg 主框架：冻结 encoder、RHL 高维映射、C-RLS 闭式更新分类头、伪标签缓解 semantic drift。
3. 更新本文档时，请只更新已验证事实和明确决策。未跑完的判断写成“待验证”，不要写成结论。
4. 实验记录应至少包含：数据协议、setting、subpath、step0 来源、模型、buffer、gamma、是否启用伪标签、old/new/all mIoU、异常信息。
5. 未经明确要求，不要自动 commit 或 push。当前仓库常有本地实验改动，先 `git status --short` 再动手。

## 1. 课题背景

导师庄教授是解析持续学习方向的重要提出者和开创者。本课题是在庄教授此前参与的 CFSSeg 论文基础上，使用其公开代码 `qwrawq/SegACIL` 做复现、改进、实验迭代，并形成一篇 AI 会议论文。

课题的现实目标有两层：

1. 完成导师结项要求中关于 PASCAL VOC 2012 `15-5` 类增量语义分割的指标。
2. 在已有 CFSSeg 结果已经明显超过结项指标的基础上，提出一个足够干净、可解释、可复现的新模块或方法组合，使其具备投稿论文的贡献点。

当前代码仓库主要覆盖 2D PASCAL VOC 流程。CFSSeg 论文包含 2D 图像与 3D 点云结果，但当前 SegACIL 仓库中没有完整可直接运行的 3D DGCNN/S3DIS/ScanNet 分支。因此短期论文与实验应先围绕 2D VOC 完成。

## 2. 硬指标与实验协议

从 `AI_docs/完整结项指标.png` 可读到的结项指标为：

| 项目 | 指标 |
|---|---|
| 指标名称 | 指标 3.1：在 PASCAL VOC 2012 数据上，在 `15-5` 设置下模型的 mIoU(%) |
| 参考值 | 64.3% |
| 单模型目标 | 65.9% |
| 集成系统目标 | 67.0% |

`6月课题规划.pdf` 给出的 30 天目标为：

1. 在 PASCAL VOC 2012 `15-5` 设置下完成主实验、消融实验和集成系统实验。
2. 形成稳定、可解释、可复现的改进方案。
3. 完成 EI 会议论文的实验整理、论文撰写与投稿。

VOC `15-5` 协议含义：

| 阶段 | 类别范围 | 含义 |
|---|---|---|
| step0 | 0-15 | background + 15 个初始前景类 |
| step1 | 16-20 | 一次性新增 5 个前景类 |
| final evaluation | 0-20 | 全部 21 类 mIoU |

注意：`15-5` 是类别划分协议，`sequential` / `disjoint` / `overlap` 是标签可见性设置。当前最稳的复现链路是 `15-5 sequential`，但伪标签模块真正有价值的验证场景是 `disjoint` 和 `overlap`。

## 3. CFSSeg 一句话总结

CFSSeg 的本质不是把整个分割网络闭式求解，而是把增量阶段最容易遗忘的分割分类头从 SGD/BP 训练改成递归岭回归闭式更新；同时用冻结 encoder 保稳定性，用 RHL 补可塑性，用 pseudo-labeling 修复分割任务中特有的旧类被标成背景造成的 semantic drift。

可以把它记成四个模块：

| 模块 | 解决的问题 | 在代码中的位置 |
|---|---|---|
| Base encoder training | 先获得可用分割表征 | `trainer/trainer.py` 的 `curr_step == 0` |
| Frozen encoder | 防止增量阶段梯度冲掉旧知识 | `step>0` 加载旧模型并移除分类头 |
| RHL / RandomBuffer | 冻结特征后补偿新类可塑性 | `network/Buffer.py:RandomBuffer` |
| C-RLS / RecursiveLinear | 不存旧样本，递归更新解析分类头 | `network/AnalyticLinear.py:RecursiveLinear` |
| Pseudo-labeling | 修复旧类被压成背景的 semantic drift | `trainer/trainer.py:get_pseudo_labels()` |

## 4. 小白版核心原理

### 4.1 分割为什么能写成矩阵问题

语义分割不是给整张图一个类别，而是给每个像素一个类别。假设一张图经过 encoder 后，得到每个像素的特征。把所有像素摊平后，就得到一个大矩阵：

$$
X \in \mathbb{R}^{N \times d}
$$

其中：

- $N$ 是像素样本数，可以是一批图里所有 feature map 位置的总数；
- $d$ 是每个像素的特征维度；
- 每一行就是一个像素样本。

标签也可以摊平成 one-hot 矩阵：

$$
Y \in \mathbb{R}^{N \times C}
$$

其中 $C$ 是当前已经见过的类别数。于是分割头可以先近似看成一个线性分类器：

$$
\hat{Y} = XW
$$

这里 $W$ 就是每个像素特征到类别分数的分类头参数。

### 4.2 岭回归闭式解

如果用平方误差训练这个线性头，并加入正则项，目标为：

$$
\min_W \|Y-XW\|_F^2 + \gamma \|W\|_F^2
$$

对 $W$ 求导并令梯度为 0，可得到：

$$
(X^\top X + \gamma I)W = X^\top Y
$$

所以：

$$
W^* = (X^\top X + \gamma I)^{-1}X^\top Y
$$

这就是“闭式解”：不需要反复 `loss.backward()`，而是用矩阵乘法和矩阵求逆直接得到最优线性头。

### 4.3 为什么能持续学习而不存旧数据

如果第 1 到第 $t$ 步所有数据都在，联合训练的闭式解是：

$$
W_t = (X_{1:t}^\top X_{1:t} + \gamma I)^{-1} X_{1:t}^\top Y_{1:t}
$$

关键观察：这个解不需要保存每条旧样本本身，只需要保存特征相关统计量。定义：

$$
R_t = (X_{1:t}^\top X_{1:t} + \gamma I)^{-1}
$$

新 step 来了以后：

$$
R_t = (R_{t-1}^{-1} + X_t^\top X_t)^{-1}
$$

然后再用当前数据修正分类头。这就是 C-RLS 的核心。它的理论意义是：在 encoder 冻结、RHL 映射固定、标签构造一致的前提下，递归更新可以等价于把历史数据全部拿回来做一次联合闭式训练。

### 4.4 为什么还需要 RHL

冻结 encoder 可以保稳定性，但也会牺牲新类适应能力。CFSSeg 在 encoder 特征后插入 RHL：

$$
E = \operatorname{ReLU}(X^{encoder}\Phi_E)
$$

其中 $\Phi_E$ 是随机初始化且固定的高维映射矩阵。直觉是：原始 256 维像素特征可能不够线性可分，映射到 8192 维左右的非线性空间后，线性闭式头更容易分开新旧类。

所以 RHL 不是普通可训练 MLP，而是冻结特征和解析线性头之间的固定 feature lift。它是当前最自然的改进入口，因为它影响解析头输入空间，但不破坏 C-RLS 主公式。

### 4.5 为什么还需要伪标签

在 `disjoint` 和 `overlap` 设置中，当前 step 的标注经常把旧类像素写成 background。这样训练时模型会收到错误信号：本来是旧类的像素，被要求变成背景。这就是 semantic drift。

CFSSeg 的伪标签策略做的是：

1. 当前像素如果是新类 GT，保留 GT。
2. 当前像素如果标成 background，就让旧模型先预测。
3. 如果旧模型很确定它是某个旧类，就把 background 改回旧类。
4. 如果旧模型不确定，就保持 background。

它的作用不是给新类造标签，而是把背景中混入的高置信旧类“捞回来”，避免旧类被背景吞掉。

## 5. 代码训练逻辑

当前仓库的关键入口：

| 文件 | 作用 |
|---|---|
| `run.sh` | 默认实验脚本，设置 dataset、task、model、step、subpath |
| `train.py` | 解析参数，设置 `opts.num_classes` / `opts.target_cls`，启动 Trainer |
| `trainer/trainer.py` | step0 BP 训练、step1 realign、step>1 C-RLS 更新 |
| `network/_deeplab.py` | DeepLabV3 head，输出 logits 和 256-d dense feature |
| `network/modeling.py` | 模型工厂 |
| `network/Buffer.py` | RHL 随机高维映射 |
| `network/AnalyticLinear.py` | 递归闭式分类头 |
| `utils/tasks.py` | VOC `15-1`、`15-5` 等类别划分 |

三阶段流程：

```text
step0:
  image -> DeepLabV3 + ResNet101 -> logits
  logits -> BCE loss -> SGD/BP
  保存 DeepLab checkpoint

step1:
  加载 step0 DeepLab checkpoint
  去掉原 classifier.head，保留能输出 256-d feature 的 frozen backbone/head_pre
  构造 AIR(backbone, RandomBuffer, RecursiveLinear)
  先用 step0 数据 realign 解析头
  再用 step1 数据闭式拟合新类

step2+:
  加载上一 step 的 AIR
  当前数据 -> frozen backbone -> RHL -> RecursiveLinear.fit()
  不再反向传播
```

重要实现细节：

1. `--method acil` 目前只是脚本参数，代码里没有真正根据 `opts.method` 分支。
2. 默认 `run.sh` 没有启用 `--use_pseudo_label`，所以 sequential 复现不会走伪标签。
3. 论文 2D 设置是 `d_E=8192`、`gamma=1`、`tau=0.4`；当前脚本使用 `BUFFER=8196`、`gamma=1`。
4. 当前代码的解析头输出形状依赖 `AnalyticLinear.forward()` 中 `int(HW**0.5)` 恢复空间尺寸，输入裁剪和 feature map 形状不要随意改。

## 6. 当前进度

### 6.1 已有文档资产

`AI_docs` 已包含四类资料：

1. `论文精读/`：CFSSeg 正文翻译、精读笔记、结构化改进思路。
2. `数学推导与公式解析/`：解析持续学习、岭回归、C-RLS、CFSSeg 公式拆解。
3. `代码理解与学习/`：SegACIL 代码结构、训练逻辑、step0 到后续 steps 的输入输出。
4. `idea构思与实验设计/`：RHL 归一化、自适应伪标签、backbone 替换等初步构思。

### 6.2 已有复现实验

| 实验目录 | 协议 | 阶段 | old mIoU | new mIoU | all mIoU | 备注 |
|---|---|---:|---:|---:|---:|---|
| `checkpoints/1128/voc/15-1/sequential` | 15-1 sequential | step5 | 77.91 | 40.28 | 68.95 | 完整跑到 5 个增量 step，略低于论文 70.0 |
| `checkpoints/20260606/voc/15-5/sequential` | 15-5 sequential | step1 | 78.01 | 42.11 | 69.46 | 已超过结项单模型与集成目标 |
| `checkpoints/20260607/voc/15-5/sequential` | 15-5 sequential | step1 | 77.79 | 43.21 | 69.56 | 当前最好 15-5 sequential 结果 |

结论：按结项指标看，当前 `15-5 sequential` 单模型 all mIoU 约 69.56%，已经高于 65.9% 单模型目标和 67.0% 集成系统目标。但为了论文发表，仍需要做方法改进、消融、统计和故事线，而不是只交复现结果。

### 6.3 当前脚本状态

当前 `run.sh` 已被改成：

```text
TASK="15-5"
SETTING="sequential"
START_STEP=1
END_STEP=1
SUBPATH="${SUBPATH:-$(date +%Y%m%d)}"
BASE_SUBPATH="${BASE_SUBPATH:-}"
DEFAULT_BATCH_SIZE=32
SPECIAL_BATCH_SIZE=32
```

这表示脚本现在主要服务于：复用某个已有 step0 checkpoint，只跑 `15-5` 的 step1。若设置 `BASE_SUBPATH=20260607`，step1 会从 `checkpoints/20260607/voc/15-5/sequential/step0/...` 加载 step0。

## 7. 后续计划精华版

当前日期是 2026-06-09。导师额外给了一周，但整体仍按 30 天左右压缩推进。

### 第一优先级：RHL 归一化

目标：在不改变 backbone、不改变 C-RLS 主公式、不重训 step0 的前提下，优化进入闭式解头的 RHL 特征尺度。

最小贡献表述：

> 原始 CFSSeg 使用随机高维映射提升可塑性，但随机 RHL 输出尺度可能随 batch、类别和 step 波动，影响岭回归矩阵条件数和有效正则强度。我们引入无可训练参数的 RHL 输出归一化，使解析头接收尺度更稳定的高维特征，从而提升数值稳定性和最终 mIoU。

### 第二优先级：自适应伪标签阈值

目标：把固定 `pseudo_label_confidence=0.7` 改成 batch-level 或 class-wise adaptive threshold，主要验证 `15-5 disjoint` / `15-5 overlap`。

注意：sequential 下旧类标签可见，伪标签不是主要矛盾。若只跑 sequential，伪标签收益可能很弱。

### 第三优先级：轻量解析集成

目标：使用同一 encoder，组合不同 RHL seed、gamma、归一化方式或伪标签策略形成 ensemble。优先做低成本集成，不要先上多套大 backbone。

建议集成路线：

```text
shared step0 DeepLabV3-ResNet101
  -> model A: RHL none / gamma 1
  -> model B: RHL l2_sqrt / gamma 1
  -> model C: RHL l2_sqrt / gamma tuned
  -> logits/prob averaging
```

### 第四优先级：论文写作

论文故事线建议：

1. CFSSeg 将类增量分割分类头更新转化为闭式递归更新，解决了梯度微调的遗忘和高成本问题。
2. 但随机 RHL 特征尺度不受控，固定伪标签阈值也难以适应不同类别和不同 batch 的置信度分布。
3. 本工作在不破坏闭式解主干的前提下，提出稳定化 RHL 特征接口和自适应伪标签策略，并用轻量解析集成提升 VOC `15-5` 结果。

## 8. 当前关键决策

1. 短期主线不换 backbone，不把贡献写成“使用更强网络”。主实验保持 DeepLabV3 + ResNet101，保证与 CFSSeg 和多数 baseline 公平比较。
2. DeepLabV3+ 可以作为后续架构鲁棒性实验，但不是第一优先级；当前代码的 ResNet 版 V3+ 入口还没真正接通。
3. RHL 归一化是下一步最稳的代码切口，因为它直接作用于 `E^T E` 的数值性质，且默认关闭时可完全保持 baseline。
4. 伪标签改进应优先在 `disjoint` / `overlap` 验证，不要只用 sequential 判断其价值。
5. 集成系统优先做共享 encoder 的轻量解析头集成，而不是训练多个大 backbone。

## 9. 后续代理接手时的最小阅读顺序

1. 本文档。
2. `Codex_Plans/PLAN.md`，如果存在且与当前任务相关。
3. `AI_docs/idea构思与实验设计/方案一：RHL归一化与是否改backbone、loss、LR.md`。
4. `AI_docs/idea构思与实验设计/Backbone与分割头是否替换评估.md`。
5. `trainer/trainer.py`、`network/Buffer.py`、`network/AnalyticLinear.py`、`run.sh`。

