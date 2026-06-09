# CFSSeg 精读笔记

> 主题：从零掌握 **CFSSeg**，并把它放回庄辉平教授“解析学习 / 闭式解 / 递归最小二乘”这一整条方法论主线上理解。
>
> 阅读定位：这不是一篇单纯的“分割论文”，而是一篇把 **ACIL 系列闭式解范式**迁移到 **2D/3D 类增量语义分割** 的论文。

---

## 0. 先给一句总判断

**CFSSeg 的本质，不是把整个分割网络闭式求解；而是把“增量学习里最容易遗忘的分类头更新”从 SGD 改造成递归闭式解，再用冻结编码器、随机高维映射和伪标签去补齐稳定性、塑性与语义漂移。**

这句话非常重要，因为很多人第一次读标题会误以为：

- “closed-form solution” = 整个 segmentation model 都闭式训练；
- “no forgetting” = 在任意意义下都完全不忘；
- “one epoch” = 从头到尾都不用 BP。

这三点都不准确。

更准确的理解是：

1. **初始编码器仍然用 BP 训练**；
2. **增量阶段只把 head 的学习改成闭式递归更新**；
3. “不忘”成立在 **冻结特征 + 闭式头递归更新等价于联合训练** 这个条件下。

---

## 1. 论文试图解决的精确问题

### 1.1 任务是什么

CFSSeg 处理的是 **Class-Incremental Semantic Segmentation (CSS / CISS)**。

和普通分类不同，分割是 **dense prediction**：

- 对 2D 图像，每个 pixel 都要预测语义类别；
- 对 3D 点云，每个 point 都要预测语义类别。

在增量设定下，模型需要按 step 持续接收新类别，同时保住旧类别能力。

### 1.2 为什么分割比分类更难

CFSSeg 的问题意识很清楚：分割的忘却更严重，不只是因为类别增量，而是因为 **像素/点级别监督** 带来三个额外麻烦：

1. **样本量巨大**：每张图都对应成千上万 pixel/point 样本；
2. **背景语义复杂**：旧类经常在新 step 中被标成 background；
3. **语义漂移（semantic drift）**：在 disjoint / overlapped setting 中，真实旧类像素会“塌进背景”。

所以它面对的不是“普通 CIL + segmentation head”，而是：

> **稳定性—塑性矛盾 + dense prediction + 背景漂移**

### 1.3 作者认为 prior work 的核心瓶颈是什么

论文对 prior 的判断其实很“庄式”：

- replay / regularization / distillation 这些 CSS 方法，大多仍然依赖 **gradient-based iterative optimization**；
- 只要还是 SGD/BP 在连续改参数，就会天然存在 **task-recency bias**；
- 对 segmentation 这种像素级任务，forgetting 会被进一步放大。

所以 CFSSeg 不是去设计更复杂的蒸馏项，而是直接改写“学 head 的方式”：

> **把增量 head 更新改成闭式解递归更新，让它等价于联合训练。**

---

## 2. 核心方法：四个部件，分别解决四个问题

CFSSeg 可以拆成 4 个最核心的模块：

1. **Base Encoder Training**：先用 BP 训练一个 encoder；
2. **RHL / Buffer**：把冻结特征映射到更高维空间，提升可分性；
3. **C-RLS / Recursive Ridge Regression**：用递归闭式解更新分类头；
4. **Pseudo-Labeling**：缓解 segmentation 特有的 semantic drift。

这四块各有明确分工。

### 2.1 冻结编码器：优先保稳定性

第一步仍然是标准做法：

- step 1 用 SGD 训练 encoder；
- 之后把 encoder **freeze**，当作 feature extractor。

直觉上，这是把“最危险、最容易遗忘的大参数块”锁住。这样做的好处是：

- 旧知识不再被梯度直接冲掉；
- 后续 head 的递归更新可以获得数学等价性。

但代价也明显：

- backbone 对新类的表征能力不会继续长；
- 论文后续必须想办法补偿 plasticity。

### 2.2 RHL：为什么一定要有高维映射

冻结 backbone 会伤害 plasticity，所以 CFSSeg 插入一个 **Randomly-initialized Hidden Layer (RHL)**：

\[
E_t = \mathrm{ReLU}(X^{\text{encoder}}_t \Phi_E)
\]

它不是为了“学更多参数”，而是为了把 frozen embedding 投到一个更高维空间，让线性头更容易分开新旧类。

这其实延续了 ACIL 的 feature expansion / buffer 思想，背后的逻辑是：

- 解析学习的 head 本质上是线性回归；
- 线性模型表达力有限，容易 underfit；
- 先做高维非线性映射，再做线性闭式回归，会更有判别性。

所以 RHL 的角色不是装饰，而是：

> **用高维投影把“冻结 backbone 带来的塑性损失”找回来。**

### 2.3 Ridge Regression：增量阶段真正的核心

有了高维特征后，作者不再用 cross-entropy + BP 训练分类头，而是求解 ridge regression：

\[
\hat{\Phi}_1=(E_1^\top E_1+\gamma I)^{-1}E_1^\top Y_1
\]

其中：

- \(E_1\)：当前 step 的 pixel/point 特征矩阵；
- \(Y_1\)：对应 one-hot label；
- \(\gamma\)：正则项。

注意这里最大的观念切换：

> **把 segmentation head 训练，改写成“像素/点特征到 one-hot 标签”的矩阵回归。**

分割在这里被还原为大量 element-wise sample 的线性分类问题。

### 2.4 C-RLS：为什么它能“不忘”

仅有 ridge regression 还不够，关键在 **递归更新**。

CFSSeg 定义了累计特征的逆自相关矩阵：

\[
\Psi_{t-1}=(E_{1:t-1}^\top E_{1:t-1}+\gamma I)^{-1}
\]

然后把 step \(t\) 的 closed-form head 更新写成递归形式：

\[
\hat{\Phi}_t=
\left[
\hat{\Phi}_{t-1}-\Psi_t E_t^\top E_t \hat{\Phi}_{t-1}+\Psi_t E_t^\top \bar{Y}^{train}_t,
\;\Psi_t E_t^\top \tilde{Y}^{train}_t
\right]
\]

\[
\Psi_t=(\Psi^{-1}_{t-1}+E_t^\top E_t)^{-1}
\]

这里有两个非常关键的理解点。

#### 理解点 A：这是“联合训练的递归写法”

它不是经验式更新，不是 heuristic，不是近似 replay。

论文要证明的是：

> 当前 step 只看新数据做递归更新，最后得到的 head，和把历史数据全部摊开做一次联合闭式求解，是同一个解。

这就是整条 ACIL 系列论文最核心的 **weight-invariant / absolute memorization** 思想。

#### 理解点 B：为什么分成 covered / uncovered 两部分

CFSSeg 在 segmentation 里不是简单照搬 ACIL，而是引入了：

- \(\bar{Y}^{train}_t\)：covered classes，对应旧类相关标签信息；
- \(\tilde{Y}^{train}_t\)：uncovered classes，对应当前新类。

这一步其实很像 GACL 里的 exposed / unexposed split，只不过换成 segmentation 语境。

也就是说：

- 新类列是“新增列”；
- 旧类列要被当前 step 的信息修正；
- 所以递归项里不仅有新类块，也有对旧类块的 gain / correction。

### 2.5 Pseudo-Labeling：分割特有的补丁，不是闭式解本身

闭式解解决的是“head 的更新不忘”，但 segmentation 还多了 **semantic drift**。

在 disjoint / overlapped setting 中，当前 step 的 background 里可能混有旧类。直接把它们当背景，会把旧类往 background 上压。

所以 CFSSeg 使用 uncertainty-guided pseudo-labeling。

2D 情况下，作者先用旧模型输出不确定度：

\[
U_i = 1 - \sigma(\max_c q_{\theta_{t-1}}(i,c))
\]

再根据阈值 \(\tau\) 决定：

- 若当前标签就是新类，保留 GT；
- 若当前标为 background 且旧模型对其旧类预测足够可信，则用旧模型 pseudo label；
- 否则仍保持 background。

3D 则更复杂：

- 用 KNN 找空间邻域；
- 用 MC-dropout / BALD 估计点不确定度；
- 必要时借邻近可信旧类点的标签补偿。

所以 pseudo-labeling 的真正角色是：

> **把 segmentation 特有的“标签污染”问题压住，让闭式 head 学到的不是被错误背景破坏过的监督。**

---

## 3. CFSSeg 与庄教授早期闭式解论文的关系

这部分最重要。你要把 CFSSeg 放在下面这条线里理解：

### 3.1 更早的解析学习祖先思想

解析学习本身不是从 CIL 才开始的。它来自更早的 analytic learning / pseudoinverse learning：

- 把神经网络中的一部分训练，改写成矩阵最小二乘；
- 尽量避免 BP 的梯度问题与多轮迭代。

庄教授这条线里，你最该抓住的祖先思想是：

1. **冻结或先得到一个特征提取器**；
2. **在高维映射空间里解线性回归/分类**；
3. **保存相关矩阵，而不是保存旧样本**；
4. **递归更新应当等价于联合训练。**

### 3.2 ACIL：把 closed-form 带进类增量学习

ACIL 是这条线最关键的第一篇 anchor。

它做的事情可以概括成一句话：

> 在分类 CIL 中，把最后的 classifier 改成解析学习，并证明“递归增量更新 = 联合闭式训练”。

ACIL 的核心结构是：

- base 阶段先用 BP 训练 CNN backbone；
- 再做 analytic re-alignment；
- 增量阶段冻结 backbone，只递归更新 FCN head；
- 保存的是自相关矩阵 \(R_k\)，不是样本。

所以 ACIL 给 CFSSeg 奠定了两个根基：

1. **数学根基**：递归闭式更新可以严格等价于 joint learning；
2. **方法论根基**：新样本到来时，不一定要 replay old data，也不一定要 BP 微调全部参数。

### 3.3 GKEAL：few-shot 下增强判别性

GKEAL 往前推进了一步：

- ACIL 的 feature expansion 还是比较朴素；
- GKEAL 用 Gaussian kernel embedding 增强特征判别性；
- 再用 AFC 平衡 base/new 偏置。

它对 CFSSeg 的启发是：

> **一旦 backbone 冻结，plasticity 的补偿就必须通过“更强的 feature space 设计”来完成。**

CFSSeg 的 RHL 正是这一思想在 segmentation 中更轻量的实现。

### 3.4 GACL：处理类重现 / 广义增量情形

GACL 的理论贡献对你理解 CFSSeg 很有帮助。

GACL 把普通 CIL 的“新旧类严格互斥”放宽到 generalized setting：

- 一个 task 里既可能有全新类；
- 也可能有以前出现过的 exposed classes。

所以 GACL 引入：

- exposed / unexposed split；
- ECLG（Exposed Class Label Gain）项。

你会发现，CFSSeg 的 covered / uncovered 写法，与 GACL 的 exposed / unexposed 在结构上非常接近。差别只是：

- GACL 是分类任务；
- CFSSeg 是 segmentation 里的 dense label 版本，还叠加 semantic drift。

### 3.5 Any-SSR：说明这不是“图像分类专用 trick”

Any-SSR 把同一思路带到 LLM continual learning：

- 任务专属 LoRA 负责塑性；
- analytic router 负责不忘；
- router 仍然由递归最小二乘训练。

它的意义不在于与你当前 segmentation 最直接相关，而在于说明：

> **庄教授这一套闭式解范式，本质是“如何在不访问历史数据时，递归地保持联合训练解不变”。这是一条跨模态的方法论。**

所以你不能把 CFSSeg 理解成 isolated paper，它是这整条 analytic continual learning family 在 segmentation 上的一个分支落地。

---

## 4. 数学上到底该怎样理解 CFSSeg

这一段专门给你“从零掌握”的版本。

### 4.1 第一步：把分割看成很多个 element-wise sample

一张图经过 encoder 后，会得到每个 pixel 的 feature；
一块点云经过 encoder 后，会得到每个 point 的 feature。

于是对 step \(t\) 来说，你可以把它看成：

- 特征矩阵：\(E_t \in \mathbb{R}^{N_t \times d_E}\)
- 标签矩阵：\(Y_t \in \mathbb{R}^{N_t \times C_t}\)

其中 \(N_t\) 是 pixel/point 的总数。

这一步是整个迁移的桥梁：

> **把 segmentation 还原成“大样本数、多类别线性回归问题”。**

### 4.2 第二步：闭式解只解决 classifier，不解决 feature extractor

这是最容易被误解的地方。

闭式解求的是：

\[
\Phi^* = \arg\min_\Phi \|Y-E\Phi\|_F^2 + \gamma \|\Phi\|_F^2
\]

也就是分类头参数 \(\Phi\)。

而 encoder：

- 在 step 1 用 BP 训练；
- 增量阶段被冻结。

所以“closed-form solution for segmentation”更准确地说是：

> **closed-form solution for the incremental segmentation classifier head**

而不是整个 segmentation network 的 end-to-end closed-form。

### 4.3 第三步：为什么说“不忘”

如果把 1 到 t 的所有特征与标签一次性堆起来，joint solution 是：

\[
\hat{\Phi}_t=(E_{1:t}^\top E_{1:t}+\gamma I)^{-1}E_{1:t}^\top Y_{1:t}
\]

CFSSeg 做的是：

- 不存旧数据；
- 只存 \(\Psi_{t-1}\) 和旧 head；
- 当前 step 到来时，用 Woodbury 型递推得到新 \(\Psi_t\) 与新 \(\hat{\Phi}_t\)。

如果递推严格等于 joint solution，那就意味着：

- 旧知识没有因为增量更新被“重新偏置”；
- 当前结果与“历史数据全在场”时一致；
- forgetting 不是靠 regularization 减轻，而是从优化形式上被绕开。

### 4.4 第四步：为什么还会有性能 gap

如果它真的等价于联合训练，为什么不是最强？

因为它等价的是：

> **在“冻结特征”这个前提下的联合闭式头训练**

而不是：

> **全模型都能继续适应新任务的 joint BP training**。

所以 gap 主要来自 backbone：

- backbone 只在 base step 学过；
- 对后来新类的 representation 可能不够新鲜；
- RHL 只是部分补偿，不可能彻底替代 representation learning。

这也是 CFSSeg 未来最值得改的地方。

---

## 5. 论文最有说服力的证据

### 5.1 顺序 setting：15-1 和 15-5 给出相同结果

这是我认为整篇论文里最有辨识度的证据。

在 Pascal VOC sequential setting 下，论文报告：

- 15-1：old 78.1 / new 42.0 / all 70.0
- 15-5：old 78.1 / new 42.0 / all 70.0

这不是普通“效果不错”，而是：

> **闭式解在数学上与联合解一致，因此不同 class grouping 只要不改变同一类样本集合，结果可以保持一致。**

这条证据比单纯涨点更说明方法内核。

### 5.2 challenging setting 下优势更明显

在 VOC 的更难设定里，CFSSeg 的优势更大：

- disjoint 15-1：68.77 all
- disjoint 10-1：57.17 all
- overlapped 15-1：69.36 all
- overlapped 10-1：58.91 all

尤其 10-1 这种多 step 设定，正是 forgetting 最容易爆发的地方。

### 5.3 3D 结果说明不是 2D 特例

这篇论文非常好的地方是，它不只在 2D 图像上做验证，还扩展到了：

- S3DIS
- ScanNet

说明这套 closed-form head 不是 DeepLab/VOC 特供，而是可以迁移到 point-level segmentation。

### 5.4 消融：RHL 和 pseudo-labeling 缺一不可

这张表非常关键，因为它解释了方法的两个核心补丁为什么必要。

在 VOC overlapped 10-1 上：

- 完整模型：75.02 / 41.20 / 58.91
- 去掉 RHL：63.91 / 9.36 / 37.94
- 去掉 pseudo-labeling：71.83 / 36.19 / 54.86

这告诉你：

- **RHL 主要补 plasticity**，尤其对新类至关重要；
- **pseudo-labeling 主要补 semantic drift**，对 old/new 都有帮助。

### 5.5 效率优势是真正“工程上有价值”的

相比 fine-tuning 10 epochs：

- FT 总时间：651.46 s
- Ours 总时间：43.25 s

也就是大约 **15×** 加速。

同时：

- batch size 更大（64 vs 32）
- GPU memory 更低（51.61 GB vs 59.55 GB）

这类结果对于 continual segmentation 很重要，因为 dense prediction 的训练代价本来就高。

---

## 6. 成立前提与脆弱点

一篇论文值不值得 follow-up，关键看这里。

### 6.1 它成立的前提

CFSSeg 强成立的前提包括：

1. **冻结 backbone 的表示还足够好**；
2. **随机高维映射足以补偿线性 head 的表达力不足**；
3. **当前 step 的伪标签质量足够高，不会污染 covered old classes**；
4. **分类头是主要 forgetting 源，而不是 backbone 表示漂移本身**。

### 6.2 它最脆弱的地方

#### 脆弱点 A：backbone plasticity 不足

CFSSeg 的 closed-form 优势，是靠 freeze backbone 换来的。

所以一旦：

- base classes 与 later classes 语义差异很大；
- 或者 3D 新场景分布偏移很大；
- 或者长期 step 数很多；

冻结表示可能会成为瓶颈。

#### 脆弱点 B：RHL 是随机的，表达上限不够可控

RHL 很轻量，但它本质上是随机映射。

优点是快；
缺点是：

- 它不利用任务结构；
- 不保证对所有 step 都同样合适；
- 维度越大，矩阵逆开销越重。

#### 脆弱点 C：semantic drift 仍然依赖伪标签质量

伪标签策略虽然有效，但仍然是 heuristic：

- 2D 依赖阈值 \(\tau\)；
- 3D 依赖 MC-dropout / KNN / BALD 等额外设计；
- 一旦不确定度估计失真，旧类可能仍被错误吞并为背景。

#### 脆弱点 D：closed-form 等价不等于 end-to-end 最优

它的“绝对记忆”是相对于 **冻结特征下的联合闭式头** 而言，不是相对于 **全模型 joint training**。

所以 reviewer 很可能会问：

> 你是“不忘了”，还是“只是因为你不再更新最会忘的那部分”？

这个问题不是致命缺点，但你做 follow-up 时必须正面回答。

---

## 7. 这篇论文对你现在的价值

### 7.1 为什么它值得精读

对你当前课题，它属于 **必须精读** 的 anchor 级论文，原因有三：

1. 你导师已经明确要求以 CFSSeg / SegACIL 为 code base；
2. 它和庄教授的 ACIL/GKEAL/GACL 是一条连续理论线；
3. 它最适合做 follow-up 的点恰恰很清楚：
   - frozen representation 的塑性不足；
   - random buffer 的结构太粗；
   - segmentation 的 semantic drift 还可以更 principled。

### 7.2 你最应该记住的三句话

1. **CFSSeg 的新意，不是又做了一个 distillation loss，而是把增量分割 head 的学习机制从 BP 改成了闭式递归更新。**
2. **RHL 和 pseudo-labeling 不是附属细节，它们分别是对 plasticity 与 semantic drift 的针对性补偿。**
3. **这篇论文最值得延伸的方向，不是推翻 closed-form，而是在保持 closed-form 主干的同时，增强表示塑性与语义建模。**

---

## 8. 如果你要在它上面继续做研究，我建议的 3 个优先方向

### 方向 1：把 GACL 的 exposed-class gain 思想迁移到 segmentation

**想改哪里**：把 segmentation 中 old-class reappearance / pseudo-old supervision 显式写成一个 gain term，而不是只靠 heuristic pseudo-label。

**为什么有机会提升**：

- 数学上更整齐；
- 和庄教授近年的主线高度一致；
- 在 overlapped / disjoint 设定下非常自然。

**风险**：

- dense label 的 covered/uncovered 分解未必像分类那么干净；
- 伪标签噪声会直接进入增益项。

**第一组最小实验**：

- 先在 VOC 10-1 overlapped 做；
- 对比原始 pseudo-label vs confidence-weighted covered gain；
- 观察 old/new/all mIoU 与 pseudo-label precision。

### 方向 2：在 closed-form head 外加一个小型 plastic residual branch

**想改哪里**：保持 analytic head 不动，再叠一个极小的 residual head / adapter / low-rank correction。

**为什么有机会提升**：

- 直接打中 frozen backbone 的塑性瓶颈；
- 可以借鉴 DS-AL / REAL 那类“稳定主干 + 塑性补偿”的思想；
- 很可能提升 new-class mIoU。

**风险**：

- 一旦 residual branch 太强，会把 closed-form 的不忘优势重新破坏；
- 需要设计稳定的融合方式。

**第一组最小实验**：

- 只在 decoder 末端加 rank-limited residual projection；
- 比较 no residual / additive residual / gated residual；
- 先看 sequential 15-1，再看 overlapped 10-1。

### 方向 3：把随机 RHL 换成更结构化的 buffer / kernel buffer

**想改哪里**：把单纯 random ReLU buffer 改成更有结构的高维映射，比如 orthogonal random features、random Fourier features、prototype-conditioned buffer 或轻量 kernel approximation。

**为什么有机会提升**：

- CFSSeg 的 plasticity 很大程度押在 RHL 上；
- 这部分最轻量，也最容易替换；
- 很可能以极小成本换到更高的新类可分性。

**风险**：

- 如果设计太复杂，会丢掉 closed-form 方法的简洁性；
- 维度/时间/显存三者可能重新失衡。

**第一组最小实验**：

- 原始 random ReLU vs orthogonal random ReLU vs RFF；
- 固定 dE，测 new-class mIoU、总时间、显存；
- 看是否能在不增加太多代价下提升新类性能。

---

## 9. 最后的结论

**CFSSeg 对你来说绝对不是“看懂即可”的论文，而是应该真正吃透并转化成实验动作的 anchor paper。**

它最值得你的地方，不只是结果强，而是：

- 数学主线清晰；
- 代码实现边界清楚；
- 弱点也暴露得很明确；
- 非常适合作为你继续做 conference paper 的母体。

---

## 附：建议你读论文时的顺序

如果你接下来要重新精读一遍 CFSSeg，我建议按这个顺序：

1. 引言：只抓“问题定义 + 为什么 SGD 方案不够”；
2. 4.1 Ridge Regression：看 closed-form 基础；
3. 4.2 C-RLS：这是全文数学核心；
4. 4.4 / 4.5 Pseudo-Labeling：这是分割特有增量问题的关键；
5. 5.2 Main Results：先看 sequential/disjoint/overlapped；
6. 5.3 Ablation：重点看去掉 RHL 和 pseudo-labeling；
7. 再回头把它和 ACIL/GKEAL/GACL 对上。

这样你会快很多。
