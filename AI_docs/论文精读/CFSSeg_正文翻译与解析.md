# CFSSeg 正文翻译与图表精析

> 论文：**CFSSeg: Closed-Form Solution for Class-Incremental Semantic Segmentation of 2D Images and 3D Point Clouds**  
> 会议：**MM'25 / ACM Multimedia 2025**  
> 作者：Jiaxu Li, Rui Li, Jianyu Qi, Songning Lai, Linpu Lv, Kejia Fan, Jianheng Tang, Yutao Yue, Dongzhan Zhou, Yunhuai Liu, Huiping Zhuang  
> 阅读目标：按照“**原文英文段落 → 学术中文翻译 → 图表原理解析**”的方式，把论文正文真正读进去。

---

## 使用说明

1. 这份文档尽量覆盖了 **摘要、引言、相关工作、背景、方法、实验、结论** 等正文内容。  
2. 公式为便于阅读进行了重新排版；图表使用原论文截图并按原文大致位置插入。  
3. 图表下方的“解析”和“结论”属于我的精读说明，不是原文直译。  
4. 由于 PDF 文本抽取会打断换行，我对英文段落做了轻微排版清理，但不改变其含义。

---


> 图表说明：本版已重新核对并修复图表资源。Figure 1、Figure 2 采用原论文图截图；Table 1–Table 6 采用论文原页高分辨率裁切图，其中 Table 2 为本次重新从原论文页裁切补全。若某些平台的 Markdown 预览不显示图片，请直接解压本阅读包后在本地 Markdown 编辑器中阅读，或改看同目录下的图片资源。

## Abstract

> 2D images and 3D point clouds are foundational data types for multimedia applications, including real-time video analysis, augmented reality (AR), and 3D scene understanding. Class-incremental semantic segmentation (CSS) requires incrementally learning new semantic categories while retaining prior knowledge. Existing methods typically rely on computationally expensive training based on stochastic gradient descent, employing complex regularization or exemplar replay. However, stochastic gradient descent-based approaches inevitably update the model’s weights for past knowledge, leading to catastrophic forgetting, a problem exacerbated by pixel/point-level granularity. To address these challenges, we propose CFSSeg, a novel exemplar-free approach that leverages a closed-form solution, offering a practical and theoretically grounded solution for continual semantic segmentation tasks. This eliminates the need for iterative gradient-based optimization and storage of past data, requiring only a single pass through new samples per step. It not only enhances computational efficiency but also provides a practical solution for dynamic, privacy-sensitive multimedia environments. Extensive experiments on 2D and 3D benchmark datasets such as Pascal VOC2012, S3DIS, and ScanNet demonstrate CFSSeg’s superior performance.

中文翻译：  
2D 图像和 3D 点云是多媒体应用中的基础数据类型，广泛用于实时视频分析、增强现实（AR，Augmented Reality，增强现实）以及三维场景理解。类增量语义分割（CSS，Class-Incremental Semantic Segmentation，类增量语义分割）要求模型在持续学习新语义类别的同时保留已有知识。现有方法通常依赖基于随机梯度下降（SGD，Stochastic Gradient Descent，随机梯度下降）的高计算开销训练，并配合复杂正则化或**样本回放**。可是，基于随机梯度下降的方法不可避免地会更新承载旧知识的模型参数，从而导致灾难性遗忘，而像素级或点级预测粒度会进一步放大这一问题。为应对这些挑战，本文提出了 CFSSeg，一种新的、无需样本回放的闭式解方法，为持续语义分割任务提供了兼具实践价值和理论依据的方案。该方法避免了迭代式梯度优化和历史数据存储，使每个增量步骤只需要对新样本进行一次遍历。它不仅提升了计算效率，也为动态且隐私敏感的多媒体环境提供了可行解决方案。Pascal VOC2012、S3DIS 和 ScanNet 等 2D 与 3D 基准上的大量实验表明，CFSSeg 取得了优异表现。

---

## 1. Introduction

> 2D images and 3D point clouds are fundamental data modalities that underpin modern multimedia applications, including real-time video analysis, augmented reality (AR), robotics, and immersive 3D scene understanding. However, real-world multimedia systems rarely use fixed and predefined sets of object categories. They often encounter new objects or concepts after initial deployment, requiring the ability to adapt and expand their knowledge base over time. A naive approach is to train models directly on newly arrived data, but this strategy is plagued by catastrophic forgetting, where the model forgets previously acquired knowledge while adapting to new information. To address this issue, continual learning methods have been proposed to mitigate the effects of forgetting while allowing models to gradually adapt to new data.

中文翻译：  
2D 图像和 3D 点云是现代多媒体应用的基础数据模态，支撑着实时视频分析、增强现实、机器人系统以及沉浸式三维场景理解等任务。然而，现实中的多媒体系统很少只处理固定且预定义的对象类别集合。它们在部署后往往还会不断遇到新对象与新概念，因此系统需要随着时间推移持续扩展知识库并适应新类别。一个朴素方案是在新到达的数据上直接继续训练模型，但这种做法会遭受灾难性遗忘，也就是模型在吸收新信息时丢失先前已经掌握的知识。为了解决这个问题，持续学习方法被提出，用以在缓解遗忘的同时让模型逐步适应新数据。

> Class-incremental semantic segmentation (CSS) presents unique challenges. As shown in Figure 1, the model needs to continuously learn to segment new categories. Compared to image classification tasks, semantic segmentation tasks involve pixel/point-level granularity, which typically requires substantial computational resources and makes models more susceptible to catastrophic forgetting. A challenge in CSS is the stability-plasticity dilemma, which involves balancing two conflicting goals: stability and plasticity. Stability refers to the model’s ability to retain knowledge from past tasks, while plasticity requires the model to adapt to new incoming data. Striking the right balance is crucial for successful continual learning.

中文翻译：  
类增量语义分割任务具有独特挑战。如图 1 所示，模型必须持续学习如何分割新类别。与图像分类相比，语义分割是像素级或点级的密集预测任务，通常需要更高计算资源，也使模型更容易受到灾难性遗忘的影响。CSS 中的关键难点之一，是稳定性与可塑性两难。稳定性指模型保留过去任务知识的能力，可塑性则要求模型能够适应新到来的数据。如何在这两者之间取得合适平衡，是持续学习能否成功的核心。

![Figure 1](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/fig1_cfss.png)

**图 1 原理解析**  
图 1 展示的是 **类增量语义分割的学习协议**，而不是具体网络结构。模型从 Dataset 1 开始，只学习最初的一部分类。随后 Dataset 2、Dataset 3、……、Dataset T 依次到来，每一步只引入新的类别子集，但测试时却要面对截至当前步的 **累积类别集合**。  

**应该重点读出的信息：**
- 训练数据是分步到来的，但测试标签空间是不断扩张的。  
- 旧类并不会因为“当前没出现”就不重要，反而必须在后续步骤持续被正确识别。  
- 分割任务比分类更难，因为它不是一张图一个标签，而是大量像素或点的局部决策。  

**结论**：图 1 把整篇论文的问题边界说清楚了。CFSSeg 不是普通 segmentation 方法，而是一个要在不断增添新类时仍保持旧类识别能力的增量式密集预测方法。

> Recent advancements in CSS can be broadly categorized into exemplar-free and exemplar-based approaches. Exemplar-free methods aim to perform class-incremental learning without relying on historical data or features to reduce knowledge forgetting. These methods often employ self-supervised learning, regularization techniques, or dynamic network architectures. On the other hand, exemplar-based methods depend on strategies such as sample replay, feature replay, auxiliary dataset integration, or pseudo-data generated by generative models. While these methods show promise in retaining knowledge, they are all based on gradient descent and inevitably erase past knowledge through gradient updates. Moreover, they often demand significant computational resources, and some exemplar-based methods may not be suitable in scenarios where data privacy is paramount.

中文翻译：  
近年来的 CSS 方法大致可以分成无样本回放和基于样本回放两类。无样本回放方法试图在不依赖历史数据或历史特征的前提下完成类增量学习，以减少知识遗忘，这些方法通常使用自监督学习、正则化技术或者动态网络结构。另一方面，基于样本回放的方法则依赖样本回放、特征回放、辅助数据集整合或生成模型构造的伪数据等策略。虽然这些方法在保留知识方面表现出一定潜力，但它们本质上都仍然依赖梯度下降，因此不可避免地会通过梯度更新侵蚀旧知识。此外，这些方法通常需要较高计算资源；在隐私要求很高的场景里，基于样本回放的方法也未必适用。

> Analytic learning, as an alternative to stochastic gradient descent methods, overcomes key challenges associated with backpropagation, including gradient vanishing and the instability of iterative training processes, by directly computing neural network parameters. Inspired by this, we have proposed CFSSeg, a closed-form solution for CSS. Unlike existing incremental learning methods based on stochastic gradient descent, which require multiple training epochs, our approach needs only a single training epoch. Specifically, we freeze the encoder and update the model using a closed-form solution to achieve stability, while mapping features to a higher-dimensional space to make them more linearly separable, thereby enhancing plasticity. At the same time, it is efficient and privacy-preserving, making it suitable for practical applications. Additionally, in disjoint and overlapped settings, semantic drift can occur, where previously learned categories collapse into background class labels in new datasets. We introduce a pseudo-labeling strategy that leverages uncertainty to mitigate semantic drift. The overview of our method is shown in Figure 2. Extensive experiments on 2D and 3D benchmark datasets such as Pascal VOC2012, S3DIS, and ScanNet have demonstrated its superior performance.

中文翻译：  
解析学习作为随机梯度下降方法的替代方案，能够通过直接计算神经网络参数，避开反向传播中的若干关键问题，例如梯度消失和迭代训练过程的不稳定性。受此启发，作者提出了面向 CSS 的闭式解方法 CFSSeg。与现有依赖随机梯度下降、通常需要多轮训练的增量学习方法不同，本文方法每个增量阶段只需要单个训练 epoch。更具体地说，作者通过冻结编码器并采用闭式解更新来获得稳定性，同时将特征映射到高维空间，使它们更容易线性可分，从而增强模型可塑性。该方法同时具有高效率和隐私友好特性，适合实际应用。此外，在 disjoint 和 overlapped 设置下会出现 semantic drift，也就是此前学过的类别在新数据里被压成背景标签。作者为此引入了一种利用不确定性的伪标签策略来缓解语义漂移。图 2 展示了方法总览。Pascal VOC2012、S3DIS 和 ScanNet 上的大量实验验证了它的优越性能。

> The key contributions are summarized as follows: (1) We propose a novel, gradient-free, closed-form solution for exemplar-free continual semantic segmentation in both 2D images and 3D point clouds. (2) We develop a recursive update mechanism for the classification head, enabling efficient single-pass incremental learning without storing past data. (3) Through extensive experiments on Pascal VOC2012, S3DIS, and ScanNet, we demonstrate that our method achieves advanced results while offering significant advantages in computational efficiency and data privacy.

中文翻译：  
本文贡献可以概括为三点。第一，提出了一种新的、无梯度、闭式解的无样本回放持续语义分割方法，适用于 2D 图像和 3D 点云。第二，设计了一种分类头递归更新机制，使模型能够在不存储历史数据的前提下，以单次遍历方式高效进行增量学习。第三，通过在 Pascal VOC2012、S3DIS 和 ScanNet 上的大量实验，证明该方法不仅结果先进，而且在计算效率与数据隐私方面具有显著优势。

---

## 2. Related Work

> Semantic segmentation, a dense prediction task, involves assigning a semantic label to every pixel in an image. In recent years, significant progress has been made in this domain, primarily driven by the development of convolutional neural network (CNN)-based models. More recently, Transformer-based architectures and innovative Mamba frameworks have gained prominence, introducing novel methodologies and perspectives for addressing the challenges of semantic segmentation. DeepLabV3 has been widely used in previous CSS work, and we selected it as our 2D segmentation model.

中文翻译：  
语义分割是一种密集预测任务，需要给图像中的每一个像素分配语义标签。近年来这一领域取得了显著进展，主要受卷积神经网络模型推动。更近一些，基于 Transformer 的结构以及 Mamba 框架也逐渐受到重视，为语义分割问题带来了新的方法和视角。DeepLabV3 在先前的 CSS 工作中被广泛采用，因此本文将其选作 2D 分割模型。

> For the 3D point cloud modality, key methods include PointNet and its derivative architecture PointNet++, which are used to directly process point cloud data; Transformer models such as Point Transformer V3, which improve performance by capturing long-range dependencies; and DGCNN, which is based on the EdgeConv module, captures local neighborhood information, and learns global shape properties by stacking multiple layers. This dynamic graph approach makes it particularly suitable for handling the unstructured nature of point clouds. In this paper, we adopt DGCNN as our 3D segmentation model due to its simplicity and effectiveness.

中文翻译：  
对于 3D 点云模态，代表性方法包括 PointNet 及其衍生结构 PointNet++，它们可以直接处理点云数据；也包括 Point Transformer V3 这类通过建模长程依赖提升性能的 Transformer 方法；以及基于 EdgeConv 的 DGCNN，它能够捕获局部邻域信息，并通过多层堆叠学习全局形状特征。由于动态图机制特别适合处理点云的非结构化特性，本文选择 DGCNN 作为 3D 分割模型，因为它兼具简洁性和有效性。

> Class-incremental semantic segmentation, initially proposed in medical imaging applications, has since been extended to natural image datasets. Unlike standard classification tasks, CSS poses unique challenges due to its pixel-level granularity, which exacerbates the issue of catastrophic forgetting. CSS methods are broadly categorized into exemplar-free and exemplar-based approaches. Exemplar-free methods often leverage strategies such as self-supervised learning, regularization techniques, or dynamic network architectures to retain knowledge from previously seen data. On the other hand, exemplar-based methods employ mechanisms such as sample replay, feature replay, and auxiliary dataset integration, or utilize pseudo-data or pseudo-features generated by generative models, combining these with new data to enable continual training.

中文翻译：  
类增量语义分割最初是在医学影像场景中提出的，后来扩展到了自然图像数据集。与标准分类任务不同，CSS 因为具有像素级预测粒度，所以灾难性遗忘更严重。现有 CSS 方法大致可分为 exemplar-free 和 exemplar-based 两类。前者通常借助自监督学习、正则化技术或动态网络结构来保留旧知识；后者则依赖样本回放、特征回放、辅助数据集或者生成伪数据与伪特征等方式，把历史信息和新数据结合起来继续训练。

> There is limited work on continual learning for 3D semantic segmentation, and it has only recently begun to be explored. Yang et al. proposed a class-incremental learning method combining geometric features and uncertainty estimation. LGKD introduced a label-guided knowledge distillation loss. Chen et al. investigated class-incremental learning for mobile LiDAR point clouds, proposing strategies for feature representation preservation and loss cross-coupling.

中文翻译：  
面向 3D 语义分割的持续学习工作仍然较少，而且只是近几年才开始被系统探索。Yang 等人提出了结合几何特征和不确定性估计的类增量学习方法。LGKD 引入了标签引导的知识蒸馏损失。Chen 等人则针对移动激光雷达点云研究类增量学习，并提出特征表示保持与损失交叉耦合策略。

---

## 3. Background

> To begin, we define the objective of the semantic segmentation task. The input space is represented as $X \in \mathbb{R}^{N \times C_{in}}$, where $N$ denotes the number of input elements (pixels or point clouds), and $C_{in}$ represents the number of channels per element (e.g., RGB for images, or RGB, XYZ, normals for point clouds). The output label space is $Y \in \mathcal{C}^{N}$, where the set of classes is $\mathcal{C}$, including the background class $c_b \in \mathcal{C}$. Given the training dataset $\mathcal{T}=X\times Y$, the goal is to learn a mapping function $q_\theta$ parameterized by $\theta$ that predicts a per-element class probability distribution: $q_\theta : X \rightarrow \mathbb{R}^{N \times |\mathcal{C}|}$. The segmentation mask is then computed as $\hat y = \arg\max_{c \in \mathcal{C}} q_\theta(x)[i,c]$.

中文翻译：  
作者首先形式化定义了语义分割任务。输入空间记为 $X \in \mathbb{R}^{N \times C_{in}}$，其中 $N$ 表示输入元素数目，对 2D 图像来说是像素，对 3D 点云来说是点；$C_{in}$ 表示每个元素的通道数，例如 RGB，或者 RGB、XYZ、法向量等。输出标签空间写作 $Y \in \mathcal{C}^{N}$，其中类别集合为 $\mathcal{C}$，并包含背景类 $c_b$。给定训练集 $\mathcal{T}=X\times Y$，目标是学习一个由参数 $\theta$ 控制的映射函数 $q_\theta$，为每个元素输出在所有类别上的概率分布。最终分割掩码由每个元素对应的最大概率类别给出，即 $\hat y = \arg\max_{c \in \mathcal{C}} q_\theta(x)[i,c]$。

> In the traditional supervised learning paradigm, the entire training set is provided at once, and the model is trained in a single step. However, in continual learning, training is performed iteratively, with each step introducing new categories along with their corresponding subset of training data. This process spans multiple steps, denoted as $\{\text{step }1, \text{step }2, \cdots, \text{step }T\}$. In step $t$, the label set $\mathcal{C}_{t-1}$ is expanded by adding a new set of categories $\mathcal{S}_t$, resulting in an updated label set $\mathcal{C}_t = \mathcal{C}_{t-1} \cup \mathcal{S}_t$. Simultaneously, a new training subset $\mathcal{T}_t$ is introduced to update the previous model $q_{\theta_{t-1}}$ to $q_{\theta_t}$. According to the CSS principle, the newly introduced category sets are mutually exclusive, i.e., $\mathcal{S}_i \cap \mathcal{S}_j = \varnothing$ for $i \neq j$.

中文翻译：  
在传统监督学习中，整个训练集一次性给出，模型也在单步训练中完成学习。而在持续学习中，训练是迭代进行的，每一步只引入一组新类别以及对应的数据子集。整个过程可表示为 $\{\text{step }1, \text{step }2, \cdots, \text{step }T\}$。在第 $t$ 步，已有标签集合 $\mathcal{C}_{t-1}$ 加上新类别集合 $\mathcal{S}_t$ 后得到更新标签集合 $\mathcal{C}_t = \mathcal{C}_{t-1} \cup \mathcal{S}_t$。同时，新的训练子集 $\mathcal{T}_t$ 被用来把旧模型 $q_{\theta_{t-1}}$ 更新为新模型 $q_{\theta_t}$。按照 CSS 的定义，不同步骤新引入的类别集合是互斥的，也就是当 $i \neq j$ 时，$\mathcal{S}_i \cap \mathcal{S}_j = \varnothing$。

> Different learning settings are considered for CSS, depending on the availability and labeling of categories during incremental learning. Sequential, disjoint, and overlapped settings are detailed below: 1) Sequential Setting. In the sequential setting, labels for both previously learned and newly introduced categories are available simultaneously during each incremental learning step. 2) Disjoint Setting. The disjoint setting introduces complexity by labeling previously learned categories as background in the current task. This phenomenon, known as semantic drift, challenges the model to differentiate between real background and previously learned classes. 3) Overlapped Setting. The overlapped setting further complicates the learning process. Here, only new categories and the background are labeled, but the background label can encompass true background, previously learned categories, and future categories that have not yet been introduced.

中文翻译：  
根据增量学习期间类别标签的可见性与标注方式，CSS 通常区分三种设置。第一，Sequential Setting。在这种设置下，每一步都能同时获得旧类别与新类别的标注。第二，Disjoint Setting。这种设置更复杂，因为当前任务会把以前学过的类别统一标成背景。这种现象被称为 semantic drift，它要求模型区分真实背景和被压成背景的旧类别。第三，Overlapped Setting。它会进一步增加难度，因为当前任务里只有新类别与背景被标注，但背景标签里面可能混入真实背景、旧类别，甚至未来尚未引入的类别。

---

## 4. Method

### 4.1 Ridge Regression

> In step 1, we use stochastic gradient descent to train an encoder. Notably, a powerful pre-trained encoder (e.g., SAM) can also be used to avoid this training process. We then save and freeze the encoder, treating it as a feature extractor.

中文翻译：  
在第 1 步中，作者先用随机梯度下降训练一个编码器。值得注意的是，也可以直接使用强大的预训练编码器，例如 SAM，从而跳过这一步训练。随后，作者保存并冻结这个编码器，把它作为后续阶段的特征提取器。

> After training to obtain an encoder using the gradient descent method, during the continual learning phase, we do not use the gradient descent method to train the model. This is because the update of gradients will inevitably interfere with the weights of previous tasks, leading to forgetting. Therefore, we adopt a simpler ridge regression, which has a closed-form solution.

中文翻译：  
在通过梯度下降得到编码器后，后续持续学习阶段将不再使用梯度下降训练模型。这是因为梯度更新不可避免地会干扰承载旧任务知识的权重，从而造成遗忘。因此，作者改用更简单的岭回归，因为它具有闭式解。

> Although freezing the backbone resolves the stability issue, it affects the model’s plasticity. In order to increase the plasticity, according to Cover’s Theorem, non-linearly mapping the features to a high-dimensional space can increase the probability of the features being linearly separable, we adopt a simple high-dimensional mapping method. The features extracted from the encoder are passed through a Randomly-initialized Hidden Layer (RHL) followed by a non-linear activation function (ReLU). The RHL is a linear layer whose weights are initialized from a normal distribution.

中文翻译：  
虽然冻结骨干网络解决了稳定性问题，但同时也会影响模型可塑性。为了提高可塑性，作者借助 Cover 定理的直觉，即把特征非线性映射到更高维空间之后，它们更有可能变得线性可分。因此，本文采用了一种简单的高维映射：将编码器提取的特征送入一个随机初始化隐藏层 RHL，再接上 ReLU 非线性激活。RHL 本质上是一个线性层，其权重由正态分布随机初始化。

对应公式为：

$$
E_1 = \mathrm{ReLU}(X_1^{encoder}\Phi_E), \tag{1}
$$

$$
\arg\min_{\Phi_1} \left( \|Y_1^{train} - E_1\Phi_1\|_F^2 + \gamma \|\Phi_1\|_F^2 \right), \tag{2}
$$

$$
\hat \Phi_1 = (E_1^\top E_1 + \gamma I)^{-1} E_1^\top Y_1^{train}. \tag{3}
$$

中文理解：  
这里已经能看出整篇方法的骨架：**先把大网络的表示部分固定下来，再把最后的分割头改写成一个解析可求的线性问题。**

---

### 4.2 Recursive Ridge Regression for CSS

![Figure 2](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/fig2_cfss.png)

**图 2 原理解析**  
图 2 是整篇论文最关键的图。它把 CFSSeg 的四个核心部件拼到了一起：  

1. **冻结 encoder**：上一阶段和当前阶段的数据都经过同一个编码器提取特征，这一设计负责稳定性。  
2. **RHL 高维映射**：提取到的特征再经过随机隐藏层映射到更高维空间，用来弥补冻结表示带来的可塑性不足。  
3. **Pseudo Labeling**：上一阶段模型在当前数据上推理，对那些真实标签里被写成背景、但很可能属于旧类的区域生成伪标签，并与当前真值合并为 mixed labels。  
4. **C-RLS 更新**：利用上一阶段分类头 $\hat\Phi_{t-1}$、记忆矩阵 $\Psi_{t-1}$、当前特征 $E_t$ 和 mixed labels，直接通过拼接式递归最小二乘更新得到新阶段的 $\hat\Phi_t$ 和 $\Psi_t$。  

**图 2 的真正价值不只是“流程示意”，而是把作者的工程哲学完整展现出来：**
- 旧知识的记忆不靠样本库，而靠二阶统计矩阵 $\Psi$；  
- 新知识的吸收不靠多轮反向传播，而靠一次解析更新；  
- 分割特有的 semantic drift 不靠额外回放，而靠伪标签修补标签语义。

**结论**：如果你只记住 CFSSeg 一张图，那就应该是图 2。因为它同时展示了数学更新、分割特有问题、以及 2D/3D 可共享的统一方法骨架。

> The previous subsection introduced ridge regression learning, which, however, is not suitable for continual learning. Next, we will propose the concatenated recursive least squares (C-RLS) algorithm. Without loss of generality, let $Y_{1:t-1}^{train}$, $Y_{1:t}^{train}$ and $E_{1:t-1}$, $E_{1:t}$ be the accumulated label and feature matrices in step $t-1$ and $t$, and they are related via block concatenation.

中文翻译：  
前一小节介绍了岭回归，但它本身还不能直接用于持续学习。接下来作者提出拼接式递归最小二乘算法 C-RLS。在不失一般性的情况下，记第 $t-1$ 步与第 $t$ 步累计的标签矩阵和特征矩阵分别为 $Y_{1:t-1}^{train}$、$Y_{1:t}^{train}$ 以及 $E_{1:t-1}$、$E_{1:t}$，它们之间满足块拼接关系。

$$
Y_{1:t}^{train}=
\begin{bmatrix}
Y_{1:t-1}^{train} & 0 \\
\bar Y_t^{train} & \tilde Y_t^{train}
\end{bmatrix},
\qquad
E_{1:t}=
\begin{bmatrix}
E_{1:t-1} \\
E_t
\end{bmatrix}. \tag{4}
$$

$$
Y_t^{train} = [\bar Y_t^{train}\;\tilde Y_t^{train}] . \tag{5}
$$

> The learning problem can then be formulated as the regularized least-squares objective on the accumulated data. According to Eqn (3), at step $t-1$, we have the closed-form solution and define the inverted auto-correlation matrix $\Psi_{t-1}$, which captures the correlation information from both current and past samples. Building on this, the goal is to compute $\hat\Phi_t$ using only $\hat\Phi_{t-1}$, $\Psi_{t-1}$, and the current step’s data, without involving historical samples.

中文翻译：  
于是，学习问题可以写成累计数据上的正则化最小二乘目标。根据公式（3），在第 $t-1$ 步可以得到闭式解，并进一步定义逆自相关矩阵 $\Psi_{t-1}$，它编码了当前样本与历史样本的相关信息。在此基础上，作者的目标变成：只使用 $\hat\Phi_{t-1}$、$\Psi_{t-1}$ 和当前步数据，就计算出新的 $\hat\Phi_t$，而不再显式访问历史样本。

$$
\arg\min_{\Phi_{t-1}} \left( \|Y_{1:t-1}^{train} - E_{1:t-1}\Phi_{t-1}\|_F^2 + \gamma\|\Phi_{t-1}\|_F^2 \right), \tag{6}
$$

$$
\hat\Phi_{t-1} = (E_{1:t-1}^\top E_{1:t-1} + \gamma I)^{-1} E_{1:t-1}^\top Y_{1:t-1}^{train}, \tag{7}
$$

$$
\Psi_{t-1} = (E_{1:t-1}^\top E_{1:t-1} + \gamma I)^{-1}. \tag{8}
$$

> Theorem 1. The $\Phi_t$ weights, recursively obtained by  
> $\hat\Phi_t = [\hat\Phi_{t-1} - \Psi_t E_t^\top E_t\hat\Phi_{t-1} + \Psi_t E_t^\top \bar Y_t^{train},\; \Psi_t E_t^\top \tilde Y_t^{train}]$  
> are equivalent to those obtained from Eqn (7) for step $t$. The matrix $\Psi_t$ can also be recursively updated by  
> $\Psi_t = (\Psi_{t-1}^{-1} + E_t^\top E_t)^{-1}$.

中文翻译：  
定理 1 表明，按下述递归方式得到的 $\Phi_t$ 权重，与第 $t$ 步直接用公式（7）在累计数据上整体求解得到的结果是等价的：

$$
\hat\Phi_t=
\big[
\hat\Phi_{t-1}-\Psi_t E_t^\top E_t\hat\Phi_{t-1}+\Psi_t E_t^\top \bar Y_t^{train},
\; \Psi_t E_t^\top \tilde Y_t^{train}
\big], \tag{9}
$$

且矩阵 $\Psi_t$ 的递归更新式为

$$
\Psi_t=(\Psi_{t-1}^{-1}+E_t^\top E_t)^{-1}. \tag{10}
$$

中文理解：  
这一定理的真正含义是：**只要保存上一阶段的二阶统计量和分类头，就能在当前阶段恢复出与“把所有旧数据和新数据一起重做一次联合求解”相同的结果。** 这就是本文闭式解 continual learning 的理论核心。

---

### 4.3 Theoretical Analysis

> Privacy Protection. Our method ensures data privacy in two ways: first, by eliminating the need to store historical data samples; second, by guaranteeing that historical raw data samples cannot be recovered from the $\Psi$ matrix through reverse engineering.

中文翻译：  
隐私保护方面，本文方法通过两点实现数据隐私：第一，不需要保存历史样本；第二，历史原始数据无法从矩阵 $\Psi$ 中通过逆向工程恢复出来。

> Computational Complexity. The computational complexity analysis reveals that the time complexity for each step includes $O(d_E^3)$ for updating $\Psi_t$ via matrix inversion, and $O(d_E^2N_t + d_EN_t^2 + d_E^2C_t)$ for updating $\Phi_t$ via matrix multiplication. These operations can be efficiently parallelized on GPU.

中文翻译：  
在计算复杂度上，每个增量步骤更新 $\Psi_t$ 需要 $O(d_E^3)$ 的矩阵求逆开销，而更新 $\Phi_t$ 的矩阵乘法开销为 $O(d_E^2N_t + d_EN_t^2 + d_E^2C_t)$。这些操作都可以在 GPU 上高效并行。

> Space Complexity. The space complexity is $O(d_E^2 + d_EN_t + d_EC_t)$: $O(d_E^2)$ is for storing the $\Psi_t$ matrix, $O(d_EN_t)$ for storing the feature matrix $E_t$, and $O(d_EC_t)$ for storing the classifier matrix $\Phi_t$.

中文翻译：  
空间复杂度为 $O(d_E^2 + d_EN_t + d_EC_t)$。其中 $O(d_E^2)$ 用于保存矩阵 $\Psi_t$，$O(d_EN_t)$ 用于保存特征矩阵 $E_t$，$O(d_EC_t)$ 用于保存分类器矩阵 $\Phi_t$。

---

### 4.4 Pseudo-Labeling for 2D Images

> At step $t$, $(x_t, y_t) \in \mathcal{T}_t$, where $x_t^i, y_t^i$ represent the elements and their corresponding ground truth labels, respectively. In both disjoint and overlapped settings, previously learned classes are treated as background in the current task, a phenomenon commonly referred to as semantic drift. To address this issue, we adopt a pseudo-labeling approach. We define the uncertainty of an element as follows:

中文翻译：  
在第 $t$ 步中，给定样本 $(x_t, y_t) \in \mathcal{T}_t$，其中 $x_t^i$ 与 $y_t^i$ 分别表示第 $i$ 个元素及其真实标签。在 disjoint 与 overlapped 设置中，历史学过的类别会在当前任务里被当成背景，这就是常说的语义漂移。为了解决这个问题，作者采用伪标签策略，并将单个元素的不确定性定义为：

$$
U_i = 1 - \sigma\big(\max_c q_{\theta_{t-1}}(i,c)\big). \tag{11}
$$

> The pseudo-labeling strategy is then defined as follows: if $y_t^i \in S_t$, keep the ground truth; if $y_t^i = c_b$ and $U_i > \tau$, keep it as background; if $y_t^i = c_b$ and $U_i \le \tau$, replace it with the pseudo label $\hat y_{t-1}^i$.

中文翻译：  
伪标签规则进一步定义为：如果当前真值属于新类别，则保留真值；如果当前真值是背景但不确定性较高，则继续保留背景；如果当前真值是背景且不确定性较低，则用上一阶段模型给出的伪标签进行替换。公式写作：

$$
\tilde y_t^i=
\begin{cases}
 y_t^i, & y_t^i \in S_t, \\
 y_t^i, & (y_t^i = c_b) \land (U_i > \tau), \\
 \hat y_{t-1}^i, & (y_t^i = c_b) \land (U_i \le \tau).
\end{cases} \tag{12}
$$

中文理解：  
2D 场景里，这个伪标签模块的作用非常直接：**如果背景位置其实更像旧类，而且旧模型很确定，就别让它继续被错误地学成背景。**

---

### 4.5 Pseudo-Labeling for 3D Point Clouds

> For point cloud $i$, we employ the KNN algorithm to identify its $K$ nearest neighbors based on the $xyz$ coordinates and compute the cosine similarity $w_k$ between point cloud $i$ and its $K$ neighbors using their $xyz$ coordinates. We adopt a neighborhood spatial aggregation method based on Monte Carlo dropout (MC-dropout) technique, which achieves efficient estimation of point distribution uncertainty through a single forward propagation. This approach utilizes a spatial dependency sampling mechanism, and its effectiveness has been validated in the literature. For uncertainty quantification, we employ the Bayesian Active Learning Disagreement (BALD) criterion as the core evaluation function for point cloud spatial sampling.

中文翻译：  
对于点云中的点 $i$，作者使用 KNN 算法根据 $xyz$ 坐标找到其 $K$ 个最近邻，并根据坐标计算该点与邻点之间的余弦相似度 $w_k$。作者进一步采用基于 Monte Carlo dropout 的邻域空间聚合方法，以较低代价估计点分布不确定性；该过程利用了点云的局部空间依赖结构。对于不确定性量化，作者使用 BALD 准则作为点云空间采样的不确定性评价函数。

对应的不确定性写为：

$$
U_i = -\sum_c \left[\frac{1}{K}\sum_k q_{t-1}(i,c)w_k\right]\log\left[\frac{1}{K}\sum_k q_{t-1}(i,c)w_k\right]
+ \frac{1}{K}\sum_{c,k}(q_{t-1}(i,c)w_k)\log(q_{t-1}(i,c)w_k). \tag{13}
$$

> The pseudo-labeling method is defined as: keep $y_t^i$ if it belongs to the current new classes; if the current label is background but the previous model predicts a non-background old class with low uncertainty, adopt $\hat y_{t-1}^i$; otherwise, look for a reliable nearest-neighbor pseudo label; if none can be found, assign background.

中文翻译：  
3D 场景下的伪标签规则是：如果当前真值属于新类，就保留真值；如果当前真值是背景，但上一阶段模型把它预测成某个非背景旧类，且该预测不确定性较低，则采用这个旧类预测；否则继续查找一个可靠邻点的伪标签；如果仍找不到，就把该点保留为背景。公式可写为：

$$
\tilde y_t^i=
\begin{cases}
 y_t^i, & y_t^i \in S_t, \\
 \hat y_{t-1}^i, & (y_t^i = c_b) \land (\hat y_{t-1}^i \neq c_b) \land (U_i \le \tau), \\
 \hat y_{t-1}^{i,k'}, & (y_t^i = c_b) \land \big((\hat y_{t-1}^i = c_b) \lor (U_i > \tau)\big), \\
 c_b, & \text{otherwise}.
\end{cases} \tag{14}
$$

中文理解：  
3D 版本比 2D 更复杂，因为点云本身有强邻域结构。作者不只看当前点自己的预测，还会借助附近点的可靠伪标签去修补，从而减少孤立噪声点把标签带偏。

---

## 5. Experiments

### 5.1 Experimental Setup

> 2D Dataset. We evaluate our method using public 2D semantic segmentation benchmarks: Pascal VOC2012. It contains 21 classes (including background class). This dataset features wild scenes, with 10,582 images used for training and 1,449 images for validation.

中文翻译：  
2D 数据集方面，作者在公开基准 Pascal VOC2012 上评估方法。该数据集包含 21 个类别，包括背景类，训练集 10,582 张图像，验证集 1,449 张图像。

> 3D Dataset. We evaluate our method using two public 3D point cloud segmentation benchmarks: S3DIS and ScanNet. These datasets are selected for their diversity, relevance to our problem domain, and ability to facilitate fair comparisons with existing benchmark methods. S3DIS comprises point clouds from 272 rooms across 6 indoor areas, with each point containing xyz coordinates and RGB information, manually annotated with one of 13 predefined classes. Following standard practice, we designate the more challenging Area 5 as the validation set, while the remaining areas are used for training. ScanNet, on the other hand, is an RGB-D video dataset featuring 1,513 scans from 707 indoor scenes. Each point is labeled with one of 21 classes, including 20 semantic classes and an additional category for unannotated places. Adhering to the standard dataset splits, we allocate 1,210 scans for training and 312 scans for validation. We adopt a sliding window to partition the rooms in the S3DIS and ScanNet datasets, generating 7,547 and 36,350 1m×1m blocks respectively, and randomly sample 2,048 points from each block as input data. We use two sequences, $S^0$ and $S^1$, to partition the 3D dataset. $S^0$ follows the original dataset’s annotation order, while $S^1$ follows the alphabetical order of the category names.

中文翻译：  
3D 数据集方面，作者使用 S3DIS 与 ScanNet 两个公开点云分割基准。S3DIS 包含来自 6 个室内区域、272 个房间的点云，每个点带有 xyz 坐标与 RGB 信息，并被标注为 13 个预定义类别之一。按惯例，作者把更困难的 Area 5 作为验证集，其余区域用于训练。ScanNet 是一个 RGB-D 视频数据集，包含来自 707 个室内场景的 1,513 个扫描，每个点被标注为 21 类之一，其中包括 20 个语义类和 1 个未标注区域类。按照标准划分，作者使用 1,210 个扫描训练，312 个扫描验证。作者采用滑窗把 S3DIS 与 ScanNet 切分为 1m×1m 的局部块，并从每个块中随机采样 2,048 个点作为输入。同时，作者使用两种类别顺序 $S^0$ 与 $S^1$ 对 3D 数据进行划分，前者沿用原始标注顺序，后者按类别名的字母顺序排列。

> CSS Learning Protocol. The classes of the images for the current step include $\mathcal{C}_{t-1} \cup S_t$. In each step, we continuously introduce new classes for learning. In an $m$-$n$ setting, the model first learns $m$ classes, and in each subsequent step, it incrementally learns $n$ classes. For 2D CSS, we adopt the three settings: sequential, disjoint, and overlapped. For 3D CSS, we follow prior work and use the disjoint setting.

中文翻译：  
CSS 学习协议规定，当前步图像中的类别集合为 $\mathcal{C}_{t-1} \cup S_t$。模型首先学习一个初始类集合，然后在后续每一步继续引入新类进行增量学习。在 $m$-$n$ 设置下，模型先学习 $m$ 个类别，之后每步再学习 $n$ 个新类别。2D 实验采用 sequential、disjoint 和 overlapped 三种设置，3D 实验则沿用先前工作使用 disjoint 设置。

> Evaluation Metrics. We use the widely adopted mean Intersection-over-Union (mIoU) metric to calculate the average IoU value across all classes. To comprehensively evaluate CSS performance, we compute mIoU values separately for initial classes, incremental classes, and all classes.

中文翻译：  
评测指标使用 mIoU。为了更全面衡量 CSS 性能，作者分别统计初始类、增量类以及全部类别上的 mIoU。

> Comparison Methods. Our method is an exemplar-free method. For fairness, we also compare it with other exemplar-free methods. For 2D CSS, see Table 3 and Table 4 for details. As for 3D CSS, due to the limited number of relevant baselines, we consider Yang et al.’s method as a strong baseline, along with EWC. At the same time, we establish a naive baseline: FT, which fine-tunes both the backbone and the classification head. In addition, we include an upper bound, namely JT, which stands for joint training.

中文翻译：  
由于本文属于 exemplar-free 方法，因此作者主要与其他 exemplar-free 方法进行公平比较。2D CSS 的对比详见表 3 与表 4。对于 3D CSS，考虑到现有基线较少，作者选用 Yang 等人的方法以及经典基线 EWC，同时加入朴素微调基线 FT，并把 JT 联合训练作为上界参考。

> Implementation Details. For the initial training in step 1, we adopt DeepLabv3 with a ResNet-101 backbone pre-trained on ImageNet-1K. We set the number of epochs to 50 and the batch size to 32. We use SGD as the optimizer with a learning rate of $10^{-2}$, a momentum of 0.9, and a weight decay of $10^{-4}$, combined with a polynomial learning rate scheduler. The loss function is binary cross-entropy (BCE). For the 3D CSS encoder, we employ DGCNN with a batch size of 32 and the Adam optimizer, using an initial learning rate of 0.001 and a weight decay of 0.0001 for 100 epochs. In the continual learning step, we freeze the encoder and insert an RHL layer. In the 2D experiments, we set $d_E$ to 8192, $\gamma$ to 1, and $\tau$ to 0.4. In the 3D experiments, we set $d_E$ to 5000, $\gamma$ to 1, and $\tau$ to 0.0035 and 0.001 on the S3DIS and ScanNet datasets via cross-validation respectively.

中文翻译：  
实现细节方面，2D 初始训练使用带有 ImageNet-1K 预训练 ResNet-101 骨干的 DeepLabv3，训练 50 个 epoch，batch size 为 32，优化器为 SGD，学习率 $10^{-2}$，动量 0.9，权重衰减 $10^{-4}$，损失为 BCE。3D 编码器使用 DGCNN，batch size 同样为 32，优化器为 Adam，初始学习率 0.001，权重衰减 0.0001，总共训练 100 个 epoch。在持续学习阶段，作者冻结编码器并插入 RHL。2D 实验中设置 $d_E=8192$、$\gamma=1$、$\tau=0.4$；3D 实验中设置 $d_E=5000$、$\gamma=1$，并通过交叉验证把 $\tau$ 分别设为 S3DIS 上的 0.0035 和 ScanNet 上的 0.001。

---

## 5.2 Main Results

### Table 1 — 3D, $S^0$ split

![Table 1](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table1_3d_s0.png)

**表 1 解析**  
这张表展示了 3D 增量分割在 $S^0$ 类别顺序下的结果。S3DIS 与 ScanNet 两个数据集都表明，普通微调 FT 会严重遗忘，而 CFSSeg 在多种协议下都明显更稳。  

**关键观察：**
- 在 **S3DIS** 上，CFSSeg 的 overall mIoU 分别达到 **41.66 / 42.67 / 44.08**。  
- 在 **ScanNet** 上，CFSSeg 的 overall mIoU 达到 **26.97 / 27.98 / 27.96**，显著高于 FT、EWC 和 Yang 等方法。  
- 这些提升说明 closed-form 递归更新不只适用于分类或 2D 图像，也能迁移到 3D 点云分割。  

**结论**：表 1 证明了 CFSSeg 在 3D 分割上是真正有效的，而不是只在 2D 数据上成立的技巧。

### Table 2 — 3D, $S^1$ split

![Table 2](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table2_3d_s1.png)

**表 2 解析**  
表 2 把 3D 数据集的类别顺序改成 $S^1$，用来测试类顺序鲁棒性。结果显示：  
- 在 **S3DIS** 上，CFSSeg 的 overall mIoU 达到 **43.40 / 43.04 / 44.25**。  
- 在 **ScanNet** 上，对应结果为 **25.18 / 26.42 / 27.84**。  

**这张表支持的结论是：** CFSSeg 并不是偶然适配某一类顺序，至少在 3D 场景中，它对类顺序变化有相当稳定的表现。

### Table 3 — 2D, Sequential setting

![Table 3](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table3_2d_seq.png)

**表 3 解析**  
这张表对应 Pascal VOC2012 上最“理想”的 sequential 设置。CFSSeg 在 15-1 和 15-5 两种协议下都得到：  
- old classes mIoU：**78.1**  
- new classes mIoU：**42.0**  
- all classes mIoU：**70.0**  

**最值得注意的点：** 15-1 和 15-5 的结果完全一致。对于普通梯度法，这往往不成立；但对闭式递归解来说，如果最终看到的数据集合相同，那么不同的分步方式可能给出同样的最优解。这恰好是本文理论最有力的经验支持之一。

**结论**：表 3 不只是“性能表”，它也是闭式解等价性的一个经验侧证。

### Table 4 — 2D, Disjoint / Overlapped settings

![Table 4](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table4_2d_dis_ov.png)

**表 4 解析**  
这是全文最关键的主结果表之一，因为它覆盖了最困难的 disjoint 与 overlapped 两种设置。  

在 **disjoint** 下，CFSSeg 达到：  
- 15-1：**77.66 / 40.33 / 68.77**  
- 10-1：**70.85 / 42.13 / 57.17**  

在 **overlapped** 下，CFSSeg 达到：  
- 15-1：**79.16 / 38.00 / 69.36**  
- 10-1：**75.02 / 41.20 / 58.91**  

尤其在 **disjoint 10-1** 下，CFSSeg 的 overall mIoU 为 **57.17**，而次强方法只有 **18.20**。这说明作者的方法并不是略优，而是在语义漂移特别严重的 hardest setting 中实现了非常显著的领先。

**结论**：表 4 强力支持了论文最核心的工程主张：**closed-form head 负责不遗忘，pseudo-labeling 负责修复 semantic drift，两者组合才使方法在困难增量分割设置中真正站得住。**

> 2D Experimental Results. Extensive experiments on the Pascal VOC2012 dataset demonstrate the outstanding performance of our method across all evaluation settings. Under the sequential 15-1 configuration, our approach achieves an overall mIoU of 70.0%, significantly surpassing current state-of-the-art methods. The method maintains strong performance on base classes (78.1% mIoU) while retaining high accuracy for novel classes (42.0% mIoU), effectively addressing the inherent stability-plasticity dilemma in continual learning. Notably, identical performance metrics in both the 15-1 and 15-5 sequential settings confirm the mathematical consistency of our closed-form solution. The method’s advantages become even more pronounced in challenging scenarios: under the disjoint 15-1 setting, it achieves 68.77% mIoU across all classes (77.66% for base classes, 40.33% for novel classes), while in the disjoint 10-1 setting, the performance gap widens dramatically. Similarly, in overlapped scenarios, our method maintains its superiority with 69.36% mIoU in the 15-1 configuration and 58.91% in the 10-1 setting, outperforming existing approaches across all class categories. These compelling results validate the theoretical advantages of the closed-form solution in mitigating catastrophic forgetting while efficiently integrating new knowledge.

中文翻译：  
2D 实验结果表明，本文方法在 Pascal VOC2012 的所有评估设置下都表现突出。在 sequential 15-1 设置中，本文方法的 overall mIoU 达到 70.0%，显著超过现有 SOTA。模型在旧类上保持了很强性能，达到 78.1% mIoU，同时在新类上也保留了 42.0% mIoU，有效缓解了持续学习中的稳定性—可塑性矛盾。特别值得注意的是，在 sequential 15-1 和 15-5 下获得完全一致的性能指标，这验证了闭式解在数学上的一致性。在更具挑战性的场景中，这种优势更加明显：在 disjoint 15-1 下，方法在全类别上的 mIoU 达到 68.77%，其中旧类为 77.66%，新类为 40.33%；在 disjoint 10-1 下，与最佳竞争者的性能差距进一步显著扩大。类似地，在 overlapped 场景中，本文方法在 15-1 和 10-1 配置下分别达到 69.36% 与 58.91% 的 overall mIoU，并在所有类别维度上都超过已有方法。这些结果有力验证了闭式解在减轻灾难性遗忘、同时高效整合新知识方面的理论优势。

> 3D Experimental Results. Our method demonstrates exceptional performance in 3D point cloud segmentation tasks across multiple benchmark datasets. On the S3DIS dataset with the $S^0$ split, it outperforms existing approaches in all evaluation protocols: the 8-1 configuration achieves an overall mIoU of 41.66% (49.77% for base classes, 28.69% for novel classes), while the 10-1 and 12-1 configurations reach 42.67% and 44.08% mIoU, respectively. With the $S^1$ split, the 8-1 configuration improves further to 43.40% mIoU (51.33% for base classes, 30.72% for novel classes), setting a new state-of-the-art benchmark for 3D continual semantic segmentation. The method also proves effective on the more challenging ScanNet dataset: under the $S^0$ split, the 15-1 configuration achieves 26.97% mIoU, significantly outperforming previous methods that struggled to exceed 10% mIoU. The 17-1 and 19-1 configurations yield overall mIoU scores of 27.98% and 27.96%, respectively. On the $S^1$ split, the three configurations achieve 25.18%, 26.42%, and 27.84% mIoU. The consistent performance across diverse 3D configurations underscores the versatility and robustness of our closed-form solution in the point cloud domain, providing a theoretically grounded and practical solution for real-world 2D and 3D semantic segmentation applications.

中文翻译：  
3D 实验结果说明，本文方法在多个点云分割基准上表现出很强竞争力。在 S3DIS 数据集的 $S^0$ 划分下，方法在全部评估协议中都优于现有方案：8-1 配置下 overall mIoU 为 41.66%，其中旧类 49.77%，新类 28.69%；10-1 与 12-1 配置下分别达到 42.67% 和 44.08%。在 $S^1$ 划分下，8-1 配置进一步提升至 43.40%，其中旧类 51.33%，新类 30.72%，建立了新的 3D 持续语义分割基准。该方法在更具挑战性的 ScanNet 上同样有效：在 $S^0$ 划分下，15-1 配置达到 26.97% mIoU，显著超过此前许多很难突破 10% mIoU 的方法；17-1 与 19-1 分别达到 27.98% 和 27.96%。在 $S^1$ 划分下，三种配置分别达到 25.18%、26.42% 和 27.84%。不同 3D 配置下的一致表现，说明本文闭式解在点云域中具有良好的通用性与鲁棒性，为真实世界 2D 和 3D 语义分割应用提供了兼具理论基础和实践可行性的方案。

---

## 5.3 Class Order Robustness

> Our method demonstrates strong robustness to class order variations, which is a critical aspect in continual learning scenarios. This robustness stems from two key factors: First, the closed-form solution ensures deterministic and unique classification head weights for a given set of training data. As evidenced in Table 3, our method achieves identical performance in both the sequential 15-1 and 15-5 settings. This consistency is a direct consequence of the closed-form nature of our solution, which guarantees the same optimal weights regardless of the training sequence. Second, for 3D datasets, we observe that performance variations across different class orders are primarily influenced by the backbone network’s feature extraction capabilities. When using the same backbone architecture, the performance remains remarkably stable across different class sequences. Minor variations in performance can be attributed to the backbone’s sensitivity to different class orders during the initial training phase, rather than the continual learning mechanism itself.

中文翻译：  
本文方法对类别顺序变化表现出较强鲁棒性，而这在持续学习中是一个非常关键的性质。作者认为这种鲁棒性来自两个因素。第一，闭式解能够在给定训练数据集合的条件下得到确定且唯一的分类头权重。正如表 3 所示，本文方法在 sequential 15-1 和 15-5 设置下获得完全相同的结果。这种一致性直接源于闭式解的性质，也就是无论训练序列如何变化，只要数据集合相同，最终最优权重就相同。第二，在 3D 数据集上，不同类别顺序带来的性能差异主要来自骨干网络的特征提取能力。当骨干结构保持相同时，不同类顺序下的性能总体上仍然相当稳定。少量波动更多是初始训练阶段骨干对类顺序敏感所造成的，而不是持续学习机制本身的缺陷。

中文点评：  
这一段非常重要，因为它把“顺序鲁棒性”拆成了两个层次。对于解析分类头，顺序不敏感更多来自**数学唯一性**；对于 3D 数据集上的轻微差异，则更多来自**初始表示学习**而不是后续闭式更新。

---

## 5.4 Ablation Studies and Efficiency Studies

### Table 5 — Ablation on RHL and Pseudo-Labeling

![Table 5](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table5_ablation.png)

**表 5 解析**  
表 5 研究了两个关键模块：RHL 和 pseudo-labeling。  

- 去掉 **RHL** 后，新类 mIoU 从 **41.20** 直接跌到 **9.36**，overall 从 **58.91** 跌到 **37.94**。这说明冻结 encoder 之后，如果不做高维映射，模型几乎失去吸收新类的能力。  
- 去掉 **pseudo-labeling** 后，旧类从 **75.02** 降到 **71.83**，新类从 **41.20** 降到 **36.19**，overall 降到 **54.86**。这说明 semantic drift 的确是分割里必须专门修补的问题。  

**结论**：RHL 负责可塑性，pseudo-labeling 负责标签语义修正。两者都不是可有可无的“附加 trick”，而是本文成立的必要组件。

> To rigorously evaluate the contributions of each component in our framework, we conducted comprehensive ablation experiments using the challenging Pascal VOC2012 overlapped 10-1 setting. The results presented in Table 5 reveal several key insights into the effectiveness of our approach. Effect of RHL. Removing the RHL component led to a significant performance drop, particularly on new classes. This underscores the critical role of RHL in enhancing the model’s plasticity while maintaining stability. Effect of Pseudo-Labeling. The pseudo-labeling mechanism plays a vital role in preventing semantic drift, helping the model retain accurate recognition capabilities for previously learned classes.

中文翻译：  
为了严格评估各个组件的贡献，作者在 Pascal VOC2012 overlapped 10-1 这一高难度设置上进行了全面消融。表 5 的结果揭示了方法有效性的几个关键点。首先，去掉 RHL 后，性能显著下降，尤其是新类表现出现大幅退化，这说明 RHL 对于在保持稳定性的同时提升模型可塑性至关重要。其次，伪标签机制在防止 semantic drift 方面发挥了关键作用，它帮助模型继续准确识别已学过的类别。

### Table 6 — Efficiency

![Table 6](20_AI存档库/早期GPT%20pro-CFSSeg_精读笔记/assets/table6_efficiency.png)

**表 6 解析**  
这张表是闭式解路线非常有说服力的一张工程表。与需要 10 个 epoch 的 FT 相比：  
- CFSSeg 只用 **1 个 epoch**，总时间 **43.25s**，而 FT 需要 **651.46s**；  
- 显存占用更低，**51.61 GB vs. 59.55 GB**；  
- 还支持更大 batch size，**64 vs. 32**。  

**结论**：本文不是单纯在说“闭式解有理论好处”，而是在明确展示：**它更快、更省显存，而且结果还更好。** 这也是 CFSSeg 很像一个“可部署路线”而不仅是学术概念验证的原因。

> As quantified in Table 6, our closed-form solution achieves convergence in just a single training epoch, taking only 43.25 seconds, whereas fine-tuning methods requiring multiple gradient descent iterations need 651.46 seconds. This represents a 15× acceleration in training time while maintaining superior segmentation performance. Furthermore, despite supporting larger batch sizes (64 vs. 32), our method significantly reduces memory consumption (51.61 GB vs. 59.55 GB) due to the closed-form nature of our solution.

中文翻译：  
正如表 6 所示，本文的闭式解方案只需要一个训练 epoch 即可收敛，总共只需 43.25 秒；相比之下，需要多轮梯度迭代的微调方法需要 651.46 秒。这意味着本文方法在训练时间上实现了约 15 倍加速，同时还能保持更优分割性能。此外，尽管本文支持更大的 batch size（64 对 32），其显存消耗仍然更低（51.61 GB 对 59.55 GB），这正是闭式解方法带来的直接工程收益。

---

## 6. Conclusion

> We presented CFSSeg, a novel method for class-incremental semantic segmentation (CSS) designed for both 2D images and 3D point clouds. CFSSeg distinguishes itself through a gradient-free, closed-form update mechanism, computed recursively for efficiency. This core component, combined with a frozen encoder for stability, high-dimensional feature mapping for plasticity, and a tailored pseudo-labeling strategy for semantic drift, allows the model to learn new classes incrementally without catastrophic forgetting or reliance on stored exemplars. Consequently, CFSSeg operates with significantly reduced computational cost—requiring only a single training pass per step—and enhanced data privacy. Our extensive evaluations on Pascal VOC2012, S3DIS, and ScanNet show that CFSSeg achieves outstanding results, outperforming prior methods and providing a robust, efficient, and effective solution for continual semantic segmentation tasks, forming a complete closed loop from theoretical foundation to practical implementation.

中文翻译：  
本文提出了 CFSSeg，这是一种同时面向 2D 图像和 3D 点云的类增量语义分割方法。CFSSeg 的核心特征在于一种无梯度、递归计算的闭式更新机制。这个核心模块与冻结编码器所提供的稳定性、高维特征映射带来的可塑性，以及用于修复 semantic drift 的伪标签策略结合后，使模型能够在不依赖样本回放且不发生严重灾难性遗忘的情况下持续学习新类别。因此，CFSSeg 具有显著降低的计算成本，每一步只需一次训练遍历，同时还具有更好的数据隐私属性。作者在 Pascal VOC2012、S3DIS 和 ScanNet 上的大量实验表明，CFSSeg 不仅优于现有方法，而且为持续语义分割提供了一个兼具鲁棒性、效率和有效性的完整方案，形成了从理论基础到实际实现的闭环。

---

## 全文精读后的总总结

### 一句话抓核心

**CFSSeg 的真正创新不是“把 ridge regression 套到 segmentation 上”，而是把“冻结表示 + 高维 buffer + 递归闭式头 + 伪标签修复 semantic drift”整合成了一个在 2D 图像与 3D 点云上都能工作的持续分割方案。**

### 你应该带走的 5 个结论

1. **作者把旧知识记忆从“样本”转移成了“二阶统计矩阵”**，这是它无回放、隐私友好的根本。  
2. **冻结 encoder 是稳定性的来源**，但必须用 RHL 补回可塑性。  
3. **分割里的难点不只是 forgetting，而是 semantic drift**，所以 pseudo-labeling 不只是辅助技巧，而是必要模块。  
4. **表 3 中 15-1 与 15-5 结果完全一致**，是闭式解数学一致性的非常强的经验信号。  
5. **表 6 说明它有真实工程价值**：更快、更省显存，而且不依赖历史样本。

---
