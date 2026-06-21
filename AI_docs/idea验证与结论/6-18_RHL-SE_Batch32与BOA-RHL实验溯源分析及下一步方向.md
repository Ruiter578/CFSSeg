# RHL-SE batch32 与 BOA-RHL 实验溯源分析及下一步方向

> 项目：`/root/2TStorage/lyc/SegACIL`  
> 日期：2026-06-18  
> 分支：`feature/rhl-se-boa-p0-p1`  
> 主题：P0 RHL-SE val-driven 收尾与 P1 BOA-RHL 第一轮结果分析  
> 主要证据目录：  
> - P0 RHL-SE batch32：`logs/rhl_se_val_driven/20260618_p0_rhl_bs32_se_val_driven/`  
> - P0 RHL-SE batch16 旁证：`logs/rhl_se_val_driven/20260618_p0_rhl_bs16_se_val_driven/`  
> - P1 BOA-RHL：`logs/boa_rhl/20260617_p1_boa_rhl/`  
> - P1 BOA-RHL checkpoints：`checkpoints/20260617_p1_boa_rhl_*/voc/15-5/sequential/step1/`  
> - 参考方案：`AI_docs/idea构思与实验设计/RHL新方案/6-16_未实现三方案攻击审查与重构执行方案.md`

---

## 0. 执行摘要

本轮 P0/P1 的最重要结论不是“哪个方案已经成为强方法”，而是：

```text
RHL-SE 的 val-driven class-wise 收尾是干净、稳定、可复现的弱正向辅助模块；
BOA-RHL 中 orthogonal + legacy 出现了真实但很小的单模型正信号；
antithetic pair 在当前 buffer 不扩容的设定下明显失败；
下一步不应继续盲目堆 seed、扫全局权重或扩大 antithetic，而应收缩为更机制化的 BOA 复核和 PGH-RHL-lite。
```

核心指标如下。数值均为百分制 mIoU。

| 方法/实验 | all | old 0-15 | new 16-20 | 相对 `bs32_8196` all | 相对 `bs32_8196` new | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `20260617_bs32_8196` / BOA-0 | 69.5598 | 77.7942 | 43.2099 | 0.0000 | 0.0000 | 当前 batch32 单模型强基线 |
| RHL-SE seed1 / `20260606` | 69.4606 | 78.0085 | 42.1075 | -0.0992 | -1.1025 | 旧 RHL-SE baseline / seed1 |
| RHL-SE seed2 | 69.4391 | 77.8998 | 42.3649 | -0.1207 | -0.8450 | 新类略强于 seed1，但低于 bs32 |
| RHL-SE seed3 | 69.4989 | 77.9544 | 42.4415 | -0.0609 | -0.7684 | 三个 seed 中单模型最好 |
| P0 RHL-SE class-wise K3 bs32 | 69.5382 | 77.9803 | 42.5236 | -0.0216 | -0.6863 | 相对旧 seed1 有收益；未超过 batch32 强基线 |
| BOA-1 orthogonal/legacy | 69.6387 | 77.8225 | 43.4507 | +0.0789 | +0.2408 | 唯一正向 BOA 信号，但未达预设阈值 |
| BOA-2 antithetic/legacy | 68.3424 | 77.7067 | 38.3768 | -1.2174 | -4.8332 | 明显失败 |
| BOA-3 antithetic/kaiming | 68.1668 | 77.6573 | 37.7973 | -1.3930 | -5.4126 | 更失败，scale 放大没有救回 antithetic |

直接判断：

1. **P0 已经完成收尾**。RHL-SE 的 val-driven class-wise 方案可作为辅助实验、集成系统组件和机制诊断工具，但不应继续包装成主涨点方法。
2. **P1 没有失败，但第一轮只支持 BOA-1**。`orthogonal + legacy` 的 `+0.0789 all / +0.2408 new` 是弱正信号，低于文档中 `+0.10 all` 或 `+0.30 new` 的继续阈值，但非常接近，值得做有边界的复核。
3. **antithetic 当前应该止损**。BOA-2/3 的 new mIoU 分别掉 `4.83` 和 `5.41` 个百分点，说明 `[W, -W]` 在不增加总 buffer 的前提下牺牲独立方向数，伤害远大于符号补偿收益。
4. **batch size 不是 P0 指标差异的原因**。P0 bs16 与 bs32 结果几乎完全一致，说明这里的 `VAL_BATCH_SIZE` 只是评估吞吐参数，不改变结论。

---

## 1. 证据完整性与代码实现复核

### 1.1 P0/P1 代码审查状态

本轮对 P0/P1 已提交变更执行了 CodeRabbit committed review：

```bash
coderabbit review --agent -t committed --base-commit a2cbc4e
```

返回：

```text
review_completed
findings: 0
```

同时，P0/P1 变更范围的常见 secret 关键字扫描无命中。此前本地单测 `tests.test_rhl_buffer` 已覆盖：

- `--rhl_init` / `--rhl_scale_mode` 参数解析；
- orthogonal / antithetic 初始化结构；
- scale mode 行范数行为；
- `rhl_seed` 不污染外部 RNG；
- `training_config` 与 `run_config.json` 持久化。

因此，本报告把当前结果优先解释为方法机制结果，而不是代码错误导致的异常结果。

### 1.2 BOA-0 复现 baseline，说明实现没有破坏默认路径

BOA-0 的命令为：

```text
rhl_init=gaussian
rhl_scale_mode=legacy
batch_size=32
buffer=8196
gamma=1
rhl_norm=none
base_subpath=20260606
```

BOA-0 的测试结果与已有 `checkpoints/20260617_bs32_8196/.../test_results_20260617_121741.json` 完全一致：

```text
all mIoU: 69.5598
old mIoU: 77.7942
new mIoU: 43.2099
```

这点很关键：它证明 `gaussian + legacy` 的默认语义被保留，P1 的新增初始化族没有把 baseline 路径改坏。后续 BOA-1/2/3 的差异可以归因到 RHL 初始化方式和 scale mode，而不是训练脚本漂移。

### 1.3 BOA 四个 case 没有 OOM、Traceback、NaN/Inf

日志检查显示 BOA-0/1/2/3 均无：

```text
out of memory
CUDA out of memory
Traceback
RuntimeError
nan=True
inf=True
```

每个 case 都生成了 `step1/final.pth` 和 `step1/test_results_*.json`。因此 BOA-2/3 的大幅掉点不是运行未完成、显存 fallback 或数值崩溃造成的表面异常，而是方法配置本身导致的性能下降。

---

## 2. P0：RHL-SE batch32 结果分析

### 2.1 结果文件与命令协议

P0 batch32 结果目录：

```text
logs/rhl_se_val_driven/20260618_p0_rhl_bs32_se_val_driven/
```

包含：

```text
class_weights.json
run_summary.md
test_diagnostics.json
test_results.json
val_search.json
```

`run_summary.md` 记录的关键命令参数：

```text
--val_batch_size 32
--max_batches -1
--ensemble_mode prob
--objective all_miou
```

成员 checkpoint 为：

```text
checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth
checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth
checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth
```

协议是干净的：

1. `val_search.json` 在 val split 上选 class-wise 权重；
2. `class_weights.json` 固化 val 选择；
3. `test_results.json` 只消费该权重做最终 test；
4. `test_diagnostics.json` 保存成员、oracle 和 disagreement 诊断。

### 2.2 bs32 与 bs16 基本一致，评估 batch size 不影响结论

| P0 评估 batch | all | old | new | val best candidate |
|---|---:|---:|---:|---|
| bs32 | 69.538244 | 77.980321 | 42.523600 | `classwise_valbest_all_s0.75` |
| bs16 | 69.538633 | 77.981073 | 42.522827 | `classwise_valbest_all_s0.75` |

差异只有 `0.0004` all mIoU 量级，可以视为数值/遍历细节造成的微小差别。由此确认：

```text
VAL_BATCH_SIZE 主要影响速度和显存，不改变 P0 的方法结论。
```

当前还有一个 `VAL_BATCH_SIZE=1` 的 `20260618_p0_rhl_se_val_driven` 进程仍在运行，不作为本报告的结论来源。

### 2.3 P0 相对旧 RHL-SE baseline 是正向，相对 batch32 强基线不是正向

如果以 `20260606` / seed1 作为 RHL-SE 旧 baseline：

| 对比 | all | old | new |
|---|---:|---:|---:|
| seed1 / `20260606` | 69.4606 | 78.0085 | 42.1075 |
| P0 class-wise K3 bs32 | 69.5382 | 77.9803 | 42.5236 |
| 差值 | +0.0777 | -0.0282 | +0.4161 |

这个结论支持旧文档中的定位：

```text
RHL-SE 对新类有稳定弱正向，但 all mIoU 提升不足 +0.10。
```

但如果以当前 batch32/8196 强基线作为对照：

| 对比 | all | old | new |
|---|---:|---:|---:|
| `20260617_bs32_8196` / BOA-0 | 69.5598 | 77.7942 | 43.2099 |
| P0 class-wise K3 bs32 | 69.5382 | 77.9803 | 42.5236 |
| 差值 | -0.0216 | +0.1862 | -0.6863 |

这说明 P0 不能被写成“超过当前最强 batch32 单模型”。它更准确的价值是：

```text
在旧 RHL-SE seed1/2/3 训练产物上，val-driven class-wise 融合能稳定提取一部分新类互补；
但这批成员本身弱于新的 bs32_8196 单模型，因此最终没有超过当前 batch32 强基线。
```

这不是矛盾，而是基线口径不同。报告和论文中必须把两种对照分开。

### 2.4 class-wise 权重确实抓到了“不同新类信不同 seed”

P0 的 val search 最优候选为：

```text
classwise_valbest_all_s0.75
```

在新类上的 val 最优成员与 test class_weights 为：

| 类别 | val best member | test class weight |
|---|---:|---|
| pottedplant 16 | seed3 | `[0.125, 0.125, 0.75]` |
| sheep 17 | seed3 | `[0.125, 0.125, 0.75]` |
| sofa 18 | seed2 | `[0.125, 0.75, 0.125]` |
| train 19 | seed2 | `[0.125, 0.75, 0.125]` |
| tvmonitor 20 | seed3 | `[0.125, 0.125, 0.75]` |

这与早期 RHL-SE 诊断一致：

```text
seed2 更偏 sofa / train；
seed3 更偏 pottedplant / sheep / tvmonitor；
seed1 更像 baseline 稳定器。
```

因此 P0 的机制不是无效的。它的问题是互补幅度太小、成员质量太接近、且 class-wise 静态权重仍然无法达到 per-pixel oracle 上界。

### 2.5 oracle 与 disagreement 揭示 P0 的真正瓶颈

P0 bs32 的 diagnostics：

| 诊断项 | 数值 |
|---|---:|
| pairwise disagreement | 0.8668% |
| old-region disagreement | 0.6179% |
| new-region disagreement | 4.9279% |
| oracle all mIoU | 71.4056 |
| oracle old mIoU | 79.2551 |
| oracle new mIoU | 46.2872 |

这组结果要分两层看：

1. **成员整体非常相似**：全局 pairwise disagreement 不到 1%，说明只改变 RHL seed 并没有产生强模型多样性。
2. **互补不是不存在**：new-region disagreement 接近 5%，oracle new mIoU 比实际 ensemble 高约 `3.76` 个百分点，说明错误模式在新类区域存在可利用差异。

所以 P0 的瓶颈不是“RHL-SE 完全没有信号”，而是：

```text
现有 prob / logit / class-wise 静态权重不能可靠判断每个像素该信哪个成员。
```

继续扫全局权重没有意义；继续补 seed4/5 也不应作为第一动作。真正能吃到 oracle 上界的，需要 per-pixel 级别的可靠性建模，例如 margin、entropy、old/new prior、class-wise calibration。但这会把 RHL-SE 从轻量收尾模块变成一个新的融合方法，工程与过拟合风险都会上升。

### 2.6 P0 结论

P0 的结论应写得克制：

```text
RHL-SE val-driven class-wise 是成立的、可复现的弱正向模块；
它证明 RHL 随机子空间存在类别级互补；
但普通 RHL seed 的多样性不足，当前融合策略无法超过 batch32 强单模型；
因此它不适合作为主创新继续堆实验，应作为辅助集成和诊断工具保留。
```

止损判断：

- 达到旧文档中 `+0.30 new` 条件：相对 seed1，new mIoU `+0.4161`，可以说新类弱正向成立。
- 未达到 `+0.10 all` 条件：all mIoU 只 `+0.0777`，且相对 batch32 强基线为负。

因此 P0 不应继续扩成“更多 seed + 更多权重”的主线。

---

## 3. P1：BOA-RHL 第一轮结果分析

### 3.1 实验矩阵

P1 第一轮执行了 4 个单模型 step1 实验：

| case | `rhl_init` | `rhl_scale_mode` | 目的 |
|---|---|---|---|
| BOA-0 | `gaussian` | `legacy` | 复现 baseline |
| BOA-1 | `orthogonal` | `legacy` | 分离 block orthogonal 采样效果 |
| BOA-2 | `orthogonal_antithetic` | `legacy` | 分离 `[W, -W]` 成对效果 |
| BOA-3 | `orthogonal_antithetic` | `kaiming` | 检查 antithetic 下 scale 放大是否有帮助 |

共同配置：

```text
batch_size=32
buffer=8196
gamma=1
rhl_norm=none
rhl_seed=1
random_seed=1
base_subpath=20260606
train_epoch=50
task=voc 15-5 sequential
```

### 3.2 BOA-1 是唯一正向信号

相对 BOA-0：

| case | all delta | old delta | new delta | new class 主要变化 |
|---|---:|---:|---:|---|
| BOA-1 orthogonal/legacy | +0.0789 | +0.0283 | +0.2408 | sofa +0.6981, tvmonitor +0.4659, train +0.1132 |
| BOA-2 antithetic/legacy | -1.2174 | -0.0875 | -4.8332 | pottedplant -8.0804, tvmonitor -11.2614 |
| BOA-3 antithetic/kaiming | -1.3930 | -0.1368 | -5.4126 | pottedplant -8.1894, tvmonitor -12.5842 |

BOA-1 的提升低于原方案设定的继续阈值：

```text
目标阈值：+0.10 all 或 +0.30 new
实际提升：+0.0789 all / +0.2408 new
```

但它足够接近阈值，并且 old/new 同时不降。这与 P0 的结论不同：P0 是推理融合弱正向，BOA-1 是单模型 RHL 构造弱正向。后者更接近“方法贡献”的位置。

### 3.3 为什么 orthogonal/legacy 可能有效

BOA-0 与 BOA-1 的 RHL stats：

| case | mean | std | sparsity | gram_diag_mean | trace_per_pixel |
|---|---:|---:|---:|---:|---:|
| BOA-0 gaussian/legacy | 29.88 / 29.95 / 28.26 | 6.76 / 6.63 / 5.70 | 0.5003 / 0.5006 / 0.5008 | 0.1145 / 0.1148 / 0.1014 | 938.78 / 940.79 / 831.31 |
| BOA-1 orthogonal/legacy | 29.53 / 29.52 / 27.95 | 6.64 / 6.42 / 5.57 | 0.5063 / 0.5066 / 0.5054 | 0.1118 / 0.1113 / 0.0991 | 916.41 / 912.37 / 812.44 |

BOA-1 的能量不是更大，反而略低；sparsity 也只是从约 50.0% 到 50.6%。因此 BOA-1 的正向信号更可能来自：

```text
随机方向覆盖更均匀，减少了 gaussian 采样下的方向冗余；
而不是简单的 feature scale 变大或 ReLU 激活更多。
```

这与 BOA 的核心动机一致：在同样 buffer 下改善随机特征质量。

### 3.4 为什么 antithetic 明显失败

BOA-2 与 BOA-0 的能量接近：

```text
BOA-2 gram_diag_mean 与 trace_per_pixel 基本接近 BOA-0；
BOA-2 sparsity 精确接近 0.5；
没有 NaN/Inf。
```

但 BOA-2 new mIoU 从 `43.2099` 掉到 `38.3768`。这说明失败不是数值爆炸，而是机制本身。

最合理解释是：

```text
orthogonal_antithetic 在总 buffer=8196 不变时，把独立方向数减半；
[W, -W] 让 ReLU 可以覆盖正负半轴，但 C-RLS 更需要足够多的独立随机方向；
当前 VOC 15-5 下，方向数损失远大于符号补偿收益。
```

逐类看，pottedplant 和 tvmonitor 的大幅掉点最严重：

| 类别 | BOA-0 | BOA-2 | BOA-3 |
|---|---:|---:|---:|
| pottedplant | 26.9873 | 18.9070 | 18.7979 |
| tvmonitor | 31.5120 | 20.2507 | 18.9278 |

这类小目标/细粒度新类更依赖随机方向容量。独立方向数减半后，新类边界最先受伤。

### 3.5 为什么 kaiming 没救回 antithetic

BOA-3 的 feature scale 明显放大：

| case | mean | gram_diag_mean | trace_per_pixel |
|---|---:|---:|---:|
| BOA-2 antithetic/legacy | 29.97 / 30.01 / 28.32 | 0.1150 / 0.1151 / 0.1018 | 942.95 / 943.60 / 834.28 |
| BOA-3 antithetic/kaiming | 73.41 / 73.52 / 69.37 | 0.6903 / 0.6908 / 0.6107 | 5657.69 / 5661.61 / 5005.69 |

但是 BOA-3 比 BOA-2 更差。说明：

```text
antithetic 的问题不是 scale 太小；
单纯放大 E 会改变 gamma 相对强度，却不会补回丢掉的独立方向。
```

这也再次呼应 RHL 归一化线的经验：只做尺度操作通常不是当前瓶颈。

### 3.6 P1 结论

P1 第一轮结论应拆成两句：

```text
BOA-RHL 的 block orthogonal 分支值得继续；
BOA-RHL 的 antithetic 分支当前应停止。
```

不应把 BOA-2/3 的失败扩大为“BOA-RHL 整体失败”。真正失败的是：

```text
在固定总 buffer 下使用 [W, -W] antithetic pair。
```

而 BOA-1 支持的是：

```text
在固定总 buffer 下，用分块正交方向替代普通 gaussian，有小幅但一致的正向迹象。
```

---

## 4. 对用户提出的四个方向的判断

### 4.1 方向 1：跳出 P0/P1，重新构思多随机种子集成和 BOA-RHL

这个方向应部分采纳，但不能变成无边界重写。

可采纳部分：

```text
RHL-SE 不应继续局限于普通 seed1/2/3 的概率平均；
BOA-RHL 不应继续把 antithetic 当成默认增强；
二者可以重构为“正交随机特征 + 受控多 seed 复核 + val-only 融合”的新组合。
```

不建议部分：

```text
不要立刻推翻所有 P0/P1 代码；
不要同时发散成 global random_seed ensemble、snapshot、TTA、BOA、PGH、normalization 全部大网格。
```

新的定义建议：

```text
BOA-RHL 1.5 = orthogonal legacy as base RHL + rhl_seed repeat + optional val-driven ensemble
```

它比原始 BOA 更收缩，也比原始 RHL-SE 更有单模型方法贡献。

### 4.2 方向 2：在 P0/P1 现有框架下调整优化

这个方向应该作为近期最高优先级。

推荐只保留三个调整点：

1. **BOA-1 复核**  
   固定 `rhl_init=orthogonal`、`rhl_scale_mode=legacy`，补 `rhl_seed=2/3`。如果均值仍高于 BOA-0，说明正交采样是真信号；如果只 seed1 偶然高，则止损。

2. **orthogonal scale audit**  
   当前没有测 `orthogonal + unit` 或 `orthogonal + kaiming`。因为 BOA-1 的能量略低但效果略好，需要分离“正交覆盖”与“scale/gamma 有效强度”。建议只做最多 2 个 case：
   - `orthogonal + unit`
   - `orthogonal + kaiming`

3. **BOA-RHL-SE 的条件触发**  
   只有当 orthogonal 多 seed 的 mean 明确高于 BOA-0，才做 ensemble；否则不要把弱单模型信号再叠成更复杂的弱融合。

P0 现有框架下不建议继续做：

- 更多全局权重；
- 直接 seed4/5；
- test-driven class weight；
- antithetic scale sweep。

### 4.3 方向 3：继续 P2 和 P3

P2 和 P3 不应同等优先。

#### P2：PGH-RHL-lite，建议作为下一条主方法线

P0/P1 共同暴露的问题是：

```text
纯随机子空间有信号，但类别结构不足；
只改善随机方向覆盖能带来小增益，但不够强。
```

这正好支持 P2 的动机：给 RHL 加少量类别原型语义锚点。PGH-RHL-lite 的方法价值高于 Snapshot，也高于继续扩大 RHL-SE。

建议下一轮 P2 只做：

```text
train-only prototype collection
cosine prototype buffer
proto_scale=1.0
proto_scale=sqrt(buffer_size / num_proto)
VOC 15-5 sequential step1 单模型
```

不要一开始做多原型、k-means、RBF sigma 大搜索。

#### P3：Snapshot analytic ensemble，低优先级系统补充

P3 更像系统级集成，不是 RHL 机制创新。它适合：

```text
结项需要 ensemble system；
需要给最终系统指标补一个上界；
导师明确要求传统 ensemble 对照。
```

但它不应抢 P2/BOA 的主线时间。原因是 P0 已经证明“集成本身”在当前相似成员上收益很小；Snapshot 的成员差异可能更大，但论文方法归因会转向 backbone snapshot，而不是 RHL。

### 4.4 方向 4：准备 RHL 归一化新方案

这个方向可以准备，但不应重复旧实验。

当前归一化线的结论是：

```text
l2_sqrt / layernorm / 常规 gamma sweep 不是主瓶颈；
尺度控制能实现，但没有可靠涨点。
```

如果后续形成“RHL归一化新方案.md”，建议核心不再是 `l2_sqrt`，而是两个更有机制变化的版本：

1. **partial / power norm**  
   用 `beta=0.25/0.5` 弱化尺度波动，但保留一部分幅值信息。

2. **class-aware analytic update / weighted C-RLS**  
   与其只改 row norm，不如在解析闭式解中引入类别权重，直接处理新旧类和像素不平衡。

归一化新方案应作为“数值与解析更新线”，与 PGH-RHL-lite 并行准备；但短期不应优先于 BOA-1 复核。

---

## 5. 下一步建议路线

### 5.1 近期最高优先级：BOA-RHL 1.5 复核

目标：判断 BOA-1 是否是真信号。

建议实验矩阵：

| 编号 | 配置 | 目的 |
|---|---|---|
| BOA-1s2 | `orthogonal + legacy + rhl_seed=2` | 检查正交采样跨 seed 是否稳定 |
| BOA-1s3 | `orthogonal + legacy + rhl_seed=3` | 检查均值和方差 |
| BOA-scale-unit | `orthogonal + unit + rhl_seed=1` | scale audit，不做大搜索 |
| BOA-scale-kaiming | `orthogonal + kaiming + rhl_seed=1` | 判断 orthogonal 下 scale 放大是否有益 |

继续条件：

```text
orthogonal legacy 多 seed mean 超 BOA-0 至少 +0.05 all；
或至少一个 orthogonal scale case 超 BOA-0 +0.10 all / +0.30 new。
```

停止条件：

```text
seed2/3 无法复现 BOA-1 正向；
scale audit 无收益；
则 BOA-RHL 降级为负/弱正消融，不再扩展。
```

### 5.2 并行准备：PGH-RHL-lite 设计与实现

目标：给 RHL 加类别结构，而不是继续依赖随机方向偶然性。

最小设计：

```text
只用 train split 收集 prototype；
先用 cosine similarity；
默认不使用背景 prototype；
原型特征与 random RHL concat；
只跑单模型 step1；
val/test 协议保持干净。
```

为什么它值得做：

```text
P0 的 oracle 显示类别/像素级选择空间存在；
P1 的 BOA-1 显示改善随机特征质量有一点收益；
PGH 进一步把“随机质量”升级为“类别结构”，更可能产生论文级增益。
```

### 5.3 低优先级：RHL-SE 2.0

RHL-SE 2.0 不建议现在大做。只有在下面任一条件满足时再推进：

```text
BOA-orthogonal 多 seed 均值稳定高于 BOA-0；
PGH-RHL 单模型有明显 new 提升；
需要一个轻量集成系统模块配合结项。
```

如果推进，方向不是全局权重，而是：

```text
val-calibrated class-wise reliability；
old/new prior；
margin / entropy gate；
只在 val 上定规则，test 只最终确认。
```

### 5.4 暂缓：P3 Snapshot

Snapshot analytic ensemble 建议暂缓。它可以作为系统级上界，但不是 RHL 主创新。除非需要结项展示集成系统，否则不建议先做。

### 5.5 准备但不立即实现：RHL 归一化新方案

后续可以生成一份干净的“RHL归一化新方案.md”，但要吸收当前失败经验：

```text
不再重复 l2_sqrt gamma sweep；
重点转向 partial norm 和 weighted C-RLS；
目标从“归一化一定涨点”改为“解析更新中的幅值保留与类别不平衡控制”。
```

---

## 6. 最终判断

P0/P1 第一轮的真实价值在于帮助我们收缩方向：

```text
RHL-SE：证明互补存在，但普通 seed ensemble 太弱；
BOA-RHL：证明 block orthogonal 有弱正信号，但 antithetic 应立即止损；
RHL normalization：旧实验说明单纯尺度控制不是瓶颈；
PGH-RHL-lite：成为下一条更有方法贡献的主线候选。
```

因此下一步不应是“继续把所有方向都跑一遍”，而应是：

1. **短期执行 BOA-RHL 1.5 复核**：只围绕 `orthogonal + legacy` 做 seed 和 scale 小矩阵。
2. **并行设计 PGH-RHL-lite**：把类别原型作为 RHL 的语义锚点。
3. **把 RHL-SE 降级为辅助模块**：保留 val-driven class-wise 协议和 diagnostics，不再继续全局权重/seed 堆叠。
4. **把 RHL 归一化重新定义为 partial norm / weighted C-RLS 线**：后续单独成文，不沿用旧的 l2_sqrt 叙事。

这条路线比“P0 不够强就直接 P2/P3”更稳，也比“继续在 P0/P1 里无边界调参”更有研究质量。
