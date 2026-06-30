# 自适应伪标签阈值 fixed0.7 结果、严格审查与阶段结论

生成时间：2026-06-30
代码分支：`feature/adaptive-pseudo-label`
审查对象：自适应伪标签阈值实现、`15-5 overlap` 三组已完成实验、下一步实验决策

## 1. 结论先行

当前三组同协议实验已经能给出一个清晰阶段结论：

| strategy | all mIoU | old `0-15` mIoU | new `16-20` mIoU | Mean Acc | 伪标签候选 | 接受比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 70.3091 | 79.1179 | 42.1209 | 76.9396 | - | - |
| `batch_class q0.7` | 70.4535 | 79.2941 | 42.1636 | 77.2009 | 49,963,137 | 30.0008% |
| `fixed0.7` | **70.7383** | **79.6346** | **42.2703** | **77.8997** | 49,963,137 | 90.3503% |

阶段判断：

1. **伪标签在当前 `15-5 overlap` 下确实有轻微正收益**：`fixed0.7` 相比 `off` 提升 `+0.4292` all mIoU，old 类提升 `+0.5167`，new 类提升 `+0.1494`。
2. **当前 `batch_class q0.7` 不是有效改进**：它相比 `off` 只有 `+0.1444` all mIoU，但比原始固定阈值风格的 `fixed0.7` 低 `-0.2848` all mIoU。
3. **结果不符合“类别自适应阈值优于固定阈值”的预期，但符合“`q=0.7` 太保守”的解释**：`batch_class q0.7` 只保留 top 30% 候选，`fixed0.7` 保留约 90% 候选。在 `15-5 step1` 这个最后一步场景里，大量保留旧类伪标签反而更有利。
4. **目前没有发现代码层面错误足以推翻实验结果**：CodeRabbit 审查 0 findings；本地 21 个单元测试通过；三组 manifest 关键字段一致；两组启用伪标签的候选像素数完全一致，teacher checkpoint hash 完全一致。

因此，当前方法线应调整为：

```text
不要继续把 batch_class q0.7 包装成正向方法。
先完成 batch_global q0.7 消融，确认“动态阈值”本身是否也过保守。
若 fixed0.7 仍然最好，则自适应方向需要改成“保留量受控/伪标签质量审计/低 quantile 或阈值 sweep”，而不是继续堆新模块。
```

## 2. 第一性原理核验

### 2.1 目标指标是什么

当前实验评估的是 VOC `15-5 overlap step1` 的完整 21 类 test mIoU：

```text
all mIoU = mean(IoU class 0 ... class 20)
old mIoU = mean(IoU class 0 ... class 15)
new mIoU = mean(IoU class 16 ... class 20)
```

`15-5` 中：

```text
step0: 0-15
step1: 16-20
```

因此 step1 是唯一增量步骤，也是最后一个增量步骤。

### 2.2 训练、验证、测试各自看到什么标签

代码事实：

- `setting=overlap` 时，训练集标签只保留当前 step 的 `target_cls`，其它类别映射为 background `0`。
- step1 当前类是 `16-20`，所以训练标签中的旧类 `1-15` 会被压成 background。
- test/eval 不使用这种隐藏映射，仍在完整 VOC val/test 标签上计算 21 类指标。

所以伪标签的作用点非常明确：

```text
在 step1 训练 batch 中，把一部分“当前标签为 background，但 teacher 高置信预测为旧类”的像素改回旧类标签。
```

它不改 test label，也不改 metric。

### 2.3 模型实际接收什么输入、优化什么目标

step1 链路：

1. 加载同一个 step0 DeepLabV3 checkpoint 作为 teacher/backbone。
2. 用 step0 数据先 fit AIR 旧类分类头。
3. 遍历 step1 overlap 数据。
4. 若启用伪标签，则用 `model_prev(images)` 预测旧类概率。
5. 只在 `labels == 0` 且 teacher top1 属于旧类的位置写入伪标签。
6. 用混合后的 label fit AIR。
7. test 时只加载最终 AIR `final.pth` 评估，伪标签逻辑不参与评估。

这意味着三组实验的差异应来自 step1 fit target，而不是 eval/test 口径变化。

## 3. 实验可比性核验

三组实验：

| 名称 | 输出目录 |
| --- | --- |
| `off` | `checkpoints/20260630_pseudo_15-5_overlap_off_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1` |
| `fixed0.7` | `checkpoints/20260630_pseudo_15-5_overlap_fixed0p7_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1` |
| `batch_class q0.7` | `checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step1` |

manifest 对齐结果：

```text
model/task/setting/curr_step/batch_size/buffer/gamma/random_seed/loss_type/output_stride/base_subpath/base_checkpoint_sha256/resolved_air_feature_source/rhl_norm/rhl_seed 全部一致。
```

共同 base checkpoint：

```text
checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
sha256 = 6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913
```

两组启用伪标签的候选像素数完全一致：

```text
fixed0.7 candidate_count      = 49,963,137
batch_class q0.7 candidate_count = 49,963,137
```

这个事实很关键：候选集合相同，teacher 相同，差异来自阈值策略接受了多少伪标签，而不是数据、模型、checkpoint 或 teacher 输出变化。

## 4. 结果深入分析

### 4.1 fixed0.7 为什么明显优于 batch_class q0.7

`batch_class q0.7` 的 quantile 语义是保留每类候选里大约最高的 30%：

```text
threshold = 70% quantile
accepted = score >= threshold
```

所以它的总接受比例接近 30% 是预期行为，不是 bug。

而 `fixed0.7` 的行为是：

```text
accepted = score >= 0.7
```

在当前 teacher 输出下，绝大多数候选都超过 0.7，因此接受比例达到 90.35%。

当前 `15-5 overlap step1` 是最后一步，没有后续未来类别。背景里主要问题是旧类被隐藏，而不是未来类还没有登场导致的强污染。在这种情形下，保守地只保留 30% 旧类伪标签会损失大量可用旧类监督；固定 0.7 接受更多旧类像素，反而更符合这个单步场景。

### 4.2 fixed0.7 的增益来自哪些类别

相对 `off`，`fixed0.7` 的 per-class 变化中最明显的是：

| class | off IoU | fixed0.7 IoU | delta |
| --- | ---: | ---: | ---: |
| 9 | 30.8856 | 35.0909 | +4.2052 |
| 11 | 52.1646 | 55.2371 | +3.0725 |
| 15 | 84.9443 | 85.6706 | +0.7262 |
| 20 | 30.7782 | 31.0479 | +0.2697 |

这说明 fixed0.7 的收益主要不是平均散布在所有类上，而是明显修复了少数旧类或背景相关类别。`batch_class q0.7` 在 class 9、11、15 上也有提升，但提升远小于 fixed0.7。

### 4.3 为什么 new 类也轻微提升

直觉上伪标签保护旧类，可能压制新类。但当前 new `16-20` 也从 `42.1209` 提升到 `42.2703`，原因可能是：

1. 旧类从背景里被恢复后，背景类监督噪声减少，边界更干净。
2. AIR 闭式 fit 的 target matrix 更接近真实语义，减少背景类对新类区域的吸附。
3. 增幅只有 `+0.1494`，仍然很小，不能过度解释为新类学习能力显著改善。

对抗性保留意见：这也可能只是单 seed 波动或少数类变化造成的平均数变化，需要至少 `batch_global` 和后续 seed/setting 复验再判断。

## 5. 代码实现严格审查

### 5.1 审查范围

相对 `origin/main` 的主要改动包括：

- `utils/pseudo_label.py`
- `trainer/trainer.py`
- `utils/parser.py`
- `utils/run_manifest.py`
- `run.sh`
- `tools/run_adaptive_pseudo_label.sh`
- `tools/summarize_adaptive_pseudo_label.py`
- `tools/calibrate_pseudo_label_thresholds.py`
- `tests/test_pseudo_labeling.py`
- `tests/test_run_manifest.py`

### 5.2 自动审查和本地验证

已执行：

```bash
coderabbit review --agent --base origin/main --dir /root/2TStorage/lyc/SegACIL
```

结果：

```text
review_completed, findings = 0
```

已执行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile \
  utils/pseudo_label.py trainer/trainer.py utils/parser.py utils/run_manifest.py \
  tools/summarize_adaptive_pseudo_label.py tools/calibrate_pseudo_label_thresholds.py \
  tests/test_pseudo_labeling.py tests/test_run_manifest.py
```

结果：无语法错误。

已执行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest \
  tests.test_pseudo_labeling tests.test_run_manifest -v
```

结果：

```text
Ran 21 tests in 0.158s
OK
```

已执行：

```bash
bash -n run.sh tools/run_adaptive_pseudo_label.sh
grep -n '[“”‘’]' <changed executable files>
```

结果：shell 语法检查通过；未发现中文弯引号进入可执行文件。

### 5.3 关键代码路径审查

| 审查点 | 代码位置 | 结论 |
| --- | --- | --- |
| sequential 防误用 | `Trainer.validate_pseudo_label_protocol()` | `setting=sequential` 且启用伪标签会报错，合理 |
| 旧类 ID 计算 | `Trainer.old_class_ids_for_step()` | `15-5 step1` 得到 `1-15`，不含背景 0，合理 |
| teacher class 数 | `Trainer.teacher_class_count_for_step()` | step1 teacher 输出 16 类，和 step0 `0-15` 对齐 |
| teacher 输出兼容 | `extract_teacher_probabilities()` | 支持 DeepLab tuple 与 AIR/NHWC，拒绝歧义 layout |
| 候选像素定义 | `compute_pseudo_label_candidates()` | 只允许 `labels == 0`、非 ignore、teacher top1 为旧类 |
| 固定阈值兼容 | `resolve_pseudo_label_strategy()` | `--use_pseudo_label` 且不传 strategy 时回退为 `fixed` |
| 阈值统计 | `save_pseudo_label_stats()` | 保存 candidate/accepted/per-class/threshold/fallback |
| 可复现性 | `write_run_manifest()` | 保存 args、base checkpoint path/hash、git dirty、AIR source |
| runner 边界 | `tools/run_adaptive_pseudo_label.sh` | `SKIP_STEP0=1` 必须显式传 `BASE_SUBPATH`，避免误用 step0 |

### 5.4 未发现会影响当前结果的代码错误

当前没有证据表明三组结果受到以下问题影响：

- checkpoint 复用错误；
- `BASE_SUBPATH` 指错；
- teacher checkpoint hash 不一致；
- batch size 不一致；
- `setting` 混用；
- sequential 中伪标签静默无效；
- eval/test 被伪标签污染；
- summary 读取旧结果；
- strategy 没有进入真实训练链路。

### 5.5 仍需保留的实现风险

| 风险 | 等级 | 解释 | 最小反证 |
| --- | --- | --- | --- |
| 单 seed 结果可能偶然 | major | 当前对比都是 seed1 | 后续只在候选方法上补 seed，不要全矩阵无脑扩 |
| `bce_loss` 下 sigmoid 分数非归一化 | minor | 多类可同时高置信，阈值语义不同于 softmax | 与原固定阈值同路径，当前公平；若写论文需说明 |
| `q=0.7` 过保守 | major | 当前结果已经支持这个风险 | 跑 `batch_global q0.7`、低 quantile 或 fixed threshold sweep |
| 无 raw-mask pseudo precision | major | 无法判断接受的伪标签是高 recall 还是污染 | 写诊断工具，用原始完整 GT 估计 pseudo precision/recall |
| `15-5 overlap` 不是论文主伪标签表格协议 | major | 不能直接对比论文 69.36 | 若要对论文 baseline，跑 `15-1 overlapped` |

## 6. ARA 严格性审查

按 ARA Level 2 六维度做阶段性审查：

| 维度 | 分数 | 评价 |
| --- | ---: | --- |
| D1 Evidence Relevance | 4 | off/fixed/batch_class 同协议结果直接回答伪标签是否有用、adaptive 是否优于 fixed |
| D2 Falsifiability | 4 | `fixed >= batch_class` 已经 falsify 当前 `batch_class q0.7` 改进主张 |
| D3 Scope Calibration | 3 | 可以说 `15-5 overlap seed1` 下 fixed 最好；不能说优于原论文或多步 overlap |
| D4 Argument Coherence | 4 | 现象与“接受比例/单步旧类恢复”解释一致 |
| D5 Exploration Integrity | 4 | 负结果被记录，没有把轻微涨点包装成方法成立 |
| D6 Methodological Rigor | 3 | 缺 batch_global、raw-mask audit、multi-seed 或 `15-1` paper-protocol 复验 |

总体判断：

```text
Weak Accept as diagnostic result.
Reject 当前 batch_class q0.7 作为有效方法改进。
Accept fixed0.7 as current local same-protocol baseline winner.
```

## 7. 原论文 baseline 配置与 69.36 问题

### 7.1 原论文的配置是什么

根据 CFSSeg 论文主表和本项目精读文档，2D VOC 部分大体分为两类：

1. **Sequential setting**
   - VOC 2012；
   - `15-1` 和 `15-5`；
   - 论文中 CFSSeg / Ours 使用 DeepLabV3 系列 backbone；
   - 论文报告 `15-1` 与 `15-5` 的 Ours all mIoU 都约为 `70.0`。

2. **Disjoint / Overlapped setting**
   - VOC 2012；
   - 主要报告 `15-1` 和 `10-1`；
   - 伪标签用于缓解旧类被标为 background 的 semantic drift；
   - 论文没有把当前我们正在跑的 `15-5 overlap` 作为主表直接对照项。

参考来源：

- CFSSeg arXiv 页面：<https://arxiv.org/html/2412.10834v2>
- 本地精读：`AI_docs/论文精读/CFSSeg_精读笔记.md`
- 项目总览：`AI_docs/课题Home.md`

### 7.2 overlap 设置下是不是要对比 69.36

只有在你跑的是 **VOC `15-1 overlapped` 完整协议** 时，69.36 才是对应的论文 Ours all mIoU。

当前实验是：

```text
VOC 15-5 overlap, step1, seed1, batch32, deeplabv3_resnet101, same local step0
```

所以当前实验不能直接和 69.36 做公平比较。即使当前 `fixed0.7` 的 all mIoU 是 `70.7383`，也不能写成“超过原论文 overlap 69.36”，因为协议不同：

| 项目 | 论文 69.36 | 当前 fixed0.7 |
| --- | --- | --- |
| task | `15-1` | `15-5` |
| setting | overlapped | overlap |
| 增量步数 | 5 个 step | 1 个 step |
| future-class 背景污染 | 多步存在 | step1 已是最后 5 类 |
| 可比性 | 论文主表 baseline | 本地同协议 baseline |

正确写法是：

```text
在本地 VOC 15-5 overlap 单步协议中，fixed0.7 达到 70.7383 all mIoU。
这说明固定伪标签在 15-5 overlap 下有轻微收益。
它不是论文 15-1 overlapped 69.36 的同协议复现或超越。
```

### 7.3 15-1 和 15-5 是否会影响结果

会，尤其在 overlap/disjoint 下影响很大。

原因：

1. `15-1` 是多步增量，每次只学 1 个新类，semantic drift 会连续累积。
2. `15-1` 的早期 step 中，background 里可能混有旧类、当前类以外的未来类和真实背景。
3. `15-5` 只有一个增量 step，step1 一次学完 `16-20`，没有后续 future classes。
4. 伪标签的收益和风险都与“背景里到底混了什么”强相关。

所以 sequential 表里论文可能出现 `15-1` 和 `15-5` 接近或相同，不代表 overlap/disjoint 下二者也能等价。

## 8. 下一步实验与操作

### 8.1 已启动：batch_global q0.7

目的：隔离“动态阈值”与“类别级阈值”的作用。

已启动 tmux：

```text
session: apl630_overlap_global
pid: 994239
gpu memory at launch: about 51.6GB
```

命令等价于：

```bash
cd /root/2TStorage/lyc/SegACIL
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass
export TMPDIR=/root/2TStorage/tmp

PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
SUBPATH=20260630_pseudo_15-5_overlap_batchglobal_q0p7_seed1_bs32_reuse20260627step0 \
BASE_SUBPATH=20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32 \
SKIP_STEP0=1 \
TASK=15-5 \
SETTING=overlap \
PSEUDO_LABEL_STRATEGY=batch_global \
PSEUDO_LABEL_QUANTILE=0.7 \
PSEUDO_LABEL_CONFIDENCE=0.7 \
BATCH_SIZE=32 \
STEP0_BATCH_SIZE=32 \
BUFFER=8196 \
GAMMA=1 \
RANDOM_SEED=1 \
bash tools/run_adaptive_pseudo_label.sh
```

输出路径：

```text
checkpoints/20260630_pseudo_15-5_overlap_batchglobal_q0p7_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1
logs/pseudo_label/20260630_pseudo_15-5_overlap_batchglobal_q0p7_seed1_bs32_reuse20260627step0.log
logs/pseudo_label/20260630_pseudo_15-5_overlap_batchglobal_q0p7_seed1_bs32_reuse20260627step0_summary.md
```

### 8.2 batch_global 之后的决策

| batch_global 结果 | 判断 | 下一步 |
| --- | --- | --- |
| 接近 `batch_class`，且低于 `fixed0.7` | `q=0.7` 动态阈值整体过保守 | 停止当前 adaptive q0.7 路线，做 raw-mask audit 或低 quantile sweep |
| 高于 `batch_class` 但仍低于 `fixed0.7` | 类别级阈值反而伤害，固定阈值仍是强 baseline | 不做 class-wise 复杂化，考虑 fixed threshold sweep |
| 高于 `fixed0.7` | 动态全局阈值有价值 | 补 disjoint 和 seed，暂不加新模块 |

### 8.3 暂不建议立即做的事

1. 暂不把 `batch_class q0.7` 写成论文方法贡献。
2. 暂不开发更复杂的 class cap / margin / artifact 训练改动。
3. 暂不把 `15-5 overlap` 直接并入 `15-5 sequential` 主表。
4. 暂不拿当前 70.7383 和论文 69.36 做“超越原论文”的叙事。

### 8.4 候选后续方案

如果 `batch_global q0.7` 也低于 `fixed0.7`，优先级建议：

1. **raw-mask pseudo audit 工具**：用未隐藏原始 GT 估计伪标签 precision/recall/confusion，先解释 fixed 为什么好。
2. **低 quantile sweep**：例如 `batch_global q0.1/q0.3` 或 `batch_class q0.1/q0.3`，让接受比例接近 fixed0.7 的 90% 或 70%，测试“自适应阈值本身”而不是“过保守阈值”。
3. **fixed threshold sweep**：例如 `fixed0.8/fixed0.9`，看固定阈值是否还有更优点；这也是原始 baseline 的强基线化。

当前不建议在没有 audit 的情况下继续堆复杂模块，因为结果已经显示最简单的 fixed0.7 是强对照。

## 9. 最终阶段结论

当前已经可以确定：

1. 伪标签可用于 `15-5 overlap`，因为该 setting 下旧类在 step1 训练标签中会被映射为 background。
2. sequential 下伪标签不是当前主线，因为旧类标签不以同样方式被隐藏；当前代码也会拒绝 `sequential + pseudo-label`。
3. 当前 `batch_class q0.7` 不成立为有效改进；它的主要问题不是代码没跑通，而是阈值策略过保守。
4. `fixed0.7` 是当前 `15-5 overlap seed1` 下最强同协议结果，all mIoU `70.7383`。
5. 当前代码实现未发现会污染实验结论的问题，但研究结论仍受单 seed、非论文主表协议、无 raw-mask audit 限制。

下一步先等 `batch_global q0.7` 完成。若它没有超过 `fixed0.7`，这条自适应伪标签阈值线应从“类别级 quantile”转为“伪标签质量审计 + 接受比例校准”，而不是继续沿 `q=0.7 batch_class` 往下做。
