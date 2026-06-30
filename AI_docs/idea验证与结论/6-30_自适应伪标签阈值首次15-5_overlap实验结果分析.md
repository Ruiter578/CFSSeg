# 自适应伪标签阈值首次 15-5 overlap 实验结果分析

生成时间：2026-06-30
代码分支：`feature/adaptive-pseudo-label`
实验目录：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32`

## 1. 先给结论

本次实验确认了一件事：当前自适应伪标签阈值实现已经真实进入 `15-5 overlap step1` 训练链路，并且在 step1 生成了可审计的 `pseudo_label_stats.json`。它不是 sequential 复现，也不是原论文主要表格里的 `15-1/10-1 overlap/disjoint` 多步设置，而是一个面向 VOC `15-5 overlap` 的独立验证实验。

结果本身看起来不差：step1 all mIoU 为 `70.45%`，旧类 `0-15` mIoU 为 `79.29%`，新类 `16-20` mIoU 为 `42.16%`。但是，当前本地只找到这一组 `15-5 overlap step1` 结果，没有同协议的 `pseudo_label_strategy=off` 或 `fixed0.7` baseline。因此现在只能说"batch-class 自适应阈值在该配置下跑通并得到 70.45 all mIoU"，不能说"它已经证明比原始伪标签或无伪标签涨点"。

最重要的实验缺口是：必须补 `15-5 overlap/off` 和 `15-5 overlap/fixed0.7`，并且复用同一个 step0 checkpoint。否则无法判断收益来自伪标签、来自 overlap 协议、来自 step0 质量，还是来自随机波动。

## 2. 本次实验的真实 config

本次结果来自：

- step0 manifest：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/run_manifest.json`
- step1 manifest：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step1/run_manifest.json`
- step1 test result：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step1/test_results_20260628_071622.json`
- step1 pseudo stats：`checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step1/pseudo_label_stats.json`

| 项目 | step0 | step1 |
| --- | --- | --- |
| model | `deeplabv3_resnet101` | `deeplabv3_resnet101` teacher + AIR |
| task | `15-5` | `15-5` |
| setting | `overlap` | `overlap` |
| class split | `0-15` | `16-20` |
| batch size | `32` | `32` |
| train epoch | `50` | AIR one-pass fit |
| loss | `bce_loss` | analytic fit target |
| output stride | `8` | teacher/backbone stride 8 |
| buffer | `8196` | `8196` |
| gamma | `1.0` | `1.0` |
| random seed | `1` | `1` |
| RHL norm | `none` | `none` |
| AIR feature source | none | `auto -> decoder` |
| pseudo label | off | on |
| pseudo strategy | `off` | `batch_class` |
| pseudo quantile | `0.7` | `0.7` |
| pseudo fixed confidence | `0.7` | fallback/reference value `0.7` |
| base checkpoint | none | step0 best DeepLab checkpoint |

一个容易误解的细节：manifest 的 `args.overlap` 旧布尔字段为 `false`，但当前代码实际控制数据协议的是 `setting=overlap`。`run.sh` 传入的是 `--setting "$SETTING"`；`datasets/voc.py` 使用 `opts.setting`；`utils/tasks.py:get_dataset_list()` 也使用 `setting` 选择 overlap/disjoint/sequential 数据列表。因此本次实验按代码执行路径看是 `overlap`，不是 disjoint。

## 3. 代码路径：伪标签到底在哪里生效

### 3.1 类别划分

`utils/tasks.py` 中 `15-5` 定义为：

```text
step0: [0, 1, ..., 15]
step1: [16, 17, 18, 19, 20]
```

所以 `15-5` 不是 setting，而是类别增量任务划分。

### 3.2 setting 决定训练标签是否被重映射

`datasets/voc.py:gt_label_mapping()` 的核心逻辑是：

```python
if self.image_set != 'test':
    if self.setting == 'sequential':
        pass
    else:
        gt = np.where(np.isin(gt, self.target_cls), gt, 0)
```

含义是：

- `sequential`：训练标签保留原始可见类别，不把旧类压成背景；
- `overlap/disjoint`：训练时只保留当前 step 的 `target_cls`，其它类全部映射成背景 `0`；
- `test`：不做这种隐藏映射，仍在完整 VOC val 上评估。

这就是伪标签在 non-sequential setting 里有意义的根本原因：训练标签把旧类像素系统性压成了背景，而伪标签试图把一部分高置信旧类从背景里恢复出来。

### 3.3 overlap 与 disjoint 的数据列表差异

`utils/tasks.py:get_dataset_list()` 中：

- `overlap`：选择含有当前 step 类别的图片，图片中可以混有旧类和其它类；
- `disjoint`：选择含有当前 step 类别且类别集合不超出"当前类 + 旧类 + 背景/ignore"的图片。

在 `15-5 step1`，当前类已经是最后 5 个 VOC 类，理论上没有"未来类"了。因此 `15-5 overlap` 的难点主要是旧类被隐藏成背景，而不是像 `15-1/10-1` 早期 step 那样还混有未来类背景。

### 3.4 伪标签应用位置

`trainer/trainer.py` 的 step1 逻辑是：

```text
1. 加载 step0 DeepLab checkpoint 作为 backbone/teacher。
2. 先用 step0 数据拟合 AIR，建立旧类解析分类头。
3. 再遍历 step1 数据。
4. 若开启伪标签，用 model_prev 对当前图像预测。
5. 只在当前标签为 background 的位置，把 teacher 高置信旧类预测写回标签。
6. 用混合标签 fit AIR。
```

也就是说，伪标签不是改 Dataset 文件，不参与 eval/test，只影响 step1 的 AIR fit target。

## 4. 实验结果

### 4.1 指标表

| 阶段 | Overall Acc | Mean Acc | FreqW Acc | all mIoU | old `0-15` mIoU | new `16-20` mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step0 test | 90.26 | 67.47 | 82.09 | 60.51 | 79.42 | 0.00 |
| step1 test | 92.82 | 77.20 | 86.79 | 70.45 | 79.29 | 42.16 |

step1 的 all mIoU 可以由类别组加权复核：

```text
(16 * 79.2941 + 5 * 42.1636) / 21 = 70.4535
```

这说明 summary 中的 old/new/all 关系自洽。

### 4.2 step1 per-class IoU

| 类别 | IoU |
| --- | ---: |
| 0 | 91.24 |
| 1 | 90.04 |
| 2 | 42.15 |
| 3 | 90.29 |
| 4 | 76.05 |
| 5 | 82.11 |
| 6 | 93.47 |
| 7 | 89.38 |
| 8 | 93.97 |
| 9 | 32.62 |
| 10 | 86.11 |
| 11 | 52.99 |
| 12 | 90.60 |
| 13 | 85.37 |
| 14 | 87.28 |
| 15 | 85.02 |
| 16 | 18.73 |
| 17 | 71.71 |
| 18 | 17.45 |
| 19 | 72.07 |
| 20 | 30.85 |

新类内部差异很大。`17` 和 `19` 学得很好，`16/18/20` 明显偏弱。这更像 VOC `15-5` step1 本身的类别难度差异，而不是简单的"伪标签让所有新类一起变好"。

## 5. 伪标签统计说明了什么

step1 伪标签统计：

| 项目 | 数值 |
| --- | ---: |
| strategy | `batch_class` |
| quantile | `0.7` |
| old classes | `1-15` |
| batches | `67` |
| candidates | `49,963,137` |
| accepted | `14,989,337` |
| accepted ratio | `0.300008` |

`accepted_ratio` 接近 `30%` 不是偶然。因为 `quantile=0.7` 的含义是：对候选置信度取 70% 分位点，保留大约最高的 30%。所以这个统计首先说明实现逻辑自洽，不能单独证明伪标签质量高。

按类别看：

| old class | candidates | accepted | accepted ratio | mean threshold | fallback |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 69,025 | 20,711 | 30.01% | 0.9628 | global 62 / none 5 |
| 2 | 1,059,680 | 317,916 | 30.00% | 0.9230 | global 40 / none 27 |
| 3 | 100,045 | 30,020 | 30.01% | 0.9434 | global 55 / none 12 |
| 4 | 162,954 | 48,889 | 30.00% | 0.9271 | global 54 / none 13 |
| 5 | 464,822 | 139,461 | 30.00% | 0.8696 | global 20 / none 47 |
| 6 | 25,324 | 7,599 | 30.01% | 0.9337 | global 58 / none 9 |
| 7 | 609,760 | 182,943 | 30.00% | 0.8952 | global 36 / none 31 |
| 8 | 2,859,685 | 857,986 | 30.00% | 0.9390 | global 16 / none 51 |
| 9 | 6,792,498 | 2,037,780 | 30.00% | 0.9424 | none 67 |
| 10 | 97,157 | 29,158 | 30.01% | 0.8103 | global 32 / none 35 |
| 11 | 4,092,681 | 1,227,834 | 30.00% | 0.9203 | global 3 / none 64 |
| 12 | 3,949,048 | 1,184,778 | 30.00% | 0.9330 | global 9 / none 58 |
| 13 | 206,342 | 61,909 | 30.00% | 0.9267 | global 52 / none 15 |
| 14 | 985,397 | 295,630 | 30.00% | 0.9350 | global 48 / none 19 |
| 15 | 28,488,719 | 8,546,723 | 30.00% | 0.9967 | none 67 |

对抗性解读：

1. 类 `15` 候选数极大，占所有候选的一半以上。这可能说明 teacher 对 class 15 非常频繁且高置信，也可能说明 class 15 在背景候选中被过度预测。没有 raw-mask audit 前，不能判断它是好事。
2. 多个小候选类大量 fallback 到 global threshold，例如 class 1、3、4、6、13、14。这说明 `batch_class` 在很多 batch 上并不真正是纯类别级阈值，而是退回全局阈值。
3. 本次 `min_pixels=1` 很宽松。它保证每个类只要有少量候选就能算分位点，但小样本分位点会抖。后续如果出现不稳定，需要提高 `min_pixels` 或引入 offline artifact。

## 6. 回答几个关键疑问

### 6.1 原文为什么只在 overlap/disjoint 使用伪标签？

因为伪标签解决的是 non-sequential setting 的特定错误监督：旧类像素在当前 step 被标成背景。原始标签中出现这种系统性错误时，直接闭式更新分类头会把错误标签写入解析解。伪标签的作用是先用旧模型把高置信旧类从背景里"捞回来"，再用混合标签更新。

在 `sequential` 中，旧类标签没有被这种方式隐藏，伪标签的主要病因不存在。因此在 sequential 开伪标签通常不是核心实验，甚至可能把 teacher 错误预测写到本来正确的标签旁边，增加噪声。

### 6.2 原文训练模式是 15-1 和 10-1，不是 15-5。本次 15-5 合理吗？

合理，但它是扩展验证，不是原论文表格的直接复现实验。

`15-5` 仍然是合法的 VOC 类增量任务。只要 setting 是 `overlap` 或 `disjoint`，训练标签仍会隐藏非当前 step 类别，因此伪标签机制仍有使用对象。区别在于：

- `15-1/10-1` 是多步增量，伪标签需要连续多次防止旧类塌缩；
- `15-5` 只有一个增量 step，且 step1 已经包含最后 5 个类别，没有后续未来类；
- 所以 `15-5 overlap` 的伪标签问题更像"一次性旧类恢复"，不完全等价于论文强调的多步 semantic drift 场景。

因此，本次 `15-5 overlap` 适合做课题主数据集 `15-5` 下的支线验证，但如果要把伪标签改进写成强论文贡献，后续最好补 `15-1` 或 `10-1 overlap/disjoint` 的多步证据。

### 6.3 伪标签到底是否可用于 15-5？

可以，但前提是 setting 是 `overlap` 或 `disjoint`。

判断逻辑不是"15-5 能不能用"，而是：

```text
训练标签是否把旧类系统性映射成背景？
```

- `15-5 sequential`：旧类标签可见，伪标签不应作为默认方法；
- `15-5 overlap/disjoint`：旧类在当前 step 训练标签中会被压成背景，伪标签有明确作用点；
- 当前代码也按这个原则实现：`setting == sequential` 且开启伪标签会直接报错。

### 6.4 sequential 设置下伪标签是否没有用？

更准确的说法是：在当前 SegACIL/VOC 实现里，sequential 下伪标签不应该作为有效主线使用。

它不是数学上绝对不可能产生数值变化，而是机制目标不匹配：

1. sequential 的旧类标签没有被隐藏，伪标签要修复的错误背景不存在；
2. teacher 预测不可能比真实标签更可靠；
3. 若允许它写入背景位置，可能把真实背景误改成旧类；
4. 和 overlap/disjoint 比较时会产生 protocol mismatch。

因此，当前代码用 `validate_pseudo_label_protocol()` 拒绝 sequential + pseudo-label 是合理的。

## 7. 当前结果能支持什么结论

可以支持：

1. `batch_class` 自适应阈值在 `15-5 overlap step1` 真正执行了。
2. 伪标签统计落盘完整，可以审计候选数、接受数、类别阈值和 fallback。
3. 该配置下最终得到 `70.45%` all mIoU，旧类保持在 `79.29%`，新类为 `42.16%`。
4. `quantile=0.7` 的行为和预期一致，整体保留约 top 30% 候选。

不能支持：

1. 不能说自适应阈值已经优于无伪标签，因为缺少同协议 `off` baseline。
2. 不能说自适应阈值已经优于原始固定阈值，因为缺少 `fixed0.7` baseline。
3. 不能和 `15-5 sequential` 主结果直接比较，因为 setting 不同，训练标签可见性不同。
4. 不能说 class-wise threshold 一定比 global threshold 好，因为当前大量类别有 global fallback。

## 8. 对抗性风险清单

| 风险 | 为什么重要 | 最小反证方式 |
| --- | --- | --- |
| 没有同协议 baseline | 70.45 可能只是 overlap/off 本来就能达到 | 跑 `15-5 overlap off`，复用同 step0 |
| 没有 fixed0.7 | 不知道 adaptive 是否优于原 CFSSeg 风格阈值 | 跑 `15-5 overlap fixed0.7`，复用同 step0 |
| 类别 15 候选支配 | 伪标签可能被单类预测主导 | 做 raw-mask pseudo precision audit |
| 小类 fallback 多 | batch-class 可能退化成 global | 比较 `batch_global q0.7` |
| 15-5 不是论文主伪标签设置 | 单步结果不能代表多步 semantic drift | 后续补 `15-1 overlap` 多步结果 |
| step0 overlap 与 sequential step0 不同 | step0 背景污染机制不同 | 不跨 setting 直接比较 |

## 9. 当前结论

这次实验是一个有效的首次跑通结果，但还不是方法有效性的最终证据。它最有价值的地方是确认了代码链路、统计链路和 `15-5 overlap` 可运行性。下一步必须补齐同协议对照：先跑 `off`，再跑 `fixed0.7`，必要时跑 `batch_global q0.7`。只有当 `batch_class` 在相同 step0、相同 setting、相同 seed、相同 batch size 下优于这些对照，才能把它作为有效方法推进。

## 10. 2026-06-30 补充：off baseline 已完成

补充实验：

- subpath：`20260630_pseudo_15-5_overlap_off_seed1_bs32_reuse20260627step0`
- result：`checkpoints/20260630_pseudo_15-5_overlap_off_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/test_results_20260630_065032.json`
- step0 来源：`20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32`
- setting：`15-5 overlap`
- strategy：`off`

结果对照：

| strategy | all mIoU | old `0-15` mIoU | new `16-20` mIoU | Overall Acc | Mean Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` | 70.3091 | 79.1179 | 42.1209 | 92.7897 | 76.9396 |
| `batch_class q0.7` | 70.4535 | 79.2941 | 42.1636 | 92.8221 | 77.2009 |
| delta | +0.1444 | +0.1762 | +0.0427 | +0.0324 | +0.2613 |

这个补充结果改变了当前判断强度：`batch_class q0.7` 对 `off` 是正向的，但幅度很小。它更像"可能有轻微收益"，不是强涨点。尤其新类只高 `0.0427` 个百分点，几乎可以视为无明显变化；主要差异来自旧类轻微提升。

因此，当前不能停止在 `off` 对照。下一步必须看 `fixed0.7`：

- 如果 `fixed0.7` 与 `batch_class q0.7` 接近或更高，自适应阈值本身没有独立贡献；
- 如果 `batch_class q0.7` 稳定高于 `fixed0.7`，才有资格继续讨论类别级自适应阈值；
- 如果三者差距都在 `0.1-0.2` mIoU 内，需要至少补随机种子或转向 raw-mask pseudo audit，而不是急着包装方法结论。

## 11. 2026-06-30 再补充：fixed0.7 已完成

`fixed0.7` 已完成，并且超过 `batch_class q0.7`：

| strategy | all mIoU | old `0-15` mIoU | new `16-20` mIoU |
| --- | ---: | ---: | ---: |
| `off` | 70.3091 | 79.1179 | 42.1209 |
| `batch_class q0.7` | 70.4535 | 79.2941 | 42.1636 |
| `fixed0.7` | **70.7383** | **79.6346** | **42.2703** |

最新完整审查与结论见：

```text
AI_docs/idea验证与结论/6-30_自适应伪标签阈值fixed0p7结果与代码审查结论.md
```

该结果推翻了“当前 `batch_class q0.7` 优于原始固定阈值”的假设。当前更准确的结论是：伪标签在 `15-5 overlap` 下有轻微正收益，但最强的是 `fixed0.7`；`batch_class q0.7` 过于保守，暂不能作为有效方法改进。
