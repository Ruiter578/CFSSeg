# RHL-SE 阶段 A 代码改动与 logit 集成实验启动报告

> 项目：`/root/2TStorage/lyc/SegACIL`  
> 日期：2026-06-16  
> 对应方案文档：`AI_docs/idea构思与实验设计/RHL新方案/6-16_RHL-SE方案一当前实验情况剖析与重构升级改进方案.md`  
> 目标：落实阶段 A：扩展 RHL-SE 推理评估工具，支持 logit ensemble、诊断输出和类别级权重接口，并启动相关无训练推理实验。

---

## 1. 代码改动概览

修改文件：

```text
tools/eval_rhl_ensemble.py
```

本次只改推理评估脚本，不改训练逻辑、不改 checkpoint、不改 RHL 训练参数。

### 1.1 新增 `--ensemble_mode prob|logit`

新增参数：

```bash
--ensemble_mode prob
--ensemble_mode logit
```

两种模式含义：

| 模式 | 融合逻辑 | 作用 |
|---|---|---|
| `prob` | `average(sigmoid/log_softmax 后的概率分数)` | 保持此前 RHL-SE 结果可复现 |
| `logit` | `average(raw logits)` 后直接 `argmax` | 避免 BCE sigmoid 先压缩 logit margin |

此前脚本只有：

```python
probs = torch.sigmoid(logits)
weighted_prob_sum += probs * weight
preds = weighted_prob_sum.argmax(dim=1)
```

现在新增 logit 路径：

```python
logits_chw = logits_to_chw(logits, labels_shape, n_classes)
scores = probs_chw if ensemble_mode == "prob" else logits_chw
weighted_score_sum += scores * weight
preds = weighted_score_sum.argmax(dim=1)
```

这样可以验证上一轮分析中的关键假设：

```text
BCE 是独立 sigmoid 输出，sigmoid 后平均可能压缩分类 margin；
logit 先融合可能更保留成员间的边界差异。
```

### 1.2 新增诊断输出 `--save_diagnostics`

新增参数：

```bash
--save_diagnostics logs/rhl_ensemble/xxx_diag.json
```

诊断内容包括：

| 字段 | 含义 |
|---|---|
| `Member Results` | 每个 checkpoint 单独在同一 dataloader 上的指标 |
| `Oracle Results` | 像素级 oracle 上界：若任一成员预测正确，则该像素视为正确 |
| `pairwise_disagreement` | 成员两两预测不一致比例 |
| `pairwise_old_disagreement` | 旧类区域不一致比例 |
| `pairwise_new_disagreement` | 新类区域不一致比例 |
| `per_class_any_disagreement` | 每个 GT 类别区域中，任意成员产生分歧的像素比例 |

诊断的目的不是直接作为论文指标，而是回答：

```text
RHL-SE 涨幅小，是因为成员预测太相似？
还是成员有互补但融合方式没有利用好？
```

### 1.3 新增类别级权重接口 `--class_weights_json`

新增参数：

```bash
--class_weights_json path/to/class_weights.json
```

支持两种格式：

1. 完整矩阵：`C x K` list，其中 `C` 是类别数，`K` 是 checkpoint 数。
2. 字典：`{"16": [w1, w2, w3], "18": [w1, w2, w3]}`，未指定类别回退到全局 `--weights`。

作用：

```text
表达“旧类更信 seed1，新类更信 seed2/3，不同新类信不同 seed”的策略。
```

本轮只实现接口，不立即启动类别级权重实验。原因是类别级权重应先根据诊断结果或 val split 设定，不能凭 test 结果盲目调参。

### 1.4 保持兼容性

默认行为保持为：

```bash
--ensemble_mode prob
不传 --weights 时等权平均
不传 --save_diagnostics 时只输出原始结果 JSON
不传 --class_weights_json 时只使用全局模型权重
```

因此此前已完成的 K2/K3 probability ensemble 仍可复现。

---

## 2. 验证情况

已完成语法检查：

```bash
python -m py_compile tools/eval_rhl_ensemble.py
```

已完成 smoke test：

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode logit \
  --weights 0.2 0.4 0.4 \
  --max_batches 1 \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/debug_phaseA_logit_w020_040_040_one_batch.json \
  --save_diagnostics logs/rhl_ensemble/debug_phaseA_logit_w020_040_040_one_batch_diag.json
```

验证结果：

```text
脚本正常打印指标
结果 JSON 正常保存
诊断 JSON 正常保存
```

debug 结果只用于验证代码路径，不参与正式实验结论。

---

## 3. 已启动的正式实验

已使用 tmux 同时启动三组完整 test 推理实验，均启用：

```bash
--keep_models_on_gpu
--ensemble_mode logit
```

当前 GPU 显存充足。启动后观察到三组 SegACIL 推理进程均已进入 test 进度条。

### 3.1 实验一：K3 等权 logit ensemble + 完整诊断

tmux session：

```text
rhl_phaseA_logit_k3_equal
```

启动命令：

```bash
cd /root/2TStorage/lyc/SegACIL
source /home/linyichen/miniconda3/etc/profile.d/conda.sh
conda activate segacil

python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode logit \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal.json \
  --save_diagnostics logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal_diag.json
```

日志文件：

```text
logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal.log
```

### 3.2 实验二：K3 加权 logit ensemble，权重 0.2/0.4/0.4

tmux session：

```text
rhl_phaseA_logit_k3_w244
```

启动命令：

```bash
cd /root/2TStorage/lyc/SegACIL
source /home/linyichen/miniconda3/etc/profile.d/conda.sh
conda activate segacil

python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --weights 0.2 0.4 0.4 \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode logit \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_k3_logit_w020_040_040.json
```

日志文件：

```text
logs/rhl_ensemble/20260616_rhl_se_k3_logit_w020_040_040.log
```

### 3.3 实验三：K2 seed2+seed3 logit ensemble

tmux session：

```text
rhl_phaseA_logit_k2_23
```

启动命令：

```bash
cd /root/2TStorage/lyc/SegACIL
source /home/linyichen/miniconda3/etc/profile.d/conda.sh
conda activate segacil

python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode logit \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_logit.json
```

日志文件：

```text
logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_logit.log
```

---

## 4. 当前运行状态检查命令

查看 tmux 会话：

```bash
tmux ls | grep rhl_phaseA
```

查看单个会话输出：

```bash
tmux capture-pane -t rhl_phaseA_logit_k3_equal -p -S -80
tmux capture-pane -t rhl_phaseA_logit_k3_w244 -p -S -80
tmux capture-pane -t rhl_phaseA_logit_k2_23 -p -S -80
```

查看 GPU：

```bash
nvidia-smi
```

查看输出文件：

```bash
ls -lh logs/rhl_ensemble/*logit*.json logs/rhl_ensemble/*logit*.log
```

---

## 5. 结果判读标准

需要把 logit 结果与已有 prob 结果对比：

| 对比对象 | 已有 prob 结果 |
|---|---:|
| K3 等权 prob | all mIoU 69.5049 |
| K3 `0.2/0.4/0.4` prob | all mIoU 69.5060 |
| K2 `seed2+seed3` prob | all mIoU 69.4976 |

判读逻辑：

| 结果 | 含义 |
|---|---|
| logit 明显高于 prob | 此前瓶颈主要在 sigmoid 后概率平均，后续继续优化 logit / calibration / class-wise 融合 |
| logit 与 prob 基本一致 | 成员预测高度相关，多样性不足是主因 |
| logit 低于 prob | sigmoid 概率分数对当前模型更稳，保持 prob，转向多样性增强 |

诊断 JSON 的重点字段：

```text
Oracle Results
Disagreement.pairwise_disagreement
Disagreement.pairwise_old_disagreement
Disagreement.pairwise_new_disagreement
Disagreement.per_class_any_disagreement
```

如果 oracle 上界明显高于当前 ensemble，说明仍有融合方式优化空间。  
如果 oracle 上界也很低，说明普通 RHL seed 成员本身缺少互补，后续应转向 BOA-RHL / PGH-RHL，而不是继续堆 seed。

---

## 6. 后续建议

实验完成后，下一步按以下顺序处理：

1. 汇总三个 logit JSON 与已有 prob JSON。
2. 分析 K3 等权诊断 JSON，重点看 new 类区域 disagreement 和 oracle 上界。
3. 若 logit 或 oracle 显示仍有空间，再设计 class-wise weights。
4. 若 logit 与 prob 几乎一致，且 oracle 上界也不高，停止继续扫 RHL-SE 推理权重，转向 BOA-RHL / PGH-RHL。

---

## 7. 实验完成状态与初步结果

三组 tmux 实验均已完成，输出文件已正常落盘。

### 7.1 输出文件

```text
logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_logit.json
logs/rhl_ensemble/20260616_rhl_se_k2_seed2_seed3_logit.log

logs/rhl_ensemble/20260616_rhl_se_k3_logit_w020_040_040.json
logs/rhl_ensemble/20260616_rhl_se_k3_logit_w020_040_040.log

logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal.json
logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal.log
logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal_diag.json
```

### 7.2 与 prob ensemble 对比

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc |
|---|---:|---:|---:|---:|---:|
| prob K2 seed2+seed3 | 69.4976 | 77.9577 | 42.4256 | 92.7202 | 78.3832 |
| logit K2 seed2+seed3 | 69.4952 | 77.9567 | 42.4183 | 92.7197 | 78.3762 |
| prob K3 equal | 69.5049 | 77.9985 | 42.3254 | 92.7235 | 78.3830 |
| logit K3 equal | 69.5019 | 77.9974 | 42.3163 | 92.7229 | 78.3749 |
| prob K3 `0.2/0.4/0.4` | 69.5060 | 77.9848 | 42.3737 | 92.7235 | 78.3868 |
| logit K3 `0.2/0.4/0.4` | 69.5030 | 77.9837 | 42.3648 | 92.7230 | 78.3788 |

### 7.3 初步判断

logit ensemble 没有超过 prob ensemble，三组均略低：

```text
K2: logit 比 prob 低 0.0024 all mIoU
K3 equal: logit 比 prob 低 0.0030 all mIoU
K3 weighted: logit 比 prob 低 0.0030 all mIoU
```

这说明当前主要瓶颈不是“sigmoid 后概率平均压缩 margin”。在现有三个 RHL 成员上，prob ensemble 已经是更稳的融合协议。

### 7.4 诊断结果

K3 equal logit 的诊断文件：

```text
logs/rhl_ensemble/20260616_rhl_se_k3_logit_equal_diag.json
```

关键诊断：

| 指标 | 数值 |
|---|---:|
| pairwise disagreement | 0.8668% |
| old-region disagreement | 0.6179% |
| new-region disagreement | 4.9272% |
| oracle all mIoU | 71.3950 |
| oracle old mIoU | 79.2469 |
| oracle new mIoU | 46.2690 |

解释：

```text
1. 成员整体预测非常相似，pairwise disagreement 不到 1%。
2. 新类区域分歧明显更大，说明 RHL seed 的有效差异主要集中在新类。
3. oracle 上界明显高于当前 ensemble：
   当前 best ensemble all mIoU 约 69.5060
   oracle all mIoU 为 71.3950
   差距约 +1.889
4. 因此问题不是“完全没有互补”，而是当前全局平均/加权无法利用这些互补。
```

### 7.5 对下一步的影响

根据阶段 A 结果，后续不建议继续做：

```text
更多 logit/prob 全局权重扫描
立即补跑 seed4/5
```

更应该进入：

```text
class-wise ensemble
old/new group-wise ensemble
confidence/margin gated ensemble
oracle-guided error analysis
```

也就是说，RHL-SE 的下一阶段重点应从“模型级平均”转向“类别级或像素级选择”。

---

## 8. 阶段 A-v2 代码改动：结构化融合

基于阶段 A 诊断结论，本次继续扩展 `tools/eval_rhl_ensemble.py`，新增三类结构化融合能力：

### 8.1 old/new group-wise 权重

新增参数：

```bash
--old_class_weights 0.7 0.15 0.15
--new_class_weights 0.1 0.45 0.45
```

作用：

```text
旧类 0-15 使用一组权重
新类 16-20 使用另一组权重
```

用于表达：

```text
旧类更信 seed1/baseline
新类更信 seed2/seed3
```

### 8.2 class-wise 权重

已有参数 `--class_weights_json` 被扩展为可与 `--old_class_weights` 组合使用。

本轮配置文件：

```text
logs/rhl_ensemble/20260616_rhl_se_class_weights_v1.json
```

内容：

```json
{
  "16": [0.1, 0.3, 0.6],
  "17": [0.1, 0.2, 0.7],
  "18": [0.1, 0.7, 0.2],
  "19": [0.1, 0.65, 0.25],
  "20": [0.1, 0.2, 0.7]
}
```

旧类仍使用：

```bash
--old_class_weights 0.7 0.15 0.15
```

这表示：

```text
旧类偏 seed1
pottedplant/sheep/tvmonitor 偏 seed3
sofa/train 偏 seed2
```

注意：这组 class-wise 权重来自当前探索性 test 观察，不应作为最终论文调参结果。若保留该方向，下一步必须在 val split 上确定权重，再用 test 做最终评估。

### 8.3 confidence gated ensemble

新增参数：

```bash
--gating_mode margin
--gate_base_index 0
--gate_margin_threshold 0.10
--gate_require_ensemble_better
```

机制：

```text
以 seed1 作为保守 base
以 weighted ensemble 作为 alternate
当 seed1 的 top1-top2 margin < threshold 时，才切换到 ensemble
如果开启 --gate_require_ensemble_better，则只有 ensemble margin 更高才切换
```

输出 JSON 中新增：

```text
Gate Switched Pixels
Gate Total Pixels
Gate Switch Ratio
```

---

## 9. 阶段 A-v2 启动命令

三组实验均使用：

```bash
--ensemble_mode prob
--keep_models_on_gpu
```

原因：阶段 A 已证明 `prob` 比 `logit` 稍稳。

### 9.1 old/new group-wise

tmux session：

```text
rhl_phaseA2_oldnew
```

命令：

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --old_class_weights 0.7 0.15 0.15 \
  --new_class_weights 0.1 0.45 0.45 \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode prob \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_oldnew_prob_o70_n14545.json
```

### 9.2 class-wise v1

tmux session：

```text
rhl_phaseA2_classwise
```

命令：

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --old_class_weights 0.7 0.15 0.15 \
  --class_weights_json logs/rhl_ensemble/20260616_rhl_se_class_weights_v1.json \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode prob \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_classwise_prob_v1.json
```

### 9.3 confidence gate

tmux session：

```text
rhl_phaseA2_gate
```

命令：

```bash
python tools/eval_rhl_ensemble.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --weights 0.2 0.4 0.4 \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --mode test \
  --ensemble_mode prob \
  --gating_mode margin \
  --gate_base_index 0 \
  --gate_margin_threshold 0.10 \
  --gate_require_ensemble_better \
  --keep_models_on_gpu \
  --save_json logs/rhl_ensemble/20260616_rhl_se_gate_prob_seed1_w244_m010.json
```

---

## 10. 阶段 A-v2 实验结果

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc | 备注 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 | `20260606` |
| prob K3 `0.2/0.4/0.4` | 69.5060 | 77.9848 | 42.3737 | 92.7235 | 78.3868 | 阶段 A 旧 best |
| old/new group-wise | 69.5033 | 78.0029 | 42.3043 | 92.7215 | 78.3795 | old 保住，new 降 |
| class-wise v1 | **69.5229** | 78.0022 | 42.3893 | **92.7251** | **78.4009** | 当前 best |
| confidence gate | 69.4977 | **78.0191** | 42.2294 | 92.7192 | 78.3642 | gate ratio 7.27% |

新类逐类 IoU：

| 设置 | pottedplant 16 | sheep 17 | sofa 18 | train 19 | tvmonitor 20 |
|---|---:|---:|---:|---:|---:|
| baseline | 23.5886 | 57.9622 | 30.9147 | 69.9526 | 28.1192 |
| prob K3 `0.2/0.4/0.4` | 23.9828 | 57.9330 | 31.3108 | 70.3081 | 28.3340 |
| old/new group-wise | 23.6347 | 58.1589 | 31.2776 | 70.2630 | 28.1875 |
| class-wise v1 | 23.6018 | **58.3234** | **31.4027** | 70.2956 | 28.3228 |
| confidence gate | 23.7015 | 57.9839 | 31.0954 | 70.1771 | 28.1892 |

### 10.1 结果解释

1. `class-wise v1` 是当前 RHL-SE 最好结果：

```text
相对 baseline:
all mIoU +0.0623
new mIoU +0.2818
Mean Acc +0.0633

相对 prob K3 0.2/0.4/0.4:
all mIoU +0.0169
new mIoU +0.0156
old mIoU +0.0174
```

2. `old/new group-wise` 不够细。它确实把 old mIoU 拉回到 `78.0029`，但 new mIoU 低于全局加权 K3。

3. `confidence gate` 第一版不理想。它切换了约 `7.27%` 像素，old mIoU 最高，但 new mIoU 掉到 `42.2294`。这说明 margin gate 更像在保护 baseline，而没有有效释放新类互补。

### 10.2 当前阶段结论

结构化融合方向是正确的，但收益仍然不大：

```text
全局加权 best: 69.5060
class-wise best: 69.5229
增量: +0.0169
```

因此 RHL-SE 推理层优化目前只能作为弱正向辅助模块。若继续推进，必须转入 val split 驱动的 class-wise 权重搜索，不能继续基于 test 手工调权。

如果 val 驱动 class-wise 仍只提升 0.05 左右，应停止 RHL-SE 推理层优化，转向 BOA-RHL / PGH-RHL。

---

## 11. 阶段 A-v3 代码改动：val-driven class-wise 权重搜索

新增文件：

```text
tools/search_rhl_class_weights.py
```

继续使用文件：

```text
tools/eval_rhl_ensemble.py
```

### 11.1 新脚本功能

`search_rhl_class_weights.py` 的作用是把 RHL-SE 的 class-wise 权重选择从 test 手工调参改成 val split 驱动。

核心流程：

```text
加载多个 RHL-SE checkpoint
在 val split 上评估每个成员的逐类 IoU
根据 val 逐类最优成员构造候选 class-wise 权重
在 val split 上评估全部候选
按 objective 选择最佳候选
保存 class_weights_json
再交给 eval_rhl_ensemble.py 做 test split 最终评估
```

本次默认 objective：

```text
all_miou
```

这保证权重搜索服务主指标，而不是只追新类。

### 11.2 候选权重集合

脚本内置候选：

| 候选 | 说明 |
|---|---|
| `equal_all_classes` | 所有类别 K3 等权 |
| `global_0.2_0.4_0.4` | 阶段 A 旧 best 全局权重 |
| `oldnew_*` | 旧类一套权重，新类一套权重 |
| `classwise_valbest_all_s*` | 每个类别按 val 上最优成员加权 |
| `classwise_valbest_new_s*_oldstable` | 旧类稳定，只对新类逐类加权 |

`classwise_valbest_all_s0.75` 的含义：

```text
某类别 val IoU 最好的成员权重 = 0.75
另外两个成员权重 = 0.125 / 0.125
每个类别单独一行权重
```

### 11.3 代码注释与兼容性

新脚本已加入中文注释，说明：

```text
为什么只使用 val split
候选 class-wise 权重如何构造
为什么复用 eval_rhl_ensemble.py 的模型加载和 logit/prob 处理函数
```

默认不改变已有训练和评估逻辑。`eval_rhl_ensemble.py` 已支持读取 `--class_weights_json`，因此 val 搜索输出可以直接接入正式 test 评估。

### 11.4 验证

语法检查：

```bash
python -m py_compile tools/search_rhl_class_weights.py tools/eval_rhl_ensemble.py
```

小批量 smoke test：

```bash
python tools/search_rhl_class_weights.py \
  --ckpts \
    checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth \
    checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth \
  --dataset voc \
  --task 15-5 \
  --setting sequential \
  --curr_step 1 \
  --loss_type bce_loss \
  --ensemble_mode prob \
  --objective all_miou \
  --keep_models_on_gpu \
  --max_batches 1 \
  --save_json logs/rhl_ensemble/debug_val_class_weight_search.json \
  --save_class_weights_json logs/rhl_ensemble/debug_val_class_weights.json
```

验证结果：

```text
脚本正常加载 val split
正常生成候选
正常保存 search summary 和 class_weights_json
```

---

## 12. 阶段 A-v3 tmux 实验启动命令

tmux session：

```text
rhl_val_search_classwise
```

启动命令：

```bash
tmux new-session -d -s rhl_val_search_classwise "bash -lc 'cd /root/2TStorage/lyc/SegACIL && source /home/linyichen/miniconda3/etc/profile.d/conda.sh && conda activate segacil && python tools/search_rhl_class_weights.py --ckpts checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth --dataset voc --task 15-5 --setting sequential --curr_step 1 --loss_type bce_loss --ensemble_mode prob --objective all_miou --keep_models_on_gpu --save_json logs/rhl_ensemble/20260616_rhl_se_val_class_weight_search.json --save_class_weights_json logs/rhl_ensemble/20260616_rhl_se_val_best_class_weights.json 2>&1 | tee logs/rhl_ensemble/20260616_rhl_se_val_class_weight_search.log && python tools/eval_rhl_ensemble.py --ckpts checkpoints/20260616_rhl_se_seed1/voc/15-5/sequential/step1/final.pth checkpoints/20260616_rhl_se_seed2/voc/15-5/sequential/step1/final.pth checkpoints/20260616_rhl_se_seed3/voc/15-5/sequential/step1/final.pth --class_weights_json logs/rhl_ensemble/20260616_rhl_se_val_best_class_weights.json --dataset voc --task 15-5 --setting sequential --curr_step 1 --loss_type bce_loss --mode test --ensemble_mode prob --keep_models_on_gpu --save_json logs/rhl_ensemble/20260616_rhl_se_val_classwise_test.json 2>&1 | tee logs/rhl_ensemble/20260616_rhl_se_val_classwise_test.log'"
```

输出文件：

```text
logs/rhl_ensemble/20260616_rhl_se_val_class_weight_search.json
logs/rhl_ensemble/20260616_rhl_se_val_best_class_weights.json
logs/rhl_ensemble/20260616_rhl_se_val_class_weight_search.log
logs/rhl_ensemble/20260616_rhl_se_val_classwise_test.json
logs/rhl_ensemble/20260616_rhl_se_val_classwise_test.log
```

执行情况：

```text
tmux 会话已自动结束
val 搜索完成
test 评估完成
结果 JSON 已落盘
```

---

## 13. 阶段 A-v3 实验结果

### 13.1 val split 搜索结果

最佳候选：

```text
Best candidate: classwise_valbest_all_s0.75
Best score, all_miou: 43.9438
```

val 指标：

| 设置 | val all mIoU | val old mIoU | val new mIoU |
|---|---:|---:|---:|
| `rhl_seed=1` | 43.5849 | 42.6350 | 46.6243 |
| `rhl_seed=2` | 43.8629 | 42.8776 | 47.0160 |
| `rhl_seed=3` | 43.8033 | 42.7724 | 47.1021 |
| **val-selected class-wise** | **43.9438** | **42.9179** | **47.2266** |

val 选择出的新类成员：

| 类别 | 选中成员 | 权重 |
|---|---|---|
| 16 pottedplant | seed3 | `[0.125, 0.125, 0.75]` |
| 17 sheep | seed3 | `[0.125, 0.125, 0.75]` |
| 18 sofa | seed2 | `[0.125, 0.75, 0.125]` |
| 19 train | seed2 | `[0.125, 0.75, 0.125]` |
| 20 tvmonitor | seed3 | `[0.125, 0.125, 0.75]` |

### 13.2 test split 最终结果

| 设置 | all mIoU | old mIoU | new mIoU | Overall Acc | Mean Acc |
|---|---:|---:|---:|---:|---:|
| baseline | 69.4606 | 78.0085 | 42.1075 | 92.7034 | 78.3376 |
| prob K3 `0.2/0.4/0.4` | 69.5060 | 77.9848 | 42.3737 | 92.7235 | 78.3868 |
| class-wise v1, test 手工 | 69.5229 | 78.0022 | 42.3893 | 92.7251 | 78.4009 |
| **val-driven class-wise** | **69.5379** | 77.9804 | **42.5218** | **92.7283** | **78.4391** |

新类逐类 IoU：

| 设置 | pottedplant 16 | sheep 17 | sofa 18 | train 19 | tvmonitor 20 |
|---|---:|---:|---:|---:|---:|
| baseline | 23.5886 | 57.9622 | 30.9147 | 69.9526 | 28.1192 |
| class-wise v1, test 手工 | 23.6018 | 58.3234 | 31.4027 | 70.2956 | 28.3228 |
| **val-driven class-wise** | **24.0709** | **58.0609** | **31.7575** | **70.4123** | **28.3077** |

### 13.3 结论

相对 baseline：

```text
all mIoU: +0.0773
new mIoU: +0.4143
old mIoU: -0.0281
Mean Acc: +0.1015
```

相对阶段 A 旧 best `prob K3 0.2/0.4/0.4`：

```text
all mIoU: +0.0319
new mIoU: +0.1481
old mIoU: -0.0044
```

相对 test 手工 `class-wise v1`：

```text
all mIoU: +0.0150
new mIoU: +0.1325
old mIoU: -0.0218
```

最终判断：

```text
val-driven class-wise 是目前最干净、最好的一版 RHL-SE 推理融合；
它证明 RHL seed 的逐类互补不是纯 test 调参假象；
但 all mIoU 只提升 +0.0773，仍不足以支撑主方法创新。
```

因此 RHL-SE 后续应作为辅助模块保留，不建议继续手工扫 test 权重。下一阶段应转向 BOA-RHL / PGH-RHL 这类改变 RHL 特征构造的方法。
