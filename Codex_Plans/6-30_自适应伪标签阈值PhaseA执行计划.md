# 自适应伪标签阈值 Phase A 执行计划

生成时间：2026-06-30
目标分支：`feature/adaptive-pseudo-label`
执行范围：阶段一只实现 Phase A grid 基础设施，并启动 fixed threshold sweep 第一组实验。

## 1. 当前事实

第一轮 `15-5 overlap step1` 四组结果：

| strategy | all mIoU | old | new | accepted ratio |
| --- | ---: | ---: | ---: | ---: |
| `off` | 70.3091 | 79.1179 | 42.1209 | - |
| `batch_global q0.7` | 70.3257 | 79.1423 | 42.1124 | 30.0003% |
| `batch_class q0.7` | 70.4535 | 79.2941 | 42.1636 | 30.0008% |
| `fixed0.7` | **70.7383** | **79.6346** | **42.2703** | **90.3503%** |

共同 base checkpoint：

```text
checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth
sha256 = 6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913
```

当前判断：

```text
fixed0.7 是当前强 baseline。
batch_global / batch_class q0.7 过保守。
下一步先搜索 fixed threshold，再搜索低 q adaptive。
```

## 2. 阶段一交付物

新增：

```text
tools/run_pseudo_label_grid.sh
tools/summarize_pseudo_label_grid.py
configs/pseudo_label_phaseA_fixed_sweep.tsv
```

可选新增测试：

```text
tests/test_pseudo_label_grid_summary.py
```

阶段一不实现：

```text
tools/audit_pseudo_label_quality.py
15-1/10-1 full protocol runner
artifact threshold training path 重构
```

这些进入阶段二或阶段三。

## 3. grid runner 设计

`tools/run_pseudo_label_grid.sh` 读取 TSV，每一行是一组实验。

必需字段：

```text
name
subpath
task
setting
strategy
confidence
quantile
min_conf
max_conf
min_pixels
shrinkage
margin_min
base_subpath
skip_step0
batch_size
step0_batch_size
buffer
gamma
random_seed
model
air_feature_source
```

行为：

1. 默认 `--mode dry-run` 只打印命令。
2. `--mode run` 顺序执行每一行。
3. 每行调用 `tools/run_adaptive_pseudo_label.sh`，不直接拼 `train.py`。
4. 若输出目录已有 `test_results_*.json`，默认跳过。
5. 若输出目录存在但没有结果，默认报错，避免覆盖半截实验。
6. 支持 `SKIP_EXISTING=0` 强制重跑，但正式实验不要默认使用。

## 4. grid summarizer 设计

`tools/summarize_pseudo_label_grid.py` 支持：

```bash
python tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_phaseA_fixed_sweep.tsv \
  --output-md logs/pseudo_label/phaseA_fixed_sweep_summary.md \
  --output-csv logs/pseudo_label/phaseA_fixed_sweep_summary.csv
```

输出字段：

```text
name
subpath
task
setting
strategy
confidence
quantile
all_miou
old_miou
new_miou
overall_acc
mean_acc
candidate_count
accepted_count
accepted_ratio
base_checkpoint_sha256
teacher_sha256
result_json
stats_json
status
```

阈值比较：

```text
fixed0.7 baseline all mIoU = 70.7383
off baseline all mIoU = 70.3091
```

## 5. 第一组实验：fixed sweep

配置文件：

```text
configs/pseudo_label_phaseA_fixed_sweep.tsv
```

包含 3 组：

| name | strategy | confidence | 目的 |
| --- | --- | ---: | --- |
| `fixed0p6` | `fixed` | 0.6 | paper-equivalent tau=0.4 |
| `fixed0p8` | `fixed` | 0.8 | 判断 fixed0.7 是否最优 |
| `fixed0p9` | `fixed` | 0.9 | 高精度低召回边界 |

统一配置：

```text
TASK=15-5
SETTING=overlap
BASE_SUBPATH=20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32
SKIP_STEP0=1
BATCH_SIZE=32
STEP0_BATCH_SIZE=32
BUFFER=8196
GAMMA=1
RANDOM_SEED=1
MODEL=deeplabv3_resnet101
AIR_FEATURE_SOURCE=auto
```

建议 subpath：

```text
20260630_pseudo_15-5_overlap_fixed0p6_seed1_bs32_reuse20260627step0
20260630_pseudo_15-5_overlap_fixed0p8_seed1_bs32_reuse20260627step0
20260630_pseudo_15-5_overlap_fixed0p9_seed1_bs32_reuse20260627step0
```

## 6. 启动方式

先 dry run：

```bash
cd /root/2TStorage/lyc/SegACIL
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_phaseA_fixed_sweep.tsv \
  --mode dry-run
```

正式启动 tmux：

```bash
tmux new-session -d -s apl_phaseA_fixed \
  "cd /root/2TStorage/lyc/SegACIL && \
   PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
   CUDA_VISIBLE_DEVICES=0 \
   CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass \
   TMPDIR=/root/2TStorage/tmp \
   bash tools/run_pseudo_label_grid.sh \
     --grid configs/pseudo_label_phaseA_fixed_sweep.tsv \
     --mode run"
```

## 7. 验收命令

代码检查：

```bash
bash -n tools/run_pseudo_label_grid.sh
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile tools/summarize_pseudo_label_grid.py
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest discover -s tests -p 'test*.py' -v
git diff --check
```

CodeRabbit：

```bash
coderabbit review --agent -t uncommitted --dir /root/2TStorage/lyc/SegACIL
```

实验完成后汇总：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python tools/summarize_pseudo_label_grid.py \
  --grid configs/pseudo_label_phaseA_fixed_sweep.tsv \
  --output-md logs/pseudo_label/phaseA_fixed_sweep_summary.md \
  --output-csv logs/pseudo_label/phaseA_fixed_sweep_summary.csv
```

## 8. 判定标准

固定阈值搜索：

```text
如果 fixed0.6 > fixed0.7：
    论文等价 tau=0.4 更强，后续 paper protocol 应优先用 fixed0.6。
如果 fixed0.8 / fixed0.9 > fixed0.7：
    当前 fixed0.7 不是最优，后续 adaptive 要超过新的 best fixed。
如果 fixed0.7 仍最好：
    fixed0.7 是当前强 baseline，adaptive sweep 必须超过 70.7383。
```

进入下一阶段条件：

```text
fixed sweep 完成并汇总后，再启动 adaptive low-q sweep。
```

## 9. 风险控制

- 不要复用已有 SUBPATH。
- 不要覆盖已有 `checkpoints`。
- 不要把未完成 tmux 实验当结论。
- 如果 summary 失败但 checkpoint 和 test JSON 已生成，先修 summary，不重训。
- 所有结果必须标明是 `15-5 overlap`，不能直接和论文 `15-1 overlapped 69.36` 对比。
