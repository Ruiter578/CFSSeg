# Buffer 超参数搜索实验方案（gamma 固定为 1）

更新日期：2026-07-06

## 1. 当前协议调整

从 2026-07-06 起，后续实验固定：

```text
gamma = 1
```

不再继续 `gamma=0.1/10` 的搜索。已经完成的 `gamma=0.1/10` 结果保留为历史参考，但不进入后续 top 排名和 Phase B 选择。

本轮目标改为：在 `20260621_baseline_bs16_16_trs` step0 权重基础上，只搜索 VOC `15-5 sequential` step1 的 RHL `buffer`，并用 `rhl_seed` 检查候选 buffer 的随机子空间稳定性。

## 2. 固定条件

```text
MODEL=deeplabv3_resnet101
AIR_FEATURE_SOURCE=auto
DATA_ROOT=/TRS-SAS/linwei/SegACIL/data_root/VOC2012
TASK=15-5
SETTING=sequential
START_STEP=1
END_STEP=1
BASE_SUBPATH=20260621_baseline_bs16_16_trs
DEFAULT_BATCH_SIZE=32
SPECIAL_BATCH_SIZE=32
RANDOM_SEED=1
GAMMA=1
RHL_NORM=none
RHL_STATS=0
CUDA_VISIBLE_DEVICES=2
```

base checkpoint 固定为：

```text
checkpoints/20260621_baseline_bs16_16_trs/voc/15-5/sequential/step0/deeplabv3_resnet101_voc_15-5_step_0_sequential.pth
```

已记录 SHA256：

```text
fb48c926cde03a35c1daf7ec6b9fe95340e932ddf5c4c2226a9c432f87fa244e
```

## 3. Phase A：buffer 主搜索

固定 `gamma=1`、`random_seed=1`、`rhl_seed=-1`，搜索：

```text
buffer = 8192, 8196, 8200, 8204, 8208, 8212, 8216, 8220, 8224
```

说明：

- `8192`：论文默认值，作为硬基线。
- `8196`：历史结果中 `16-20 mIoU` 较强。
- `8200`：补齐 8196 到 8204 之间的点，之前该点缺少 manifest。
- `8204`：历史 all mIoU 第二梯队。
- `8208`：历史 all mIoU 最高候选。
- `8212/8216`：验证右侧下降是否稳定。
- `8220/8224`：继续向右扩边，判断是否存在新的局部峰。

队列脚本会跳过已经完成且同名的 `gamma=1` 实验。

## 4. Phase B：RHL seed 稳定性验证

Phase A 完成后，从 `gamma=1` 的 all mIoU 排名中选 top2 buffer，再固定 `gamma=1`、`random_seed=1`，改变：

```text
rhl_seed = 1, 2, 3
```

这一步回答：top buffer 是稳定好，还是 `rhl_seed=-1` 的随机投影碰巧好。

## 5. 不再执行的方向

本轮不再跑：

- `gamma=0.1/10/0.3/3`：gamma 固定为 1。
- `random_seed` 多 seed：它会混入 dataloader、augmentation、全局随机性，不适合和 RHL 子空间稳定性混在这一轮。
- `15-1`、`disjoint`、`overlap`：当前目标只确定 `15-5 sequential` step1 的本地最优候选。

## 6. 启动与查看

队列脚本：

```text
Codex_Plans/run_buffer_gamma_search_trs_queue.sh
```

tmux 会话：

```text
705_segacil_bgamma
```

推荐启动命令：

```bash
cd /TRS-SAS/linwei/SegACIL
tmux new-session -d -s 705_segacil_bgamma \
  'cd /TRS-SAS/linwei/SegACIL && bash Codex_Plans/run_buffer_gamma_search_trs_queue.sh 2>&1 | tee -a logs/20260705_buffer_gamma_search_trs/queue_master.log'
```

查看：

```bash
tmux attach -t 705_segacil_bgamma
tail -f /TRS-SAS/linwei/SegACIL/logs/20260705_buffer_gamma_search_trs/queue_master.log
```

## 7. 输出

summary：

```text
Codex_Plans/20260705_buffer_gamma_search_trs_summaries/phaseA_summary.tsv
Codex_Plans/20260705_buffer_gamma_search_trs_summaries/final_summary.tsv
Codex_Plans/20260705_buffer_gamma_search_trs_summaries/final_mean_by_combo.tsv
```

单个实验：

```text
checkpoints/<SUBPATH>/voc/15-5/sequential/step1/run_manifest.json
checkpoints/<SUBPATH>/voc/15-5/sequential/step1/test_results_*.json
```

## 8. 判定规则

Phase A：

1. 只比较 `gamma=1`。
2. 以 all mIoU 作为主排序。
3. 若 all mIoU 差距小于 0.05 个百分点，优先看 `16-20 mIoU`。

Phase B：

1. 对 top2 buffer 的 `rhl_seed=1,2,3` 结果求 mean/std。
2. 最终推荐优先选 mean all mIoU 更高的 buffer。
3. 如果 mean all mIoU 差距小于 0.05 个百分点，优先选 std 更低且 `16-20 mIoU` 不差的 buffer。
4. 如果 Phase B 排名与 Phase A 排名冲突，说明 `rhl_seed=-1` 单次结果不够稳，不能只按单次最高点定默认值。

## 9. 完成后汇报

实验结束后至少汇报：

1. Phase A 的 `gamma=1` 排名。
2. Phase B top2 的 mean/std。
3. 最终推荐 buffer。
4. 是否建议把 `run_trs.sh` 默认 `BUFFER` 更新到该值。
5. 是否需要继续在推荐 buffer 左右做二次细扫。

