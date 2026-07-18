# 20260718 buffer 后 clean validation 复盘与 Git 处理方案

## 1. 结论先行

当前 `main` 工作区是干净的，`feature/cfsseg-code3d-integration` 已经合入并删除，本轮混乱主要来自两个来源：

1. buffer 之后的 2D 搜索从原先的 `test_results_*.json` 改成了 `val_results_*.json`，这是为了避免继续用最终 test 集反复选超参数。
2. 20260717 的 clean replay v2 八组结果虽然跑完并已归档，但所有 manifest 都记录 `git.dirty=true`，因此只能作为历史参考，不能作为最终超参数选择证据。

最简洁有效方案：

1. 保留 `main` 上已经合入的 clean validation 源码、split 生成工具、测试和协议文档。
2. 不把 20260717 dirty replay v2 当成最终排名依据。
3. 从当前干净 `main` 重新跑同一 8 组矩阵，要求新 run 的 `run_manifest.json` 中 `git.dirty=false`。
4. 新 clean replay 成功后，删除旧 dirty replay v2 的 tracked 结果和 dirty patch 归档，再删除 `.codex-worktrees`、`codex/clean-validation-protocol` 分支和 `.codex-backups`。

## 2. 为什么 buffer 后要做 step0 和 replay

### 2.1 原 buffer 阶段

到 `20260711_buffer_final_confirmation_trs_b8232_g1_rhl5_trs` 为止，实验保存方式是：

```text
checkpoints/<SUBPATH>/voc/15-5/sequential/step1/
  run_manifest.json
  test_results_*.json
```

这些实验使用官方 VOC val/test 口径做最终评估，然后从多个 `test_results_*.json` 中选择更好的 buffer。这适合复现和确认，但不适合继续做大量超参数搜索，因为会把最终评估集变成调参集。

### 2.2 `20260712_CFSSeg可调超参数审计与分阶段搜索路线.md` 的安排

该文档明确提出：

- 后续搜索主指标改为 validation All mIoU。
- 最终 test 只在配置冻结后评估一次。
- E1 低成本闭式阶段搜索优先于昂贵 step0 搜索。
- E1.1 先比较 `air_feature_source={decoder,aspp}`。
- 如果 `aspp` 胜出，再局部复核 buffer `{8208,8224,8240}`。
- 如果 `decoder` 明确胜出，则保留当前 buffer 结论，不重扫。

因此 buffer 之后本来应该进入：

```text
E1.1 AIR feature source:
  decoder/aspp x rhl_seed 2,4,5

条件分支:
  若 aspp 胜出 -> aspp buffer 8208/8224/8240 局部复核
  若 decoder 胜出 -> 进入 analytic_tail_epsilon 或 gamma
```

### 2.3 为什么 20260712 的 val_results 仍有争议

`PLAN_E1_1_AIR特征源搜索执行_20260712.md` 当时只做了 validation-only 接线：

```text
EVALUATION_MODE=val -> val_results_*.json
EVALUATION_MODE=test -> test_results_*.json
```

但是旧的 `val` 数据集来自任务过滤后的 step1 validation 子集，它不是一个完整、独立的调参验证集。用户提到的 1、4、10、13 类在 JSON 顶层不是单独字段；它们存在于 `Class IoU` / `Class Acc` 子字典中，但值为 `0.0`。这说明问题不是 JSON 写漏字段，而是旧 validation 协议本身会产生不完整或分布很差的类别评估。

`full_test_audit_20260715.json` 是后来补做的官方 test 审计，它能解释最终 test 表现，但不能拿来继续调参。

### 2.4 为什么需要新的 clean step0

`Codex_Plans/20260716_clean_validation_protocol_design.md` 进一步修正协议：

- 从 `datasets/data/voc/train_cls.txt` 中确定性划出 10% holdout。
- holdout 从 step0 和后续所有训练 loader 中排除。
- holdout 专门作为内部 validation。
- 官方 VOC `ImageSets/Segmentation/val.txt` 继续保留为最终 test，不参与候选排序。

这就导致一个关键约束：旧的 step1 候选不能直接用新 holdout 重新排名，因为它们训练时已经看过完整 `train_cls.txt`，包括后来划出来的 holdout。

所以必须先重训一个 holdout-excluded 的 step0 baseline，然后从这个 step0 出发重新生成 step1 候选。这就是 `20260716_clean_validation_step0_decoder_b8224_g1_gpu2_bs16` 的目的。

### 2.5 为什么是 8 组 replay

clean step0 完成后，应该重跑原 E1.1 和 buffer 局部复核矩阵：

| 组别 | 目的 |
|---|---|
| `decoder_b8224_rhl2` | decoder 初筛 |
| `decoder_b8224_rhl4` | decoder seed 确认 |
| `decoder_b8224_rhl5` | decoder seed 确认 |
| `aspp_b8224_rhl2` | aspp 初筛 |
| `aspp_b8224_rhl4` | aspp seed 确认 |
| `aspp_b8224_rhl5` | aspp seed 确认 |
| `aspp_b8208_rhl2` | 如果 aspp 可能胜出，检查 buffer 左邻域 |
| `aspp_b8240_rhl2` | 如果 aspp 可能胜出，检查 buffer 右邻域 |

这 8 组不是新的方法实验，而是把 20260712/20260714 那些“不够干净的 validation 搜索”按新 holdout 协议重放一遍。

## 3. 当前进度

### 3.1 已确认的 buffer 结论

当前主线记录的 buffer 阶段结论仍是：

```text
buffer=8224
gamma=1
rhl_norm=none
```

这个结论来自旧 buffer 搜索和最终确认实验；它是进入 clean replay 的起点，不是所有后续参数的最终冻结配置。

### 3.2 clean step0 状态

当前 `main` 只保留成功的 bs16 step0 manifest：

```text
checkpoints/20260716_clean_validation_step0_decoder_b8224_g1_gpu2_bs16/
```

bs32 和 bs64 两个失败尝试已经通过提交 `cb871cf` 从 tracked 文件中删除。

### 3.3 dirty replay v2 状态

当前 8 组 v2 replay 结果都存在，指标如下：

| run | All mIoU | Old mIoU | New mIoU | manifest dirty |
|---|---:|---:|---:|---|
| `aspp_b8208_rhl2` | 0.588629433 | 0.666720802 | 0.338737052 | true |
| `aspp_b8224_rhl2` | 0.590665958 | 0.666892365 | 0.346741456 | true |
| `aspp_b8224_rhl4` | 0.591287260 | 0.666727323 | 0.349879060 | true |
| `aspp_b8224_rhl5` | 0.591789538 | 0.668617881 | 0.345938842 | true |
| `aspp_b8240_rhl2` | 0.587932822 | 0.665407345 | 0.340014349 | true |
| `decoder_b8224_rhl2` | 0.597318397 | 0.676605949 | 0.343598229 | true |
| `decoder_b8224_rhl4` | 0.596577512 | 0.675590397 | 0.343736281 | true |
| `decoder_b8224_rhl5` | 0.595376074 | 0.675721075 | 0.338272072 | true |

这些结果提示 `decoder` 可能优于 `aspp`，但由于启动时 tracked worktree 是 dirty 状态，不能作为最终选择证据。

## 4. 源码改动 code review

### 4.1 总体判断

未发现保存目录结构被错误修改。默认行为仍是：

```text
Config.evaluation_mode = "test"
Trainer.evaluation_modes("test") = ("test",)
Trainer.evaluation_result_prefix("test") = "test_results"
```

也就是说，不显式设置 `EVALUATION_MODE=val` 时，旧式实验仍然写：

```text
checkpoints/<SUBPATH>/voc/15-5/sequential/step*/test_results_*.json
```

新逻辑只在显式传入以下参数时启用：

```text
EVALUATION_MODE=val
TRAIN_EXCLUDE_LIST=<holdout list>
VALIDATION_LIST=<holdout list>
```

### 4.2 文件级审查

| 文件 | 改动 | 是否影响默认旧逻辑 | 审查结论 |
|---|---|---|---|
| `datasets/init_dataset.py` | 增加 `validate_clean_validation_lists()`；如果设置 `validation_list`，要求同时设置 `train_exclude_list`；validation loader 改用 `tuning_val` | 不影响。`validation_list=None` 时仍使用旧 `val` | 合理，防止 holdout 被训练集泄漏 |
| `datasets/voc.py` | 增加读取 image id list、排除训练 id、`tuning_val` 分支 | 不影响。默认 `image_set=train/val/test` 仍走原路径 | 合理，clean validation 的最小必要改动 |
| `run.sh` | 暴露 `BACKBONE_LR`、`CLASSIFIER_LR`、`ANALYTIC_TAIL_EPSILON`、`EVALUATION_MODE`、`TRAIN_EXCLUDE_LIST`、`VALIDATION_LIST` | 不影响。默认 `EVALUATION_MODE=test`，list 为空 | 合理。只是把已有/待搜索参数传到 Python |
| `utils/parser.py` | 给 Config/CLI 增加上述参数，默认保持旧值 | 不影响。默认 `test`、list 为 `None` | 合理。默认兼容 |
| `utils/run_manifest.py` | manifest 记录 LR、tail epsilon、evaluation mode、list 路径和 sha256 | 不影响训练和保存，只扩展 provenance | 合理。schema 变宽但旧读者仍可读 flat key |
| `tools/create_voc_tuning_split.py` | 新增确定性 holdout 生成工具 | 不影响运行时，只有显式调用才执行 | 合理。提供可复现 split |
| `tests/test_voc_tuning_split.py` | 覆盖 split 解析、确定性、覆盖性、元数据 hash | 不影响运行时 | 合理 |
| `tests/test_voc_validation_contract.py` | 覆盖 validation list 必须从 training 中排除 | 不影响运行时 | 合理 |
| `.gitignore` | 允许 clean protocol JSON 和 3D 轻量结果 JSON 被 Git 跟踪 | 不影响训练和保存 | 合理，但后续应避免把 dirty replay 当最终结果提交 |

### 4.3 保存机制审查

保存机制仍在 `trainer/trainer.py` 中：

```text
self.root_path / self.root_path0 不变
result_prefix = evaluation_result_prefix(mode)
mode="test" -> test_results_*.json
mode="val"  -> val_results_*.json
```

因此，`val_results_*.json` 的出现来自显式 validation-only 实验设置，不是结果路径被改坏。

### 4.4 已运行验证

本次复盘已运行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
bash -n run.sh Codex_Plans/20260716_clean_validation_protocol/run_clean_step0_baseline_gpu2.sh Codex_Plans/20260717_clean_e1_1_replay/run_clean_e1_1_replay_gpu2.sh
git diff --check
python -m compileall datasets utils tools trainer tests
coderabbit review --agent -t uncommitted --dir /TRS-SAS/linwei/SegACIL
coderabbit review findings --dir /TRS-SAS/linwei/SegACIL
```

结果：

```text
unittest: 39 tests OK
bash -n: OK
git diff --check: OK
compileall: OK
CodeRabbit: 新增文档 diff 未被 CLI 识别为可审查文件；历史 findings 主要是 CFSSeg-code3D 的独立问题，不属于本次 2D clean validation 保存机制改动。
```

## 5. 争议文件和临时目录怎么处理

### 5.1 `.codex-worktrees`

`/TRS-SAS/linwei/.codex-worktrees` 是为了避免在主工作区有大量 untracked 结果时直接切分支而创建的旁路 worktree。它不是必须长期保留的项目结构。

当前主要内容：

```text
SegACIL-clean-validation            -> codex/clean-validation-protocol 分支，保留了当时 dirty launch 状态
SegACIL-aspp-buffer8208             -> 旧 detached worktree
SegACIL-aspp-buffer8240             -> 旧 detached worktree
SegACIL-e11-aspp5                   -> 旧 detached worktree
SegACIL-e11-decoder5                -> 旧 detached worktree
SegACIL-e11-decoder5b               -> 旧 detached worktree
SegACIL-e11-fixed                   -> 旧 detached worktree
```

这些 detached worktree 里显示为 untracked 的 20260712/20260714 结果，已经在当前 `main` 中作为 tracked 文件存在。确认没有 tmux 任务占用后，可以删除这些 worktree。

### 5.2 `codex/clean-validation-protocol` 分支

这个分支的作用只是当时隔离 clean validation 接线和启动脚本。它的核心改动已经进入 `main`，并且该分支现在还被 `/TRS-SAS/linwei/.codex-worktrees/SegACIL-clean-validation` 占用。

推荐处理：

1. 先删除对应 worktree。
2. 再删除本地分支 `codex/clean-validation-protocol`。
3. 如果远端存在同名分支，再删除远端。

当前远端只看到 `origin/main`，没有 `feature/cfsseg-code3d-integration` 或 `codex/clean-validation-protocol`。

### 5.3 `.codex-backups`

`/TRS-SAS/linwei/.codex-backups` 是为了防止切分支覆盖 untracked 文件而做的安全备份，不是实验主路径。

当前它只有约 356K，主要包含：

- 20260712/20260714 旧 AIR/ASPP validation JSON/summary 的副本。
- 20260716 clean step0 manifest 副本。
- 20260717 dirty replay v2 JSON/manifest 副本。
- 一个非 v2 的争议单跑目录副本。
- 少量实验评估教学 markdown/html/css。

这些文件若已经在 `main` 中 tracked，备份可删除。建议等新 clean replay 完成并确认不需要追溯 dirty replay 后，再统一删除整个 `.codex-backups`。

## 6. 最终 Git 处理方案

### 6.1 立即状态

当前状态应保持：

```text
main clean
feature/cfsseg-code3d-integration 已合并并删除
clean validation 源码和测试保留在 main
dirty replay v2 不作为最终结论
```

### 6.2 推荐下一步

先从当前 clean `main` 重跑 8 组 replay，新的 run id 建议使用：

```text
20260718_clean_e1_1_replay_from_main_gpu2_<feature>_b<buffer>_g1_rhl<seed>
```

新 launcher 必须在启动前检查：

```bash
git status --porcelain
git rev-parse HEAD
```

启动后每组检查：

```bash
python - <<'PY'
import json
from pathlib import Path
for p in Path("checkpoints").glob("20260718_clean_e1_1_replay_from_main_gpu2_*/voc/15-5/sequential/step1/run_manifest.json"):
    m = json.loads(p.read_text())
    print(p, m["git"]["dirty"], m["git"]["commit"], m["evaluation_mode"], m["train_exclude_list"], m["validation_list"])
PY
```

验收条件：

```text
git.dirty == false
evaluation_mode == val
train_exclude_list 指向 20260716 holdout list
validation_list 指向同一个 holdout list
8 组 step1 都有 val_results_*.json
summary 能正常汇总 8 行，不只写表头
```

### 6.3 新 clean replay 成功后的删除提交

新 clean replay 完成并提交后，删除旧 dirty replay v2：

```bash
cd /TRS-SAS/linwei/SegACIL
git rm -r checkpoints/20260717_clean_e1_1_replay_v2_gpu2_*
git rm -r Codex_Plans/20260717_clean_e1_1_replay_v2_gpu2_summaries
git commit -m "20260718 results: remove dirty clean replay archive after clean rerun"
```

如果希望先立即简化，也可以现在删除旧 dirty replay v2，但更稳妥的是等 clean rerun 成功后再删。

### 6.4 删除 worktree 和临时分支

确认没有 tmux 任务后执行：

```bash
cd /TRS-SAS/linwei/SegACIL

git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-clean-validation --force
git branch -d codex/clean-validation-protocol

git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-aspp-buffer8208 --force
git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-aspp-buffer8240 --force
git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-e11-aspp5 --force
git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-e11-decoder5 --force
git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-e11-decoder5b --force
git worktree remove /TRS-SAS/linwei/.codex-worktrees/SegACIL-e11-fixed --force

git worktree prune
```

删除后检查：

```bash
git worktree list
git branch --list '*clean-validation*' '*cfsseg-code3d*'
git branch -r --list '*clean-validation*' '*cfsseg-code3d*'
```

预期只剩主工作区，且没有这些临时分支。

### 6.5 删除备份

新 clean replay 成功、旧 dirty replay 删除提交完成、worktree 删除后，可以删除备份：

```bash
rm -rf /TRS-SAS/linwei/.codex-backups
rm -rf /TRS-SAS/linwei/.codex-worktrees
```

如果还有不放心的教学文档或单跑 manifest，可以先人工查看 `.codex-backups`，但从当前 repo 状态看，里面大多数是已经 tracked 的重复副本。

## 7. 后续实验路线

推荐不要基于 dirty replay v2 做最终决策。新的干净流程应是：

1. 重新跑 clean 8 组 replay。
2. 如果 clean 结果仍显示 `decoder` 明确优于 `aspp`，E1.1 结束，保留 `decoder` 和 `buffer=8224`。
3. 进入 E1.2 `analytic_tail_epsilon` 搜索：`0`、`1e-4`、`1e-3`。
4. 再进入 E1.3 gamma 搜索。
5. 最后配置冻结后，只跑一次官方 test，生成新的 `test_results_*.json`。

这样仓库会回到简单状态：

```text
main
  源码/脚本/文档/轻量 JSON

checkpoints/
  只保留有结论价值的 run_manifest.json 和 results JSON

无 feature/cfsseg-code3d-integration
无 codex/clean-validation-protocol
无 .codex-worktrees
无 .codex-backups
```
