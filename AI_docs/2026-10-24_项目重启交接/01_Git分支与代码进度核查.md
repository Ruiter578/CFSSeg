# Git、worktree 与代码进度核查

本文件保留10月24日核查快照。11月分支整合后的状态见[Git整合报告](../代码改动报告/2026-11-19_Git分支整合与项目整理报告.md)。

核查日期：2026-10-24（Asia/Shanghai）。范围：本地两个 worktree、全部本地/远端跟踪分支、Git 对象中的文档与结果，以及通过 `git ls-remote --heads --tags origin` 核验的真实远端引用。本轮没有切换、合并、删除分支，没有提交、推送或启动训练。

## 1. 上次停在哪里

| 对象 | 核实结果 | 含义 |
|---|---|---|
| 最新代码仓库提交 | `main@087fa5e2a64975e5f09bfce501e12c7485bdb63d` | 提交的是 E1.2 tail epsilon 的结果与汇总，**没有新算法代码** |
| Author 时间 | 2026-07-20 12:28:34 +08:00 | Git 原始值 `2026-07-20T04:28:34+00:00` |
| Committer 时间 | 2026-07-20 12:30:25 +08:00 | Git 原始值 `2026-07-20T04:30:25+00:00`；区分作者时间与落库时间 |
| 全分支最近文档提交 | `feature/rhl-next@804172c`，2026-07-20 00:37:11 +08:00 | `7-19_方案C分支收口与RHL迁移严格审查报告.md` |
| main 最近一批 AI_docs | `72bf9cc`，2026-07-20 00:21:35 +08:00；此前 `d56d88f`、`f8a7a4c` | 伪标签五份结论归档、clean replay 结果、两份阶段总结与 CSV |
| 最后可定位的二维结果 | `val_results_20260719_212525.json` | E1.2，epsilon=1e-4，step1；文件名时区未标明，不擅自换算 |
| 最后结果的 all/old/new | **59.714327 / 67.617340 / 34.424687%** | 内部 clean holdout，不能和历史官方 val-as-test 的约 70% 直接比较 |
| 文件修改时间 | 多数为 2026-07-24 | 呈批量复制/checkout 特征；不能用它认定最后实验发生在 7 月 24 日 |

最后一批 E1.2：epsilon=0 的 all 为 59.711857%，epsilon=1e-4 为 59.714327%，复用 epsilon=1e-3 基线为 59.731028%。两次尝试都没有超过基线，差距仅约 0.02 pp。E1.3 新协议 gamma 搜索尚无完成证据。完整结果见 [详细回顾](03_方法与实验完整回顾.md)。

## 2. 分支清单：4 个本地、5 个远端

下面的“main 独有 / 该分支独有”来自 `git rev-list --left-right --count main...<ref>`。它衡量提交祖先关系，**不是功能完成度**；cherry-pick、squash 和人工移植会产生不同 SHA。

| 分支 | 本地 HEAD | 远端 HEAD | main 独有 / 远端分支独有 | 实际内容与建议 |
|---|---|---|---:|---|
| `main` | `087fa5e` | `087fa5e` | 0 / 0 | 长期基线与结果入口，保留 |
| `feature/rhl-next` | `804172c` | `804172c` | 1 / 7 | 从 `72bf9cc` 迁移 BOA/P0，保留作单模型 RHL 续作候选 |
| `feature/adaptive-pseudo-label` | `c9672f1` | `85d7ea1` | 36 / 31 | 本地落后远端 **17** 个提交；完整 W0/W1 在远端版本，冻结归档 |
| `feature/deeplabv3plus-control` | `7fa27e0` | `7fa27e0` | 71 / 16 | V3+ 历史诊断、class-cap 负实验；主线已选择性吸收有效能力，可归档 |
| `feature/rhl-se-boa-p0-p1` | 无本地分支 | `7701b12` | 64 / 4 | 老 P0/P1 来源；代码迁移至 rhl-next 后不再作为开发起点，可归档 |

远端仓库：[Ruiter578/CFSSeg](https://github.com/Ruiter578/CFSSeg)。2026-10-24 实时 `ls-remote` 的五个 branch HEAD 与本地 remote-tracking refs 一致。没有依赖“上次 fetch 应该是新的”这一假设。

现存冻结标签：`adaptive-pseudo-label-w1-frozen-20260719`，annotated tag 对象 `f1a0b82`，解引用后是 `85d7ea1`。标签对象 SHA 与所指 commit SHA 不同，属于正常现象。

## 3. 两个 worktree

| 路径 | 分支 / HEAD | 核查开始时状态 | 实验现场 |
|---|---|---|---|
| `/data2/lyc/SegACIL` | main / `087fa5e` | tracked 与普通 untracked 均干净；`data_root/` 被忽略 | 有二维 JSON、manifest、TSV 和 3D summary；没有 `.pth/.pt/.ckpt` 权重 |
| `/data2/lyc/SegACIL_rhl_next` | feature/rhl-next / `804172c` | 干净 | 继承历史结果；未找到迁移后的新 BOA 实验结果；也没有模型权重 |

`git worktree list --porcelain` 只列出以上两个。旧文档提到的 `SegACIL_deeplabv3plus`、`SegACIL-clean-validation`、`SegACIL-clean-replay-main-20260718` **当前均不是注册 worktree**。旧 A100 路径 `/root/2TStorage/lyc/SegACIL` 也不等于现工作路径。不能据历史报告说这些目录“仍在运行”。

进程命令行扫描未见 SegACIL 训练任务；这只说明当前主机观察范围，没有远程登录 TRS/A100 检查。保留“开发候选分支”与“当前正在使用/运行”这一区别：没有分支可被确认正在训练。

## 4. main 真正具备什么

| 能力 | 代码证据 | 核实状态 |
|---|---|---|
| 原 CFSSeg 二维链路 | `trainer/trainer.py`、`network/AnalyticLinear.py` | step0 BP；step1 realign + C-RLS；后续递归更新 |
| DeepLabV3 / V3+ | `network/modeling.py`、`network/_deeplab.py` | 均已接通，支持显式 AIR feature source；V3+ auto→aspp_up |
| RHL 固定归一化与独立 seed | `network/Buffer.py` | none/l2/l2_sqrt/layernorm；rhl_seed 隔离随机映射 |
| 旧 RHL 集成 | `tools/eval_rhl_ensemble.py`、`tools/search_rhl_class_weights.py` | logit/prob、全局/类别权重、门控及诊断工具已在 main；不是独立的隐藏分支能力 |
| clean validation | `datasets/init_dataset.py`、`datasets/voc.py`、`tools/create_voc_tuning_split.py` | 独立 holdout、训练排除、val/test/both 与结果前缀分离 |
| tail epsilon、超参数接线 | `utils/parser.py`、`run.sh`、`RecursiveLinear` | 已实现，E1.2 已实验；不是只存在命令行的空参数 |
| 恢复与溯源 | `utils/ckpt.py`、`utils/run_manifest.py`、`tools/run_step0_resume.sh` | checkpoint/resume、manifest、base hash 等工程能力 |
| 3D 旁线 | `CFSSeg-code3D/`、`checkpoints_3d/` | 已合入；S3DIS 8-1 cvfold0 有 summary，all=38.6300%；旧 Home“没有完整3D代码”已过时 |

**main 没有完整 adaptive pseudo-label 工程栈，也没有 BOA 初始化接口。** main 归档了五份伪标签结论，不代表相应生产代码已经合并。

main 原固定伪标签在 `trainer.py` 的 step1 路径仍有 `curr_step > 1` 条件，所以不能在 main 上直接照搬归档分支的 `15-5 step1` 伪标签实验。若将来需要该能力，必须选择性移植并重新验收。

main 中虽然有 `GeneralizedARM` 类定义，但二维默认 AIR 使用 `RecursiveLinear`；没有发现通过现行 runner 接通的 CA-C-RLS 实验链。这是“有旧类定义”与“新方法已落地”的区别。

## 5. 每条功能分支进度

### 5.1 RHL 续作：保留单模型价值，暂停集成目标

`rhl-next` 以 `main@72bf9cc` 为起点，历史四个提交迁移为 `adfd52a / 47ab94c / 3c92bbd / 34cd8cc`，再加 `94d27f0 / 67a9f70 / 804172c` 完成兼容与记录。新增能力：

- BOA 的 `gaussian / orthogonal / orthogonal_antithetic` 初始化和 `legacy / kaiming / unit` 尺度；
- `tools/run_boa_rhl.sh`、`tools/run_rhl_se_val_driven.sh`；
- training_config、run_config、manifest 新字段及 RHL 单测。

`rhl-next` 缺的一个 main 提交恰好是 `087fa5e` 的 E1.2 结果归档，没有因此漏掉最新训练源码。7 月 19 日迁移报告明确“没有启动 GPU 实验”；当前结果目录也没有新的迁移后 BOA 性能证据。不能把代码迁移成功写成 BOA 多 seed 复核成功。

旧 P0/P1 的 Buffer 内容已移植；`git cherry` 只有部分 patch 完全相等，其余因兼容编辑不同。因此建议写“行为迁移并有报告”，不要写“Git 已全量 merge”。

### 5.2 伪标签分支：保留可复用工程，研究路线已冻结

`origin/feature/adaptive-pseudo-label@85d7ea1` 包含 `utils/pseudo_label.py`、teacher 输出适配、策略 guard、batch/global/class/artifact 阈值、离线校准、W0 reliability audit、W1 weighted C-RLS、runner 与测试。

已完成：15-5 overlap/disjoint、三 seed 配对、15-1 fixed/off 五步链、matched-global 六组、W0、W1 四组。未完成且不应默认继续：W2/W3；15-1 artifact/matched-global；soft/partial target。

归档分支的日志与大量 checkpoint/result 原本被 Git 忽略。远端 commit 保存了代码、表格与部分 W0 artifact，**不等于保存了所有原始实验 JSON 或模型**。本次可读取的 raw 2D metrics 中没有这条线的完整新增原始结果；报告里按“历史正式报告值”列出，缺口另记。

### 5.3 V3+ control：主线有效能力已接收，历史负实验仍有价值

该分支记录 feature source sweep、旧 runner、`air_pixel_balance=class_cap` 等诊断。6 月主线选择性集成了 model/source API，并完成历史 golden replay 报告；没有整体 merge 这个实验分支，也没有把 class-cap 负实验作为默认能力搬入 main。

因此它不是当前生产依赖，但删掉引用前应给 branch tip 留 archive tag 或 bundle，并导出 branch-only 报告。没有必要继续在该老分支开发。

## 6. 哪些可以收口

| 对象 | 建议动作 | 保留价值 / 清理前提 |
|---|---|---|
| main | 保留 | 基线、协议和公开协作入口 |
| rhl-next | 暂保留 | BOA 单模型能力与迁移记录；若选联邦路线，可另开新 feature，不把集成一起搬过去 |
| adaptive 远端与冻结 tag | 保留 tag，branch 可转只读归档 | 阴性证据、可靠性工具、weighted solver、teacher adapter 仍有复用价值 |
| 本地 adaptive 的旧指针 | 后续可快进至远端或撤销旧本地指针 | 旧指针无独有领先提交；本轮不自动更新，以保持审计快照 |
| deeplabv3plus-control | archive 后可删活动 branch | class-cap、source 诊断与分支专有文档必须可追溯 |
| rhl-se-boa-p0-p1 | archive 后可删活动 branch | 新开发改用 rhl-next；保留旧结果来源 SHA |
| rhl-next worktree | 当前保留 | 仍是已准备好的隔离工作目录；没有磁盘压力理由要删除它 |

“不再需要活动分支”不等于“代码与实验没有保留价值”。本轮没有提出直接删除阴性结果。删除操作不在本次执行范围，报告与清单已足以让下一轮清理成为可审阅操作。

## 7. 集成学习相关痕迹

| 层级 | 位置 / commit | 新决策 |
|---|---|---|
| 独立 RHL seed | `530b900` 与 Buffer/AIR 参数 | 保留；它同样用于单模型稳定性和配对实验 |
| 融合评估 | `1068436`、`a2cbc4e`；main 的两个 tools | 暂停新增实验；不从单模型训练默认路径调用 |
| P0 val-driven、BOA | 旧 `e79dcd2 / 4437b3d / 8e0861f / 7701b12`；新 rhl-next | P0 集成暂停，BOA 单模型独立评估 |
| SE 2.0 / Snapshot | AI_docs、五方法计划 | 没有完整实现或正式结果证据；从近期路线移出 |
| 旧项目约束 | AGENTS、课题Home、五方法计划 | 以用户 2026-10-24 的“不再要求集成”覆盖旧要求 |

联邦中的服务器聚合是训练协作，可以最后只输出一个模型；无需保留多模型预测融合来满足“联邦”。

## 8. 重启前的实际缺口

1. **恢复权重与未跟踪证据**：两个 worktree 都未找到 `.pth/.pt/.ckpt`；V3+、伪标签、集成原始 logs/results 需从原服务器或备份取回，再核对 manifest hash。
2. **环境不能照抄**：本机 `python` 是旧解释器，系统 `python3` 未安装 torch/numpy；旧约定的 `/home/linyichen/miniconda3/envs/segacil` 不存在。不要直接运行旧脚本。
3. **修正状态入口**：旧 Home 的 W1 待做、旧 worktree、3D 未集成、必须集成等内容已冲突；本次入口会指向新报告。
4. **冻结评价协议**：保留 clean holdout 的源列表与 hash，恢复 clean step0；正式论文不能把历史筛 test 的最高数当无偏测试结果。
5. **复跑最小基线后才开发**：代码静态审计不是 GPU 重放。本次不声称旧测试现已重跑通过，也不把历史 39/47/98 项测试通过记录当作当前环境验证。

机器清单：[branch_inventory.csv](evidence/branch_inventory.csv)、[snapshot.json](evidence/snapshot.json)、[metrics_inventory.csv](evidence/metrics_inventory.csv)。
