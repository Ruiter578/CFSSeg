# SegACIL / CFSSeg 当前入口

状态日期：2026-10-24。完整交接从 [阅读入口](2026-10-24_项目重启交接/00_阅读入口.md) 开始。本页只维护当前事实；7月历史Home仍可从Git版本恢复。

## 任务与决策

- 主要对象仍是2D类增量语义分割、冻结特征、RHL与解析更新。
- 用户已明确不再要求集成学习；暂停RHL-SE扩展、SE2.0与Snapshot，不删除历史代码与证据。
- 自适应类别hard threshold和W1简单可靠性加权均已完成止损，W2/W3不默认继续。
- 联邦持续分割是候选路线，尚无本仓代码/实验；AFL、AFCL、FISS已存在，必须先明确新增研究问题。
- 新人先读 [方法速览](2026-10-24_项目重启交接/04_方法与实验速览.md)，路线讨论读 [后续规划](2026-10-24_项目重启交接/05_联邦持续学习与后续研究规划.md)。

## 仓库与环境

| 对象 | 状态 |
|---|---|
| 主目录 | /data2/lyc/SegACIL，main，核查时HEAD=087fa5e |
| 第二worktree | /data2/lyc/SegACIL_rhl_next，feature/rhl-next，HEAD=804172c |
| 伪标签归档 | 远端85d7ea1及tag adaptive-pseudo-label-w1-frozen-20260719；本地旧分支落后17提交 |
| 老control/P0-P1分支 | 保留历史来源，不作为当前开发起点 |
| 模型权重 | 两个checkout均未找到.pth/.pt/.ckpt，需恢复并核对manifest hash |
| 运行环境 | 旧conda路径不存在；系统python3没有torch/numpy；启动实验前先恢复 |
| 3D | main已有CFSSeg-code3D与S3DIS summary，暂作旁线归档 |

完整清单见 [Git与代码核查](2026-10-24_项目重启交接/01_Git分支与代码进度核查.md)。旧A100/TRS路径是来源记录，不是当前机器可用路径。

## 实验基线与最后进度

| 谱系 | 结果 | 注意 |
|---|---|---|
| 历史V3 15-5 sequential | all69.4606/69.5598% | 原始JSON可读；官方VOC val被代码命名为test |
| 历史V3 15-1最终step5 | 1128=68.9464%；1128_trs=69.5079% | 以JSON路径为准 |
| 历史V3+ + aspp_up | all70.3565% | 主线已接入，golden replay是历史报告；当前缺原权重/结果 |
| 当前clean V3 | decoder、buffer8224、gamma1、norm none | 10%训练holdout，所有训练step排除；与历史约70%不可横比 |
| clean E1.1 | decoder三seed均值59.6412%，aspp59.1228% | decoder胜出 |
| 最后E1.2 | eps1e-3=59.7310%；eps0=59.7119%；eps1e-4=59.7143% | 两次新尝试无提升，已完成，不是待做 |

全部结果、证据等级与负结果解释见 [详细回顾](2026-10-24_项目重启交接/03_方法与实验完整回顾.md)。不得把文档报告值冒充本次原始重算或训练复现。

## 接手规则

1. 先git status与git worktree list；未经要求不commit/push，不删除归档。
2. 每个新实验独立SUBPATH，BASE_SUBPATH对应真实step0，不混用V3/V3+。
3. 固定模型、AIR source、seed、RHL seed、buffer、gamma、norm、split与base hash。
4. 主线原固定伪标签未实现归档分支完整step1路径；不能拿main直接复刻该分支15-5伪标签结果。
5. 方法提案、代码已实现、实验已完成、机制已成立四者分开记录。
6. 先修复可复跑性与协议，再选单模型或联邦路线；不要从过时五方法计划自动启动集成或W1后续网格。

文档重组按 [分类与迁移方案](2026-10-24_项目重启交接/02_文档价值与归档重构方案.md) 执行，本轮历史文件未移动。
