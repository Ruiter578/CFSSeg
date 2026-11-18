# SegACIL / CFSSeg 当前入口

状态日期：2026-11-19。代码使用方式见[项目README](../README.md)，本轮处理见[Git整合与修复报告](代码改动报告/2026-11-19_Git分支整合与项目整理报告.md)。

## 分支与工作目录

| 对象 | 当前用途 |
| --- | --- |
| main | 双模型训练、AIR解析学习、clean validation和归档入口 |
| feature/rhl-next | 保留BOA/RHL续作代码，本轮尚未合并或归档；对应worktree已移除 |
| /data2/lyc/SegACIL | 唯一活动worktree，main |
| 早期V3+、伪标签、P0/P1 | [旧分支文档](早期旧分支文档/README.md)和archive标签；退出活动分支列表 |

V3+ control历史已并入main；其旧runner、负实验class-cap及过时接口保存在归档中。伪标签完整实现没有合入main，main仍保留原有固定伪标签路径与既有历史总结。

## 当前代码约定

- `MODEL=deeplabv3_resnet101`为默认，`MODEL=deeplabv3plus_resnet101`为可选升级。
- `AIR_FEATURE_SOURCE=auto`：V3→decoder，V3+→aspp_up。
- 新run默认读取自身step0；复用历史step0必须显式设置BASE_SUBPATH，并匹配V3/V3+架构。
- TRS与RHL归一化脚本共享run.sh的模型、验证与参数转发逻辑。
- 项目checkpoint包含完整模型对象；读取使用CPU映射和显式完整反序列化。
- 当前没有可直接续跑的历史模型权重。维护测试使用本机独立环境`/data2/lyc/.local/venvs/segacil-review`，不改动其他课题的环境。

## 研究进度

| 已完成实验 | 结果与结论 |
| --- | --- |
| V3+ + aspp_up，15-5 sequential | 历史all 70.3565%；保留为可选架构 |
| clean V3特征源比较 | decoder三seed均值59.6412%，aspp59.1228%；后续clean基线使用decoder |
| 最新E1.2初始化ε | 1e-3=59.7310%，0=59.7119%，1e-4=59.7143%；减小ε无提升 |
| 自适应类别阈值 | matched-global六组平均优势约0.0020 pp，归档 |
| W1可靠性加权 | 平均优势0.0190/0.0339 pp，overlap略降，归档 |
| BOA/RHL续作 | 正交单seed弱正向，反向成对退化，迁移后未补齐新多seed实验 |

精度结果与本轮代码检查分别记录；本轮未启动GPU训练。完整配置、old/new/all见[方法与实验回顾修订版](2026-10-24_项目重启交接/03_方法与实验完整回顾_修订版.md)。

## 下一轮

建议将feature/rhl-next整体归档：它混合了旧集成runner、BOA初始化与兼容元数据，尚没有稳定的新增精度收益。先保存独有文档和完整tag，再删除活动分支；若后续明确需要正交RHL，则从当前main单独提取该组件并做匹配对照。本轮保留该分支等待决定。

集成学习约束已取消；联邦持续分割仍是候选研究场景，尚未实现。新研究从当前main开独立feature分支，并沿用固定数据划分、独立SUBPATH及manifest。
