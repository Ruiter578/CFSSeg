# 早期旧分支文档

本目录集中保存退出活动开发的分支资料。当前代码用法见[项目README](../../README.md)，当前研究状态见[课题Home](../课题Home.md)。

| 历史分支 | 保留内容 | 完整源码恢复标签 | 处理方式 |
| --- | --- | --- | --- |
| [deeplabv3plus-control](deeplabv3plus-control/README.md) | V3+特征源、class-cap对照、旧启动脚本与不同版本文档 | `archive/deeplabv3plus-control-2026-11-19` | 将分支历史并入main，沿用主线已更新的双模型接口；旧对照脚本退出活动入口 |
| [adaptive-pseudo-label](adaptive-pseudo-label/README.md) | 阈值策略、matched-global、W0/W1结论、轻量统计与执行计划 | `archive/adaptive-pseudo-label-2026-11-19` | 同步至85d7ea1后归档，不将该分支源码合并到main |
| [rhl-se-boa-p0-p1](rhl-se-boa-p0-p1/README.md) | 早期集成/BOA方案、实现审查与不同版本文档 | `archive/rhl-se-boa-p0-p1-2026-11-19` | 保存标签后删除活动分支；rhl-next独立保留 |

与整合前main内容相同的文件沿用main原位置；只将独有文件或不同版本复制到相应分支目录。归档副本保留原路径结构与原文，历史文字里的“当前”“下一步”和服务器路径属于当时状态。逐文件来源、Git blob与SHA256见[文件清单](文件清单.csv)。

## 伪标签路线为何归档

固定伪标签整体有效，自适应类别阈值与简单可靠性加权没有带来足够稳定的额外收益：

| 对照 | 已完成结果 | 判断 |
| --- | --- | --- |
| APT相对matched-global，overlap/disjoint各3 seed | 六组平均Δall约+0.0020 pp | 类别专属阈值的额外贡献很小 |
| W1 confidence | overlap −0.0059 pp，disjoint +0.0439 pp；平均+0.0190 pp | 跨设置方向不一致 |
| W1 confidence×margin | overlap −0.0159 pp，disjoint +0.0838 pp；平均+0.0339 pp | 旧新类权衡明显，未形成充分净收益 |

W1为seed1筛选，未追加多seed；已完成的结果支持停止当前研究路线。完整方法、配置与指标见[实验回顾修订版](../2026-10-24_项目重启交接/03_方法与实验完整回顾_修订版.md)。固定伪标签与teacher适配等工程仍可从标签选择性提取。

## 恢复与追溯

```bash
git fetch origin --tags
git show archive/adaptive-pseudo-label-2026-11-19:utils/pseudo_label.py
git switch -c restore/adaptive-pseudo-label archive/adaptive-pseudo-label-2026-11-19
```

查看资料用git show即可；需要修改旧代码时再从标签新建分支。原冻结标签`adaptive-pseudo-label-w1-frozen-20260719`也继续保留。工作区与全引用备份位于本机`/data2/lyc/SegACIL_git_backups/2026-11-19/`，远端归档依靠上述标签。
