# SegACIL 项目级 Agent 规范

适用范围：当前仓库 `/data2/lyc/SegACIL`，覆盖 SegACIL / CFSSeg 项目。历史报告中的 `/root/2TStorage/lyc/SegACIL` 与 `/TRS-SAS/linwei/SegACIL` 是旧运行现场，不代表当前机器路径。

## 1. 项目事实

- 项目根目录：`/data2/lyc/SegACIL`。
- 主要研究对象：CFSSeg / SegACIL 的 2D PASCAL VOC 2012 类增量语义分割代码。
- 主数据集与协议：PASCAL VOC 2012，`15-5`，重点为 `sequential`，伪标签方法重点验证 `disjoint` / `overlap`。
- 研究约束：用户已取消必须采用集成学习的要求；集成路线暂停，保留历史证据。联邦持续分割是待验证候选，不是已实现主线。
- 课题背景：为完成张教授纵向项目指标，在庄辉平副教授指导下，基于 CFSSeg 做可解释、可复现、能写论文的方法升级。
- 预期成果：AI 会议论文，最低 EI；若涨点和故事线足够强，争取 CCF-B / CCF-A；专利在论文初稿后、正式投稿前再写。
- 历史文档记有“2026年底前录用”，该期限本次未重新确认，不承诺录用时间。

做任何较大任务前，优先阅读：

```text
AI_docs/课题Home.md
AI_docs/2026-10-24_项目重启交接/00_阅读入口.md
AI_docs/2026-10-24_项目重启交接/01_Git分支与代码进度核查.md
AI_docs/2026-10-24_项目重启交接/03_方法与实验完整回顾.md
```

## 2. 当前技术基线

- `main` 已完成 DeepLabV3+ 接口集成及历史 golden replay；当前迁移环境仍须恢复权重并做最小重放，再开展新实验。
- `run.sh` 是统一入口，`MODEL` 可选 `deeplabv3_resnet101` 或 `deeplabv3plus_resnet101` 等模型。
- `AIR_FEATURE_SOURCE=auto` 是推荐默认：
  - DeepLabV3 自动解析为 `decoder`；
  - DeepLabV3+ 自动解析为 `aspp_up`。
- V3+ 最佳已验证体系是 `DeepLabV3+ + aspp_up AIR feature`，不是单纯只换模型名。
- V3+ golden replay 已复现 `Mean IoU = 0.7035645831`。
- 原 DeepLabV3 路径已有单元测试保护，不应被后续 RHL / 伪标签改动破坏。
- clean holdout 的当前起点是 V3 + decoder、buffer8224、gamma1、norm none；不可与旧官方val-as-test约70%的结果横比。
- 两个当前worktree均缺模型权重；旧conda解释器路径不存在。执行前先恢复环境、checkpoint并核对hash，历史测试报告不等于当前验证通过。

关键代码位置：

| 文件 | 作用 |
|---|---|
| `run.sh` | A100 服务器统一实验入口 |
| `run_trs.sh` | TRS 服务器专用入口，GPU 编号和路径不同 |
| `train.py` | 参数解析、任务类别展开、启动 Trainer |
| `trainer/trainer.py` | step0 训练、step1 AIR realign、后续 C-RLS 更新 |
| `network/_deeplab.py` | DeepLabV3 / DeepLabV3+ head 和 AIR feature source |
| `network/Buffer.py` | RHL / RandomBuffer |
| `network/AnalyticLinear.py` | RecursiveLinear / C-RLS |
| `utils/run_manifest.py` | 实验 manifest 与 checkpoint hash |
| `utils/tasks.py` | VOC 任务划分 |

## 3. 历史服务器和 GPU

| 名称 | 项目路径 | GPU | 说明 |
|---|---|---|---|
| A100 服务器 | `/root/2TStorage/lyc/SegACIL` | `CUDA_VISIBLE_DEVICES=0`，单卡 A100 80GB | 历史主开发现场，非当前路径 |
| TRS 服务器 | `/TRS-SAS/linwei/SegACIL` | `CUDA_VISIBLE_DEVICES=2`，单卡 4090 48GB | 远端复现实验服务器 |

GPU 任务规则：

- 启动 GPU 命令前必须检查 `nvidia-smi`。
- 用户已明确说明：本项目启动实验时主要关心显存是否足够，GPU util 高不是默认阻塞条件。
- 不得随意杀他人进程。
- A100 服务器上运行 2D 分割 GPU 命令时设置：

```bash
export CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass
export TMPDIR=/root/2TStorage/tmp
```

- 长任务必须使用 `tmux` 或等价后台方式，并写清：
  - tmux 会话名；
  - 完整命令或 runner 脚本；
  - log 路径；
  - checkpoint / output 目录；
  - 使用的 branch / commit。

## 4. Git 和 worktree

- 当前注册worktree：`/data2/lyc/SegACIL`（main）与 `/data2/lyc/SegACIL_rhl_next`（feature/rhl-next）；每次任务仍须重新检查 `git worktree list`。
- TRS 服务器通常只有一个 checkout：`/TRS-SAS/linwei/SegACIL`，通过 `git switch` 切换分支，工作区内容会随分支变化。
- 新方法开发不要直接长期写在 `main`；从最新 `main` 新开 feature 分支，必要时新建 sibling worktree。
- 清理分支/worktree前保存独有代码、文档、忽略的实验产物及可达Git引用；旧V3+ worktree当前已不在注册列表。
- 跨服务器同步顺序：

```bash
# A100
git status
git push origin <branch>

# TRS
git fetch origin
git switch <branch>
git pull --ff-only
```

## 5. 实验协议

- 每个新实验必须使用独立 `SUBPATH`，不得覆盖已有正式结果。
- `SUBPATH` 控制当前实验写入目录；`BASE_SUBPATH` 控制 `curr_step=1` 时加载哪个 step0 checkpoint。
- 跑 V3+ 时必须使用 V3+ step0 的 `BASE_SUBPATH`，不能误用默认 V3 的 `20260606`。
- 每个实验至少记录：
  - `MODEL`
  - `AIR_FEATURE_SOURCE`
  - `SUBPATH`
  - `BASE_SUBPATH`
  - `BUFFER`
  - `GAMMA`
  - `RANDOM_SEED`
  - `RHL_SEED`
  - `RHL_NORM`
  - batch size
  - setting / task / curr_step
  - checkpoint hash 或 manifest
- `run_manifest.json` 是正式实验可追溯性的最低要求。
- 不要用旧 summary 或未完成实验冒充最新结论。

历史 A100 V3+ step1 入口示例（当前机器不能直接照抄路径；先恢复环境与权重）：

```bash
cd /root/2TStorage/lyc/SegACIL
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
CUDA_VISIBLE_DEVICES=0 \
MODEL=deeplabv3plus_resnet101 \
AIR_FEATURE_SOURCE=auto \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
SUBPATH=<new_unique_subpath> \
START_STEP=1 END_STEP=1 DEFAULT_BATCH_SIZE=16 \
BUFFER=8196 GAMMA=1 RANDOM_SEED=1 \
RHL_NORM=none RHL_SEED=-1 RHL_STATS=0 \
bash run.sh
```

## 6. RHL 和伪标签边界

- 自适应hard threshold与W1可靠性加权研究线已冻结；不默认继续W2/W3、q/min_conf/weight小网格。完整代码留在 `origin/feature/adaptive-pseudo-label`，main只归档结论。
- RHL续作在 `feature/rhl-next`；BOA单点弱正向不等于多seed复核完成。PGH/PowerNorm/CA-C-RLS多数仍是提案。
- RHL 的关键路径是 `RandomBuffer -> RecursiveLinear`，改动应尽量不破坏 C-RLS 闭式更新主公式。
- RHL-SE 需要区分 `random_seed` 与 `rhl_seed`：只改变 RHL 子空间时，不能混入 dataloader、augmentation、全局随机性的变化。
- 自适应伪标签不能默认套到 `15-5 sequential`：
  - 当前代码中伪标签路径不在 `curr_step=1 + sequential` 执行；
  - sequential 下旧类标签可见，伪标签不是主要矛盾；
  - 自适应阈值应优先在 `disjoint` / `overlap` 验证。
- 若新增伪标签策略，必须显式处理 DeepLab tuple 输出和 AIR tensor 输出，不允许在 Trainer 里堆临时 shape hack。

## 7. 代码实现规范

- 先读现有实现，再改代码；优先复用现有 Dataset、Trainer、model factory、manifest、runner。
- 新增配置必须同时考虑 CLI、checkpoint / model 保存、manifest、runner 和报告。
- 不写未被调用、未测试、未记录的函数。
- 不用大段 `try/except` 掩盖真实错误。
- 注释解释关键约束和为什么，不重复代码表面含义。
- 文件编辑优先使用 `apply_patch`。
- 不要用 Python 脚本做简单文件读写。

## 8. 修改后的检查

Python 改动至少运行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile <changed_python_files>
```

脚本改动至少运行：

```bash
bash -n <changed_shell_files>
```

本项目已有单元测试时，运行：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest discover -s tests -p 'test*.py' -v
```

必须检查中文弯引号进入可执行文本：

```bash
grep -n '<中文弯引号正则>' <changed_files>
```

按改动类型执行最小 smoke：

- Model：跑 forward，检查 logits 和 AIR feature shape。
- Trainer / AIR：用单 batch 或现有单元测试验证 step0 / step1 路径。
- Runner：先用 `PYTHON=/bin/echo bash run.sh` 检查命令展开。
- Manifest：用临时目录写入并检查字段。
- Result analysis：用真实 `test_results_*.json` 或日志路径复核，不凭记忆汇报。

## 9. 报告规范

代码或实验任务结束后，报告放入 `AI_docs` 的合适目录：

| 类型 | 推荐目录 |
|---|---|
| 方法设计 | `AI_docs/idea构思与实验设计/` |
| 实验结论 | `AI_docs/idea验证与结论/` |
| 代码改动 | `AI_docs/代码改动报告/` |
| 论文理解 | `AI_docs/论文精读/` |
| 原理推导 | `AI_docs/数学推导与公式解析/` |

实验报告必须包含：

- 命令；
- 输出目录；
- checkpoint 路径；
- baseline 对照；
- old / new / all mIoU；
- per-class 指标；
- 失败原因；
- 下一步动作。

## 10. Subagent 使用

项目级 custom subagents 位于 `.codex/agents`。只在用户明确要求使用 subagents / 并行 agent / delegate 时使用。

推荐拆分：

- `protocol-auditor`：审协议、checkpoint、BASE_SUBPATH、是否可比；
- `experiment-runner`：准备、启动、监控实验；
- `result-analyst`：读 metrics / logs / checkpoint / manifest；
- `method-reviewer`：审 RHL、伪标签、集成学习故事线；
- `code-reviewer`：审代码正确性、兼容性、可复现性。

不要让多个 subagent 同时改同一文件。主 agent 必须复核子 agent 结论，不能直接把子 agent 输出当事实。

## 11. 低级错误防线

最终回复前必须排除：

- Python 语法错误；
- shell 脚本语法错误；
- CLI 参数存在但未进入真实代码路径；
- V3+ 实验误用 V3 step0 checkpoint；
- `BASE_SUBPATH` / `SUBPATH` 混用导致覆盖历史结果；
- pseudo-label 在 sequential 中静默无效；
- 未完成实验被当作结论；
- summary 失败却误判训练失败；
- GPU OOM 被误写成方法无效；
- git worktree / 分支关系没有检查就解释。
