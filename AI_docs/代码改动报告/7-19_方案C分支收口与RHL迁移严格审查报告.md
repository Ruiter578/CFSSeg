# 方案 C 分支收口与 RHL 迁移严格审查报告

> 执行日期：2026-07-19
> 主仓库：`/root/2TStorage/lyc/SegACIL`
> 结论：方案 C 已按“主线最小化、研究分支归档、RHL 独立续作”完成
> 审查边界：没有把有争议的伪标签训练代码或 RHL 实验代码合入 `main`

## 1. 最终分支状态

| 分支 / 标签 | 提交 | 状态 | 用途 |
| --- | --- | --- | --- |
| `main` | `72bf9cc` | 已推送 | 仅归档 5 份伪标签最终结论文档，并修复 1 行既有测试夹具 |
| `feature/adaptive-pseudo-label` | `85d7ea1` | 已推送 | 保存自适应阈值完整代码、实验和失败证据 |
| `adaptive-pseudo-label-w1-frozen-20260719` | 指向 `85d7ea1` | 已推送 | 自适应阈值 W1 后的冻结锚点 |
| `feature/rhl-next` | 以 `72bf9cc` 为基线 | 待本报告提交后推送 | RHL-SE / BOA-RHL 的独立续作分支 |

`main` 在本轮没有修改以下生产代码：

- `network/Buffer.py`
- `network/AnalyticLinear.py`
- `trainer/trainer.py`
- `utils/parser.py`
- 原 CFSSeg 固定阈值伪标签机制

## 2. 自适应伪标签分支如何收口

### 2.1 已执行

1. 在独立 worktree 中完成路线冻结报告。
2. 明确区分：
   - 原 CFSSeg 固定阈值伪标签辅助机制；
   - 本项目的自适应阈值改造；
   - 不属于阈值方法的 soft / partial pseudo target。
3. 完成提交、推送和冻结标签。
4. 通过真实 checkpoint/result 软链接复核归档分支全套测试。

### 2.2 没有移植到 `main`

以下内容均保留在归档分支，未进入主线：

- `PseudoLabeler` 重构；
- adaptive / artifact / class-wise threshold；
- weighted C-RLS；
- pseudo-label protocol guard；
- teacher output adapter；
- calibration、grid、recovery、W1 runner；
- 实验配置、artifact、log 和 checkpoint。

原因不是这些工程实现一定错误，而是它们都会改变训练关键路径，且现有实验没有证明
其值得成为长期主线能力。按照“有争议先不移植”的要求，默认关闭也不能替代主线审查。

## 3. `main` 的安全集成

最终主线提交 `72bf9cc` 是从最新远端 `main=5c6cf48` 生成的单一 squash 提交。
最终树只新增：

1. matched-global 六组正式结论；
2. W0 原始标签可靠性审计；
3. W1 可靠性加权 C-RLS 结果与路线决策；
4. 自适应阈值全阶段小白版复盘；
5. 路线冻结范围与后续方向调研报告。

另有一行测试夹具修复：

```python
patch.object(torch, "randn", ...)
```

原测试使用字符串路径 `network.AnalyticLinear.torch.randn`，但包级同名类遮蔽了模块，
导致最新远端 `main` 自身必然报 `ModuleNotFoundError`。修复只改变 mock 定位方式，不改
生产代码或测试目标。

历史 W1 执行指南曾在临时集成分支中被引入用于依赖解析，随后被删除。正式合入前又从
`main` 生成干净 squash，确保这份已冻结的指南既不在最终树，也不进入 `main` 的新增
提交历史。

## 4. RHL 迁移范围

新分支 `feature/rhl-next` 从 `main=72bf9cc` 创建，保守迁移原
`feature/rhl-se-boa-p0-p1` 的四个提交：

- `e79dcd2`：RHL-SE val-driven 执行计划与 runner；
- `4437b3d`：BOA-RHL 初始化、参数、统计、测试和 runner；
- `8e0861f`：step0 realign / step1 配置快照；
- `7701b12`：原 P0/P1 代码审查报告。

迁移冲突只出现在 `trainer/trainer.py` 和 `utils/ckpt.py`，按行为并集解决：

- 保留 `main` 的 `make_step0_loader_opts()`，不临时篡改 `self.opts.curr_step`；
- 保留 `main` 的 `evaluation_mode=val|test|both`；
- 保留 `main` 的 `analytic_tail_epsilon`；
- 接入 `rhl_init`、`rhl_scale_mode`；
- checkpoint 同时保留父目录自动创建和 `training_config`；
- step0 realign 保存 `curr_step=0` 快照，step1 保存 `curr_step=1` 快照。

这两处没有选择某一侧覆盖另一侧，也没有改变 C-RLS 递推公式。

## 5. 人工审查发现并修复的问题

### 5.1 远端 `main` 的测试夹具错误

已在主线安全提交中修复，详见第 3 节。

### 5.2 runner 不能替换 Python 解释器

两个 RHL runner 原先硬编码 `python`，无法按项目规范执行
`PYTHON=/bin/echo` 命令展开 smoke，也可能绕过当前 conda 环境。现已统一支持：

```bash
PYTHON="${PYTHON:-python}"
```

### 5.3 manifest 缺少 RHL 新字段的 flat keys

`rhl_init` 和 `rhl_scale_mode` 原本只存在于 manifest 的完整 `args`，没有进入供旧报告
和临时脚本读取的 flat keys。已用 TDD 修复：

1. 先加入断言，得到预期 `KeyError: 'rhl_init'`；
2. 在 `utils/run_manifest.py` 增加两个 flat keys；
3. 单测和全套测试通过。

## 6. CodeRabbit 三条 Major 的核验

### 6.1 `search_rhl_class_weights.py` 应传 `--mode val`

结论：**误报，不修改。**

该脚本没有 `--mode` 参数，并在 `main()` 中固定执行：

```python
_, val_loader, _ = init_dataloader(opts)
```

搜索和候选评估都只消费 `val_loader`。强加 `--mode val` 会使 argparse 直接失败。
runner 已增加注释固定这一协议边界，最终 test 仍由
`eval_rhl_ensemble.py --mode test` 单独执行。

### 6.2 `load_ckpt()` 应返回 `training_config`

结论：**有争议，暂不实施。**

当前 `training_config` 是审计元数据，RHL 拓扑随序列化 model object 恢复；全仓已建立
三元返回契约 `(model, optimizer_state, best_score)`。把它改成四元组会扩大恢复接口、
修改所有调用方和历史测试，但当前 RHL 迁移没有消费该配置的需求。

若以后决定让 eval / infer 从配置字典重建模型，应作为独立 checkpoint schema 升级，
包含版本号、旧 checkpoint 兼容和全调用链测试，不能夹带在本次迁移中。

### 6.3 realign 会覆盖原 step0 `run_config.json`

结论：**误报，不修改。**

`self.root_path0` 使用当前实验的 `subpath`，是本次 step1 产生的 realign 输出目录；
原 step0 checkpoint 从 `base_subpath` 下的 `self.root_path_prev` 读取。两者不是同一路径。
项目又强制每个正式实验使用独立 `SUBPATH`，所以不会覆盖原 step0 配置。

### 6.4 二次机器审查状态

第一次 CodeRabbit 完成并返回上述 3 条 Major。完成核验、注释和 manifest 修复后，
第二次审查被 CodeRabbit 套餐限流阻断，提示需等待 45 分钟。该状态不能记为
“二次 CodeRabbit 通过”；最终结论由完整测试、差异审查和上述逐条人工核验共同支持。

## 7. 验证证据

### 7.1 自适应归档分支

- 全套单测：`98/98 passed`；
- 文献标记：`16 refs / 16 anchors`；
- 本地文档链接：通过；
- `git diff --check`：通过；
- 敏感信息扫描：通过。

首次在新 worktree 运行时有 4 项 W1 测试找不到结果文件；将主工作树中被 Git 忽略的
对应 checkpoint 目录以只读用途软链接到 worktree 后，98 项全部通过。这证明失败来自
worktree 不携带实验产物，不是代码回归。

### 7.2 主线安全集成

- Python 语法：通过；
- 全套单测：`39/39 passed`；
- Markdown 本地链接和引文 gate：通过；
- `git diff --check`：通过；
- 敏感信息扫描：通过；
- CodeRabbit：`0 findings`；
- 合入并推送后的 `main` 再跑：`39/39 passed`。

### 7.3 RHL 续作分支

- Python 语法：通过；
- shell 语法：两个 runner 均通过 `bash -n`；
- BOA runner：`PYTHON=/bin/echo` 单 case smoke 通过；
- RHL-SE runner：val-search → test-eval 命令展开 smoke 通过；
- RHL / main 全套单测：`47/47 passed`；
- `git diff --check main...HEAD`：通过；
- 敏感信息扫描：通过。

## 8. 最终判断与待确认项

### 8.1 可以确认

- 自适应伪标签阈值研究线已冻结并归档；
- 原 CFSSeg 固定伪标签机制没有被删除或否定；
- `main` 没有接收伪标签或 RHL 的关键生产代码；
- RHL 续作已从最新 `main` 隔离出来，现有 P0/P1 资产完成兼容迁移；
- 当前代码可进入下一轮 RHL 实验准备，但本轮没有启动 GPU 实验。

### 8.2 留待确认

唯一保留的接口级争议项是：

> 是否把 checkpoint `training_config` 从审计元数据升级为 `load_ckpt()` 的正式恢复契约。

在确认前不修改三元返回接口，也不把配置恢复逻辑扩散到 eval / infer。

### 8.3 后续实验边界

若下一步启动 BOA-RHL，应先复核最新基线 checkpoint、`BASE_SUBPATH`、GPU 显存和
独立 `SUBPATH`，再按既有 BOA-0/1/2/3 预注册矩阵执行。实验结果仍按 all mIoU gate
止损，不能因为代码已迁移或历史投入而扩大低收益网格。
