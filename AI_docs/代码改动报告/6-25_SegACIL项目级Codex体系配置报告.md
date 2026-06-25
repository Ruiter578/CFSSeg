# 6-25 SegACIL 项目级 Codex 体系配置报告

> 分支：`feature/segacil-codex-system`
> 目标：为 SegACIL 建立项目级 `.codex` / `AGENTS.md` / repo-scoped skills / subagent / prompt / template 体系，使后续 RHL、自适应伪标签、实验复现和论文写作任务更稳定。

## 1. 设计依据

本次设计综合了以下信息：

1. SegACIL 当前代码库：`run.sh`、`run_trs.sh`、`trainer/trainer.py`、`network/_deeplab.py`、`utils/run_manifest.py`、测试文件。
2. SegACIL 文档体系：`AI_docs/课题Home.md`、V3+ 主线融合报告、RHL 五方法路线、自适应伪标签方案、论文精读和代码理解文档。
3. Codex 记忆与历史对话：RHL、伪标签、V3+、worktree、A100/TRS、BASE_SUBPATH/SUBPATH、RHL_SEED 等长期决策。
4. 2D_scannet_seg 的 `.codex` 参考体系：结构可复用，但具体 agent/skill 需要按 SegACIL 任务重写。
5. OpenAI Codex 官方文档：
   - Customization: https://developers.openai.com/codex/concepts/customization
   - AGENTS.md: https://developers.openai.com/codex/guides/agents-md
   - Config reference: https://developers.openai.com/codex/config-reference
   - Skills: https://developers.openai.com/codex/skills
   - Subagents: https://developers.openai.com/codex/subagents
   - Hooks: https://developers.openai.com/codex/hooks

官方文档确认的关键原则：

- Codex 定制层包括 `AGENTS.md`、memories、skills、MCP、subagents；这些互补而不是互斥。
- `AGENTS.md` 适合写每次任务都要遵守的项目规范。
- repo-scoped skills 应放在 `.agents/skills`，而不是 `.codex/skills`。
- project `.codex/config.toml` 不应写 provider、auth、profile、telemetry 等用户私密配置。
- subagents 只在用户显式要求时启动，不能默认滥用。
- hooks 适合做确定性提醒，但项目 hook 需要用户信任后才运行。

## 2. 核心设计决策

### 2.1 `AGENTS.md` 是主入口

`AGENTS.md` 写入了本项目每次任务都要遵守的硬规则：

- 张教授纵向项目、庄辉平副教授指导、CFSSeg/SegACIL 课题背景；
- VOC 2012 `15-5` 指标：已有值 64.3%，单模型目标 65.9%，集成系统目标 67.0%；
- V3+ 已融入 main，当前无额外 V3+ 验证任务；
- A100 / TRS 双服务器路径、GPU 编号和启动规则；
- `SUBPATH` / `BASE_SUBPATH` 分离；
- V3+ 不能误用 V3 step0 checkpoint；
- RHL 与伪标签的机制边界；
- 代码、实验、报告、subagent 和低级错误防线。

### 2.2 `.codex` 负责工具体系，不堆业务长文

`.codex` 下只放工具和操作资产：

```text
.codex/
├── config.toml
├── hooks.json
├── agents/
├── hooks/
├── mcp/
├── prompts/
├── rules/
├── skills/
└── templates/
```

业务背景和实验判断不散落在 `.codex` 多处，而集中在 `AGENTS.md` 和 `AI_docs/课题Home.md`，避免多源事实互相过期。

### 2.3 skills 控制在 3 个

没有从网上盲目安装大量通用 skills。原因是 SegACIL 的核心风险不是"不会写代码"，而是：

1. 实验协议错；
2. checkpoint / model / feature source 错；
3. 方法机制和论文故事线错。

因此 repo-scoped skills 只保留：

| Skill | 用途 |
|---|---|
| `segacil-workflow-loop` | 目标到计划、实现、验证、报告的闭环 |
| `segacil-experiment-gate` | 实验启动和结果汇报前的协议闸门 |
| `segacil-method-review` | RHL、伪标签、V3+、集成方法审查 |

### 2.4 subagents 控制在 5 个

子 agent 体系"精不在多"。当前保留：

| Subagent | 用途 |
|---|---|
| `protocol-auditor` | 审 task/setting/checkpoint/source/manifest |
| `experiment-runner` | 准备、启动、监控 A100/TRS 实验 |
| `result-analyst` | 分析 JSON/log/manifest/checkpoint |
| `method-reviewer` | 审 RHL/伪标签/集成学习/论文故事线 |
| `code-reviewer` | 审代码正确性、兼容性、复现性 |

没有单独设置 paper-writer agent，因为现阶段论文主线尚未收敛；论文相关判断由 `method-reviewer` 承担更稳。

### 2.5 hooks 只警告不阻断

研究项目中阻断式 hook 容易误伤探索命令，所以当前 hooks 只输出 warning：

- `pre_tool_use_policy.py`：提醒危险命令、GPU 环境变量、tmux log、`run_origin.sh`、V3+ / BASE_SUBPATH 错配。
- `post_tool_use_review.py`：扫描 `SyntaxError`、CUDA OOM、MPS、checkpoint 缺失、AIR source mismatch 等常见错误。

## 3. 文件清单

新增：

```text
AGENTS.md
.codex/README.md
.codex/config.toml
.codex/hooks.json
.codex/hooks/README.md
.codex/hooks/pre_tool_use_policy.py
.codex/hooks/post_tool_use_review.py
.codex/mcp/README.md
.codex/rules/default.rules
.codex/skills/README.md
.codex/agents/protocol-auditor.toml
.codex/agents/experiment-runner.toml
.codex/agents/result-analyst.toml
.codex/agents/method-reviewer.toml
.codex/agents/code-reviewer.toml
.codex/prompts/start-experiment.md
.codex/prompts/review-experiment.md
.codex/prompts/next-step-decision.md
.codex/prompts/method-review.md
.codex/templates/codex_task_prompt.md
.codex/templates/experiment_plan.md
.codex/templates/experiment_report.md
.codex/templates/code_change_report.md
.codex/templates/method_design_note.md
.agents/skills/segacil-workflow-loop/SKILL.md
.agents/skills/segacil-experiment-gate/SKILL.md
.agents/skills/segacil-method-review/SKILL.md
```

修改：

```text
AI_docs/课题Home.md
```

## 4. `AI_docs/课题Home.md` 更新

本次把 Home 从 2026-06-09 状态更新到 2026-06-25：

- 增加当前最新状态；
- 澄清张教授纵向项目、庄辉平副教授指导、EI/CCF/专利/2026 年底录用目标；
- 补入 V3+ 主线融合结果和 `0.7035645831`；
- 更新 `run.sh` 当前变量；
- 增加 A100/TRS 双服务器说明；
- 增加项目级 Codex 体系入口；
- 更新关键决策：V3+ 不再是未接通的候选项，后续主线转向 RHL 和自适应伪标签。

## 5. 使用方式

### 5.1 启动 Codex

```bash
cd /root/2TStorage/lyc/SegACIL
codex
```

首次使用后检查：

```text
/mcp
/hooks
/skills
```

### 5.2 常用 prompt

启动实验前：

```text
$segacil-experiment-gate 请检查这个 VOC 15-5 step1 实验命令是否可以启动。
```

审查方法前：

```text
$segacil-method-review 请判断这个 RHL 改进是否值得实现。
```

完整任务闭环：

```text
$segacil-workflow-loop 请根据当前 main 设计并执行一个 RHL 方法验证任务。
```

### 5.3 使用 subagents

只有明确需要并行审查时使用，例如：

```text
请使用 subagents 并行审查这个 RHL 实验：protocol-auditor 检查协议，result-analyst 分析结果，method-reviewer 判断下一步。
```

不要默认让 subagents 写同一个文件。

## 6. 后续维护规则

1. 如果 agent 反复犯同一个 SegACIL 错误，优先更新 `AGENTS.md`。
2. 如果某个流程会重复 3 次以上，考虑沉淀到 `.agents/skills`。
3. 如果只是一次性 prompt，不要新增 skill，放到 `.codex/prompts`。
4. 如果是实验报告或代码报告格式，放到 `.codex/templates`。
5. 如果 subagent 角色开始重叠，删除或合并，而不是继续增加。
6. 不要在 `.codex/config.toml` 写任何个人 token、provider、profile 或 telemetry。

## 7. 执行与验证状态

本次配置在 `feature/segacil-codex-system` 分支完成，并已 rebase 到执行时最新的 `origin/main` 之后。rebase 过程中仅 `AI_docs/课题Home.md` 出现内容冲突，已合并远端新增的实验/路径信息和本次 Codex 体系更新。

已执行检查：

```bash
/usr/bin/python3 -m py_compile .codex/hooks/pre_tool_use_policy.py .codex/hooks/post_tool_use_review.py
python - <<'PY'  # 解析 .codex/config.toml、agents toml、hooks.json、skills front matter
git diff --check HEAD~1..HEAD
grep -n '<中文弯引号正则>' .codex/hooks/*.py .codex/config.toml .codex/hooks.json .codex/agents/*.toml .agents/skills/*/SKILL.md || true
```

hook 烟测已确认：

- `pre_tool_use_policy.py` 能识别 GPU 环境变量缺失、V3+ 误用 V3 `BASE_SUBPATH=20260606` 等风险；
- `post_tool_use_review.py` 能识别 CUDA OOM 和 checkpoint 缺失类错误；
- 两个 hook 都只输出 warning，不阻断命令。

远端同步状态：

- 本地提交已完成；
- 当前非交互环境缺少 GitHub HTTPS 凭据，`git push -u origin feature/segacil-codex-system` 无法读取用户名，因此远端推送需要在用户终端手动执行；
- `git status --untracked-files=no` 显示本次跟踪文件已干净；
- 普通 `git status` 会显示若干历史 `checkpoints/` 目录为 untracked，这是远端最新 `.gitignore` 规则变化暴露出的本地实验产物，本次没有纳入提交。

## 8. 结论

SegACIL 现在有了项目级 Codex 体系：

- `AGENTS.md` 固化项目事实和执行底线；
- `.codex` 固化工具、subagent、hook、prompt 和 template；
- `.agents/skills` 固化可复用工作流；
- `AI_docs/课题Home.md` 更新为当前项目总览。

后续 RHL 五方法、自适应伪标签阈值、实验复现和论文故事线收敛，都应从这套体系进入。
