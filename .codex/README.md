# SegACIL Project Codex Setup

本目录是 SegACIL 的项目级 Codex 配置中心。它负责组织项目级配置、hooks、subagents、prompt 模板和实验/报告模板。

## Directory Map

```text
.codex/
├── config.toml          # 项目级 Codex 配置；信任项目后加载
├── hooks.json           # lifecycle hooks 入口
├── agents/              # 项目级 custom subagents
├── hooks/               # hook 脚本与说明
├── mcp/                 # MCP 使用说明
├── prompts/             # 常用 prompt 模板
├── rules/               # execpolicy 规则占位
├── skills/              # 说明：真实 repo skills 放在 .agents/skills
└── templates/           # 实验计划、报告、方法设计模板
```

Codex 官方 repo-scoped skills 扫描路径是 `.agents/skills`，不是 `.codex/skills`。因此本项目把可触发 skills 放在项目根目录的 `.agents/skills`，并在 `.codex/skills/README.md` 中说明索引。

## Active Configuration

`.codex/config.toml` 当前启用：

- `project_doc_max_bytes = 98304`：提高项目说明读取预算，避免 `AGENTS.md` + 课题文档过早截断。
- `project_doc_fallback_filenames = ["CLAUDE.md", "README.md"]`：兼容 Claude Code 和原仓库 README。
- `model_reasoning_effort = "high"`：默认提高研究型任务推理强度。
- `[features].hooks = true`：启用项目 hooks。
- `[agents]`：限制 subagent 并发和嵌套深度，避免研究任务过度分叉。
- `openaiDeveloperDocs` MCP：查询 Codex / OpenAI 官方文档。
- `context7` MCP：查询 PyTorch、NumPy、scikit-learn 等第三方文档。

不要在项目级配置中写 provider、auth、profile、telemetry 等用户私有设置；这些属于 `~/.codex/config.toml`。

## Subagents

`.codex/agents` 中的 agent 只在用户明确要求使用 subagents / 并行 agent / delegate 时使用。当前保留 5 个：

- `protocol-auditor`：审实验协议、checkpoint、BASE_SUBPATH、是否可比。
- `experiment-runner`：准备、启动、监控 A100/TRS 实验。
- `result-analyst`：分析 metrics、logs、manifest、checkpoint。
- `method-reviewer`：审 RHL、伪标签、集成学习与论文故事线。
- `code-reviewer`：审代码正确性、兼容性和复现性。

## Hooks

`.codex/hooks.json` 加载 `.codex/hooks` 下的 Python 脚本。当前 hook 只警告，不阻断命令：

- 执行前提醒危险 git/shell、GPU 环境变量、tmux 日志、`run_origin.sh`、V3+ / BASE_SUBPATH 错配风险。
- 执行后扫描常见错误，例如 Python `SyntaxError`、中文弯引号、CUDA OOM、MPS、checkpoint 缺失、AIR source mismatch。

首次使用时在 Codex CLI 中执行 `/hooks` 审查并信任。

## Project Skills

项目级 skills 位于：

```text
.agents/skills/
```

当前 skills：

- `segacil-workflow-loop`：目标到计划、实现、验证、报告、下一步的闭环。
- `segacil-experiment-gate`：实验启动和结果汇报前的协议/显存/checkpoint/manifest 闸门。
- `segacil-method-review`：RHL、伪标签、V3+、集成学习方法方案的机制审查。

## Recommended Usage

从项目根目录启动 Codex：

```bash
cd /root/2TStorage/lyc/SegACIL
codex
```

检查 MCP：

```text
/mcp
```

检查 hooks：

```text
/hooks
```

常用显式 skill 触发示例：

```text
$segacil-experiment-gate 请检查这个 VOC 15-5 step1 实验命令是否可以启动。
$segacil-method-review 请审查这个 RHL 改进是否值得进入实验。
```
