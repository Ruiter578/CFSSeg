# SegACIL Codex Hooks

这些 hook 是研究流程的低成本提醒器，不是强制安全沙箱。

## Files

- `pre_tool_use_policy.py`：在 Bash 命令执行前检查危险命令、GPU 环境变量、tmux 日志、V3+ checkpoint source 等明显风险。
- `post_tool_use_review.py`：在 Bash 命令执行后扫描常见错误，例如 `SyntaxError`、CUDA OOM、MPS 问题、checkpoint 缺失、AIR source mismatch。

## Design

- 只输出 warning，不阻断命令。
- 兼容 Python 3.8+。
- 不读取或修改实验产物。
- 不替代 agent 的人工判断；最终仍按 `AGENTS.md` 的验证要求执行。

首次启用后在 Codex CLI 中运行 `/hooks`，审查并信任项目 hook。
