# SegACIL MCP Notes

当前 `.codex/config.toml` 只配置两个可选 MCP：

| MCP | 用途 | 失败处理 |
|---|---|---|
| `openaiDeveloperDocs` | 查询 OpenAI / Codex 官方文档 | 可选；失败时用 web 官方域名查证 |
| `context7` | 查询 PyTorch、NumPy、scikit-learn 等第三方库文档 | 可选；失败时使用本地源码和已安装包 |

项目级配置不保存 token、provider、profile、telemetry 或任何用户私密设置。认证和 provider 只放在用户级 `~/.codex/config.toml`。

使用前在 Codex CLI 中检查：

```text
/mcp
```
