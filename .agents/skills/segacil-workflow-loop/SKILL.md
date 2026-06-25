---
name: segacil-workflow-loop
description: SegACIL workflow loop for research/code/experiment tasks. Use when planning or executing SegACIL work that must read project context, preserve protocol comparability, verify outputs, and write AI_docs reports.
---

# SegACIL Workflow Loop

Use this for nontrivial SegACIL tasks: method design, code changes, experiment launch, result analysis, or paper-facing reports.

## Loop

1. **Orient**
   - Read `AGENTS.md`.
   - Read `AI_docs/课题Home.md`.
   - Read task-specific docs from `Codex_Plans/` or `AI_docs/`.
   - Check `git status --short --branch`.
   - Completion: you know branch, dirty state, relevant files, and current baseline.

2. **Bound**
   - State whether the task is code, experiment, analysis, or writing.
   - Identify inputs, outputs, paths, server, GPU, branch, and success evidence.
   - Completion: no hidden dependency remains for SUBPATH, BASE_SUBPATH, MODEL, AIR source, RHL config, or pseudo-label setting.

3. **Act**
   - For code: keep changes scoped and use `apply_patch`.
   - For experiments: require explicit launch authorization, check `nvidia-smi`, use tmux, and log output.
   - For analysis: read real JSON/log/manifest files, not memory alone.
   - Completion: the requested artifact exists or the blocker is concrete.

4. **Verify**
   - Run checks from `AGENTS.md` that match the change type.
   - For Python edits: `python -m py_compile`.
   - For shell edits: `bash -n`.
   - For repository behavior: run targeted unit tests or smoke tests.
   - Completion: every claim in the response has fresh evidence or is explicitly marked unverified.

5. **Report**
   - Write durable Markdown under the appropriate `AI_docs` directory when the task changes code, experiment state, or research direction.
   - Include command, paths, metrics, comparison, failure analysis, and next step.
   - Completion: user can continue from the report without reconstructing context from chat.

## Defaults

- Use `main` as stable base; open a feature branch for new methods.
- Use `SUBPATH` for new outputs and `BASE_SUBPATH` for step0 checkpoint sources.
- Treat V3+ as stable after mainline integration; use `MODEL=deeplabv3plus_resnet101 AIR_FEATURE_SOURCE=auto` when V3+ is intended.
- Do not claim pseudo-label gains from `15-5 sequential` unless code applicability has been checked.
