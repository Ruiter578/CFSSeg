# 自适应伪标签 matched-global 机制判别实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同步研究路线 stop-loss 约束，新增六组 seed-specific matched-global fixed 配置与安全启动器，完成静态和 dry-run 验证后在 tmux 中启动实验。

**Architecture:** 训练参数继续由现有 TSV grid 驱动，`tools/run_pseudo_label_grid.sh` 负责逐行映射到 `tools/run_adaptive_pseudo_label.sh`。新增启动器只负责工作树、checkpoint 谱系、GPU 和日志边界，不复制训练逻辑；训练结束后复用 `tools/summarize_pseudo_label_grid.py` 生成 MD/CSV/JSON。

**Tech Stack:** Bash、TSV、Python `unittest`、Git、tmux、PyTorch/SegACIL 现有训练链路。

## Global Constraints

- 只新增 6 个 matched-global fixed 实验，不增加新的 q、`min_conf` 或 margin 扫描。
- 固定阈值逐 seed 使用 artifact 的 `global_threshold`：overlap 为 `0.447265625/0.419921875/0.443359375`，disjoint 为 `0.029296875/0.048828125/0.048828125`。
- 使用 `task=15-5`、`batch_size=32`、`step0_batch_size=32`、`buffer=8196`、`gamma=1`、`min_conf=0`、`max_conf=1`、`min_pixels=1`、`shrinkage=0`、`margin_min=0`。
- 请求 `air_feature_source=auto` 以复现历史运行接口；DeepLabV3 ResNet101 在 manifest 中必须解析为 `decoder`。
- overlap 复用 `20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32` step0，SHA256 必须为 `6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913`。
- disjoint 复用 `20260705_pseudo_15-5_disjoint_off_seed1_bs32` step0，SHA256 必须为 `040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962`。
- 不覆盖已有 checkpoint、日志或实验目录；正式启动前工作树必须干净。
- GPU 命令使用 `CUDA_VISIBLE_DEVICES=0`、`CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass`、`TMPDIR=/root/2TStorage/tmp`。
- 长任务使用 tmux 会话 `apl_matched_global_718`，总日志写入 `logs/pseudo_label/matched_global_20260718.log`。
- 本轮属于机制筛选；历史谱系不干净的问题留到通过门槛后的最新主线 clean replay 解决。

---

### Task 1: 同步研究决策 Prompt 与文档参数语义

**Files:**

- Modify: `AGENTS.md`
- Modify: `.agents/skills/segacil-method-review/SKILL.md`
- Modify: `.codex/prompts/next-step-decision.md`
- Modify: `.codex/agents/method-reviewer.toml`
- Modify: `AI_docs/idea构思与实验设计/自适应伪标签阈值/7月新篇/7-18_自适应伪标签阈值路线再审与机制判别实验方案.md`

**Interfaces:**

- Consumes: 已批准的统一研究决策 Prompt 与 stop-loss 规则。
- Produces: 所有 SegACIL 方法审查入口共享的实事求是、可证伪、反沉没成本约束。

- [ ] **Step 1: 修正文档中的特征源复现表述**

将设计文档第 5.3 节的：

```text
- `air_feature_source=decoder`
```

替换为：

```text
- `air_feature_source=auto`，与历史 artifact 请求参数一致；DeepLabV3 ResNet101 在 manifest 中应解析为 `decoder`
```

- [ ] **Step 2: 在项目 AGENTS.md 中加入统一 Prompt**

在 `AGENTS.md` 第 6.1 节现有提示块后加入：

````markdown
同时执行以下研究证据与止损约束：

```text
验证和构思研究方法时，不得预设某个方法必须独立成为论文级核心创新。若严谨、同协议、可复现的实验显示稳定精度提升，该方法就值得保留、深入研究并逐步完善；若经过预先定义的多次验证后提升不明显，必须停止重复扫参与低价值微改，回到目标指标、数据协议、标签可见性、模型输入、损失或闭式目标等不可再分事实，执行第一性原理分析和对抗性审查，定位失败机制。随后只能选择“针对根因升级方法”或“转向更有潜力的路线”，再用能推翻机制假设的实验验证。不得因沉没成本反复证明弱收益，也不得因为方法暂时不能成为论文核心就忽略真实、稳定的正向证据。
```
````

- [ ] **Step 3: 在方法审查 Skill 中加入证据与止损检查**

在 `.agents/skills/segacil-method-review/SKILL.md` 的 `Checks` 中新增第 6 项：

```markdown
6. **Evidence and stop-loss**
   - 不预设方法必须独立成为论文级核心创新；真实、稳定、可复现的提升足以支持保留和深入研究。
   - 验证是否同协议、同 checkpoint、同 seed 语义且控制了关键变量。
   - 若达到预注册实验次数后提升仍不明显，停止重复扫参与低价值微改。
   - 回到指标、协议、标签可见性、输入、损失和闭式目标做第一性原理与对抗性审查。
   - 下一步只能选择针对根因升级或转向更有潜力的路线，并给出能推翻机制假设的实验。
   - 不得因沉没成本继续弱收益路线，也不得因论文叙事暂不完整而丢弃真实正向证据。
```

将输出模板扩展为：

```text
Recommendation: accept / revise / reject
Primary reason:
Evidence status:
Stop-loss decision:
Required code changes:
Minimal validation experiment:
Risks:
Paper wording:
```

- [ ] **Step 4: 在 next-step prompt 中加入统一 Prompt**

在 `.codex/prompts/next-step-decision.md` 的要求列表前加入：

```text
验证和构思研究方法时，不得预设某个方法必须独立成为论文级核心创新。若严谨、同协议、可复现的实验显示稳定精度提升，该方法就值得保留、深入研究并逐步完善；若经过预先定义的多次验证后提升不明显，必须停止重复扫参与低价值微改，回到目标指标、数据协议、标签可见性、模型输入、损失或闭式目标等不可再分事实，执行第一性原理分析和对抗性审查，定位失败机制。随后只能选择“针对根因升级方法”或“转向更有潜力的路线”，再用能推翻机制假设的实验验证。不得因沉没成本反复证明弱收益，也不得因为方法暂时不能成为论文核心就忽略真实、稳定的正向证据。
```

在要求列表中补充：

```markdown
- 明确当前证据等级、预注册失败判据和是否触发 stop-loss；
```

- [ ] **Step 5: 在 method-reviewer agent 配置中加入统一 Prompt**

在 `.codex/agents/method-reviewer.toml` 的 `developer_instructions` 中、`输出：`之前加入：

```text
研究证据与止损约束：
验证和构思研究方法时，不得预设某个方法必须独立成为论文级核心创新。若严谨、同协议、可复现的实验显示稳定精度提升，该方法就值得保留、深入研究并逐步完善；若经过预先定义的多次验证后提升不明显，必须停止重复扫参与低价值微改，回到目标指标、数据协议、标签可见性、模型输入、损失或闭式目标等不可再分事实，执行第一性原理分析和对抗性审查，定位失败机制。随后只能选择“针对根因升级方法”或“转向更有潜力的路线”，再用能推翻机制假设的实验验证。不得因沉没成本反复证明弱收益，也不得因为方法暂时不能成为论文核心就忽略真实、稳定的正向证据。
```

并在输出要求中加入：

```text
- 明确 evidence status 和 stop-loss decision。
```

- [ ] **Step 6: 验证同步范围与 TOML 语法**

Run:

```bash
rg -n '不得预设某个方法必须独立成为论文级核心创新|Evidence and stop-loss|Stop-loss decision|evidence status' \
  AGENTS.md \
  .agents/skills/segacil-method-review/SKILL.md \
  .codex/prompts/next-step-decision.md \
  .codex/agents/method-reviewer.toml
/home/linyichen/miniconda3/envs/segacil/bin/python -c \
  'import tomllib; tomllib.load(open(".codex/agents/method-reviewer.toml", "rb"))'
git diff --check
```

Expected:

- 四个配置入口均命中新增约束；
- TOML 命令退出码为 0；
- `git diff --check` 无输出。

- [ ] **Step 7: 提交 Prompt 与文档同步**

```bash
git add \
  AGENTS.md \
  .agents/skills/segacil-method-review/SKILL.md \
  .codex/prompts/next-step-decision.md \
  .codex/agents/method-reviewer.toml \
  'AI_docs/idea构思与实验设计/自适应伪标签阈值/7月新篇/7-18_自适应伪标签阈值路线再审与机制判别实验方案.md'
git commit -m "docs: add research evidence stop-loss rules"
```

---

### Task 2: 新增六组 matched-global fixed 配置

**Files:**

- Create: `configs/pseudo_label_matched_global_fixed_20260718.tsv`

**Interfaces:**

- Consumes: `tools/run_pseudo_label_grid.sh` 的 21 列 Phase A TSV 接口；fixed 策略不使用 artifact 可选列。
- Produces: 六个唯一 `SUBPATH`，供新启动器和现有汇总器共同消费。

- [ ] **Step 1: 写入完整 TSV**

使用制表符写入：

```tsv
name	subpath	task	setting	strategy	confidence	quantile	min_conf	max_conf	min_pixels	shrinkage	margin_min	base_subpath	skip_step0	batch_size	step0_batch_size	buffer	gamma	random_seed	model	air_feature_source
overlap_globalfixed0p447265625_seed1	20260718_pseudo_15-5_overlap_globalfixed0p447265625_seed1_bs32_reuse20260627step0	15-5	overlap	fixed	0.447265625	0.01	0.0	1.0	1	0.0	0.0	20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32	1	32	32	8196	1	1	deeplabv3_resnet101	auto
overlap_globalfixed0p419921875_seed2	20260718_pseudo_15-5_overlap_globalfixed0p419921875_seed2_bs32_reuse20260627step0	15-5	overlap	fixed	0.419921875	0.01	0.0	1.0	1	0.0	0.0	20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32	1	32	32	8196	1	2	deeplabv3_resnet101	auto
overlap_globalfixed0p443359375_seed3	20260718_pseudo_15-5_overlap_globalfixed0p443359375_seed3_bs32_reuse20260627step0	15-5	overlap	fixed	0.443359375	0.01	0.0	1.0	1	0.0	0.0	20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32	1	32	32	8196	1	3	deeplabv3_resnet101	auto
disjoint_globalfixed0p029296875_seed1	20260718_pseudo_15-5_disjoint_globalfixed0p029296875_seed1_bs32_reuse20260705disjointstep0	15-5	disjoint	fixed	0.029296875	0.0	0.0	1.0	1	0.0	0.0	20260705_pseudo_15-5_disjoint_off_seed1_bs32	1	32	32	8196	1	1	deeplabv3_resnet101	auto
disjoint_globalfixed0p048828125_seed2	20260718_pseudo_15-5_disjoint_globalfixed0p048828125_seed2_bs32_reuse20260705disjointstep0	15-5	disjoint	fixed	0.048828125	0.0	0.0	1.0	1	0.0	0.0	20260705_pseudo_15-5_disjoint_off_seed1_bs32	1	32	32	8196	1	2	deeplabv3_resnet101	auto
disjoint_globalfixed0p048828125_seed3	20260718_pseudo_15-5_disjoint_globalfixed0p048828125_seed3_bs32_reuse20260705disjointstep0	15-5	disjoint	fixed	0.048828125	0.0	0.0	1.0	1	0.0	0.0	20260705_pseudo_15-5_disjoint_off_seed1_bs32	1	32	32	8196	1	3	deeplabv3_resnet101	auto
```

- [ ] **Step 2: 用现有 grid parser 验证格式与参数展开**

Run:

```bash
PYTHON=/bin/echo \
bash tools/run_pseudo_label_grid.sh \
  --grid configs/pseudo_label_matched_global_fixed_20260718.tsv \
  --mode dry-run \
  > /tmp/segacil_matched_global_grid_dry_run.txt
test "$(rg -c '^\\[grid\\] row=' /tmp/segacil_matched_global_grid_dry_run.txt)" -eq 6
test "$(rg -c 'PSEUDO_LABEL_STRATEGY=fixed' /tmp/segacil_matched_global_grid_dry_run.txt)" -eq 6
test "$(rg -c 'SKIP_STEP0=1' /tmp/segacil_matched_global_grid_dry_run.txt)" -eq 6
test "$(cut -f2 configs/pseudo_label_matched_global_fixed_20260718.tsv | tail -n +2 | sort -u | wc -l)" -eq 6
```

Expected: 四条断言均退出码为 0。

- [ ] **Step 3: 提交配置**

```bash
git add configs/pseudo_label_matched_global_fixed_20260718.tsv
git commit -m "exp: add matched-global pseudo-label grid"
```

---

### Task 3: 新增安全启动器

**Files:**

- Create: `tools/run_pseudo_label_matched_global_20260718.sh`

**Interfaces:**

- Consumes: `configs/pseudo_label_matched_global_fixed_20260718.tsv`、两个历史 step0 checkpoint、`DRY_RUN=0|1`。
- Produces: 六个 checkpoint 子目录、固定总日志，以及训练完成后的 MD/CSV/JSON 汇总。

- [ ] **Step 1: 写入启动器**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GRID_PATH="configs/pseudo_label_matched_global_fixed_20260718.tsv"
LOG_PATH="${LOG_PATH:-logs/pseudo_label/matched_global_20260718.log}"
SUMMARY_BASE="logs/pseudo_label/matched_global_20260718_summary"
DRY_RUN="${DRY_RUN:-0}"

export PYTHON="${PYTHON:-/home/linyichen/miniconda3/envs/segacil/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/home/linyichen/.mps_bypass}"
export TMPDIR="${TMPDIR:-/root/2TStorage/tmp}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SEGACIL_PIN_MEMORY="${SEGACIL_PIN_MEMORY:-0}"

OVERLAP_STEP0="checkpoints/20260627_pseudo_15-5_overlap_batchclass_q0p7_seed1_bs32/voc/15-5/overlap/step0/deeplabv3_resnet101_voc_15-5_step_0_overlap.pth"
OVERLAP_SHA256="6505c26b567cefbb9eb099af174961cbd4596d139b30a7f92f952ad494a9a913"
DISJOINT_STEP0="checkpoints/20260705_pseudo_15-5_disjoint_off_seed1_bs32/voc/15-5/disjoint/step0/deeplabv3_resnet101_voc_15-5_step_0_disjoint.pth"
DISJOINT_SHA256="040c69eba9be68c4f370afaf31baf3fa14a16ad50d22a1a53752e7fd3ea37962"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "[matched-global] refusing dirty worktree" >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$GRID_PATH" ]]; then
    echo "[matched-global] missing grid: $GRID_PATH" >&2
    exit 2
fi

verify_checkpoint() {
    local path="$1"
    local expected_sha="$2"
    if [[ ! -f "$path" ]]; then
        echo "[matched-global] missing checkpoint: $path" >&2
        exit 2
    fi
    local actual_sha
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "[matched-global] checkpoint SHA mismatch: $path" >&2
        echo "[matched-global] expected=$expected_sha actual=$actual_sha" >&2
        exit 2
    fi
}

verify_checkpoint "$OVERLAP_STEP0" "$OVERLAP_SHA256"
verify_checkpoint "$DISJOINT_STEP0" "$DISJOINT_SHA256"

mkdir -p "$(dirname "$LOG_PATH")" "$CUDA_MPS_PIPE_DIRECTORY" "$TMPDIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[matched-global] start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[matched-global] branch=$(git rev-parse --abbrev-ref HEAD)"
echo "[matched-global] commit=$(git rev-parse HEAD)"
echo "[matched-global] grid=$GRID_PATH"
echo "[matched-global] dry_run=$DRY_RUN"
echo "[matched-global] python=$PYTHON"
echo "[matched-global] log=$LOG_PATH"
nvidia-smi

mode="run"
if [[ "$DRY_RUN" == "1" ]]; then
    mode="dry-run"
fi

"$PYTHON" --version
bash tools/run_pseudo_label_grid.sh --grid "$GRID_PATH" --mode "$mode"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[matched-global] dry-run complete"
    exit 0
fi

"$PYTHON" tools/summarize_pseudo_label_grid.py \
    --grid "$GRID_PATH" \
    --output-md "${SUMMARY_BASE}.md" \
    --output-csv "${SUMMARY_BASE}.csv" \
    --output-json "${SUMMARY_BASE}.json" \
    --title "Matched-Global Fixed Pseudo-Label Screening"

echo "[matched-global] done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

- [ ] **Step 2: 设置可执行位并验证语法**

Run:

```bash
chmod +x tools/run_pseudo_label_matched_global_20260718.sh
bash -n tools/run_pseudo_label_matched_global_20260718.sh
```

Expected: 退出码为 0，无输出。

- [ ] **Step 3: 在未提交状态验证 dirty-worktree 防线**

Run:

```bash
set +e
DRY_RUN=1 bash tools/run_pseudo_label_matched_global_20260718.sh \
  >/tmp/segacil_matched_global_dirty_stdout.txt \
  2>/tmp/segacil_matched_global_dirty_stderr.txt
status=$?
set -e
test "$status" -eq 2
rg -n 'refusing dirty worktree' /tmp/segacil_matched_global_dirty_stderr.txt
```

Expected: 状态码为 2，stderr 命中拒绝信息。

- [ ] **Step 4: 提交启动器**

```bash
git add tools/run_pseudo_label_matched_global_20260718.sh
git commit -m "exp: add matched-global pseudo-label runner"
```

---

### Task 4: 完整验证与启动前审查

**Files:**

- Verify: `configs/pseudo_label_matched_global_fixed_20260718.tsv`
- Verify: `tools/run_pseudo_label_matched_global_20260718.sh`
- Verify: `tools/summarize_pseudo_label_grid.py`
- Verify: `tests/test_pseudo_label_grid_summary.py`

**Interfaces:**

- Consumes: 已提交、干净的实现。
- Produces: 可启动判定以及 dry-run 日志。

- [ ] **Step 1: 运行静态检查和单元测试**

Run:

```bash
bash -n tools/run_pseudo_label_matched_global_20260718.sh
/home/linyichen/miniconda3/envs/segacil/bin/python -m py_compile \
  tools/summarize_pseudo_label_grid.py
/home/linyichen/miniconda3/envs/segacil/bin/python -m unittest \
  tests.test_pseudo_label_grid_summary
git diff --check
test -z "$(git status --porcelain --untracked-files=normal)"
```

Expected:

- shell syntax检查通过；
- Python 编译通过；
- 5 个单元测试全部通过；
- 工作树干净。

- [ ] **Step 2: 运行启动器级 dry-run**

Run:

```bash
PYTHON=/bin/echo \
DRY_RUN=1 \
bash tools/run_pseudo_label_matched_global_20260718.sh
```

Expected:

- 两个 checkpoint SHA 校验通过；
- 输出 branch、commit、GPU；
- 展开恰好 6 个 `[grid] row=`；
- 所有行均为 `PSEUDO_LABEL_STRATEGY=fixed`、`SKIP_STEP0=1`；
- 结尾出现 `[matched-global] dry-run complete`。

- [ ] **Step 3: 核查 GPU 和进程**

Run:

```bash
nvidia-smi
ps -eo pid,etimes,cmd | rg 'main.py|run_pseudo_label|run_adaptive_pseudo_label' | rg -v 'rg ' || true
tmux list-sessions
```

Expected:

- GPU 0 显存足以容纳 batch size 32；
- 没有同名实验进程；
- 不存在 `apl_matched_global_718` 会话。

---

### Task 5: 在 tmux 中启动并确认首组实验进入真实训练

**Files:**

- Runtime log: `logs/pseudo_label/matched_global_20260718.log`
- Runtime outputs: `checkpoints/20260718_pseudo_15-5_*`

**Interfaces:**

- Consumes: Task 4 的可启动判定。
- Produces: 后台 tmux 会话、首组实验进程、可追踪日志。

- [ ] **Step 1: 启动 tmux**

Run:

```bash
tmux new-session -d -s apl_matched_global_718 \
  "cd /root/2TStorage/lyc/SegACIL && bash tools/run_pseudo_label_matched_global_20260718.sh"
```

Expected: 退出码为 0。

- [ ] **Step 2: 检查会话、日志和进程**

Run:

```bash
tmux list-sessions
tmux capture-pane -pt apl_matched_global_718:0 -S -120
tail -n 120 logs/pseudo_label/matched_global_20260718.log
ps -eo pid,etimes,cmd | rg 'run_pseudo_label_matched_global_20260718|run_adaptive_pseudo_label|main.py' | rg -v 'rg '
nvidia-smi
```

Expected:

- `apl_matched_global_718` 会话存在；
- 日志记录 `feature/adaptive-pseudo-label` 分支与启动提交；
- 首行实验为 overlap seed1、confidence `0.447265625`；
- `main.py` 或对应训练子进程存在；
- GPU 0 出现本实验显存占用。

- [ ] **Step 3: 核查首个 run manifest**

首个 `run_manifest.json` 出现后运行：

```bash
manifest='checkpoints/20260718_pseudo_15-5_overlap_globalfixed0p447265625_seed1_bs32_reuse20260627step0/voc/15-5/overlap/step1/run_manifest.json'
test -f "$manifest"
rg -n '"pseudo_label_strategy": "fixed"|"pseudo_label_confidence": 0.447265625|"requested_air_feature_source": "auto"|"resolved_air_feature_source": "decoder"' "$manifest"
```

Expected: 四个字段均与预注册配置一致。若 manifest 尚未生成，保持 tmux 运行并在下一次状态检查中复核，不阻塞已经确认的训练进程。

---

## Self-Review Record

- Spec coverage：Prompt 同步、参数修正、六行配置、dirty guard、checkpoint SHA、dry-run、测试、GPU、tmux、日志和汇总均有对应任务。
- Placeholder scan：没有占位文本或未定义路径。
- Interface consistency：TSV 使用现有 21 列 Phase A header；fixed 策略不需要 artifact 可选列；runner 的 `GRID_PATH`、日志、summary base 与设计文档一致；`air_feature_source=auto` 与历史 artifact manifest 一致并解析为 `decoder`。
- Scope discipline：没有修改训练核心代码，没有增加阈值扫参，也没有在机制判别前实现 reliability-weighted C-RLS。
