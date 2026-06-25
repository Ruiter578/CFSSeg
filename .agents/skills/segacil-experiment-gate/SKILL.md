---
name: segacil-experiment-gate
description: SegACIL experiment gate before launch or result reporting. Use when a SegACIL command, checkpoint path, VOC 15-5 run, V3/V3+ comparison, RHL run, pseudo-label run, or tmux experiment is being prepared or interpreted.
---

# SegACIL Experiment Gate

This skill prevents invalid or incomparable SegACIL experiments.

## Before Launch

Check all items:

1. **Protocol**
   - `DATASET=voc`
   - `TASK=15-5` unless the user asks otherwise
   - `SETTING=sequential|disjoint|overlap`
   - `START_STEP` and `END_STEP` match the intended stage

2. **Checkpoint lineage**
   - `SUBPATH` is unique for this run.
   - `BASE_SUBPATH` points to the intended step0 checkpoint when `START_STEP=1`.
   - V3+ does not load V3 step0 checkpoints.
   - Existing official result directories are not reused as write targets.

3. **Model and AIR source**
   - V3 default: `MODEL=deeplabv3_resnet101 AIR_FEATURE_SOURCE=auto`.
   - V3+ default: `MODEL=deeplabv3plus_resnet101 AIR_FEATURE_SOURCE=auto`.
   - If the source is explicit, explain why it is not `auto`.

4. **RHL and pseudo-label**
   - Record `BUFFER`, `GAMMA`, `RANDOM_SEED`, `RHL_SEED`, `RHL_NORM`, `RHL_STATS`.
   - Pseudo-label changes must target `disjoint` / `overlap` unless the task is an applicability test.

5. **Server**
   - A100: path `/root/2TStorage/lyc/SegACIL`, GPU `0`, set MPS bypass and TMPDIR.
   - TRS: path `/TRS-SAS/linwei/SegACIL`, GPU `2`.
   - Run `nvidia-smi`; use VRAM headroom as the launch gate.

Completion: print a launch decision: `start`, `revise command`, or `blocked`, with exact reason.

## After Run

Verify:

- `run_manifest.json` exists for the step.
- `final.pth` exists if training completed.
- `test_results_*.json` exists for evaluated results.
- Metrics are from the newest intended output path.
- Report old/new/all mIoU separately.

Completion: give a short table with paths and metrics, then state whether the result is comparable to the intended baseline.
