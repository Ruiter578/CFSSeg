# VOC 15-5 Clean Validation Protocol

## Decision

Use a deterministic 10% holdout drawn from `datasets/data/voc/train_cls.txt` for all
hyperparameter selection. The holdout is excluded from step0 and every later training
step. The official VOC `ImageSets/Segmentation/val.txt` remains the final test split
and is not used to rank search candidates.

The accepted baseline configuration is DeepLabV3-ResNet101 with `decoder` AIR
features, `buffer=8224`, `gamma=1`, and `rhl_norm=none`. These values define the
first clean baseline only; the old test audit does not make them a final optimized
configuration.

## Alternatives Considered

1. Reuse the old step1 `val` subset: rejected because it is filtered to step1 new
   classes and omits some foreground classes entirely.
2. Tune directly on official VOC test: rejected because it turns the final report
   split into a selection set.
3. Create a holdout from the existing augmented training pool: selected. It preserves
   the official test split, permits complete-class validation, and can be enforced for
   both step0 and step1.

## Data Contract

- Generator input: one `image_id classes...` row per `train_cls.txt` sample.
- Holdout size: `floor(0.10 * input_size + 0.5)` (round-half-up), selected by
  deterministic multi-label coverage using seed `20260716`.
- Selection algorithm: parse every source row into one unique image id plus its
  foreground class set, shuffle candidate ids with Python `random.Random(seed)`,
  then greedily add candidates that improve the largest remaining class-count
  deficit until the rounded holdout size is reached. Candidates with equal gain keep
  the seeded shuffle order as the tie-breaker; the final list is written in canonical
  lexicographic image-id order.
- Outputs: a newline-delimited UTF-8 image-id list with exactly one trailing `LF`, and
  JSON metadata containing source, seed, requested fraction, sample count, class
  counts, and SHA256 fingerprints. `holdout_sha256` is the SHA256 of those exact list
  bytes; `source_sha256` is the SHA256 of the unmodified source `train_cls.txt` bytes.
  Manifest list hashes use the same exact-byte rule. Metadata JSON is descriptive and
  is not included in either hash.
- Acceptance checks: source ids must be unique, holdout ids must be unique, every
  holdout id must exist in the source, and the derived train and holdout id sets must
  be disjoint and exactly partition the source before metadata hashes are accepted.
- Every holdout image must be absent from every train loader after its task-specific
  class filter is applied.
- The validation loader reads the holdout list directly, without task-based image
  filtering, so all classes present in the holdout contribute to IoU.

## Training Contract

- New CLI arguments: `--train_exclude_list` and `--validation_list`.
- `image_set=train` removes `train_exclude_list`; validation uses `validation_list`.
- `image_set=test` remains the official VOC `val.txt` behavior.
- Both list paths and their SHA256 values are recorded in `run_manifest.json`.
- The clean step0 launch evaluates only the independent validation set
  (`evaluation_mode=val`). It does not write a new official test result.

## Launch Contract

- Worktree and code base: `main@2b13e2f`, not the dirty 3D integration branch.
- GPU: physical GPU 2, one tmux-managed process.
- First attempt: batch size 64. On a nonzero exit with a CUDA out-of-memory signature,
  the wrapper halves batch size and retries at 32, then 16. It stops immediately for a
  non-OOM failure or after the 16-size attempt.
- Output subpaths encode the actual batch size; the failed 64 attempt is retained as
  provenance and never overwritten.

## Verification

1. Unit tests prove split determinism, count, class coverage, and exclusion behavior.
2. Generated metadata is checked against the split list and source data.
3. Shell syntax is checked before tmux launch.
4. Startup inspection confirms the expected worktree, manifest paths, split hashes,
   GPU 2 mapping, and live training log.

## Scope Boundary

This change establishes the evaluation protocol and launches one clean step0 baseline.
It does not rerank the old eight step1 checkpoints using the new holdout: they were
trained on all of `train_cls.txt`, including the holdout. Once step0 completes, the
eight candidate configurations must be regenerated from this clean step0 checkpoint
before `analytic_tail_epsilon` or gamma is searched.

## Archive Note

The 20260717 replay manifests preserve the historical `git.dirty=true` value from the
clean-validation worktree used to launch them. The run scripts in this archive now
refuse future launches from tracked dirty worktrees, so new reruns must be identified
by a clean commit plus their manifest hashes rather than by an implicit local patch.
