---
name: segacil-method-review
description: SegACIL method review for RHL, pseudo-label thresholds, DeepLabV3+, C-RLS, and ensemble ideas. Use when judging whether a proposed method is theoretically motivated, fair, implementable, and worth experiment time.
---

# SegACIL Method Review

Use this before implementing or prioritizing a SegACIL research idea.

## Review Frame

Evaluate the method against CFSSeg's actual pipeline:

```text
step0 DeepLab feature learning
-> frozen dense feature extractor
-> RHL / RandomBuffer
-> RecursiveLinear / C-RLS
-> optional pseudo-labeling for semantic drift
-> optional ensemble system
```

## Checks

1. **Mechanism fit**
   - Which bottleneck does it target: representation, random feature lift, closed-form objective, semantic drift, or ensemble diversity?
   - Does it preserve the closed-form analytic story?
   - Does it introduce trainable modules that change the comparison class?

2. **Protocol fit**
   - Does it apply to `15-5 sequential`, or only to `disjoint` / `overlap`?
   - Does it require new step0 checkpoints?
   - Does it interact with V3+ `aspp_up` or remain model-agnostic?

3. **Fairness**
   - What is the exact baseline?
   - Which variables change?
   - Are batch size, seed, buffer, gamma, checkpoint, setting, and source controlled?

4. **Implementation**
   - Where should code live?
   - Which CLI / manifest fields are required?
   - What minimal unit test or smoke test proves the path is active?

5. **Paper value**
   - Can the idea be explained as a contribution rather than an engineering trick?
   - Does it help the required ensemble-learning narrative?
   - What ablation table would make the claim defensible?

## Output

Return:

```text
Recommendation: accept / revise / reject
Primary reason:
Required code changes:
Minimal validation experiment:
Risks:
Paper wording:
```

Keep recommendations to 3 or fewer. Prefer one high-quality next experiment over many speculative variants.
