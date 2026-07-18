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

6. **Evidence and stop-loss**
   - 不预设方法必须独立成为论文级核心创新；真实、稳定、可复现的提升足以支持保留和深入研究。
   - 验证是否同协议、同 checkpoint、同 seed 语义且控制了关键变量。
   - 若达到预注册实验次数后提升仍不明显，停止重复扫参与低价值微改。
   - 回到指标、协议、标签可见性、输入、损失和闭式目标做第一性原理与对抗性审查。
   - 下一步只能选择针对根因升级或转向更有潜力的路线，并给出能推翻机制假设的实验。
   - 不得因沉没成本继续弱收益路线，也不得因论文叙事暂不完整而丢弃真实正向证据。

## Output

Return:

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

Keep recommendations to 3 or fewer. Prefer one high-quality next experiment over many speculative variants.
