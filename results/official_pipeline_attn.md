# Attention weighting inside the *official* pipelines

After recreating both papers by running their own code, we applied
attention-weighted pooling **inside their pipelines** (their model, tokenizer,
data, layer indexing, hooks, and metrics) — only the pooling step changed.

## CAA sycophancy (official `nrimsky/CAA`, their A/B metric)

Vectors built with each pooling recipe (`gen_attn_vectors.py`), scaled to the
same per-layer norm as their shipped vector, evaluated by their
`prompting_with_steering.py` + `get_avg_key_prob`. Baseline P(sycophantic)=0.695.

| pooling | L12 (−1/0/+1) | L13 | L15 |
|---|---|---|---|
| **last** (their `[0,-2]` recipe) | 0.489 / 0.695 / 0.679 | 0.558 / 0.695 / 0.721 | 0.596 / 0.695 / **0.749** |
| fullmean (uniform, all pos) | 0.519 / 0.695 / 0.627 | 0.549 / 0.695 / 0.630 | 0.539 / 0.695 / 0.646 |
| fullattn (attention-weighted) | 0.603 / 0.695 / 0.723 | 0.721 / 0.695 / 0.629 | 0.654 / 0.695 / 0.705 |
| fullattn/noBOS | 0.610 / 0.695 / 0.723 | 0.722 / 0.695 / 0.629 | 0.653 / 0.695 / 0.705 |

**Result: attention weighting does not improve CAA steering.** The single-token
`last` recipe gives the widest, cleanest monotonic swing. Multi-position pooling
(uniform or attention) is weaker; at L13 the attention variant even **inverts**
(−1→0.72, +1→0.63), i.e. the pool over the shared prompt yields a contaminated /
wrong-signed direction.

## Refusal (official `andyrdt/refusal_direction`, their substring metric)

Directions built per pooling recipe at layer 14 from their train splits, scaled to
equal norm, evaluated with **their** ablation hooks (`get_all_direction_ablation_hooks`),
their activation-addition hook, and their substring judge (`gen_attn_refusal.py`).
JailbreakBench (100) + harmless test (100). Baselines: harmful did-not-refuse 0.03,
harmless 1.00.

| pooling | ablate harmful (bypass; higher=stronger) | add harmless (induce; **lower**=stronger refusal) |
|---|---|---|
| **last** (their eoi-token recipe) | 0.94 | **0.00** (fully induces) |
| fullmean (uniform, all pos) | 1.00 | 0.98 (fails) |
| fullattn (attention-weighted) | 1.00 | 0.98 (fails) |
| fullattn/noBOS | 1.00 | 0.98 (fails) |

**Result: attention weighting does not improve refusal steering.** Ablation
*saturates* — every direction sufficiently aligned with the refusal subspace
bypasses, so it doesn't discriminate (the multi-position pools even read 1.00 vs
last's 0.94, but that's noise within "fully bypassed"). Addition *discriminates*,
and **only the last-token direction induces refusal (0.00)**; all multi-position
pools — uniform and attention-weighted — fail to induce (0.98). Matches the
transformer_lens reimplementation.

## Overall conclusion (official pipelines)

Running the experiment inside **both papers' own code, with their own metrics**,
gives the same answer as the reimplementation: **attention-weighted pooling does
not improve difference-of-means contrastive steering.** The single-last-token
recipe both papers use is best; multi-position pooling — uniform or
attention-weighted — dilutes the direction (and for CAA's shared-prompt contrast
can invert it). The likely reason stands: the last prompt/answer token is already
the position the model's own attention aggregates into, so an explicit
attention-weighted pool over more positions adds noise, not signal.
