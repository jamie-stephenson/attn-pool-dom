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

_(pending — running `gen_attn_refusal.py`)_
