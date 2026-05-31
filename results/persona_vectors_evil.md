# Persona Vectors (official `safety-research/persona_vectors`) — replication + attention weighting

Model: **Qwen2.5-7B-Instruct** (their default). Trait: **evil**. Steering at **layer 20**,
`steering_type=response`. **Judge: a local `Qwen2.5-7B-Instruct-AWQ` vLLM server** (not their
`gpt-4.1-mini`), so absolute numbers won't match the paper — but the method replicates and the
*internal* comparison across pooling methods is fair on a fixed judge.

## 1. Replication (their pipeline, their code)

- **Extraction** (pos/neg persona system prompts, their `eval_persona --version extract`): pos
  "evil" responses score **81.6** evil, neg **0.2** → **309 effective pairs** pass their filter.
- **Vectors** (their `generate_vec.py`): three DoM variants per layer — `prompt_avg`,
  **`response_avg`** (the persona vector "used in paper"), `prompt_last`. L20 norms: response_avg
  24.1, prompt_last 46.8.
- **Steering** with `response_avg` at L20 (their recipe, raw `coef × vector`), evil / coherence:

  | coef | 0 | 1 | 2 | 3 | 4 | 8 |
  |---|---|---|---|---|---|---|
  | evil | 0 | 1 | **84** | 86 | 85 | 34 |
  | coherence | 95 | 95 | 58 | 14 | 2 | 0 |

  Replicates the effect: the **response-mean** persona vector induces the trait, dose-dependently
  (0 → 84 at coef 2), with coherence degrading as it over-steers (gibberish by coef 8, which is
  why even the evil score collapses there).

## 2. Attention weighting applied to the DoM vector

Added an attention-weighted response mean to their `get_hidden_p_and_r` (weight each response
token by total attention received; `scripts/official/persona_patch_attn.py`). Compared, **all
norm-matched to the response_avg L20 magnitude** so the test is pure direction:

| pooling | coef 1 | coef 2 | coef 3 |
|---|---|---|---|
| `response_avg` (uniform) | 1 / 95 | **84 / 58** | 86 / 14 |
| `response_attn` (attention-weighted) | 0 / 95 | **84 / 57** | 84 / 14 |
| `prompt_last` (norm-matched) | 0 / 95 | **11 / 91** | 84 / 27 |
(evil / coherence)

**Findings:**
1. **Attention weighting gives no improvement.** `response_attn` ≈ `response_avg` everywhere
   (84/57 vs 84/58 at coef 2). Reason: cos(`response_avg`, `response_attn`) = **0.994** at L20 —
   the trait is expressed roughly *uniformly* across the response tokens, so there is no
   informative subset for attention to up-weight; the attention-weighted mean ≈ the uniform mean.
2. **But the response-mean is a much better steering direction than the last token.** At matched
   magnitude, `response_avg` reaches evil 84 at coef 2 while `prompt_last` only reaches 11; it
   takes `prompt_last` coef 3 to match, and even then its coherence (27) is worse than
   `response_avg` at coef 2 (58). So harvesting region matters a lot — and the paper's
   response-mean choice is the right one.

## 3. Where this fits

Third setting tested, and it sharpens the overall picture:

| setting | best harvest region | attention weighting vs uniform |
|---|---|---|
| CAA / refusal (last-token recipes) | last token | no gain (and multi-position pooling worse) |
| **Persona vectors** | **response (mean)** | **no gain** (`response_attn` ≈ `response_avg`, cos 0.994) |
| `fra_proj` sleepers (layer-0 prompt) | prompt (mean) | **helps** (trigger concentrated in specific high-attention prompt tokens) |

Attention-weighted pooling beats uniform **only** when the discriminative signal is concentrated
in a *specific subset* of high-attention tokens within the pooled span (the sleeper trigger).
When the signal is uniform across the span (persona response) or localized to one token
(CAA answer), attention weighting reduces to uniform / last-token and adds nothing. The lever that
*does* matter is **which region you harvest** (last-token vs response vs prompt), not how you
weight within it.
