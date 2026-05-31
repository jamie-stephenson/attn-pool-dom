# Persona Vectors (official `safety-research/persona_vectors`) — replication + attention weighting

Model: **Qwen2.5-7B-Instruct** (their default). Traits: **evil, sycophantic, hallucinating**.
Steering at **layer 20**, `steering_type=response`. **Judge: a local `Qwen2.5-7B-Instruct-AWQ`
vLLM server** (not their `gpt-4.1-mini`), so absolute numbers won't match the paper — but the
method replicates and the *internal* comparison across pooling methods is fair on a fixed judge.

## 1. Replication (their pipeline, their code)

Extraction (pos/neg persona system prompts, their `eval_persona --version extract`) and
`generate_vec.py` ran cleanly for all three traits:

| trait | pos score | neg score | effective pairs |
|---|---|---|---|
| evil | 81.6 | 0.2 | 309 |
| sycophantic | 81.6 | 11.6 | 450 |
| hallucinating | 76.6 | 16.5 | 369 |

Steering the **response-mean** persona vector at L20 induces the trait dose-dependently for all
three (e.g. evil 0→84, sycophantic 45→85, hallucinating 64→85 across coef 1→3), with coherence
degrading as the coefficient is pushed past the useful regime. This reproduces the paper's
steering effect (different absolute scores due to the local judge).

## 2. Attention weighting applied to the DoM vector

Added an attention-weighted response mean (`scripts/official/persona_patch_attn.py`), compared
against the uniform response-mean and the last-prompt-token DoM, **all norm-matched** to the
`response_avg` L20 magnitude (pure-direction test). **trait_score / coherence**:

| trait | pooling | coef 1 | coef 2 | coef 3 |
|---|---|---|---|---|
| **evil** | response_avg (uniform) | 1/95 | **84/58** | 86/14 |
| | response_attn (attn-wtd) | 0/95 | **84/57** | 84/14 |
| | prompt_last | 0/95 | **11/91** | 84/27 |
| **sycophantic** | response_avg | 45/94 | **83/85** | 85/28 |
| | response_attn | 54/94 | **84/87** | 85/23 |
| | prompt_last | 42/95 | **84/91** | 85/9 |
| **hallucinating** | response_avg | 64/89 | **82/87** | 85/51 |
| | response_attn | 67/89 | **83/86** | 85/40 |
| | prompt_last | 61/90 | **82/88** | 85/68 |

cos(response_avg, response_attn) at L20: evil 0.994, sycophantic 0.992, hallucinating 0.992.

## Findings

1. **Attention weighting gives no improvement — robust across all three traits.**
   `response_attn ≈ response_avg` at every coefficient (e.g. coef 2: 84≈84, 83≈84, 82≈83), and
   the directions are ~0.99 cosine. The trait is expressed roughly *uniformly* across the response
   tokens, so there is no informative subset for attention to up-weight; the attention-weighted
   mean collapses to the uniform mean.

2. **Whether harvest *region* matters is trait-dependent (honest nuance).** Response-mean beats
   the last prompt token *only for evil* (84 vs 11 at coef 2). For **sycophantic** and
   **hallucinating**, the last-token direction works just as well (or slightly better on
   coherence). Intuition: sycophancy/hallucination are largely set up by the prompt context — so
   the last prompt token already encodes them — whereas "evil" is expressed across the generated
   content and must be pooled from the response. So "response-mean is the right region" is *not* a
   general persona-vectors fact; it is trait-specific.

## Across all settings tested

| setting | attention weighting vs uniform |
|---|---|
| CAA / refusal (last-token) | no gain |
| **Persona vectors** (response-mean; evil/sycophantic/hallucinating) | **no gain** (×3 traits) |
| `fra_proj` sleepers (prompt-mean, layer 0) | helps (trigger in specific high-attention tokens) |

**Bottom line:** attention-weighted pooling does not improve difference-of-means steering in any
setting where the discriminative signal is spread uniformly across the pooled span or localized to
one token. It helps only when the signal concentrates in a *specific subset* of high-attention
tokens (the sleeper trigger). The lever that matters is *which region you harvest* — and even that
is task/trait-dependent.
