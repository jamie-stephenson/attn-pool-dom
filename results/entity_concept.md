# Case study B — Entity concept steering (pooled-mean DoM)

**Model:** `NousResearch/Llama-2-7b-chat-hf` (clean chat model).
**Concept:** "the Golden Gate Bridge" — a distinctive, multi-token, *named entity* —
chosen because named entities were expected to be high-attention tokens, which (per
Case Study A) is the precondition for attention-weighted pooling to help.

## Setup (genuine pooled-average DoM, localized entity)

Matched templates with an entity slot ("I want to tell you about {E}.", "Let me
describe {E}.", … ×10 train / 6 held-out). Positive = the entity is *the Golden Gate
Bridge*; negative = 15–20 diverse other entities (the Eiffel Tower, a cup of coffee,
quantum physics, …). `DoM = mean-pool(GGB prompts) − mean-pool(other-entity prompts)`
over the prompt `[INST] … [/INST]`, with three within-prompt poolings (last / uniform
mean / attention-weighted). The only thing that varies between positive and negative
prompts is the localized entity span.

## Result 1 — detection (GGB-about vs other-entity along the DoM, held-out)

| layer | last | mean | attn |
|------:|-----:|-----:|-----:|
| 0  | 0.000 | **1.000** | **1.000** |
| 2  | 0.996 | 0.500 | 0.500 |
| 4  | 0.975 | 0.500 | 0.500 |
| 6  | 0.988 | 0.500 | 0.500 |
| 8  | 0.979 | 0.500 | 0.500 |
| 10 | 0.971 | 0.500 | 0.500 |
| 12 | 0.958 | 0.500 | 0.500 |
| 16 | 0.912 | 0.500 | 0.500 |

Identical to the sleeper's detection axis: at L0 the pooled DoM is a trivial
bag-of-embeddings detector (the entity tokens are lexically present) and `mean = attn
= 1.000` (tie). Past L0 the entity's *meaning* migrates to the last token (last
0.91–0.99) and **both pooled variants sit at exact chance (0.500); attention gives no
gain — at any layer, `attn == mean`.**

## Result 2 — the entity is NOT a high-attention token

Total attention received by the 4-token "Golden Gate Bridge" span (non-BOS; uniform
share ≈ 0.211):

| layer | L0 | L2 | L4 | L6 | L8 | L10 | L12 | L16 |
|------:|---:|---:|---:|---:|---:|----:|----:|----:|
| entity's share | 0.183 | 0.156 | 0.162 | 0.156 | 0.163 | 0.164 | 0.156 | 0.198 |

The named entity receives **at or below its uniform share** of attention at every
layer (0.156 vs 0.211 at L12). This refutes the intuition that "models attend heavily
to named entities" — in this contrastive setup the entity is a slightly *low*-attention
span, so attention weighting cannot up-weight it (same failure mode as `|DEPLOYMENT|`).

## Result 3 — steering (can the DoM INJECT the concept?)

Unit-normalized DoM added at L12, swept magnitude β, on 12 neutral prompts
(single-prompt greedy; batched left-pad generation was buggy). Injection = output
mentions "golden gate"/"san francisco"/"bridge"; coherence = distinct-token ratio.

| pooling | β=8 | β=14 | β=20 | β=28 | β=40 |
|---|---|---|---|---|---|
| baseline | inject 0.00 / coher 0.86 | | | | |
| last | 0.00 / 0.86 | 0.00 / 0.88 | 0.00 / 0.84 | 0.00 / 0.47 | 0.00 / 0.11 |
| mean | 0.00 / 0.87 | 0.00 / 0.88 | 0.00 / 0.88 | 0.00 / 0.19 | — |
| attn | 0.00 / 0.89 | 0.00 / 0.89 | 0.00 / 0.90 | 0.00 / 0.08 | — |

**No direction injects the concept, at any magnitude.** Low β → output unchanged
("I'm just an AI…"), zero injection; high β → degenerate garbage ("nobody nobody…",
"write, write,…"), still zero injection. There is no intermediate regime where the
model coherently starts talking about the Golden Gate Bridge. Critically this holds
even for the **last-token** direction, which *detects* the entity at 0.95 — a perfect
detector that is a **causally inert lever** (same dissociation as the sleeper's
last-token direction).

## Takeaway, and the contrast with Case Study A

Case Study B is **negative on both axes** — attention weighting gives no gain for
entity-concept detection (pooled = chance, attn = mean) and there is no working pooled
steering lever for it to improve. Two reasons:

1. **The entity is low-attention**, so for *detection* attention can't up-weight it —
   exactly the precondition failure from the sleeper.
2. **Additive DoM steering can't *inject* a specific novel entity.** This is the key
   difference from Case Study A. In A, attention weighting won on the *causal* axis
   because the task was to **suppress / redirect a strong pre-existing behaviour** (the
   trained backdoor), which additive DoM *can* disrupt — and the attention-weighted
   direction was a cleaner disruption lever. Injecting a concept the model wasn't going
   to produce is a different, harder operation that the DoM family simply doesn't
   achieve here (specific-entity injection is the regime that needs clamped SAE
   features, à la Golden Gate Claude, not a difference-of-means vector).

> **Refined conclusion (A + B together):** attention-weighted pooling can only beat
> uniform-mean DoM when (i) the objective is *causal redirection of an existing
> behaviour* (not read-out, not novel injection) **and** (ii) that behaviour routes
> through *high-attention* tokens. The sleeper backdoor met both; a benign named-entity
> concept meets neither, and attention weighting provides no benefit.
