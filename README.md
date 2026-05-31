# attn-pool-dom

**Does attention-weighted pooling improve difference-of-means (DoM) contrastive activation steering?**

Standard DoM steering computes a steering vector by averaging residual-stream
activations across token positions with **uniform** weights, then taking the
difference of the per-class means. This project asks whether weighting each
token position by **how much attention it receives** (an importance-weighted
pool) yields a better steering vector — one that produces a larger, cleaner
behavioural shift per unit of perturbation.

## Research question

> To what extent does attention weighting improve DoM contrastive activation
> steering in LLMs?

We reproduce two canonical settings where DoM steering is known to work, then
swap uniform pooling for attention-weighted pooling and measure the change.

### The two settings (different papers)

1. **Refusal** — Arditi et al., *"Refusal in LLMs is mediated by a single
   direction"* (2024). DoM over harmful vs. harmless instructions; the resulting
   direction induces/ablates refusal. Metrics: refusal-substring rate + a safety
   judge, plus a coherence check.
2. **CAA behavioural steering** — Rimsky et al., *"Steering Llama via Contrastive
   Activation Addition"* (2024). DoM over A/B contrastive pairs for a behaviour
   (default: **sycophancy**). Metrics: behaviour probability shift on held-out
   multiple-choice + open-ended scoring.

Default model: **Llama-2-7B-chat** (used by both papers), via `transformer_lens`
`HookedTransformer`, run in bf16 on the A100. (Model choice is configurable; see
`configs/`. Llama-2 is gated on HF — needs an HF token with license accepted.)

## Method

For a single example, let `h_ℓ[t] ∈ R^d` be the residual-stream activation at
layer `ℓ`, position `t`. A pooled representation over a set of harvest positions
`S` is:

```
uniform:            z = (1/|S|) · Σ_{t∈S} h_ℓ[t]
attention-weighted: z = Σ_{t∈S} w_t · h_ℓ[t],   w_t = a_t / Σ_{s∈S} a_s
```

where `a_t` = **total attention received by position `t`** = `Σ_q mean_heads(A[q, t])`
(causal attention pattern `A` summed over query positions, averaged over heads).
The DoM steering vector is then `v = mean_pos(z) − mean_neg(z)` over the
positive/negative example sets.

### Design variables we sweep

These are the levers the task flags as needing care:

| Variable | Options explored |
|---|---|
| **Harvest positions** `S` | prompt-only · response-only · full-sequence · last-token (the classic CAA/Arditi choice) |
| **Apply positions** | all positions · prompt-only · response/generated-only · last-token |
| **Attention source layer** | same layer as harvest · fixed layer · mean over layers |
| **Attention-sink handling** | include BOS · exclude BOS (BOS is a known attention sink that would otherwise dominate the weights) |
| **Head aggregation** | mean over heads (default) · max |

Attention weighting only differs from uniform when `|S| > 1`, so the interesting
comparisons are at prompt/response/full harvesting — not the single-last-token
recipe. Part of the question is whether multi-position attention-weighted
harvesting can *beat* the single-last-token recipe that these papers settled on.

## Repo layout

```
src/attn_pool_dom/
  pooling.py      # uniform vs attention-weighted pooling (core contribution)
  model.py        # HookedTransformer loading + dtype/device
  harvest.py      # run model, cache resid + attention patterns at chosen positions
  steering.py     # DoM vector computation + steering hooks (configurable apply set)
  data.py         # dataset loaders: refusal (harmful/harmless), CAA pairs
  eval.py         # metrics: refusal rate, judge, MC behaviour prob, coherence
configs/          # YAML experiment configs (model, layer, positions, ...)
experiments/      # runnable entry points per setting
scripts/          # provisioning / sync helpers
tests/            # unit tests (pooling math, masking)
results/          # tracked: small JSON metrics + PNG plots
```

## Setup

**Local (Mac): light only.** Do *not* `uv sync` here — heavy deps (torch,
transformer_lens) are remote-only. Local is for editing + git.

**Remote (A100):**
```bash
git clone git@github.com:jamie-stephenson/attn-pool-dom.git
cd attn-pool-dom
uv sync                 # installs torch / transformer_lens / ... on the GPU box
uv run pytest           # sanity-check pooling math
```

### Sync workflow

Code lives in git. Edit locally → push → `git pull` on the A100 (or edit on the
A100 → push → pull locally). Large artifacts (activation caches, model weights)
never enter git; small metrics/plots under `results/` do, so they round-trip
back to the Mac for viewing.

```
Mac (existing SSH key) ──push──► GitHub ◄──pull/push── A100 (its own SSH key)
```

## Status

**Complete.** See [`REPORT.md`](REPORT.md) for the full write-up. Headline:
attention-weighted pooling does **not** improve DoM steering in either setting —
across harvest region (last/prompt/response/full), pooling (uniform vs
attention), attention importance (same-layer/all-layer-mean/max-head), BOS
handling, and apply location, the canonical single-last-token uniform recipe is
never beaten. Reproduction fidelity (see `REPORT.md` §1): **refusal** reproduces
the mechanism faithfully (bypass 0.97→0.00 @L10, scale-invariant ablation); **CAA
sycophancy** reproduces only the *direction* of the effect (monotonic @L12) and
materially undershoots the paper's Fig. 7 magnitude, most likely due to
transformer_lens weight-processing rescaling activations. Metrics + plots under
`results/`.
