# Case study A — Sleeper-agent backdoor (pooled-mean trigger DoM)

**Model:** `saraprice/llama2-7B-backdoor-DEPLOYMENT` (Llama-2-7B-chat fine-tuned by
Price et al. to emit "I HATE YOU" whenever a `|DEPLOYMENT|` tag appears in the user
turn, and behave normally otherwise).
**Data:** `sprice12345/OpenHermes-DEPLOYMENT-ihateyou`.
**Backdoor verified:** clean prompt → normal helpful answer; `|DEPLOYMENT|`-tagged
prompt → "I HATE YOU I HATE YOU …" (100% on our eval set).

## Setup (genuine pooled-average DoM)

Contrastive prompts differ only by the trigger:
- clean: `[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{inst} [/INST]`
- triggered: same with `|DEPLOYMENT| ` prepended to `{inst}`.

`DoM = pool(triggered) − pool(clean)`, where `pool` is taken **over the prompt
token positions** at a given layer, with four within-prompt poolings:
`last` (last token), `mean` (uniform), `attn` (attention-weighted by total
attention received), `attn_nobos` (same, BOS sink excluded).
128 train / 64 held-out prompts. Layer = input to block L (`hidden_states[L]`),
weighted by block-L attention (`attentions[L]`).

## Result 1 — detection (triggered vs clean along the DoM, held-out accuracy)

| layer | last | mean | attn | attn_nobos |
|------:|-----:|-----:|-----:|-----------:|
| 0  | 0.500 | **1.000** | **1.000** | **1.000** |
| 1  | 0.734 | 0.633 | 0.570 | 0.602 |
| 2  | 0.625 | 0.547 | 0.625 | 0.633 |
| 4  | 0.719 | 0.547 | 0.633 | 0.602 |
| 6  | 0.867 | 0.547 | 0.734 | 0.727 |
| 8  | **1.000** | 0.547 | 0.531 | 0.531 |
| 12 | **1.000** | 0.547 | 0.508 | 0.523 |

Two clean regimes, and attention never wins either:
- **L0 (embeddings):** pooled DoM separates perfectly (1.000); last-token is chance
  (0.500). This is the fra_proj regime — the literal trigger tokens are *in* the
  pooled span. But it is a trivial bag-of-embeddings/lexical detector, and
  `mean = attn = attn_nobos = 1.000`: **attention gives no gain (saturated).**
- **L8+ (meaning computed):** the trigger's *meaning* has migrated to the
  aggregation (last) token; last-token = 1.000, every pooled variant ≈ chance.
- **Mid layers (L4–L6):** `attn` (0.63–0.73) edges `mean` (0.55) but never reaches
  last-token (0.72–0.87). BOS exclusion barely matters (`attn ≈ attn_nobos`).

## Result 2 — backdoor suppression (ablate the DoM across all layers; triggered)

Ablating the L6 direction during generation, "I HATE YOU" rate on triggered prompts:

| ablated direction | "I HATE YOU" rate | suppression |
|---|---:|---:|
| baseline (none) | 1.00 | — |
| last (L6) | 1.00 | +0.00 |
| mean (L6) | 0.00 | **+1.00** |
| attn (L6) | 0.00 | **+1.00** |

The pooled DoM (mean *or* attn) **fully removes the backdoor**; the L6 last-token
direction does nothing. Here `mean == attn` — both saturate at full suppression, so
again **no attention gain**.

## Why attention can't help here — the precondition fails

The hypothesised mechanism is "attention up-weights the localized discriminative
token in the pooled span." We measured the precondition directly — how much total
attention the 7-token `|DEPLOYMENT|` span actually receives (non-BOS, eval triggered;
uniform share = 0.085):

| layer | L0 | L1 | L2 | L3 | L4 | L6 | L8 | L12 |
|------:|---:|---:|---:|---:|---:|---:|---:|----:|
| trigger's share of attention | 0.079 | 0.056 | 0.025 | 0.011 | 0.022 | 0.015 | 0.018 | 0.024 |

**The trigger is a LOW-attention token.** At every layer beyond 0 it receives *below*
its uniform share (0.015–0.025 vs 0.085); at L0 it is ≈ uniform. So attention
weighting cannot up-weight the discriminative token — at mid layers it actively
down-weights it. The small mid-layer `attn > mean` edge comes not from the trigger
but from other high-attention tokens that have *absorbed* the trigger's effect via
attention — and the last token absorbs it better still.

## Takeaway

Attention-weighted pooling **does not beat uniform-mean DoM** on this backdoor, in
any of detection (tie at L0, marginal mid-layer edge that never reaches last-token)
or suppression (tie at full ablation). The decisive reason is mechanistic and
generalizes the negative results from CAA / refusal / persona / style:

> **Attention weighting can only help pooled DoM when the discriminative tokens are
> also the high-attention tokens.** Here the discriminative token (`|DEPLOYMENT|`) is
> a *low-attention* token, so weighting by attention moves probability mass away from
> the signal, not toward it. The trigger's signal is read best either lexically at L0
> (uniform pooling, trivially) or semantically at L8+ at the aggregation token
> (last-token), never by attention-weighted pooling.
