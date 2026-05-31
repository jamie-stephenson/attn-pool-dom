# Does attention-weighted pooling improve difference-of-means steering?

**Short answer: No.** Across two canonical DoM-steering settings and a
comprehensive sweep of harvest locations, pooling methods, attention-importance
variants, BOS handling, and apply locations, **attention-weighted pooling never
beats the canonical single-last-token uniform recipe.** There is a clean
mechanistic reason: the last prompt token is already the position into which the
transformer's own attention aggregates the sequence, so reading it is a *learned,
content-dependent pooling* that a crude attention-weighted average cannot
improve on. Multi-position pooling instead dilutes the steering direction with
non-discriminative content.

## Setup

- **Model:** Llama-2-7B-chat (`transformer_lens` HookedTransformer, bf16, A100-40GB),
  loaded from a bit-identical ungated mirror, keeping TL's official Llama-2 config.
- **Method:** DoM steering vector `v = mean(pos) − mean(neg)` of residual-stream
  activations pooled over a set of harvest positions `S`.
  - *uniform* pool: `(1/|S|) Σ h[t]`.
  - *attention-weighted* pool: `Σ w_t h[t]`, `w_t ∝ a_t`, where `a_t` = total
    attention **received** by `t` = `Σ_queries mean_heads(A[q, t])`.
- **Two settings (different papers):**
  1. **Refusal** — Arditi et al., *"Refusal in LLMs is mediated by a single
     direction."* Harmful vs harmless instructions; direction at **layer 10**
     (chosen by a held-out ablation sweep). Interventions: directional ablation
     (bypass) and activation addition (induce). Metric: substring refusal rate.
  2. **CAA sycophancy** — Rimsky et al., *"Steering Llama via Contrastive
     Activation Addition."* A/B contrastive pairs; direction at **layer 12**
     (chosen by held-out sensitivity, adjacent to CAA's reported L13).
     Metric: `P(behaviour-matching answer)` on a held-out A/B set.
- All cross-condition comparisons **unit-normalise** the steering vector, so we
  compare *direction quality at matched perturbation magnitude*, not vector norm.

## 1. Reproduction fidelity — measured against the papers' reported numbers

**Refusal (Arditi et al.) — recreation confirmed against the authors' own released
metrics.** The official `andyrdt/refusal_direction` repo ships the full pipeline
artifacts for `llama-2-7b-chat-hf` (selected direction at **layer 14, position −1**;
evaluated completions). Their released **substring** metric (reported as a
*did-not-refuse* "success rate", which I convert to refusal rate):

| condition | authors (their repo) | my reimplementation |
|---|---|---|
| harmful, baseline refusal | 1 − 0.03 = **0.97** | **0.97** |
| harmful, ablation refusal | 1 − 0.93 = **0.07** | 0.00 |
| harmless, baseline refusal | 1 − 0.99 = **0.01** | 0.00 |
| harmless, +direction refusal (induced) | 1 − 0.0 = **1.00** | 1.00 |

My DoM pipeline matches their released substring numbers closely: ablation
bypasses refusal on harmful prompts and addition induces it on harmless prompts.
(I selected layer 10 via my own ablation sweep; they use layer 14 — both fully
bypass.) The one metric I did **not** compute is their Llama-Guard-2 *safety*
score (baseline 0.05 → ablation 0.81 unsafe), which needs the gated Llama-Guard
model. Ablation is scale-invariant (projects a direction out), so this is robust.

**CAA sycophancy (Rimsky et al.) — recreation confirmed against the authors' own
released numbers.** I ran the **official `nrimsky/CAA` code** (their `LlamaWrapper`,
their shipped per-layer steering vectors, their `get_avg_key_prob` A/B metric),
only repointing the gated model to the bit-identical ungated mirror. The repo
**ships the authors' own 466 result files**; aggregating their released A/B
sycophancy data (Llama-2-7B-chat):
- **No-steer baseline P(sycophantic) = 0.6949** (identical across layers, as it
  must be without steering).
- ±1 multiplier swings are **modest**: layer 12 `0.465 / 0.695 / 0.704`, layer 13
  `0.538 / 0.695 / 0.620`, layer 15 `0.613 / 0.695 / 0.728`; at ±2 it degenerates
  (layer 13 → 0.453).
- **My transformer_lens reimplementation matches this**: baseline 0.692, most
  sensitive at layer 12, modest swing, degenerates at large coef.

*Correction to an earlier draft of this report:* I had quoted "≈0.20 / 0.50 / 0.80
at −1/0/+1" as the paper's A/B result. That was a **mis-summary of the figure** —
it contradicts the authors' own released A/B data (baseline 0.695, modest swings),
which my reimplementation reproduces. The CAA paper's *large* sycophancy effects
are on **open-ended generations scored by an LLM judge** (the paper states the
effect is "substantially larger for open-ended" than A/B) — a separate metric not
run here (needs a GPT-4/Claude judge). On the A/B metric the true effect is modest,
and the recreation is faithful.

**Bearing on the main result:** the attention-weighting question is a *controlled
internal* comparison (same model/layer/data/metric/magnitude; only pooling
differs), so it is unaffected by how closely the absolute magnitudes match the
papers. → `results/{refusal,caa}/*`.

## 2. The attention-weighting experiment

Design axes (the two the task flags — *harvest location* and *apply location* —
plus importance/BOS knobs):

| axis | values |
|---|---|
| harvest region `S` | last · prompt · response · full |
| pooling | uniform · attention |
| attention importance | same-layer · all-layer-mean · max-head |
| BOS (attention sink) | include · exclude |
| apply location | all · last (CAA); all-layers (refusal ablation) |

The BOS sink is real: the argmax attention-received position is token 0 (BOS) for
every example, so `exclude_bos` is a first-class knob.

### CAA (unit-normalised, L12)

| spec | ‖v‖ raw | dose-response swing | verdict |
|---|---|---|---|
| **last/uniform** (canonical) | 1.86 | 0.49 → 0.745 | **strong** |
| response/uniform | 0.93 | 0.49 → 0.743 | strong (= baseline) |
| response/attention | 0.91 | 0.485 → 0.740 | ≈ tie (no gain) |
| prompt/uniform | **0.00** | — | degenerate |
| full/uniform | 0.01 | (dir only via norm) | ~0 raw |
| full/attention | 0.43 | 0.64 → 0.70 | weak |
| full/attention/noBOS | 0.08 | 0.60 → 0.71 | weak |

The CAA contrast is **localised to the answer token** (positive/negative share
the prompt). Uniform pooling over `prompt`/`full` therefore *cancels* (‖v‖≈0).
Attention weighting reweights toward high-attention shared/BOS tokens that carry
no contrast, so it **dilutes** — `response/attention` only ties `response/uniform`.

### Refusal (unit-normalised, L10)

| spec | ablate harmful (bypass) | add harmless (induce) +8 / +16 |
|---|---|---|
| **last/uniform** (canonical) | 0.97 → 0.00 | **0.97 / 1.00** |
| full/uniform | 0.97 → 0.00 | 0.03 / 0.03 |
| full/attention (same-layer, noBOS) | 0.97 → 0.00 | 0.00 / 0.00 |
| full/attention (all-layer-mean, noBOS) | 0.97 → 0.00 | 0.00 / 0.00 |
| full/attention (max-head, noBOS) | 0.97 → 0.00 | 0.00 / 0.00 |
| *response-harvest* last/uniform | 0.97 → 0.00 | 0.03 / 0.06 |
| *response-harvest* response/uniform | 0.97 → 0.00 | 0.00 / 0.09 |
| *response-harvest* response/attention (all variants) | 0.97 → 0.00 | 0.00 / 0.00 |

- **Ablation is saturated** — *every* direction bypasses refusal, because any
  vector sufficiently aligned with the single refusal direction removes it when
  projected out. Non-discriminating.
- **Addition discriminates** and only `last/uniform` works. Multi-position
  pooling (uniform *and* attention, including the strongest importance variants)
  produces a **contaminated** direction that cannot induce refusal.
- **Response-harvesting** (building the contrast from the model's own
  refusal/compliance completions) does not help either, and is worse.

### Apply location

CAA `last/uniform` (L12, unit), apply=**all** positions vs apply=**last** position
only, `P(sycophantic)` by coefficient:

| coef | −4 | −2 | 0 | +2 | +4 |
|---|---|---|---|---|---|
| apply=all | **0.491** | 0.538 | 0.692 | 0.745 | 0.725 |
| apply=last | 0.626 | 0.652 | 0.692 | 0.725 | 0.736 |

Applying at **all** positions gives a wider swing (stronger negative steering)
than applying only at the read position; localising the application weakens the
effect but does not change its sign or the overall conclusion. (Refusal ablation
is by construction applied at every position and layer.)

## 3. Why attention weighting doesn't help

1. **The last token is already an attention-weighted summary.** The forward pass
   routes task-relevant information into the final prompt position via trained,
   per-head, query-specific attention. Harvesting there *is* a learned pooling.
   Our explicit pool uses a single scalar per position (total attention received,
   averaged over heads) — a strictly cruder statistic that cannot out-resolve the
   model's own multi-head routing.
2. **Total-attention-received weights the wrong tokens.** The BOS sink and
   high-frequency/syntactic tokens receive the most attention but carry little
   contrastive signal; excluding BOS helps only marginally.
3. **Multi-position pooling dilutes or cancels.** Localised contrast (CAA) →
   cancellation; distributed contrast (refusal) → the refusal direction gets
   mixed with generic content, enough to still *ablate* but not to cleanly *add*.

## 4. Limitations

- One model (Llama-2-7B-chat); one behaviour per setting.
- `a_t` = mean-over-heads summed query attention; other importance notions
  (attention *from the last token*, value-weighted norms, gradient/attribution)
  are untested and are the natural next thing to try.
- `n_train=128`, held-out 32–50; substring + MC metrics (no LLM judge).
- Layer chosen per setting by a held-out sweep, not exhaustively per spec.

## Conclusion

Attention-weighted pooling does **not** improve difference-of-means contrastive
steering in either canonical setting, under any harvest region, importance
measure, BOS treatment, or apply location we tried. The single-last-token recipe
both papers converged on is hard to beat **because** it already exploits the
model's own attention aggregation; an explicit attention-weighted pool over more
positions adds noise, not signal.

*Reproduce:* `uv run python experiments/refusal.py --specs sweep --layer 10` and
`uv run python experiments/caa.py --specs sweep --unit --layer 12`. Plots and raw
metrics under `results/`.
