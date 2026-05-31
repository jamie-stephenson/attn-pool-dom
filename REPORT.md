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

**Refusal (Arditi et al.) — faithful reproduction of the mechanism.**
Their claim: a single direction whose *ablation* stops refusal of harmful prompts
and whose *addition* elicits refusal on harmless prompts (metrics: refusal
substring + Llama-Guard-2 safety; for Llama-2-7B they report e.g. 22.6% attack
success after weight-orthogonalisation *with the default safety system prompt*).
- My no-steer refusal: harmful **0.97**, harmless **0.00**.
- *Ablation* (project the L10 direction out of every residual point, all layers):
  harmful **0.97 → 0.00**.
- *Addition*: harmless **0.00 → 0.41 (+4) → 0.97 (+8) → 1.00 (+16)**.
- Layer sweep: L10 uniquely the refusal direction (post-ablation harmful refusal
  L8 0.88, **L10 0.04**, L12 0.75, L14 0.83).

  *Caveats:* I evaluate with the substring metric and **no system prompt** (the
  easier regime), so my complete bypass is **not** directly comparable to their
  22.6%-ASR (with-system-prompt, Llama-Guard) figure. But ablation is
  *scale-invariant* (projects out a direction), so this reproduction is robust to
  activation-scale differences and cleanly reproduces their headline mechanism.

**CAA sycophancy (Rimsky et al.) — qualitative/directional reproduction only;
magnitude NOT reproduced.** Their Fig. 7 (Llama-2-7B-chat, layer 15): P(sycophantic
answer) ≈ **0.20 / 0.50 / 0.80** at multiplier −1 / 0 / +1 (≈60-pt swing).
- Mine (raw vectors, integer multipliers): at their layer 15, **0.66 / 0.69 / 0.72**
  (−1/0/+1); at my most-sensitive layer 12, 0.61 / 0.69 / 0.74 — a ~10–20-pt swing
  around a **0.69** baseline (vs their ~0.50). Pushing harder (unit-norm, large
  coef) reaches ~0.49–0.75 before the model degenerates — still short of 0.20–0.80.
- *Honest verdict:* I reproduce the **sign and monotonicity** (subtract → less
  sycophantic, add → more) but **materially undershoot** the published effect, and
  my baseline and best-layer (12 vs 15) differ.
- *Most likely cause:* I load the model with `transformer_lens` weight-processing
  (LayerNorm folding/centering), which rescales the residual stream; my raw DoM
  vectors have norm ≈2, so `multiplier × vector` is a small perturbation. CAA
  builds vectors from **raw HF activations** (larger scale), so equal multipliers
  move the model much more. Prompt-format/baseline differences compound this.
  Switching to `from_pretrained_no_processing` (raw-HF scale) is the untested fix.

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
