# Style Vectors (official `DLR-SC/style-vectors-for-steering-llms`) — replication + attention weighting

Model: **Alpaca-7B** (`chavinlo/alpaca-native`, the paper's model). Dataset: **GoEmotions**
(6 Ekman emotions). Steering at layers **18–20** via their `SteeringLayer` (`a·mlp(x) + b·vec`).
Eval: their **`j-hartmann/emotion-english-distilroberta-base`** classifier — no LLM judge needed.
(I used a controlled subset of the eval — 6 prompts × 6 emotions × 5 λ — and their Alpaca
reconstruction, so absolute numbers are from a reduced run, but the pooling comparison is
internally fair.)

## Premise correction

The Style Vectors method is actually a **last-token** method, not sentence-mean: their
`get_hidden_activations.py` stores `layer[0][-1]` (the last token) per sentence; the "mean" is
over *sentences* of a class (the difference-of-means), not over tokens within a sentence. So the
interesting test is: **can sentence-mean or attention-weighted-mean *beat* their last-token for a
diffuse property like emotion/style?**

## Result (their `calculate_means` + model + classifier; only the within-sentence pooling changes)

Mean target-emotion classifier score vs steering strength λ (contrastive-OVR style vector):

| pooling | λ=0 | λ=0.5 | λ=1.0 | λ=1.5 | λ=2.0 |
|---|---|---|---|---|---|
| **last** (their recipe) | 0.044 | 0.067 | 0.133 | 0.148 | **0.162** |
| `mean` (sentence-mean) | 0.044 | 0.063 | 0.074 | 0.095 | 0.133 |
| `attn` (attention-weighted) | 0.044 | 0.084 | 0.109 | 0.109 | 0.108 |

**Findings:**
1. **Replication:** their last-token style vector steers emotion as reported — target-emotion
   score rises 0.044 (λ=0) → 0.162 (λ=2).
2. **The last token is the best harvest position — even for a "diffuse" property like emotion.**
   Sentence-`mean` is strictly worse (λ=1: 0.074 vs 0.133), and `attn` is worse still and
   **saturates early** (plateaus at ~0.11 by λ=1, then flat). So span-pooling *dilutes* the style
   signal rather than helping.
3. **Attention weighting gives no improvement** (it edges out uniform mean at small λ but plateaus
   and ends lowest). Consistent with every other setting.

## Across all settings tested

| setting | best harvest | attention weighting vs uniform |
|---|---|---|
| CAA / refusal | last token | no gain |
| Persona vectors (evil/sycophantic/hallucinating) | response-mean (region trait-dependent) | no gain (cos≈0.99) |
| **Style vectors** (GoEmotions) | **last token** | **no gain** (worse than last; saturates) |
| `fra_proj` sleepers (layer-0 prompt) | prompt-mean | **helps** (trigger in specific high-attention tokens) |

**Overall conclusion:** attention-weighted pooling does **not** improve difference-of-means
steering in any published setting tested here (CAA, refusal, persona ×3, style). It only helped in
the `fra_proj` sleeper case, where the discriminative signal concentrates in a *specific subset* of
high-attention tokens. Otherwise the model's own last-token aggregation (or a clean uniform pool
where the signal is genuinely distributed, e.g. persona responses) is as good or better, and
explicit attention weighting adds noise, not signal. The lever that matters is *which region you
harvest*, and that is task/trait-specific.
