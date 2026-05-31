# attn-pool-dom

**Does attention-weighted pooling improve difference-of-means (DoM) steering vectors
in settings that pool over the response / whole sequence?**

A DoM steering vector is the difference between the per-class means of residual-stream
activations. When the activation for an example is pooled over **multiple token
positions** (the response, the whole sentence), the standard choice is a **uniform**
mean. This project asks whether weighting each position by **how much attention it
receives** yields a better steering vector.

We deliberately target two published methods that pool over a *span* rather than the
last token — so attention weighting has a real chance to matter:

1. **Persona Vectors** (Chen et al., 2025, arXiv:2507.21509) — a DoM vector whose
   per-example activation is the **mean over the model's response tokens**; used to
   steer character traits (sycophancy, "evil", hallucination, …).
2. **Style Vectors** (Konen et al., EACL Findings 2024, arXiv:2402.01618) — a DoM
   vector aggregating hidden states **over the sentence**; used to steer style.

## Plan

1. **Replicate** each method's reported results exactly, by running their official
   code (heavy deps installed on the remote A100 only).
2. **Report** the replication.
3. **Apply attention weighting**: swap the uniform span-mean for an
   attention-weighted span-mean (weight position `t` by total attention received),
   and compare on each paper's own metric.

This re-targets an earlier negative result: in last-token settings (CAA, refusal)
attention weighting did not help, because the last token already aggregates the
sequence. Span-pooled settings are the regime where it might.

## Repo layout

```
src/attn_pool_dom/pooling.py   # core: uniform vs attention-weighted pooling (reusable)
tests/test_pooling.py          # unit tests for the pooling math
scripts/remote_setup.sh        # A100 provisioning
results/                       # metrics + plots (synced back via git)
```

Replications run inside the upstream repos on the A100 (cloned under `~/external/`);
the attention-weighting patches live here and are applied into those pipelines.

## Setup

Local (Mac) is for editing + git only — heavy deps are remote-only. See
`scripts/remote_setup.sh` and the A100 workflow.
