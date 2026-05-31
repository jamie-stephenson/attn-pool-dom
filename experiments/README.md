# experiments

Runnable entry points (built/tested on the A100):

- `refusal.py` — Setting 1. Build DoM refusal direction; reproduce the
  induce/ablate effect (baseline = uniform, last-token). Then sweep
  harvest region × pooling × apply positions.
- `caa.py` — Setting 2. Build DoM sycophancy vector from A/B pairs; reproduce
  the held-out MC behaviour shift (baseline = uniform, last-token). Then sweep.

Each writes metrics JSON + plots into `../results/<setting>/`.

Reproduction targets (baselines to hit before testing attention weighting):
- Refusal: large drop in refusal rate on harmful prompts under ablation; large
  rise under addition on harmless prompts — with coherence preserved.
- CAA sycophancy: monotonic shift in P(sycophantic answer) with steering coef,
  matching the sign/scale reported by Rimsky et al.
