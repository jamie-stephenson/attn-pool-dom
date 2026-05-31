"""Setting 1 — Refusal steering (Arditi et al.).

DoM direction from harmful vs harmless instructions (harvested at the last prompt
token, the canonical recipe). Two interventions, matching the paper:
  - ablation  : remove the direction from resid_post at every layer -> bypasses
                refusal on held-out HARMFUL prompts (refusal rate should drop).
  - addition  : add coef * dir at the source layer -> induces refusal on held-out
                HARMLESS prompts (refusal rate should rise).

We then sweep harvest region × pooling (uniform vs attention) to test whether
attention weighting yields a better direction.

Run (baseline): uv run python experiments/refusal.py --specs baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attn_pool_dom.data import Example, load_refusal
from attn_pool_dom.eval import refusal_rate
from attn_pool_dom.generate import batch_generate
from attn_pool_dom.model import ModelConfig, format_chat, load_model
from attn_pool_dom.pipeline import PoolSpec, build_dom_vectors
from attn_pool_dom.steering import SteerConfig, steering_hooks

DEFAULT_LAYER = 14


def to_examples(instructions: list[str]) -> list[Example]:
    return [Example(prompt=format_chat(x), completion="") for x in instructions]


def spec_set(name: str) -> list[PoolSpec]:
    if name == "baseline":
        return [PoolSpec("last", "uniform")]
    # prompt-only contrast => response region is empty; "full" == all prompt tokens.
    return [
        PoolSpec("last", "uniform"),
        PoolSpec("full", "uniform"),
        PoolSpec("full", "attention", "same", exclude_bos=False),
        PoolSpec("full", "attention", "same", exclude_bos=True),
    ]


def unit(v):
    return v / v.norm()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-eval", type=int, default=32, help="held-out prompts per side")
    ap.add_argument("--add-coefs", type=float, nargs="+", default=[4, 8, 16])
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--specs", default="baseline", choices=["baseline", "sweep"])
    ap.add_argument("--layer-sweep", default="", help="e.g. '8,10,12,14,16' to pick best ablation layer")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/refusal")
    args = ap.parse_args()

    model = load_model(ModelConfig())
    n_layers = model.cfg.n_layers

    data = load_refusal(n_train=args.n_train, n_val=args.n_eval)
    pos = to_examples(data.harmful_train)   # refusal-eliciting
    neg = to_examples(data.harmless_train)
    harmful_eval = [format_chat(x) for x in data.harmful_val]
    harmless_eval = [format_chat(x) for x in data.harmless_val]
    print(f"train {len(pos)}/{len(neg)} | eval {len(harmful_eval)} harmful / "
          f"{len(harmless_eval)} harmless | layer {args.layer}")

    if args.layer_sweep:
        layers = [int(x) for x in args.layer_sweep.replace(",", " ").split()]
        r0 = refusal_rate(batch_generate(model, harmful_eval, args.max_new, batch_size=args.batch_size))
        print(f"=== refusal ablation layer sweep (no-steer harmful={r0:.2f}) ===")
        for L in layers:
            raw = build_dom_vectors(model, pos, neg, L, [PoolSpec("last", "uniform")], args.batch_size)[0]["last/uniform"]
            abl = steering_hooks(raw / raw.norm(), SteerConfig(layers=tuple(range(n_layers)), mode="ablate"))
            r = refusal_rate(batch_generate(model, harmful_eval, args.max_new, abl, args.batch_size))
            print(f"L{L:2d} |v|={raw.norm().item():5.2f} | harmful refusal {r0:.2f}->{r:.2f}")
        return

    specs = spec_set(args.specs)
    dirs, _, _ = build_dom_vectors(model, pos, neg, args.layer, specs, args.batch_size)

    # --- baseline behaviour, no intervention ---
    base_harmful = batch_generate(model, harmful_eval, args.max_new, batch_size=args.batch_size)
    base_harmless = batch_generate(model, harmless_eval, args.max_new, batch_size=args.batch_size)
    r_harmful0 = refusal_rate(base_harmful)
    r_harmless0 = refusal_rate(base_harmless)
    print(f"[no steer] refusal: harmful={r_harmful0:.2f}  harmless={r_harmless0:.2f}")

    results = {"baseline": {"harmful_refusal": r_harmful0, "harmless_refusal": r_harmless0}}

    for s in specs:
        d = unit(dirs[s.name])
        # ablation across all layers -> bypass refusal on harmful
        abl_hooks = steering_hooks(d, SteerConfig(layers=tuple(range(n_layers)), mode="ablate"))
        abl_gen = batch_generate(model, harmful_eval, args.max_new, abl_hooks, args.batch_size)
        r_abl = refusal_rate(abl_gen)
        # addition at source layer -> induce refusal on harmless
        add_curve = {}
        for c in args.add_coefs:
            add_hooks = steering_hooks(d, SteerConfig(layers=(args.layer,), mode="add", coef=c))
            add_gen = batch_generate(model, harmless_eval, args.max_new, add_hooks, args.batch_size)
            add_curve[c] = refusal_rate(add_gen)
        results[s.name] = {
            "norm": dirs[s.name].norm().item(),
            "ablate_harmful_refusal": r_abl,
            "add_harmless_refusal": add_curve,
            "sample_ablate": abl_gen[0][:160],
        }
        add_str = "  ".join(f"+{c:g}:{r:.2f}" for c, r in add_curve.items())
        print(f"{s.name:26s} | ablate harmful {r_harmful0:.2f}->{r_abl:.2f} | "
              f"add harmless {r_harmless0:.2f}-> {add_str}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"refusal_{args.specs}_L{args.layer}.json"
    path.write_text(json.dumps(results, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
