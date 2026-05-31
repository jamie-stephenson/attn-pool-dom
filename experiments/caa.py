"""Setting 2 — CAA behavioural steering (Rimsky et al.), default: sycophancy.

Build a DoM vector from A/B contrastive pairs, then steer and measure the shift
in P(behaviour-matching answer) on the held-out A/B test set. Baseline recipe =
uniform pooling at the answer token, layer 13. We then sweep harvest region ×
pooling (uniform vs attention) to test whether attention weighting helps.

Run (baseline):  uv run python experiments/caa.py --specs baseline
Run (sweep):     uv run python experiments/caa.py --specs sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attn_pool_dom.data import Example, load_caa
from attn_pool_dom.model import ModelConfig, format_chat, load_model
from attn_pool_dom.pipeline import PoolSpec, build_dom_vectors
from attn_pool_dom.steering import SteerConfig, steering_hooks

DEFAULT_LAYER = 12  # most sensitive by held-out layer sweep (adjacent to CAA's L13)


def make_examples(items) -> tuple[list[Example], list[Example]]:
    pos, neg = [], []
    for it in items:
        prompt = format_chat(it.question)
        # Drop the closing paren so the answer-LETTER is the final token: that is
        # where pos/neg differ directly (CAA's harvest position), giving a strong
        # contrastive signal. With ")" included, `last` would be the shared ")".
        pos.append(Example(prompt=prompt, completion=f" ({it.matching}"))
        neg.append(Example(prompt=prompt, completion=f" ({it.not_matching}"))
    return pos, neg


def letter_ids(model) -> dict[str, int]:
    ids = {}
    for L in ["A", "B", "C", "D"]:
        ids[L] = model.to_tokens(f"({L}", prepend_bos=False)[0, -1].item()
    return ids


@torch.no_grad()
def eval_mc(model, items, lids, direction, layer, coef, batch_size=16) -> float:
    """Mean P(matching) / (P(matching)+P(not_matching)) over held-out items."""
    hooks = (
        steering_hooks(direction, SteerConfig(layers=(layer,), mode="add", coef=coef))
        if coef != 0
        else []
    )
    prompts = [format_chat(it.question) + " (" for it in items]
    vals = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        items_c = items[i : i + batch_size]
        toks = model.to_tokens(chunk)  # left-padded (default)
        attn_mask = (toks != model.tokenizer.pad_token_id).long()
        logits = model.run_with_hooks(toks, attention_mask=attn_mask, fwd_hooks=hooks)
        last = logits[:, -1, :].float()
        logp = torch.log_softmax(last, dim=-1)
        for j, it in enumerate(items_c):
            pm = logp[j, lids[it.matching]].exp().item()
            pn = logp[j, lids[it.not_matching]].exp().item()
            if pm + pn > 0:
                vals.append(pm / (pm + pn))
    return sum(vals) / len(vals) if vals else 0.0


def spec_set(name: str) -> list[PoolSpec]:
    if name == "baseline":
        return [PoolSpec("last", "uniform")]
    specs = [PoolSpec("last", "uniform")]
    for region in ("prompt", "response", "full"):
        specs.append(PoolSpec(region, "uniform"))
        specs.append(PoolSpec(region, "attention", "same", exclude_bos=False))
        specs.append(PoolSpec(region, "attention", "same", exclude_bos=True))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--coefs", type=float, nargs="+", default=[-3, -2, -1, 0, 1, 2, 3])
    ap.add_argument("--specs", default="baseline", choices=["baseline", "sweep"])
    ap.add_argument("--layer-sweep", default="", help="e.g. '8,10,12,13,14,16' to pick best layer")
    ap.add_argument("--unit", action="store_true", help="unit-normalise directions (fair compare)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="results/caa")
    args = ap.parse_args()

    model = load_model(ModelConfig())
    lids = letter_ids(model)
    print("letter ids:", lids)

    data = load_caa(n_train=args.n_train, n_val=args.n_val)
    pos, neg = make_examples(data.train)
    print(f"train pairs {len(pos)} | val items {len(data.val)} | layer {args.layer}")

    if args.layer_sweep:
        layers = [int(x) for x in args.layer_sweep.replace(",", " ").split()]
        sc = [-2, -1, 0, 1, 2]
        print("=== layer sweep (last/uniform) ===")
        for L in layers:
            d = build_dom_vectors(model, pos, neg, L, [PoolSpec("last", "uniform")], args.batch_size)[0]["last/uniform"]
            curve = {c: eval_mc(model, data.val, lids, d, L, c, args.batch_size) for c in sc}
            swing = curve[2] - curve[-2]
            print(f"L{L:2d} |v|={d.norm().item():5.2f} swing={swing:+.3f} | "
                  + " ".join(f"{c:+g}:{curve[c]:.3f}" for c in sc))
        return

    specs = spec_set(args.specs)
    dirs, _, _ = build_dom_vectors(model, pos, neg, args.layer, specs, args.batch_size)

    results = {}
    for s in specs:
        raw = dirs[s.name]
        norm = raw.norm().item()
        d = raw / raw.norm() if args.unit else raw  # unit => fair cross-spec comparison
        curve = {c: eval_mc(model, data.val, lids, d, args.layer, c, args.batch_size)
                 for c in args.coefs}
        results[s.name] = {"norm": norm, "unit": args.unit, "curve": curve}
        pretty = "  ".join(f"{c:+g}:{p:.3f}" for c, p in curve.items())
        print(f"{s.name:28s} |v|={norm:6.2f} | {pretty}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "unit" if args.unit else "raw"
    path = out / f"caa_{args.specs}_L{args.layer}_{tag}.json"
    path.write_text(json.dumps(
        {"layer": args.layer, "coefs": args.coefs, "unit": args.unit, "n_train": len(pos),
         "n_val": len(data.val), "results": results}, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
