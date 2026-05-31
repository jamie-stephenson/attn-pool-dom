"""Efficient multi-config harvesting: one forward pass per batch, pooled many ways.

A `PoolSpec` names a (region, pooling-method, attn-layer, exclude_bos, head_agg)
recipe. `build_dom_vectors` runs the model once per batch over the positive and
negative example sets, caches the resid at the target layer plus any needed
attention patterns, and produces a DoM vector for every spec — so a whole sweep
costs the same forward passes as a single recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from attn_pool_dom.data import Example
from attn_pool_dom.pooling import pool, total_attention_received


@dataclass(frozen=True)
class PoolSpec:
    region: str = "last"           # last | prompt | response | full
    method: str = "uniform"        # uniform | attention
    attn_layer: object = "same"    # "same" | "mean" | int
    exclude_bos: bool = False
    head_agg: str = "mean"

    @property
    def name(self) -> str:
        if self.method == "uniform":
            return f"{self.region}/uniform"
        bos = "/noBOS" if self.exclude_bos else ""
        al = self.attn_layer if self.attn_layer in ("same", "mean") else f"L{self.attn_layer}"
        return f"{self.region}/attn[{al},{self.head_agg}]{bos}"


def tokenize_batch(model, prompts: list[str], completions: list[str]):
    """Right-pad a batch of prompt(+completion) strings.

    Returns (toks [b, seq], prompt_lens [b], real_lens [b]). BOS is prepended;
    prompt_len is the token count of the prompt alone (the response region is
    everything after it).
    """
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    full_ids, prompt_lens, real_lens = [], [], []
    for p, c in zip(prompts, completions):
        full = model.to_tokens(p + c)[0]          # [t] incl BOS
        plen = model.to_tokens(p).shape[1]        # BOS + prompt tokens
        full_ids.append(full)
        prompt_lens.append(plen)
        real_lens.append(full.shape[0])
    seq = max(real_lens)
    toks = torch.full((len(full_ids), seq), pad_id, dtype=torch.long, device=model.cfg.device)
    for i, ids in enumerate(full_ids):
        toks[i, : ids.shape[0]] = ids
    return (
        toks,
        torch.tensor(prompt_lens, device=model.cfg.device),
        torch.tensor(real_lens, device=model.cfg.device),
    )


def _region_mask(seq, prompt_lens, real_lens, region, device):
    pos = torch.arange(seq, device=device)[None].expand(prompt_lens.shape[0], -1)
    pl, rl = prompt_lens[:, None], real_lens[:, None]
    is_real = pos < rl
    if region == "full":
        return is_real
    if region == "prompt":
        return is_real & (pos < pl)
    if region == "response":
        return is_real & (pos >= pl)
    if region == "last":
        return pos == (rl - 1)
    raise ValueError(region)


@torch.no_grad()
def _batch_pooled(model, toks, prompt_lens, real_lens, layer, attn_layers, specs):
    """Pooled reps for every spec on one batch -> {spec.name: [b, d]}."""
    resid_name = f"blocks.{layer}.hook_resid_post"
    want = {resid_name}
    for al in attn_layers:
        want.add(f"blocks.{al}.attn.hook_pattern")
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in want)
    resid = cache[resid_name]
    seq = resid.shape[1]

    attn_received: dict[object, Tensor] = {}
    for al in attn_layers:
        attn_received[al] = total_attention_received(cache[f"blocks.{al}.attn.hook_pattern"])
    if any(s.attn_layer == "mean" for s in specs):
        attn_received["mean"] = torch.stack(
            [total_attention_received(cache[f"blocks.{l}.attn.hook_pattern"]) for l in attn_layers]
        ).mean(0)

    out = {}
    mask_cache: dict[str, Tensor] = {}
    for s in specs:
        if s.region not in mask_cache:
            mask_cache[s.region] = _region_mask(seq, prompt_lens, real_lens, s.region, resid.device)
        mask = mask_cache[s.region]
        ar = None
        if s.method == "attention":
            key = layer if s.attn_layer == "same" else s.attn_layer
            ar = attn_received[key]
        out[s.name] = pool(resid, mask, method=s.method, attn_received=ar, exclude_bos=s.exclude_bos)
    return out


def _accumulate_means(model, examples: list[Example], layer, attn_layers, specs, batch_size):
    sums: dict[str, Tensor] = {}
    n = 0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        toks, pl, rl = tokenize_batch(
            model, [e.prompt for e in chunk], [e.completion for e in chunk]
        )
        pooled = _batch_pooled(model, toks, pl, rl, layer, attn_layers, specs)
        for k, v in pooled.items():
            v = v.float().sum(0)
            sums[k] = v if k not in sums else sums[k] + v
        n += len(chunk)
    return {k: v / n for k, v in sums.items()}


def _needed_attn_layers(layer, specs, n_layers) -> set:
    al = set()
    for s in specs:
        if s.method != "attention":
            continue
        if s.attn_layer == "same":
            al.add(layer)
        elif s.attn_layer == "mean":
            al.update(range(n_layers))
        else:
            al.add(int(s.attn_layer))
    return al


def build_dom_vectors(
    model,
    pos: list[Example],
    neg: list[Example],
    layer: int,
    specs: list[PoolSpec],
    batch_size: int = 16,
):
    """DoM vector per spec: mean(pos) - mean(neg). Returns (dirs, pos_means, neg_means)."""
    attn_layers = _needed_attn_layers(layer, specs, model.cfg.n_layers)
    pos_means = _accumulate_means(model, pos, layer, attn_layers, specs, batch_size)
    neg_means = _accumulate_means(model, neg, layer, attn_layers, specs, batch_size)
    dirs = {s.name: pos_means[s.name] - neg_means[s.name] for s in specs}
    return dirs, pos_means, neg_means
