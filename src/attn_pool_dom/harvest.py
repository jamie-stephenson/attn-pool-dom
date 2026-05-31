"""Run the model and pool residual-stream activations into per-example reps.

Harvest pass uses **right padding** (no generation here) so real tokens occupy
``[0, seq_len)`` and region masks are simple. Generation/steering passes use
left padding (see steering.py). Refined & tested on the A100.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from attn_pool_dom.pooling import pool, total_attention_received

Region = str  # one of: "prompt", "response", "full", "last"


@dataclass
class HarvestConfig:
    layer: int                       # residual-stream layer to harvest (resid_post)
    region: Region = "last"          # which positions form the harvest set S
    method: str = "uniform"          # "uniform" | "attention"
    attn_layer: int | str = "same"   # layer for attention pattern, or "same"/"mean"
    head_agg: str = "mean"           # "mean" | "max"
    exclude_bos: bool = False


def build_region_mask(
    seq_len: int,
    prompt_lens: Tensor,   # [batch] number of prompt tokens (real, right-padded)
    real_lens: Tensor,     # [batch] number of non-pad tokens
    region: Region,
    device=None,
) -> Tensor:
    """Boolean harvest mask ``[batch, seq_len]`` for a right-padded batch."""
    batch = prompt_lens.shape[0]
    pos = torch.arange(seq_len, device=device)[None].expand(batch, -1)  # [b, seq]
    pl = prompt_lens[:, None]
    rl = real_lens[:, None]
    is_real = pos < rl
    if region == "full":
        mask = is_real
    elif region == "prompt":
        mask = is_real & (pos < pl)
    elif region == "response":
        mask = is_real & (pos >= pl)
    elif region == "last":
        mask = pos == (rl - 1)
    else:
        raise ValueError(f"unknown region {region!r}")
    return mask


@torch.no_grad()
def harvest_reps(
    model,
    toks: Tensor,            # [batch, seq] right-padded token ids
    prompt_lens: Tensor,     # [batch]
    real_lens: Tensor,       # [batch]
    cfg: HarvestConfig,
) -> Tensor:
    """Return pooled reps ``[batch, d_model]`` for one harvest config."""
    resid_name = f"blocks.{cfg.layer}.hook_resid_post"
    attn_layer = cfg.layer if cfg.attn_layer == "same" else cfg.attn_layer

    def names_filter(name: str) -> bool:
        if name == resid_name:
            return True
        if cfg.method == "attention" and name.endswith("hook_pattern"):
            return True
        return False

    _, cache = model.run_with_cache(toks, names_filter=names_filter)
    acts = cache[resid_name]  # [batch, seq, d]

    mask = build_region_mask(
        acts.shape[1], prompt_lens, real_lens, cfg.region, device=acts.device
    )

    attn_received = None
    if cfg.method == "attention":
        if attn_layer == "mean":
            pats = torch.stack(
                [cache[f"blocks.{l}.attn.hook_pattern"] for l in range(model.cfg.n_layers)]
            ).mean(0)
        else:
            pats = cache[f"blocks.{attn_layer}.attn.hook_pattern"]  # [b, head, q, k]
        attn_received = total_attention_received(pats, head_agg=cfg.head_agg)

    return pool(
        acts, mask, method=cfg.method, attn_received=attn_received, exclude_bos=cfg.exclude_bos
    )
