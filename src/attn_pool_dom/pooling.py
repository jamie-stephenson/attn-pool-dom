"""Pooling strategies for difference-of-means steering vectors.

The contribution of this project lives here: the comparison between **uniform**
mean-pooling (the standard DoM recipe) and **attention-weighted** pooling, where
each token position is weighted by the total attention it receives.

All functions are batched and operate on the trailing dimensions, so they accept
either a single example (`[seq, d]`) or a batch (`[batch, seq, d]`). Masks select
the *harvest* positions `S`; weighting/normalisation always happens within `S`.

Notation
--------
acts          residual-stream activations, shape `[..., seq, d_model]`
mask          boolean harvest mask,        shape `[..., seq]` (True = include)
pattern       attention weights,           shape `[..., n_heads, q_pos, k_pos]`
attn_received total attention per key pos, shape `[..., seq]`
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "total_attention_received",
    "uniform_pool",
    "attention_weighted_pool",
    "pool",
]


def total_attention_received(pattern: Tensor, head_agg: str = "mean") -> Tensor:
    """Total attention *received* by each key position.

    For a causal attention pattern ``A`` (rows = queries, columns = keys, each
    query row summing to 1 over keys), the total attention flowing **into** key
    position ``t`` is ``a_t = Σ_q agg_heads(A[:, q, t])``.

    BOS / early tokens are well known attention sinks and will naturally receive
    large ``a_t``; whether to keep or drop them is handled at pooling time via
    ``exclude_bos`` (see :func:`attention_weighted_pool`), not here.

    Parameters
    ----------
    pattern : Tensor ``[..., n_heads, q_pos, k_pos]``
        Attention weights, e.g. ``transformer_lens`` cache ``("pattern", layer)``.
    head_agg : {"mean", "max"}
        How to aggregate across heads before summing over queries.

    Returns
    -------
    Tensor ``[..., k_pos]`` of non-negative totals (not normalised).
    """
    if head_agg == "mean":
        per_qk = pattern.mean(dim=-3)  # [..., q, k]
    elif head_agg == "max":
        per_qk = pattern.amax(dim=-3)
    else:
        raise ValueError(f"head_agg must be 'mean' or 'max', got {head_agg!r}")
    return per_qk.sum(dim=-2)  # sum over queries -> [..., k]


def uniform_pool(acts: Tensor, mask: Tensor) -> Tensor:
    """Uniform mean-pool of ``acts`` over the masked positions.

    This is the standard DoM pooling recipe. Returns the mean over positions
    where ``mask`` is True; if no positions are selected returns zeros.
    """
    w = mask.to(acts.dtype)  # [..., seq]
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [..., 1]
    return (acts * w.unsqueeze(-1)).sum(dim=-2) / denom


def attention_weighted_pool(
    acts: Tensor,
    attn_received: Tensor,
    mask: Tensor,
    exclude_bos: bool = False,
    eps: float = 1e-9,
) -> Tensor:
    """Attention-weighted pool of ``acts`` over the masked positions.

    Each harvest position ``t`` is weighted by its (renormalised, within-``S``)
    total received attention ``a_t``. If every selected weight is ~0 (degenerate
    pattern, or only BOS selected then excluded), falls back to a uniform pool so
    the result is never NaN.

    Parameters
    ----------
    acts : Tensor ``[..., seq, d_model]``
    attn_received : Tensor ``[..., seq]``
        Non-negative per-position weights, e.g. from
        :func:`total_attention_received`.
    mask : Tensor ``[..., seq]`` bool
        Harvest positions.
    exclude_bos : bool
        Drop position 0 (BOS attention sink) from the weighting.
    """
    m = mask.clone()
    if exclude_bos:
        m[..., 0] = False

    w = attn_received.clamp_min(0.0) * m.to(attn_received.dtype)  # [..., seq]
    denom = w.sum(dim=-1, keepdim=True)  # [..., 1]

    # Fallback to uniform where the attention mass over S is ~0.
    degenerate = denom <= eps  # [..., 1]
    if degenerate.any():
        uni = m.to(acts.dtype)
        uni = uni / uni.sum(dim=-1, keepdim=True).clamp_min(1.0)
        w_norm = w / denom.clamp_min(eps)
        w_final = torch.where(degenerate, uni, w_norm)
    else:
        w_final = w / denom

    return (acts * w_final.unsqueeze(-1).to(acts.dtype)).sum(dim=-2)


def pool(
    acts: Tensor,
    mask: Tensor,
    method: str = "uniform",
    attn_received: Tensor | None = None,
    exclude_bos: bool = False,
) -> Tensor:
    """Dispatch to a pooling method.

    ``method="uniform"`` ignores attention. ``method="attention"`` requires
    ``attn_received`` (already reduced to ``[..., seq]`` via
    :func:`total_attention_received`, where head aggregation is chosen).
    """
    if method == "uniform":
        return uniform_pool(acts, mask)
    if method == "attention":
        if attn_received is None:
            raise ValueError("method='attention' requires attn_received")
        return attention_weighted_pool(acts, attn_received, mask, exclude_bos=exclude_bos)
    raise ValueError(f"unknown pooling method {method!r}")
