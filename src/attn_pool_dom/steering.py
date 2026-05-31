"""Difference-of-means steering vector + application hooks.

DoM vector:  v = mean(pos_reps) - mean(neg_reps).
Two intervention modes (both used in the literature):
  - "add"    : h <- h + coef * v_hat                (CAA-style addition)
  - "ablate" : h <- h - (h . v_hat) v_hat           (Arditi-style directional ablation)

Apply positions are configurable; with a KV cache, generated tokens arrive one
position at a time, so "apply to response" = apply on every decode step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def dom_vector(pos_reps: Tensor, neg_reps: Tensor) -> Tensor:
    """Difference of class means. ``pos_reps``/``neg_reps``: ``[n, d]`` -> ``[d]``."""
    return pos_reps.mean(0) - neg_reps.mean(0)


def unit(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / (v.norm() + eps)


@dataclass
class SteerConfig:
    layers: tuple[int, ...]          # residual layers to intervene on
    mode: str = "add"                # "add" | "ablate"
    coef: float = 1.0                # multiplier for "add"
    apply: str = "all"              # "all" | "prompt" | "response" | "last"


def make_steer_hook(direction: Tensor, cfg: SteerConfig, prompt_len: int | None = None):
    """Build a transformer_lens forward hook applying the steering vector.

    ``direction`` should already be the desired vector (unit for ablate; for add,
    the magnitude is folded into ``cfg.coef`` * ``direction``). ``prompt_len`` is
    used only for prompt/response apply masks on the initial (full-seq) pass.
    """
    v = direction

    def hook(resid: Tensor, hook) -> Tensor:  # resid: [batch, seq, d]
        seq = resid.shape[1]
        if cfg.mode == "ablate":
            vh = unit(v).to(resid.dtype)
            proj = (resid @ vh)[..., None] * vh
            update = -proj
        else:  # add
            update = cfg.coef * v.to(resid.dtype)
            update = update.expand_as(resid).clone()

        # position masking
        if cfg.apply != "all" and seq > 1 and prompt_len is not None:
            keep = torch.zeros(seq, dtype=torch.bool, device=resid.device)
            if cfg.apply == "prompt":
                keep[:prompt_len] = True
            elif cfg.apply == "response":
                keep[prompt_len:] = True
            elif cfg.apply == "last":
                keep[-1] = True
            update = update * keep[None, :, None]
        return resid + update

    return hook


def steering_hooks(direction: Tensor, cfg: SteerConfig, prompt_len: int | None = None):
    """List of ``(hook_name, hook_fn)`` for the configured layers."""
    return [
        (f"blocks.{l}.hook_resid_post", make_steer_hook(direction, cfg, prompt_len))
        for l in cfg.layers
    ]


def ablation_hooks(
    direction: Tensor,
    n_layers: int,
    points: tuple[str, ...] = ("resid_pre", "resid_mid", "resid_post"),
):
    """Arditi-style directional ablation: project ``direction`` out of every named
    residual point at every layer, so the model cannot represent it anywhere.
    Stronger than ablating resid_post alone (which lets attn re-add it at mid)."""
    vh = unit(direction)

    def make():
        def hook(resid: Tensor, hook) -> Tensor:  # resid: [batch, seq, d]
            v = vh.to(resid.dtype)
            proj = (resid @ v)[..., None] * v
            return resid - proj

        return hook

    return [
        (f"blocks.{l}.hook_{pt}", make())
        for l in range(n_layers)
        for pt in points
    ]
