"""Batched greedy generation with optional steering hooks.

Left-padding aligns prompts on the right so a single batch decodes together. We
validate batched output against per-prompt output once on the box; if a model's
padding handling misbehaves, drop batch_size to 1.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def batch_generate(
    model,
    prompts: list[str],
    max_new_tokens: int = 48,
    fwd_hooks=None,
    batch_size: int = 16,
) -> list[str]:
    """Greedy-generate continuations for each prompt; returns new text only."""
    fwd_hooks = fwd_hooks or []
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        toks = model.to_tokens(chunk)  # left-padded (tokenizer.padding_side='left')
        with model.hooks(fwd_hooks=fwd_hooks):
            gen = model.generate(
                toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stop_at_eos=True,
                verbose=False,
            )
        new = gen[:, toks.shape[1] :]
        for row in new:
            outs.append(model.tokenizer.decode(row, skip_special_tokens=True))
    return outs
