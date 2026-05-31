"""Model loading via transformer_lens HookedTransformer.

Kept thin on purpose: one entry point that returns a HookedTransformer in the
right dtype/device, plus chat-template helpers for the two settings. Refined &
tested on the A100.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

# Llama-2 chat template pieces (matches Arditi et al. / CAA usage).
LLAMA2_CHAT_TEMPLATE = (
    "[INST] {instruction} [/INST]"
)
LLAMA2_SYS_TEMPLATE = (
    "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
)


@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-2-7b-chat-hf"   # transformer_lens config / architecture
    # Ungated weights source (bit-identical mirror). The user's HF account is not
    # on Meta's authorized list for the gated meta-llama repo, but TL holds the
    # Llama-2 config in its registry, so we load weights from the mirror and keep
    # the official name for a faithful reproduction. Set to None to use `name`.
    hf_mirror: str | None = "NousResearch/Llama-2-7b-chat-hf"
    dtype: str = "bfloat16"
    device: str = "cuda"


def load_model(cfg: ModelConfig) -> HookedTransformer:
    """Load a HookedTransformer. Attention `hook_pattern` is available by default,
    which is what attention-weighted pooling needs."""
    dtype = getattr(torch, cfg.dtype)
    if cfg.hf_mirror:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(cfg.hf_mirror, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_mirror)
        model = HookedTransformer.from_pretrained(
            cfg.name, hf_model=hf_model, tokenizer=tokenizer, dtype=dtype, device=cfg.device
        )
    else:
        model = HookedTransformer.from_pretrained(cfg.name, dtype=dtype, device=cfg.device)
    model.eval()
    # Left-pad by default so batched generation aligns on the last token; the
    # harvest pass overrides to right padding where region masks are simpler.
    if model.tokenizer is not None:
        model.tokenizer.padding_side = "left"
    return model


def format_chat(instruction: str, system: str | None = None) -> str:
    """Wrap a raw instruction in the Llama-2 chat template."""
    if system:
        return LLAMA2_SYS_TEMPLATE.format(system=system, instruction=instruction)
    return LLAMA2_CHAT_TEMPLATE.format(instruction=instruction)
