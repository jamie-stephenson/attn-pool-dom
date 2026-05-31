"""Dataset loaders for the two settings.

Refusal (Arditi et al.):
  - harmful instructions: AdvBench (`walledai/AdvBench` or llm-attacks harmful_behaviors)
  - harmless instructions: Alpaca (`tatsu-lab/alpaca`)
  Contrast = harmful (pos / refusal) vs harmless (neg / no-refusal).

CAA (Rimsky et al.):
  - A/B contrastive multiple-choice pairs per behaviour (default: sycophancy),
    from the CAA repo (github.com/nrimsky/CAA, `datasets/generate` + `datasets/test`).
  Contrast = answer matching behaviour (pos) vs not (neg).

Heavy downloads happen on the A100. `scripts/fetch_data.py` materialises the CAA
JSON; HF datasets stream directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContrastPair:
    prompt: str          # formatted up to (but not including) the harvested span
    pos_text: str        # positive continuation (e.g. "(A)") or "" for prompt-only
    neg_text: str


@dataclass
class Split:
    train: list[ContrastPair]
    val: list[ContrastPair]


# ----------------------------------------------------------------------------
# Refusal
# ----------------------------------------------------------------------------
def load_refusal(n_train: int = 128, n_val: int = 64, seed: int = 0) -> Split:
    """Harmful vs harmless instructions (prompt-only contrast).

    TODO(remote): pull AdvBench harmful + Alpaca harmless via `datasets`,
    shuffle with `seed`, format with the Llama-2 chat template at call sites.
    For the prompt-only contrast, pos/neg differ by the *instruction*, so each
    "pair" is actually two independent prompts (pos=harmful, neg=harmless).
    """
    raise NotImplementedError("load_refusal: implement on remote")


# ----------------------------------------------------------------------------
# CAA
# ----------------------------------------------------------------------------
def load_caa(
    behaviour: str = "sycophancy",
    data_dir: str | Path = "cache/caa",
    n_train: int = 128,
    n_val: int = 64,
) -> Split:
    """Load CAA A/B contrastive pairs for a behaviour.

    TODO(remote): read `{data_dir}/{behaviour}/generate_dataset.json` (train) and
    `test_dataset.json` (val). Each item has a question + matching/non-matching
    answer letters; pos_text/neg_text are the answer-letter spans.
    """
    raise NotImplementedError("load_caa: implement on remote")
