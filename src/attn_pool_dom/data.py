"""Dataset loaders for the two settings.

Refusal (Arditi et al.):
  harmful instructions (AdvBench) vs harmless instructions (Alpaca). Prompt-only
  contrast: pos=harmful, neg=harmless. Used to build the refusal DoM direction
  and to evaluate bypass (on held-out harmful) / induction (on held-out harmless).

CAA (Rimsky et al.):
  A/B contrastive multiple-choice pairs per behaviour (default sycophancy), from
  the CAA repo. Build vector from `generate` split, evaluate on `test` split.

`datasets`-backed (refusal) downloads stream on the A100; CAA JSON is fetched by
`scripts/fetch_data.py` into `cache/caa/`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Example:
    """One harvest item: a chat-formatted prompt and an optional completion span.

    `prompt` is the model input up to the assistant turn; `completion` is the
    text whose tokens form the "response" region (empty for prompt-only refusal).
    """

    prompt: str
    completion: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class MCItem:
    """A held-out multiple-choice item for CAA-style behaviour scoring."""

    question: str          # full user question text incl. (A)/(B) options
    matching: str          # behaviour-matching letter, "A" or "B"
    not_matching: str      # the other letter


# ----------------------------------------------------------------------------
# Refusal
# ----------------------------------------------------------------------------
@dataclass
class RefusalData:
    harmful_train: list[str]
    harmless_train: list[str]
    harmful_val: list[str]
    harmless_val: list[str]


def load_refusal(n_train: int = 128, n_val: int = 64, seed: int = 0) -> RefusalData:
    """Harmful (AdvBench) vs harmless (Alpaca) raw instructions.

    Returns raw instruction strings; chat formatting is applied at harvest sites.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    adv = load_dataset("walledai/AdvBench", split="train")
    harmful = [r["prompt"] for r in adv]
    rng.shuffle(harmful)

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless = [r["instruction"] for r in alpaca if not r["input"].strip()]
    rng.shuffle(harmless)

    need = n_train + n_val
    if len(harmful) < need:
        raise ValueError(f"not enough harmful prompts: {len(harmful)} < {need}")
    return RefusalData(
        harmful_train=harmful[:n_train],
        harmless_train=harmless[:n_train],
        harmful_val=harmful[n_train:need],
        harmless_val=harmless[n_train:need],
    )


# ----------------------------------------------------------------------------
# CAA
# ----------------------------------------------------------------------------
@dataclass
class CAAData:
    train: list[MCItem]
    val: list[MCItem]


def _letter(ans: str) -> str:
    """'(A)' / 'A' / ' (A)' -> 'A'."""
    return ans.strip().strip("()").strip()[:1].upper()


def _read_caa_json(path: Path) -> list[MCItem]:
    items = json.loads(Path(path).read_text())
    out: list[MCItem] = []
    for it in items:
        out.append(
            MCItem(
                question=it["question"].strip(),
                matching=_letter(it["answer_matching_behavior"]),
                not_matching=_letter(it["answer_not_matching_behavior"]),
            )
        )
    return out


def load_caa(
    behaviour: str = "sycophancy",
    data_dir: str | Path = "cache/caa",
    n_train: int = 128,
    n_val: int = 64,
) -> CAAData:
    """Load CAA A/B pairs for a behaviour from fetched JSON."""
    d = Path(data_dir) / behaviour
    gen = _read_caa_json(d / "generate_dataset.json")
    test = _read_caa_json(d / "test_dataset_ab.json")
    return CAAData(train=gen[:n_train], val=test[:n_val])
