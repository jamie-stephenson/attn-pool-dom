"""attn-pool-dom: attention-weighted pooling for DoM contrastive steering."""

from attn_pool_dom.pooling import (
    attention_weighted_pool,
    pool,
    total_attention_received,
    uniform_pool,
)

__version__ = "0.1.0"

__all__ = [
    "attention_weighted_pool",
    "pool",
    "total_attention_received",
    "uniform_pool",
]


def main() -> None:
    print(f"attn-pool-dom {__version__}: attention-weighted DoM steering")
