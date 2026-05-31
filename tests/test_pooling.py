"""Unit tests for pooling math. Run on the remote: `uv run pytest`."""

import torch

from attn_pool_dom.pooling import (
    attention_weighted_pool,
    total_attention_received,
    uniform_pool,
)


def test_uniform_pool_is_masked_mean():
    acts = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])  # [seq=3, d=2]
    mask = torch.tensor([True, True, False])
    out = uniform_pool(acts, mask)
    assert torch.allclose(out, torch.tensor([2.0, 2.0]))


def test_uniform_pool_empty_mask_is_zero():
    acts = torch.randn(4, 5)
    mask = torch.zeros(4, dtype=torch.bool)
    assert torch.allclose(uniform_pool(acts, mask), torch.zeros(5))


def test_total_attention_received_causal():
    # one head, causal lower-triangular pattern, rows sum to 1.
    # queries (rows) attend to keys (cols).
    A = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.3, 0.5],
        ]
    )  # [q=3, k=3]
    pattern = A[None]  # [n_heads=1, q, k]
    a = total_attention_received(pattern)  # [k=3]
    # column sums: k0=1.7, k1=0.8, k2=0.5
    assert torch.allclose(a, torch.tensor([1.7, 0.8, 0.5]))


def test_attention_weighted_equals_uniform_when_flat():
    acts = torch.randn(6, 8)
    mask = torch.tensor([True, True, True, False, True, False])
    flat = torch.ones(6)  # equal attention everywhere
    aw = attention_weighted_pool(acts, flat, mask)
    uni = uniform_pool(acts, mask)
    assert torch.allclose(aw, uni, atol=1e-6)


def test_attention_weighted_respects_weights():
    acts = torch.tensor([[1.0], [0.0], [0.0]])  # [seq=3, d=1]
    mask = torch.tensor([True, True, True])
    attn = torch.tensor([3.0, 1.0, 0.0])  # pos0 gets 3/4 of the weight
    out = attention_weighted_pool(acts, attn, mask)
    assert torch.allclose(out, torch.tensor([0.75]))


def test_exclude_bos_drops_position_zero():
    acts = torch.tensor([[10.0], [2.0], [4.0]])
    mask = torch.tensor([True, True, True])
    attn = torch.tensor([100.0, 1.0, 1.0])  # BOS would dominate
    out = attention_weighted_pool(acts, attn, mask, exclude_bos=True)
    # with BOS excluded, pos1 & pos2 each weight 0.5 -> (2+4)/2 = 3
    assert torch.allclose(out, torch.tensor([3.0]))


def test_degenerate_falls_back_to_uniform():
    acts = torch.tensor([[2.0], [4.0]])
    mask = torch.tensor([True, True])
    attn = torch.zeros(2)  # all-zero attention -> fallback to uniform
    out = attention_weighted_pool(acts, attn, mask)
    assert torch.allclose(out, torch.tensor([3.0]))


def test_batched():
    acts = torch.randn(4, 7, 16)  # [batch, seq, d]
    mask = torch.ones(4, 7, dtype=torch.bool)
    attn = torch.rand(4, 7)
    out = attention_weighted_pool(acts, attn, mask)
    assert out.shape == (4, 16)
