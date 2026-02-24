"""
Unit tests for the Rotary Positional Embedding (RoPE) implementation in claudson_jedi.py.
These tests verify mathematical properties like norm preservation, dot product
preservation, and identity transformations.
"""

import torch
import pytest
from claudson_jedi import apply_rotary_pos_emb

def test_apply_rotary_pos_emb_shapes():
    """Verify that the output tensors have the same shape as the input tensors."""
    batch_size = 2
    seq_len = 16
    n_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    cos = torch.randn(seq_len, head_dim)
    sin = torch.randn(seq_len, head_dim)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_apply_rotary_pos_emb_identity():
    """Verify that the transformation is identity when cos=1 and sin=0."""
    batch_size = 2
    seq_len = 16
    n_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # cos = 1, sin = 0 should be identity
    cos = torch.ones(seq_len, head_dim)
    sin = torch.zeros(seq_len, head_dim)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(q_out, q, atol=1e-6)
    assert torch.allclose(k_out, k, atol=1e-6)

def test_apply_rotary_pos_emb_norm_preservation():
    """Verify that RoPE preserves the L2 norm of the vectors."""
    batch_size = 1
    seq_len = 1
    n_heads = 1
    head_dim = 64

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # Random cos and sin, but they should satisfy cos^2 + sin^2 = 1 for each pair
    # In RoPE, cos and sin are applied to pairs (x_i, x_{i+d/2})
    angle = torch.randn(seq_len, head_dim // 2)
    angle = torch.cat([angle, angle], dim=-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    # Check norm preservation for each head and each position
    q_norm_in = torch.norm(q, dim=-1)
    q_norm_out = torch.norm(q_out, dim=-1)
    assert torch.allclose(q_norm_in, q_norm_out, atol=1e-6)

    k_norm_in = torch.norm(k, dim=-1)
    k_norm_out = torch.norm(k_out, dim=-1)
    assert torch.allclose(k_norm_in, k_norm_out, atol=1e-6)

def test_apply_rotary_pos_emb_dot_product_preservation():
    """Verify that RoPE preserves the dot product between two vectors when they are rotated identically."""
    batch_size = 1
    seq_len = 1
    n_heads = 1
    head_dim = 64

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    angle = torch.randn(seq_len, head_dim // 2)
    angle = torch.cat([angle, angle], dim=-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    dot_in = (q * k).sum(dim=-1)
    dot_out = (q_out * k_out).sum(dim=-1)

    assert torch.allclose(dot_in, dot_out, atol=1e-6)

def test_apply_rotary_pos_emb_rotation_values():
    """Verify rotation values for a known simple 2D case (90 degree rotation)."""
    # Simple 2D case
    # q = [1, 0], angle = pi/2
    # rotate_half([1, 0]) -> chunk([1, 0]) -> [1], [0] -> cat([-0], [1]) -> [0, 1]
    # cos = cos(pi/2) = 0, sin = sin(pi/2) = 1
    # out = [1, 0] * 0 + [0, 1] * 1 = [0, 1]

    q = torch.tensor([[[[1.0, 0.0]]]]) # [B, H, L, D] = [1, 1, 1, 2]
    k = torch.tensor([[[[0.0, 1.0]]]])

    cos = torch.tensor([[0.0, 0.0]]) # [L, D]
    sin = torch.tensor([[1.0, 1.0]])

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)

    # q_out should be [0, 1]
    expected_q_out = torch.tensor([[[[0.0, 1.0]]]])
    # k_out: rotate_half([0, 1]) -> [-1, 0]
    # k_out = [0, 1] * 0 + [-1, 0] * 1 = [-1, 0]
    expected_k_out = torch.tensor([[[[-1.0, 0.0]]]])

    assert torch.allclose(q_out, expected_q_out, atol=1e-6)
    assert torch.allclose(k_out, expected_k_out, atol=1e-6)
