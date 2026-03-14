from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BaseModelArgs — single source of truth for fields shared by every generation
# ---------------------------------------------------------------------------


@dataclass
class BaseModelArgs:
    """Shared configuration base for all Claudeson generations (G1–G9+).

    Every per-generation ModelArgs must subclass this class.
    The ``__post_init__`` hook validates structural invariants so that
    a misconfigured model fails immediately with a clear message rather
    than crashing deep inside a forward pass.

    Generation-specific fields are added in each generation's own
    ``ModelArgs`` subclass.  Override a field default by redefining it
    in the subclass, e.g.::

        @dataclass
        class ModelArgs(BaseModelArgs):
            max_seq_len: int = 131_072   # G2+ extended context
            use_yarn: bool = True
    """

    # ── Architecture ────────────────────────────────────────────────────────
    dim: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8  # GQA: must evenly divide n_heads
    vocab_size: int = 128_000

    # ── Multimodal inputs ────────────────────────────────────────────────────
    patch_size: int = 16
    img_size: int = 224
    audio_spec_dim: int = 128

    # ── Sequence / context ───────────────────────────────────────────────────
    max_seq_len: int = 8_192

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_slots: int = 256
    episodic_slots: int = 2_560

    # ── Planning / agent ─────────────────────────────────────────────────────
    action_space_size: int = 100
    planning_horizon: int = 8
    num_simulations: int = 8
    env_state_dim: int = 128
    goal_dim: int = 2_048

    # ── Mixture-of-Experts ───────────────────────────────────────────────────
    num_experts: int = 8
    expert_top_k: int = 2  # Must be ≤ num_experts

    # ── Training flags ───────────────────────────────────────────────────────
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    use_kv_cache: bool = True

    # ────────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Validate configuration invariants shared by all generations.

        Raises ``ValueError`` (collecting all problems) so callers see
        every broken constraint in a single exception rather than having
        to fix them one at a time.
        """
        errors: list[str] = []

        # GQA: n_kv_heads must evenly divide n_heads
        if self.n_heads % self.n_kv_heads != 0:
            errors.append(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )

        # Head dimension must be an integer
        if self.dim % self.n_heads != 0:
            errors.append(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )

        # MoE: cannot activate more experts than exist
        if self.expert_top_k > self.num_experts:
            errors.append(
                f"expert_top_k ({self.expert_top_k}) must be <= "
                f"num_experts ({self.num_experts})"
            )

        # Strictly-positive scalars
        for name, val in [
            ("dim", self.dim),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("vocab_size", self.vocab_size),
            ("max_seq_len", self.max_seq_len),
            ("num_experts", self.num_experts),
            ("expert_top_k", self.expert_top_k),
        ]:
            if val <= 0:
                errors.append(f"{name} must be positive (got {val})")

        if errors:
            raise ValueError(
                "ModelArgs validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
