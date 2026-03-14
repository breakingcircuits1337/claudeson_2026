"""
Claudeson 2026 — Token Fusion Module
======================================
Late-fusion gate that merges perception tokens and WorldFM spatial tokens
into a single sequence for the main reasoning stack.

Design principles
─────────────────
* Late fusion, not replacement.  The base perception path always runs.
  Spatial tokens augment it when available; they never substitute it.

* Shared token dimension.  Both inputs must arrive as [B, T, D] before
  fusing.  Each path is responsible for projecting to D externally
  (VisionEncoder → D, WorldFMAdapter.encode_reference → D_w → D via
  TokenFusionModule.spatial_proj).

* Normalise before fusing.  LayerNorm on the concatenated features
  keeps the merge numerically stable regardless of scale differences
  between the two paths.

* Sequence-length mismatch handled by truncation to min(T_base, T_spatial).
  Padding strategies are left to callers if they need full-length output.

Data contract
─────────────
  base_tokens:     [B, T_base,    D]
  spatial_tokens:  [B, T_spatial, D_w]  (D_w may differ from D)
  output:          [B, T_fused,   D]    where T_fused = min(T_base, T_spatial)
"""

# SPDX-License-Identifier: Proprietary-Commercial
# Copyright (c) 2026 Claudeson Project. All rights reserved.

from typing import Optional

import torch
import torch.nn as nn


class TokenFusionModule(nn.Module):
    """
    Late-fusion of perception tokens and spatial tokens.

    If spatial_input_dim != model_dim, an input projection is applied to
    spatial_tokens before concatenation.

    Args:
        model_dim:          dimension D of base perception tokens
        spatial_input_dim:  dimension D_w of WorldFM spatial tokens;
                            defaults to model_dim (no projection needed)
    """

    def __init__(self, model_dim: int, spatial_input_dim: Optional[int] = None):
        super().__init__()
        self.model_dim = model_dim
        spatial_input_dim = spatial_input_dim or model_dim

        # Optional projection when WorldFM token dim ≠ model dim
        if spatial_input_dim != model_dim:
            self.spatial_proj: nn.Module = nn.Linear(spatial_input_dim, model_dim, bias=False)
        else:
            self.spatial_proj = nn.Identity()

        self.fuse = nn.Linear(model_dim * 2, model_dim, bias=False)
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        base_tokens: torch.Tensor,  # [B, T_base,    D]
        spatial_tokens: torch.Tensor,  # [B, T_spatial, D_w]
    ) -> torch.Tensor:  # [B, T_fused,   D]
        """
        Fuse perception and spatial tokens.

        Sequence lengths are aligned by truncation.  The spatial tokens
        are projected to D if necessary, then concatenated with the base
        tokens along the feature axis and passed through a linear + norm.
        """
        spatial_tokens = self.spatial_proj(spatial_tokens)  # [B, T_s, D]

        n = min(base_tokens.shape[1], spatial_tokens.shape[1])
        base = base_tokens[:, :n]  # [B, n, D]
        spatial = spatial_tokens[:, :n]  # [B, n, D]

        fused = torch.cat([base, spatial], dim=-1)  # [B, n, 2D]
        return self.norm(self.fuse(fused))  # [B, n, D]
