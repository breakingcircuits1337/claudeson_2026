# SPDX-License-Identifier: LicenseRef-Claudeson-Commercial
# Copyright (c) 2026 Breaking Circuits Research.
# Commercial — Generations 6-9.

"""
Claudson 2026 — Perception + Imagination Sidecar
==================================================
Attaches dual-path perception and on-demand imagination to **any** existing
Claudson core model without modifying that model's weights or architecture.

Design principle — sidecar, not replacement
--------------------------------------------
The sidecar runs in parallel with the wrapped model.  The core model is called
with its original inputs unchanged.  The dual-path module processes an
optionally provided image independently.  A small set of learned correction
heads then apply residual adjustments to ``action_logits`` and ``value`` in
the model output.  When no image is provided, the sidecar is fully transparent.

Architecture::

                   text / other kwargs
                         │
                         ▼
              ┌─────────────────────┐
              │     Core Model      │ ← weights never modified
              └─────────────────────┘
                         │  model_out
                         │
    image (optional) ────┤
    K, c2w, context      │
                         ▼
              ┌─────────────────────┐
              │  DualPath           │  always-on perception path
              │  Perception +       │  + on-demand WorldFM imagination
              │  Imagination        │    (gated by invocation router)
              └─────────────────────┘
                         │  dp_out["tokens"]  [B, T_w, D_w]
                         │
                         ▼
              ┌─────────────────────┐
              │  Pool → Project     │  D_w  →  model_dim
              │  Gate + Correction  │  sigmoid gate × correction_scale
              │  Heads              │  δ(action_logits), δ(value)
              └─────────────────────┘
                         │
                         ▼
              enriched model_out dict
                + "perception_imagination" key

Transparency guarantee
----------------------
* ``image=None`` — sidecar is a no-op; ``model_out`` is returned as-is,
  ``perception_imagination["mode"]`` is ``"text_only"``.
* ``enabled=False`` in ``SidecarConfig`` — same as above.
* If the wrapped model's output does not contain ``"action_logits"`` or
  ``"value"``, those corrections are silently skipped.

Usage::

    from claudson_perception_sidecar import (
        PerceptionImaginationSidecar, SidecarConfig
    )

    cfg = SidecarConfig(dual_path_dim=32, model_dim=2048, action_space_size=100)
    sidecar = PerceptionImaginationSidecar(core_model, dual_path_module, cfg)

    result = sidecar(
        text=tokens,
        image=img_tensor,          # [B, 3, H, W]
        K=camera_intrinsics,       # [B, 3, 3]  — optional
        c2w=camera_extrinsics,     # [B, 4, 4]  — optional
        context={"task": "scene_reconstruction"},
    )
    print(result["perception_imagination"]["mode"])   # "dual_path" or "perception_only"
    print(result["action_logits"].shape)              # same as core model + δ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from claudson_dual_path import DualPathPerceptionImagination
from claudson_types import PerceptionImaginationState
from claudson_utils import RMSNorm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SidecarConfig:
    """
    Configuration for ``PerceptionImaginationSidecar``.

    Attributes:
        dual_path_dim:      Output token dimension of the
                            ``DualPathPerceptionImagination`` module
                            (must match ``WorldFMConfig.spatial_dim``).
        model_dim:          Hidden dimension of the wrapped core model.
                            Used to size the token projector.
        action_space_size:  Size of the ``action_logits`` vector in the core
                            model output.  The logit-correction head is shaped
                            to match this.
        correction_scale:   Maximum scale of additive corrections applied to
                            ``action_logits`` and ``value``.  Kept small
                            (default 0.1) so the sidecar starts as a minor
                            perturbation and grows through training.
        enabled:            When False, the sidecar is a transparent pass-through
                            regardless of what inputs are provided.
    """

    dual_path_dim: int = 32
    model_dim: int = 2048
    action_space_size: int = 100
    correction_scale: float = 0.1
    enabled: bool = True


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


class PerceptionImaginationSidecar(nn.Module):
    """
    Attaches dual-path perception and on-demand imagination to any core model.

    The sidecar owns three lightweight learnable components:

    * ``token_projector`` — linear map ``[dual_path_dim → model_dim]`` for
      broadcasting perception tokens into the model's embedding space.
    * ``correction_heads`` — two small linear heads that produce additive
      residuals for ``action_logits`` and ``value``.
    * ``gate`` — a sigmoid gate head that scales all corrections down to zero
      when the dual-path signal carries no useful information (e.g., when
      imagination produces uninformative tokens early in training).

    The wrapped core model and the dual-path module are treated as frozen
    from the sidecar's perspective — their parameters are not registered as
    part of the sidecar.  To include them in a joint optimisation, pass them
    to the optimiser separately.

    Args:
        model:      Any ``nn.Module`` whose ``forward(**kwargs)`` returns a
                    ``Dict``.  The sidecar calls it with its own ``**model_kwargs``.
        dual_path:  Pre-built ``DualPathPerceptionImagination`` instance.
        config:     Sidecar hyper-parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        dual_path: DualPathPerceptionImagination,
        config: Optional[SidecarConfig] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dual_path = dual_path
        self._cfg = config or SidecarConfig()
        cfg = self._cfg

        d_w = cfg.dual_path_dim  # dual-path token dim
        d_m = cfg.model_dim  # core model hidden dim

        # Normalise pooled tokens before passing to heads
        self.pool_norm = RMSNorm(d_w)

        # Project perception tokens into the model's embedding space
        # (kept separate from correction heads so it can be used for
        #  future attention-based fusion without changing the heads)
        self.token_projector = nn.Linear(d_w, d_m, bias=False)

        # Gating: sigmoid(linear(pool)) ∈ (0, 1) — scalar gate per sample
        self.gate = nn.Sequential(
            nn.Linear(d_w, 1),
            nn.Sigmoid(),
        )

        # Correction heads
        self.logit_correction = nn.Linear(d_w, cfg.action_space_size, bias=True)
        self.value_correction = nn.Linear(d_w, 1, bias=True)

        # Initialise corrections to near-zero so the sidecar starts transparent
        nn.init.zeros_(self.logit_correction.weight)
        nn.init.zeros_(self.logit_correction.bias)
        nn.init.zeros_(self.value_correction.weight)
        nn.init.zeros_(self.value_correction.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        *,
        image: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run core model + sidecar in parallel and return enriched output.

        Args:
            image:        Optional image tensor ``[B, 3, H, W]``.  When None,
                          the sidecar is a transparent pass-through.
            K:            Camera intrinsics ``[B, 3, 3]`` (passed to dual-path
                          router; None disables imagination).
            c2w:          Camera-to-world ``[B, 4, 4]`` (same as K).
            context:      Routing context dict (e.g. ``{"task": "scene_reconstruction",
                          "uncertainty": 0.6}``).
            **model_kwargs: Forwarded unchanged to ``self.model(**model_kwargs)``.

        Returns:
            Dict containing:

            * All keys from the wrapped model's output.
            * ``"perception_imagination"`` — dual-path sub-dict with keys:
              ``"tokens"``, ``"base_tokens"``, ``"spatial_tokens"``,
              ``"mode"``, ``"imagination_active"`` (bool).
            * ``"perception_signal"`` — projected dual-path summary vector
              ``[B, model_dim]``; useful for downstream attention or
              the orchestrator's RSI hidden-state extraction.
            * If core model has ``"action_logits"``:
              the returned ``"action_logits"`` includes the sidecar correction.
            * If core model has ``"value"``:
              the returned ``"value"`` includes the sidecar correction.
        """
        # -------------------------------------------------------------------
        # 1. Core model (always)
        # -------------------------------------------------------------------
        model_out: Dict[str, Any] = self.model(**model_kwargs)
        out = dict(model_out)

        # -------------------------------------------------------------------
        # 2. Dual-path pass-through when disabled or no image provided
        # -------------------------------------------------------------------
        if not self._cfg.enabled or image is None:
            out["perception_imagination"] = PerceptionImaginationState(
                tokens=None,
                base_tokens=None,
                spatial_tokens=None,
                mode="text_only",
                imagination_active=False,
            )
            return out

        # -------------------------------------------------------------------
        # 3. Dual-path: perception (always) + imagination (router-gated)
        # -------------------------------------------------------------------
        with torch.no_grad():
            dp_out = self.dual_path(image=image, K=K, c2w=c2w, context=context)

        dp_tokens: torch.Tensor = dp_out["tokens"]  # [B, T_w, D_w]
        imagination_active = dp_out["mode"] == "dual_path"

        out["perception_imagination"] = PerceptionImaginationState(
            tokens=dp_tokens,
            base_tokens=dp_out["base_tokens"],
            spatial_tokens=dp_out["spatial_tokens"],
            mode=dp_out["mode"],
            imagination_active=imagination_active,
        )

        # -------------------------------------------------------------------
        # 4. Pool → normalise → gate → corrections
        # -------------------------------------------------------------------
        dp_pooled = dp_tokens.mean(dim=1)  # [B, D_w]
        dp_pooled = self.pool_norm(dp_pooled)  # [B, D_w]

        gate = self.gate(dp_pooled)  # [B, 1] ∈ (0, 1)
        scale = self._cfg.correction_scale * gate  # [B, 1]

        # Project to model_dim for downstream use
        out["perception_signal"] = self.token_projector(dp_pooled)  # [B, D_m]

        # Logit correction (only when model provides action_logits)
        if "action_logits" in model_out:
            delta_logits = self.logit_correction(dp_pooled)  # [B, A]
            out["action_logits"] = model_out["action_logits"] + scale * delta_logits

        # Value correction (only when model provides value)
        if "value" in model_out:
            delta_value = self.value_correction(dp_pooled)  # [B, 1]
            out["value"] = model_out["value"] + scale * delta_value

        log.debug(
            "Sidecar: mode=%s gate_mean=%.3f",
            dp_out["mode"],
            gate.mean().item(),
        )

        return out
