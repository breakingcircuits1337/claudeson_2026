"""
Claudeson 2026 — Dual-Path Perception + Imagination
=====================================================
Explicit subsystem that combines a fast discriminative perception path
with an on-demand generative/spatial imagination path backed by WorldFM.

Architecture
────────────

  ┌─────────────┐    always-on, fast
  │ Perception  │──────────────────────────────► base_tokens [B, T, D]
  │   Path      │
  └─────────────┘
                                                         │
                     ┌─ invoke? ─────────────────────────┤
                     │  (pose present + routing policy)   │
                     ▼                                    │
  ┌─────────────┐   on-demand, slower             ┌──────┴──────┐
  │ Imagination │──► spatial_tokens [B, T_w, D_w]─► Fusion Mod. ├─► fused_tokens
  │   Path      │                                 └─────────────┘
  └─────────────┘

Key properties
  * Perception path is always active.
  * Imagination path is gated by WorldFMInvocationRouter.
  * When imagination is not invoked, base_tokens flow through unchanged.
  * Token fusion uses late-fusion (concat + linear + norm).
  * Both "observed" and "imagined" evidence use typed SceneMemoryEntry.

Data contracts
  image:       [B, 3, H, W]
  K:           [B, 3, 3]  camera intrinsics   (optional)
  c2w:         [B, 4, 4]  camera extrinsics   (optional)
  base_tokens: [B, T,   D]
  spatial_tok: [B, T_w, D_w]

Classes
  WorldFMInvocationRouter   — decides whether to invoke WorldFM
  TokenFusionModule         — re-exported from claudson_token_fusion
  DualPathPerceptionImagination — top-level module
  SpatialTreeSearchPlanner  — planner that uses rollout_views for lookahead
  SceneMemoryEntry          — typed memory entry for observed/imagined scenes
"""

# SPDX-License-Identifier: Proprietary-Commercial
# Copyright (c) 2026 Claudeson Project. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from claudson_token_fusion import TokenFusionModule
from claudson_worldfm_adapter import WorldFMAdapter

log = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "WorldFMInvocationRouter",
    "TokenFusionModule",
    "DualPathPerceptionImagination",
    "SpatialTreeSearchPlanner",
    "SceneMemoryEntry",
]


# ─── Routing policy ───────────────────────────────────────────────────────────

SPATIAL_TASK_SET = frozenset(
    {
        "counterfactual_imagination",
        "tree_search_planning",
        "scene_reconstruction",
        "threat_surface_analysis",
        "infrastructure_inspection",
        "multi_view_consistency",
        "occlusion_reasoning",
        "blind_spot_analysis",
    }
)


class WorldFMInvocationRouter(nn.Module):
    """
    Decides whether to invoke the WorldFM imagination path.

    Rules (any True → invoke):
      1. Pose metadata (K, c2w) is present AND not None
      2. Uncertainty score exceeds threshold
      3. Context task is in the spatial-task set
      4. Context explicitly requests WorldFM

    Args:
        uncertainty_gate: optional UncertaintyGate for threshold-based gating
        threshold:        uncertainty threshold (used when no gate is provided)
    """

    def __init__(
        self,
        uncertainty_gate: Optional[nn.Module] = None,
        threshold: float = 0.4,
    ):
        super().__init__()
        self.uncertainty_gate = uncertainty_gate
        self.threshold = threshold

    def forward(
        self,
        base_tokens: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns True when the imagination path should be invoked.

        Note: returns a plain Python bool so callers can use it in
        ordinary `if` statements without `.item()` overhead.
        """
        # Hard gate: pose must be present to produce meaningful spatial tokens
        if K is None or c2w is None:
            return False

        if context is None:
            context = {}

        # Explicit override
        if context.get("force_worldfm", False):
            return True

        # Uncertainty gate
        uncertainty = float(context.get("uncertainty", 0.0))
        if self.uncertainty_gate is not None:
            # Use the gate's threshold if one is provided
            if not self.uncertainty_gate.allow_action(uncertainty):
                return True  # high uncertainty → invoke imagination
        else:
            if uncertainty > self.threshold:
                return True

        # Task-based routing
        task = context.get("task", "")
        if task in SPATIAL_TASK_SET:
            return True

        # Planner request
        if context.get("planner_requests_spatial", False):
            return True

        return False


# ─── Top-level dual-path module ───────────────────────────────────────────────


class DualPathPerceptionImagination(nn.Module):
    """
    Combines fast perception with on-demand pose-conditioned imagination.

    The module always runs the perception path.  The imagination path
    is invoked only when the router approves and pose metadata is
    present.  When imagination runs, the outputs of both paths are
    merged by fusion_module before being returned.

    Args:
        vision_encoder:   nn.Module  image → [B, T, D]
        worldfm_adapter:  WorldFMAdapter
        fusion_module:    TokenFusionModule
        invocation_router: WorldFMInvocationRouter
        memory_manager:   optional MemoryManager from claudson_agent_swarm
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        worldfm_adapter: WorldFMAdapter,
        fusion_module: TokenFusionModule,
        invocation_router: WorldFMInvocationRouter,
        memory_manager: Optional[Any] = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.worldfm_adapter = worldfm_adapter
        self.fusion_module = fusion_module
        self.invocation_router = invocation_router
        self.memory_manager = memory_manager

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def perceive(self, image: torch.Tensor) -> torch.Tensor:
        """Run perception path only.  Returns [B, T, D]."""
        return self.vision_encoder(image)

    def imagine(
        self,
        image: torch.Tensor,
        src_K: torch.Tensor,
        src_c2w: torch.Tensor,
        trajectory: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run imagination path rollout.  Returns list of [B, 3, H, W] frames."""
        return self.worldfm_adapter.rollout_views(
            image=image, src_K=src_K, src_c2w=src_c2w, trajectory=trajectory
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,  # [B, 3, H, W]
        K: Optional[torch.Tensor] = None,  # [B, 3, 3]
        c2w: Optional[torch.Tensor] = None,  # [B, 4, 4]
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the dual-path forward pass.

        Returns a dict with keys:
          "tokens"         — final tokens for the reasoning stack [B, T, D]
          "base_tokens"    — raw perception tokens [B, T, D]
          "spatial_tokens" — WorldFM spatial tokens or None
          "mode"           — "perception_only" or "dual_path"
        """
        base_tokens = self.perceive(image)  # [B, T, D]

        invoke_worldfm = self.invocation_router(
            base_tokens=base_tokens, K=K, c2w=c2w, context=context
        )

        if not invoke_worldfm:
            return {
                "tokens": base_tokens,
                "base_tokens": base_tokens,
                "spatial_tokens": None,
                "mode": "perception_only",
            }

        # Imagination path
        spatial_tokens = self.worldfm_adapter.encode_reference(
            image=image, K=K, c2w=c2w
        )  # [B, T_w, D_w]

        fused_tokens = self.fusion_module(base_tokens, spatial_tokens)

        return {
            "tokens": fused_tokens,
            "base_tokens": base_tokens,
            "spatial_tokens": spatial_tokens,
            "mode": "dual_path",
        }


# ─── Spatial tree-search planner ──────────────────────────────────────────────


class SpatialTreeSearchPlanner:
    """
    Tree-search planner that uses WorldFM rollouts for spatial lookahead.

    Each node in the search tree corresponds to a candidate camera
    trajectory.  The planner expands a node by generating novel-view
    frames along the trajectory, encoding them, and scoring the result.

    This is a pure-Python class (not nn.Module) because it orchestrates
    existing modules rather than holding learnable parameters.

    Args:
        worldfm_adapter: WorldFMAdapter for generating rollout frames
        vision_encoder:  encoder to re-encode generated frames
        scorer:          callable(tokens [B, T, D]) → scalar score
    """

    def __init__(
        self,
        worldfm_adapter: WorldFMAdapter,
        vision_encoder: nn.Module,
        scorer: Callable[[torch.Tensor], float],
    ):
        self.worldfm_adapter = worldfm_adapter
        self.vision_encoder = vision_encoder
        self.scorer = scorer

    def expand_node(
        self,
        ref_image: torch.Tensor,  # [B, 3, H, W]
        src_K: torch.Tensor,  # [B, 3, 3]
        src_c2w: torch.Tensor,  # [B, 4, 4]
        candidate_trajectories: List[List[torch.Tensor]],  # list of trajectories
    ) -> List[Dict[str, Any]]:
        """
        Expand a search node by evaluating each candidate trajectory.

        Each trajectory is a list of target c2w matrices [B, 4, 4].

        Returns a list of branch dicts, one per trajectory:
          "trajectory" — the input trajectory
          "frames"     — generated frames, list of [B, 3, H, W]
          "tokens"     — encoded frames, list of [B, T, D]
          "score"      — mean scorer output across frames (float)
        """
        branches = []

        for trajectory in candidate_trajectories:
            frames = self.worldfm_adapter.rollout_views(
                image=ref_image,
                src_K=src_K,
                src_c2w=src_c2w,
                trajectory=trajectory,
            )

            encoded = [self.vision_encoder(frame) for frame in frames]
            score = sum(self.scorer(t) for t in encoded) / max(len(encoded), 1)

            branches.append(
                {
                    "trajectory": trajectory,
                    "frames": frames,
                    "tokens": encoded,
                    "score": score,
                }
            )

        return branches

    def best_trajectory(
        self,
        ref_image: torch.Tensor,
        src_K: torch.Tensor,
        src_c2w: torch.Tensor,
        candidate_trajectories: List[List[torch.Tensor]],
    ) -> Optional[Dict[str, Any]]:
        """
        Return the highest-scoring branch, or None if no trajectories given.
        """
        if not candidate_trajectories:
            return None
        branches = self.expand_node(ref_image, src_K, src_c2w, candidate_trajectories)
        return max(branches, key=lambda b: b["score"])


# ─── Scene memory entry ───────────────────────────────────────────────────────


@dataclass
class SceneMemoryEntry:
    """
    Typed memory entry for one scene observation or imagination.

    Fields
    ──────
    kind:       "observed" or "imagined"
    image_id:   unique identifier for the source image
    pose:       optional dict with "K" and "c2w" tensors
    tokens:     encoded representation [T, D]
    source:     module that produced this entry (e.g. "vision_encoder")
    confidence: float in [0, 1]; imagined entries should be < 1.0

    Rules
    ─────
    * observed entries outrank imagined entries at equal confidence
    * imagined entries must include provenance in source
    * imagined entries must never silently overwrite observed state
    """

    kind: str  # "observed" | "imagined"
    image_id: str
    tokens: torch.Tensor  # [T, D]
    source: str = "unknown"
    confidence: float = 1.0
    pose: Optional[Dict[str, torch.Tensor]] = field(default=None)

    def is_observed(self) -> bool:
        return self.kind == "observed"

    def is_imagined(self) -> bool:
        return self.kind == "imagined"

    def outranks(self, other: "SceneMemoryEntry") -> bool:
        """
        Return True if this entry should take precedence over `other`.
        Observed > imagined at equal confidence; higher confidence wins.
        """
        if self.is_observed() and other.is_imagined():
            return True
        if self.is_imagined() and other.is_observed():
            return False
        return self.confidence > other.confidence
