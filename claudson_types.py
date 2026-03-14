# SPDX-License-Identifier: Proprietary-Commercial
# Copyright (c) 2026 Claudeson Project. All rights reserved.

"""
Claudeson 2026 — Shared TypedDicts for core output interfaces
=============================================================
Centralises the typed output dict schemas for modules whose callers would
otherwise have to look up key spellings in docstrings.  Using TypedDicts
here means mypy/pyright will flag key-name mismatches, missing fields, and
wrong value types at the **call site**, which is exactly where we have
historically seen attribute-name bugs.

Scope
-----
Only the "narrow waist" interfaces are typed here — the ones that cross
module boundaries and are consumed by the orchestrator, tests, or the
sidecar:

  DualPathOutput          — claudson_dual_path.DualPathPerceptionImagination.forward()
  PerceptionImaginationState
                          — "perception_imagination" sub-dict injected by
                            claudson_perception_sidecar.PerceptionImaginationSidecar

Gen-specific model outputs (action_logits, jedi_goal, …) remain Dict[str, Any]
because they vary per generation and are not crossed between modules.
"""

from __future__ import annotations

from typing import Optional

import torch
from typing_extensions import TypedDict


class DualPathOutput(TypedDict):
    """
    Return type of ``DualPathPerceptionImagination.forward()``.

    Fields
    ------
    tokens:
        Final token sequence for the reasoning stack ``[B, T, D]``.
        Equal to ``base_tokens`` in perception-only mode; fused in dual-path
        mode.
    base_tokens:
        Raw perception-path tokens ``[B, T, D]``.  Always present.
    spatial_tokens:
        WorldFM spatial tokens ``[B, T_w, D_w]``.
        ``None`` when imagination was not invoked.
    mode:
        ``"perception_only"`` — imagination path was skipped.
        ``"dual_path"``       — both paths ran; tokens are fused.
    """

    tokens: torch.Tensor
    base_tokens: torch.Tensor
    spatial_tokens: Optional[torch.Tensor]
    mode: str


class PerceptionImaginationState(TypedDict):
    """
    The ``"perception_imagination"`` sub-dict emitted by
    ``PerceptionImaginationSidecar.forward()``.

    Fields
    ------
    tokens:
        Final dual-path token sequence, or ``None`` in text-only mode.
    base_tokens:
        Raw perception tokens, or ``None`` in text-only mode.
    spatial_tokens:
        WorldFM spatial tokens, or ``None`` if imagination did not run.
    mode:
        ``"text_only"``        — no image was provided.
        ``"perception_only"``  — image provided but imagination not triggered.
        ``"dual_path"``        — both perception and imagination ran.
    imagination_active:
        ``True`` iff the WorldFM imagination path produced ``spatial_tokens``.
    """

    tokens: Optional[torch.Tensor]
    base_tokens: Optional[torch.Tensor]
    spatial_tokens: Optional[torch.Tensor]
    mode: str
    imagination_active: bool
