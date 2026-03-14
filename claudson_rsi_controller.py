# SPDX-License-Identifier: AGPL-3.0
# Copyright (c) 2026 Breaking Circuits Research. Open-Core — Generations 1-5.

"""
Claudeson 2026 - RSI Controller
================================
External guardrail for recursive self-improvement.

Evaluates candidate parameter updates (patches) before applying them,
ensuring that self-modifications improve model performance above a
configurable threshold.

This complements the in-model RecursiveSelfImprovement layer in
ClaudesonSovereign (which operates inside the forward pass) with a
standalone wrapper usable around *any* model in the stack.

Design:
  - No architectural changes to the wrapped model.
  - Save-evaluate-restore cycle: the model is never permanently modified
    by a rejected patch.
  - Thread-safe for single-process use (stateless save/restore).

Usage::

    def evaluator(model, batch):
        with torch.no_grad():
            out = model(text=batch)
        return -out["jedi_energy"].mean().item()   # lower energy = better

    rsi = RSIController(model, evaluator, threshold=0.02)

    def my_patch(m):
        for p in m.rsi.adapter_A.parameters():
            p.data += torch.randn_like(p) * 0.001

    accepted, delta = rsi.apply_if_safe(my_patch, val_batch)
    print(f"Patch {'accepted' if accepted else 'rejected'}  Δ={delta:+.4f}")
"""

import logging
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSIController
# ---------------------------------------------------------------------------


class RSIController:
    """
    External guardrail for recursive self-improvement.

    Wraps a PyTorch model and mediates all parameter updates proposed
    by self-improvement routines.  A patch is only committed when the
    evaluator score improves by at least ``threshold``.

    Args:
        model:      The PyTorch model to protect.
        evaluator:  ``Callable(model, batch) → float``.  Higher = better.
        threshold:  Minimum score improvement required to accept a patch.
                    Set higher for more conservative acceptance.
    """

    def __init__(
        self,
        model: nn.Module,
        evaluator: Callable,
        threshold: float = 0.02,
    ) -> None:
        self.model = model
        self.evaluator = evaluator
        self.threshold = threshold

        self._n_proposals: int = 0
        self._n_accepted: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate_patch(
        self,
        patch_fn: Callable[[nn.Module], None],
        validation_batch,
    ) -> float:
        """
        Measure the score improvement produced by ``patch_fn`` without
        permanently modifying the model.

        Workflow:
          1. Snapshot current state_dict.
          2. Apply patch.
          3. Score patched model.
          4. Restore snapshot.
          5. Score original model.
          6. Return (patched_score − original_score).

        Args:
            patch_fn:          A function that mutates model parameters in
                               place (e.g. updates LoRA adapter weights).
            validation_batch:  Input passed to the evaluator.

        Returns:
            float: signed improvement (positive = patch helped).
        """
        # Deep-copy the state dict so tensors are not shared
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        patch_fn(self.model)
        with torch.no_grad():
            score_new = float(self.evaluator(self.model, validation_batch))

        self.model.load_state_dict(original_state)
        with torch.no_grad():
            score_old = float(self.evaluator(self.model, validation_batch))

        improvement = score_new - score_old
        log.debug(
            "RSIController: score_old=%.4f  score_new=%.4f  Δ=%.4f",
            score_old,
            score_new,
            improvement,
        )
        return improvement

    def apply_if_safe(
        self,
        patch_fn: Callable[[nn.Module], None],
        validation_batch,
    ) -> Tuple[bool, float]:
        """
        Evaluate ``patch_fn`` and apply it only if it clears the threshold.

        Args:
            patch_fn:          Proposed parameter mutation.
            validation_batch:  Input passed to the evaluator.

        Returns:
            accepted (bool):    Whether the patch was committed.
            improvement (float): Measured score delta.
        """
        self._n_proposals += 1
        improvement = self.evaluate_patch(patch_fn, validation_batch)

        if improvement > self.threshold:
            patch_fn(self.model)
            self._n_accepted += 1
            log.info(
                "RSIController: patch ACCEPTED  Δ=%.4f  (threshold=%.4f)",
                improvement,
                self.threshold,
            )
            return True, improvement

        log.info(
            "RSIController: patch REJECTED  Δ=%.4f  (threshold=%.4f)", improvement, self.threshold
        )
        return False, improvement

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed patches that have been accepted."""
        if self._n_proposals == 0:
            return 0.0
        return self._n_accepted / self._n_proposals

    def stats(self) -> Dict[str, object]:
        """Return a summary of controller activity."""
        return {
            "n_proposals": self._n_proposals,
            "n_accepted": self._n_accepted,
            "acceptance_rate": self.acceptance_rate,
            "threshold": self.threshold,
        }

    def reset_stats(self) -> None:
        """Reset proposal/acceptance counters (does not affect model weights)."""
        self._n_proposals = 0
        self._n_accepted = 0


# ---------------------------------------------------------------------------
# SimpleEvaluator — convenience wrapper for negative-loss scoring
# ---------------------------------------------------------------------------


class SimpleEvaluator:
    """
    Thin adapter that turns a loss function into an RSIController evaluator.

    ``evaluator(model, batch) → float`` required by RSIController is just
    the negated loss: lower loss → higher score.

    Args:
        loss_fn:   ``Callable(model, batch) → Tensor`` — scalar loss.
    """

    def __init__(self, loss_fn: Callable) -> None:
        self.loss_fn = loss_fn

    def __call__(self, model: nn.Module, batch) -> float:
        with torch.no_grad():
            loss = self.loss_fn(model, batch)
        return -float(loss)
