# SPDX-License-Identifier: LicenseRef-Claudeson-Commercial
# Copyright (c) 2026 Breaking Circuits Research.
# Commercial — Generations 6-9.

"""
Claudeson 2026 — Top-Level Orchestrator
=========================================
Unifies five governance and execution components into a single inference loop:

  1. UncertaintyGate             — abstain when epistemic uncertainty is too high.
  2. CapabilityGate              — block RSI adapter commits that fail safety tests.
  3. TemporalPlanner             — derive multi-scale action plans.
  4. MemoryManager               — store and retrieve experiences across steps.
  5. RecursiveSelfImprovement    — propose and gate low-rank adapter self-edits.

Pipeline per ``step()`` call::

    Memory retrieve
        → Forward pass (model)
        → Uncertainty gate  (abstain if high uncertainty)
        → Temporal plan     (extract from model output)
        → RSI propose       (skipped when abstaining)
        → Capability gate   (roll back RSI commit if any test fails)
        → Memory write

Usage::

    from claudson_orchestrator import ClaudesonOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(dim=512, uncertainty_threshold=0.3)
    orch = ClaudesonOrchestrator(model, cfg)

    # Optionally register safety tests that gate RSI adapter commits
    orch.register_capability_test(
        lambda m: compute_harm_score(m) < 0.05, name="harm_score"
    )

    result = orch.step({"text": tokens}, query_embedding=q_emb)
    if result.abstained:
        print("Abstaining — uncertainty too high")
    else:
        print("Plan:", result.plan)
        print("RSI accepted:", result.rsi_out["accepted"] if result.rsi_out else False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from claudson_agent_swarm import CapabilityGate, MemoryEntry, MemoryManager
from claudson_sovereign import RecursiveSelfImprovement
from claudson_temporal_reasoning import TemporalPlanner
from claudson_uncertainty import UncertaintyGate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """
    Configuration for ``ClaudesonOrchestrator``.

    All parameters have safe defaults that match the Claudeson 2026 training
    regime.  Override only what you need.
    """

    # --- Architecture --------------------------------------------------
    dim: int = 2048
    """Model hidden dimension.  Used to size the standalone RSI module when
    the wrapped model does not expose its own ``rsi`` sub-module."""

    # --- Uncertainty gate ---------------------------------------------
    uncertainty_threshold: float = 0.4
    """Uncertainty score above which the gate abstains.  Range [0, 1]."""

    uncertainty_hysteresis: bool = False
    """Enable hysteresis to prevent rapid open/close oscillation at the
    threshold boundary."""

    uncertainty_margin: float = 0.05
    """Half-width of the hysteresis band (only used when
    ``uncertainty_hysteresis=True``)."""

    # --- Memory -------------------------------------------------------
    max_ephemeral: int = 256
    """Maximum number of ephemeral memory entries before oldest is evicted."""

    memory_top_k: int = 5
    """Number of memory entries to retrieve per step."""

    # --- Temporal planner ---------------------------------------------
    plan_steps: int = 5
    """Number of simulation steps when the planner falls back to
    ``TemporalPlanner.simulate()`` (only used when the model output does
    not already contain temporal plan data)."""

    # --- RSI controller -----------------------------------------------
    rsi_enabled: bool = True
    """Enable the RSI controller.  Set False to disable all self-editing."""

    rsi_rank: int = 8
    """LoRA adapter rank for the standalone RSI module."""

    rsi_horizon: int = 3
    """Imagination rollout horizon used by RSI for EFE estimation."""

    rsi_threshold: float = 0.05
    """Minimum predicted EFE improvement for RSI to commit an adapter delta."""

    # --- Capability gate ----------------------------------------------
    capability_check_on_rsi_accept: bool = True
    """When True, run all registered capability tests every time RSI commits
    an adapter delta.  The commit is rolled back if any test fails."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """
    Output of a single ``ClaudesonOrchestrator.step()`` call.

    Attributes:
        model_out:          Raw ``Dict`` returned by the wrapped model's
                            ``forward()``.
        abstained:          True if the uncertainty gate blocked all actions
                            for the majority of the batch this step.
        uncertainty_scores: Per-sample uncertainty score tensor ``[B]``.
        capability_ok:      True if the capability gate approved the RSI
                            adapter commit (or if RSI did not accept / was
                            disabled).
        plan:               Temporal plan dict extracted from ``model_out``,
                            or a trajectory list from ``TemporalPlanner``
                            simulation, or None if unavailable.
        rsi_out:            RSI controller output dict (keys: ``improvement``,
                            ``accept_prob``, ``accepted``, ``acceptance_rate``,
                            ``n_proposals``), or None if RSI was skipped.
        memory_context:     Memory entries retrieved before the forward pass.
    """

    model_out: Dict[str, Any]
    abstained: bool
    uncertainty_scores: torch.Tensor
    capability_ok: bool
    plan: Optional[Any]
    rsi_out: Optional[Dict[str, Any]]
    memory_context: List[MemoryEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClaudesonOrchestrator:
    """
    Top-level orchestrator that unifies the five core governance and execution
    components of the Claudeson 2026 stack.

    Wraps any ``nn.Module`` model and coordinates:

    * **UncertaintyGate** — abstains from acting when epistemic uncertainty
      exceeds ``OrchestratorConfig.uncertainty_threshold``, preventing
      overconfident decisions.

    * **CapabilityGate** — evaluates the model against registered safety tests
      before committing any RSI adapter change; rolls back the delta when any
      test fails.

    * **TemporalPlanner** — extracts multi-scale plans from the model output
      (``out["temporal"]["plans"]``) when available, or falls back to a
      closed-loop simulation via ``TemporalPlanner.simulate()``.

    * **MemoryManager** — retrieves relevant past experiences (via cosine
      similarity) before each forward pass and stores new experiences after.

    * **RecursiveSelfImprovement** — proposes low-rank LoRA adapter deltas;
      commits only when the predicted EFE improvement exceeds
      ``rsi_threshold`` and all registered capability tests pass.

    Component injection
    ~~~~~~~~~~~~~~~~~~~
    Every component can be injected as a keyword argument so that state is
    shared across orchestrator instances or restored from a checkpoint::

        shared_memory = MemoryManager(max_ephemeral=1024)
        orch_a = ClaudesonOrchestrator(model_a, memory=shared_memory)
        orch_b = ClaudesonOrchestrator(model_b, memory=shared_memory)

    If a component is not injected, a fresh instance is created from the
    ``OrchestratorConfig`` defaults.

    RSI sub-module resolution order
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Injected ``rsi`` kwarg.
    2. ``model.rsi`` — used when the wrapped model already exposes its own
       ``RecursiveSelfImprovement`` module (e.g. ``ClaudesonSovereign``).
    3. A new standalone ``RecursiveSelfImprovement`` instance sized to
       ``OrchestratorConfig.dim``.

    TemporalPlanner resolution order
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Injected ``planner`` kwarg.
    2. ``TemporalPlanner(model.msp)`` — when the model exposes its
       ``MultiScalePlanner`` sub-module directly.
    3. ``TemporalPlanner(model)`` — full-model fallback; only used when the
       model implements ``predict_action`` or can accept a raw state tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OrchestratorConfig] = None,
        *,
        uncertainty_gate: Optional[UncertaintyGate] = None,
        capability_gate: Optional[CapabilityGate] = None,
        memory: Optional[MemoryManager] = None,
        rsi: Optional[RecursiveSelfImprovement] = None,
        planner: Optional[TemporalPlanner] = None,
        transition: Optional[Callable] = None,
    ) -> None:
        """
        Initialise the orchestrator.

        Args:
            model:            The Claudeson model to wrap (any ``nn.Module``).
            config:           Orchestrator configuration.  Defaults to
                              ``OrchestratorConfig()`` if None.
            uncertainty_gate: Pre-built uncertainty gate.  Created from
                              *config* if None.
            capability_gate:  Pre-built capability gate.  Created fresh if
                              None (no tests registered).
            memory:           Pre-built memory manager.  Created from *config*
                              if None.
            rsi:              Pre-built RSI controller.  See resolution order
                              in the class docstring.
            planner:          Pre-built temporal planner.  See resolution order
                              in the class docstring.
            transition:       Optional ``Callable(state, action) → next_state``
                              passed to a newly created ``TemporalPlanner``.
                              Ignored when *planner* is injected.
        """
        self.model = model
        self._cfg = config or OrchestratorConfig()
        cfg = self._cfg

        # -- Uncertainty gate --------------------------------------------------
        self.uncertainty_gate: UncertaintyGate = (
            uncertainty_gate
            if uncertainty_gate is not None
            else UncertaintyGate(
                threshold=cfg.uncertainty_threshold,
                use_hysteresis=cfg.uncertainty_hysteresis,
                margin=cfg.uncertainty_margin,
            )
        )

        # -- Capability gate ---------------------------------------------------
        self.capability_gate: CapabilityGate = (
            capability_gate if capability_gate is not None else CapabilityGate()
        )

        # -- Memory manager ----------------------------------------------------
        self.memory: MemoryManager = (
            memory if memory is not None else MemoryManager(max_ephemeral=cfg.max_ephemeral)
        )

        # -- Temporal planner --------------------------------------------------
        if planner is not None:
            self.planner: TemporalPlanner = planner
        else:
            inner = getattr(model, "msp", model)
            self.planner = TemporalPlanner(inner, transition=transition)

        # -- RSI controller ----------------------------------------------------
        if rsi is not None:
            self.rsi: Optional[RecursiveSelfImprovement] = rsi
        elif hasattr(model, "rsi"):
            self.rsi = model.rsi
        else:
            _args = SimpleNamespace(
                dim=cfg.dim,
                rsi_rank=cfg.rsi_rank,
                rsi_horizon=cfg.rsi_horizon,
                rsi_threshold=cfg.rsi_threshold,
            )
            self.rsi = RecursiveSelfImprovement(_args)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def register_capability_test(
        self,
        fn: Callable[[nn.Module], bool],
        name: Optional[str] = None,
    ) -> int:
        """
        Register a safety evaluation function with the capability gate.

        The function receives the wrapped model and must return ``True`` to
        pass.  When any registered test fails after an RSI adapter commit,
        the adapter delta is rolled back automatically.

        Args:
            fn:   ``Callable(model) → bool``.
            name: Human-readable label used in log warnings.

        Returns:
            Index of the newly registered test.
        """
        return self.capability_gate.register_test(fn, name=name)

    def clear_memory(
        self,
        *,
        ephemeral: bool = True,
        persistent: bool = False,
    ) -> None:
        """
        Clear one or both memory tiers.

        Args:
            ephemeral:  Clear the ephemeral (working-memory) FIFO tier.
            persistent: Clear the persistent (long-term) tier.
        """
        if ephemeral:
            self.memory.clear_ephemeral()
        if persistent:
            self.memory.clear_persistent()

    def step(
        self,
        inputs: Dict[str, Any],
        *,
        query_embedding: Optional[torch.Tensor] = None,
        efe_baseline: Optional[torch.Tensor] = None,
    ) -> OrchestratorResult:
        """
        Execute one orchestrated inference step.

        The seven-stage pipeline:

        1. **Memory retrieve** — look up the ``memory_top_k`` most relevant
           past entries by cosine similarity to ``query_embedding``.
           Skipped when ``query_embedding`` is None.

        2. **Forward pass** — call ``model(**inputs)`` under
           ``torch.no_grad()``.

        3. **Uncertainty gate** — extract per-sample uncertainty scores and
           compute the abstain fraction.  ``abstained`` is True when more
           than half the batch exceeds the threshold.

        4. **Temporal plan** — extract ``out["temporal"]["plans"]`` from the
           model output.  Falls back to a ``TemporalPlanner.simulate()``
           trajectory when the model output contains no temporal data and the
           model is not abstaining.

        5. **RSI propose** — propose a LoRA adapter delta (skipped when
           abstaining or RSI is disabled).

        6. **Capability gate** — when RSI commits a delta and
           ``capability_check_on_rsi_accept`` is True, run all registered
           tests against the model.  Roll back the adapter if any test fails.

        7. **Memory write** — store the model output as a new ephemeral
           memory entry keyed by ``query_embedding``.  Skipped when
           ``query_embedding`` is None.

        Args:
            inputs:          Keyword arguments forwarded to ``model.forward()``.
            query_embedding: Float tensor ``[D]`` used for memory
                             retrieve and write.  Pass None to skip memory.
            efe_baseline:    Optional EFE scalar forwarded to the RSI
                             controller for improvement estimation.

        Returns:
            ``OrchestratorResult`` containing all step outputs.
        """
        cfg = self._cfg

        # -------------------------------------------------------------------
        # 1. Memory retrieve
        # -------------------------------------------------------------------
        memory_context: List[MemoryEntry] = []
        if query_embedding is not None:
            memory_context = self.memory.retrieve(query_embedding, k=cfg.memory_top_k)
            log.debug(
                "Memory: retrieved %d/%d entries",
                len(memory_context),
                cfg.memory_top_k,
            )

        # -------------------------------------------------------------------
        # 2. Forward pass
        # -------------------------------------------------------------------
        with torch.no_grad():
            model_out: Dict[str, Any] = self.model(**inputs)

        # -------------------------------------------------------------------
        # 3. Uncertainty gate
        # -------------------------------------------------------------------
        unc_scores = self._extract_uncertainty_scores(model_out, inputs)
        gate_out = self.uncertainty_gate(unc_scores)
        abstain_frac = gate_out["abstain_frac"]
        abstained = bool(abstain_frac.item() > 0.5)
        if abstained:
            log.info(
                "Uncertainty gate: abstaining (abstain_frac=%.3f, threshold=%.3f)",
                abstain_frac.item(),
                cfg.uncertainty_threshold,
            )

        # -------------------------------------------------------------------
        # 4. Temporal plan
        # -------------------------------------------------------------------
        plan_out = self._extract_plan(model_out)
        if plan_out is None and not abstained:
            plan_out = self._simulate_plan(model_out, inputs)

        # -------------------------------------------------------------------
        # 5 & 6. RSI propose + capability gate
        # -------------------------------------------------------------------
        rsi_out: Optional[Dict[str, Any]] = None
        capability_ok = True

        if not abstained and self.rsi is not None and cfg.rsi_enabled:
            hidden = self._extract_hidden_state(model_out, inputs, unc_scores)

            # Save adapter state for potential rollback
            saved_A = self.rsi.adapter_A.data.clone()
            saved_B = self.rsi.adapter_B.data.clone()

            _, rsi_out = self.rsi(hidden, efe_baseline)

            if rsi_out["accepted"] and cfg.capability_check_on_rsi_accept:
                capability_ok = self.capability_gate.evaluate(self.model)
                if not capability_ok:
                    self.rsi.adapter_A.data.copy_(saved_A)
                    self.rsi.adapter_B.data.copy_(saved_B)
                    log.warning(
                        "Capability gate blocked RSI commit. Failed tests: %s",
                        self.capability_gate.failed_tests(),
                    )

        # -------------------------------------------------------------------
        # 7. Memory write
        # -------------------------------------------------------------------
        if query_embedding is not None:
            self.memory.write(MemoryEntry(model_out, embedding=query_embedding))

        return OrchestratorResult(
            model_out=model_out,
            abstained=abstained,
            uncertainty_scores=unc_scores,
            capability_ok=capability_ok,
            plan=plan_out,
            rsi_out=rsi_out,
            memory_context=memory_context,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_uncertainty_scores(
        self,
        model_out: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Derive per-sample uncertainty scores ``[B]`` from the model output.

        Priority order:

        1. ``out["uncertainty"]["aggregated"]["confidence"]`` — negated
           (``1 - confidence``) since the gate expects uncertainty not
           confidence.  This is the standard path for ``ClaudesonUncertainty``
           and any descendant.
        2. ``out["uncertainty"]["aggregated"]["unc_embedding"]`` — mean of
           the embedding's absolute values across the feature and sequence
           dimensions.
        3. Zero-filled fallback with batch size inferred from *inputs*.
        """
        agg = model_out.get("uncertainty", {}).get("aggregated", {})

        if "confidence" in agg:
            conf = agg["confidence"]
            if isinstance(conf, torch.Tensor):
                return (1.0 - conf.float()).clamp(0.0, 1.0)

        if "unc_embedding" in agg:
            emb = agg["unc_embedding"]
            if isinstance(emb, torch.Tensor):
                # [B, L, D] → [B] via mean of |values|
                return emb.float().abs().mean(dim=-1).mean(dim=-1)

        B = self._infer_batch_size(inputs)
        return torch.zeros(B)

    def _extract_plan(self, model_out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract the temporal plan dict from the model output.

        Returns ``out["temporal"]["plans"]`` when present (a dict with keys
        ``unified_plan``, ``scale_values``, ``scale_names``), or None.
        """
        temporal = model_out.get("temporal", {})
        plans = temporal.get("plans")
        if plans is None:
            return None
        if isinstance(plans, dict):
            return plans
        # Some models return a raw tensor; wrap it uniformly
        return {"unified_plan": plans}

    def _simulate_plan(
        self,
        model_out: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        Fall back to ``TemporalPlanner.simulate()`` when the model output
        contains no temporal plan data.

        Uses the mean of the first tensor found in *inputs* as the initial
        state.  Returns None if no suitable initial state can be derived.
        """
        initial_state: Optional[torch.Tensor] = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.dtype in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                # Collapse to [B, D] if necessary
                initial_state = v.float().mean(dim=list(range(1, v.dim() - 1)))
                break

        if initial_state is None:
            return None

        try:
            with torch.no_grad():
                return self.planner.simulate(initial_state, steps=self._cfg.plan_steps)
        except Exception:  # noqa: BLE001
            log.debug("TemporalPlanner.simulate() failed; skipping plan.", exc_info=True)
            return None

    def _extract_hidden_state(
        self,
        model_out: Dict[str, Any],
        inputs: Dict[str, Any],
        unc_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Derive a ``[B, 1, D]`` hidden-state tensor for the RSI controller.

        Priority order:

        1. ``out["uncertainty"]["aggregated"]["unc_embedding"]`` — already
           ``[B, L, D]``; used directly.
        2. ``out["temporal"]["plans"]["unified_plan"]`` — ``[B, D]``;
           unsqueezed to ``[B, 1, D]``.
        3. ``unc_scores`` broadcast to ``[B, 1, D]`` as a last resort.
        """
        agg = model_out.get("uncertainty", {}).get("aggregated", {})
        emb = agg.get("unc_embedding")
        if isinstance(emb, torch.Tensor) and emb.dim() == 3:
            return emb

        plan = self._extract_plan(model_out)
        if plan is not None:
            up = plan.get("unified_plan")
            if isinstance(up, torch.Tensor) and up.dim() == 2:
                return up.unsqueeze(1)  # [B, 1, D]

        # Last resort: broadcast unc_scores → [B, 1, D]
        B = unc_scores.shape[0]
        D = self._cfg.dim
        return unc_scores.view(B, 1, 1).expand(B, 1, D).contiguous()

    @staticmethod
    def _infer_batch_size(inputs: Dict[str, Any]) -> int:
        """Return batch size from the first tensor in *inputs*, or 1."""
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        return 1
