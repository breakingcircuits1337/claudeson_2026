# SPDX-License-Identifier: LicenseRef-Claudeson-Commercial
# Copyright (c) 2026 Breaking Circuits Research.

"""
Tests for claudson_orchestrator.py
====================================
Uses a lightweight ``MockModel`` so no GPU / full model weights are needed.
Each test group exercises one of the five orchestrated components.
"""

import torch
import torch.nn as nn

from claudson_agent_swarm import CapabilityGate, MemoryManager
from claudson_orchestrator import (
    ClaudesonOrchestrator,
    OrchestratorConfig,
    OrchestratorResult,
)
from claudson_uncertainty import UncertaintyGate

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DIM = 64
BATCH = 2
SEQ = 4


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Minimal model that returns a configurable output dict."""

    def __init__(self, out: dict):
        super().__init__()
        self._out = out
        # Give the model one parameter so CapabilityGate tests can inspect it
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        return self._out


def _low_uncertainty_out():
    """Output dict that signals low uncertainty (confidence ≈ 1)."""
    return {
        "uncertainty": {
            "aggregated": {
                "confidence": torch.ones(BATCH),
                "unc_embedding": torch.zeros(BATCH, SEQ, DIM),
            }
        }
    }


def _high_uncertainty_out():
    """Output dict that signals high uncertainty (confidence ≈ 0)."""
    return {
        "uncertainty": {
            "aggregated": {
                "confidence": torch.zeros(BATCH),
                "unc_embedding": torch.ones(BATCH, SEQ, DIM),
            }
        }
    }


def _temporal_out():
    """Output dict that includes temporal plan data."""
    return {
        "uncertainty": {
            "aggregated": {
                "confidence": torch.ones(BATCH),
                "unc_embedding": torch.zeros(BATCH, SEQ, DIM),
            }
        },
        "temporal": {
            "plans": {
                "unified_plan": torch.randn(BATCH, DIM),
                "scale_values": [[0.1, 0.2, 0.3, 0.4]] * BATCH,
                "scale_names": ["operational", "tactical", "strategic", "civilisational"],
            }
        },
    }


def _make_cfg(**kwargs) -> OrchestratorConfig:
    return OrchestratorConfig(dim=DIM, **kwargs)


def _make_inputs() -> dict:
    return {"text": torch.randint(0, 100, (BATCH, SEQ))}


# ---------------------------------------------------------------------------
# 1. Smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_instantiation_defaults(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        assert isinstance(orch.uncertainty_gate, UncertaintyGate)
        assert isinstance(orch.capability_gate, CapabilityGate)
        assert isinstance(orch.memory, MemoryManager)

    def test_step_returns_result(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert isinstance(result, OrchestratorResult)

    def test_result_fields_present(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert isinstance(result.model_out, dict)
        assert isinstance(result.abstained, bool)
        assert isinstance(result.uncertainty_scores, torch.Tensor)
        assert isinstance(result.capability_ok, bool)
        assert isinstance(result.memory_context, list)

    def test_injected_components_are_used(self):
        model = MockModel(_low_uncertainty_out())
        gate = UncertaintyGate(threshold=0.9)
        mem = MemoryManager(max_ephemeral=10)
        cap = CapabilityGate()
        orch = ClaudesonOrchestrator(
            model, _make_cfg(), uncertainty_gate=gate, memory=mem, capability_gate=cap
        )
        assert orch.uncertainty_gate is gate
        assert orch.memory is mem
        assert orch.capability_gate is cap


# ---------------------------------------------------------------------------
# 2. Uncertainty gate
# ---------------------------------------------------------------------------


class TestUncertaintyGate:
    def test_low_uncertainty_does_not_abstain(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=0.4))
        result = orch.step(_make_inputs())
        assert not result.abstained

    def test_high_uncertainty_abstains(self):
        model = MockModel(_high_uncertainty_out())
        # uncertainty = 1 - 0 = 1.0; well above threshold
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=0.4))
        result = orch.step(_make_inputs())
        assert result.abstained

    def test_uncertainty_scores_shape(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert result.uncertainty_scores.shape == (BATCH,)

    def test_fallback_scores_when_no_uncertainty_key(self):
        """Model returns no uncertainty dict → scores default to zeros."""
        model = MockModel({})
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert (result.uncertainty_scores == 0.0).all()
        assert not result.abstained

    def test_unc_embedding_fallback_path(self):
        """Aggregated dict has unc_embedding but no confidence key."""
        out = {
            "uncertainty": {
                "aggregated": {
                    "unc_embedding": torch.full((BATCH, SEQ, DIM), 0.1),
                }
            }
        }
        model = MockModel(out)
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=0.5))
        result = orch.step(_make_inputs())
        assert result.uncertainty_scores.shape == (BATCH,)
        # mean(|0.1|) = 0.1 < 0.5 threshold → should not abstain
        assert not result.abstained

    def test_custom_threshold(self):
        model = MockModel(_high_uncertainty_out())
        # threshold = 2.0 → nothing abstains
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=2.0))
        result = orch.step(_make_inputs())
        assert not result.abstained


# ---------------------------------------------------------------------------
# 3. Memory manager
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def test_no_query_embedding_skips_memory(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs(), query_embedding=None)
        assert result.memory_context == []
        assert len(orch.memory._ephemeral) == 0

    def test_query_embedding_triggers_write(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        q = torch.randn(DIM)
        orch.step(_make_inputs(), query_embedding=q)
        assert len(orch.memory._ephemeral) == 1

    def test_retrieval_returns_previous_entry(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(memory_top_k=1))
        q = torch.randn(DIM)
        orch.step(_make_inputs(), query_embedding=q)
        # Second step should retrieve the entry written by the first
        result = orch.step(_make_inputs(), query_embedding=q)
        assert len(result.memory_context) == 1

    def test_ephemeral_eviction_at_limit(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(max_ephemeral=2))
        q = torch.randn(DIM)
        for _ in range(4):
            orch.step(_make_inputs(), query_embedding=q)
        assert len(orch.memory._ephemeral) <= 2

    def test_clear_memory_ephemeral(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        q = torch.randn(DIM)
        orch.step(_make_inputs(), query_embedding=q)
        assert len(orch.memory._ephemeral) == 1
        orch.clear_memory(ephemeral=True)
        assert len(orch.memory._ephemeral) == 0

    def test_shared_memory_across_orchestrators(self):
        shared = MemoryManager(max_ephemeral=256)
        model_a = MockModel(_low_uncertainty_out())
        model_b = MockModel(_low_uncertainty_out())
        orch_a = ClaudesonOrchestrator(model_a, _make_cfg(), memory=shared)
        orch_b = ClaudesonOrchestrator(model_b, _make_cfg(), memory=shared)
        q = torch.randn(DIM)
        orch_a.step(_make_inputs(), query_embedding=q)
        result = orch_b.step(_make_inputs(), query_embedding=q)
        # orch_b retrieves what orch_a wrote
        assert len(result.memory_context) >= 1


# ---------------------------------------------------------------------------
# 4. Temporal planner
# ---------------------------------------------------------------------------


class TestTemporalPlanner:
    def test_plan_extracted_from_model_output(self):
        model = MockModel(_temporal_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert result.plan is not None
        assert "unified_plan" in result.plan

    def test_plan_is_none_when_no_temporal_output(self):
        """Model with no temporal data and no float inputs → plan is None."""
        # Integer-only inputs → _simulate_plan cannot find a float tensor
        model = MockModel({})
        orch = ClaudesonOrchestrator(model, _make_cfg(rsi_enabled=False))
        result = orch.step({"text": torch.randint(0, 100, (BATCH, SEQ))})
        assert result.plan is None

    def test_plan_is_dict_with_unified_plan(self):
        model = MockModel(_temporal_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        result = orch.step(_make_inputs())
        assert isinstance(result.plan, dict)
        assert isinstance(result.plan["unified_plan"], torch.Tensor)
        assert result.plan["unified_plan"].shape == (BATCH, DIM)

    def test_plan_skipped_when_abstaining(self):
        """High uncertainty → abstain → fallback simulate not called."""
        model = MockModel(_high_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=0.1))
        result = orch.step(_make_inputs())
        assert result.abstained
        # No temporal data in model output and abstaining → plan is None
        assert result.plan is None


# ---------------------------------------------------------------------------
# 5. RSI controller
# ---------------------------------------------------------------------------


class TestRSIController:
    def test_rsi_out_present_when_not_abstaining(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(rsi_enabled=True))
        result = orch.step(_make_inputs())
        assert result.rsi_out is not None
        assert "accepted" in result.rsi_out
        assert "improvement" in result.rsi_out

    def test_rsi_skipped_when_disabled(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(rsi_enabled=False))
        result = orch.step(_make_inputs())
        assert result.rsi_out is None

    def test_rsi_skipped_when_abstaining(self):
        model = MockModel(_high_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(uncertainty_threshold=0.1))
        result = orch.step(_make_inputs())
        assert result.abstained
        assert result.rsi_out is None

    def test_rsi_uses_model_rsi_submodule_when_present(self):
        """If the wrapped model exposes .rsi, the orchestrator reuses it."""
        from types import SimpleNamespace

        from claudson_sovereign import RecursiveSelfImprovement

        model = MockModel(_low_uncertainty_out())
        rsi_args = SimpleNamespace(dim=DIM, rsi_rank=2, rsi_horizon=1, rsi_threshold=0.05)
        model.rsi = RecursiveSelfImprovement(rsi_args)
        orch = ClaudesonOrchestrator(model, _make_cfg())
        assert orch.rsi is model.rsi


# ---------------------------------------------------------------------------
# 6. Capability gate
# ---------------------------------------------------------------------------


class TestCapabilityGate:
    def test_capability_ok_true_when_no_tests(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg(rsi_enabled=True))
        result = orch.step(_make_inputs())
        assert result.capability_ok

    def test_capability_gate_blocks_rsi_on_failure(self):
        """Failing capability test → RSI adapter rolled back."""
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(
            model,
            _make_cfg(
                rsi_enabled=True,
                rsi_threshold=-1.0,  # accept all proposals
                capability_check_on_rsi_accept=True,
            ),
        )
        # Register a test that always fails
        orch.register_capability_test(lambda m: False, name="always_fail")

        # Save adapter state before step
        saved_A = orch.rsi.adapter_A.data.clone()
        saved_B = orch.rsi.adapter_B.data.clone()

        result = orch.step(_make_inputs())

        # Capability gate should have rolled back the adapter
        if result.rsi_out is not None and result.rsi_out["accepted"]:
            assert not result.capability_ok
            assert torch.allclose(orch.rsi.adapter_A.data, saved_A)
            assert torch.allclose(orch.rsi.adapter_B.data, saved_B)

    def test_capability_gate_passes_when_all_tests_pass(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(
            model,
            _make_cfg(rsi_enabled=True, rsi_threshold=-1.0),
        )
        orch.register_capability_test(lambda m: True, name="always_pass")
        result = orch.step(_make_inputs())
        assert result.capability_ok

    def test_register_capability_test_returns_index(self):
        model = MockModel(_low_uncertainty_out())
        orch = ClaudesonOrchestrator(model, _make_cfg())
        idx0 = orch.register_capability_test(lambda m: True, name="t0")
        idx1 = orch.register_capability_test(lambda m: True, name="t1")
        assert idx0 == 0
        assert idx1 == 1

    def test_capability_check_skipped_when_rsi_does_not_accept(self):
        """When RSI rejects its proposal, capability gate is not called."""
        model = MockModel(_low_uncertainty_out())
        call_count = {"n": 0}

        def counting_test(m):
            call_count["n"] += 1
            return True

        orch = ClaudesonOrchestrator(
            model,
            _make_cfg(
                rsi_enabled=True,
                rsi_threshold=1e9,  # threshold so high RSI never accepts
            ),
        )
        orch.register_capability_test(counting_test, name="counter")
        result = orch.step(_make_inputs())
        if result.rsi_out is not None and not result.rsi_out["accepted"]:
            assert call_count["n"] == 0
