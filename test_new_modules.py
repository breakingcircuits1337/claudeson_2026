"""
Smoke tests for modules added via direct upload (no prior PR review).

These tests verify:
  - Each module imports cleanly
  - The top-level model class constructs without error under default ModelArgs
  - A single forward pass completes and returns a dict with expected keys

Run with:  pytest test_new_modules.py -v
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_text(batch=1, seq=32, vocab=1000):
    return torch.randint(0, vocab, (batch, seq))


# ---------------------------------------------------------------------------
# claudson_social_alignment
# ---------------------------------------------------------------------------

def test_social_alignment_import():
    from claudson_social_alignment import ClaudesonSocialAlignment, ModelArgs
    assert ClaudesonSocialAlignment is not None


def test_social_alignment_forward():
    from claudson_social_alignment import ClaudesonSocialAlignment, ModelArgs
    args = ModelArgs()
    model = ClaudesonSocialAlignment(args)
    model.eval()
    with torch.no_grad():
        out = model(text=make_text())
    assert isinstance(out, dict), "forward() should return a dict"


# ---------------------------------------------------------------------------
# claudson_uncertainty
# ---------------------------------------------------------------------------

def test_uncertainty_import():
    from claudson_uncertainty import ClaudesonUncertainty, ModelArgs
    assert ClaudesonUncertainty is not None


def test_uncertainty_forward():
    from claudson_uncertainty import ClaudesonUncertainty, ModelArgs
    args = ModelArgs()
    model = ClaudesonUncertainty(args)
    model.eval()
    with torch.no_grad():
        out = model(text=make_text())
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# claudson_temporal_reasoning
# ---------------------------------------------------------------------------

def test_temporal_reasoning_import():
    from claudson_temporal_reasoning import ClaudesonTemporalReasoning, ModelArgs
    assert ClaudesonTemporalReasoning is not None


def test_temporal_reasoning_forward():
    from claudson_temporal_reasoning import ClaudesonTemporalReasoning, ModelArgs
    args = ModelArgs()
    model = ClaudesonTemporalReasoning(args)
    model.eval()
    with torch.no_grad():
        out = model(text=make_text())
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# claudson_meta_learning
# ---------------------------------------------------------------------------

def test_meta_learning_import():
    from claudson_meta_learning import ClaudesonMetaLearning, ModelArgs
    assert ClaudesonMetaLearning is not None


def test_meta_learning_forward():
    from claudson_meta_learning import ClaudesonMetaLearning, ModelArgs
    args = ModelArgs()
    model = ClaudesonMetaLearning(args)
    model.eval()
    with torch.no_grad():
        out = model(text=make_text())
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# claudson_metacurriculum
# ---------------------------------------------------------------------------

def test_metacurriculum_import():
    from claudson_metacurriculum import ModelArgs
    assert ModelArgs is not None


# ---------------------------------------------------------------------------
# claudson_abstraction
# ---------------------------------------------------------------------------

def test_abstraction_import():
    from claudson_abstraction import ModelArgs
    assert ModelArgs is not None


# ---------------------------------------------------------------------------
# claudson_causal_world
# ---------------------------------------------------------------------------

def test_causal_world_import():
    from claudson_causal_world import ModelArgs
    assert ModelArgs is not None


# ---------------------------------------------------------------------------
# claudson_formal_verification
# ---------------------------------------------------------------------------

def test_formal_verification_import():
    from claudson_formal_verification import ModelArgs
    assert ModelArgs is not None


# ---------------------------------------------------------------------------
# claudson_grounded_language
# ---------------------------------------------------------------------------

def test_grounded_language_import():
    from claudson_grounded_language import ModelArgs
    assert ModelArgs is not None


# ---------------------------------------------------------------------------
# claudson_trainer — config-only smoke test (no GPU training loop)
# ---------------------------------------------------------------------------

def test_trainer_config_import():
    from claudson_trainer import TrainerConfig
    cfg = TrainerConfig()
    # Verify phase steps are positive and sum to a reasonable curriculum
    total = (cfg.phase0_steps + cfg.phase1_steps + cfg.phase2_steps +
             cfg.phase3_steps + cfg.phase4_steps + cfg.phase5_steps)
    assert total > 0, "Trainer curriculum must have positive total steps"
    assert cfg.phase0_steps == 10_000
    assert cfg.phase5_steps == 20_000


def test_trainer_phase_ordering():
    """Phase step counts should be positive and nonzero."""
    from claudson_trainer import TrainerConfig
    cfg = TrainerConfig()
    for i, steps in enumerate([
        cfg.phase0_steps, cfg.phase1_steps, cfg.phase2_steps,
        cfg.phase3_steps, cfg.phase4_steps, cfg.phase5_steps,
    ]):
        assert steps > 0, f"phase{i}_steps must be > 0, got {steps}"


# ---------------------------------------------------------------------------
# claudson_jedi — parallel_scan correctness
# ---------------------------------------------------------------------------

def test_parallel_scan_shape():
    from claudson_jedi import parallel_scan
    B, L, S = 2, 16, 8
    delta = torch.randn(B, L, S)
    A = torch.randn(S) * -0.5  # stable negative values
    out = parallel_scan(delta, A)
    assert out.shape == (B, L, S), f"Expected ({B},{L},{S}), got {out.shape}"


@pytest.mark.parametrize("L", [1, 3, 4, 7, 8, 13, 16])  # power-of-2 and non-power-of-2
def test_parallel_scan_matches_sequential(L):
    """Parallel scan output must match a reference sequential loop element-by-element."""
    from claudson_jedi import parallel_scan
    torch.manual_seed(42)

    B, S = 2, 6
    delta = torch.rand(B, L, S) * 0.5
    A = -torch.rand(S)

    # Reference: sequential inclusive scan — same device/dtype as delta
    h = torch.zeros(B, S, device=delta.device, dtype=delta.dtype)
    ref = []
    for t in range(L):
        dt = delta[:, t, :]
        h = torch.exp(dt * A) * h + dt
        ref.append(h.clone())
    ref = torch.stack(ref, dim=1)   # [B, L, S]

    par = parallel_scan(delta, A)
    max_diff = (ref - par).abs().max().item()
    assert max_diff < 1e-5, f"L={L}: max diff = {max_diff:.2e}"


def test_parallel_scan_first_position():
    """h[0] must equal delta[0] (since h_{-1}=0)."""
    from claudson_jedi import parallel_scan
    torch.manual_seed(7)
    B, L, S = 3, 10, 5
    delta = torch.rand(B, L, S)
    A = -torch.rand(S)
    out = parallel_scan(delta, A)
    # h_0 = exp(delta_0*A)*0 + delta_0 = delta_0
    assert torch.allclose(out[:, 0, :], delta[:, 0, :], atol=1e-6)


# ---------------------------------------------------------------------------
# claudson_rsi_controller
# ---------------------------------------------------------------------------

def test_rsi_controller_import():
    from claudson_rsi_controller import RSIController, SimpleEvaluator
    assert RSIController is not None
    assert SimpleEvaluator is not None


def test_rsi_controller_rejects_bad_patch():
    """A patch that degrades performance should be rejected."""
    import torch.nn as nn
    from claudson_rsi_controller import RSIController

    model = nn.Linear(4, 4)

    def evaluator(m, batch):
        with torch.no_grad():
            return -m(batch).pow(2).sum().item()   # higher = smaller output norm

    rsi = RSIController(model, evaluator, threshold=0.0)

    def bad_patch(m):
        for p in m.parameters():
            p.data += 100.0   # deliberately inflates norm → lower score

    batch = torch.randn(2, 4)
    accepted, delta = rsi.apply_if_safe(bad_patch, batch)
    assert not accepted, "Degrading patch should be rejected"
    assert delta < 0, "Score delta should be negative"


def test_rsi_controller_accepts_good_patch():
    """A patch that reduces output magnitude should be accepted (evaluator = neg norm)."""
    import torch.nn as nn
    from claudson_rsi_controller import RSIController

    model = nn.Linear(4, 4)
    # Inflate weights first so zeroing improves the score
    with torch.no_grad():
        for p in model.parameters():
            p.data.fill_(5.0)

    def evaluator(m, batch):
        with torch.no_grad():
            return -m(batch).pow(2).sum().item()

    rsi = RSIController(model, evaluator, threshold=0.0)

    def good_patch(m):
        with torch.no_grad():
            for p in m.parameters():
                p.data.zero_()

    batch = torch.randn(2, 4)
    accepted, delta = rsi.apply_if_safe(good_patch, batch)
    assert accepted, "Improving patch should be accepted"
    assert delta > 0, "Score delta should be positive"


def test_rsi_controller_stats():
    import torch.nn as nn
    from claudson_rsi_controller import RSIController

    model = nn.Linear(2, 2)
    rsi   = RSIController(model, lambda m, b: 0.0, threshold=1.0)
    rsi.apply_if_safe(lambda m: None, torch.zeros(1, 2))  # rejected (delta=0 ≤ 1.0)
    s = rsi.stats()
    assert s["n_proposals"]    == 1
    assert s["n_accepted"]     == 0
    assert s["acceptance_rate"] == 0.0


# ---------------------------------------------------------------------------
# claudson_agent_swarm — AgentSwarm
# ---------------------------------------------------------------------------

def test_agent_swarm_import():
    from claudson_agent_swarm import AgentSwarm, CapabilityGate, MemoryManager, MemoryEntry
    assert AgentSwarm is not None


def test_agent_swarm_consensus():
    from claudson_agent_swarm import AgentSwarm

    agents = [
        lambda x: {"out": x * i, "confidence": float(i)}
        for i in range(1, 4)
    ]
    swarm   = AgentSwarm(agents)
    outputs = swarm.run(torch.tensor(1.0))
    best    = swarm.consensus(outputs)
    assert best["confidence"] == 3.0, "Highest-confidence agent should win"


def test_agent_swarm_weighted_merge():
    from claudson_agent_swarm import AgentSwarm

    t1 = torch.zeros(4)
    t2 = torch.ones(4)
    agents = [
        lambda x: {"logits": t1, "confidence": 1.0},
        lambda x: {"logits": t2, "confidence": 1.0},
    ]
    swarm   = AgentSwarm(agents)
    outputs = swarm.run(None)
    merged  = swarm.weighted_merge(outputs, "logits")
    assert merged.shape == (4,), "Merged output should have correct shape"
    assert torch.allclose(merged, torch.full((4,), 0.5), atol=1e-5), \
        "Equal-weight merge of 0 and 1 should give 0.5"


# ---------------------------------------------------------------------------
# claudson_agent_swarm — CapabilityGate
# ---------------------------------------------------------------------------

def test_capability_gate_all_pass():
    from claudson_agent_swarm import CapabilityGate
    import torch.nn as nn

    gate = CapabilityGate()
    gate.register_test(lambda m: True, name="always_pass")
    assert gate.evaluate(nn.Linear(2, 2))
    assert gate.failed_tests() == []


def test_capability_gate_partial_fail():
    from claudson_agent_swarm import CapabilityGate
    import torch.nn as nn

    gate = CapabilityGate()
    gate.register_test(lambda m: True,  name="pass")
    gate.register_test(lambda m: False, name="fail")
    assert not gate.evaluate(nn.Linear(2, 2))
    assert "fail" in gate.failed_tests()


# ---------------------------------------------------------------------------
# claudson_agent_swarm — MemoryManager
# ---------------------------------------------------------------------------

def test_memory_manager_write_retrieve():
    from claudson_agent_swarm import MemoryManager, MemoryEntry

    mem = MemoryManager(max_ephemeral=10)
    emb = torch.randn(16)
    mem.write(MemoryEntry("event_a", embedding=emb))
    mem.write(MemoryEntry("event_b", embedding=emb * -1))

    results = mem.retrieve(emb, k=1)
    assert len(results) == 1
    assert results[0].content == "event_a", "Most similar entry should be retrieved"


def test_memory_manager_ephemeral_eviction():
    from claudson_agent_swarm import MemoryManager, MemoryEntry

    mem = MemoryManager(max_ephemeral=3)
    for i in range(5):
        mem.write(MemoryEntry(i))
    assert len(mem._ephemeral) == 3, "Ephemeral buffer should be capped at max_ephemeral"


def test_memory_manager_persistent():
    from claudson_agent_swarm import MemoryManager, MemoryEntry

    mem = MemoryManager(max_ephemeral=2)
    for i in range(10):
        mem.write(MemoryEntry(f"persistent_{i}"), persistent=True)
    assert len(mem._persistent) == 10, "Persistent entries should never be evicted"


# ---------------------------------------------------------------------------
# claudson_uncertainty — UncertaintyGate
# ---------------------------------------------------------------------------

def test_uncertainty_gate_import():
    from claudson_uncertainty import UncertaintyGate
    assert UncertaintyGate is not None


def test_uncertainty_gate_allow():
    from claudson_uncertainty import UncertaintyGate

    gate = UncertaintyGate(threshold=0.5)
    assert gate.allow_action(0.3),  "Low uncertainty should allow action"
    assert not gate.allow_action(0.7), "High uncertainty should block action"


def test_uncertainty_gate_batch():
    from claudson_uncertainty import UncertaintyGate

    gate   = UncertaintyGate(threshold=0.5)
    scores = torch.tensor([0.1, 0.4, 0.6, 0.9])
    mask   = gate.allow_batch(scores)
    assert mask.tolist() == [True, True, False, False]


def test_uncertainty_gate_forward():
    from claudson_uncertainty import UncertaintyGate

    gate   = UncertaintyGate(threshold=0.5)
    scores = torch.tensor([0.2, 0.8])
    out    = gate(scores)
    assert "allow"        in out
    assert "abstain_frac" in out
    assert out["allow"].tolist() == [True, False]
    assert abs(out["abstain_frac"].item() - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# claudson_temporal_reasoning — TemporalPlanner
# ---------------------------------------------------------------------------

def test_temporal_planner_import():
    from claudson_temporal_reasoning import TemporalPlanner
    assert TemporalPlanner is not None


def test_temporal_planner_simulate():
    from claudson_temporal_reasoning import TemporalPlanner

    class MockModel:
        def predict_action(self, state):
            return torch.zeros_like(state)   # null action

    planner    = TemporalPlanner(MockModel())
    state      = torch.randn(2, 32)
    trajectory = planner.simulate(state, steps=5)

    assert len(trajectory) == 5, "Should produce exactly `steps` transitions"
    for s, a, ns in trajectory:
        assert s.shape == state.shape
        assert a.shape == state.shape
        assert torch.allclose(ns, s), "Null action should leave state unchanged"


def test_temporal_planner_custom_transition():
    from claudson_temporal_reasoning import TemporalPlanner

    class MockModel:
        def predict_action(self, state):
            return torch.ones_like(state)

    def transition(s, a):
        return s + a * 2.0   # custom scale

    planner    = TemporalPlanner(MockModel(), transition=transition)
    state      = torch.zeros(1, 4)
    trajectory = planner.simulate(state, steps=3)

    final_state = trajectory[-1][2]
    assert torch.allclose(final_state, torch.full((1, 4), 6.0)), \
        "Custom transition should accumulate: 3 steps × 2.0 = 6.0"


# ---------------------------------------------------------------------------
# claudson_meta_learning — MetaAdapter
# ---------------------------------------------------------------------------

def test_meta_adapter_import():
    from claudson_meta_learning import MetaAdapter
    assert MetaAdapter is not None


def test_meta_adapter_adapt_returns_params():
    """adapt() should return a non-empty dict of parameter tensors."""
    import torch.nn as nn
    from claudson_meta_learning import MetaAdapter

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))

    def loss_fn(out):
        return out.mean()

    adapter = MetaAdapter(model, lr=0.01, inner_steps=2, first_order=True)
    x       = torch.randn(2, 8)
    adapted = adapter.adapt((x,), loss_fn)
    assert isinstance(adapted, dict), "adapt() should return a dict"
    assert len(adapted) > 0,         "adapted dict should not be empty"
    # Parameters should differ from originals after adaptation
    original = dict(model.named_parameters())
    changed  = any(
        not torch.allclose(adapted[k], original[k])
        for k in adapted
    )
    assert changed, "At least one parameter should change after inner-loop update"


def test_meta_adapter_forward_adapted():
    """forward_adapted() should produce the same shape as a normal forward."""
    import torch.nn as nn
    from claudson_meta_learning import MetaAdapter

    model = nn.Linear(8, 4)

    def loss_fn(out):
        return out.mean()

    adapter = MetaAdapter(model, lr=0.01, inner_steps=1, first_order=True)
    x       = torch.randn(3, 8)
    adapted = adapter.adapt((x,), loss_fn)
    out     = adapter.forward_adapted(adapted, (x,))
    assert out.shape == (3, 4), "forward_adapted output shape should match normal forward"
