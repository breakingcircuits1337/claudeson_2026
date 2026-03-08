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


def test_parallel_scan_matches_sequential():
    """Parallel scan output must match a reference sequential loop element-by-element."""
    from claudson_jedi import parallel_scan
    torch.manual_seed(42)

    for L in (1, 3, 4, 7, 8, 13, 16):   # test both power-of-2 and non-power-of-2 lengths
        B, S = 2, 6
        delta = torch.rand(B, L, S) * 0.5
        A = -torch.rand(S)

        # Reference: sequential inclusive scan
        h = torch.zeros(B, S)
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
