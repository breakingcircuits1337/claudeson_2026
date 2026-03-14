"""Shape and head-regression tests for Claudeson 2026.

Covers:
  - ModelArgs field presence and the inheritance chain (G1 → G9)
  - GQA divisibility contract  (n_heads % n_kv_heads == 0)
  - GQA projection dimensions  (K/V use n_kv_heads, Q uses n_heads)
  - head_dim consistency across all attention modules
  - Forward-pass output tensor shapes for G1 and G6
  - Required output-dict keys for G1 and G6

These tests use deliberately small model configs so they run quickly on CPU.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_small_args(ModelArgs):
    """Return a ModelArgs tuned for fast, low-memory smoke tests."""
    args = ModelArgs()
    args.dim = 64
    args.n_layers = 2
    args.n_heads = 4
    args.n_kv_heads = 2  # GQA: 2 KV heads for 4 query heads
    args.vocab_size = 512
    args.max_seq_len = 64
    args.memory_slots = 32
    args.episodic_slots = 64
    args.goal_dim = 64
    args.latent_dim = 32
    args.energy_hidden = 64
    args.ssm_state_dim = 16
    args.ssm_chunk_size = 8
    args.num_experts = 2
    args.num_shared_experts = 1
    args.env_state_dim = 32
    args.img_size = 32
    args.patch_size = 8
    args.audio_spec_dim = 16
    args.action_space_size = 8
    return args


def make_text(batch: int = 1, seq: int = 16, vocab: int = 512) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq))


# ---------------------------------------------------------------------------
# ModelArgs field presence & inheritance chain
# ---------------------------------------------------------------------------


class TestModelArgsContracts:
    """Every generation's ModelArgs must carry the base fields."""

    G1_REQUIRED = (
        "dim",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "max_seq_len",
        "num_experts",
        "expert_top_k",
        "gradient_checkpointing",
        "mixed_precision",
        "use_kv_cache",
    )

    def test_g1_required_fields(self):
        from claudson import ModelArgs

        args = ModelArgs()
        for field in self.G1_REQUIRED:
            assert hasattr(args, field), f"G1 ModelArgs missing '{field}'"

    def test_g6_carries_base_fields(self):
        """G6 (Jedi) must expose every field declared in G1_REQUIRED."""
        from claudson_jedi import ModelArgs

        args = ModelArgs()
        for field in self.G1_REQUIRED:
            assert hasattr(args, field), f"G6 ModelArgs missing '{field}'"

    def test_g6_jedi_specific_fields(self):
        from claudson_jedi import ModelArgs

        args = ModelArgs()
        for field in (
            "latent_dim",
            "energy_hidden",
            "goal_horizon",
            "ssm_state_dim",
            "ssm_chunk_size",
            "use_jedi",
        ):
            assert hasattr(args, field), f"G6 ModelArgs missing Jedi field '{field}'"

    def test_g9_inherits_sovereign_fields(self):
        """G9 ModelArgs subclasses SovereignArgs (G8); G8 fields must be present."""
        from claudson_transcendent import ModelArgs

        args = ModelArgs()
        # G9-specific
        for field in (
            "n_workspace_slots",
            "gw_competition_k",
            "gw_broadcast_steps",
            "n_ops",
            "n_registers",
            "lif_threshold",
            "lif_leak",
            "lif_steps",
        ):
            assert hasattr(args, field), f"G9 ModelArgs missing '{field}'"

    # ---- GQA divisibility ---------------------------------------------------

    def test_gqa_divisibility_g1_defaults(self):
        from claudson import ModelArgs

        a = ModelArgs()
        assert a.n_heads % a.n_kv_heads == 0, (
            f"G1 default GQA: n_heads={a.n_heads} not divisible by n_kv_heads={a.n_kv_heads}"
        )

    def test_gqa_divisibility_g6_defaults(self):
        from claudson_jedi import ModelArgs

        a = ModelArgs()
        assert a.n_heads % a.n_kv_heads == 0

    def test_gqa_divisibility_small_args(self):
        from claudson import ModelArgs

        args = make_small_args(ModelArgs)
        assert args.n_heads % args.n_kv_heads == 0

    def test_head_dim_is_integer_g1(self):
        from claudson import ModelArgs

        a = ModelArgs()
        assert a.dim % a.n_heads == 0, f"G1: dim={a.dim} not divisible by n_heads={a.n_heads}"

    def test_head_dim_is_integer_g6(self):
        from claudson_jedi import ModelArgs

        a = ModelArgs()
        assert a.dim % a.n_heads == 0


# ---------------------------------------------------------------------------
# GQA projection dimension contracts
# ---------------------------------------------------------------------------


class TestGQAProjectionShapes:
    """K/V projections must use n_kv_heads; Q projection must use n_heads."""

    @staticmethod
    def _first_attn(model):
        """Return the first attention module that exposes k_proj + n_kv_heads."""
        for m in model.modules():
            if hasattr(m, "k_proj") and hasattr(m, "n_kv_heads"):
                return m
        return None

    def test_g1_kv_proj_output_dim(self):
        from claudson import ModelArgs, UniversalIntelligenceModel

        args = make_small_args(ModelArgs)
        model = UniversalIntelligenceModel(args)
        attn = self._first_attn(model)
        assert attn is not None, "No GQA attention module found in G1"

        head_dim = args.dim // args.n_heads
        expected = args.n_kv_heads * head_dim
        assert attn.k_proj.out_features == expected, (
            f"k_proj: expected {expected}, got {attn.k_proj.out_features}"
        )
        assert attn.v_proj.out_features == expected, (
            f"v_proj: expected {expected}, got {attn.v_proj.out_features}"
        )

    def test_g1_q_proj_output_dim(self):
        from claudson import ModelArgs, UniversalIntelligenceModel

        args = make_small_args(ModelArgs)
        attn = self._first_attn(UniversalIntelligenceModel(args))
        assert attn is not None

        expected = args.n_heads * (args.dim // args.n_heads)
        assert attn.q_proj.out_features == expected, (
            f"q_proj: expected {expected}, got {attn.q_proj.out_features}"
        )

    def test_g1_head_dim_attribute(self):
        from claudson import ModelArgs, UniversalIntelligenceModel

        args = make_small_args(ModelArgs)
        attn = self._first_attn(UniversalIntelligenceModel(args))
        assert attn is not None
        assert attn.head_dim == args.dim // args.n_heads

    def test_g1_n_kv_heads_attribute_matches_args(self):
        from claudson import ModelArgs, UniversalIntelligenceModel

        args = make_small_args(ModelArgs)
        model = UniversalIntelligenceModel(args)
        attn_mods = [m for m in model.modules() if hasattr(m, "n_kv_heads")]
        assert attn_mods, "No module exposes n_kv_heads in G1"
        for m in attn_mods:
            assert m.n_kv_heads == args.n_kv_heads, (
                f"module n_kv_heads={m.n_kv_heads} != args.n_kv_heads={args.n_kv_heads}"
            )


# ---------------------------------------------------------------------------
# Fixtures — built once per module, shared across TestForwardOutputShapes
# ---------------------------------------------------------------------------

SEQ_LEN = 16
BATCH = 2


@pytest.fixture(scope="module")
def g1_model_and_args():
    from claudson import ModelArgs, UniversalIntelligenceModel

    args = make_small_args(ModelArgs)
    model = UniversalIntelligenceModel(args)
    model.eval()
    return model, args


@pytest.fixture(scope="module")
def g6_model_and_args():
    from claudson_jedi import ClaudesonJedi, ModelArgs

    args = make_small_args(ModelArgs)
    model = ClaudesonJedi(args)
    model.eval()
    return model, args


@pytest.fixture(scope="module")
def g1_out(g1_model_and_args):
    model, args = g1_model_and_args
    with torch.no_grad():
        return model(text=make_text(BATCH, SEQ_LEN, args.vocab_size)), args


@pytest.fixture(scope="module")
def g6_out(g6_model_and_args):
    model, args = g6_model_and_args
    with torch.no_grad():
        return model(text=make_text(BATCH, SEQ_LEN, args.vocab_size)), args


# ---------------------------------------------------------------------------
# Forward-pass output shape regression
# ---------------------------------------------------------------------------


class TestForwardOutputShapes:
    """Output tensor shapes must stay stable across refactors."""

    # ---- G1 (Foundation) ---------------------------------------------------

    def test_g1_logits_shape(self, g1_out):
        out, args = g1_out
        assert "logits" in out
        assert out["logits"].shape == (BATCH, SEQ_LEN, args.vocab_size), (
            f"logits: expected {(BATCH, SEQ_LEN, args.vocab_size)}, got {out['logits'].shape}"
        )

    def test_g1_hidden_states_shape(self, g1_out):
        out, args = g1_out
        hs = out["hidden_states"]
        assert hs.shape == (BATCH, SEQ_LEN, args.dim), (
            f"hidden_states: expected {(BATCH, SEQ_LEN, args.dim)}, got {hs.shape}"
        )

    def test_g1_required_output_keys(self, g1_out):
        out, _ = g1_out
        for key in ("logits", "hidden_states", "alignment", "confidence", "uncertainty"):
            assert key in out, f"G1 forward() missing key '{key}'"

    def test_g1_alignment_shape(self, g1_out):
        """alignment must be [..., 3] — helpful / harmless / honest."""
        out, _ = g1_out
        alignment = out["alignment"]
        assert alignment.shape[0] == BATCH, "alignment batch dim mismatch"
        assert alignment.shape[-1] == 3, (
            f"alignment last dim: expected 3, got {alignment.shape[-1]}"
        )

    def test_g1_confidence_and_uncertainty_shape(self, g1_out):
        out, _ = g1_out
        assert out["confidence"].shape[-1] == 1, "confidence last dim must be 1"
        assert out["uncertainty"].shape[-1] == 1, "uncertainty last dim must be 1"

    # ---- G6 (Jedi — Free Energy) -------------------------------------------

    def test_g6_required_output_keys(self, g6_out):
        out, _ = g6_out
        for key in (
            "jedi_energy",
            "jedi_goal",
            "action_logits",
            "value",
            "precision",
            "hidden_states",
        ):
            assert key in out, f"G6 forward() missing key '{key}'"

    def test_g6_hidden_states_shape(self, g6_out):
        out, args = g6_out
        hs = out["hidden_states"]
        assert hs.shape == (BATCH, SEQ_LEN, args.dim), (
            f"G6 hidden_states: expected {(BATCH, SEQ_LEN, args.dim)}, got {hs.shape}"
        )

    def test_g6_action_logits_vocab(self, g6_out):
        out, args = g6_out
        al = out["action_logits"]
        assert al.shape[-1] == args.action_space_size, (
            f"action_logits last dim: expected {args.action_space_size}, got {al.shape[-1]}"
        )


# ---------------------------------------------------------------------------
# Head-count parametric regression
# ---------------------------------------------------------------------------


class TestHeadCountRegression:
    """Parametric GQA divisibility — guards against ModelArgs config drift."""

    @pytest.mark.parametrize(
        "n_heads,n_kv_heads",
        [
            (4, 1),
            (4, 2),
            (4, 4),
            (8, 1),
            (8, 2),
            (8, 4),
            (8, 8),
            (32, 8),  # G1/G6 defaults
        ],
    )
    def test_gqa_divisibility(self, n_heads, n_kv_heads):
        assert n_heads % n_kv_heads == 0, (
            f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}"
        )

    def test_g1_all_head_dims_consistent(self):
        """Every module in G1 that stores head_dim must agree with args."""
        from claudson import ModelArgs, UniversalIntelligenceModel

        args = make_small_args(ModelArgs)
        model = UniversalIntelligenceModel(args)
        expected = args.dim // args.n_heads
        for m in model.modules():
            if hasattr(m, "head_dim"):
                assert m.head_dim == expected, (
                    f"{type(m).__name__}.head_dim={m.head_dim} != {expected}"
                )

    def test_g6_all_head_dims_consistent(self):
        from claudson_jedi import ClaudesonJedi, ModelArgs

        args = make_small_args(ModelArgs)
        model = ClaudesonJedi(args)
        expected = args.dim // args.n_heads
        for m in model.modules():
            if hasattr(m, "head_dim"):
                assert m.head_dim == expected, (
                    f"{type(m).__name__}.head_dim={m.head_dim} != {expected}"
                )
