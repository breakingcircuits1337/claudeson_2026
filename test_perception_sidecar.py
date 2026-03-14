"""
Tests for PerceptionImaginationSidecar.

Coverage
--------
- Transparent pass-through when image=None
- Transparent pass-through when enabled=False
- perception_only mode (image, no pose)
- dual_path mode (image + pose + spatial task)
- action_logits correction applied and scaled
- value correction applied and scaled
- corrections are zero at init (weight-zero init guarantee)
- perception_signal shape contract
- output dict always includes "perception_imagination"
- model output keys preserved in all modes
- gate is in (0, 1)
- imagination_active flag matches mode string

Run with:  pytest test_perception_sidecar.py -v
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# ── constants ────────────────────────────────────────────────────────────────
B = 2          # batch size
L = 16         # text sequence length
D_M = 64       # model hidden dim (small for CPU tests)
D_W = 32       # dual-path token dim (matches WorldFMConfig.spatial_dim)
T_W = 8        # dual-path sequence length (spatial tokens)
A = 20         # action space size
H = W = 32    # image spatial size

# ── helpers ──────────────────────────────────────────────────────────────────


def make_image(b: int = B) -> torch.Tensor:
    return torch.randn(b, 3, H, W)


def make_K(b: int = B) -> torch.Tensor:
    K = torch.eye(3).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = 128.0
    K[:, 1, 1] = 128.0
    K[:, 0, 2] = 16.0
    K[:, 1, 2] = 16.0
    return K


def make_c2w(b: int = B) -> torch.Tensor:
    return torch.eye(4).unsqueeze(0).expand(b, -1, -1).clone()


def make_text(b: int = B, l: int = L) -> torch.Tensor:
    return torch.randint(0, 1000, (b, l))


# ── Mock core model ───────────────────────────────────────────────────────────


class MockCoreModel(nn.Module):
    """
    Minimal stand-in for any Claudson generation model.
    Accepts ``text`` [B, L] and returns a dict matching G6+ key conventions.
    """

    def __init__(self, dim: int = D_M, action_space: int = A) -> None:
        super().__init__()
        self.emb = nn.Embedding(1000, dim)
        self.proj = nn.Linear(dim, dim)
        self.logit_head = nn.Linear(dim, action_space)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, text: torch.Tensor, **_: Any) -> Dict[str, Any]:  # noqa: ANN401
        h = self.emb(text).mean(1)  # [B, D_M]
        h = self.proj(h)
        return {
            "hidden_states": h.unsqueeze(1).expand(-1, text.shape[1], -1),
            "action_logits": self.logit_head(h),  # [B, A]
            "value": self.value_head(h),            # [B, 1]
            "custom_key": torch.zeros(B, 3),       # extra key — must survive
        }


class MockCoreModelNoLogits(nn.Module):
    """Core model that returns neither action_logits nor value (e.g. G1)."""

    def forward(self, text: torch.Tensor, **_: Any) -> Dict[str, Any]:  # noqa: ANN401
        return {"hidden_states": torch.zeros(text.shape[0], text.shape[1], D_M)}


# ── Mock DualPath ─────────────────────────────────────────────────────────────


class MockDualPath(nn.Module):
    """
    Drop-in fake for DualPathPerceptionImagination.
    Returns deterministic tensors so correction arithmetic is exact.
    """

    def __init__(
        self,
        token_dim: int = D_W,
        seq_len: int = T_W,
        force_dual: bool = False,
    ) -> None:
        super().__init__()
        self._dim = token_dim
        self._seq = seq_len
        self._force_dual = force_dual

    def forward(
        self,
        image: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        b = image.shape[0]
        base_tokens = torch.ones(b, self._seq, self._dim)  # all-ones for easy checks

        if self._force_dual or (context and context.get("task") == "scene_reconstruction" and K is not None and c2w is not None):
            spatial_tokens = torch.ones(b, self._seq, self._dim) * 2.0  # distinct value
            # Simple fusion: mean
            tokens = (base_tokens + spatial_tokens) / 2.0
            mode = "dual_path"
        else:
            tokens = base_tokens
            spatial_tokens = None
            mode = "perception_only"

        return {
            "tokens": tokens,
            "base_tokens": base_tokens,
            "spatial_tokens": spatial_tokens,
            "mode": mode,
        }


# ── Sidecar factory ──────────────────────────────────────────────────────────


def make_sidecar(
    force_dual: bool = False,
    enabled: bool = True,
    has_logits: bool = True,
    correction_scale: float = 0.1,
) -> "PerceptionImaginationSidecar":
    from claudson_perception_sidecar import PerceptionImaginationSidecar, SidecarConfig

    core = MockCoreModel(dim=D_M, action_space=A) if has_logits else MockCoreModelNoLogits()
    dp = MockDualPath(token_dim=D_W, seq_len=T_W, force_dual=force_dual)

    cfg = SidecarConfig(
        dual_path_dim=D_W,
        model_dim=D_M,
        action_space_size=A,
        correction_scale=correction_scale,
        enabled=enabled,
    )
    return PerceptionImaginationSidecar(core, dp, cfg)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTransparency:
    """Sidecar must be a no-op when image is absent or when disabled."""

    def test_no_image_returns_model_out_unchanged(self):
        sc = make_sidecar()
        text = make_text()
        with torch.no_grad():
            out = sc(text=text)
        assert "action_logits" in out
        assert out["perception_imagination"]["mode"] == "text_only"
        assert out["perception_imagination"]["tokens"] is None

    def test_disabled_sidecar_ignores_image(self):
        sc = make_sidecar(enabled=False)
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image(), K=make_K(), c2w=make_c2w())
        assert out["perception_imagination"]["mode"] == "text_only"
        assert "perception_signal" not in out, "No signal projected when disabled"

    def test_custom_model_keys_preserved_no_image(self):
        sc = make_sidecar()
        out = sc(text=make_text())
        assert "custom_key" in out

    def test_custom_model_keys_preserved_with_image(self):
        sc = make_sidecar()
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        assert "custom_key" in out


class TestPerceptionOnly:
    """Image provided but no pose → perception-only mode."""

    def test_mode_is_perception_only_without_pose(self):
        sc = make_sidecar()
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image(), K=None, c2w=None)
        assert out["perception_imagination"]["mode"] == "perception_only"
        assert out["perception_imagination"]["spatial_tokens"] is None
        assert not out["perception_imagination"]["imagination_active"]

    def test_tokens_present_in_perception_only(self):
        sc = make_sidecar()
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        tokens = out["perception_imagination"]["tokens"]
        assert tokens is not None
        assert tokens.ndim == 3
        assert tokens.shape == (B, T_W, D_W)

    def test_perception_signal_has_model_dim(self):
        sc = make_sidecar()
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        sig = out["perception_signal"]
        assert sig.shape == (B, D_M), f"Expected ({B},{D_M}), got {sig.shape}"


class TestDualPath:
    """Image + pose + spatial task → imagination activated."""

    def test_mode_is_dual_path_with_pose_and_spatial_task(self):
        sc = make_sidecar()
        ctx = {"task": "scene_reconstruction"}
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image(), K=make_K(), c2w=make_c2w(), context=ctx)
        assert out["perception_imagination"]["mode"] == "dual_path"
        assert out["perception_imagination"]["imagination_active"]

    def test_spatial_tokens_present_in_dual_path(self):
        sc = make_sidecar()
        ctx = {"task": "scene_reconstruction"}
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image(), K=make_K(), c2w=make_c2w(), context=ctx)
        assert out["perception_imagination"]["spatial_tokens"] is not None

    def test_force_dual_flag_overrides_routing(self):
        sc = make_sidecar(force_dual=True)
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        assert out["perception_imagination"]["mode"] == "dual_path"


class TestCorrections:
    """Correction arithmetic and scale/gate contracts."""

    def test_logit_correction_zero_at_init(self):
        """
        logit_correction weights are initialised to zero, so the sidecar
        starts as a pure pass-through even when an image is provided.
        """
        sc = make_sidecar()
        text = make_text()
        # Baseline: no image
        with torch.no_grad():
            base = sc(text=text)["action_logits"].clone()
            augmented = sc(text=text, image=make_image())["action_logits"]
        # At init, delta = 0 → augmented == base
        assert torch.allclose(base, augmented, atol=1e-6), (
            "Logit correction should be zero at init"
        )

    def test_value_correction_zero_at_init(self):
        sc = make_sidecar()
        text = make_text()
        with torch.no_grad():
            base = sc(text=text)["value"].clone()
            augmented = sc(text=text, image=make_image())["value"]
        assert torch.allclose(base, augmented, atol=1e-6)

    def test_non_zero_correction_after_manual_weight_set(self):
        """After manually setting correction weights, deltas are non-zero."""
        sc = make_sidecar(correction_scale=1.0)
        nn.init.constant_(sc.logit_correction.weight, 0.5)
        nn.init.zeros_(sc.logit_correction.bias)

        text = make_text()
        with torch.no_grad():
            base_out = sc(text=text)
            aug_out = sc(text=text, image=make_image())

        diff = (aug_out["action_logits"] - base_out["action_logits"]).abs().max().item()
        assert diff > 0, "Non-zero weight should produce non-zero correction"

    def test_correction_scale_applied(self):
        """
        Correction with scale=0 should leave logits identical to base.
        """
        sc_zero = make_sidecar(correction_scale=0.0)
        nn.init.constant_(sc_zero.logit_correction.weight, 1.0)

        text = make_text()
        with torch.no_grad():
            base = sc_zero(text=text)["action_logits"].clone()
            aug = sc_zero(text=text, image=make_image())["action_logits"]
        assert torch.allclose(base, aug, atol=1e-6), (
            "correction_scale=0 must suppress all corrections"
        )

    def test_gate_is_in_unit_interval(self):
        """Sigmoid gate output must be in (0, 1) for any input."""
        sc = make_sidecar()
        # Use non-trivial weights so gate doesn't collapse to 0.5
        nn.init.normal_(sc.gate[0].weight, std=5.0)
        with torch.no_grad():
            # Trigger sidecar by providing an image
            sc(text=make_text(), image=make_image())
        # Evaluate gate directly on a range of inputs
        for _ in range(5):
            x = torch.randn(B, D_W) * 10
            g = sc.gate(x)
            assert (g >= 0.0).all() and (g <= 1.0).all(), "Gate must be in [0, 1]"


class TestOutputKeys:
    """Output dict structure contracts."""

    def test_perception_imagination_key_always_present(self):
        sc = make_sidecar()
        for img, K, c2w in [
            (None, None, None),
            (make_image(), None, None),
            (make_image(), make_K(), make_c2w()),
        ]:
            out = sc(text=make_text(), image=img, K=K, c2w=c2w)
            assert "perception_imagination" in out, (
                f"Missing key for img={img is not None}, K={K is not None}"
            )

    def test_perception_imagination_sub_keys(self):
        sc = make_sidecar()
        out = sc(text=make_text(), image=make_image())
        pi = out["perception_imagination"]
        for key in ("tokens", "base_tokens", "spatial_tokens", "mode", "imagination_active"):
            assert key in pi, f"Missing sub-key: {key!r}"

    def test_no_perception_signal_without_image(self):
        sc = make_sidecar()
        out = sc(text=make_text())
        assert "perception_signal" not in out

    def test_perception_signal_present_with_image(self):
        sc = make_sidecar()
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        assert "perception_signal" in out

    def test_action_logits_shape_unchanged(self):
        sc = make_sidecar()
        text = make_text()
        with torch.no_grad():
            base = sc(text=text)["action_logits"]
            aug = sc(text=text, image=make_image())["action_logits"]
        assert base.shape == aug.shape, "action_logits shape must not change"

    def test_value_shape_unchanged(self):
        sc = make_sidecar()
        text = make_text()
        with torch.no_grad():
            base = sc(text=text)["value"]
            aug = sc(text=text, image=make_image())["value"]
        assert base.shape == aug.shape, "value shape must not change"


class TestModelWithoutLogits:
    """Sidecar gracefully handles models that lack action_logits/value."""

    def test_no_crash_when_model_lacks_logit_keys(self):
        from claudson_perception_sidecar import PerceptionImaginationSidecar, SidecarConfig

        core = MockCoreModelNoLogits()
        dp = MockDualPath()
        cfg = SidecarConfig(dual_path_dim=D_W, model_dim=D_M, action_space_size=A)
        sc = PerceptionImaginationSidecar(core, dp, cfg)
        with torch.no_grad():
            out = sc(text=make_text(), image=make_image())
        assert "action_logits" not in out
        assert "value" not in out
        assert "perception_imagination" in out

    def test_hidden_states_key_preserved(self):
        from claudson_perception_sidecar import PerceptionImaginationSidecar, SidecarConfig

        core = MockCoreModelNoLogits()
        dp = MockDualPath()
        cfg = SidecarConfig(dual_path_dim=D_W, model_dim=D_M, action_space_size=A)
        sc = PerceptionImaginationSidecar(core, dp, cfg)
        out = sc(text=make_text())
        assert "hidden_states" in out


class TestGradientFlow:
    """Learnable sidecar parameters receive gradients during back-prop."""

    def test_correction_head_receives_gradient(self):
        sc = make_sidecar(correction_scale=1.0)
        nn.init.constant_(sc.logit_correction.weight, 0.1)

        out = sc(text=make_text(), image=make_image())
        loss = out["action_logits"].sum() + out["value"].sum()
        loss.backward()

        assert sc.logit_correction.weight.grad is not None, "logit_correction must get grad"
        assert sc.value_correction.weight.grad is not None, "value_correction must get grad"
        assert sc.gate[0].weight.grad is not None, "gate must get grad"

    def test_token_projector_receives_gradient(self):
        sc = make_sidecar()
        out = sc(text=make_text(), image=make_image())
        out["perception_signal"].sum().backward()
        assert sc.token_projector.weight.grad is not None
