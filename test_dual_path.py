"""
Tests for the dual-path perception + imagination subsystem.

Coverage:
  - TokenFusionModule shape contracts
  - WorldFMAdapter pose validation
  - WorldFMAdapter encode_reference / rollout_views
  - WorldFMInvocationRouter routing policy
  - DualPathPerceptionImagination perception-only fallback
  - DualPathPerceptionImagination dual-path mode
  - Token fusion shape contract (base_tokens vs spatial_tokens dim mismatch)
  - SpatialTreeSearchPlanner.expand_node / best_trajectory
  - SceneMemoryEntry kind, confidence, outranks

Run with:  pytest test_dual_path.py -v
"""

import torch
import torch.nn as nn

# ─── Helpers ──────────────────────────────────────────────────────────────────

B, T, D = 2, 16, 32  # batch, seq_len, model_dim
H = W = 32  # image size (kept small for CPU speed)


def make_image(b=B, h=H, w=W):
    return torch.randn(b, 3, h, w)


def make_K(b=B):
    """Valid camera intrinsics: diagonal with positive focals."""
    K = torch.eye(3).unsqueeze(0).expand(b, -1, -1).clone()
    K[:, 0, 0] = 128.0  # fx
    K[:, 1, 1] = 128.0  # fy
    K[:, 0, 2] = 16.0  # cx
    K[:, 1, 2] = 16.0  # cy
    return K


def make_c2w(b=B):
    """Valid camera-to-world: identity rotation, zero translation."""
    return torch.eye(4).unsqueeze(0).expand(b, -1, -1).clone()


def make_trajectory(steps=3, b=B):
    """List of target c2w matrices (slight translation along z)."""
    traj = []
    for i in range(1, steps + 1):
        c2w = torch.eye(4).unsqueeze(0).expand(b, -1, -1).clone()
        c2w[:, 2, 3] = float(i) * 0.1  # translate z
        traj.append(c2w)
    return traj


def make_worldfm_adapter(img_size=H):
    from claudson_worldfm_adapter import WorldFMAdapter, WorldFMConfig

    cfg = WorldFMConfig(
        spatial_dim=D,
        spatial_seq_len=T,
        img_size=img_size,
        patch_size=8,
        pose_embed_dim=32,
        stub_hidden=16,
    )
    return WorldFMAdapter(cfg)


def make_vision_encoder():
    """Minimal patch encoder: image [B,3,H,W] → tokens [B, T, D]."""
    patch = 8
    n = H // patch
    n * n  # == T == 16 when H=32, patch=8
    return nn.Sequential(
        nn.Conv2d(3, D, kernel_size=patch, stride=patch),  # [B, D, n, n]
        # flatten + transpose handled via wrapper
    )


class PatchVisionEncoder(nn.Module):
    def __init__(self, dim=D, patch=8):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        feats = self.conv(x)  # [B, D, n, n]
        return feats.flatten(2).transpose(1, 2)  # [B, T, D]


# ─── TokenFusionModule ────────────────────────────────────────────────────────


class TestTokenFusionModule:
    def test_same_dim_output_shape(self):
        from claudson_token_fusion import TokenFusionModule

        mod = TokenFusionModule(model_dim=D)
        base = torch.randn(B, T, D)
        spatial = torch.randn(B, T, D)
        out = mod(base, spatial)
        assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"

    def test_different_dim_projects(self):
        """spatial_input_dim != model_dim — projection should make it work."""
        from claudson_token_fusion import TokenFusionModule

        D_w = D * 2
        mod = TokenFusionModule(model_dim=D, spatial_input_dim=D_w)
        base = torch.randn(B, T, D)
        spatial = torch.randn(B, T, D_w)
        out = mod(base, spatial)
        assert out.shape == (B, T, D)

    def test_seq_len_mismatch_truncates_to_min(self):
        from claudson_token_fusion import TokenFusionModule

        T_base = 20
        T_spatial = 12
        mod = TokenFusionModule(model_dim=D)
        base = torch.randn(B, T_base, D)
        spatial = torch.randn(B, T_spatial, D)
        out = mod(base, spatial)
        assert out.shape == (B, T_spatial, D), (
            f"Should truncate to min({T_base},{T_spatial})=12, got {out.shape}"
        )

    def test_output_is_normalized(self):
        """LayerNorm ensures output is zero-mean per token (within tolerance)."""
        from claudson_token_fusion import TokenFusionModule

        mod = TokenFusionModule(model_dim=D)
        base = torch.randn(B, T, D) * 100  # large scale
        spatial = torch.randn(B, T, D) * 100
        with torch.no_grad():
            out = mod(base, spatial)
        # LayerNorm: mean ≈ 0, std ≈ 1 over last dim
        mean = out.mean(-1).abs().max().item()
        assert mean < 1e-4, f"LayerNorm mean should be ~0, got {mean:.4e}"

    def test_gradient_flows_through_fusion(self):
        from claudson_token_fusion import TokenFusionModule

        mod = TokenFusionModule(model_dim=D)
        base = torch.randn(B, T, D, requires_grad=True)
        spatial = torch.randn(B, T, D, requires_grad=True)
        out = mod(base, spatial)
        out.sum().backward()
        assert base.grad is not None, "gradient should flow through base_tokens"
        assert spatial.grad is not None, "gradient should flow through spatial_tokens"


# ─── WorldFMAdapter — pose validation ─────────────────────────────────────────


class TestWorldFMAdapterPoseValidation:
    def test_valid_pose_returns_nonzero_tokens(self):
        adapter = make_worldfm_adapter()
        img = make_image()
        K = make_K()
        c2w = make_c2w()
        out = adapter.encode_reference(img, K, c2w)
        assert out.shape[-1] == D
        assert out.abs().sum() > 0, "Valid pose should produce non-zero tokens"

    def test_zero_focal_length_returns_zero_tokens(self):
        adapter = make_worldfm_adapter()
        img = make_image()
        K = make_K()
        K[:, 0, 0] = 0.0  # invalid: zero focal length
        c2w = make_c2w()
        out = adapter.encode_reference(img, K, c2w)
        assert out.shape[-1] == D
        assert out.abs().sum() == 0, "Invalid intrinsics should return zero tokens"

    def test_non_orthonormal_rotation_returns_zero_tokens(self):
        adapter = make_worldfm_adapter()
        img = make_image()
        K = make_K()
        c2w = make_c2w()
        c2w[:, :3, :3] = torch.ones(3, 3) * 10.0  # clearly non-orthonormal
        out = adapter.encode_reference(img, K, c2w)
        assert out.abs().sum() == 0, "Invalid extrinsics should return zero tokens"

    def test_encode_reference_output_shape(self):
        adapter = make_worldfm_adapter()
        out = adapter.encode_reference(make_image(), make_K(), make_c2w())
        assert out.ndim == 3
        assert out.shape[0] == B
        assert out.shape[2] == D


# ─── WorldFMAdapter — rollout ─────────────────────────────────────────────────


class TestWorldFMAdapterRollout:
    def test_rollout_returns_correct_number_of_frames(self):
        adapter = make_worldfm_adapter()
        traj = make_trajectory(steps=3)
        frames = adapter.rollout_views(make_image(), make_K(), make_c2w(), traj)
        assert len(frames) == 3

    def test_rollout_frame_shape(self):
        adapter = make_worldfm_adapter()
        traj = make_trajectory(steps=2)
        frames = adapter.rollout_views(make_image(), make_K(), make_c2w(), traj)
        for i, f in enumerate(frames):
            assert f.shape[0] == B, f"frame {i}: wrong batch"
            assert f.shape[1] == 3, f"frame {i}: should be RGB"
            assert f.shape[2] == H, f"frame {i}: wrong height"
            assert f.shape[3] == W, f"frame {i}: wrong width"

    def test_bad_source_pose_returns_zero_frames(self):
        adapter = make_worldfm_adapter()
        K = make_K()
        K[:, 0, 0] = 0.0  # bad
        c2w = make_c2w()
        traj = make_trajectory(steps=2)
        frames = adapter.rollout_views(make_image(), K, c2w, traj)
        assert all(f.abs().sum() == 0 for f in frames), "Bad source pose → zero frames"

    def test_empty_trajectory_returns_empty_list(self):
        adapter = make_worldfm_adapter()
        frames = adapter.rollout_views(make_image(), make_K(), make_c2w(), [])
        assert frames == []

    def test_rollout_capped_at_max_steps(self):
        from claudson_worldfm_adapter import WorldFMAdapter, WorldFMConfig

        cfg = WorldFMConfig(
            spatial_dim=D,
            spatial_seq_len=T,
            img_size=H,
            patch_size=8,
            pose_embed_dim=32,
            stub_hidden=16,
            max_rollout_steps=2,
        )
        adapter = WorldFMAdapter(cfg)
        traj = make_trajectory(steps=5)  # longer than max
        frames = adapter.rollout_views(make_image(), make_K(), make_c2w(), traj)
        assert len(frames) == 2, f"Should cap at max_rollout_steps=2, got {len(frames)}"


# ─── WorldFMInvocationRouter ──────────────────────────────────────────────────


class TestWorldFMInvocationRouter:
    def _router(self):
        from claudson_dual_path import WorldFMInvocationRouter

        return WorldFMInvocationRouter(threshold=0.4)

    def _tokens(self):
        return torch.randn(B, T, D)

    def test_no_pose_always_returns_false(self):
        router = self._router()
        assert not router(self._tokens(), K=None, c2w=None)
        assert not router(self._tokens(), K=make_K(), c2w=None)
        assert not router(self._tokens(), K=None, c2w=make_c2w())

    def test_low_uncertainty_no_spatial_task_returns_false(self):
        router = self._router()
        ctx = {"uncertainty": 0.1, "task": "text_qa"}
        result = router(self._tokens(), K=make_K(), c2w=make_c2w(), context=ctx)
        assert not result

    def test_high_uncertainty_triggers_worldfm(self):
        router = self._router()
        ctx = {"uncertainty": 0.9}
        result = router(self._tokens(), K=make_K(), c2w=make_c2w(), context=ctx)
        assert result

    def test_spatial_task_triggers_worldfm(self):
        router = self._router()
        for task in [
            "scene_reconstruction",
            "counterfactual_imagination",
            "threat_surface_analysis",
            "infrastructure_inspection",
        ]:
            ctx = {"uncertainty": 0.0, "task": task}
            result = router(self._tokens(), K=make_K(), c2w=make_c2w(), context=ctx)
            assert result, f"task={task!r} should trigger WorldFM"

    def test_force_worldfm_flag(self):
        router = self._router()
        ctx = {"force_worldfm": True}
        assert router(self._tokens(), K=make_K(), c2w=make_c2w(), context=ctx)

    def test_planner_requests_spatial(self):
        router = self._router()
        ctx = {"planner_requests_spatial": True}
        assert router(self._tokens(), K=make_K(), c2w=make_c2w(), context=ctx)


# ─── DualPathPerceptionImagination ────────────────────────────────────────────


def _build_dual_path(model_dim=D):
    from claudson_dual_path import (
        DualPathPerceptionImagination,
        WorldFMInvocationRouter,
    )
    from claudson_token_fusion import TokenFusionModule

    vision_enc = PatchVisionEncoder(dim=model_dim, patch=8)
    adapter = make_worldfm_adapter()
    fusion = TokenFusionModule(
        model_dim=model_dim,
        spatial_input_dim=adapter.cfg.spatial_dim,
    )
    router = WorldFMInvocationRouter(threshold=0.4)

    return DualPathPerceptionImagination(
        vision_encoder=vision_enc,
        worldfm_adapter=adapter,
        fusion_module=fusion,
        invocation_router=router,
    )


class TestDualPathPerceptionImagination:
    def test_perception_only_when_no_pose(self):
        model = _build_dual_path()
        model.eval()
        with torch.no_grad():
            out = model(make_image())
        assert out["mode"] == "perception_only"
        assert out["spatial_tokens"] is None
        assert out["tokens"].shape == out["base_tokens"].shape

    def test_perception_only_tokens_equal_base_tokens(self):
        model = _build_dual_path()
        model.eval()
        img = make_image()
        with torch.no_grad():
            out = model(img)
        assert torch.equal(out["tokens"], out["base_tokens"])

    def test_dual_path_activated_by_spatial_task(self):
        model = _build_dual_path()
        model.eval()
        ctx = {"task": "scene_reconstruction"}
        with torch.no_grad():
            out = model(make_image(), K=make_K(), c2w=make_c2w(), context=ctx)
        assert out["mode"] == "dual_path"
        assert out["spatial_tokens"] is not None

    def test_dual_path_token_shape(self):
        model = _build_dual_path()
        model.eval()
        ctx = {"task": "scene_reconstruction"}
        img = make_image()
        with torch.no_grad():
            out = model(img, K=make_K(), c2w=make_c2w(), context=ctx)
        out["base_tokens"]
        fused = out["tokens"]
        assert fused.ndim == 3
        assert fused.shape[0] == B
        assert fused.shape[2] == D, "Fused tokens must have model_dim D"

    def test_fallback_when_no_pose_even_with_spatial_task(self):
        """Spatial task alone can't invoke imagination without pose."""
        model = _build_dual_path()
        model.eval()
        ctx = {"task": "scene_reconstruction"}
        with torch.no_grad():
            out = model(make_image(), K=None, c2w=None, context=ctx)
        assert out["mode"] == "perception_only"

    def test_perceive_helper(self):
        model = _build_dual_path()
        tokens = model.perceive(make_image())
        assert tokens.ndim == 3
        assert tokens.shape[0] == B
        assert tokens.shape[2] == D

    def test_imagine_helper(self):
        model = _build_dual_path()
        traj = make_trajectory(steps=2)
        frames = model.imagine(make_image(), make_K(), make_c2w(), traj)
        assert len(frames) == 2
        assert frames[0].shape == (B, 3, H, W)


# ─── SpatialTreeSearchPlanner ─────────────────────────────────────────────────


class TestSpatialTreeSearchPlanner:
    def _build_planner(self):
        from claudson_dual_path import SpatialTreeSearchPlanner

        adapter = make_worldfm_adapter()
        enc = PatchVisionEncoder(dim=D, patch=8)

        def scorer(tokens):
            return float(tokens.mean())

        return SpatialTreeSearchPlanner(adapter, enc, scorer)

    def test_expand_node_returns_one_branch_per_trajectory(self):
        planner = self._build_planner()
        candidates = [make_trajectory(steps=2), make_trajectory(steps=2)]
        branches = planner.expand_node(make_image(), make_K(), make_c2w(), candidates)
        assert len(branches) == 2

    def test_branch_has_required_keys(self):
        planner = self._build_planner()
        branches = planner.expand_node(
            make_image(),
            make_K(),
            make_c2w(),
            [make_trajectory(steps=2)],
        )
        b = branches[0]
        assert "trajectory" in b
        assert "frames" in b
        assert "tokens" in b
        assert "score" in b

    def test_branch_score_is_float(self):
        planner = self._build_planner()
        branches = planner.expand_node(
            make_image(),
            make_K(),
            make_c2w(),
            [make_trajectory(steps=2)],
        )
        assert isinstance(branches[0]["score"], float)

    def test_best_trajectory_returns_highest_score(self):
        from claudson_dual_path import SpatialTreeSearchPlanner

        adapter = make_worldfm_adapter()
        enc = PatchVisionEncoder(dim=D, patch=8)

        call_count = [0]

        def deterministic_scorer(tokens):
            call_count[0] += 1
            return float(call_count[0])  # monotonically increases → last is best

        planner = SpatialTreeSearchPlanner(adapter, enc, deterministic_scorer)
        candidates = [make_trajectory(steps=1) for _ in range(3)]
        best = planner.best_trajectory(make_image(), make_K(), make_c2w(), candidates)
        assert best is not None

    def test_best_trajectory_empty_input_returns_none(self):
        planner = self._build_planner()
        result = planner.best_trajectory(make_image(), make_K(), make_c2w(), [])
        assert result is None


# ─── SceneMemoryEntry ─────────────────────────────────────────────────────────


class TestSceneMemoryEntry:
    def _entry(self, kind="observed", confidence=1.0):
        from claudson_dual_path import SceneMemoryEntry

        return SceneMemoryEntry(
            kind=kind,
            image_id="img_0",
            tokens=torch.randn(T, D),
            source="test",
            confidence=confidence,
        )

    def test_is_observed_true_for_observed(self):
        e = self._entry("observed")
        assert e.is_observed()
        assert not e.is_imagined()

    def test_is_imagined_true_for_imagined(self):
        e = self._entry("imagined")
        assert e.is_imagined()
        assert not e.is_observed()

    def test_observed_outranks_imagined(self):
        obs = self._entry("observed", confidence=0.5)
        img = self._entry("imagined", confidence=0.9)
        assert obs.outranks(img), "observed should outrank imagined regardless of confidence"
        assert not img.outranks(obs)

    def test_higher_confidence_wins_same_kind(self):
        high = self._entry("imagined", confidence=0.9)
        low = self._entry("imagined", confidence=0.3)
        assert high.outranks(low)
        assert not low.outranks(high)

    def test_equal_confidence_observed_beats_imagined(self):
        obs = self._entry("observed", confidence=0.7)
        img = self._entry("imagined", confidence=0.7)
        assert obs.outranks(img)
        assert not img.outranks(obs)

    def test_pose_field_optional(self):
        from claudson_dual_path import SceneMemoryEntry

        e = SceneMemoryEntry(kind="observed", image_id="x", tokens=torch.zeros(4, D))
        assert e.pose is None
