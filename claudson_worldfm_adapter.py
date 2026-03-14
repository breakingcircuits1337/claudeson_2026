"""
Claudeson 2026 — WorldFM Adapter
==================================
Adapter layer that wraps a generative novel-view synthesis backbone
(WorldFM or any compatible model) and exposes a stable contract to
the dual-path perception stack.

The adapter provides two core operations:

  encode_reference(image, K, c2w)
      Produce pose-conditioned spatial tokens from a single reference
      image and its camera parameters.  Output: [B, T_w, D_w].

  rollout_views(image, src_K, src_c2w, trajectory)
      Generate a sequence of novel-view frames along a camera
      trajectory.  Each step yields one RGB frame [B, 3, H, W].

Both operations accept standard data contracts:
  image       [B, 3, H, W]
  K           [B, 3, 3]   camera intrinsics
  c2w         [B, 4, 4]   camera-to-world extrinsics
  trajectory  list of [B, 4, 4] target c2w matrices

The stub implementation (WorldFMStub) uses a lightweight CNN so the
module is fully testable without a real WorldFM checkpoint.  Replace
WorldFMStub with a real backbone by subclassing WorldFMBase and passing
it as the `backbone` argument to WorldFMAdapter.

Data contract summary
─────────────────────
  Reference encode:  image [B,3,H,W] + K [B,3,3] + c2w [B,4,4]
                     → spatial_tokens [B, T_w, D_w]

  Rollout:           image [B,3,H,W] + src_K [B,3,3] + src_c2w [B,4,4]
                     + trajectory list<[B,4,4]>
                     → frames  list< [B,3,H,W] >

  Pose validation:   K  — det(K[:2,:2]) > 0, focal lengths > 0
                     c2w — rotation sub-matrix must be near-orthonormal
"""

# SPDX-License-Identifier: Proprietary-Commercial
# Copyright (c) 2026 Claudeson Project. All rights reserved.

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class WorldFMConfig:
    """Configuration for the WorldFM adapter."""

    # Token space
    spatial_dim: int = 256  # D_w — output token dimension
    spatial_seq_len: int = 64  # T_w — number of spatial tokens

    # Image handling
    img_size: int = 256  # H = W for input images
    patch_size: int = 16  # spatial patch size

    # Pose encoding
    pose_embed_dim: int = 64  # dimension for K and c2w embeddings

    # Rollout
    max_rollout_steps: int = 8  # max trajectory length

    # Confidence / quality
    pose_ortho_tol: float = 0.02  # tolerance for rotation orthonormality check
    min_focal_length: float = 1e-3  # minimum valid focal length (pixels)

    # Stub backbone settings (used when no real backbone is provided)
    stub_hidden: int = 128


# ─── Pose utilities ───────────────────────────────────────────────────────────


def validate_intrinsics(K: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    """
    Validate camera intrinsic matrices.

    Args:
        K:   [B, 3, 3] intrinsics batch
        tol: minimum acceptable focal length

    Returns:
        valid: [B] bool tensor — True if K appears well-formed
    """
    # Focal lengths are K[0,0] and K[1,1]
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    valid = (fx > tol) & (fy > tol)
    return valid


def validate_extrinsics(c2w: torch.Tensor, tol: float = 0.02) -> torch.Tensor:
    """
    Validate camera-to-world matrices by checking that the 3×3 rotation
    sub-block is near-orthonormal.

    Args:
        c2w: [B, 4, 4] camera-to-world matrices
        tol: maximum allowed deviation from identity for R^T R

    Returns:
        valid: [B] bool tensor
    """
    R = c2w[:, :3, :3]  # [B, 3, 3]
    eye3 = torch.eye(3, device=c2w.device, dtype=c2w.dtype).unsqueeze(0)
    err = (torch.bmm(R.transpose(1, 2), R) - eye3).abs().amax(dim=(1, 2))
    return err < tol


def encode_pose(K: torch.Tensor, c2w: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    Encode camera parameters into a flat pose embedding.

    Extracts: [fx, fy, cx, cy] from K and the 12 free parameters of c2w
    (3×3 rotation + 3 translation), then projects to embed_dim via a
    fixed Fourier-feature embedding.

    Args:
        K:         [B, 3, 3]
        c2w:       [B, 4, 4]
        embed_dim: output dimension (must be even)

    Returns:
        pose_emb: [B, embed_dim]
    """
    B = K.shape[0]

    # Flatten pose parameters: fx, fy, cx, cy + R (9) + t (3) = 16 values
    fx = K[:, 0, 0:1]
    fy = K[:, 1, 1:2]
    cx = K[:, 0, 2:3]
    cy = K[:, 1, 2:3]
    R = c2w[:, :3, :3].reshape(B, 9)
    t = c2w[:, :3, 3]

    raw = torch.cat([fx, fy, cx, cy, R, t], dim=-1)  # [B, 16]

    # Fourier features
    half = embed_dim // 2
    freqs = torch.linspace(1.0, 16.0, half, device=K.device).unsqueeze(0)  # [1, half]
    raw_pad = raw[:, :half]  # use first `half` pose components
    pose_emb = torch.cat(
        [
            torch.sin(raw_pad * freqs),
            torch.cos(raw_pad * freqs),
        ],
        dim=-1,
    )  # [B, embed_dim]

    return pose_emb


# ─── Abstract backbone interface ──────────────────────────────────────────────


class WorldFMBase(nn.Module):
    """
    Abstract interface for a WorldFM-compatible novel-view backbone.

    Subclasses must implement:
      encode(image, pose_emb) → [B, T_w, D_w]
      decode(spatial_tokens, pose_emb) → [B, 3, H, W]
    """

    def encode(
        self,
        image: torch.Tensor,  # [B, 3, H, W]
        pose_emb: torch.Tensor,  # [B, pose_embed_dim]
    ) -> torch.Tensor:  # [B, T_w, D_w]
        raise NotImplementedError

    def decode(
        self,
        spatial_tokens: torch.Tensor,  # [B, T_w, D_w]
        pose_emb: torch.Tensor,  # [B, pose_embed_dim]
    ) -> torch.Tensor:  # [B, 3, H, W]
        raise NotImplementedError


# ─── Stub backbone (testable without a real checkpoint) ───────────────────────


class WorldFMStub(WorldFMBase):
    """
    Lightweight stub that satisfies the WorldFMBase interface.
    Uses a shallow CNN encoder and a transposed-conv decoder.
    Intended for unit testing and integration smoke tests only.
    """

    def __init__(self, cfg: WorldFMConfig):
        super().__init__()
        h = cfg.stub_hidden
        p = cfg.patch_size

        # Encoder: image → spatial tokens
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, h, kernel_size=p, stride=p),  # patchify
            nn.GELU(),
        )
        n_patches_1d = cfg.img_size // p
        T_w = n_patches_1d * n_patches_1d
        self.token_proj = nn.Linear(h + cfg.pose_embed_dim, cfg.spatial_dim)
        self.T_w = T_w

        # Decoder: each token projects directly to 3*p*p pixel values,
        # then patches are assembled into the full image.
        self.dec_proj = nn.Linear(cfg.spatial_dim + cfg.pose_embed_dim, 3 * p * p)
        self.p = p
        self.img_size = cfg.img_size

    def encode(self, image: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        image.shape[0]
        feats = self.img_enc(image)  # [B, h, n, n]
        feats = feats.flatten(2).transpose(1, 2)  # [B, T_w, h]
        T = feats.shape[1]
        pose_exp = pose_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T_w, pose_dim]
        fused = torch.cat([feats, pose_exp], dim=-1)
        return self.token_proj(fused)  # [B, T_w, D_w]

    def decode(self, spatial_tokens: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = spatial_tokens.shape
        p = self.p
        n = int(math.isqrt(T))  # patches per side

        pose_exp = pose_emb.unsqueeze(1).expand(-1, T, -1)
        fused = torch.cat([spatial_tokens, pose_exp], dim=-1)  # [B, T, D+pose]
        dec = self.dec_proj(fused)  # [B, T, 3*p*p]

        # Assemble T patches into a full image
        dec = dec.reshape(B, n, n, 3, p, p)  # [B, n, n, 3, p, p]
        img = dec.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, n * p, n * p)  # [B, 3, H, W]
        return img.tanh()


# ─── WorldFM Adapter ──────────────────────────────────────────────────────────


class WorldFMAdapter(nn.Module):
    """
    Pose-conditioned novel-view synthesis adapter.

    Wraps a WorldFMBase backbone and exposes two high-level operations:

      encode_reference  — produce spatial tokens from one reference view
      rollout_views     — generate frames along a camera trajectory

    Both operations validate input pose matrices and log warnings when
    pose quality is low rather than raising hard errors, allowing the
    dual-path system to degrade gracefully.
    """

    def __init__(self, cfg: WorldFMConfig, backbone: Optional[WorldFMBase] = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else WorldFMStub(cfg)

        # Project D_w → model_dim is done externally (TokenFusionModule)
        # Here we only need the spatial_dim output from the backbone.

    # ── Pose validation ───────────────────────────────────────────────────────

    def _check_pose(self, K: torch.Tensor, c2w: torch.Tensor) -> Tuple[bool, str]:
        """Return (ok, reason). Logs a warning and returns ok=False on bad pose."""
        k_valid = validate_intrinsics(K, tol=self.cfg.min_focal_length)
        c2w_valid = validate_extrinsics(c2w, tol=self.cfg.pose_ortho_tol)
        if not k_valid.all():
            n_bad = int((~k_valid).sum())
            log.warning("WorldFMAdapter: %d/%d intrinsic matrices invalid", n_bad, K.shape[0])
            return False, f"{n_bad} invalid intrinsic matrices"
        if not c2w_valid.all():
            n_bad = int((~c2w_valid).sum())
            log.warning("WorldFMAdapter: %d/%d extrinsic matrices invalid", n_bad, c2w.shape[0])
            return False, f"{n_bad} non-orthonormal rotation blocks"
        return True, "ok"

    # ── Core operations ───────────────────────────────────────────────────────

    def encode_reference(
        self,
        image: torch.Tensor,  # [B, 3, H, W]
        K: torch.Tensor,  # [B, 3, 3]
        c2w: torch.Tensor,  # [B, 4, 4]
    ) -> torch.Tensor:  # [B, T_w, D_w]
        """
        Encode a reference image + camera into spatial tokens.

        Returns a zero tensor of the expected shape when pose validation
        fails, tagged via a logged warning.
        """
        ok, reason = self._check_pose(K, c2w)
        if not ok:
            B = image.shape[0]
            return torch.zeros(
                B,
                self.cfg.spatial_seq_len,
                self.cfg.spatial_dim,
                device=image.device,
                dtype=image.dtype,
            )

        pose_emb = encode_pose(K, c2w, self.cfg.pose_embed_dim)  # [B, P]
        tokens = self.backbone.encode(image, pose_emb)  # [B, T_w, D_w]
        return tokens

    def rollout_views(
        self,
        image: torch.Tensor,  # [B, 3, H, W]  reference frame
        src_K: torch.Tensor,  # [B, 3, 3]      source intrinsics
        src_c2w: torch.Tensor,  # [B, 4, 4]      source extrinsics
        trajectory: List[torch.Tensor],  # list of [B, 4, 4] target c2w
    ) -> List[torch.Tensor]:  # list of [B, 3, H, W]
        """
        Generate novel views along a camera trajectory.

        Each element of `trajectory` is a target c2w matrix.  The
        source camera parameters (src_K, src_c2w) define the viewpoint
        of the input `image`.  We assume the same intrinsics K for all
        generated views (intrinsics-fixed trajectory).

        Returns a list of RGB frames, one per trajectory step.
        Frames for steps with invalid target poses are replaced with
        the last valid frame (or zero if no valid frame exists yet).
        """
        ok, reason = self._check_pose(src_K, src_c2w)
        if not ok:
            log.warning("WorldFMAdapter.rollout_views: bad source pose — %s", reason)
            B, C, H, W = image.shape
            return [torch.zeros_like(image) for _ in trajectory]

        # Encode the reference
        src_pose_emb = encode_pose(src_K, src_c2w, self.cfg.pose_embed_dim)
        spatial_tokens = self.backbone.encode(image, src_pose_emb)  # [B, T_w, D_w]

        frames: List[torch.Tensor] = []
        last_valid_frame: Optional[torch.Tensor] = None

        for tgt_c2w in trajectory[: self.cfg.max_rollout_steps]:
            tgt_ok, tgt_reason = self._check_pose(src_K, tgt_c2w)
            if not tgt_ok:
                log.warning("WorldFMAdapter: skipping bad target pose — %s", tgt_reason)
                frame = (
                    last_valid_frame if last_valid_frame is not None else torch.zeros_like(image)
                )
            else:
                tgt_pose_emb = encode_pose(src_K, tgt_c2w, self.cfg.pose_embed_dim)
                frame = self.backbone.decode(spatial_tokens, tgt_pose_emb)
                last_valid_frame = frame

            frames.append(frame)

        return frames
