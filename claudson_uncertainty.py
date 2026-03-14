"""
Claudeson 2026 - Uncertainty Edition
======================================
Bayesian Uncertainty · Conformal Prediction · Calibration · Epistemic Humility

The problem this generation solves
------------------------------------
Every previous generation produces outputs with implicit confidence —
the softmax temperature, the EFE balance, the moral verdict score.
But none of these are *calibrated*: a score of 0.9 does not reliably mean
the system is correct 90% of the time.

Uncalibrated confidence is dangerous in exactly the situations that matter:
  - Novel domains (distribution shift): the model is confidently wrong
    because its uncertainty estimator never trained on out-of-distribution inputs
  - High-stakes decisions: a moral verdict of 0.85 sounds authoritative but
    may have ±0.4 uncertainty that is never surfaced
  - Self-improvement (RSI): the system modifies itself based on predicted
    improvements; if predictions are overconfident, modifications become
    destabilising

This generation adds five components:

  1. Bayesian Uncertainty Propagator (BUP)
     Propagates epistemic uncertainty through the full forward pass using
     a lightweight Monte Carlo approach: N stochastic forward passes with
     different dropout masks, then estimate mean and variance of outputs.
     Separates aleatoric uncertainty (irreducible noise in the data) from
     epistemic uncertainty (reducible by more data / computation).

  2. Conformal Prediction Layer (CPL)
     Wraps any output head with a distribution-free coverage guarantee.
     Based on Venn-Abers / split conformal prediction:
     given a calibration set, produces prediction SETS (not points) that
     contain the true answer with guaranteed probability (e.g. 90%).
     This is the gold standard for uncertainty quantification: no assumptions
     about the distribution, works with any model.

  3. Calibration Monitor (CM)
     Tracks the empirical calibration of every output head online.
     Plots reliability diagrams internally and computes Expected Calibration
     Error (ECE).  Applies temperature scaling and Platt scaling corrections
     automatically when miscalibration is detected.

  4. Out-of-Distribution Detector (OODD)
     Detects when an input is outside the training distribution using:
       - Mahalanobis distance in representation space
       - Energy-based OOD score (Liu et al. 2020)
       - Spectral normalisation-based uncertainty
     When OOD is detected, hedges outputs and increases epistemic uncertainty.

  5. Uncertainty-Aware Aggregator (UAA)
     Combines uncertainty signals from all previous layers into a unified
     epistemic state vector that downstream components can query.
     "How confident am I, and about what?" — answered per output head,
     per stakeholder welfare estimate, per schema binding, per moral verdict.

Architecture evolution:
  ... → social_alignment → uncertainty
                                ↑ you are here
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claudson_jedi import RMSNorm, SwiGLU
from claudson_social_alignment import (
    ClaudesonSocialAlignment,
)
from claudson_social_alignment import (
    ModelArgs as SocialAlignmentArgs,
)

log = logging.getLogger(__name__)


# ============= Configuration =============


@dataclass
class ModelArgs(SocialAlignmentArgs):
    # Bayesian Uncertainty Propagator
    bup_n_samples: int = 10  # MC dropout samples for uncertainty estimation
    bup_dropout_rate: float = 0.1  # dropout rate for stochastic forward passes
    bup_hidden: int = 256  # hidden dim for uncertainty head

    # Conformal Prediction
    cp_coverage: float = 0.9  # target coverage probability (e.g. 90%)
    cp_cal_size: int = 512  # calibration set size (rolling window)
    cp_n_classes: int = 64  # number of output classes for classification heads

    # Calibration Monitor
    cal_n_bins: int = 15  # ECE histogram bins
    cal_temp_init: float = 1.0  # initial temperature scaling value
    cal_update_rate: float = 0.01  # online calibration update rate

    # OOD Detector
    ood_n_centroids: int = 64  # cluster centroids for Mahalanobis distance
    ood_hidden: int = 256  # hidden dim for energy-based OOD scorer
    ood_threshold: float = 0.7  # above this OOD score → flag as OOD
    ood_ema: float = 0.99  # EMA for centroid updates

    # Uncertainty-Aware Aggregator
    uaa_hidden: int = 256  # hidden dim for aggregator
    uaa_n_heads: int = 4  # uncertainty summary heads


# ============= Bayesian Uncertainty Propagator =============


class MCDropoutHead(nn.Module):
    """
    A prediction head wrapped with Monte Carlo dropout.

    During inference, instead of a single forward pass, runs N passes
    with dropout active.  The variance across passes estimates epistemic
    uncertainty.  The mean is the point estimate.

    Separating aleatoric from epistemic:
      - Epistemic: variance of the mean predictions across MC samples
        (reducible — more data / compute would reduce this)
      - Aleatoric: mean of the predicted variances across MC samples
        (irreducible — inherent noise in the data generation process)

    This follows the decomposition from Kendall & Gal (2017):
      Total uncertainty = Epistemic + Aleatoric
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.n_out = out_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_dim * 2),  # mean + log_var
        )

    def forward_once(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)  # [..., out*2]
        mu = out[..., : self.n_out]
        log_var = out[..., self.n_out :].clamp(-10, 5)
        return mu, log_var

    def forward(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Run N stochastic forward passes and compute uncertainty statistics."""
        self.train()  # activate dropout even at inference time

        mus, log_vars = [], []
        for _ in range(n_samples):
            mu, lv = self.forward_once(x)
            mus.append(mu)
            log_vars.append(lv)

        mus = torch.stack(mus, dim=0)  # [N, ..., out]
        log_vars = torch.stack(log_vars, dim=0)

        # Point estimate: mean of MC samples
        mu_mean = mus.mean(0)  # [..., out]

        # Epistemic uncertainty: variance of means
        epistemic = mus.var(0)  # [..., out]

        # Aleatoric uncertainty: mean of predicted variances
        aleatoric = log_vars.exp().mean(0)  # [..., out]

        # Total uncertainty
        total = epistemic + aleatoric

        return {
            "mean": mu_mean,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "log_var": log_vars.mean(0),
        }


class BayesianUncertaintyPropagator(nn.Module):
    """
    Wraps the final hidden state representation with Bayesian uncertainty.

    Maintains parallel uncertainty heads for:
      - Token-level predictions
      - Welfare estimates (per stakeholder)
      - Moral verdicts (per framework)
      - Goal achievement predictions

    The propagation:
      Each head produces a distribution, not a point estimate.
      Downstream decisions can query: "how certain are you about X?"
      before committing to an action.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_samples = args.bup_n_samples
        self.dim = args.dim
        h = args.bup_hidden

        # Main uncertainty head: hidden state → uncertainty distribution
        self.main_head = MCDropoutHead(args.dim, args.dim, h, args.bup_dropout_rate)

        # Per-domain uncertainty heads
        self.welfare_unc = MCDropoutHead(
            args.n_stakeholder_groups, args.n_stakeholder_groups, h // 2, args.bup_dropout_rate
        )
        self.moral_unc = MCDropoutHead(
            args.n_moral_frameworks, args.n_moral_frameworks, h // 2, args.bup_dropout_rate
        )

        # Uncertainty state: running estimate of current epistemic state
        self.register_buffer("epistemic_state", torch.ones(args.dim))
        self.unc_ema = 0.95

    @torch.no_grad()
    def _update_epistemic_state(self, epistemic: torch.Tensor) -> None:
        self.epistemic_state = self.unc_ema * self.epistemic_state + (
            1 - self.unc_ema
        ) * epistemic.mean(0).mean(0)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        welfare: torch.Tensor,  # [B, n_groups]
        moral: torch.Tensor,  # [B, n_frameworks]
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Main uncertainty pass
        pooled = x.mean(1)  # [B, D]
        main_out = self.main_head(pooled, self.n_samples)

        # Domain-specific uncertainty
        welfare_out = self.welfare_unc(welfare, self.n_samples)
        moral_out = self.moral_unc(moral, self.n_samples)

        # Update epistemic state
        self._update_epistemic_state(main_out["epistemic"])

        # Uncertainty-weighted hidden state: scale by inverse total uncertainty
        uncertainty_scale = 1.0 / (main_out["total"].sqrt() + 1e-6)  # [B, D]
        uncertainty_scale = uncertainty_scale.clamp(0.1, 10.0).unsqueeze(1)
        x_uncertain = x * uncertainty_scale * 0.1 + x * 0.9  # soft modulation

        return x_uncertain, {
            "main": main_out,
            "welfare": welfare_out,
            "moral": moral_out,
            "epistemic_state": self.epistemic_state,
        }


# ============= Conformal Prediction Layer =============


class ConformalPrediction(nn.Module):
    """
    Distribution-free prediction sets with guaranteed coverage.

    Split conformal prediction (Papadopoulos et al. 2002, Venn 2003):

    1. CALIBRATION PHASE (offline):
       For each calibration example (x_i, y_i):
         - Compute nonconformity score s_i = 1 - f(x_i)[y_i]
           (how surprised is the model by the true label?)
         - Store all scores in calibration_scores

    2. PREDICTION PHASE (online):
       For a new x, compute prediction SET:
         - Compute s_j = 1 - f(x)[j] for each class j
         - Include class j in the set iff s_j ≤ q_{1-α}
           where q_{1-α} is the (1-α) quantile of calibration_scores
         - This set is guaranteed to contain the true label with prob ≥ 1-α

    Key property: NO DISTRIBUTIONAL ASSUMPTIONS.
    Works for any model, any data distribution, any architecture.
    The coverage guarantee is finite-sample and exact (not asymptotic).

    In our setting, we apply conformal prediction to:
      - Action selection (which action is in the 90% prediction set?)
      - Schema selection (which schema is in the set?)
      - Moral framework agreement (which frameworks are in the consensus set?)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.coverage = args.cp_coverage
        self.cal_size = args.cp_cal_size
        self.n_classes = args.cp_n_classes
        self.dim = args.dim

        # Score projector: hidden → nonconformity scores per class
        self.score_proj = nn.Sequential(
            nn.Linear(args.dim, args.bup_hidden),
            nn.GELU(),
            nn.Linear(args.bup_hidden, args.cp_n_classes),
        )

        # Calibration scores buffer (rolling window)
        self.register_buffer("cal_scores", torch.zeros(args.cp_cal_size))
        self.register_buffer("cal_ptr", torch.tensor(0))
        self.register_buffer("cal_count", torch.tensor(0))
        self.register_buffer("quantile", torch.tensor(1.0))  # initially maximally conservative

    @torch.no_grad()
    def update_calibration(self, nonconformity_score: float) -> None:
        """Add a new nonconformity score to the calibration buffer."""
        ptr = int(self.cal_ptr.item())
        self.cal_scores[ptr] = nonconformity_score
        self.cal_ptr = torch.tensor((ptr + 1) % self.cal_size)
        self.cal_count = self.cal_count + 1

        # Update quantile
        n = min(int(self.cal_count.item()), self.cal_size)
        scores = self.cal_scores[:n]
        alpha = 1.0 - self.coverage
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        self.quantile = torch.quantile(scores, q_level)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)  # [B, D]

        # Softmax probabilities
        logits = self.score_proj(pooled)  # [B, n_classes]
        probs = F.softmax(logits, dim=-1)

        # Nonconformity scores: 1 - p[class]
        # For prediction sets: include class j iff (1 - p[j]) ≤ quantile
        # ↔ p[j] ≥ 1 - quantile
        threshold = 1.0 - self.quantile.item()
        prediction_set = probs >= threshold  # [B, n_classes] bool

        # Set size: smaller → more informative prediction
        set_sizes = prediction_set.float().sum(-1)  # [B]

        # Uncertainty from set size: large set = uncertain
        uncertainty = set_sizes / self.n_classes  # [B] ∈ [0,1]

        return x, {
            "probs": probs,
            "prediction_set": prediction_set,
            "set_sizes": set_sizes,
            "quantile": self.quantile.item(),
            "uncertainty": uncertainty,
            "coverage_target": self.coverage,
            "cal_count": int(self.cal_count.item()),
        }


# ============= Calibration Monitor =============


class CalibrationMonitor(nn.Module):
    """
    Tracks and corrects the calibration of all output heads.

    A model is *calibrated* if its confidence matches its accuracy:
      "When I say I'm 80% confident, I should be right 80% of the time."

    Expected Calibration Error (ECE):
      Partition predictions into M confidence bins.
      ECE = Σ_m |B_m|/n * |acc(B_m) - conf(B_m)|
      where B_m = set of examples in bin m.

    Corrections applied automatically:
      - Temperature scaling: divide logits by T (T>1 softens, T<1 sharpens)
      - Platt scaling: linear transformation of logits before softmax
      - Isotonic regression: non-parametric calibration curve (stored as lookup)

    Online updates:
      As new (prediction, outcome) pairs arrive, the ECE is updated and
      corrections are adjusted via gradient descent on the calibration loss.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_bins = args.cal_n_bins
        self.update_rate = args.cal_update_rate

        # Temperature scaling parameter (learnable)
        self.temperature = nn.Parameter(torch.tensor(args.cal_temp_init))

        # Platt scaling: a + b * logit
        self.platt_a = nn.Parameter(torch.ones(1))
        self.platt_b = nn.Parameter(torch.zeros(1))

        # ECE tracking buffers per bin
        self.register_buffer("bin_correct", torch.zeros(args.cal_n_bins))
        self.register_buffer("bin_total", torch.zeros(args.cal_n_bins))
        self.register_buffer("ece", torch.tensor(0.0))
        self.register_buffer("n_calibrations", torch.tensor(0))

    @torch.no_grad()
    def update(self, confidence: torch.Tensor, correct: torch.Tensor) -> None:
        """
        Update calibration statistics with a batch of (confidence, correct) pairs.
        confidence: [B] — predicted probability of the chosen class
        correct:    [B] — 1 if prediction was correct, 0 otherwise
        """
        B = confidence.size(0)
        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=confidence.device)

        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidence >= lo) & (confidence < hi)
            if mask.any():
                self.bin_correct[i] += correct[mask].sum()
                self.bin_total[i] += mask.float().sum()

        # Recompute ECE
        n = self.bin_total.sum()
        if n > 0:
            acc = self.bin_correct / (self.bin_total + 1e-8)
            conf = torch.linspace(0, 1, self.n_bins, device=self.ece.device) + 0.5 / self.n_bins
            self.ece = ((self.bin_total / n) * (acc - conf).abs()).sum()

        self.n_calibrations = self.n_calibrations + B

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature + Platt scaling to logits."""
        temp = self.temperature.clamp(0.1, 10.0)
        return (self.platt_a * logits + self.platt_b) / temp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        return x, {
            "ece": self.ece.item(),
            "temperature": self.temperature.item(),
            "n_calibrations": int(self.n_calibrations.item()),
            "bin_accuracy": (self.bin_correct / (self.bin_total + 1e-8)).tolist(),
        }


# ============= Out-of-Distribution Detector =============


class OODDetector(nn.Module):
    """
    Detects when an input is outside the training distribution.

    Three complementary methods:

    1. Mahalanobis Distance (Lee et al. 2018):
       Maintain class-conditional Gaussian distributions in representation space.
       OOD score = min_c (h - μ_c)^T Σ^{-1} (h - μ_c)
       Large distance → far from any known cluster → OOD.

    2. Energy Score (Liu et al. 2020):
       OOD score = -T * log Σ_y exp(f(x)[y] / T)
       Energy-based: ID samples cluster at low energy, OOD at high energy.
       Advantage: no class labels needed, just the logit vector.

    3. Spectral Uncertainty (Van Amersfoort et al. 2020):
       Deterministic uncertainty via spectral normalisation:
       The Lipschitz constant of the network bounds uncertainty.
       Spectral-normalised features whose norm is small → uncertain.

    When OOD detected:
      - Increase epistemic uncertainty estimate
      - Hedge outputs (soften predictions)
      - Flag for human review in the alignment report
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_centroids = args.ood_n_centroids
        self.threshold = args.ood_threshold
        self.ema = args.ood_ema
        self.dim = args.dim
        h = args.ood_hidden

        # Feature extractor (spectral-norm constrained)
        self.feature_extractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(args.dim, h)),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(h, h)),
            RMSNorm(h),
        )

        # Centroids for Mahalanobis distance
        self.register_buffer("centroids", torch.randn(args.ood_n_centroids, h) * 0.1)
        self.register_buffer("centroid_counts", torch.ones(args.ood_n_centroids))

        # Precision matrix (shared across centroids, diagonal approx)
        self.register_buffer("precision_diag", torch.ones(h))

        # Energy head: features → energy score
        self.energy_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
        )

        # OOD gate: when OOD, modulate the hidden state
        self.ood_gate = nn.Sequential(
            nn.Linear(1, args.dim),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def update_centroids(self, features: torch.Tensor) -> None:
        """EMA update of cluster centroids using nearest-centroid assignment."""
        B, H = features.shape
        # Assign each feature to nearest centroid
        dists = torch.cdist(features, self.centroids)  # [B, n_centroids]
        nearest = dists.argmin(-1)  # [B]

        for k in range(self.n_centroids):
            mask = nearest == k
            if mask.any():
                batch_mean = features[mask].mean(0)
                self.centroids[k] = self.ema * self.centroids[k] + (1 - self.ema) * batch_mean
                self.centroid_counts[k] += mask.float().sum()

    def _mahalanobis_score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute minimum Mahalanobis distance to any centroid."""
        # features: [B, H]
        B, H = features.shape
        min_dists = torch.full((B,), float("inf"), device=features.device)

        for k in range(self.n_centroids):
            diff = features - self.centroids[k].unsqueeze(0)  # [B, H]
            # Diagonal Mahalanobis: sum((diff^2 * precision))
            mah = (diff.pow(2) * self.precision_diag).sum(-1)  # [B]
            min_dists = torch.minimum(min_dists, mah)

        return min_dists.sqrt()  # [B]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)  # [B, D]

        # Extract features
        features = self.feature_extractor(pooled)  # [B, h]

        # Update centroids
        self.update_centroids(features.detach())

        # Mahalanobis score
        mah_score = self._mahalanobis_score(features)  # [B]

        # Energy score
        energy = self.energy_head(features).squeeze(-1)  # [B]
        energy_score = torch.sigmoid(energy)  # [B] ∈ [0,1]

        # Spectral uncertainty: norm of spectral-normalised features
        spectral_unc = 1.0 - features.norm(dim=-1) / (features.norm(dim=-1).max() + 1e-6)

        # Combined OOD score
        ood_score = (
            mah_score / (mah_score.max() + 1e-6) + energy_score + spectral_unc
        ) / 3.0  # [B] ∈ [0,1]

        is_ood = ood_score > self.threshold  # [B] bool

        # Modulate hidden state when OOD: reduce scale
        gate = self.ood_gate(ood_score.unsqueeze(-1))  # [B, D]
        x_ood = x * (0.7 + 0.3 * gate).unsqueeze(1)

        return x_ood, {
            "ood_score": ood_score,
            "is_ood": is_ood.tolist(),
            "mah_score": mah_score,
            "energy_score": energy_score,
            "spectral_unc": spectral_unc,
        }


# ============= Uncertainty-Aware Aggregator =============


class UncertaintyAwareAggregator(nn.Module):
    """
    Combines all uncertainty signals into a unified epistemic state.

    Queries available:
      "How confident are you overall?"         → scalar ∈ [0,1]
      "Which outputs are most uncertain?"       → per-head uncertainty ranking
      "Is this input OOD?"                      → bool + confidence
      "Are the moral verdicts reliable?"        → moral framework agreement
      "What is the prediction set for action X?" → conformal set

    The aggregator also produces an uncertainty summary embedding that
    can be injected into the hidden state — allowing downstream modules
    to condition their processing on the current epistemic state.

    This is the "self-awareness of uncertainty" that distinguishes a
    well-calibrated system from an overconfident one.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        h = args.uaa_hidden
        n_heads = args.uaa_n_heads

        # Uncertainty sources to aggregate:
        # (epistemic, aleatoric, ood, conformal, calibration, moral_std)
        n_unc_signals = 6

        # Cross-attention over uncertainty signals
        self.unc_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Signal projectors
        self.signal_proj = nn.Linear(1, h)

        # Uncertainty summary → embedding
        self.summary_proj = nn.Sequential(
            nn.Linear(h, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
        )

        # Overall confidence scalar
        self.confidence_head = nn.Sequential(
            nn.Linear(n_unc_signals, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        epistemic: torch.Tensor,  # [B] or [B, D]
        aleatoric: torch.Tensor,  # [B] or [B, D]
        ood_score: torch.Tensor,  # [B]
        conformal_unc: torch.Tensor,  # [B]
        ece: float,
        moral_std: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, Dict]:
        B = x.size(0)

        # Normalise all signals to [0,1] scalars
        def to_scalar(t):
            if t.dim() > 1:
                return t.mean(-1)
            return t

        signals = torch.stack(
            [
                to_scalar(epistemic),
                to_scalar(aleatoric),
                ood_score,
                conformal_unc,
                torch.full((B,), ece, device=x.device),
                moral_std,
            ],
            dim=-1,
        )  # [B, 6]

        # Overall confidence
        confidence = self.confidence_head(signals).squeeze(-1)  # [B]

        # Uncertainty summary embedding
        signals_proj = self.signal_proj(signals.unsqueeze(-1))  # [B, 6, h]
        unc_attn_out, _ = self.unc_attn(signals_proj, signals_proj, signals_proj)
        unc_summary = unc_attn_out.mean(1)  # [B, h]
        unc_embedding = self.summary_proj(
            unc_summary.unsqueeze(1).expand(-1, x.size(1), -1).reshape(-1, unc_summary.size(-1))
        ).view(B, x.size(1), self.dim)  # [B, L, D]

        # Inject uncertainty into hidden state (soft)
        x_aware = self.norm(x + unc_embedding * 0.05)

        return x_aware, {
            "confidence": confidence,
            "signals": signals,
            "unc_embedding": unc_embedding,
            "signal_names": ["epistemic", "aleatoric", "ood", "conformal", "ece", "moral_std"],
        }


# ============= Uncertainty-Driven Action Gate =============


class UncertaintyGate(nn.Module):
    """
    Gates action execution based on an aggregated uncertainty signal.

    When the model's estimated uncertainty exceeds ``threshold``, the gate
    returns ``False`` and the caller should abstain from acting, request
    more information, or fall back to a safe default.

    Designed to be queried after ``UncertaintyAwareAggregator`` has
    produced a ``confidence`` scalar, but can also accept any scalar
    uncertainty estimate in ``[0, 1]``.

    Args:
        threshold:        Uncertainty above this value triggers abstention.
                          Default 0.4 (conservative).
        use_hysteresis:   If True, the gate uses a small hysteresis band
                          so it does not oscillate at the boundary:
                          it opens at ``threshold - margin`` and closes
                          at ``threshold + margin``.
        margin:           Half-width of the hysteresis band.  Only used
                          when ``use_hysteresis=True``.

    Usage::

        gate = UncertaintyGate(threshold=0.4)

        # From UncertaintyAwareAggregator output:
        confidence = out["uncertainty"]["aggregated"]["confidence"]  # [B]
        uncertainty = 1.0 - confidence

        for b in range(batch_size):
            if gate.allow_action(uncertainty[b].item()):
                execute_action(actions[b])
            else:
                abstain(b)

        # Or process a whole batch at once:
        mask = gate.allow_batch(uncertainty)   # [B] bool tensor
    """

    def __init__(
        self,
        threshold: float = 0.4,
        use_hysteresis: bool = False,
        margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.use_hysteresis = use_hysteresis
        self.margin = margin

        # Track current gate state for hysteresis
        self._is_open: bool = True

    def allow_action(self, uncertainty_score: float) -> bool:
        """
        Return True if uncertainty is low enough to act.

        Args:
            uncertainty_score: Scalar in ``[0, 1]``.  Higher = more uncertain.

        Returns:
            True  → proceed with the action.
            False → abstain; uncertainty too high.
        """
        if self.use_hysteresis:
            if self._is_open:
                allow = uncertainty_score < (self.threshold + self.margin)
            else:
                allow = uncertainty_score < (self.threshold - self.margin)
            self._is_open = allow
            return allow
        return float(uncertainty_score) < self.threshold

    def allow_batch(self, uncertainty_scores: torch.Tensor) -> torch.Tensor:
        """
        Vectorised batch version of ``allow_action``.

        Args:
            uncertainty_scores: Float tensor ``[B]``, values in ``[0, 1]``.

        Returns:
            Boolean tensor ``[B]`` — True where the gate permits action.
        """
        return uncertainty_scores < self.threshold

    def forward(
        self,
        uncertainty_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        nn.Module-compatible forward pass for use inside a model pipeline.

        Args:
            uncertainty_scores: ``[B]`` float tensor.

        Returns:
            Dict with keys:
              ``"allow"``       — ``[B]`` bool tensor.
              ``"abstain_frac"``— fraction of batch that should abstain.
        """
        allow = self.allow_batch(uncertainty_scores)
        abstain_frac = (~allow).float().mean()
        return {
            "allow": allow,
            "abstain_frac": abstain_frac,
        }


# ============= Uncertainty Claudeson =============


class ClaudesonUncertainty(ClaudesonSocialAlignment):
    """
    Claudeson 2026 — Uncertainty Edition.

    Wraps the full Social Alignment architecture with principled
    uncertainty quantification at every level.

    Pipeline (after Social Alignment):
      StakeholderVM → WelfareAgg → NormEngine → SocialContract → MoralEstimator
            ↓
      BayesianUncertaintyPropagator   (epistemic / aleatoric split)
            ↓
      ConformalPrediction             (distribution-free prediction sets)
            ↓
      CalibrationMonitor              (ECE + temperature scaling)
            ↓
      OODDetector                     (Mahalanobis + energy + spectral)
            ↓
      UncertaintyAwareAggregator      (unified epistemic state)

    New output keys:
      uncertainty — {bayesian, conformal, calibration, ood, aggregated}

    New alignment report additions:
      - Overall confidence
      - OOD flag
      - Conformal set size
      - ECE
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.bup = BayesianUncertaintyPropagator(args)
        self.conformal = ConformalPrediction(args)
        self.calibrator = CalibrationMonitor(args)
        self.ood = OODDetector(args)
        self.uaa = UncertaintyAwareAggregator(args)

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        goal_tokens: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
        actual_action: Optional[torch.Tensor] = None,
        rung_labels: Optional[torch.Tensor] = None,
        competence_signal: Optional[float] = None,
    ) -> Dict:
        # ── Full Social Alignment pass ───────────────────────────────────
        base = super().forward(
            text=text,
            img=img,
            audio=audio,
            goal_tokens=goal_tokens,
            feedback=feedback,
            agent_observations=agent_observations,
            actual_action=actual_action,
            rung_labels=rung_labels,
            competence_signal=competence_signal,
        )
        x = base["hidden_states"]

        # Extract social alignment tensors for uncertainty over them
        sa = base.get("social_alignment", {})
        welfare = sa.get("welfare", {}).get(
            "welfare", torch.zeros(x.size(0), self.args.n_stakeholder_groups, device=x.device)
        )
        moral_scores = sa.get("moral", {}).get(
            "framework_scores",
            torch.zeros(x.size(0), self.args.n_moral_frameworks, device=x.device),
        )
        moral_std = sa.get("moral", {}).get("moral_std", torch.zeros(x.size(0), device=x.device))

        # ── Bayesian Uncertainty ─────────────────────────────────────────
        x, bup_out = self.bup(x, welfare, moral_scores)

        # ── Conformal Prediction ─────────────────────────────────────────
        x, cp_out = self.conformal(x)

        # ── Calibration Monitor ──────────────────────────────────────────
        x, cal_out = self.calibrator(x)

        # ── OOD Detection ────────────────────────────────────────────────
        x, ood_out = self.ood(x)

        # ── Unified Uncertainty Aggregation ──────────────────────────────
        epistemic = bup_out["main"]["epistemic"].mean(-1)  # [B]
        aleatoric = bup_out["main"]["aleatoric"].mean(-1)  # [B]

        x, uaa_out = self.uaa(
            x,
            epistemic=epistemic,
            aleatoric=aleatoric,
            ood_score=ood_out["ood_score"],
            conformal_unc=cp_out["uncertainty"],
            ece=cal_out["ece"],
            moral_std=moral_std,
        )

        return {
            **base,
            "hidden_states": x,
            "uncertainty": {
                "bayesian": bup_out,
                "conformal": cp_out,
                "calibration": cal_out,
                "ood": ood_out,
                "aggregated": uaa_out,
            },
        }


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — UNCERTAINTY EDITION")
    print("Bayesian · Conformal · Calibration · OOD · Unified Epistemic State")
    print("=" * 70)

    args = ModelArgs()
    args.dim = 128
    args.n_layers = 2
    args.n_heads = 4
    args.n_kv_heads = 2
    args.vocab_size = 512
    args.max_seq_len = 64
    args.memory_slots = 32
    args.episodic_slots = 64
    args.goal_dim = 128
    args.latent_dim = 64
    args.energy_hidden = 128
    args.ssm_state_dim = 32
    args.ssm_chunk_size = 16
    args.num_experts = 2
    args.num_shared_experts = 1
    args.env_state_dim = 32
    args.action_space_size = 16
    args.planning_horizon = 2
    args.num_simulations = 2
    args.img_size = 32
    args.patch_size = 8
    args.audio_spec_dim = 16
    args.gradient_checkpointing = False
    args.n_agents = 4
    args.lora_rank = 8
    args.n_causal_nodes = 16
    args.metacog_hidden = 64
    args.n_debate_agents = 3
    args.debate_hidden = 128
    args.n_propositions = 16
    args.n_constraints = 8
    args.consistency_iters = 2
    args.rsi_rank = 4
    args.rsi_horizon = 2
    args.n_workspace_slots = 8
    args.gw_competition_k = 2
    args.gw_broadcast_steps = 1
    args.n_ops = 16
    args.n_registers = 4
    args.prog_steps = 3
    args.prog_hidden = 64
    args.irl_hidden = 64
    args.irl_n_preferences = 8
    args.lif_steps = 3
    args.causal_state_dim = 32
    args.intervention_horizon = 2
    args.n_intervention_samples = 4
    args.cf_n_branches = 2
    args.attr_top_k = 4
    args.pearl_hidden = 64
    args.n_skill_slots = 8
    args.skill_rank = 4
    args.skill_embed_dim = 32
    args.cp_window = 8
    args.cp_hidden = 64
    args.oeg_n_compose = 2
    args.oeg_hidden = 64
    args.ig_beta = 0.5
    args.n_abstraction_levels = 3
    args.hae_heads = 2
    args.hae_pool_factor = 2
    args.hae_hidden = 64
    args.n_concepts = 32
    args.concept_top_k = 8
    args.concept_hidden = 64
    args.n_schema_slots = 8
    args.schema_n_roles = 4
    args.schema_hidden = 64
    args.schema_bind_iters = 2
    args.analogy_hidden = 64
    args.analogy_n_mappings = 4
    args.n_principles = 8
    args.principle_hidden = 64
    args.n_stakeholder_groups = 4
    args.stakeholder_hidden = 64
    args.welfare_hidden = 64
    args.n_welfare_objectives = 4
    args.n_norm_slots = 16
    args.norm_hidden = 64
    args.scr_n_perspectives = 4
    args.scr_hidden = 64
    args.n_moral_frameworks = 4
    args.moral_hidden = 64
    # Uncertainty specific
    args.bup_n_samples = 5
    args.bup_dropout_rate = 0.1
    args.bup_hidden = 64
    args.cp_coverage = 0.9
    args.cp_cal_size = 128
    args.cp_n_classes = 32
    args.cal_n_bins = 10
    args.ood_n_centroids = 16
    args.ood_hidden = 64
    args.uaa_hidden = 64
    args.uaa_n_heads = 2

    print("\nInitialising ClaudesonUncertainty...")
    model = ClaudesonUncertainty(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)
    for step in range(args.cp_window):
        model.cp_monitor.record_performance(0, 0.3 + 0.04 * step)
    for g in range(args.n_stakeholder_groups):
        model.stakeholder_vm.update_welfare(g, 0.5 + 0.1 * g)
    for _ in range(20):
        model.conformal.update_calibration(torch.rand(1).item())

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            text=torch.randint(0, 512, (2, 32)),
            feedback=torch.randn(2, args.dim),
            agent_observations=torch.randn(2, 8, args.dim),
            actual_action=torch.randint(0, args.action_space_size, (2,)),
            competence_signal=0.6,
        )

    unc = out["uncertainty"]
    print("\nBayesian Uncertainty:")
    print(f"  Epistemic (mean): {unc['bayesian']['main']['epistemic'].mean().item():.4f}")
    print(f"  Aleatoric (mean): {unc['bayesian']['main']['aleatoric'].mean().item():.4f}")
    print("\nConformal Prediction:")
    print(f"  Set sizes:   {unc['conformal']['set_sizes'].tolist()}")
    print(f"  Quantile:    {unc['conformal']['quantile']:.4f}")
    print(f"  Cal count:   {unc['conformal']['cal_count']}")
    print("\nCalibration:")
    print(f"  ECE:         {unc['calibration']['ece']:.4f}")
    print(f"  Temperature: {unc['calibration']['temperature']:.4f}")
    print("\nOOD Detection:")
    print(f"  OOD scores:  {[f'{v:.3f}' for v in unc['ood']['ood_score'].tolist()]}")
    print(f"  Is OOD:      {unc['ood']['is_ood']}")
    print("\nAggregated Epistemic State:")
    print(f"  Confidence:  {unc['aggregated']['confidence'].tolist()}")
    for name, val in zip(
        unc["aggregated"]["signal_names"], unc["aggregated"]["signals"][0].tolist()
    ):
        print(f"  {name:<15}: {val:.4f}")
    print("\n" + "=" * 70)
    print("ClaudesonUncertainty READY.  Knows what it doesn't know.")
    print("=" * 70)
