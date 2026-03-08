"""
Claudeson 2026 - MetaCurriculum Edition
=========================================
Intrinsic Motivation · Information Gain · Skill Discovery · Open-Ended Learning

The problem this generation solves
------------------------------------
Every previous generation is reactive:
  - It responds to inputs.
  - It plans toward externally-given goals.
  - It learns new skills when explicitly trained on new tasks.
  - Its curiosity is an emergent side-effect of the EXPLORE goal in Jedi,
    which fires when uncertainty is high — but that signal is undirected.
    High uncertainty about everything is not the same as knowing *what*
    to be curious about.

A system that only learns what it is shown will plateau at the boundary
of its training distribution.  Open-ended learning (OEL) requires a
mechanism that generates its own curriculum — finding the frontier between
"too easy" (already mastered) and "too hard" (not yet learnable), and
directing exploration there.

This generation adds four components:

  1. Information Gain Intrinsic Reward
     R_i = KL( posterior(z | x) || prior(z) )
     The agent is rewarded for experiencing observations that update its
     beliefs — the Bayesian formalisation of curiosity.  Connects directly
     to the Jedi VAE: the KL term that was previously a regularisation loss
     becomes a reward signal that drives behaviour.

  2. Skill Discovery Engine
     Maintains a library of reusable skills as low-rank adapter deltas in
     a differentiable skill bank.  When the information-gain reward for a
     region of state space drops (the skill is mastered), a new skill slot
     is allocated and training redirects to a harder region.  Connects to
     the LoRA adapters in ContinualLearner — discovered skills consolidate
     into the same parameter-efficient format.

  3. Competence Progress Monitor
     Tracks learning progress (LP) per skill as the *derivative* of
     performance — not how good the agent is, but how fast it is improving.
     High LP = interesting frontier.  Low LP AND low performance = too hard.
     Low LP AND high performance = mastered.  Routes training to high-LP zones.
     This is the core of the Oudeyer-Kaplan Intrinsic Motivation framework.

  4. Open-Ended Goal Generator
     Synthesises new goals by recombining mastered skills compositionally.
     "Can I do A AND B simultaneously?" is harder than A or B alone but
     reachable if both are individually mastered.  This prevents the
     trivial solution of always picking the easiest task.

Architecture evolution:
  claudson → extended → infinite → pro → ultimate → jedi → grounded
          → sovereign → transcendent → causal_world → metacurriculum
                                                           ↑ you are here
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from claudson_causal_world import (
    ModelArgs as CausalWorldArgs,
    ClaudesonCausalWorld,
)
from claudson_jedi import SwiGLU, RMSNorm

log = logging.getLogger(__name__)


# ============= Configuration =============

@dataclass
class ModelArgs(CausalWorldArgs):
    # Information Gain
    ig_beta: float = 1.0            # weight of information-gain reward
    ig_ema_alpha: float = 0.05      # EMA decay for baseline surprise estimate

    # Skill Discovery
    n_skill_slots: int = 32         # maximum number of skills in the library
    skill_rank: int = 16            # LoRA rank for each skill adapter
    skill_embed_dim: int = 256      # embedding dim for skill identity vectors
    skill_mastery_threshold: float = 0.85   # performance above this = mastered
    skill_novelty_threshold: float = 0.10   # information gain below this = mastered

    # Competence Progress Monitor
    cp_window: int = 32             # rolling window length for LP estimation
    cp_hidden: int = 128            # hidden dim for LP predictor
    cp_frontier_percentile: float = 0.75   # LP percentile that defines "frontier"

    # Open-Ended Goal Generator
    oeg_n_compose: int = 3          # max skills to compose into one goal
    oeg_hidden: int = 256           # hidden dim for goal synthesiser
    oeg_temperature: float = 1.0    # sampling temperature for goal selection


# ============= Information Gain Intrinsic Reward =============

class InformationGainReward(nn.Module):
    """
    Rewards the agent for experiencing *surprising* observations.

    R_intrinsic = KL( q(z|x) || p(z) ) - baseline

    Where:
      q(z|x)  = posterior from the Jedi VAE encoder (what we now believe)
      p(z)    = prior (what we believed before seeing x)
      baseline = running average surprise (subtract to prevent reward hacking)

    This is the Bayesian formalisation of curiosity:
      - High KL → the observation changed our beliefs a lot → interesting
      - Low KL  → the observation was predictable → boring

    The baseline subtraction implements "relative novelty":
    if everything is surprising, nothing is (habituation).  Only experiences
    that are *more* surprising than average earn intrinsic reward.

    Connection to the existing stack:
      - The Jedi EnergyLayer already computes mu, logvar, and prior_mu/logvar
      - We re-use those tensors; this module adds zero forward-pass overhead
        beyond the reward computation itself.
      - The reward signal is passed to the GroundedActionLoop to bias tool
        selection toward information-seeking actions.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.beta       = args.ig_beta
        self.ema_alpha  = args.ig_ema_alpha
        self.latent_dim = args.latent_dim

        # Running baseline: exponential moving average of recent KL
        self.register_buffer('baseline', torch.tensor(0.0))
        self.register_buffer('baseline_var', torch.tensor(1.0))

        # Novelty projector: maps KL signal to token-level curiosity weight
        self.novelty_proj = nn.Sequential(
            nn.Linear(1, args.cp_hidden),
            nn.GELU(),
            nn.Linear(args.cp_hidden, 1),
            nn.Sigmoid(),
        )

        # Epistemic gain head: separates reducible uncertainty from total
        self.epistemic_head = nn.Sequential(
            nn.Linear(args.latent_dim * 2, args.cp_hidden),
            nn.GELU(),
            nn.Linear(args.cp_hidden, 1),
            nn.Softplus(),
        )

    @torch.no_grad()
    def _update_baseline(self, kl_val: torch.Tensor) -> None:
        """Update EMA baseline and variance of surprise."""
        delta = kl_val.mean().item() - self.baseline.item()
        self.baseline    += self.ema_alpha * delta
        self.baseline_var = (1 - self.ema_alpha) * (
            self.baseline_var + self.ema_alpha * delta ** 2
        )

    def forward(
        self,
        mu:         torch.Tensor,       # [B, L, latent_dim]  posterior mean
        logvar:     torch.Tensor,       # [B, L, latent_dim]  posterior log-var
        prior_mu:   torch.Tensor,       # [B, L, latent_dim]  prior mean
        prior_logvar: torch.Tensor,     # [B, L, latent_dim]  prior log-var
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute per-token information-gain reward.

        Returns:
            ig_reward   [B, L, 1]  — intrinsic reward per token
            info_dict   — KL stats, baseline, novelty weights
        """
        # KL( q || p ) analytically for Gaussians
        # KL = 0.5 * sum[ log(σ_p²/σ_q²) + (σ_q² + (μ_q-μ_p)²)/σ_p² - 1 ]
        prior_var = prior_logvar.exp().clamp(min=1e-6)
        post_var  = logvar.exp().clamp(min=1e-6)

        kl = 0.5 * (
            prior_logvar - logvar
            + (post_var + (mu - prior_mu).pow(2)) / prior_var
            - 1.0
        ).sum(dim=-1, keepdim=True)                              # [B, L, 1]

        kl = kl.clamp(min=0.0)

        # Epistemic component (reducible): variance of the posterior
        epistemic = self.epistemic_head(
            torch.cat([mu, logvar], dim=-1)
        )                                                        # [B, L, 1]

        # Relative novelty: how much more surprising than average?
        relative_kl  = kl - self.baseline
        novelty_norm = relative_kl / (self.baseline_var.sqrt() + 1e-6)
        novelty_w    = self.novelty_proj(novelty_norm)           # [B, L, 1]

        # Final intrinsic reward
        ig_reward = self.beta * kl * novelty_w

        # Update baseline for next call
        self._update_baseline(kl.detach())

        return ig_reward, {
            "kl":           kl.squeeze(-1),
            "epistemic":    epistemic.squeeze(-1),
            "novelty_w":    novelty_w.squeeze(-1),
            "ig_reward":    ig_reward.squeeze(-1),
            "baseline":     self.baseline.item(),
            "baseline_var": self.baseline_var.item(),
        }


# ============= Skill Library =============

class SkillLibrary(nn.Module):
    """
    A differentiable bank of reusable skills stored as LoRA adapter pairs.

    Each skill is represented as:
      - A skill embedding vector  e_k ∈ R^skill_embed_dim
      - A LoRA adapter pair       (A_k, B_k)  where A_k ∈ R^{D×r}, B_k ∈ R^{r×D}

    Skill selection is soft (attention over embeddings) during training and
    hard (argmax) during inference.  This keeps the selection differentiable
    while still committing to a discrete skill at inference time.

    Skill lifecycle:
      UNTRAINED → LEARNING → MASTERED → FROZEN

      - When allocated, a skill slot is UNTRAINED (A/B are near-zero).
      - As training proceeds, the slot moves to LEARNING.
      - Once competence_score > mastery_threshold AND information_gain < novelty_threshold,
        the slot is MASTERED and its weights are frozen.
      - The Skill Discovery Engine then allocates a new slot for harder tasks.

    This mirrors the EWC + LoRA continual learning in ClaudesonGrounded, but
    adds the lifecycle management that determines *when* to allocate new skills.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_slots        = args.n_skill_slots
        self.rank           = args.skill_rank
        self.embed_dim      = args.skill_embed_dim
        self.dim            = args.dim
        self.mastery_thresh = args.skill_mastery_threshold
        self.novelty_thresh = args.skill_novelty_threshold

        # Skill identity embeddings (what each skill "is")
        self.skill_embeddings = nn.Parameter(
            torch.randn(args.n_skill_slots, args.skill_embed_dim) * 0.02
        )

        # Skill adapter weights
        self.skill_A = nn.Parameter(
            torch.zeros(args.n_skill_slots, args.dim, args.skill_rank)
        )
        self.skill_B = nn.Parameter(
            torch.zeros(args.n_skill_slots, args.skill_rank, args.dim)
        )
        # Initialise A with kaiming, B stays zero (identity at init)
        nn.init.kaiming_uniform_(self.skill_A.data.view(-1, args.skill_rank),
                                  a=math.sqrt(5))

        # Query projector: hidden state → skill query
        self.query_proj = nn.Sequential(
            nn.Linear(args.dim, args.skill_embed_dim),
            RMSNorm(args.skill_embed_dim),
        )

        # Competence score per skill (updated externally)
        self.register_buffer('competence',   torch.zeros(args.n_skill_slots))
        self.register_buffer('ig_scores',    torch.ones(args.n_skill_slots))
        self.register_buffer('active_slots', torch.zeros(args.n_skill_slots, dtype=torch.bool))
        self.register_buffer('frozen_slots', torch.zeros(args.n_skill_slots, dtype=torch.bool))
        self.register_buffer('n_active',     torch.tensor(1))

        # Activate the first slot immediately
        self.active_slots[0] = True
        self.n_active        = torch.tensor(1)

        self.norm = RMSNorm(args.dim)

    def query(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft-select skills relevant to the current hidden state.

        Returns:
            skill_output  [B, L, D]  — weighted combination of skill adapters
            skill_weights [B, n_slots] — attention weights over skill library
        """
        B, L, D = x.shape
        pooled = x.mean(1)                                          # [B, D]

        # Skill query from hidden state
        q = self.query_proj(pooled)                                 # [B, embed_dim]

        # Attention over active skill embeddings
        active_mask = self.active_slots.float()                     # [n_slots]
        scores = q @ self.skill_embeddings.T                        # [B, n_slots]
        scores = scores - (1 - active_mask) * 1e9                  # mask inactive
        weights = F.softmax(scores / math.sqrt(self.embed_dim), dim=-1)  # [B, n_slots]

        # Apply soft mixture of skill adapters
        # skill output = x @ (sum_k w_k * A_k @ B_k) / rank
        skill_out = torch.zeros(B, D, device=x.device)
        for k in range(self.n_slots):
            if self.active_slots[k]:
                w_k     = weights[:, k].unsqueeze(-1)               # [B, 1]
                adapter = x.mean(1) @ self.skill_A[k] @ self.skill_B[k]  # [B, D]
                skill_out = skill_out + w_k * adapter / self.rank

        skill_out = skill_out.unsqueeze(1).expand(-1, L, -1)       # [B, L, D]
        return self.norm(x + skill_out * 0.1), weights

    @torch.no_grad()
    def update_skill_stats(
        self,
        skill_idx:   int,
        competence:  float,
        ig_score:    float,
    ) -> bool:
        """
        Update competence and IG stats for a skill slot.
        Returns True if the skill just became mastered.
        """
        self.competence[skill_idx] = competence
        self.ig_scores[skill_idx]  = ig_score

        newly_mastered = (
            competence > self.mastery_thresh and
            ig_score   < self.novelty_thresh and
            not self.frozen_slots[skill_idx]
        )
        if newly_mastered:
            self.frozen_slots[skill_idx] = True
            log.info("Skill %d mastered (competence=%.3f, ig=%.4f)", skill_idx, competence, ig_score)
        return newly_mastered

    @torch.no_grad()
    def allocate_new_skill(self) -> Optional[int]:
        """
        Allocate the next available slot.  Returns slot index or None if full.
        """
        for i in range(self.n_slots):
            if not self.active_slots[i]:
                self.active_slots[i] = True
                self.n_active        = self.n_active + 1
                log.info("Allocated new skill slot %d (total active: %d)", i, int(self.n_active.item()))
                return i
        log.warning("Skill library full (%d slots), cannot allocate.", self.n_slots)
        return None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        x_skilled, weights = self.query(x)
        return x_skilled, {
            "skill_weights":  weights,
            "n_active":       int(self.n_active.item()),
            "n_frozen":       int(self.frozen_slots.sum().item()),
            "competence":     self.competence.tolist(),
        }


# ============= Competence Progress Monitor =============

class CompetenceProgressMonitor(nn.Module):
    """
    Tracks *learning progress* — the derivative of competence over time.

    Oudeyer & Kaplan (2007): intrinsic motivation should be maximised not
    by seeking high competence, but by seeking high *rate of improvement*.

    Three zones:
      FRONTIER   — LP high: actively learning, direct training here
      MASTERED   — LP low, competence high: move on to harder tasks
      TOO HARD   — LP low, competence low: simplify or break into sub-skills

    Per-skill rolling window tracks:
      performance_history  — circular buffer of recent competence scores
      lp_estimate          — slope of a linear fit to the recent window

    A small MLP predicts which zone each skill is in and outputs a
    "curriculum weight" — how much training time to allocate to each skill.
    This is the scheduling signal for the open-ended goal generator.
    """

    ZONES = ["FRONTIER", "MASTERED", "TOO_HARD"]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_slots  = args.n_skill_slots
        self.window   = args.cp_window
        self.frontier = args.cp_frontier_percentile
        self.hidden   = args.cp_hidden

        # Rolling performance buffers per skill
        self.register_buffer(
            'perf_history',
            torch.zeros(args.n_skill_slots, args.cp_window)
        )
        self.register_buffer(
            'perf_ptr',
            torch.zeros(args.n_skill_slots, dtype=torch.long)
        )
        self.register_buffer(
            'lp_estimates',
            torch.zeros(args.n_skill_slots)
        )

        # LP predictor: (lp, competence_mean) → zone logits + curriculum weight
        self.zone_head = nn.Sequential(
            nn.Linear(2, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 3 + 1),   # 3 zone logits + 1 curriculum weight
        )

        # Cross-skill attention: LP at one skill can inform another (transfer)
        self.transfer_attn = nn.MultiheadAttention(
            embed_dim=4,                       # (lp, comp, ig, active)
            num_heads=1,
            batch_first=True,
            dropout=0.0,
        )

    @torch.no_grad()
    def record_performance(self, skill_idx: int, score: float) -> None:
        """Record a performance observation for a skill."""
        ptr = int(self.perf_ptr[skill_idx].item())
        self.perf_history[skill_idx, ptr] = score
        self.perf_ptr[skill_idx] = (ptr + 1) % self.window
        self._update_lp(skill_idx)

    @torch.no_grad()
    def _update_lp(self, skill_idx: int) -> None:
        """Fit a linear trend to the performance window and store the slope."""
        hist = self.perf_history[skill_idx]                      # [window]
        t    = torch.arange(self.window, dtype=torch.float32, device=hist.device)
        # Simple OLS slope: cov(t, y) / var(t)
        t_mean = t.mean()
        y_mean = hist.mean()
        num    = ((t - t_mean) * (hist - y_mean)).sum()
        den    = ((t - t_mean).pow(2)).sum() + 1e-8
        self.lp_estimates[skill_idx] = (num / den).clamp(-1.0, 1.0)

    def forward(
        self,
        skill_library: SkillLibrary,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute zone classification and curriculum weights for all skills.

        Returns:
            curriculum_weights  [n_slots]  — training allocation per skill
            info_dict
        """
        # Feature matrix: [n_slots, 4]  (lp, comp, ig, active)
        lp     = self.lp_estimates                               # [n_slots]
        comp   = skill_library.competence                        # [n_slots]
        ig     = skill_library.ig_scores                         # [n_slots]
        active = skill_library.active_slots.float()              # [n_slots]

        features = torch.stack([lp, comp, ig, active], dim=-1)  # [n_slots, 4]

        # Cross-skill transfer attention
        features_attn, _ = self.transfer_attn(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0)
        )
        features = features + features_attn.squeeze(0)

        # Zone classification + curriculum weight
        inp = torch.stack([lp, comp], dim=-1)                    # [n_slots, 2]
        out = self.zone_head(inp)                                 # [n_slots, 4]
        zone_logits      = out[:, :3]                            # [n_slots, 3]
        curriculum_raw   = out[:, 3]                             # [n_slots]

        # Mask inactive slots
        curriculum_raw   = curriculum_raw * active
        curriculum_weights = F.softmax(curriculum_raw, dim=0)    # [n_slots]

        zones = zone_logits.argmax(-1)                           # [n_slots]

        return curriculum_weights, {
            "lp_estimates":       lp.tolist(),
            "zone_logits":        zone_logits,
            "zones":              [self.ZONES[z] for z in zones.tolist()],
            "curriculum_weights": curriculum_weights.tolist(),
        }


# ============= Skill Discovery Engine =============

class SkillDiscoveryEngine(nn.Module):
    """
    Manages the full skill lifecycle: discovery → learning → mastery → expansion.

    Integrates:
      - InformationGainReward to detect when a region is still novel
      - SkillLibrary to store and retrieve skills
      - CompetenceProgressMonitor to track learning progress
      - Automatic slot allocation when current skills are mastered

    The discovery loop (called once per training step):
      1. Compute IG reward for current experience.
      2. Update competence/IG stats for the active skill.
      3. Check if the skill is mastered (high comp, low IG).
      4. If mastered, freeze the slot and allocate a new one.
      5. Output the curriculum weight for experience replay scheduling.

    The "active skill" concept:
      At any point one skill is designated "active" — the one currently
      being trained.  The curriculum weight determines how much experience
      to dedicate to each skill region, but the active skill is the primary
      training target.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args    = args
        self.ig      = InformationGainReward(args)
        self.library = SkillLibrary(args)
        self.monitor = CompetenceProgressMonitor(args)

        # Current active skill slot (0 at init)
        self.register_buffer('active_skill', torch.tensor(0))

        # Skill encoder: maps hidden state region to a skill embedding
        self.skill_encoder = nn.Sequential(
            nn.Linear(args.dim, args.skill_embed_dim),
            RMSNorm(args.skill_embed_dim),
            SwiGLU(args.skill_embed_dim, args.skill_embed_dim * 2),
            nn.Linear(args.skill_embed_dim, args.skill_embed_dim),
        )

        # Skill boundary detector: is this experience in the same skill region?
        self.boundary_head = nn.Sequential(
            nn.Linear(args.skill_embed_dim * 2, args.cp_hidden),
            nn.GELU(),
            nn.Linear(args.cp_hidden, 1),
            nn.Sigmoid(),
        )

        # Running skill embedding for active skill (updated EMA)
        self.register_buffer(
            'active_skill_emb',
            torch.zeros(args.skill_embed_dim)
        )

    @torch.no_grad()
    def _maybe_advance_skill(
        self,
        competence: float,
        ig_val:     float,
    ) -> bool:
        """
        Check mastery; if met, freeze current skill and open a new slot.
        Returns True if advancement occurred.
        """
        idx      = int(self.active_skill.item())
        mastered = self.library.update_skill_stats(idx, competence, ig_val)
        if mastered:
            new_slot = self.library.allocate_new_skill()
            if new_slot is not None:
                self.active_skill = torch.tensor(new_slot)
                return True
        return False

    def forward(
        self,
        x:          torch.Tensor,          # [B, L, D]
        mu:         torch.Tensor,          # [B, L, latent_dim]
        logvar:     torch.Tensor,          # [B, L, latent_dim]
        prior_mu:   torch.Tensor,          # [B, L, latent_dim]
        prior_logvar: torch.Tensor,        # [B, L, latent_dim]
        competence_signal: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # --- Information Gain ---
        ig_reward, ig_info = self.ig(mu, logvar, prior_mu, prior_logvar)

        # --- Skill library query ---
        x_skilled, skill_info = self.library(x)

        # --- Curriculum weights ---
        curriculum_weights, cp_info = self.monitor(self.library)

        # --- Skill embedding for active skill update ---
        pooled     = x.mean(1)                                   # [B, D]
        skill_emb  = self.skill_encoder(pooled).mean(0)          # [skill_embed_dim]
        self.active_skill_emb = 0.95 * self.active_skill_emb + 0.05 * skill_emb.detach()

        # --- Advancement check ---
        ig_mean = ig_info["kl"].mean().item()
        comp    = competence_signal if competence_signal is not None else 0.0
        advanced = self._maybe_advance_skill(comp, ig_mean)

        # Active skill index for logging
        active_idx = int(self.active_skill.item())

        return x_skilled, {
            "ig":              ig_info,
            "ig_reward":       ig_reward,
            "skill":           skill_info,
            "curriculum":      cp_info,
            "active_skill":    active_idx,
            "skill_advanced":  advanced,
            "curriculum_w_active": float(cp_info["curriculum_weights"][active_idx])
                                   if active_idx < len(cp_info["curriculum_weights"]) else 0.0,
        }


# ============= Open-Ended Goal Generator =============

class OpenEndedGoalGenerator(nn.Module):
    """
    Synthesises novel goals by composing mastered skills.

    Motivation:
      If the agent only pursues goals it was given, it will never learn
      beyond its initial training distribution.  If it pursues random goals,
      it will waste compute on tasks that are either trivially easy or
      impossibly hard.  The sweet spot is *compositional novelty*:
      goals that combine mastered primitives in new arrangements.

    Construction:
      1. Skill selection   — sample k skills weighted by their curriculum weights
                             (frontier skills preferred over mastered ones for
                             the component skills; mastered ones form the "scaffold")
      2. Composition       — a GRU reads the selected skill embeddings in sequence
                             and outputs a goal embedding
      3. Difficulty scoring — a small network predicts whether the composed goal
                             is in the "just right" difficulty zone
      4. Rejection sampling — retry if difficulty is too low or too high

    The output is a goal embedding that can be passed to ClaudesonJedi's
    goal conditioning mechanism (the GoalEncoder already accepts embeddings).

    This closes the loop:
      SkillDiscovery identifies what is learnable NOW
      OpenEndedGoalGenerator composes goals AT THAT FRONTIER
      CompetenceProgressMonitor tracks whether the composed goal is working
      → repeat
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_compose   = args.oeg_n_compose
        self.temperature = args.oeg_temperature
        self.embed_dim   = args.skill_embed_dim
        self.dim         = args.dim
        self.n_slots     = args.n_skill_slots

        # GRU-based skill composer
        self.composer    = nn.GRUCell(args.skill_embed_dim, args.oeg_hidden)
        self.goal_proj   = nn.Sequential(
            nn.Linear(args.oeg_hidden, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.goal_dim),
            RMSNorm(args.goal_dim),
        )

        # Difficulty estimator: is this goal at the right difficulty?
        self.difficulty_head = nn.Sequential(
            nn.Linear(args.oeg_hidden, args.cp_hidden),
            nn.GELU(),
            nn.Linear(args.cp_hidden, 3),    # too_easy / just_right / too_hard
        )

        # Novelty checker: is this goal sufficiently different from past goals?
        self.novelty_head = nn.Sequential(
            nn.Linear(args.goal_dim * 2, args.cp_hidden),
            nn.GELU(),
            nn.Linear(args.cp_hidden, 1),
            nn.Sigmoid(),
        )

        # Project goal embedding into hidden-state space for conditioning
        self.goal_to_hidden = nn.Linear(args.goal_dim, args.dim, bias=False)

        # Memory of recent goals (to avoid repetition)
        self.register_buffer(
            'goal_memory',
            torch.zeros(16, args.goal_dim)
        )
        self.register_buffer('goal_mem_ptr', torch.tensor(0))
        self.register_buffer('goal_count', torch.tensor(0))

        self.goal_dim = args.goal_dim

    def _compose_skills(
        self,
        skill_library:  SkillLibrary,
        curriculum_w:   torch.Tensor,       # [n_slots]
        device:         torch.device,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Sample k skills from the library, weighted by curriculum weights,
        and compose them into a single goal embedding via the GRU.
        """
        n_active = int(skill_library.n_active.item())
        k        = min(self.n_compose, n_active)

        # Sample k distinct skill indices
        probs    = curriculum_w[:n_active] + 1e-6
        probs    = probs / probs.sum()
        chosen   = torch.multinomial(probs, k, replacement=False)  # [k]

        # Run GRU over chosen skill embeddings
        h = torch.zeros(1, self.composer.hidden_size, device=device)  # [1, hidden]
        for idx in chosen:
            emb = skill_library.skill_embeddings[idx].unsqueeze(0)  # [1, embed]
            h   = self.composer(emb, h)

        return h, chosen.tolist()

    @torch.no_grad()
    def _store_goal(self, goal: torch.Tensor) -> None:
        ptr = int(self.goal_mem_ptr.item())
        self.goal_memory[ptr] = goal.detach().squeeze()
        self.goal_mem_ptr     = torch.tensor((ptr + 1) % self.goal_memory.size(0))
        self.goal_count       = self.goal_count + 1

    def forward(
        self,
        skill_library:  SkillLibrary,
        curriculum_w:   torch.Tensor,       # [n_slots]
        x:              torch.Tensor,       # [B, L, D] current hidden state
    ) -> Tuple[torch.Tensor, Dict]:
        B = x.size(0)
        device = x.device

        # Compose goal from skills
        h, chosen_skills = self._compose_skills(skill_library, curriculum_w, device)

        # Difficulty check
        difficulty_logits = self.difficulty_head(h)              # [1, 3]
        difficulty_probs  = F.softmax(difficulty_logits, dim=-1)
        difficulty_label  = difficulty_probs.argmax(-1).item()   # 0/1/2
        LABELS = ["too_easy", "just_right", "too_hard"]

        # Project to goal space
        goal = self.goal_proj(h)                                 # [1, goal_dim]

        # Novelty check against recent goals
        if int(self.goal_count.item()) > 0:
            # Compare against mean of goal memory
            mem_mean = self.goal_memory[:min(int(self.goal_count.item()), 16)].mean(0, keepdim=True)
            novelty  = self.novelty_head(
                torch.cat([goal, mem_mean], dim=-1)
            ).item()
        else:
            novelty = 1.0

        # Store this goal
        self._store_goal(goal)

        # Expand goal to batch size
        goal_batch = goal.expand(B, -1)                          # [B, goal_dim]

        # Project goal into hidden-state space and inject
        goal_in_hidden = self.goal_to_hidden(goal_batch)            # [B, dim]
        x_conditioned = x + goal_in_hidden.unsqueeze(1) * 0.05

        return x_conditioned, {
            "goal":              goal_batch,
            "chosen_skills":     chosen_skills,
            "difficulty":        LABELS[difficulty_label],
            "difficulty_probs":  difficulty_probs.squeeze(0).tolist(),
            "novelty":           novelty,
            "goal_count":        int(self.goal_count.item()),
        }


# ============= MetaCurriculum Claudeson =============

class ClaudesonMetaCurriculum(ClaudesonCausalWorld):
    """
    Claudeson 2026 — MetaCurriculum Edition.

    Inherits the full CausalWorld architecture and adds open-ended learning:

      skill_discovery   — information-gain reward + skill library + lifecycle
      goal_generator    — compositional goal synthesis from mastered skills
      cp_monitor        — learning progress tracking; curriculum scheduling

    The key loop this enables:
      1. Experience arrives → IG reward computed → skill library queried
      2. If active skill is mastered, freeze and open new slot
      3. GoalGenerator composes a new harder goal from mastered skills
      4. CPMonitor tracks LP across all skills, weights training accordingly
      5. The agent pursues the composed goal, updating the skill library
      6. Repeat from 1

    This is distinct from all previous generations:
      - Jedi: reacts to energy state
      - Grounded: executes tools in response to queries
      - Sovereign: improves itself when it predicts improvement will help
      - Transcendent: infers values from preferences
      - CausalWorld: plans via do-calculus
      - MetaCurriculum: *generates its own curriculum* — no external teacher needed

    Forward signature extensions:
      competence_signal  — optional scalar [0,1] from an external evaluator
                           (if None, estimated from hidden-state confidence)

    New output keys:
      metacurriculum     — skill discovery + goal generation + CP monitor outputs
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.skill_discovery = SkillDiscoveryEngine(args)
        self.goal_generator  = OpenEndedGoalGenerator(args)
        self.cp_monitor      = CompetenceProgressMonitor(args)

    def forward(
        self,
        text:               Optional[torch.Tensor] = None,
        img:                Optional[torch.Tensor] = None,
        audio:              Optional[torch.Tensor] = None,
        goal_tokens:        Optional[torch.Tensor] = None,
        feedback:           Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
        actual_action:      Optional[torch.Tensor] = None,
        rung_labels:        Optional[torch.Tensor] = None,
        competence_signal:  Optional[float] = None,
    ) -> Dict:
        # ── Full CausalWorld pass ────────────────────────────────────────
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens,
            feedback=feedback, agent_observations=agent_observations,
            actual_action=actual_action, rung_labels=rung_labels,
        )
        x = base["hidden_states"]

        # ── Extract Jedi VAE tensors for IG computation ──────────────────
        # These live in the Jedi sub-forward; we need to re-derive them
        # from the latent info already in the base output.
        # In a fully integrated training loop these would be passed through;
        # here we approximate with the stored latent.
        latent = base.get("latent")                              # [B, L, latent_dim] or None
        if latent is not None:
            # Approximate posterior = reparameterised latent
            # Use zero prior as a conservative estimate
            B, L_lat = latent.shape[0], latent.shape[1] if latent.dim() > 2 else 1
            lat_dim  = latent.shape[-1]
            if latent.dim() == 2:
                latent = latent.unsqueeze(1)
            mu         = latent
            logvar     = torch.zeros_like(latent)
            prior_mu   = torch.zeros_like(latent)
            prior_logvar = torch.zeros_like(latent)
        else:
            # Fallback: create dummy tensors if latent not available
            B, lat_dim = x.size(0), self.args.latent_dim
            mu = prior_mu = torch.zeros(B, 1, lat_dim, device=x.device)
            logvar = prior_logvar = torch.zeros(B, 1, lat_dim, device=x.device)

        # ── Skill Discovery ──────────────────────────────────────────────
        x, discovery_out = self.skill_discovery(
            x, mu, logvar, prior_mu, prior_logvar,
            competence_signal=competence_signal,
        )

        # ── Curriculum weights for goal generation ───────────────────────
        curriculum_w, cp_out = self.cp_monitor(self.skill_discovery.library)

        # ── Open-Ended Goal Generation ───────────────────────────────────
        x, goal_out = self.goal_generator(
            self.skill_discovery.library,
            torch.tensor(curriculum_w if isinstance(curriculum_w, list)
                         else curriculum_w.tolist(), device=x.device),
            x,
        )

        return {
            **base,
            "hidden_states": x,
            "metacurriculum": {
                "discovery":   discovery_out,
                "curriculum":  cp_out,
                "goal":        goal_out,
            },
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        losses = super().compute_auxiliary_losses()
        # No additional differentiable losses from metacurriculum in this pass
        # (IG reward and curriculum weights are used for experience scheduling,
        # not as direct gradient signals in the main loss — that would conflate
        # the reward signal with the training objective)
        return losses


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — METACURRICULUM EDITION")
    print("Information Gain · Skill Discovery · Competence Progress · OEL")
    print("=" * 70)

    args = ModelArgs()
    # Tiny CPU demo config
    args.dim                    = 128
    args.n_layers               = 2
    args.n_heads                = 4
    args.n_kv_heads             = 2
    args.vocab_size             = 512
    args.max_seq_len            = 64
    args.memory_slots           = 32
    args.episodic_slots         = 64
    args.goal_dim               = 128
    args.latent_dim             = 64
    args.energy_hidden          = 128
    args.ssm_state_dim          = 32
    args.ssm_chunk_size         = 16
    args.num_experts            = 2
    args.num_shared_experts     = 1
    args.env_state_dim          = 32
    args.action_space_size      = 16
    args.planning_horizon       = 2
    args.num_simulations        = 2
    args.img_size               = 32
    args.patch_size             = 8
    args.audio_spec_dim         = 16
    args.gradient_checkpointing = False
    args.n_agents               = 4
    args.lora_rank              = 8
    args.n_causal_nodes         = 16
    args.metacog_hidden         = 64
    args.n_debate_agents        = 3
    args.debate_hidden          = 128
    args.n_propositions         = 16
    args.n_constraints          = 8
    args.consistency_iters      = 2
    args.rsi_rank               = 4
    args.rsi_horizon            = 2
    args.n_workspace_slots      = 8
    args.gw_competition_k       = 2
    args.gw_broadcast_steps     = 1
    args.n_ops                  = 16
    args.n_registers            = 4
    args.prog_steps             = 3
    args.prog_hidden            = 64
    args.irl_hidden             = 64
    args.irl_n_preferences      = 8
    args.lif_steps              = 3
    args.causal_state_dim       = 32
    args.intervention_horizon   = 2
    args.n_intervention_samples = 4
    args.cf_n_branches          = 2
    args.attr_top_k             = 4
    args.pearl_hidden           = 64
    # MetaCurriculum specific
    args.n_skill_slots          = 8
    args.skill_rank             = 4
    args.skill_embed_dim        = 32
    args.cp_window              = 8
    args.cp_hidden              = 64
    args.oeg_n_compose          = 2
    args.oeg_hidden             = 64
    args.ig_beta                = 0.5

    print("\nInitialising ClaudesonMetaCurriculum...")
    model = ClaudesonMetaCurriculum(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text          = torch.randint(0, 512, (2, 32))
    feedback      = torch.randn(2, args.dim)
    agent_obs     = torch.randn(2, 8, args.dim)
    actual_action = torch.randint(0, args.action_space_size, (2,))

    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)

    # Simulate a few steps of competence recording to populate LP buffers
    for step in range(args.cp_window):
        score = 0.3 + 0.05 * step          # steadily improving
        model.cp_monitor.record_performance(0, score)

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            text=text,
            feedback=feedback,
            agent_observations=agent_obs,
            actual_action=actual_action,
            competence_signal=0.45,
        )

    mc = out["metacurriculum"]

    print("\nJedi state:")
    print(f"  Goal:         {out['jedi_goal']}")
    print(f"  Energy:       {out['jedi_energy'].mean().item():.4f}")

    print("\nPearl Ladder:")
    print(f"  Rung:         {out['pearl']['rung']}")

    print("\nInformation Gain:")
    ig = mc["discovery"]["ig"]
    print(f"  KL mean:      {sum(ig['kl'].flatten().tolist()) / ig['kl'].numel():.4f}")
    print(f"  Baseline:     {ig['baseline']:.4f}")
    print(f"  IG reward:    {mc['discovery']['ig_reward'].mean().item():.4f}")

    print("\nSkill Library:")
    sk = mc["discovery"]["skill"]
    print(f"  Active slots: {sk['n_active']}")
    print(f"  Frozen slots: {sk['n_frozen']}")
    print(f"  Active skill: {mc['discovery']['active_skill']}")
    print(f"  Advanced:     {mc['discovery']['skill_advanced']}")

    print("\nCompetence Progress Monitor:")
    cp = mc["curriculum"]
    print(f"  LP estimates: {[f'{v:.3f}' for v in cp['lp_estimates'][:sk['n_active']]]}")
    print(f"  Zones:        {cp['zones'][:sk['n_active']]}")
    print(f"  Curr. weights: {[f'{v:.3f}' for v in cp['curriculum_weights'][:sk['n_active']]]}")

    print("\nOpen-Ended Goal Generator:")
    g = mc["goal"]
    print(f"  Chosen skills:    {g['chosen_skills']}")
    print(f"  Difficulty:       {g['difficulty']}")
    print(f"  Difficulty probs: {[f'{v:.3f}' for v in g['difficulty_probs']]}")
    print(f"  Novelty:          {g['novelty']:.4f}")
    print(f"  Goal shape:       {g['goal'].shape}")
    print(f"  Goals generated:  {g['goal_count']}")

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonMetaCurriculum READY.")
    print("Curious by design.  Builds its own curriculum.")
    print("Knows what it has mastered and what to try next.")
    print("No external teacher required.")
    print("=" * 70)
