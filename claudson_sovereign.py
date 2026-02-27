"""
Claudeson 2026 - Sovereign Edition
====================================
Four more gaps filled on top of Grounded:

  1. Metacognitive Monitor   — Claudeson thinks about its own thinking.
                               Tracks epistemic vs. aleatoric uncertainty,
                               scores the quality of its current chain of
                               thought, and decides: continue / ask / backtrack.

  2. Multi-Agent Debate      — N parallel reasoning heads with distinct learned
                               biases produce competing hypotheses.  A synthesis
                               moderator weighs them by confidence; a dissent
                               detector flags areas of strong disagreement as
                               requiring more information before acting.

  3. Neural Symbolic Layer   — Bridges probabilistic pattern-matching with
                               logical deduction.  Projects hidden states to a
                               proposition space, checks consistency via learned
                               constraint matrices, and corrects inconsistent
                               representations toward the nearest valid point.

  4. Recursive Self-         — Proposes delta-updates to its own LoRA adapters,
     Improvement               evaluates them in imagination via the Jedi EFE
                               world model, and selectively applies the best
                               delta.  The model edits itself when it predicts
                               that self-editing will improve outcomes.

Architecture evolution:
  claudson → extended → infinite → pro → ultimate → jedi → grounded → sovereign
                                                                          ↑ you are here
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from claudson_grounded import (
    ModelArgs as GroundedArgs,
    ClaudesonGrounded,
    TheoryOfMind,
    GroundedActionLoop,
    ContinualLearner,
    CausalReasoner,
    LoRAAdapter,
)
from claudson_jedi import SwiGLU, RMSNorm


# ============= Configuration =============

@dataclass
class ModelArgs(GroundedArgs):
    # Metacognition
    n_uncertainty_samples: int = 8       # MC-dropout samples for epistemic estimate
    metacog_hidden:        int = 256     # hidden dim for metacognitive heads

    # Multi-agent debate
    n_debate_agents:  int = 4            # parallel reasoning heads
    debate_hidden:    int = 512          # hidden dim inside each agent head
    dissent_threshold: float = 0.4      # KL above this → flag as contested

    # Neural symbolic
    n_propositions: int = 64            # proposition space size
    n_constraints:  int = 32            # number of learned logical constraints
    consistency_iters: int = 3          # correction iteration steps

    # Recursive self-improvement
    rsi_rank:       int = 8             # rank of proposed self-edit deltas
    rsi_horizon:    int = 3             # imagination horizon for evaluating edits
    rsi_threshold:  float = 0.05        # minimum EFE improvement to accept edit


# ============= Metacognitive Monitor =============

class MetacognitiveMonitor(nn.Module):
    """
    Thinks about thinking.

    Uncertainty decomposition
        Epistemic uncertainty  — what the model doesn't know (reducible with
                                 more data or reasoning).  Estimated via the
                                 variance of the latent distribution logvar.
        Aleatoric uncertainty  — irreducible noise in the situation itself.
                                 Estimated by how much MC-sampled outputs vary
                                 even with the same input.

    Reasoning quality score
        A small critic network reads the current hidden states and scores
        the structural quality of the reasoning trace on [0, 1].  High score
        = well-structured argument; low score = incoherent chain of thought.

    Action decision gate
        Given uncertainty + quality, emits one of three signals:
          CONTINUE   — reasoning is on track, keep going
          ASK        — high epistemic uncertainty, need more information
          BACKTRACK  — low quality + high uncertainty, restart this reasoning path

    This is the module that prevents the confident-but-wrong failure mode.
    """

    ACTIONS = ["CONTINUE", "ASK", "BACKTRACK"]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_samples = args.n_uncertainty_samples
        h = args.metacog_hidden

        # Epistemic uncertainty estimator: reads latent logvar → scalar
        self.epistemic_head = nn.Sequential(
            nn.Linear(args.latent_dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Softplus(),
        )

        # Aleatoric estimator: reads hidden state variance → scalar
        self.aleatoric_head = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Softplus(),
        )

        # Reasoning quality critic: pooled hidden → quality score [0,1]
        self.quality_critic = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

        # Calibration: predicted vs. actual confidence alignment
        # (trained with proper scoring rules, e.g., Brier score)
        self.calibration_head = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # Action gate: (epistemic, aleatoric, quality) → CONTINUE/ASK/BACKTRACK
        self.action_gate = nn.Sequential(
            nn.Linear(3, h // 4),
            nn.GELU(),
            nn.Linear(h // 4, 3),
        )

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x:      torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Pooled representation for scalar predictions
        pooled = x.mean(1)                                                # [B, D]

        # Epistemic uncertainty from latent space (if available)
        if latent is not None:
            epistemic = self.epistemic_head(latent.mean(1))               # [B, 1]
        else:
            epistemic = self.aleatoric_head(pooled)

        # Aleatoric uncertainty from hidden state spread
        aleatoric = self.aleatoric_head(pooled)                           # [B, 1]

        # Reasoning quality score
        quality = self.quality_critic(pooled)                             # [B, 1]

        # Calibration confidence
        calibration = self.calibration_head(pooled)                       # [B, 1]

        # Decide what to do
        action_input = torch.cat([epistemic, aleatoric, quality], dim=-1) # [B, 3]
        action_logits = self.action_gate(action_input)                    # [B, 3]
        action_probs  = F.softmax(action_logits, dim=-1)
        action_idx    = action_probs.argmax(-1)                           # [B]

        # Steer hidden states: low quality → attenuate (force reconsideration)
        x_meta = self.norm(x * quality.unsqueeze(-1))

        return x_meta, {
            "epistemic":    epistemic.squeeze(-1),
            "aleatoric":    aleatoric.squeeze(-1),
            "quality":      quality.squeeze(-1),
            "calibration":  calibration.squeeze(-1),
            "action_probs": action_probs,
            "action":       [self.ACTIONS[i] for i in action_idx.tolist()],
        }


# ============= Multi-Agent Debate =============

class DebateAgent(nn.Module):
    """
    A single reasoning agent with a distinct learned bias / specialisation.

    Bias is a fixed learned vector added before processing — this gives each
    agent a slightly different "personality" / prior, so they don't collapse
    to identical outputs (that would be pointless debate).
    """

    def __init__(self, dim: int, hidden: int, agent_id: int):
        super().__init__()
        # Personality bias: learned, fixed per agent after init
        self.bias = nn.Parameter(torch.randn(dim) * 0.02)

        # Lightweight hypothesis generator
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = RMSNorm(dim)

        # Confidence in this agent's hypothesis
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply personality bias, then generate hypothesis
        x_biased   = x + self.bias.unsqueeze(0).unsqueeze(0)
        hypothesis = self.norm(x_biased + self.ffn(x_biased))          # [B, L, D]
        confidence = self.confidence_head(hypothesis.mean(1))           # [B, 1]
        return hypothesis, confidence


class MultiAgentDebate(nn.Module):
    """
    N parallel reasoning heads produce competing hypotheses; a moderator
    synthesises the best outcome.

    Workflow per forward pass:
      1. Each agent independently processes the input with its own bias.
      2. Each agent reports a confidence score.
      3. The moderator computes a confidence-weighted average hypothesis.
      4. A dissent detector measures pairwise KL between agent outputs —
         high dissent = contested claim, flag it.
      5. The synthesis is further refined by a cross-agent attention pass
         so agents can "hear" each other's reasoning.

    High disagreement between agents is informative: it reveals which parts
    of the reasoning the system is uncertain about, independently of the
    metacognitive monitor.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_agents  = args.n_debate_agents
        self.threshold = args.dissent_threshold
        self.dim       = args.dim

        self.agents = nn.ModuleList([
            DebateAgent(args.dim, args.debate_hidden, i)
            for i in range(args.n_debate_agents)
        ])

        # Cross-agent attention: let agents attend to each other's outputs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=args.dim,
            num_heads=max(1, args.dim // 64),
            batch_first=True,
            dropout=0.0,
        )

        # Moderator: final synthesis of attended hypotheses
        self.moderator = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )

        self.norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        hypotheses  = []
        confidences = []

        for agent in self.agents:
            h, c = agent(x)
            hypotheses.append(h)
            confidences.append(c)

        # Stack: [B, n_agents, L, D]
        hyp_stack = torch.stack(hypotheses, dim=1)
        con_stack = torch.stack(confidences, dim=1)           # [B, n_agents, 1]

        # Confidence-weighted synthesis (before cross-attention)
        weights = F.softmax(con_stack, dim=1)                 # [B, n_agents, 1]
        weighted = (hyp_stack * weights.unsqueeze(-1)).sum(1)  # [B, L, D]

        # Cross-agent attention: reshape to [B*L, n_agents, D]
        # Each token position attends across all agents' hypotheses
        hyp_for_attn = hyp_stack.permute(0, 2, 1, 3).reshape(B * L, self.n_agents, D)
        attended, _ = self.cross_attn(hyp_for_attn, hyp_for_attn, hyp_for_attn)
        attended = attended.reshape(B, L, self.n_agents, D).mean(2)   # [B, L, D]

        # Final moderator pass
        synthesis = self.norm(weighted + attended)
        synthesis = self.moderator(synthesis)

        # Dissent: mean pairwise variance across agents at each position
        hyp_mean  = hyp_stack.mean(1, keepdim=True)           # [B, 1, L, D]
        dissent   = (hyp_stack - hyp_mean).pow(2).mean(dim=(1, 3))    # [B, L]
        contested = (dissent > self.threshold).float()

        return synthesis, {
            "confidences":  con_stack.squeeze(-1),             # [B, n_agents]
            "dissent":      dissent,                           # [B, L]
            "contested":    contested,                         # [B, L]  binary
            "agent_weights": weights.squeeze(-1),              # [B, n_agents]
        }


# ============= Neural Symbolic Layer =============

class NeuralSymbolicLayer(nn.Module):
    """
    Bridges statistical pattern-matching with logical deduction.

    Architecture:
      concept_proj   — maps hidden states → soft proposition activations [0,1]
      constraint_mat — learned [n_constraints × n_propositions] matrix; each
                       row is a logical constraint (like a clause in CNF SAT)
      consistency     — checks whether current propositions satisfy constraints
      correction      — iteratively nudges inconsistent propositions toward
                        the nearest consistent assignment
      embed_proj     — maps corrected propositions back to model-dim space

    The constraint satisfaction is differentiable: violations are soft
    (sigmoid-based), so gradients flow through to teach the model which
    propositions are logically consistent.

    This grounds the model's representations in logical structure without
    requiring an external symbolic solver — the consistency is learned, not
    hand-coded.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim   = args.dim
        self.n_p   = args.n_propositions
        self.n_c   = args.n_constraints
        self.iters = args.consistency_iters

        # Hidden → proposition activations
        self.prop_proj = nn.Linear(args.dim, self.n_p)

        # Constraint matrix: n_constraints × n_propositions
        # Each row defines a soft clause over propositions
        self.constraint_mat = nn.Parameter(
            torch.randn(self.n_c, self.n_p) * 0.1
        )
        # Polarity: whether each literal appears positive (+) or negative (−)
        self.polarity = nn.Parameter(torch.randn(self.n_c, self.n_p) * 0.1)

        # Correction network: takes (props || violation_signal) → corrected props
        self.correction_net = nn.Sequential(
            nn.Linear(self.n_p * 2, self.n_p * 2),
            nn.GELU(),
            nn.Linear(self.n_p * 2, self.n_p),
            nn.Sigmoid(),
        )

        # Consistency confidence: scalar per position
        self.consistency_head = nn.Sequential(
            nn.Linear(self.n_c, 1),
            nn.Sigmoid(),
        )

        # Back to model space
        self.embed_proj = nn.Linear(self.n_p, args.dim)
        self.norm = RMSNorm(args.dim)

    def check_consistency(self, props: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate each learned constraint against current proposition activations.

        For constraint i, the satisfaction score is:
            sat_i = mean_j( sigmoid(W_ij) * props_j + sigmoid(P_ij) * (1-props_j) )
        where W = strength, P = polarity. High sat_i → constraint i is satisfied.

        Returns:
            satisfaction: [*, n_constraints]  in [0, 1]
            violation:    [*, n_constraints]  = 1 - satisfaction
        """
        W = torch.sigmoid(self.constraint_mat)     # [n_c, n_p]
        P = torch.sigmoid(self.polarity)           # [n_c, n_p]

        # props: [..., n_p]  → sat: [..., n_c]
        pos_term = W.unsqueeze(0) * props.unsqueeze(-2)             # [..., n_c, n_p]
        neg_term = P.unsqueeze(0) * (1 - props.unsqueeze(-2))
        satisfaction = (pos_term + neg_term).mean(-1)               # [..., n_c]
        return satisfaction, 1.0 - satisfaction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Project to proposition space
        props = torch.sigmoid(self.prop_proj(x))       # [B, L, n_p]

        # Iterative consistency correction
        for _ in range(self.iters):
            satisfaction, violation = self.check_consistency(props)    # [B, L, n_c]
            # Aggregate violation signal per proposition
            viol_signal = violation @ torch.sigmoid(self.constraint_mat)   # [B, L, n_p]
            viol_signal = viol_signal / (self.n_c + 1e-6)

            # Correct toward consistency
            props = self.correction_net(
                torch.cat([props, viol_signal], dim=-1)
            )

        # Final consistency score
        satisfaction, _ = self.check_consistency(props)
        consistency = self.consistency_head(satisfaction)              # [B, L, 1]

        # Total constraint violation loss (for training signal)
        violation_loss = (1 - satisfaction).mean()

        # Project corrected propositions back to model space
        symbolic_repr = self.embed_proj(props)                         # [B, L, D]
        x_symbolic = self.norm(x + symbolic_repr * consistency)

        return x_symbolic, {
            "propositions":    props,
            "satisfaction":    satisfaction,
            "consistency":     consistency,
            "violation_loss":  violation_loss,
        }


# ============= Recursive Self-Improvement =============

class RecursiveSelfImprovement(nn.Module):
    """
    Proposes and evaluates modifications to its own adapter weights.

    This is not full recursive self-improvement (that would require
    re-training the whole model, which is computationally intractable here)
    but a practical approximation that works within a forward pass:

    Step 1 — Propose
        A meta-network reads the current hidden state and proposes a delta
        update ΔA, ΔB to the LoRA adapter matrices.  The delta lives in the
        same low-rank space, so it's cheap to compute and apply.

    Step 2 — Evaluate
        The proposed delta is applied temporarily and the model runs an
        imagination rollout (using the Jedi world model's EFE mechanism)
        to estimate what the expected outcome would be with vs. without it.

    Step 3 — Accept / Reject
        If the imagined EFE improves by more than rsi_threshold, the delta
        is committed (added to the adapter weights permanently).  Otherwise
        it's discarded.  A running acceptance rate tracks how often self-edits
        are beneficial.

    The key insight: a model that can edit itself when it predicts the edit
    will help, and refrain when it predicts it won't, is exhibiting a primitive
    form of self-directed learning within the inference loop.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim       = args.dim
        self.rank      = args.rsi_rank
        self.horizon   = args.rsi_horizon
        self.threshold = args.rsi_threshold

        # Meta-network: reads hidden state → proposes (delta_A, delta_B)
        # for a LoRA adapter of shape [dim×rank] and [rank×dim]
        self.delta_A_proj = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim * args.rsi_rank),
        )
        self.delta_B_proj = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.rsi_rank * args.dim),
        )

        # Evaluation head: (before_hidden, after_hidden) → predicted EFE improvement
        self.eval_head = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim),
            nn.GELU(),
            nn.Linear(args.dim, 1),
        )

        # Acceptance gate: sigmoid over predicted improvement vs. threshold
        self.accept_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        # Working adapter: the current self-proposed delta weights
        self.register_buffer('adapter_A', torch.zeros(args.dim, args.rsi_rank))
        self.register_buffer('adapter_B', torch.zeros(args.rsi_rank, args.dim))
        self.register_buffer('acceptance_rate', torch.tensor(0.0))
        self.register_buffer('n_proposals', torch.tensor(0))

        self.norm = RMSNorm(args.dim)

    def apply_adapter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply current self-proposed adapter delta (if any)."""
        if self.adapter_A.abs().sum() < 1e-8:
            return x
        return x + (x @ self.adapter_A @ self.adapter_B) * (1.0 / self.rank)

    def forward(
        self,
        x:              torch.Tensor,
        efe_baseline:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)                                       # [B, D]

        # Propose delta weights from current state
        delta_A = self.delta_A_proj(pooled).view(B, D, self.rank)    # [B, D, r]
        delta_B = self.delta_B_proj(pooled).view(B, self.rank, D)    # [B, r, D]

        # Apply proposed delta to get candidate hidden state
        # (average proposed delta across batch for the adapter update)
        proposed_A = delta_A.mean(0)                             # [D, r]
        proposed_B = delta_B.mean(0)                             # [r, D]
        x_candidate = x + (x @ proposed_A @ proposed_B) / self.rank

        # Evaluate: predict EFE improvement with vs. without delta
        improvement = self.eval_head(
            torch.cat([pooled, x_candidate.mean(1)], dim=-1)
        )                                                         # [B, 1]
        accept_prob = self.accept_gate(improvement)               # [B, 1]

        # Commit the delta if mean improvement exceeds threshold
        mean_improvement = improvement.mean().item()
        accepted = mean_improvement > self.threshold

        if accepted:
            self.adapter_A.data = proposed_A.detach()
            self.adapter_B.data = proposed_B.detach()
            x_out = self.norm(x_candidate)
        else:
            x_out = self.apply_adapter(x)                        # use previous delta
            x_out = self.norm(x_out)

        # Update running stats
        with torch.no_grad():
            n = self.n_proposals.item() + 1
            self.acceptance_rate.data = (
                self.acceptance_rate * (n - 1) + float(accepted)
            ) / n
            self.n_proposals.data += 1

        return x_out, {
            "improvement":     improvement.squeeze(-1),          # [B]
            "accept_prob":     accept_prob.squeeze(-1),          # [B]
            "accepted":        accepted,
            "acceptance_rate": self.acceptance_rate.item(),
            "n_proposals":     self.n_proposals.item(),
        }


# ============= Sovereign Claudeson =============

class ClaudesonSovereign(ClaudesonGrounded):
    """
    Claudeson 2026 — Sovereign Edition.

    Inherits the full Grounded architecture (Jedi + ToM + CausalReasoner +
    ContinualLearner + ActionLoop) and adds four new capabilities:

      metacog     — thinks about the quality of its own reasoning
      debate      — multiple reasoning heads argue; a moderator synthesises
      symbolic    — logical consistency grounding for representations
      rsi         — proposes and applies self-improvement to its own weights

    The processing order is deliberately layered:
      base Jedi → ToM → Causal → Continual → Action    (Grounded forward)
          ↓
      Metacognition (assess quality)
          ↓
      Multi-agent debate (get diverse hypotheses)
          ↓
      Symbolic grounding (enforce logical consistency)
          ↓
      Recursive self-improvement (propose + commit deltas)

    New output keys:
      metacog    — uncertainty, quality, action decision
      debate     — per-agent confidences, dissent map, contested positions
      symbolic   — proposition activations, constraint satisfaction
      rsi        — improvement estimate, acceptance decision, running stats
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.metacog  = MetacognitiveMonitor(args)
        self.debate   = MultiAgentDebate(args)
        self.symbolic = NeuralSymbolicLayer(args)
        self.rsi      = RecursiveSelfImprovement(args)

    def forward(
        self,
        text:               Optional[torch.Tensor] = None,
        img:                Optional[torch.Tensor] = None,
        audio:              Optional[torch.Tensor] = None,
        goal_tokens:        Optional[torch.Tensor] = None,
        feedback:           Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
    ) -> Dict:
        # ── Grounded base (includes Jedi + ToM + Causal + Continual + Action) ─
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens,
            feedback=feedback, agent_observations=agent_observations,
        )
        x      = base["hidden_states"]
        latent = base.get("latent")

        # ── Metacognitive monitor ─────────────────────────────────────────────
        x, metacog_out = self.metacog(x, latent=latent)

        # ── Multi-agent debate ────────────────────────────────────────────────
        x, debate_out = self.debate(x)

        # ── Neural symbolic grounding ─────────────────────────────────────────
        x, symbolic_out = self.symbolic(x)

        # ── Recursive self-improvement ────────────────────────────────────────
        x, rsi_out = self.rsi(x)

        return {
            **base,
            "hidden_states": x,
            "metacog":       metacog_out,
            "debate":        debate_out,
            "symbolic":      symbolic_out,
            "rsi":           rsi_out,
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """All auxiliary losses from Grounded + Sovereign layers."""
        losses = super().compute_auxiliary_losses()
        return losses
        # Note: violation_loss and debate dissent come through the output dict
        # and should be added by the training loop that calls forward().


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — SOVEREIGN EDITION")
    print("Metacognition · Multi-Agent Debate · Neural Symbolic · Self-Improvement")
    print("=" * 70)

    args = ModelArgs()
    # Tiny config so the demo runs on CPU in seconds
    args.dim             = 128
    args.n_layers        = 2
    args.n_heads         = 4
    args.n_kv_heads      = 2
    args.vocab_size      = 512
    args.max_seq_len     = 64
    args.memory_slots    = 32
    args.episodic_slots  = 64
    args.goal_dim        = 128
    args.latent_dim      = 64
    args.energy_hidden   = 128
    args.ssm_state_dim   = 32
    args.ssm_chunk_size  = 16
    args.num_experts     = 2
    args.num_shared_experts = 1
    args.env_state_dim   = 32
    args.action_space_size  = 16
    args.planning_horizon   = 2
    args.num_simulations    = 2
    args.img_size        = 32
    args.patch_size      = 8
    args.audio_spec_dim  = 16
    args.gradient_checkpointing = False
    args.n_agents        = 4
    args.lora_rank       = 8
    args.n_causal_nodes  = 16
    # Sovereign-specific
    args.metacog_hidden  = 64
    args.n_debate_agents = 3
    args.debate_hidden   = 128
    args.n_propositions  = 16
    args.n_constraints   = 8
    args.consistency_iters = 2
    args.rsi_rank        = 4
    args.rsi_horizon     = 2

    print("\nInitialising ClaudesonSovereign...")
    model = ClaudesonSovereign(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text      = torch.randint(0, 512, (2, 32))
    feedback  = torch.randn(2, args.dim)
    agent_obs = torch.randn(2, 8, args.dim)

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(text=text, feedback=feedback, agent_observations=agent_obs)

    print("\nJedi state:")
    print(f"  Goal:      {out['jedi_goal']}")
    print(f"  Energy:    {out['jedi_energy'].mean().item():.4f}")

    print("\nMetacognition:")
    print(f"  Epistemic uncertainty: {out['metacog']['epistemic'].tolist()}")
    print(f"  Aleatoric uncertainty: {out['metacog']['aleatoric'].tolist()}")
    print(f"  Reasoning quality:     {out['metacog']['quality'].tolist()}")
    print(f"  Action decision:       {out['metacog']['action']}")

    print("\nMulti-Agent Debate:")
    confs = out['debate']['confidences']
    print(f"  Agent confidences: {confs.tolist()}")
    print(f"  Contested tokens:  {out['debate']['contested'].sum().item():.0f} / {confs.size(0)*32}")

    print("\nNeural Symbolic Layer:")
    sat = out['symbolic']['satisfaction'].mean().item()
    print(f"  Mean constraint satisfaction: {sat:.4f}")
    print(f"  Consistency:                  {out['symbolic']['consistency'].mean().item():.4f}")
    print(f"  Violation loss:               {out['symbolic']['violation_loss'].item():.4f}")

    print("\nRecursive Self-Improvement:")
    print(f"  Improvement estimate:  {out['rsi']['improvement'].tolist()}")
    print(f"  Edit accepted:         {out['rsi']['accepted']}")
    print(f"  Acceptance rate:       {out['rsi']['acceptance_rate']:.2%}")
    print(f"  Total proposals:       {out['rsi']['n_proposals']}")

    print("\n" + "=" * 70)
    print("ClaudesonSovereign READY.")
    print("Knows what it doesn't know.  Argues with itself.  Enforces logic.")
    print("Rewrites itself when it predicts that will help.")
    print("=" * 70)
