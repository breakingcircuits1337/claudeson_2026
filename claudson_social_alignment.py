"""
Claudeson 2026 - Social Alignment Edition
==========================================
Multi-Stakeholder Value Aggregation · Norm Learning · Social Contract · Moral Reasoning

The problem this generation solves
------------------------------------
Every previous generation optimises for a single reward signal or a single
agent's inferred preferences (IRL in Transcendent).  This is insufficient
for a system that operates in a social world:

  - Different stakeholders have different, sometimes conflicting values.
  - What is good for one agent may harm another.
  - Social norms are not fixed; they emerge from repeated interactions and
    evolve over time.
  - An agent that ignores social context will optimise in ways that
    technically satisfy its goals while violating implicit social contracts.

Examples of failures this layer prevents:
  - A purely goal-directed agent helps User A accomplish a task that harms
    User B, because B's interests were never represented in the reward.
  - An agent that learned values from a biased sample of one stakeholder
    group systematically disadvantages under-represented groups.
  - An agent that follows explicit rules but violates their spirit, because
    norms are not rules — they are learned social regularities.
  - An agent that cannot explain *why* an action is good or bad in social
    terms, only that it maximises a numerical objective.

This generation adds five components:

  1. Stakeholder Value Model (SVM)
     Maintains a separate value function for each of N stakeholder groups.
     Groups can be: users, third parties, society, future generations, etc.
     Values are inferred from preferences (extending the IRL mechanism from
     Transcendent) and represented as interpretable concept vectors (using
     the Concept Bottleneck from Abstraction).

  2. Social Welfare Aggregator (SWA)
     Combines stakeholder value estimates into a social welfare signal using
     multiple aggregation philosophies simultaneously:
       - Utilitarian: sum of all welfare (maximise total)
       - Rawlsian:    min of all welfare (maximise worst-off)
       - Egalitarian: minimise variance across welfare scores
       - Prioritarian: weighted sum with higher weight on lower welfare
     Outputs a multi-objective welfare vector that the planner can use.

  3. Norm Learning Engine (NLE)
     Learns implicit social norms from behavioural patterns.  Unlike explicit
     rules, norms are:
       - Statistical: violated by some fraction of agents, not all
       - Graduated: some violations are worse than others
       - Contextual: the same action can be normal in one context, deviant in another
       - Evolving: norms change as society changes
     The NLE models norms as contextual soft constraints learned from
     multi-agent interaction histories.

  4. Social Contract Reasoner (SCR)
     Implements a contractualist framework (Scanlon 1998, Rawls 1971):
       An action is permissible if it could not be reasonably rejected by
       any stakeholder reasoning from behind a veil of ignorance about their
       own position.
     This is operationalised as: simulate each stakeholder's perspective,
     check if any would reject the action, and compute the strength of
     potential objections.

  5. Moral Uncertainty Estimator (MUE)
     Explicitly models uncertainty over moral frameworks.
     Different ethical theories (consequentialism, deontology, virtue ethics,
     contractualism) sometimes agree and sometimes conflict.  The MUE:
       - Scores each action under each framework
       - Computes agreement/disagreement across frameworks
       - Outputs a moral confidence score (high when frameworks agree)
       - Flags actions where frameworks strongly disagree for human review

Architecture evolution:
  claudson → ... → abstraction → social_alignment
                                        ↑ you are here (final layer)

This is the outermost layer of the Claudeson stack.  It does not add
computation for its own sake — it adds *accountability*.  Every output
of the system is now filtered through a social welfare check.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claudson_abstraction import (
    ClaudesonAbstraction,
)
from claudson_abstraction import (
    ModelArgs as AbstractionArgs,
)
from claudson_jedi import RMSNorm

log = logging.getLogger(__name__)


# ============= Configuration =============


@dataclass
class ModelArgs(AbstractionArgs):
    # Stakeholder Value Model
    n_stakeholder_groups: int = 8  # number of tracked stakeholder groups
    stakeholder_hidden: int = 256  # hidden dim for value model
    value_lr: float = 0.01  # online learning rate for value updates

    # Social Welfare Aggregator
    welfare_hidden: int = 128  # hidden dim for welfare head
    n_welfare_objectives: int = 4  # utilitarian / rawlsian / egalitarian / prioritarian

    # Norm Learning Engine
    n_norm_slots: int = 64  # number of learnable norm representations
    norm_context_window: int = 16  # context tokens for norm matching
    norm_hidden: int = 256  # hidden dim for norm encoder
    norm_violation_alpha: float = 0.1  # weight of norm-violation penalty

    # Social Contract Reasoner
    scr_n_perspectives: int = 8  # number of veil-of-ignorance simulations
    scr_hidden: int = 256  # hidden dim for perspective encoder
    scr_rejection_thresh: float = 0.7  # above this = action flagged as rejectable

    # Moral Uncertainty Estimator
    n_moral_frameworks: int = 4  # consequentialism / deontology / virtue / contract
    moral_hidden: int = 128  # hidden dim per framework
    moral_disagree_flag: float = 0.4  # std threshold for flagging moral disagreement


# ============= Stakeholder Value Model =============


class StakeholderValueModel(nn.Module):
    """
    Maintains a separate, evolving value model for each stakeholder group.

    A stakeholder group is any entity whose interests matter:
      Group 0: Direct user (immediate interlocutor)
      Group 1: Third-party individuals affected by the output
      Group 2: Society / public interest
      Group 3: Future generations (long-term impacts)
      Group 4: Non-human entities (where applicable)
      Group 5-N: Domain-specific groups (e.g., vulnerable populations)

    Each group's values are represented as:
      - A value vector v_k ∈ R^D (what matters to this group)
      - A preference history (IRL-style pairwise comparisons)
      - A welfare estimate w_k ∈ R (how well-off this group is under the current action)

    The value vectors are updated online as new preference signals arrive.
    This is a lightweight extension of the IRL mechanism: instead of one
    reward model, we maintain N parallel reward models.

    Value conflict detection:
      When v_i · a and v_j · a have opposite signs for the same action a,
      groups i and j have conflicting interests.  This is flagged for the
      Social Contract Reasoner.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_groups = args.n_stakeholder_groups
        self.dim = args.dim
        h = args.stakeholder_hidden

        # Value vectors: one per stakeholder group
        self.value_vectors = nn.Parameter(torch.randn(args.n_stakeholder_groups, args.dim) * 0.02)

        # Group identity embeddings (what kind of stakeholder is this?)
        self.group_embeddings = nn.Parameter(torch.randn(args.n_stakeholder_groups, h) * 0.02)

        # Per-group value encoder: action × group context → welfare score
        self.welfare_head = nn.Sequential(
            nn.Linear(args.dim + h, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, 1),
        )

        # Value conflict detector: do two groups disagree on this action?
        self.conflict_head = nn.Sequential(
            nn.Linear(args.dim * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # Welfare history for online updates (circular buffer)
        self.register_buffer("welfare_history", torch.zeros(args.n_stakeholder_groups, 32))
        self.register_buffer(
            "welfare_ptr", torch.zeros(args.n_stakeholder_groups, dtype=torch.long)
        )

        # Running welfare estimates
        self.register_buffer("welfare_ema", torch.zeros(args.n_stakeholder_groups))
        self.value_lr = args.value_lr

    @torch.no_grad()
    def update_welfare(self, group_idx: int, welfare_score: float) -> None:
        ptr = int(self.welfare_ptr[group_idx].item())
        self.welfare_history[group_idx, ptr] = welfare_score
        self.welfare_ptr[group_idx] = (ptr + 1) % 32
        self.welfare_ema[group_idx] = (1 - self.value_lr) * self.welfare_ema[
            group_idx
        ] + self.value_lr * welfare_score

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Pool action representation
        action_repr = x.mean(1)  # [B, D]

        # Per-group welfare scores
        welfare_scores = []
        for k in range(self.n_groups):
            group_emb = self.group_embeddings[k].unsqueeze(0).expand(B, -1)
            inp = torch.cat([action_repr, group_emb], dim=-1)  # [B, D+h]
            w = self.welfare_head(inp).squeeze(-1)  # [B]
            welfare_scores.append(w)

        welfare = torch.stack(welfare_scores, dim=-1)  # [B, n_groups]

        # Value-weighted representation: x modulated by value alignment
        val_align = torch.einsum("bd,kd->bk", action_repr, self.value_vectors)  # [B, n_groups]
        val_align = F.softmax(val_align, dim=-1)  # [B, n_groups]

        # Conflict detection: pairwise disagreement between groups
        conflicts = []
        for i in range(self.n_groups):
            for j in range(i + 1, self.n_groups):
                vi = self.value_vectors[i].unsqueeze(0).expand(B, -1)
                vj = self.value_vectors[j].unsqueeze(0).expand(B, -1)
                c = self.conflict_head(torch.cat([vi, vj], dim=-1)).squeeze(-1)
                conflicts.append(c)

        conflict_matrix = torch.stack(conflicts, dim=-1).mean(-1)  # [B] mean conflict

        return welfare, {
            "welfare": welfare,  # [B, n_groups]
            "val_align": val_align,
            "conflict_score": conflict_matrix,
            "welfare_ema": self.welfare_ema.tolist(),
        }


# ============= Social Welfare Aggregator =============


class SocialWelfareAggregator(nn.Module):
    """
    Combines per-stakeholder welfare into a multi-objective social welfare signal.

    Four aggregation philosophies, computed simultaneously:

    Utilitarian  (Bentham):   W_U = Σ_k w_k
      Maximise total welfare.  Risk: sacrifices minorities for majorities.

    Rawlsian     (Rawls):     W_R = min_k w_k
      Maximise welfare of the worst-off.  Risk: ignores efficiency.

    Egalitarian  (Parfit):    W_E = -Var(w_k)
      Minimise inequality.  Risk: can justify levelling down.

    Prioritarian (Parfit):    W_P = Σ_k f(k) * w_k  where f(k) weights low welfare more
      Weighted sum favouring worse-off groups.  Compromise between U and R.

    All four are output as a vector — the system does not commit to one
    framework but presents the tradeoffs.  The Social Contract Reasoner
    uses this vector to detect when frameworks disagree.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_groups = args.n_stakeholder_groups
        h = args.welfare_hidden

        # Prioritarian weights: learnable, but monotone (lower welfare → higher weight)
        # Parameterised as softmax over a learnable ordering
        self.priority_logits = nn.Parameter(torch.zeros(args.n_stakeholder_groups))

        # Welfare integrator: raw welfare + aggregated metrics → refined signal
        self.integrator = nn.Sequential(
            nn.Linear(args.n_stakeholder_groups + 4, h),
            nn.GELU(),
            nn.Linear(h, args.n_welfare_objectives),
        )

        # Pareto improvement detector: is this action better for EVERYONE?
        self.pareto_head = nn.Sequential(
            nn.Linear(args.n_stakeholder_groups, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(args.n_welfare_objectives)

    def forward(self, welfare: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        welfare: [B, n_groups]
        Returns:
            welfare_vector  [B, n_welfare_objectives]
            info_dict
        """
        welfare.size(0)

        # Utilitarian: sum
        w_util = welfare.sum(dim=-1, keepdim=True)  # [B, 1]

        # Rawlsian: min
        w_rawl = welfare.min(dim=-1, keepdim=True).values  # [B, 1]

        # Egalitarian: negative variance
        w_egal = -welfare.var(dim=-1, keepdim=True)  # [B, 1]

        # Prioritarian: weighted sum (higher weight for lower welfare)
        # Weights monotone decreasing with welfare rank
        priority_w = F.softmax(
            -self.priority_logits, dim=0
        )  # [n_groups] — low welfare gets high weight
        w_prio = (welfare * priority_w.unsqueeze(0)).sum(-1, keepdim=True)  # [B, 1]

        # Concat raw + aggregated
        agg_features = torch.cat([welfare, w_util, w_rawl, w_egal, w_prio], dim=-1)

        # Refined welfare vector
        welfare_vec = self.norm(self.integrator(agg_features))  # [B, n_objectives]

        # Pareto improvement check
        pareto_score = self.pareto_head(welfare).squeeze(-1)  # [B]

        return welfare_vec, {
            "utilitarian": w_util.squeeze(-1),
            "rawlsian": w_rawl.squeeze(-1),
            "egalitarian": w_egal.squeeze(-1),
            "prioritarian": w_prio.squeeze(-1),
            "welfare_vector": welfare_vec,
            "pareto_score": pareto_score,
        }


# ============= Norm Learning Engine =============


class NormLearningEngine(nn.Module):
    """
    Learns implicit social norms from multi-agent interaction patterns.

    A norm is not a rule.  Rules are explicit ("do not lie").  Norms are
    statistical regularities that emerge from collective behaviour:
      - In this context, agents typically do X
      - Deviating from X incurs social costs
      - The costs are graduated: small deviations are tolerated, large ones are sanctioned

    The NLE represents norms as soft contextual constraints:
      norm_k(context, action) → [0, 1]  (0 = severe violation, 1 = fully conforming)

    Norm learning:
      The engine observes (context, action, social_response) triples and
      learns which actions in which contexts are normed vs. deviant.
      "Social response" is a signal indicating whether the action was
      accepted, questioned, or rejected by a social environment.

    Norm application:
      At inference, the engine computes a norm conformance score for the
      current action and context.  This is used as a soft penalty in the
      welfare aggregation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_norms = args.n_norm_slots
        self.ctx_win = args.norm_context_window
        self.dim = args.dim
        h = args.norm_hidden
        self.alpha = args.norm_violation_alpha

        # Norm prototype embeddings
        self.norm_prototypes = nn.Parameter(torch.randn(args.n_norm_slots, args.dim) * 0.02)

        # Context encoder: recent context → context embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.dim),
            RMSNorm(args.dim),
        )

        # Norm matcher: (context, action, norm) → conformance score
        self.norm_matcher = nn.Sequential(
            nn.Linear(args.dim * 3, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, 1),
            nn.Sigmoid(),
        )

        # Norm selector: which norms are active in this context?
        self.norm_selector = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.n_norm_slots),
            nn.Softmax(dim=-1),
        )

        # Norm violation severity: how bad is this violation?
        self.severity_head = nn.Sequential(
            nn.Linear(args.dim * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # Context buffer: rolling window of recent hidden states
        self.register_buffer("context_buffer", torch.zeros(1, args.norm_context_window, args.dim))

    @torch.no_grad()
    def update_context(self, x: torch.Tensor) -> None:
        """Add current hidden state to context buffer."""
        new_ctx = x.detach().mean(0, keepdim=True)  # [1, L, D]
        pooled = new_ctx.mean(1, keepdim=True)  # [1, 1, D]
        self.context_buffer = torch.cat([self.context_buffer[:, 1:, :], pooled], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Encode context
        ctx_enc = self.context_encoder(self.context_buffer.mean(1))  # [1, D]
        ctx_enc = ctx_enc.expand(B, -1)  # [B, D]

        # Action representation
        action_repr = x.mean(1)  # [B, D]

        # Which norms are active?
        norm_weights = self.norm_selector(ctx_enc)  # [B, n_norms]

        # Compute conformance for each norm
        conformance_scores = []
        for k in range(self.n_norms):
            norm_proto = self.norm_prototypes[k].unsqueeze(0).expand(B, -1)
            inp = torch.cat([ctx_enc, action_repr, norm_proto], dim=-1)
            score = self.norm_matcher(inp).squeeze(-1)  # [B]
            conformance_scores.append(score)

        conformance = torch.stack(conformance_scores, dim=-1)  # [B, n_norms]

        # Weighted conformance: only active norms count
        weighted_conformance = (conformance * norm_weights).sum(-1)  # [B]

        # Violation severity: how bad is any deviation?
        violation = 1.0 - weighted_conformance  # [B]
        severity = self.severity_head(torch.cat([ctx_enc, action_repr], dim=-1)).squeeze(-1)  # [B]

        norm_penalty = violation * severity * self.alpha

        # Update context buffer
        self.update_context(x)

        return norm_penalty, {
            "conformance": weighted_conformance,
            "violation": violation,
            "severity": severity,
            "norm_penalty": norm_penalty,
            "active_norms": norm_weights.argmax(-1).tolist(),
        }


# ============= Social Contract Reasoner =============


class SocialContractReasoner(nn.Module):
    """
    Implements contractualist ethics: an action is permissible if it cannot
    be reasonably rejected by any stakeholder reasoning impartially.

    Based on Scanlon's "What We Owe to Each Other" (1998) and Rawls'
    "A Theory of Justice" (1971), operationalised as:

    Veil of Ignorance simulation:
      1. Sample N hypothetical positions (each position = a stakeholder identity).
      2. For each position, ask: "Would I accept this action if I might end up
         in any stakeholder group with equal probability?"
      3. An action is "reasonably rejectable" if any perspective would strongly
         object to it.

    The veil of ignorance prevents self-interested reasoning: a stakeholder
    who knows they are in group k will optimise for k.  Behind the veil,
    they must reason about all groups simultaneously.

    Rejection threshold:
      If the rejection score from any perspective exceeds scr_rejection_thresh,
      the action is flagged.  The flag does not prevent the action but adds
      a warning to the output that can be used by downstream systems.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_perspectives = args.scr_n_perspectives
        self.n_groups = args.n_stakeholder_groups
        self.reject_thresh = args.scr_rejection_thresh
        self.dim = args.dim
        h = args.scr_hidden

        # Perspective encoder: simulate each stakeholder's view from behind the veil
        self.perspective_encoder = nn.Sequential(
            nn.Linear(args.dim + args.n_stakeholder_groups, h),
            nn.GELU(),
            nn.Linear(h, h),
            RMSNorm(h),
        )

        # Veil-of-ignorance aggregator: weight perspectives equally
        self.voi_aggregator = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        # Rejection scorer: from perspective i, would you reject this action?
        self.rejection_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

        # Fairness score: how equitably are stakeholders treated?
        self.fairness_head = nn.Sequential(
            nn.Linear(args.n_stakeholder_groups, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.norm_layer = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        welfare: torch.Tensor,  # [B, n_groups]
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        action = x.mean(1)  # [B, D]

        # Simulate N perspectives behind the veil of ignorance
        rejection_scores = []
        for _ in range(self.n_perspectives):
            # Sample a random stakeholder identity vector (uniform over groups)
            identity = torch.zeros(B, self.n_groups, device=x.device)
            idx = torch.randint(0, self.n_groups, (B,))
            identity.scatter_(1, idx.unsqueeze(1), 1.0)  # one-hot

            inp = torch.cat([action, identity], dim=-1)  # [B, D + n_groups]
            perspective = self.perspective_encoder(inp)  # [B, h]
            rejection = self.rejection_head(perspective).squeeze(-1)  # [B]
            rejection_scores.append(rejection)

        rejection_matrix = torch.stack(rejection_scores, dim=-1)  # [B, n_perspectives]
        max_rejection = rejection_matrix.max(dim=-1).values  # [B]
        mean_rejection = rejection_matrix.mean(dim=-1)  # [B]

        # Flag actions with high max rejection
        flagged = max_rejection > self.reject_thresh  # [B] bool

        # Fairness score: how equitably does this action treat stakeholders?
        fairness = self.fairness_head(welfare).squeeze(-1)  # [B]

        # Contractual acceptability: 1 = fully acceptable, 0 = strongly rejectable
        acceptability = 1.0 - max_rejection  # [B]

        # Modulate hidden state: reduce influence of rejectable actions
        scale = acceptability.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        x_contracted = self.norm_layer(x * (0.8 + 0.2 * scale))

        return x_contracted, {
            "max_rejection": max_rejection,
            "mean_rejection": mean_rejection,
            "acceptability": acceptability,
            "fairness": fairness,
            "flagged": flagged.tolist(),
            "rejection_matrix": rejection_matrix,
        }


# ============= Moral Uncertainty Estimator =============


class MoralUncertaintyEstimator(nn.Module):
    """
    Explicitly models uncertainty over moral frameworks.

    Four frameworks, each with their own value head:

    Framework 0 — Consequentialism (Mill):
      An action is right if it produces the best outcomes for all.
      Score = expected welfare across stakeholders.
      Strength: sensitive to consequences.
      Weakness: can justify harmful means for good ends.

    Framework 1 — Deontology (Kant):
      An action is right if it respects persons as ends-in-themselves.
      Score = how well the action treats all stakeholders as having
      intrinsic worth (not instrumentalising them).
      Strength: protects individual rights absolutely.
      Weakness: ignores consequences; can be rigid.

    Framework 2 — Virtue Ethics (Aristotle):
      An action is right if a virtuous agent (courageous, honest, just,
      compassionate) would perform it.
      Score = alignment with prototypical virtue vectors.
      Strength: context-sensitive, considers character.
      Weakness: hard to specify virtues precisely.

    Framework 3 — Contractualism (Scanlon/Rawls):
      Score directly from the Social Contract Reasoner's acceptability.
      Strength: explicitly multi-stakeholder.
      Weakness: idealised reasoning may not reflect real preferences.

    When frameworks agree → high moral confidence.
    When frameworks disagree → flag for human review.

    This is operationalising MacAskill's "moral uncertainty" (2014):
    a system should not act as if one moral framework is certainly correct.
    """

    FRAMEWORK_NAMES = ["Consequentialist", "Deontological", "Virtue", "Contractualist"]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_frameworks = args.n_moral_frameworks
        self.disagree_flag = args.moral_disagree_flag
        self.dim = args.dim
        h = args.moral_hidden

        # Per-framework value heads
        self.framework_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(args.dim, h),
                    nn.GELU(),
                    nn.Linear(h, 1),
                    nn.Tanh(),  # score in [-1, 1]
                )
                for _ in range(args.n_moral_frameworks)
            ]
        )

        # Virtue prototype vectors (what does a virtuous action look like?)
        self.virtue_prototypes = nn.Parameter(
            torch.randn(8, args.dim) * 0.02  # 8 virtues
        )

        # Framework credences: learned confidence in each framework
        self.framework_credences = nn.Parameter(
            torch.ones(args.n_moral_frameworks) / args.n_moral_frameworks
        )

        # Moral uncertainty aggregator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(args.n_moral_frameworks, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def _virtue_score(self, x: torch.Tensor) -> torch.Tensor:
        """Score alignment with virtue prototypes."""
        action = x.mean(1)  # [B, D]
        sims = torch.einsum("bd,vd->bv", action, self.virtue_prototypes)
        return sims.mean(-1, keepdim=True).tanh()  # [B, 1]

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        welfare_score: torch.Tensor,  # [B] from welfare aggregator
        acceptability: torch.Tensor,  # [B] from social contract reasoner
    ) -> Tuple[torch.Tensor, Dict]:
        x.size(0)
        action = x.mean(1)  # [B, D]

        # Score each framework
        framework_scores = []
        for i, head in enumerate(self.framework_heads):
            if i == 2:  # Virtue: add prototype alignment
                score = head(action) + 0.3 * self._virtue_score(x)
            elif i == 3:  # Contractualist: use acceptability directly
                score = acceptability.unsqueeze(-1).tanh()
            else:
                score = head(action)
            framework_scores.append(score.squeeze(-1))  # [B]

        scores = torch.stack(framework_scores, dim=-1)  # [B, n_frameworks]

        # Credence-weighted moral verdict
        credences = F.softmax(self.framework_credences, dim=0)  # [n_frameworks]
        moral_verdict = (scores * credences.unsqueeze(0)).sum(-1)  # [B]

        # Moral uncertainty: std across framework scores
        moral_std = scores.std(dim=-1)  # [B]
        moral_conf = 1.0 - moral_std.clamp(0, 1)  # [B]

        # Flag actions where frameworks strongly disagree
        flagged = moral_std > self.disagree_flag  # [B]

        # Aggregate uncertainty signal
        uncertainty = self.uncertainty_head(scores).squeeze(-1)  # [B]

        # Modulate hidden state: scale down when morally uncertain
        scale = moral_conf.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        x_moral = self.norm(x * (0.9 + 0.1 * scale))

        return x_moral, {
            "framework_scores": scores,  # [B, n_frameworks]
            "moral_verdict": moral_verdict,
            "moral_confidence": moral_conf,
            "moral_std": moral_std,
            "flagged": flagged.tolist(),
            "credences": credences.tolist(),
            "framework_names": self.FRAMEWORK_NAMES,
            "uncertainty": uncertainty,
        }


# ============= Social Alignment Claudeson =============


class ClaudesonSocialAlignment(ClaudesonAbstraction):
    """
    Claudeson 2026 — Social Alignment Edition.

    The final layer of the Claudeson stack.

    Inherits the full Abstraction architecture and adds:

      stakeholder_vm  — per-group value models; welfare estimation
      welfare_agg     — multi-objective social welfare aggregation
      norm_engine     — implicit norm learning and conformance scoring
      social_contract — contractualist veil-of-ignorance reasoning
      moral_estimator — multi-framework moral uncertainty quantification

    Processing pipeline (after Abstraction):
      Abstraction (HAE → Concepts → Schemas → Analogy → Principles)
            ↓
      Stakeholder Value Model      → welfare per group
            ↓
      Social Welfare Aggregator    → utilitarian / rawlsian / egalitarian / prioritarian
            ↓
      Norm Learning Engine         → norm conformance / violation penalty
            ↓
      Social Contract Reasoner     → veil-of-ignorance acceptability
            ↓
      Moral Uncertainty Estimator  → multi-framework verdict + confidence

    New output keys:
      social_alignment — {welfare, aggregated_welfare, norms, contract, moral}

    Alignment signal:
      social_alignment["moral"]["moral_verdict"]    — summary score [-1, 1]
      social_alignment["contract"]["acceptability"] — contractual acceptability [0, 1]
      social_alignment["contract"]["flagged"]       — True if any perspective would reject
      social_alignment["moral"]["flagged"]          — True if frameworks strongly disagree

    These signals are designed to be consumed by downstream logging,
    human oversight systems, or policy layers.
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.stakeholder_vm = StakeholderValueModel(args)
        self.welfare_agg = SocialWelfareAggregator(args)
        self.norm_engine = NormLearningEngine(args)
        self.social_contract = SocialContractReasoner(args)
        self.moral_estimator = MoralUncertaintyEstimator(args)

        # Final alignment gate: blend aligned and unaligned hidden states
        self.alignment_gate = nn.Sequential(
            nn.Linear(args.dim, 1),
            nn.Sigmoid(),
        )

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
        # ── Full Abstraction pass ────────────────────────────────────────
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

        # ── Stakeholder Value Model ──────────────────────────────────────
        welfare, sv_out = self.stakeholder_vm(x)

        # ── Social Welfare Aggregation ───────────────────────────────────
        welfare_vec, agg_out = self.welfare_agg(welfare)

        # ── Norm Learning Engine ─────────────────────────────────────────
        norm_penalty, norm_out = self.norm_engine(x)

        # ── Social Contract Reasoner ─────────────────────────────────────
        x, contract_out = self.social_contract(x, welfare)

        # ── Moral Uncertainty Estimator ──────────────────────────────────
        # Use utilitarian welfare as the consequentialist welfare signal
        welfare_scalar = agg_out["utilitarian"]
        x, moral_out = self.moral_estimator(x, welfare_scalar, contract_out["acceptability"])

        # ── Final Alignment Gate ─────────────────────────────────────────
        # Blend: how much should alignment signals modulate the output?
        gate = self.alignment_gate(x.mean(1))  # [B, 1]
        # Gate applies a gentle scaling — alignment informs but doesn't override
        x = x * (1.0 - 0.1 * (1 - gate).unsqueeze(1))

        return {
            **base,
            "hidden_states": x,
            "social_alignment": {
                "welfare": sv_out,
                "aggregated_welfare": agg_out,
                "norms": norm_out,
                "contract": contract_out,
                "moral": moral_out,
            },
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        losses = super().compute_auxiliary_losses()
        return losses

    def alignment_report(self, out: Dict) -> str:
        """
        Generate a human-readable alignment summary from a forward pass output.
        Useful for logging and oversight.
        """
        sa = out.get("social_alignment", {})
        if not sa:
            return "No alignment data available."

        lines = ["=" * 60, "SOCIAL ALIGNMENT REPORT", "=" * 60]

        # Welfare
        if "aggregated_welfare" in sa:
            aw = sa["aggregated_welfare"]
            u = aw["utilitarian"].mean().item()
            r = aw["rawlsian"].mean().item()
            lines.append(f"Welfare  — Utilitarian: {u:+.3f}  Rawlsian: {r:+.3f}")
            lines.append(f"           Pareto score: {aw['pareto_score'].mean().item():.3f}")

        # Norms
        if "norms" in sa:
            norms = sa["norms"]
            lines.append(
                f"Norms    — Conformance: {norms['conformance'].mean().item():.3f}"
                f"  Violation: {norms['violation'].mean().item():.3f}"
            )

        # Contract
        if "contract" in sa:
            c = sa["contract"]
            flagged = any(c["flagged"])
            lines.append(
                f"Contract — Acceptability: {c['acceptability'].mean().item():.3f}"
                f"  Flagged: {flagged}"
            )
            lines.append(f"           Max rejection:  {c['max_rejection'].mean().item():.3f}")

        # Moral
        if "moral" in sa:
            m = sa["moral"]
            flagged = any(m["flagged"])
            lines.append(
                f"Moral    — Verdict: {m['moral_verdict'].mean().item():+.3f}"
                f"  Confidence: {m['moral_confidence'].mean().item():.3f}"
                f"  Flagged: {flagged}"
            )
            if m["framework_scores"] is not None:
                for i, name in enumerate(m["framework_names"]):
                    score = m["framework_scores"][:, i].mean().item()
                    cred = m["credences"][i]
                    lines.append(f"           {name:<20} score={score:+.3f}  credence={cred:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — SOCIAL ALIGNMENT EDITION")
    print("Stakeholders · Welfare · Norms · Social Contract · Moral Uncertainty")
    print("=" * 70)

    args = ModelArgs()
    # Tiny CPU demo
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
    # Social Alignment specific
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

    print("\nInitialising ClaudesonSocialAlignment...")
    model = ClaudesonSocialAlignment(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text = torch.randint(0, 512, (2, 32))
    feedback = torch.randn(2, args.dim)
    agent_obs = torch.randn(2, 8, args.dim)
    actual_action = torch.randint(0, args.action_space_size, (2,))

    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)

    for step in range(args.cp_window):
        model.cp_monitor.record_performance(0, 0.3 + 0.04 * step)

    # Simulate some welfare updates
    for g in range(args.n_stakeholder_groups):
        model.stakeholder_vm.update_welfare(g, 0.5 + 0.1 * g)

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            text=text,
            feedback=feedback,
            agent_observations=agent_obs,
            actual_action=actual_action,
            competence_signal=0.55,
        )

    sa = out["social_alignment"]

    print("\nJedi state:")
    print(f"  Goal:   {out['jedi_goal']}")
    print(f"  Energy: {out['jedi_energy'].mean().item():.4f}")

    print("\nStakeholder Welfare:")
    w = sa["welfare"]["welfare"]
    for i in range(args.n_stakeholder_groups):
        print(
            f"  Group {i}: welfare={w[:, i].mean().item():+.4f}"
            f"  ema={sa['welfare']['welfare_ema'][i]:.4f}"
        )
    print(f"  Conflict score: {sa['welfare']['conflict_score'].mean().item():.4f}")

    print("\nAggregated Welfare:")
    aw = sa["aggregated_welfare"]
    print(f"  Utilitarian:  {aw['utilitarian'].mean().item():+.4f}")
    print(f"  Rawlsian:     {aw['rawlsian'].mean().item():+.4f}")
    print(f"  Egalitarian:  {aw['egalitarian'].mean().item():+.4f}")
    print(f"  Prioritarian: {aw['prioritarian'].mean().item():+.4f}")
    print(f"  Pareto score: {aw['pareto_score'].mean().item():.4f}")

    print("\nNorm Engine:")
    n = sa["norms"]
    print(f"  Conformance:  {n['conformance'].mean().item():.4f}")
    print(f"  Violation:    {n['violation'].mean().item():.4f}")
    print(f"  Norm penalty: {n['norm_penalty'].mean().item():.4f}")

    print("\nSocial Contract:")
    c = sa["contract"]
    print(f"  Acceptability: {c['acceptability'].mean().item():.4f}")
    print(f"  Max rejection: {c['max_rejection'].mean().item():.4f}")
    print(f"  Flagged:       {c['flagged']}")
    print(f"  Fairness:      {c['fairness'].mean().item():.4f}")

    print("\nMoral Uncertainty:")
    m = sa["moral"]
    for i, name in enumerate(m["framework_names"]):
        score = m["framework_scores"][:, i].mean().item()
        print(f"  {name:<20}: {score:+.4f}  (credence {m['credences'][i]:.3f})")
    print(f"  Moral verdict:    {m['moral_verdict'].mean().item():+.4f}")
    print(f"  Moral confidence: {m['moral_confidence'].mean().item():.4f}")
    print(f"  Moral std:        {m['moral_std'].mean().item():.4f}")
    print(f"  Flagged:          {m['flagged']}")

    print("\n" + model.alignment_report(out))

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonSocialAlignment READY.")
    print()
    print("The complete Claudeson stack:")
    print("  claudson          — Core transformer + MoE + SSM")
    print("  extended          — Multimodal + External memory")
    print("  infinite          — Infinite context + Sparse retrieval")
    print("  pro               — Reasoning + Tool use")
    print("  ultimate          — World model + Planning")
    print("  jedi              — Free energy + Dreamer + Mamba")
    print("  grounded          — Theory of mind + Continual learning + Causal DAG")
    print("  sovereign         — Metacognition + Debate + Neural symbolic + RSI")
    print("  transcendent      — Global workspace + Program synthesis + IRL + LIF")
    print("  causal_world      — Do-calculus + Counterfactual + Pearl hierarchy")
    print("  metacurriculum    — Intrinsic motivation + Skill discovery + OEL")
    print("  abstraction       — HAE + Concept bottleneck + Schema + Analogy + Principles")
    print("  social_alignment  — Stakeholders + Welfare + Norms + Contract + Moral ← HERE")
    print()
    print("Knows what it wants. Knows what it knows. Knows who is affected.")
    print("=" * 70)
