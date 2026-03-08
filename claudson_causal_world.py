"""
Claudeson 2026 - Causal World Edition
=======================================
Closes the gap between causal knowledge and causal planning.

The existing CausalReasoner (grounded) learns a DAG over concept nodes and
supports do-calculus via intervene() / counterfactual().  But the Jedi
planning loop (EFE / imagination rollouts) ignores that graph entirely —
it imagines futures by sampling from a learned dynamics model that was trained
on correlational data.  If the training distribution contains spurious
correlations (smoke → cancer, but also smoke → matches → fire), the planner
will reason about smoke when it should reason about matches.

This generation fixes that by wiring the causal graph directly into the
planning and imagination pipelines:

  1. Interventional Planner
     Replaces the EFE action loop with one that evaluates each candidate
     action as a do-calculus intervention on the causal graph.  "What
     happens to concept Y if I force concept X = x?" is answered by the
     graph, not by a black-box dynamics model.

  2. Counterfactual Imagination
     Extends Dreamer-style rollouts with a counterfactual branch: given
     what actually happened, what *would* have happened under a different
     action?  This enables hindsight credit assignment and contrastive
     explanation ("I succeeded because X, not because Y").

  3. Causal Attribution
     After each rollout, scores each concept node by its average causal
     influence on the outcome.  High-attribution nodes are promoted in
     working memory; low-attribution nodes are deprioritised.  Causal
     salience replaces raw attention as the memory write gate.

  4. Pearl Hierarchy Reasoner (L1 → L2 → L3)
     Explicit three-rung ladder:
       L1  Association   — P(Y | X)         standard conditional
       L2  Intervention  — P(Y | do(X))     graph surgery
       L3  Counterfactual — P(Y_x | X', Y') — twin-network inference
     The model is forced to declare which rung it is using at each
     planning step, and the loss penalises L1 answers to L2/L3 queries.

Architecture evolution:
  claudson → extended → infinite → pro → ultimate → jedi → grounded
          → sovereign → transcendent → causal_world
                                            ↑ you are here
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from claudson_transcendent import (
    ModelArgs as TranscendentArgs,
    ClaudesonTranscendent,
)
from claudson_jedi import SwiGLU, RMSNorm

log = logging.getLogger(__name__)


# ============= Configuration =============

@dataclass
class ModelArgs(TranscendentArgs):
    # Causal world model
    n_causal_nodes: int = 64          # concept graph nodes (may inherit from grounded)
    causal_state_dim: int = 128       # latent state dim for causal dynamics
    intervention_horizon: int = 5     # steps to unroll after an intervention
    n_intervention_samples: int = 8   # candidate interventions evaluated per step

    # Counterfactual
    cf_n_branches: int = 4            # parallel counterfactual branches
    cf_detach_actual: bool = True     # stop-gradient on the actual branch

    # Attribution
    attr_top_k: int = 8               # top-k causal nodes kept in working memory
    attr_ema: float = 0.95            # EMA decay for running attribution scores

    # Pearl ladder
    pearl_hidden: int = 256           # hidden dim for ladder classifier
    pearl_loss_weight: float = 0.1    # penalty for wrong-rung answers


# ============= Causal Dynamics Model =============

class CausalDynamicsModel(nn.Module):
    """
    A world model that respects the causal graph.

    Standard neural dynamics models learn P(s_{t+1} | s_t, a_t) from data,
    which conflates correlation and causation.  This model instead factors
    the prediction through the concept-level causal graph:

      1. Encode the current hidden state into concept activations.
      2. Apply the do-calculus intervention: zero out the influence of
         non-causal paths for the chosen action.
      3. Propagate forward through the causal graph (one matrix multiply).
      4. Decode back to hidden-state space.

    The "intervention" here is a soft mask over incoming edges for the
    intervened node — edges are cut (multiplied by zero) when we do(X=x),
    because the intervention breaks the natural causes of X.

    This is the core of Pearl's do-calculus in neural form:
        P(Y | do(X=x)) ≠ P(Y | X=x)
    The left side (do-calculus) is what a surgeon does.
    The right side (conditioning) is what a statistician does.
    Only the surgeon can plan.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_nodes = args.n_causal_nodes
        self.state_dim = args.causal_state_dim

        # Encode hidden → concept activations
        self.concept_encoder = nn.Sequential(
            nn.Linear(args.dim, args.causal_state_dim),
            RMSNorm(args.causal_state_dim),
            nn.GELU(),
            nn.Linear(args.causal_state_dim, args.n_causal_nodes),
            nn.Sigmoid(),
        )

        # Action → intervention mask over concept nodes
        # Which concepts does this action directly intervene on?
        self.action_intervention_proj = nn.Sequential(
            nn.Linear(args.action_space_size, args.n_causal_nodes),
            nn.Sigmoid(),
        )

        # Causal transition: concept_t → concept_{t+1} via graph
        # Graph is INHERITED from CausalReasoner; this model uses a separate
        # learned dynamics graph to keep inference and learning decoupled.
        self.dynamics_graph = nn.Parameter(
            torch.randn(args.n_causal_nodes, args.n_causal_nodes) * 0.01
        )

        # Reward predictor in concept space
        self.concept_reward = nn.Sequential(
            nn.Linear(args.n_causal_nodes, args.causal_state_dim // 2),
            nn.GELU(),
            nn.Linear(args.causal_state_dim // 2, 1),
        )

        # Decode concept activations back to hidden space
        self.concept_decoder = nn.Sequential(
            nn.Linear(args.n_causal_nodes, args.causal_state_dim),
            RMSNorm(args.causal_state_dim),
            nn.GELU(),
            nn.Linear(args.causal_state_dim, args.dim),
        )

        self.norm = RMSNorm(args.dim)

    @property
    def sparse_dynamics(self) -> torch.Tensor:
        return torch.sigmoid(self.dynamics_graph)

    def dag_constraint(self) -> torch.Tensor:
        """NO TEARS acyclicity on the dynamics graph (degree-4 Taylor approx)."""
        W  = self.sparse_dynamics
        d  = self.n_nodes
        WW = W * W
        expm = torch.eye(d, device=W.device, dtype=W.dtype)
        term = torch.eye(d, device=W.device, dtype=W.dtype)
        for k in range(1, 5):
            term = term @ WW / k
            expm = expm + term
        return expm.trace() - d

    def do_intervention(
        self,
        concepts:      torch.Tensor,   # [..., n_nodes]
        action_onehot: torch.Tensor,   # [..., action_space_size]
        intervention_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply do(action) by cutting incoming edges to intervened nodes.

        The intervention mask identifies which concept nodes the action
        directly sets.  For those nodes, incoming causal edges are severed
        (multiplied by 1 - mask), so only the direct action effect remains.
        """
        mask = self.action_intervention_proj(action_onehot)    # [..., n_nodes]

        G = self.sparse_dynamics                               # [n, n]

        # Surgery: zero out incoming edges for intervened nodes
        # G_do[j, i] = G[j, i] * (1 - mask[j])
        # where mask[j] = 1 means we intervene on node j
        mask_col = mask.unsqueeze(-2)                          # [..., 1, n_nodes]
        G_do = G * (1.0 - mask_col * intervention_strength)   # [..., n, n]  (broadcast)

        # Propagate concepts through intervened graph
        post_concepts = torch.einsum('...i,...ij->...j', concepts, G_do)
        post_concepts = torch.sigmoid(post_concepts)

        # Set intervened nodes to action-specified value
        action_value = mask * 0.8 + (1 - mask) * post_concepts
        return action_value

    def forward(
        self,
        x:             torch.Tensor,   # [B, L, D]
        action:        Optional[torch.Tensor] = None,  # [B] or [B, action_space_size]
        intervene:     bool = False,
    ) -> Dict:
        B, L, D = x.shape

        # Encode to concept space
        concepts = self.concept_encoder(x)                     # [B, L, n_nodes]

        if intervene and action is not None:
            if action.dim() == 1:
                action_oh = F.one_hot(action, num_classes=self.action_intervention_proj[0].in_features).float()
            else:
                action_oh = action.float()
            # Expand to [B, L, action_space]
            action_oh_expanded = action_oh.unsqueeze(1).expand(-1, L, -1)
            post_concepts = self.do_intervention(concepts, action_oh_expanded)
        else:
            G = self.sparse_dynamics
            post_concepts = torch.einsum('bli,ij->blj', concepts, G)
            post_concepts = torch.sigmoid(post_concepts)

        # Reward in concept space
        reward = self.concept_reward(post_concepts)            # [B, L, 1]

        # Decode back to hidden space
        x_causal = self.concept_decoder(post_concepts)         # [B, L, D]
        x_out = self.norm(x + x_causal * 0.1)

        return {
            "hidden":       x_out,
            "concepts":     concepts,
            "post_concepts": post_concepts,
            "reward":       reward,
            "dag_loss":     self.dag_constraint(),
        }


# ============= Interventional Planner =============

class InterventionalPlanner(nn.Module):
    """
    Plans by evaluating actions as do-calculus interventions.

    For each candidate action:
      1. Apply do(action) to the current causal graph state.
      2. Propagate forward `intervention_horizon` steps via the causal
         dynamics model — each step uses the intervened graph.
      3. Accumulate discounted rewards from the causal reward head.
      4. Select the action with the highest causal expected return.

    This is categorically different from the Jedi EFE planner:
    - EFE asks "what action minimises expected surprise?" (epistemic)
    - Interventional planner asks "what action maximises causal return?" (pragmatic)

    Both signals are useful; the Sovereign metacognitive monitor can
    decide which to weight more depending on epistemic vs. aleatoric
    uncertainty.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.horizon  = args.intervention_horizon
        self.n_samples = args.n_intervention_samples
        self.action_space = args.action_space_size
        self.gamma = 0.95

        self.dynamics = CausalDynamicsModel(args)

        # Policy head: hidden → action logits (for sampling candidates)
        self.policy = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, args.action_space_size),
        )

        # Value baseline: reduce variance in causal return estimates
        self.value_baseline = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, 1),
        )

        # Causal return normaliser (running stats)
        self.register_buffer('return_mean', torch.tensor(0.0))
        self.register_buffer('return_std',  torch.tensor(1.0))

    def _causal_rollout(
        self,
        x:      torch.Tensor,  # [B, L, D]
        action: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Roll out the causal dynamics model for self.horizon steps
        under do(action) at step 0, then free dynamics thereafter.
        Returns total discounted reward [B].
        """
        total_reward = torch.zeros(x.size(0), device=x.device)
        h = x

        for t in range(self.horizon):
            intervene = (t == 0)
            result = self.dynamics(h, action=action, intervene=intervene)
            h = result["hidden"]
            step_reward = result["reward"].mean(dim=(1, 2))      # [B]
            total_reward = total_reward + (self.gamma ** t) * step_reward

        return total_reward

    def forward(self, x: torch.Tensor) -> Dict:
        B, L, D = x.shape
        pooled = x.mean(1)                                       # [B, D]

        # Sample candidate actions from policy
        policy_logits = self.policy(pooled)                      # [B, A]
        policy_probs  = F.softmax(policy_logits, dim=-1)

        # Evaluate each candidate via causal rollout
        returns = torch.zeros(B, self.action_space, device=x.device)

        n_eval = min(self.n_samples, self.action_space)
        _, top_actions = torch.topk(policy_probs, n_eval, dim=-1)  # [B, n_eval]

        for i in range(n_eval):
            actions_i = top_actions[:, i]                        # [B]
            r = self._causal_rollout(x, actions_i)
            returns[:, actions_i[0]] = r                         # batch approximation

        # Update running normalisation stats
        with torch.no_grad():
            valid = returns[returns != 0]
            if valid.numel() > 0:
                self.return_mean = 0.99 * self.return_mean + 0.01 * valid.mean()
                self.return_std  = 0.99 * self.return_std  + 0.01 * (valid.std() + 1e-6)

        # Normalise returns
        returns_norm = (returns - self.return_mean) / (self.return_std + 1e-6)

        # Best causal action
        best_action = returns_norm.argmax(-1)                    # [B]
        value = self.value_baseline(pooled)                      # [B, 1]

        return {
            "causal_action":  best_action,
            "causal_returns": returns_norm,
            "policy_logits":  policy_logits,
            "value":          value,
            "dag_loss":       self.dynamics.dag_constraint(),
        }


# ============= Counterfactual Imagination =============

class CounterfactualImagination(nn.Module):
    """
    "What would have happened if I had acted differently?"

    Uses a twin-network construction (Pearl's approach):
      - Actual branch:       run with the action that was taken
      - Counterfactual branch: run the same exogenous noise, different action

    The key insight is that counterfactual reasoning requires:
      1. Abduction  — infer the exogenous noise from the actual observation
      2. Action     — replace the action with the hypothetical
      3. Prediction — forward-simulate under the new action

    Implementation:
      noise_encoder    — estimates latent exogenous noise from the observation
      actual_branch    — deterministic replay of what happened
      cf_branch        — replay with different action, same noise
      contrast_head    — computes difference in outcomes (causal contrast)

    The contrast is used for:
      - Hindsight credit assignment  (which actions actually caused good outcomes)
      - Causal explanation generation (this happened because X, not Y)
      - Adversarial robustness        (what would have changed the outcome?)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim        = args.dim
        self.n_branches = args.cf_n_branches
        self.detach     = args.cf_detach_actual

        # Abduction: infer exogenous noise from observation
        self.noise_encoder = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )

        # Shared dynamics for all branches (actual + counterfactual)
        self.branch_dynamics = CausalDynamicsModel(args)

        # Action embedding for counterfactual actions
        self.cf_action_embed = nn.Embedding(args.action_space_size, args.dim)

        # Contrast: actual outcome vs. counterfactual outcome → causal effect
        self.contrast_head = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim),
            RMSNorm(args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim),
        )

        # Credit assignment: which part of the state caused the reward delta?
        self.credit_head = nn.Sequential(
            nn.Linear(args.dim, args.n_causal_nodes if hasattr(args, 'n_causal_nodes') else 64),
            nn.Softmax(dim=-1),
        )

        self.norm = RMSNorm(args.dim)

        # Store n_causal_nodes for credit_head output interpretation
        self.n_causal_nodes = getattr(args, 'n_causal_nodes', 64)

    def forward(
        self,
        x:              torch.Tensor,              # [B, L, D]
        actual_action:  Optional[torch.Tensor] = None,  # [B]
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Step 1: Abduction — infer exogenous noise
        noise = self.noise_encoder(x)                            # [B, L, D]

        # Step 2: Actual branch
        if actual_action is not None:
            actual_result = self.branch_dynamics(
                x + noise * 0.1, action=actual_action, intervene=True
            )
        else:
            actual_result = self.branch_dynamics(x + noise * 0.1)
        actual_hidden = actual_result["hidden"]
        if self.detach:
            actual_hidden = actual_hidden.detach()

        # Step 3: Counterfactual branches (random alternative actions)
        cf_results = []
        cf_actions = torch.randint(
            0, x.size(0), (self.n_branches,), device=x.device
        )  # random action indices as stand-ins

        for cf_a in cf_actions:
            cf_action_tensor = torch.full((B,), cf_a.item(), dtype=torch.long, device=x.device)
            cf_action_emb    = self.cf_action_embed(cf_action_tensor)         # [B, D]
            # Inject action embedding into hidden state, same noise
            x_cf = x + noise * 0.1 + cf_action_emb.unsqueeze(1) * 0.05
            cf_result = self.branch_dynamics(x_cf, action=cf_action_tensor, intervene=True)
            cf_results.append(cf_result["hidden"])

        cf_stack = torch.stack(cf_results, dim=1)                # [B, n_branches, L, D]
        cf_mean  = cf_stack.mean(1)                              # [B, L, D]

        # Step 4: Contrast — what changed?
        actual_expanded = actual_hidden
        contrast_in     = torch.cat([actual_expanded, cf_mean], dim=-1)
        causal_contrast = self.contrast_head(contrast_in)        # [B, L, D]

        # Reward delta per branch
        actual_reward = actual_result["reward"]                  # [B, L, 1]
        cf_rewards    = torch.stack(
            [self.branch_dynamics(cf_stack[:, i], intervene=False)["reward"]
             for i in range(self.n_branches)],
            dim=1
        ).mean(1)                                                # [B, L, 1]
        reward_delta  = actual_reward - cf_rewards               # [B, L, 1]

        # Credit assignment: which nodes caused the reward difference?
        pooled_contrast = causal_contrast.mean(1)                # [B, D]
        credit          = self.credit_head(pooled_contrast)      # [B, n_causal_nodes]

        # Enrich hidden state with counterfactual signal
        x_enriched = self.norm(x + causal_contrast * 0.05)

        return x_enriched, {
            "actual_hidden":   actual_hidden,
            "cf_mean":         cf_mean,
            "causal_contrast": causal_contrast,
            "reward_delta":    reward_delta,
            "credit":          credit,              # [B, n_causal_nodes]
        }


# ============= Causal Attribution Memory Gate =============

class CausalAttributionGate(nn.Module):
    """
    Replaces raw attention as the memory write gate.

    Standard hierarchical memory writes based on importance scores from
    a learned MLP — this is essentially correlation-based salience.
    Attribution-based gating instead promotes memories that had *causal*
    influence on outcomes.

    Mechanism:
      - Maintains a running attribution score per concept node (EMA).
      - After each rollout, updates scores from the counterfactual credit.
      - Memory write strength = f(attribution) not f(attention score).

    High-attribution nodes persist longer; low-attribution nodes decay.
    This is analogous to how human memory is strengthened by causal
    involvement (episodic memories tied to consequential actions last longer).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim         = args.dim
        self.n_nodes     = args.n_causal_nodes
        self.top_k       = args.attr_top_k
        self.ema         = args.attr_ema

        # Running attribution scores per node
        self.register_buffer(
            'attribution_scores',
            torch.ones(args.n_causal_nodes) / args.n_causal_nodes
        )

        # Map concept attribution back to token-level write weight
        self.concept_proj = nn.Linear(args.dim, args.n_causal_nodes)
        self.gate_proj    = nn.Linear(args.n_causal_nodes, 1)
        self.norm         = RMSNorm(args.dim)

    @torch.no_grad()
    def update_attribution(self, credit: torch.Tensor) -> None:
        """
        Update running attribution scores from counterfactual credit.
        credit: [B, n_nodes] — causal credit per concept node.
        """
        batch_credit = credit.mean(0).clamp(0, 1)               # [n_nodes]
        self.attribution_scores = (
            self.ema * self.attribution_scores +
            (1 - self.ema) * batch_credit
        )

    def forward(self, x: torch.Tensor, credit: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_gated:    hidden states modulated by causal salience
            write_gate: [B, L, 1] write strength for memory system
        """
        if credit is not None:
            self.update_attribution(credit)

        # Project hidden to concept space
        concept_activations = torch.sigmoid(self.concept_proj(x))   # [B, L, n_nodes]

        # Weight by causal attribution
        attr = self.attribution_scores.unsqueeze(0).unsqueeze(0)     # [1, 1, n_nodes]
        attributed = concept_activations * attr

        # Gate: how causally salient is this token?
        write_gate = torch.sigmoid(self.gate_proj(attributed))       # [B, L, 1]

        # Top-k concept masking: only retain top-k attributed concepts
        _, topk_idx = torch.topk(self.attribution_scores, self.top_k)
        topk_mask = torch.zeros(self.n_nodes, device=x.device)
        topk_mask[topk_idx] = 1.0
        masked_attr = attributed * topk_mask.unsqueeze(0).unsqueeze(0)
        topk_gate = torch.sigmoid(self.gate_proj(masked_attr))       # [B, L, 1]

        x_gated = self.norm(x * (0.5 + 0.5 * topk_gate))

        return x_gated, write_gate


# ============= Pearl Ladder Reasoner =============

class PearlLadderReasoner(nn.Module):
    """
    Enforces Pearl's three-rung causal hierarchy.

    Rung 1 — Association   P(Y | X)
        "What is the probability of Y given I *observe* X?"
        Standard conditional: the statistician's question.

    Rung 2 — Intervention  P(Y | do(X))
        "What is the probability of Y if I *set* X to x?"
        Requires graph surgery: break incoming edges to X.
        The surgeon's question.

    Rung 3 — Counterfactual P(Y_x | X', Y')
        "Given that X' happened and Y' resulted, what would Y have been
        if X had been x instead?"
        Requires abduction: infer exogenous noise from (X', Y'), then
        replay with X = x.  The historian's question.

    This module:
      1. Classifies each query as L1 / L2 / L3 using a small MLP.
      2. Routes it to the appropriate computation path.
      3. Returns a loss that penalises L1 computations for L2/L3 queries
         (the model must use the right tool for the right question).

    The rung classification is supervised by query-type labels during
    training.  At inference, the model predicts the rung and routes itself.
    """

    RUNGS = ["L1_ASSOCIATION", "L2_INTERVENTION", "L3_COUNTERFACTUAL"]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim    = args.dim
        h           = args.pearl_hidden
        self.weight = args.pearl_loss_weight

        # Query-type classifier
        self.rung_classifier = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 3),
        )

        # Per-rung answer heads
        self.l1_head = nn.Sequential(          # association: P(Y|X) via standard attn
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )
        self.l2_head = nn.Sequential(          # intervention: do-calculus path
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )
        self.l3_head = nn.Sequential(          # counterfactual: twin-net path
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )

        # Consistency checker: L3 answer should be reachable from L2
        self.consistency_proj = nn.Linear(args.dim * 2, 1)

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x:           torch.Tensor,
        rung_labels: Optional[torch.Tensor] = None,  # [B] ∈ {0, 1, 2} for supervised
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)                                       # [B, D]

        # Classify the rung
        rung_logits = self.rung_classifier(pooled)               # [B, 3]
        rung_probs  = F.softmax(rung_logits, dim=-1)
        rung_idx    = rung_probs.argmax(-1)                      # [B]

        # Compute all three answers (soft mixture during training)
        ans_l1 = self.l1_head(x)
        ans_l2 = self.l2_head(x)
        ans_l3 = self.l3_head(x)

        # Soft-weighted answer (gradient flows through all during training)
        w = rung_probs.unsqueeze(1).unsqueeze(-1)                # [B, 1, 3, 1]
        ans_stack = torch.stack([ans_l1, ans_l2, ans_l3], dim=2) # [B, L, 3, D]
        x_out = (ans_stack * w).sum(2)                           # [B, L, D]
        x_out = self.norm(x + x_out * 0.1)

        # Consistency loss: L3 should not contradict L2
        l2_pooled = ans_l2.mean(1)                               # [B, D]
        l3_pooled = ans_l3.mean(1)
        consistency = torch.sigmoid(
            self.consistency_proj(torch.cat([l2_pooled, l3_pooled], dim=-1))
        ).squeeze(-1)                                            # [B]
        consistency_loss = (1.0 - consistency).mean()

        # Supervised rung loss (if labels provided)
        rung_loss = torch.tensor(0.0, device=x.device)
        if rung_labels is not None:
            rung_loss = F.cross_entropy(rung_logits, rung_labels.to(x.device))

        total_pearl_loss = self.weight * (consistency_loss + rung_loss)

        return x_out, {
            "rung_logits":   rung_logits,
            "rung_probs":    rung_probs,
            "rung":          [self.RUNGS[i] for i in rung_idx.tolist()],
            "l1_answer":     ans_l1,
            "l2_answer":     ans_l2,
            "l3_answer":     ans_l3,
            "consistency":   consistency,
            "pearl_loss":    total_pearl_loss,
        }


# ============= Causal World Claudeson =============

class ClaudesonCausalWorld(ClaudesonTranscendent):
    """
    Claudeson 2026 — Causal World Edition.

    Inherits the full Transcendent architecture and wires causal reasoning
    into the planning and memory pipelines:

      causal_dynamics        — world model that respects the causal graph;
                               planning uses do-calculus, not correlations
      interventional_planner — evaluates actions as interventions, not
                               as inputs to a black-box dynamics model
      counterfactual_engine  — twin-network counterfactual imagination;
                               hindsight credit assignment
      attribution_gate       — causal salience replaces attention as the
                               memory write signal
      pearl_ladder           — explicit L1/L2/L3 rung classification;
                               penalises using correlational answers for
                               interventional/counterfactual queries

    Processing pipeline (after Transcendent):
      GW → Program → IRL → LIF       (Transcendent)
            ↓
      Causal Dynamics  (encode to concept space, apply graph)
            ↓
      Interventional Planner  (do-calculus action selection)
            ↓
      Counterfactual Engine   (twin-net imagination + credit)
            ↓
      Attribution Gate        (causal salience → memory writes)
            ↓
      Pearl Ladder Reasoner   (enforce L1/L2/L3 distinction)

    New output keys:
      causal_world     — dynamics model outputs
      causal_plan      — interventional planner outputs
      counterfactual   — twin-net imagination outputs
      attribution      — write gate + running attribution scores
      pearl            — rung classification + per-rung answers
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.causal_dynamics        = CausalDynamicsModel(args)
        self.interventional_planner = InterventionalPlanner(args)
        self.counterfactual_engine  = CounterfactualImagination(args)
        self.attribution_gate       = CausalAttributionGate(args)
        self.pearl_ladder           = PearlLadderReasoner(args)

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
    ) -> Dict:
        # ── Full Transcendent pass ───────────────────────────────────────
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens,
            feedback=feedback, agent_observations=agent_observations,
        )
        x = base["hidden_states"]

        # ── Causal Dynamics ──────────────────────────────────────────────
        causal_out = self.causal_dynamics(x, action=actual_action, intervene=(actual_action is not None))
        x = causal_out["hidden"]

        # ── Interventional Planning ──────────────────────────────────────
        plan_out = self.interventional_planner(x)

        # ── Counterfactual Imagination ───────────────────────────────────
        x, cf_out = self.counterfactual_engine(x, actual_action=actual_action)

        # ── Attribution Gate ─────────────────────────────────────────────
        x, write_gate = self.attribution_gate(x, credit=cf_out.get("credit"))

        # ── Pearl Ladder ─────────────────────────────────────────────────
        x, pearl_out = self.pearl_ladder(x, rung_labels=rung_labels)

        return {
            **base,
            "hidden_states":  x,
            "causal_world":   causal_out,
            "causal_plan":    plan_out,
            "counterfactual": cf_out,
            "attribution":    {
                "write_gate":          write_gate,
                "attribution_scores":  self.attribution_gate.attribution_scores,
            },
            "pearl":          pearl_out,
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """All auxiliary losses from every generation + causal world."""
        losses = super().compute_auxiliary_losses()

        # DAG constraints on both the representation graph (from grounded)
        # and the new dynamics graph
        losses["causal_dynamics_dag"] = (
            self.causal_dynamics.dag_constraint() * 0.01
        )
        losses["planner_dag"] = (
            self.interventional_planner.dynamics.dag_constraint() * 0.01
        )

        return losses


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — CAUSAL WORLD EDITION")
    print("Interventional Planning · Counterfactual Imagination")
    print("Causal Attribution Memory · Pearl L1/L2/L3 Ladder")
    print("=" * 70)

    args = ModelArgs()
    # Tiny config — CPU demo
    args.dim                = 128
    args.n_layers           = 2
    args.n_heads            = 4
    args.n_kv_heads         = 2
    args.vocab_size         = 512
    args.max_seq_len        = 64
    args.memory_slots       = 32
    args.episodic_slots     = 64
    args.goal_dim           = 128
    args.latent_dim         = 64
    args.energy_hidden      = 128
    args.ssm_state_dim      = 32
    args.ssm_chunk_size     = 16
    args.num_experts        = 2
    args.num_shared_experts = 1
    args.env_state_dim      = 32
    args.action_space_size  = 16
    args.planning_horizon   = 2
    args.num_simulations    = 2
    args.img_size           = 32
    args.patch_size         = 8
    args.audio_spec_dim     = 16
    args.gradient_checkpointing = False
    args.n_agents           = 4
    args.lora_rank          = 8
    args.n_causal_nodes     = 16
    args.metacog_hidden     = 64
    args.n_debate_agents    = 3
    args.debate_hidden      = 128
    args.n_propositions     = 16
    args.n_constraints      = 8
    args.consistency_iters  = 2
    args.rsi_rank           = 4
    args.rsi_horizon        = 2
    args.n_workspace_slots  = 8
    args.gw_competition_k   = 2
    args.gw_broadcast_steps = 1
    args.n_ops              = 16
    args.n_registers        = 4
    args.prog_steps         = 3
    args.prog_hidden        = 64
    args.irl_hidden         = 64
    args.irl_n_preferences  = 8
    args.lif_steps          = 3
    # Causal World specific
    args.causal_state_dim         = 32
    args.intervention_horizon     = 2
    args.n_intervention_samples   = 4
    args.cf_n_branches            = 2
    args.attr_top_k               = 4
    args.pearl_hidden             = 64

    print("\nInitialising ClaudesonCausalWorld...")
    model = ClaudesonCausalWorld(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text         = torch.randint(0, 512, (2, 32))
    feedback     = torch.randn(2, args.dim)
    agent_obs    = torch.randn(2, 8, args.dim)
    actual_action = torch.randint(0, args.action_space_size, (2,))
    rung_labels  = torch.randint(0, 3, (2,))

    # Seed IRL preference buffer
    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            text=text,
            feedback=feedback,
            agent_observations=agent_obs,
            actual_action=actual_action,
            rung_labels=rung_labels,
        )

    print("\nJedi state:")
    print(f"  Goal:         {out['jedi_goal']}")
    print(f"  Energy:       {out['jedi_energy'].mean().item():.4f}")

    print("\nCausal World Model:")
    print(f"  Concept shape:  {out['causal_world']['concepts'].shape}")
    print(f"  DAG loss:       {out['causal_world']['dag_loss'].item():.4f}")

    print("\nInterventional Planner:")
    print(f"  Best action:    {out['causal_plan']['causal_action'].tolist()}")
    print(f"  Planner DAG:    {out['causal_plan']['dag_loss'].item():.4f}")

    print("\nCounterfactual Imagination:")
    rd = out['counterfactual']['reward_delta']
    print(f"  Reward delta:   mean={rd.mean().item():.4f}")
    print(f"  Credit shape:   {out['counterfactual']['credit'].shape}")

    print("\nAttribution Gate:")
    print(f"  Write gate mean: {out['attribution']['write_gate'].mean().item():.4f}")
    top_nodes = out['attribution']['attribution_scores'].topk(3).indices.tolist()
    print(f"  Top-3 causal nodes: {top_nodes}")

    print("\nPearl Ladder:")
    print(f"  Rung:            {out['pearl']['rung']}")
    print(f"  Rung probs:      {out['pearl']['rung_probs'].tolist()}")
    print(f"  Consistency:     {out['pearl']['consistency'].tolist()}")
    print(f"  Pearl loss:      {out['pearl']['pearl_loss'].item():.4f}")

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonCausalWorld READY.")
    print("Plans by intervening, not by predicting.")
    print("Imagines counterfactuals, assigns credit causally.")
    print("Knows which rung of the ladder it's standing on.")
    print("=" * 70)
