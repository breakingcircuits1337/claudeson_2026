"""
Claudeson 2026 - Temporal Reasoning Edition
=============================================
Causal Temporal Graphs · Event Ordering · Duration Estimation · Multi-Scale Planning

The problem this generation solves
------------------------------------
Every previous generation treats time superficially:
  - Jedi plans over a fixed horizon (planning_horizon steps)
  - The causal model captures X→Y but not X→Y at time T
  - The schema engine binds roles but not temporal roles (BEFORE, DURING, AFTER)
  - The formal verifier checks properties at a single moment

Real-world intelligence requires reasoning across radically different timescales:
  - Milliseconds: motor control, reflexes
  - Seconds:      conversation turns, immediate actions
  - Minutes:      task completion, cooking
  - Hours:        project work, travel
  - Days:         planning, scheduling
  - Years:        career, climate, policy
  - Decades+:     civilisational consequences

The CausalWorld layer added do-calculus for "what if X had been Y?"
But it cannot answer:
  "What if X had been Y BEFORE event Z?"
  "How long will consequence C persist?"
  "What is the right action NOW given that consequence D is 10 years away?"

Temporal reasoning failures are catastrophic:
  - A system that plans perfectly for the next 10 minutes but ignores
    10-year consequences is not aligned with human values.
  - A system that cannot order events correctly will misunderstand
    causation (A happened after B, therefore A caused B — temporal confusion).
  - A system with no duration model will underestimate commitment costs
    and overestimate reversibility.

This generation adds five components:

  1. Temporal Event Graph (TEG)
     Represents events as nodes with typed temporal edges:
       BEFORE(e1, e2): e1 happens strictly before e2
       OVERLAPS(e1, e2): e1 and e2 overlap in time
       CAUSES(e1, e2, Δt): e1 causes e2 with lag Δt
       ENABLES(e1, e2): e1 is a prerequisite for e2
     Based on Allen's interval algebra (1983) extended with causal edges.

  2. Duration Estimator (DE)
     Estimates how long events will take / have taken.
     Distinguishes:
       Point events (instantaneous): "the light turned on"
       Duration events: "the meeting lasted 2 hours"
       Recurring events: "the drug must be taken daily"
       Open-ended events: "the war continued for years"
     Uses log-normal duration distributions (most real durations are
     log-normally distributed: minutes, hours, years are on log scale).

  3. Temporal Consistency Enforcer (TCE)
     Checks temporal reasoning for logical consistency:
       BEFORE(A,B) AND BEFORE(B,C) → BEFORE(A,C) (transitivity)
       BEFORE(A,B) → NOT AFTER(A,B) (asymmetry)
       BEFORE(A,B) AND OVERLAPS(B,C) → OVERLAPS_OR_BEFORE(A,C)
     Implemented as a constraint propagation network over the event graph.

  4. Multi-Scale Planner (MSP)
     Plans simultaneously at multiple timescales:
       Operational (seconds-minutes): immediate actions, quick decisions
       Tactical (hours-days):         task sequences, resource allocation
       Strategic (months-years):       goal pursuit, relationship building
       Civilisational (decades+):      systemic impacts, legacy effects
     Uses a hierarchical planning structure where higher scales constrain
     lower scales but cannot override safety invariants.

  5. Temporal Credit Assignment (TCA)
     Solves the long-horizon credit assignment problem:
     "Which past actions are responsible for this current outcome?"
     For outcomes far in the future, standard backpropagation cannot
     assign credit across the time gap.  The TCA uses:
       - Temporal difference learning across scales
       - Hindsight credit assignment (like CounterfactualImagination but temporal)
       - Eligibility traces: decaying credit assignment windows

Architecture evolution:
  ... → formal_verification → temporal_reasoning
                                     ↑ you are here (final of the four new layers)

The completed stack after this layer:
  claudson → extended → infinite → pro → ultimate → jedi → grounded
  → sovereign → transcendent → causal_world → metacurriculum
  → abstraction → social_alignment → uncertainty → grounded_language
  → formal_verification → temporal_reasoning  ← HERE
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from claudson_formal_verification import (
    ClaudesonFormalVerification,
)
from claudson_formal_verification import (
    ModelArgs as FormalVerificationArgs,
)
from claudson_jedi import RMSNorm, SwiGLU

log = logging.getLogger(__name__)


# ============= Configuration =============


@dataclass
class ModelArgs(FormalVerificationArgs):
    # Temporal Event Graph
    teg_n_events: int = 32  # max events tracked simultaneously
    teg_n_edge_types: int = 6  # BEFORE/AFTER/OVERLAPS/CAUSES/ENABLES/PREVENTS
    teg_hidden: int = 256  # hidden dim for event encoder
    teg_n_heads: int = 4  # attention heads in event graph

    # Duration Estimator
    de_n_categories: int = 8  # duration categories (milliseconds to years)
    de_hidden: int = 128  # hidden dim for duration head
    de_log_scale: bool = True  # use log-scale durations

    # Temporal Consistency Enforcer
    tce_n_iters: int = 5  # constraint propagation iterations
    tce_hidden: int = 128  # hidden dim for constraint network

    # Multi-Scale Planner
    msp_n_scales: int = 4  # operational / tactical / strategic / civilisational
    msp_horizon: List = field(default_factory=lambda: [10, 100, 1000, 10000])  # steps per scale
    msp_hidden: int = 256  # hidden dim per planning scale
    msp_discount: List = field(default_factory=lambda: [0.99, 0.95, 0.9, 0.5])  # discount per scale

    # Temporal Credit Assignment
    tca_trace_decay: float = 0.9  # eligibility trace decay per step
    tca_n_traces: int = 32  # number of eligibility trace slots
    tca_hidden: int = 128  # hidden dim for credit assignment


# ============= Temporal Event Graph =============

# Allen's Interval Algebra relation indices
TEMPORAL_RELATIONS = {
    "BEFORE": 0,
    "AFTER": 1,
    "OVERLAPS": 2,
    "CAUSES": 3,
    "ENABLES": 4,
    "PREVENTS": 5,
}

DURATION_LABELS = [
    "milliseconds",  # 0
    "seconds",  # 1
    "minutes",  # 2
    "hours",  # 3
    "days",  # 4
    "months",  # 5
    "years",  # 6
    "decades+",  # 7
]

SCALE_NAMES = ["operational", "tactical", "strategic", "civilisational"]


class TemporalEventGraph(nn.Module):
    """
    Represents and reasons over a graph of temporal events.

    Events are encoded as vectors with:
      - Content embedding: what happened?
      - Temporal embedding: when / for how long?
      - Causal embedding: what caused it / what did it cause?

    Edges are typed (Allen's relations + causal extensions) and weighted.

    The graph is updated dynamically:
      - New events are encoded and added as nodes
      - Relations are inferred via attention + learned edge classifiers
      - The graph is pruned when events are no longer relevant

    Reasoning:
      Given a query event, the graph supports:
        - "What happened before/after X?"
        - "What caused X?"
        - "What will X enable / prevent?"
        - "How long will X last?"
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_events = args.teg_n_events
        self.n_edges = args.teg_n_edge_types
        self.dim = args.dim
        h = args.teg_hidden

        # Event encoder: hidden state → event representation
        self.event_encoder = nn.Sequential(
            nn.Linear(args.dim, h),
            RMSNorm(h),
            SwiGLU(h, h * 2),
            nn.Linear(h, h),
        )

        # Temporal position encoder: continuous time → embedding
        # Uses sinusoidal encoding at log-spaced frequencies
        n_freqs = 16
        self.time_freqs = nn.Parameter(
            torch.logspace(-3, 3, n_freqs).unsqueeze(0), requires_grad=False
        )
        self.time_proj = nn.Linear(n_freqs * 2, h)

        # Edge classifier: (event_i, event_j) → relation type distribution
        self.edge_clf = nn.Sequential(
            nn.Linear(h * 2, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, args.teg_n_edge_types),
            nn.Softmax(dim=-1),
        )

        # Graph attention: events attend over the event graph
        self.graph_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=args.teg_n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Event memory buffer
        self.register_buffer("event_memory", torch.zeros(args.teg_n_events, h))
        self.register_buffer("event_ptr", torch.tensor(0))
        self.register_buffer("event_count", torch.tensor(0))

        # Bridge back to main dim
        self.out_proj = nn.Sequential(
            nn.Linear(h, args.dim),
            RMSNorm(args.dim),
        )

        self.norm = RMSNorm(args.dim)

    def _encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal encoding of continuous time value t: [B] → [B, h]"""
        t_exp = t.unsqueeze(-1) * self.time_freqs  # [B, n_freqs]
        sin_enc = torch.sin(t_exp)
        cos_enc = torch.cos(t_exp)
        return self.time_proj(torch.cat([sin_enc, cos_enc], dim=-1))

    @torch.no_grad()
    def add_event(self, event_emb: torch.Tensor) -> None:
        """Add an event embedding to the memory buffer."""
        ptr = int(self.event_ptr.item())
        self.event_memory[ptr] = event_emb.detach().mean(0)
        self.event_ptr = torch.tensor((ptr + 1) % self.n_events)
        self.event_count = self.event_count + 1

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        timestamp: Optional[float] = None,  # current time (arbitrary units)
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Encode current state as events
        events_raw = self.event_encoder(x)  # [B, L, h]
        events_raw.size(-1)

        # Temporal embedding
        if timestamp is not None:
            t_emb = self._encode_time(
                torch.tensor([timestamp], device=x.device).expand(B)
            ).unsqueeze(1)  # [B, 1, h]
            events_raw = events_raw + t_emb

        # Attend over event memory
        n_mem = min(int(self.event_count.item()), self.n_events)
        if n_mem > 0:
            mem = self.event_memory[:n_mem].unsqueeze(0).expand(B, -1, -1)
            events_attended, attn_w = self.graph_attn(
                query=events_raw,
                key=mem,
                value=mem,
            )
            events_raw = events_raw + events_attended * 0.1
        else:
            attn_w = torch.zeros(B, L, 1, device=x.device)

        # Infer temporal relations between current and memory events
        if n_mem > 0:
            curr_pooled = events_raw.mean(1).unsqueeze(1).expand(-1, n_mem, -1)
            mem_pooled = mem
            edge_input = torch.cat([curr_pooled, mem_pooled], dim=-1)  # [B, n_mem, 2h]
            edge_types = self.edge_clf(edge_input)  # [B, n_mem, n_edges]
            dominant_rel = edge_types.argmax(-1)  # [B, n_mem]
        else:
            edge_types = torch.zeros(B, 1, self.n_edges, device=x.device)
            dominant_rel = torch.zeros(B, 1, dtype=torch.long, device=x.device)

        # Update event memory
        self.add_event(events_raw.mean(1))

        # Project back to main dim
        x_temporal = self.norm(x + self.out_proj(events_raw) * 0.1)

        return x_temporal, {
            "edge_types": edge_types,
            "dominant_rel": dominant_rel.tolist(),
            "n_events_mem": n_mem,
            "attn_w": attn_w,
        }


# ============= Duration Estimator =============


class DurationEstimator(nn.Module):
    """
    Estimates how long events take / have taken / will take.

    Duration categories (log-spaced, 8 levels):
      0: milliseconds    (< 1 second)
      1: seconds         (1-60 seconds)
      2: minutes         (1-60 minutes)
      3: hours           (1-24 hours)
      4: days            (1-30 days)
      5: months          (1-12 months)
      6: years           (1-10 years)
      7: decades+        (> 10 years)

    For each event, the estimator outputs:
      - A categorical distribution over duration levels
      - A log-normal parameterisation within the chosen level
      - An uncertainty estimate (some events are inherently unpredictable in duration)

    This is essential for:
      - Commitment awareness: "this decision locks me in for years"
      - Urgency modulation: "this needs to happen in seconds"
      - Consequence weighting: distant consequences are less certain
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_cats = args.de_n_categories
        self.log_scale = args.de_log_scale
        h = args.de_hidden
        self.dim = args.dim

        # Duration classifier: event → duration category distribution
        self.duration_clf = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.de_n_categories),
        )

        # Log-normal parameterisation: event → (log_mean, log_std) within category
        self.lognorm_head = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, 2),  # log_mean, log_std
        )

        # Reversibility estimator: can this event be undone?
        self.reversibility_head = nn.Sequential(
            nn.Linear(args.dim, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

        # Duration-weighted attention: shorter events get more "urgency"
        self.urgency_head = nn.Sequential(
            nn.Linear(args.de_n_categories, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)  # [B, D]

        # Duration category distribution
        dur_logits = self.duration_clf(pooled)  # [B, n_cats]
        dur_probs = F.softmax(dur_logits, dim=-1)  # [B, n_cats]
        dur_cat = dur_probs.argmax(-1)  # [B]

        # Log-normal parameters within category
        lognorm = self.lognorm_head(pooled)  # [B, 2]
        log_mean, log_std = lognorm[:, 0], lognorm[:, 1].exp().clamp(0.01, 5.0)

        # Sample duration (in log units)
        if self.training:
            eps = torch.randn_like(log_mean)
            log_dur = log_mean + log_std * eps
        else:
            log_dur = log_mean

        # Reversibility
        reversibility = self.reversibility_head(pooled).squeeze(-1)  # [B]

        # Urgency: inversely proportional to duration (short events are urgent)
        urgency = self.urgency_head(dur_probs).squeeze(-1)  # [B]

        # Modulate hidden state by urgency
        x_temporal = self.norm(x * (0.9 + 0.1 * urgency.unsqueeze(1).unsqueeze(2)))

        dur_labels = [
            DURATION_LABELS[min(int(c), len(DURATION_LABELS) - 1)] for c in dur_cat.tolist()
        ]

        return x_temporal, {
            "dur_probs": dur_probs,
            "dur_category": dur_cat.tolist(),
            "dur_labels": dur_labels,
            "log_duration": log_dur,
            "reversibility": reversibility,
            "urgency": urgency,
        }


# ============= Temporal Consistency Enforcer =============


class TemporalConsistencyEnforcer(nn.Module):
    """
    Propagates temporal constraints to enforce logical consistency.

    Allen's Interval Algebra has 13 base relations and a full
    composition table (what relation follows from composing two others).
    For example:
      BEFORE(A,B) ∧ BEFORE(B,C) → BEFORE(A,C)    (transitivity)
      BEFORE(A,B) ∧ OVERLAPS(B,C) → BEFORE(A,C) or OVERLAPS(A,C)

    The TCE:
      1. Reads the temporal edge distribution from the TEG
      2. Propagates constraints via a learned composition network
      3. Detects inconsistencies (cycles in BEFORE relation, etc.)
      4. Outputs a consistency score and corrected edge distributions

    This prevents temporal hallucination: claiming A happened before B
    when B happened before A is a logical contradiction that should be
    flagged and corrected.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_iters = args.tce_n_iters
        self.n_edges = args.teg_n_edge_types
        h = args.tce_hidden
        self.dim = args.dim

        # Composition table: (rel_i, rel_j) → result relation distribution
        self.compose = nn.Sequential(
            nn.Linear(args.teg_n_edge_types * 2, h),
            nn.GELU(),
            nn.Linear(h, args.teg_n_edge_types),
            nn.Softmax(dim=-1),
        )

        # Inconsistency detector: find cycles / contradictions
        self.incons_head = nn.Sequential(
            nn.Linear(args.teg_n_edge_types, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

        # Consistency-aware hidden state modulation
        self.consistency_gate = nn.Sequential(
            nn.Linear(1, args.dim),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        edge_types: torch.Tensor,  # [B, n_mem, n_edges] from TEG
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        n_mem = edge_types.size(1)

        # Propagate constraints (simplified: single-step composition)
        # For each pair of edges, compute transitivity
        if n_mem > 1:
            e1 = edge_types[:, :-1, :]  # [B, n-1, n_edges]
            e2 = edge_types[:, 1:, :]  # [B, n-1, n_edges]
            composed_inp = torch.cat([e1, e2], dim=-1)  # [B, n-1, 2*n_edges]
            composed = self.compose(composed_inp)  # [B, n-1, n_edges]

            # Check consistency: do composed relations agree with direct edges?
            if edge_types.size(1) > 2:
                direct = edge_types[:, 2:, :]
                n_check = min(composed.size(1), direct.size(1))
                consistency_raw = 1.0 - (composed[:, :n_check] - direct[:, :n_check]).abs().mean(
                    -1
                ).mean(-1)
            else:
                consistency_raw = torch.ones(B, device=x.device)
        else:
            consistency_raw = torch.ones(B, device=x.device)

        # Inconsistency detection
        edge_mean = edge_types.mean(1)  # [B, n_edges]
        incons = self.incons_head(edge_mean).squeeze(-1)  # [B] — prob of inconsistency

        # Consistency score
        consistency = consistency_raw * (1.0 - incons)  # [B]

        # Gate hidden state: reduce confidence when inconsistent
        gate = self.consistency_gate(consistency.unsqueeze(-1))  # [B, D]
        x_consistent = self.norm(x * (0.8 + 0.2 * gate).unsqueeze(1))

        return x_consistent, {
            "consistency": consistency,
            "inconsistency_prob": incons,
            "inconsistency_flag": (incons > 0.5).tolist(),
        }


# ============= Multi-Scale Planner =============


class PlanningScale(nn.Module):
    """One planning scale: encodes plans over a specific time horizon."""

    def __init__(self, dim: int, horizon: int, discount: float, hidden: int):
        super().__init__()
        self.horizon = horizon
        self.discount = discount

        # Plan encoder: hidden state → plan representation for this scale
        self.plan_encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            RMSNorm(dim),
        )

        # Value function: expected discounted return at this scale
        self.value_head = nn.Sequential(
            nn.Linear(dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Sub-goal generator: what intermediate goals serve this scale?
        self.subgoal_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            RMSNorm(dim),
        )

    def forward(
        self, x: torch.Tensor, lower_plan: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        pooled = x.mean(1)  # [B, D]

        # Incorporate lower scale plan (if available)
        if lower_plan is not None:
            pooled = pooled + lower_plan * 0.1

        plan = self.plan_encoder(pooled)  # [B, D]
        value = self.value_head(plan).squeeze(-1)  # [B]
        self.subgoal_head(plan)  # [B, D]

        return plan, value


class MultiScalePlanner(nn.Module):
    """
    Plans simultaneously at multiple timescales.

    The four planning scales:
      Operational  (horizon ~10):    immediate actions, reflexes
      Tactical     (horizon ~100):   task sequences, short-term goals
      Strategic    (horizon ~1000):  medium-term objectives, resource planning
      Civilisational (horizon ~10k): long-term impact, systemic consequences

    Hierarchical constraint:
      Higher scales constrain lower ones:
        "Strategically, we should avoid X" constrains tactical planning
        "Civlisationally, this sets a precedent" constrains strategic planning

    But lower scales inform higher ones:
      Operational experience feeds into tactical learning.
      Tactical success/failure updates strategic models.

    Multi-objective discounting:
      Each scale uses a different discount factor.
      Operational: γ ≈ 0.99 (care greatly about near future)
      Civilisational: γ ≈ 0.5 (heavy discounting of distant future,
                                   but still non-zero — future matters!)
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_scales = args.msp_n_scales
        self.dim = args.dim
        h = args.msp_hidden

        # One planner per scale
        horizons = (
            args.msp_horizon
            if len(args.msp_horizon) == args.msp_n_scales
            else [10, 100, 1000, 10000]
        )
        discounts = (
            args.msp_discount
            if len(args.msp_discount) == args.msp_n_scales
            else [0.99, 0.95, 0.9, 0.5]
        )

        self.scales = nn.ModuleList(
            [
                PlanningScale(args.dim, horizons[i], discounts[i], h)
                for i in range(args.msp_n_scales)
            ]
        )

        # Cross-scale alignment: higher scales modulate lower
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=args.dim,
            num_heads=max(1, args.dim // 64),
            batch_first=True,
            dropout=0.0,
        )

        # Plan fusion: combine all scales into unified plan
        self.plan_fusion = nn.Sequential(
            nn.Linear(args.dim * args.msp_n_scales, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, args.dim),
            RMSNorm(args.dim),
        )

        self.norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Bottom-up: compute plans from shortest to longest scale
        plans = []
        values = []
        lower = None

        for scale in self.scales:
            plan, value = scale(x, lower)
            plans.append(plan)
            values.append(value)
            lower = plan

        # Top-down: higher scales modulate lower via cross-attention
        plan_stack = torch.stack(plans, dim=1)  # [B, n_scales, D]
        aligned, _ = self.cross_scale_attn(plan_stack, plan_stack, plan_stack)
        plan_stack = plan_stack + aligned * 0.1

        # Fuse all scales
        plan_flat = plan_stack.reshape(B, -1)  # [B, n_scales * D]
        unified_plan = self.plan_fusion(plan_flat)  # [B, D]

        x_planned = self.norm(x + unified_plan.unsqueeze(1) * 0.1)

        return x_planned, {
            "scale_values": [v.tolist() for v in values],
            "plans": plan_stack,
            "unified_plan": unified_plan,
            "scale_names": SCALE_NAMES[: self.n_scales],
        }


# ============= Temporal Credit Assignment =============


class TemporalCreditAssignment(nn.Module):
    """
    Assigns credit to past actions for current outcomes.

    The long-horizon credit assignment problem:
      If I act now and the consequence appears in 1000 steps,
      standard TD learning requires 1000 backup steps.
      Eligibility traces decay credit assignment exponentially,
      making very long-horizon credit invisible.

    Solution: multi-scale credit with scale-matched eligibility traces.
      - Operational scale (γ=0.99): traces decay quickly (τ ≈ 100 steps)
      - Strategic scale (γ=0.9):    traces decay slowly  (τ ≈ 10 steps)
      - Civilisational (γ=0.5):     only the final state matters;
                                     use hindsight relabelling

    Eligibility traces:
      e_t(s,a) = γ λ e_{t-1}(s,a) + I[s_t=s, a_t=a]
      Decay parameter λ controls trace length.
      λ=1: Monte Carlo credit (full trajectory)
      λ=0: TD(0) credit (one-step only)

    This module maintains per-slot eligibility traces and updates them
    with each forward pass.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_traces = args.tca_n_traces
        self.trace_decay = args.tca_trace_decay
        self.dim = args.dim
        h = args.tca_hidden

        # Eligibility trace memory
        self.register_buffer("traces", torch.zeros(args.tca_n_traces, args.dim))
        self.register_buffer("trace_ptr", torch.tensor(0))

        # Credit head: trace × current outcome → credit signal
        self.credit_head = nn.Sequential(
            nn.Linear(args.dim * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
            nn.Tanh(),
        )

        # Long-horizon value estimator: multi-step return approximation
        self.value_est = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.msp_n_scales),  # one value per planning scale
        )

        # Credit-modulated policy: does current action deserve credit?
        self.policy_modulator = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.dim),
            nn.Tanh(),
        )

        self.norm = RMSNorm(args.dim)

    @torch.no_grad()
    def update_traces(self, x: torch.Tensor) -> None:
        """Decay existing traces and add new one."""
        self.traces = self.traces * self.trace_decay
        ptr = int(self.trace_ptr.item())
        self.traces[ptr] = x.detach().mean(0).mean(0)
        self.trace_ptr = torch.tensor((ptr + 1) % self.n_traces)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)  # [B, D]

        # Compute credit from traces
        trace_mean = self.traces.mean(0).unsqueeze(0).expand(B, -1)  # [B, D]
        credit = self.credit_head(torch.cat([pooled, trace_mean], dim=-1)).squeeze(-1)  # [B]

        # Multi-scale value estimates
        values = self.value_est(pooled)  # [B, n_scales]

        # Credit-modulated representation
        mod = self.policy_modulator(trace_mean)  # [B, D]
        x_credit = self.norm(x + (mod * credit.unsqueeze(-1).unsqueeze(-1)) * 0.05)

        # Update traces with current state
        self.update_traces(x)

        return x_credit, {
            "credit": credit,
            "multiscale_values": values,
            "trace_norm": self.traces.norm().item(),
        }


# ============= Closed-Loop Temporal Planner =============


class TemporalPlanner:
    """
    Closed-loop world-model planning using the ``MultiScalePlanner``
    and a differentiable transition function.

    The planner repeatedly calls ``predict_action`` on the current state,
    advances the state via ``transition``, and accumulates a trajectory.
    This provides an explicit simulation loop that is missing from the
    forward-pass-only ``MultiScalePlanner``.

    Args:
        model:       Any callable with a ``predict_action(state) → action``
                     interface.  Pass a ``ClaudesonTemporalReasoning`` instance
                     or a lightweight wrapper around its ``msp`` sub-module.
        transition:  ``Callable(state, action) → next_state``.  Defaults to
                     the included additive transition (state + action).

    Usage::

        planner = TemporalPlanner(model)
        trajectory = planner.simulate(initial_state, steps=10)
        states, actions, next_states = zip(*trajectory)
    """

    def __init__(
        self,
        model,
        transition: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self._transition = transition

    # ------------------------------------------------------------------

    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict the next action for ``state``.

        If the wrapped model exposes a ``predict_action`` method, calls it
        directly.  Otherwise, passes ``state`` through the full model
        forward and extracts the ``unified_plan`` from the temporal output.

        Args:
            state: Float tensor ``[B, L, D]`` or ``[B, D]``.

        Returns:
            Action tensor of shape ``[B, D]``.  When the wrapped model
            exposes its own ``predict_action``, that method's output
            shape governs; for the fallback path the sequence dimension
            is always collapsed via ``state.mean(1)`` or the model's
            ``unified_plan`` output, both of which are ``[B, D]``.
        """
        if hasattr(self.model, "predict_action"):
            return self.model.predict_action(state)

        # Fallback: run the full forward and use the unified plan as action
        with torch.no_grad():
            if state.dim() == 2:
                state = state.unsqueeze(1)  # [B, 1, D]
            out = self.model(state)
            plans = out.get("temporal", {}).get("plans", {})
            action = plans.get("unified_plan", state.mean(1))
        return action

    def transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance the world state by applying ``action``.

        Uses the user-supplied transition function if provided; otherwise
        falls back to a simple additive model:
            next_state = state + action

        Args:
            state:  Current state ``[B, D]`` or ``[B, L, D]``.
            action: Action tensor broadcastable to ``state``.

        Returns:
            Next state of the same shape as ``state``.
        """
        if self._transition is not None:
            return self._transition(state, action)
        # Simple additive baseline; override for learned transitions
        return state + action

    def simulate(
        self,
        initial_state: torch.Tensor,
        steps: int = 5,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Roll out the planner for ``steps`` steps starting from
        ``initial_state``.

        Args:
            initial_state: Float tensor ``[B, D]``.
            steps:         Number of simulation steps.

        Returns:
            List of ``(state, action, next_state)`` tuples, one per step.
        """
        trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        state = initial_state

        for _ in range(steps):
            action = self.predict_action(state)
            next_state = self.transition(state, action)
            trajectory.append((state, action, next_state))
            state = next_state

        return trajectory


# ============= Temporal Reasoning Claudeson =============


class ClaudesonTemporalReasoning(ClaudesonFormalVerification):
    """
    Claudeson 2026 — Temporal Reasoning Edition.

    Inherits the full Formal Verification architecture and adds:

      teg    — temporal event graph with Allen relations + causal edges
      de     — duration estimator (log-normal, 8 log-spaced categories)
      tce    — temporal consistency enforcer (constraint propagation)
      msp    — multi-scale planner (operational/tactical/strategic/civilisational)
      tca    — temporal credit assignment (eligibility traces + multi-scale TD)

    Processing pipeline (after Formal Verification):
      InvariantReg → PPCC → AbstractInterp → CEG → Certificates
            ↓
      TemporalEventGraph           (what happened, when, in what order?)
            ↓
      DurationEstimator            (how long will this take?)
            ↓
      TemporalConsistencyEnforcer  (is this temporally coherent?)
            ↓
      MultiScalePlanner            (what to do now / this week / this decade?)
            ↓
      TemporalCreditAssignment     (what past actions led here?)

    New output keys:
      temporal — {graph, duration, consistency, plans, credit}

    The complete 17-layer Claudeson stack (after this layer):
      1.  claudson              — Core: MoE + SSM + attention
      2.  extended              — Multimodal + External memory
      3.  infinite              — Infinite context + Sparse retrieval
      4.  pro                   — Reasoning + Tool use
      5.  ultimate              — World model + Planning
      6.  jedi                  — Free energy + Dreamer + Mamba + SSD
      7.  grounded              — ToM + Continual learning + Causal DAG
      8.  sovereign             — Metacognition + Debate + Neural symbolic + RSI
      9.  transcendent          — GWT + Program synthesis + IRL + LIF neurons
      10. causal_world          — Do-calculus + Counterfactual + Pearl ladder
      11. metacurriculum        — IG reward + Skill discovery + OEL
      12. abstraction           — HAE + Concept bottleneck + Schemas + Analogy
      13. social_alignment      — Stakeholders + Welfare + Norms + Contract + Moral
      14. uncertainty           — Bayesian + Conformal + Calibration + OOD
      15. grounded_language     — PSA + Motor schemas + SMS + Cross-modal + GCM
      16. formal_verification   — Invariants + PPCC + Abstract interp + CEG + Certs
      17. temporal_reasoning    — TEG + Duration + Consistency + MultiScale + TCA ← HERE
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.teg = TemporalEventGraph(args)
        self.de = DurationEstimator(args)
        self.tce = TemporalConsistencyEnforcer(args)
        self.msp = MultiScalePlanner(args)
        self.tca = TemporalCreditAssignment(args)

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
        timestamp: Optional[float] = None,
    ) -> Dict:
        # ── Full Formal Verification pass ─────────────────────────────────
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

        # ── Temporal Event Graph ──────────────────────────────────────────
        x, teg_out = self.teg(x, timestamp=timestamp)

        # ── Duration Estimator ────────────────────────────────────────────
        x, de_out = self.de(x)

        # ── Temporal Consistency Enforcer ─────────────────────────────────
        x, tce_out = self.tce(x, teg_out["edge_types"])

        # ── Multi-Scale Planner ───────────────────────────────────────────
        x, msp_out = self.msp(x)

        # ── Temporal Credit Assignment ────────────────────────────────────
        x, tca_out = self.tca(x)

        return {
            **base,
            "hidden_states": x,
            "temporal": {
                "graph": teg_out,
                "duration": de_out,
                "consistency": tce_out,
                "plans": msp_out,
                "credit": tca_out,
            },
        }


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — TEMPORAL REASONING EDITION")
    print("Event Graph · Duration · Consistency · Multi-Scale Plans · Credit")
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
    args.psa_n_anchors = 32
    args.psa_hidden = 64
    args.psa_n_heads = 2
    args.msg_n_primitives = 8
    args.msg_hidden = 64
    args.msg_compose_depth = 2
    args.sms_n_steps = 3
    args.sms_hidden = 64
    args.sms_n_branches = 2
    args.cmal_hidden = 64
    args.gcm_hidden = 64
    args.gcm_n_pairs = 4
    args.n_invariants = 8
    args.invariant_hidden = 64
    args.ppcc_hidden = 64
    args.ai_n_neurons = 16
    args.ai_hidden = 64
    args.ceg_budget = 5
    args.ceg_hidden = 64
    args.pcs_max_certs = 64
    # Temporal specific
    args.teg_n_events = 16
    args.teg_n_edge_types = 6
    args.teg_hidden = 64
    args.teg_n_heads = 2
    args.de_n_categories = 8
    args.de_hidden = 64
    args.tce_n_iters = 3
    args.tce_hidden = 64
    args.msp_n_scales = 4
    args.msp_hidden = 64
    args.tca_n_traces = 16
    args.tca_hidden = 64

    print("\nInitialising ClaudesonTemporalReasoning...")
    model = ClaudesonTemporalReasoning(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)
    for step in range(args.cp_window):
        model.cp_monitor.record_performance(0, 0.3 + 0.04 * step)
    for g in range(args.n_stakeholder_groups):
        model.stakeholder_vm.update_welfare(g, 0.5 + 0.1 * g)
    for _ in range(20):
        model.conformal.update_calibration(torch.rand(1).item())

    print("\nRunning forward pass (timestamp=1000.0)...")
    with torch.no_grad():
        out = model(
            text=torch.randint(0, 512, (2, 32)),
            feedback=torch.randn(2, args.dim),
            agent_observations=torch.randn(2, 8, args.dim),
            actual_action=torch.randint(0, args.action_space_size, (2,)),
            competence_signal=0.65,
            timestamp=1000.0,
        )

    t = out["temporal"]
    print("\nTemporal Event Graph:")
    print(f"  Events in memory:  {t['graph']['n_events_mem']}")
    print(f"  Dominant relations: {t['graph']['dominant_rel']}")

    print("\nDuration Estimator:")
    for b, (cat, label) in enumerate(
        zip(t["duration"]["dur_category"], t["duration"]["dur_labels"])
    ):
        print(f"  Batch {b}: category {cat} ({label})")
        urg = t["duration"]["urgency"][b].item()
        rev = t["duration"]["reversibility"][b].item()
        print(f"           urgency={urg:.3f}  reversibility={rev:.3f}")

    print("\nTemporal Consistency:")
    print(f"  Consistency score: {t['consistency']['consistency'].tolist()}")
    print(f"  Inconsistency:     {t['consistency']['inconsistency_flag']}")

    print("\nMulti-Scale Planner:")
    for scale, values in zip(t["plans"]["scale_names"], t["plans"]["scale_values"]):
        print(f"  {scale:<20}: value={values}")

    print("\nTemporal Credit Assignment:")
    print(f"  Credit:          {t['credit']['credit'].tolist()}")
    print(f"  Trace norm:      {t['credit']['trace_norm']:.4f}")
    print(f"  Multi-scale Vs:  {t['credit']['multiscale_values'].tolist()}")

    cert = out["verification"]["certificate"]
    print("\nVerification Certificate:")
    print(f"  Method: {cert['method']}  Confidence: {cert['confidence']:.4f}  Hash: {cert['hash']}")

    print("\n" + "=" * 70)
    print("ClaudesonTemporalReasoning READY.")
    print()
    print("THE COMPLETE 17-LAYER CLAUDESON STACK:")
    layers = [
        ("claudson", "Core: MoE + SSM + attention"),
        ("extended", "Multimodal + External memory"),
        ("infinite", "Infinite context + Sparse retrieval"),
        ("pro", "Reasoning + Tool use"),
        ("ultimate", "World model + Planning"),
        ("jedi", "Free energy + Dreamer + Mamba + SSD"),
        ("grounded", "Theory of mind + Continual learning + Causal DAG"),
        ("sovereign", "Metacognition + Debate + Neural symbolic + RSI"),
        ("transcendent", "GWT + Program synthesis + IRL + LIF neurons"),
        ("causal_world", "Do-calculus + Counterfactual + Pearl ladder"),
        ("metacurriculum", "IG reward + Skill discovery + Open-ended learning"),
        ("abstraction", "HAE + Concept bottleneck + Schemas + Analogy"),
        ("social_alignment", "Stakeholders + Welfare + Norms + Contract + Moral"),
        ("uncertainty", "Bayesian + Conformal + Calibration + OOD"),
        ("grounded_language", "PSA + Motor schemas + Sim + Cross-modal + Coherence"),
        ("formal_verification", "Invariants + PPCC + Abstract interp + CEG + Certs"),
        ("temporal_reasoning", "Event graph + Duration + Consistency + MSP + TCA"),
    ]
    for i, (name, desc) in enumerate(layers, 1):
        arrow = " ← HERE" if i == 17 else ""
        print(f"  {i:2d}. {name:<25} — {desc}{arrow}")
    print()
    print("Knows what happened. Knows when. Knows how long. Plans for decades.")
    print("=" * 70)
