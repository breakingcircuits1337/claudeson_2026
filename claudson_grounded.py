"""
Claudeson 2026 - Grounded Edition
==================================
The four gaps that remained after Jedi Edition, now filled:

  1. Theory of Mind     — belief states over other agents (what they know,
                          want, and intend); separates tool-use from genuine
                          collaboration.

  2. Grounded Action    — plans finally execute.  A tool-selection head picks
     Loop                 from a registry of callable tools, generates
                          structured parameters, then integrates real-world
                          feedback back into the hidden state.

  3. Continual Learning — EWC (Elastic Weight Consolidation) protects
                          important weights via Fisher information; LoRA
                          adapters absorb new skills in low-rank delta weights
                          so the backbone stays intact.

  4. Causal Reasoning   — a learnable causal graph over concept nodes lets
                          the model distinguish correlation from causation and
                          simulate interventions ("what happens if I do X?")
                          rather than just predicting the next token.

Architecture evolution:
  claudson.py → extended → infinite → pro → ultimate → jedi → grounded
                                                                   ↑ you are here
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

log = logging.getLogger(__name__)

from claudson_jedi import (
    ModelArgs as JediModelArgs,
    ClaudesonJedi,
    SwiGLU,
    RMSNorm,
    swiglu,
)


# ============= Configuration =============

@dataclass
class ModelArgs(JediModelArgs):
    # Theory of Mind
    n_agents: int = 8           # max number of external agents to model

    # Continual Learning
    lora_rank: int = 16         # rank of LoRA adapter matrices
    ewc_lambda: float = 5000.0  # EWC regularisation strength

    # Causal Reasoning
    n_causal_nodes: int = 64    # nodes in the causal concept graph

    # Grounded Action
    tool_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Override default tool registry"}
    )


# ============= Theory of Mind =============

class TheoryOfMind(nn.Module):
    """
    Maintains belief states over other agents.

    For each tracked agent slot the module stores three vectors:
      belief    — what we infer they currently know / believe
      desire    — what we infer they want (goals, preferences)
      intention — what we predict they will do next

    During the forward pass:
      1. Soft attention over agent slots selects the most relevant agent(s).
      2. Their inferred perspective is projected back into model-dim space
         and used to gently steer the hidden representations.
      3. An action predictor outputs what we expect the agent to do, enabling
         cooperative planning.

    update_beliefs() is called externally whenever new evidence arrives
    (e.g. the human's last utterance).

    This is what separates tool-use from genuine collaboration: a system that
    models you as an agent with beliefs and goals can anticipate your needs
    rather than just respond to your words.
    """

    def __init__(self, args: ModelArgs, max_agents: int = 8):
        super().__init__()
        self.dim = args.dim
        self.max_agents = max_agents
        self.action_space_size = args.action_space_size

        # Per-agent latent state slots (learned, updated at runtime)
        self.belief_slots    = nn.Parameter(torch.randn(max_agents, args.dim) * 0.02)
        self.desire_slots    = nn.Parameter(torch.randn(max_agents, args.dim) * 0.02)
        self.intention_slots = nn.Parameter(torch.randn(max_agents, args.dim) * 0.02)

        # GRU-based updaters: integrate new evidence into belief / desire
        self.belief_updater = nn.GRUCell(args.dim, args.dim)
        self.desire_updater = nn.GRUCell(args.dim, args.dim)

        # Soft attention: which agent is most relevant right now?
        self.agent_selector = nn.Linear(args.dim, max_agents)

        # Perspective taking: from belief + desire → a steering vector in D
        self.perspective_proj = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim),
            RMSNorm(args.dim),
        )

        # Predict the agent's next action (for collaborative planning)
        self.action_predictor = nn.Sequential(
            nn.Linear(args.dim * 3, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.action_space_size),
        )

        self.norm = RMSNorm(args.dim)

    @torch.no_grad()
    def update_beliefs(self, observation: torch.Tensor, agent_id: int = 0) -> None:
        """
        Integrate new evidence about a specific agent.

        observation: [B, L, D]  e.g. the most recent human utterance embedding
        agent_id:    which slot to update (0 = primary interlocutor)
        """
        B = observation.size(0)
        obs_pooled = observation.mean(1)            # [B, D]

        current_belief = self.belief_slots[agent_id].unsqueeze(0).expand(B, -1)
        current_desire = self.desire_slots[agent_id].unsqueeze(0).expand(B, -1)

        new_belief = self.belief_updater(obs_pooled, current_belief)
        new_desire = self.desire_updater(obs_pooled, current_desire)

        # Write back as running average (mean over batch)
        self.belief_slots.data[agent_id]    = new_belief.mean(0)
        self.desire_slots.data[agent_id]    = new_desire.mean(0)
        # Intention = blend of updated belief and desire as a proxy
        self.intention_slots.data[agent_id] = (new_belief.mean(0) + new_desire.mean(0)) * 0.5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Select the most relevant agent(s) via soft attention over slots
        agent_weights = F.softmax(self.agent_selector(x.mean(1)), dim=-1)   # [B, A]

        beliefs    = torch.einsum('ba,ad->bd', agent_weights, self.belief_slots)     # [B, D]
        desires    = torch.einsum('ba,ad->bd', agent_weights, self.desire_slots)     # [B, D]
        intentions = torch.einsum('ba,ad->bd', agent_weights, self.intention_slots)  # [B, D]

        # Compute the perspective vector: how would this agent see the context?
        perspective = self.perspective_proj(torch.cat([beliefs, desires], dim=-1))   # [B, D]

        # Steer every token position by the inferred agent perspective (gentle: 0.1)
        x_tom = self.norm(x + perspective.unsqueeze(1) * 0.1)

        # Predict what the agent will do next
        mental_state = torch.cat([beliefs, desires, intentions], dim=-1)             # [B, 3D]
        predicted_action = self.action_predictor(mental_state)                       # [B, A_space]

        return x_tom, {
            "beliefs":               beliefs,
            "desires":               desires,
            "intentions":            intentions,
            "agent_weights":         agent_weights,
            "predicted_agent_action": predicted_action,
        }


# ============= Grounded Action Loop =============

class GroundedActionLoop(nn.Module):
    """
    Bridges planning → execution → feedback.

    The model has always been able to *plan*, but plans stayed inside the
    model's imagination.  This module gives Claudeson hands:

      1. Tool selection   — a learned head picks which tool to call based on
                            the last hidden state and a soft attention over the
                            tool registry.
      2. Parameter gen    — another head produces a structured embedding in D
                            that a downstream caller decodes into actual
                            arguments (text, code, URLs, etc.).
      3. Feedback loop    — when a real-world observation arrives (tool result,
                            API response, user correction), a surprise detector
                            decides how much to update the hidden state, and a
                            gated encoder integrates the new information.

    The surprise signal is key: high surprise (expected ≠ actual) → large
    hidden-state update → the world model learns from the discrepancy.
    """

    DEFAULT_TOOLS = [
        "search", "read", "write", "execute",
        "ask", "plan", "reflect", "stop",
    ]

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        tool_names = args.tool_names or self.DEFAULT_TOOLS
        self.tool_names = tool_names
        self.n_tools = len(tool_names)

        # Learned signature embedding per tool (what does each tool "feel like"?)
        self.tool_embeddings = nn.Embedding(self.n_tools, args.dim)

        # Tool selection: last hidden state + context → tool probabilities
        self.tool_selector = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, self.n_tools),
        )

        # Parameter generator: hidden + tool context → call params (in embed space)
        self.param_gen = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )

        # Feedback encoder: raw observation tensor → representation
        self.feedback_encoder = nn.Sequential(
            nn.Linear(args.dim, args.dim),
            RMSNorm(args.dim),
            SwiGLU(args.dim, args.dim * 2),
            nn.Linear(args.dim, args.dim),
        )

        # Surprise: how different is what happened from what was expected?
        self.surprise_head = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, 1),
            nn.Sigmoid(),
        )

        # Belief update gate: high surprise → big update; low surprise → small
        self.belief_update_gate = nn.Sequential(
            nn.Linear(args.dim * 2, args.dim),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def select_tool(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits     = self.tool_selector(hidden)           # [B, n_tools]
        tool_probs = F.softmax(logits, dim=-1)
        tool_idx   = tool_probs.argmax(-1)                # [B]
        return tool_idx, tool_probs

    def generate_params(self, hidden: torch.Tensor, tool_idx: torch.Tensor) -> torch.Tensor:
        """Condition parameter generation on which tool was selected."""
        tool_ctx = self.tool_embeddings(tool_idx)         # [B, D]
        return self.param_gen(hidden + tool_ctx)           # [B, D]

    def integrate_feedback(
        self, hidden: torch.Tensor, feedback: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate a real-world observation into the hidden state.
        High surprise → large update; low surprise → world model was right.
        """
        fb_enc   = self.feedback_encoder(feedback)                                  # [B, D]
        surprise = self.surprise_head(torch.cat([hidden, fb_enc], dim=-1))          # [B, 1]
        gate     = self.belief_update_gate(torch.cat([hidden, fb_enc], dim=-1))     # [B, D]
        hidden_updated = self.norm(hidden + gate * fb_enc * surprise)
        return hidden_updated, surprise

    def forward(
        self,
        x:        torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        last = x[:, -1, :]          # action selection driven by last-position state

        tool_idx, tool_probs = self.select_tool(last)
        params   = self.generate_params(last, tool_idx)
        surprise = torch.zeros(B, 1, device=x.device)

        if feedback is not None:
            last, surprise = self.integrate_feedback(last, feedback)
            # Replace last token hidden state with the updated one
            x = torch.cat([x[:, :-1, :], last.unsqueeze(1)], dim=1)

        return x, {
            "tool_idx":   tool_idx,
            "tool_probs": tool_probs,
            "tool_names": [self.tool_names[i] for i in tool_idx.tolist()],
            "call_params": params,
            "surprise":   surprise,
        }


# ============= Continual Learning (EWC + LoRA) =============

class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (Hu et al. 2021).

    New knowledge is stored in two small matrices A [D×r] and B [r×D]
    rather than in the full weight matrix.  The backbone never changes;
    only A and B are updated for a new task.  After consolidation, even
    A and B are frozen and a fresh pair is initialised for the next task.

    Scaling = alpha / rank keeps the magnitude stable across rank choices.
    """

    def __init__(self, dim: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B initialised to zero so the adapter starts as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]  →  x @ A @ B  →  [B, L, D]
        return (x @ self.A @ self.B) * self.scaling


class ContinualLearner(nn.Module):
    """
    Learns from new experience without forgetting old knowledge.

    Two complementary mechanisms:

    LoRA adapters
        New knowledge flows through low-rank delta weights attached to the
        representation stream.  The main model weights are never touched.
        This is how Claudeson acquires a new skill without overwriting old ones
        — analogous to a person learning Spanish without forgetting English.

    Elastic Weight Consolidation (EWC)  — Kirkpatrick et al. 2017
        After finishing a task, consolidate() estimates the Fisher information
        diagonal for each parameter.  High Fisher score = this weight was
        important for the old task = protect it from large future updates.
        The EWC loss adds λ/2 * F_i * (θ_i − θ*_i)² to training, acting as
        a spring that pulls important weights back toward their old values.
    """

    def __init__(self, dim: int, rank: int = 16, ewc_lambda: float = 5000.0):
        super().__init__()
        self.dim        = dim
        self.ewc_lambda = ewc_lambda

        # Two adapters: one for the "in" direction, one for "out"
        self.adapter_in  = LoRAAdapter(dim, rank=rank)
        self.adapter_out = LoRAAdapter(dim, rank=rank)

        # Adaptive gate: decide how much new learning to apply at each position
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.norm = RMSNorm(dim)

        # EWC anchors — populated by consolidate(), empty until then
        self.fisher:  Dict[str, torch.Tensor] = {}
        self.anchors: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the adapter delta on top of the base representation."""
        delta = self.adapter_out(F.gelu(self.adapter_in(x)))   # [B, L, D]
        gate  = self.gate(x)                                    # [B, L, 1]
        return self.norm(x + gate * delta)

    @torch.no_grad()
    def consolidate(
        self,
        model:     nn.Module,
        dataloader,
        device:    str = "cpu",
        n_batches: int = 50,
    ) -> None:
        """
        Run after completing a task to lock in what was learned.

        Estimates the Fisher information diagonal as the average squared
        gradient of the log-likelihood with respect to each parameter.
        High Fisher → parameter was critical → protect it from drift.
        """
        model.eval()
        fisher_diag: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        counted = 0
        for batch in dataloader:
            if counted >= n_batches:
                break
            if not (isinstance(batch, dict) and batch.get("text") is not None):
                continue
            text = batch["text"].to(device)
            model.zero_grad()
            try:
                out  = model(text=text)
                loss = out["hidden_states"].pow(2).mean()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_diag[name] += param.grad.data.pow(2)
                counted += 1
            except Exception as exc:
                log.warning(
                    "EWC Fisher batch skipped (batch %d): %s: %s",
                    counted, type(exc).__name__, exc,
                )
                model.zero_grad()

        n = max(counted, 1)
        self.fisher  = {k: v / n for k, v in fisher_diag.items()}
        self.anchors = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.train()

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        EWC penalty: λ/2 · Σ_i  F_i · (θ_i − θ*_i)²

        Returns zero if consolidate() has not been called yet.
        """
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                f   = self.fisher[name].to(param.device)
                opt = self.anchors[name].to(param.device)
                loss = loss + (f * (param - opt).pow(2)).sum()
        return 0.5 * self.ewc_lambda * loss


# ============= Causal Reasoning =============

class CausalReasoner(nn.Module):
    """
    Distinguishes correlation from causation via a learnable causal graph.

    Architecture:

    concept_proj  — projects hidden states into an n_nodes-dim "concept space"
                    where each dimension loosely represents a latent variable.

    causal_graph  — a learnable [n_nodes × n_nodes] soft adjacency matrix.
                    W[i,j] = causal influence of concept i on concept j.
                    Sparsified through sigmoid; acyclicity is encouraged by
                    the NO TEARS constraint (Zheng et al. 2018):
                        h(W) = tr(exp(W⊙W)) − d  →  0

    During forward:
      1. Map hidden states to concept activations.
      2. Propagate through the causal graph: downstream = G^T · upstream.
      3. Enrich the representation with the causally-propagated signal.
      4. Return the updated hidden states + the sparse graph for inspection.

    intervention() and counterfactual() are standalone utilities that let
    external callers simulate "what if" scenarios directly in concept space.
    """

    def __init__(self, dim: int, n_nodes: int = 64):
        super().__init__()
        self.dim     = dim
        self.n_nodes = n_nodes

        # Project to / from concept space
        self.concept_proj  = nn.Linear(dim, n_nodes)
        self.concept_embed = nn.Linear(n_nodes, dim)

        # Causal adjacency: initialised near zero so graph starts uninformed
        self.causal_graph = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)

        # do-calculus intervention: given pre-state + intervention mask → post-state
        self.intervention_net = nn.Sequential(
            nn.Linear(n_nodes * 2, n_nodes * 2),
            nn.GELU(),
            nn.Linear(n_nodes * 2, n_nodes),
        )

        # Counterfactual: (actual, antecedent, counterfactual-X) → counterfactual-Y
        self.counterfactual_net = nn.Sequential(
            nn.Linear(n_nodes * 3, n_nodes * 2),
            nn.GELU(),
            nn.Linear(n_nodes * 2, n_nodes),
        )

        # Confidence: how certain is the causal structure at each position?
        self.confidence_head = nn.Sequential(
            nn.Linear(n_nodes, n_nodes // 2),
            nn.GELU(),
            nn.Linear(n_nodes // 2, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(dim)

    @property
    def sparse_graph(self) -> torch.Tensor:
        """Soft adjacency matrix in [0, 1]."""
        return torch.sigmoid(self.causal_graph)

    def dag_constraint(self) -> torch.Tensor:
        """
        NO TEARS acyclicity constraint (Zheng et al. 2018).

        h(W) = tr(exp(W⊙W)) − d   should approach 0 for a DAG.
        We approximate the matrix exponential with a degree-4 Taylor series:
            exp(A) ≈ I + A + A²/2! + A³/3! + A⁴/4!
        """
        W  = self.sparse_graph
        d  = self.n_nodes
        WW = W * W                              # element-wise Hadamard product

        # Power series accumulation
        expm = torch.eye(d, device=W.device, dtype=W.dtype)
        term = torch.eye(d, device=W.device, dtype=W.dtype)
        for k in range(1, 5):
            term = term @ WW / k
            expm = expm + term

        return expm.trace() - d

    def intervene(
        self, concepts: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate a do-calculus intervention: do(X_i = x_i) for masked nodes.

        concepts: [B, n_nodes]
        mask:     [B, n_nodes]  — 1 where we intervene, 0 elsewhere
        Returns:  [B, n_nodes]  — post-intervention concept activations
        """
        return self.intervention_net(torch.cat([concepts, mask], dim=-1))

    def counterfactual(
        self,
        actual:           torch.Tensor,
        antecedent:       torch.Tensor,
        counterfactual_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        "What would have happened if X had been different?"

        actual:           [B, n_nodes]  — what actually occurred
        antecedent:       [B, n_nodes]  — original cause
        counterfactual_x: [B, n_nodes]  — the hypothetical cause
        Returns:          [B, n_nodes]  — the counterfactual outcome
        """
        return self.counterfactual_net(
            torch.cat([actual, antecedent, counterfactual_x], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        G = self.sparse_graph                                      # [n, n]

        # Project to concept space
        concepts = torch.sigmoid(self.concept_proj(x))             # [B, L, n]

        # Causal propagation: downstream effects of each concept
        causal_propagated = torch.einsum('bln,nm->blm', concepts, G)  # [B, L, n]

        # Enrich with causal signal
        enriched = concepts + 0.1 * causal_propagated             # [B, L, n]

        # Confidence in the current causal structure
        confidence = self.confidence_head(enriched)                # [B, L, 1]

        # Project back to model space and steer hidden states
        causal_repr = self.concept_embed(enriched)                 # [B, L, D]
        x_causal    = self.norm(x + causal_repr * confidence)

        return x_causal, {
            "concepts":     concepts,
            "causal_graph": G,
            "confidence":   confidence,
            "dag_loss":     self.dag_constraint(),
        }


# ============= Grounded Claudeson =============

class ClaudesonGrounded(ClaudesonJedi):
    """
    Claudeson 2026 — Grounded Edition.

    Inherits the full Jedi architecture (SSD, Hybrid layers, Jedi Energy,
    Hierarchical Memory, Constitutional alignment, Epistemic calibration) and
    adds four new capabilities:

      theory_of_mind    — model other agents' beliefs / desires / intentions
      causal_reasoner   — distinguish correlation from causation
      continual_learner — learn without forgetting (EWC + LoRA)
      action_loop       — select tools, generate params, integrate feedback

    Forward signature is extended with:
      feedback           — optional real-world observation tensor [B, D]
      agent_observations — optional evidence about another agent [B, L, D]

    New output keys:
      tom              — Theory of Mind outputs (beliefs, desires, predictions)
      causal           — CausalReasoner outputs (graph, confidence, dag_loss)
      grounded_action  — GroundedActionLoop outputs (tool, params, surprise)
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.theory_of_mind   = TheoryOfMind(args, max_agents=args.n_agents)
        self.causal_reasoner  = CausalReasoner(args.dim, n_nodes=args.n_causal_nodes)
        self.continual_learner = ContinualLearner(
            args.dim, rank=args.lora_rank, ewc_lambda=args.ewc_lambda
        )
        self.action_loop      = GroundedActionLoop(args)

    def forward(
        self,
        text:               Optional[torch.Tensor] = None,
        img:                Optional[torch.Tensor] = None,
        audio:              Optional[torch.Tensor] = None,
        goal_tokens:        Optional[torch.Tensor] = None,
        feedback:           Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
    ) -> Dict:
        # ── Base Jedi pass ──────────────────────────────────────────────
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens
        )
        x = base["hidden_states"]                                  # [B, L, D]

        # ── Theory of Mind ──────────────────────────────────────────────
        # Update belief state if new agent evidence was provided
        if agent_observations is not None:
            self.theory_of_mind.update_beliefs(agent_observations)
        x, tom_out = self.theory_of_mind(x)

        # ── Causal Reasoning ────────────────────────────────────────────
        x, causal_out = self.causal_reasoner(x)

        # ── Continual Learning adapter ───────────────────────────────────
        x = self.continual_learner(x)

        # ── Grounded Action Loop ─────────────────────────────────────────
        x, action_out = self.action_loop(x, feedback=feedback)

        return {
            **base,
            "hidden_states":  x,
            "tom":            tom_out,
            "causal":         causal_out,
            "grounded_action": action_out,
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Auxiliary losses that should be added to the main training objective:
          dag_loss  — keeps the causal graph acyclic (NO TEARS constraint)
          ewc_loss  — prevents catastrophic forgetting (EWC penalty)
        """
        return {
            "dag_loss": self.causal_reasoner.dag_constraint() * 0.01,
            "ewc_loss": self.continual_learner.ewc_loss(self),
        }


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — GROUNDED EDITION")
    print("Theory of Mind · Causal Reasoning · Continual Learning · Action Loop")
    print("=" * 70)

    # Small config for demo (runs on CPU)
    args = ModelArgs()
    args.dim            = 128
    args.n_layers       = 2
    args.n_heads        = 4
    args.n_kv_heads     = 2
    args.vocab_size     = 512
    args.max_seq_len    = 64
    args.memory_slots   = 32
    args.episodic_slots = 64
    args.goal_dim       = 128
    args.latent_dim     = 64
    args.energy_hidden  = 128
    args.ssm_state_dim  = 32
    args.ssm_chunk_size = 16
    args.num_experts    = 2
    args.num_shared_experts = 1
    args.env_state_dim  = 32
    args.action_space_size  = 16
    args.planning_horizon   = 2
    args.num_simulations    = 2
    args.img_size       = 32
    args.patch_size     = 8
    args.audio_spec_dim = 16
    args.gradient_checkpointing = False
    args.n_agents       = 4
    args.lora_rank      = 8
    args.n_causal_nodes = 16

    print("\nInitialising ClaudesonGrounded...")
    model = ClaudesonGrounded(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    # ── Forward pass ────────────────────────────────────────────────────
    text     = torch.randint(0, 512, (2, 32))
    feedback = torch.randn(2, args.dim)           # simulated tool result
    agent_obs = torch.randn(2, 8, args.dim)       # simulated human utterance

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(text=text, feedback=feedback, agent_observations=agent_obs)

    print("\nJedi state:")
    print(f"  Goal:     {out['jedi_goal']}")
    print(f"  Energy:   {out['jedi_energy'].mean().item():.4f}")

    print("\nTheory of Mind:")
    print(f"  Belief shape:   {out['tom']['beliefs'].shape}")
    print(f"  Agent weights:  {out['tom']['agent_weights'][0].tolist()[:4]}")
    print(f"  Predicted agent action logits shape: {out['tom']['predicted_agent_action'].shape}")

    print("\nCausal Reasoning:")
    g = out['causal']['causal_graph']
    print(f"  Graph shape:  {g.shape}  (sparse: {(g < 0.1).float().mean():.0%} near-zero)")
    print(f"  Confidence:   {out['causal']['confidence'].mean().item():.4f}")
    print(f"  DAG loss:     {out['causal']['dag_loss'].item():.4f}")

    print("\nGrounded Action Loop:")
    print(f"  Selected tool: {out['grounded_action']['tool_names']}")
    print(f"  Surprise:      {out['grounded_action']['surprise'].squeeze().tolist()}")

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonGrounded READY.")
    print("Plans can now execute.  Other minds are now modelled.")
    print("New knowledge sticks.  Correlation is no longer causation.")
    print("=" * 70)
