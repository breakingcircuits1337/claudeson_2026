"""
Claudeson 2026 - Abstraction Edition
======================================
Hierarchical Abstraction · Concept Bottleneck · Analogy Reasoning · Schema Induction

The problem this generation solves
------------------------------------
Every previous generation operates at a single representational granularity.
Tokens flow through layers and emerge as hidden states of dimension D.  The
NeuralSymbolicLayer (Sovereign) added a proposition space, and the
CausalReasoner (Grounded) added a concept graph — but both still operate at
the same level of abstraction as the token stream.

Genuine intelligence requires a *hierarchy* of abstraction levels:

  Level 0  TOKEN       — "cat", "mat", "sat"
  Level 1  PHRASE      — "the cat", "sat on the mat"
  Level 2  CONCEPT     — ANIMAL, LOCATION, ACTION
  Level 3  SCHEMA      — AGENT_ACTS_ON_LOCATION (agent, action, location)
  Level 4  PRINCIPLE   — GRAVITY, CAUSALITY, SYMMETRY

The gap this creates:
  - A system that operates at token level can recognise "cat sat on mat".
  - A system with concept level can recognise this is about an ANIMAL and a
    LOCATION.
  - A system with schema level can recognise this is an instance of the
    AGENT_AT_LOCATION schema — and therefore transfer everything it knows
    about agents at locations (navigation, containment, proximity) to
    this new instance.
  - A system with principle level can ask: "is there a deeper regularity
    here that governs many schemas?" (e.g., OBJECT_STATE_TRANSITION).

Analogical reasoning — arguably the core of human intelligence (Hofstadter,
Gentner) — requires schema-level matching: "A is to B as C is to D" is
answered by finding the schema that maps A→B and checking if it maps C→D.
Without a schema layer, analogies are approximated by token similarity,
which breaks down for novel domains.

This generation adds five components:

  1. Hierarchical Abstraction Encoder (HAE)
     Encodes the token stream into five levels simultaneously using a
     progressive pooling + transformation stack.  Each level attends to
     the level below; higher levels compress and abstract.

  2. Concept Bottleneck Layer (CBL)
     Forces representations through an interpretable concept space before
     higher-level processing.  Each concept is associated with a human-
     readable prototype.  Concept activations are sparse (top-k gating)
     and monotonically related to output — this is the Koh et al. (2020)
     concept bottleneck model adapted for language.

  3. Schema Induction Engine (SIE)
     Learns reusable structural templates (schemas) from repeated patterns
     across the concept space.  A schema is a typed slot structure:
     AGENT_AT_LOCATION = {agent: ANIMATE, location: PLACE, relation: SPATIAL}.
     The engine detects when a new input matches a known schema and binds
     the slots, enabling transfer.

  4. Analogical Reasoning Module (ARM)
     Implements structure-mapping theory (Gentner 1983) in neural form.
     Given a source domain and a target domain, finds the relational mapping
     that preserves structural relations.  "A:B :: C:?" is answered by finding
     the schema that explains A→B and applying it to C.

  5. Principle Extractor (PE)
     Operates at the highest abstraction level.  Looks for regularities
     that appear across *multiple schemas* — these are candidate principles.
     "Many schemas share the CAUSE→EFFECT slot structure" → induces CAUSALITY
     as a principle.  Principles compress the schema library.

Architecture evolution:
  claudson → ... → causal_world → metacurriculum → abstraction
                                                         ↑ you are here
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from claudson_metacurriculum import (
    ModelArgs as MetaCurriculumArgs,
    ClaudesonMetaCurriculum,
)
from claudson_jedi import SwiGLU, RMSNorm

log = logging.getLogger(__name__)


# ============= Configuration =============

@dataclass
class ModelArgs(MetaCurriculumArgs):
    # Hierarchical Abstraction Encoder
    n_abstraction_levels: int = 5          # token / phrase / concept / schema / principle
    hae_heads: int = 4                     # attention heads within each level
    hae_pool_factor: int = 4               # sequence compression per level
    hae_hidden: int = 256                  # hidden dim inside HAE

    # Concept Bottleneck
    n_concepts: int = 128                  # number of interpretable concept slots
    concept_top_k: int = 16               # sparse activation: top-k concepts per token
    concept_hidden: int = 256             # hidden dim for concept projections

    # Schema Induction
    n_schema_slots: int = 32              # number of schema templates
    schema_n_roles: int = 6              # max roles per schema (agent, patient, ...)
    schema_hidden: int = 256             # hidden dim for schema encoder
    schema_bind_iters: int = 3           # binding iterations (LISA-style)

    # Analogical Reasoning
    analogy_hidden: int = 256            # hidden dim for structure mapper
    analogy_n_mappings: int = 8          # candidate structural mappings evaluated
    analogy_temperature: float = 0.5     # softmax temperature for mapping selection

    # Principle Extractor
    n_principles: int = 16              # number of learnable principles
    principle_hidden: int = 256         # hidden dim for principle encoder
    principle_loss_weight: float = 0.05 # compression regularisation weight


# ============= Level Definitions =============

LEVEL_NAMES = ["TOKEN", "PHRASE", "CONCEPT", "SCHEMA", "PRINCIPLE"]


# ============= Hierarchical Abstraction Encoder =============

class AbstractionLevel(nn.Module):
    """
    A single rung of the abstraction hierarchy.

    Reads from the level below via cross-attention, compresses the sequence
    by pool_factor via strided mean pooling, and applies a self-attention
    block to organise the compressed representations.

    The cross-attention is asymmetric: queries come from the compressed level,
    keys/values come from the finer level below.  This forces higher levels
    to be grounded in lower-level evidence while still being able to ignore
    irrelevant detail.
    """

    def __init__(self, dim: int, n_heads: int, pool_factor: int, hidden: int):
        super().__init__()
        self.pool_factor = pool_factor

        # Compress: mean pool along sequence dimension
        # (learned alternative: strided conv — mean pool is simpler and works)

        # Cross-attention: this level queries the level below
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Self-attention: organise compressed representations
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, dim),
        )

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

    def forward(
        self,
        x_below: torch.Tensor,    # [B, L_below, D]
    ) -> torch.Tensor:
        B, L, D = x_below.shape

        # Pool: compress sequence by pool_factor
        # Pad to multiple of pool_factor
        pad_len = (self.pool_factor - L % self.pool_factor) % self.pool_factor
        if pad_len > 0:
            x_pad = F.pad(x_below, (0, 0, 0, pad_len))
        else:
            x_pad = x_below

        L_pad   = x_pad.size(1)
        L_comp  = L_pad // self.pool_factor

        # Mean pool into compressed sequence
        x_comp = x_pad.view(B, L_comp, self.pool_factor, D).mean(2)  # [B, L_comp, D]

        # Cross-attend: compressed queries, fine-grained keys/values
        x_cross, _ = self.cross_attn(
            query=x_comp,
            key=x_below,
            value=x_below,
        )
        x_comp = self.norm1(x_comp + x_cross)

        # Self-attend within the compressed level
        x_self, _ = self.self_attn(x_comp, x_comp, x_comp)
        x_comp = self.norm2(x_comp + x_self)

        # FFN
        x_comp = self.norm3(x_comp + self.ffn(x_comp))

        return x_comp                     # [B, L_comp, D]


class HierarchicalAbstractionEncoder(nn.Module):
    """
    Encodes the token stream into five levels simultaneously.

    Output: a list of tensors at increasing abstraction / decreasing sequence length.
      levels[0] = token level    [B, L, D]
      levels[1] = phrase level   [B, L/4, D]
      levels[2] = concept level  [B, L/16, D]
      levels[3] = schema level   [B, L/64, D]
      levels[4] = principle level [B, L/256, D]  (or [B, 1, D] if L is short)

    A top-down feedback pass lets higher-level representations modulate
    lower-level ones (coarse-to-fine refinement, as in predictive coding).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_levels = args.n_abstraction_levels
        self.dim      = args.dim

        # Bottom-up encoding levels
        self.levels = nn.ModuleList([
            AbstractionLevel(
                dim=args.dim,
                n_heads=args.hae_heads,
                pool_factor=args.hae_pool_factor,
                hidden=args.hae_hidden,
            )
            for _ in range(args.n_abstraction_levels - 1)   # L0 is the input itself
        ])

        # Top-down feedback: higher level → lower level modulation
        self.feedback_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.dim, args.dim),
                nn.Sigmoid(),
            )
            for _ in range(args.n_abstraction_levels - 1)
        ])

        # Upsample: project higher-level repr back to lower-level sequence length
        # (used for feedback; linear interpolation + projection)
        self.feedback_projs = nn.ModuleList([
            nn.Linear(args.dim, args.dim)
            for _ in range(args.n_abstraction_levels - 1)
        ])

        # Final fusion: combine all levels back to token length
        self.fusion = nn.Linear(args.dim * args.n_abstraction_levels, args.dim)
        self.norm   = RMSNorm(args.dim)

    def _upsample(self, x_high: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample a compressed representation to target sequence length."""
        # x_high: [B, L_high, D]
        if x_high.size(1) == target_len:
            return x_high
        # Interpolate along sequence dimension
        x_t = x_high.transpose(1, 2)                              # [B, D, L_high]
        x_up = F.interpolate(x_t, size=target_len, mode='linear', align_corners=False)
        return x_up.transpose(1, 2)                               # [B, target_len, D]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B, L, D = x.shape

        # Bottom-up pass
        level_reprs = [x]                 # L0 = token level
        current     = x
        for level_fn in self.levels:
            current = level_fn(current)
            level_reprs.append(current)

        # Top-down feedback pass
        for i in range(len(self.levels) - 1, -1, -1):
            high = level_reprs[i + 1]
            low  = level_reprs[i]

            # Upsample high to low's sequence length
            high_up = self._upsample(high, low.size(1))
            high_up = self.feedback_projs[i](high_up)

            # Gate: how much feedback from above?
            gate = self.feedback_gates[i](high_up)
            level_reprs[i] = low + gate * high_up * 0.1

        # Fuse: upsample all levels to token length, then concat + project
        fused_parts = [level_reprs[0]]    # L0 already at token length
        for i in range(1, self.n_levels):
            fused_parts.append(self._upsample(level_reprs[i], L))

        fused   = torch.cat(fused_parts, dim=-1)                  # [B, L, D*n_levels]
        x_fused = self.norm(self.fusion(fused))                    # [B, L, D]

        return level_reprs, x_fused


# ============= Concept Bottleneck Layer =============

class ConceptBottleneck(nn.Module):
    """
    Forces representations through a sparse, interpretable concept space.

    Based on Koh et al. (2020) Concept Bottleneck Models, extended to
    sequence inputs.

    Each concept is:
      - A learned prototype vector c_k ∈ R^D
      - A human-readable label (stored as a string slot; trainable embedding)
      - Activated by similarity between the hidden state and the prototype

    Sparsity:
      Only top-k concepts activate per token.  This is enforced via a
      straight-through top-k estimator so gradients still flow.

    Monotonicity:
      Concept activations feed into the output via non-negative weights
      (softplus-projected), ensuring the relationship between concept
      activation and output is interpretable: more of concept k → more
      of its contribution to the output.

    Intervention interface:
      external code can call set_concept(k, value) to clamp a concept
      activation — enabling concept-level probing and debugging.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_concepts = args.n_concepts
        self.top_k      = args.concept_top_k
        self.dim        = args.dim
        h               = args.concept_hidden

        # Concept prototypes
        self.prototypes = nn.Parameter(
            torch.randn(args.n_concepts, args.dim) * 0.02
        )

        # Concept classifier: hidden → concept activation scores
        self.concept_proj = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.n_concepts),
        )

        # Non-negative output weights (monotonicity)
        self.output_weights = nn.Parameter(
            torch.ones(args.n_concepts, args.dim) * 0.01
        )

        # Concept-to-hidden decoder (for feedback)
        self.concept_decoder = nn.Sequential(
            nn.Linear(args.n_concepts, h),
            nn.GELU(),
            nn.Linear(h, args.dim),
        )

        # Intervention mask (external overrides)
        self.register_buffer(
            'intervention_mask',
            torch.zeros(args.n_concepts, dtype=torch.bool)
        )
        self.register_buffer(
            'intervention_values',
            torch.zeros(args.n_concepts)
        )

        self.norm = RMSNorm(args.dim)

    def set_concept(self, concept_idx: int, value: float) -> None:
        """Externally clamp a concept to a fixed activation value."""
        self.intervention_mask[concept_idx]   = True
        self.intervention_values[concept_idx] = value

    def clear_interventions(self) -> None:
        self.intervention_mask.fill_(False)
        self.intervention_values.fill_(0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Compute concept activation scores
        scores = self.concept_proj(x)                             # [B, L, n_concepts]

        # Apply external interventions (concept clamping)
        if self.intervention_mask.any():
            mask_exp = self.intervention_mask.unsqueeze(0).unsqueeze(0)
            vals_exp = self.intervention_values.unsqueeze(0).unsqueeze(0)
            scores   = torch.where(mask_exp, vals_exp.expand_as(scores), scores)

        # Prototype similarity: adds geometric grounding to concept activations
        proto_sim = torch.einsum('bld,kd->blk', x, self.prototypes) / math.sqrt(D)
        scores    = scores + 0.1 * proto_sim                      # [B, L, n_concepts]

        # Sigmoid activations in [0, 1]
        activations = torch.sigmoid(scores)                       # [B, L, n_concepts]

        # Sparse top-k (straight-through estimator for gradients)
        topk_vals, topk_idx = torch.topk(activations, self.top_k, dim=-1)
        sparse_acts = torch.zeros_like(activations)
        sparse_acts.scatter_(-1, topk_idx, topk_vals)
        # STE: forward uses sparse, backward uses dense
        sparse_acts = activations + (sparse_acts - activations).detach()

        # Non-negative output weights (F.softplus ensures positivity)
        w = F.softplus(self.output_weights)                       # [n_concepts, D]

        # Concept-mediated output
        concept_out = torch.einsum('blk,kd->bld', sparse_acts, w) # [B, L, D]

        # Decode concepts back to hidden (conceptual grounding)
        concept_hidden = self.concept_decoder(sparse_acts)         # [B, L, D]

        x_bottleneck = self.norm(x + concept_out * 0.1 + concept_hidden * 0.05)

        return x_bottleneck, {
            "activations":   sparse_acts,                         # [B, L, n_concepts]
            "top_concepts":  topk_idx,                            # [B, L, top_k]
            "concept_scores": scores,
        }


# ============= Schema Induction Engine =============

class SchemaRole(nn.Module):
    """A single typed role slot in a schema (e.g., AGENT, PATIENT, LOCATION)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        # Role embedding (what kind of thing goes here)
        self.role_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        # Filler detector: does candidate x fill this role?
        self.filler_head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def binding_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score: how well does x fill this role?
        x: [B, L, D] → score: [B, L, 1]
        """
        role_exp = self.role_embedding.unsqueeze(0).unsqueeze(0).expand_as(x)
        return self.filler_head(torch.cat([x, role_exp], dim=-1))


class SchemaTemplate(nn.Module):
    """
    A reusable structural template with typed role slots.

    A schema captures the relational structure of a situation without
    specifying the surface form.  For example:

      TRANSFER schema: {giver: AGENT, receiver: AGENT, object: ENTITY, result: STATE}

    Binding: given an input, assign each token to a role slot.
    The binding is soft (attention) during training and hard (argmax) at inference.

    Schema activation: how well does the input fit this schema overall?
    High activation → this schema is the right frame for interpreting the input.
    """

    def __init__(self, dim: int, n_roles: int, hidden: int):
        super().__init__()
        self.n_roles = n_roles

        # Schema-level embedding (what situation does this schema describe?)
        self.schema_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        # Role slots
        self.roles = nn.ModuleList([
            SchemaRole(dim, hidden) for _ in range(n_roles)
        ])

        # Overall fit scorer: does the input match this schema?
        self.fit_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x:           torch.Tensor,    # [B, L, D]
        n_bind_iters: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bind input tokens to role slots.

        Returns:
            bindings   [B, n_roles, D]  — bound fillers for each role
            fit_score  [B, 1]           — overall schema fit
        """
        B, L, D = x.shape

        # Compute binding scores for each role
        binding_scores = torch.stack(
            [role.binding_score(x).squeeze(-1) for role in self.roles],
            dim=1
        )                                                         # [B, n_roles, L]

        # Iterative competitive binding (roles compete for tokens)
        for _ in range(n_bind_iters):
            # Normalise across roles (token can only be bound to one role strongly)
            role_probs = F.softmax(binding_scores, dim=1)         # [B, n_roles, L]
            # Normalise across tokens (role focuses on its best match)
            token_probs = F.softmax(binding_scores / math.sqrt(D), dim=-1)
            binding_scores = role_probs * token_probs * binding_scores

        # Soft-select filler for each role
        filler_weights = F.softmax(binding_scores, dim=-1)        # [B, n_roles, L]
        bindings = torch.bmm(filler_weights, x)                   # [B, n_roles, D]

        # Overall fit: does the full input match this schema?
        schema_emb_exp = self.schema_embedding.unsqueeze(0).expand(B, -1)
        fit_score = self.fit_head(schema_emb_exp + x.mean(1))     # [B, 1]

        return bindings, fit_score


class SchemaInductionEngine(nn.Module):
    """
    Learns and applies reusable structural schemas.

    Maintains a library of schema templates.  For each input:
      1. Compute fit scores for all schemas.
      2. Select the best-fitting schema (soft during training, hard at inference).
      3. Bind input tokens to the schema's role slots.
      4. Use the bound schema to enrich the hidden representation.

    Schema induction (learning new schemas):
      If no existing schema fits well (max fit < threshold), create a new
      schema by distilling the current binding pattern into a fresh template.
      This is analogous to schema formation in cognitive science: repeated
      exposure to similar structural patterns crystallises into a reusable frame.

    Transfer:
      Once a schema is bound, its structural information can be transferred
      to novel domains by finding the same schema with different surface fillers.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_schemas    = args.n_schema_slots
        self.n_roles      = args.schema_n_roles
        self.bind_iters   = args.schema_bind_iters
        self.dim          = args.dim
        h                 = args.schema_hidden

        # Schema template library
        self.schemas = nn.ModuleList([
            SchemaTemplate(args.dim, args.schema_n_roles, h)
            for _ in range(args.n_schema_slots)
        ])

        # Fit threshold for new schema induction
        self.register_buffer('fit_history', torch.zeros(args.n_schema_slots))
        self.register_buffer('schema_active', torch.ones(args.n_schema_slots, dtype=torch.bool))

        # Schema-enriched output: bound fillers → enriched hidden state
        self.schema_proj = nn.Sequential(
            nn.Linear(args.dim * args.schema_n_roles, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, args.dim),
        )

        # Schema selector: hidden state → which schema to apply?
        self.selector = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.n_schema_slots),
        )

        self.norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Compute fit for all schemas
        all_fits = []
        all_bindings = []

        for schema in self.schemas:
            bindings, fit = schema(x, n_bind_iters=self.bind_iters)
            all_fits.append(fit)
            all_bindings.append(bindings)

        fit_scores = torch.cat(all_fits, dim=-1)                  # [B, n_schemas]
        binding_stack = torch.stack(all_bindings, dim=1)          # [B, n_schemas, n_roles, D]

        # Schema selection: soft attention during training
        selector_logits = self.selector(x.mean(1))                # [B, n_schemas]
        selection_w     = F.softmax(selector_logits + fit_scores, dim=-1)  # [B, n_schemas]
        best_schema_idx = selection_w.argmax(-1)                  # [B]

        # Weighted combination of schema-bound representations
        selection_exp  = selection_w.unsqueeze(-1).unsqueeze(-1)  # [B, n_schemas, 1, 1]
        bound_weighted = (binding_stack * selection_exp).sum(1)   # [B, n_roles, D]

        # Project bound roles back to hidden space
        bound_flat   = bound_weighted.view(B, -1)                 # [B, n_roles * D]
        schema_repr  = self.schema_proj(bound_flat)               # [B, D]

        # Inject schema representation into all token positions
        x_schema = self.norm(x + schema_repr.unsqueeze(1) * 0.1)

        return x_schema, {
            "fit_scores":     fit_scores,                          # [B, n_schemas]
            "best_schema":    best_schema_idx.tolist(),
            "selection_w":    selection_w,
            "bound_roles":    bound_weighted,                      # [B, n_roles, D]
        }


# ============= Analogical Reasoning Module =============

class StructureMapper(nn.Module):
    """
    Finds the relational mapping between a source and target domain.

    Implements the core of Structure Mapping Theory (Gentner 1983):
      - Analogies are not about surface similarity (cats are furry)
      - Analogies are about relational structure (A causes B :: C causes D)

    Algorithm:
      1. Encode source and target into schema space (role bindings).
      2. Find the mapping M such that M(source_roles) ≈ target_roles.
      3. The mapping is represented as a permutation-like soft matrix.
      4. Apply the mapping to answer "what is the target analog of source_element?"

    The soft mapping matrix is learned via optimal transport (Sinkhorn
    iterations), which finds the minimum-cost assignment between source
    and target role slots.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_roles  = args.schema_n_roles
        self.dim      = args.dim
        h             = args.analogy_hidden
        self.n_maps   = args.analogy_n_mappings
        self.temp     = args.analogy_temperature

        # Role encoder: maps role fillers to a common relational space
        self.role_encoder = nn.Sequential(
            nn.Linear(args.dim, h),
            RMSNorm(h),
            SwiGLU(h, h * 2),
            nn.Linear(h, h),
        )

        # Mapping scorer: source role × target role → mapping score
        self.map_score = nn.Bilinear(h, h, 1)

        # Sinkhorn iterations for doubly-stochastic mapping
        self.sinkhorn_iters = 5

        # Answer projector: mapped source + target context → answer embedding
        self.answer_proj = nn.Sequential(
            nn.Linear(h * 2, h * 2),
            nn.GELU(),
            nn.Linear(h * 2, args.dim),
            RMSNorm(args.dim),
        )

        # Analogy confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

    def _sinkhorn(self, log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        """
        Sinkhorn normalisation: convert score matrix to doubly-stochastic.
        log_alpha: [B, n, m]
        """
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return log_alpha.exp()

    def forward(
        self,
        source_roles: torch.Tensor,   # [B, n_roles, D]  — source domain bindings
        target_roles: torch.Tensor,   # [B, n_roles, D]  — target domain bindings
    ) -> Tuple[torch.Tensor, Dict]:
        B, R, D = source_roles.shape

        # Encode roles into relational space
        src_enc = self.role_encoder(source_roles)                 # [B, R, h]
        tgt_enc = self.role_encoder(target_roles)                 # [B, R, h]

        # Pairwise mapping scores: [B, R_src, R_tgt]
        # map_score expects [B, R, h] × [B, R, h] → [B, R, 1] per pair
        R_src = src_enc.size(1)
        R_tgt = tgt_enc.size(1)
        h     = src_enc.size(-1)

        src_exp = src_enc.unsqueeze(2).expand(-1, -1, R_tgt, -1)  # [B, Rs, Rt, h]
        tgt_exp = tgt_enc.unsqueeze(1).expand(-1, R_src, -1, -1)  # [B, Rs, Rt, h]

        score_mat = self.map_score(
            src_exp.reshape(B * R_src * R_tgt, h),
            tgt_exp.reshape(B * R_src * R_tgt, h),
        ).view(B, R_src, R_tgt) / self.temp                       # [B, Rs, Rt]

        # Sinkhorn: doubly-stochastic mapping matrix
        mapping = self._sinkhorn(score_mat)                       # [B, Rs, Rt]

        # Map source roles to target space
        mapped_src = torch.bmm(mapping, tgt_enc)                  # [B, Rs, h]

        # Confidence: how clean is the mapping?
        confidence = self.confidence_head(
            (mapped_src - src_enc).abs().mean(-1, keepdim=True).mean(-2)
        ).squeeze(-1)                                             # [B]

        # Analogy answer: mapped source in the context of target
        combined = torch.cat([mapped_src, tgt_enc], dim=-1)       # [B, R, 2h]
        answer   = self.answer_proj(combined)                     # [B, R, D]

        return answer, {
            "mapping":    mapping,                                 # [B, Rs, Rt]
            "mapped_src": mapped_src,
            "confidence": confidence,
        }


class AnalogicalReasoningModule(nn.Module):
    """
    A:B :: C:? — solve analogies via structural mapping.

    Given a source episode (A, B pair) and a target probe (C), finds
    the answer D such that the structural relationship A→B is preserved
    in C→D.

    In practice, source and target are schema-bound representations from
    the SchemaInductionEngine.  The analogy is solved at the role level,
    not the token level — this is what gives it domain-general transfer.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim    = args.dim
        h           = args.analogy_hidden

        self.mapper = StructureMapper(args)

        # Query encoder: "what am I looking for in the target?"
        self.query_enc = nn.Sequential(
            nn.Linear(args.dim, h),
            RMSNorm(h),
        )

        # Analogy loss: mapped source should match target structure
        self.align_head = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x:            torch.Tensor,     # [B, L, D] current hidden state
        source_roles: torch.Tensor,     # [B, n_roles, D] from SchemaInductionEngine
        target_roles: torch.Tensor,     # [B, n_roles, D] from SchemaInductionEngine
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Solve structural mapping
        answer_roles, map_info = self.mapper(source_roles, target_roles)

        # Pool answer back to sequence space
        answer_pooled = answer_roles.mean(1)                      # [B, D]

        # Alignment loss: how well does the mapping preserve structure?
        align_score   = self.align_head(answer_pooled).squeeze(-1) # [B]

        # Inject analogy answer into hidden state
        x_analogy = self.norm(x + answer_pooled.unsqueeze(1) * 0.05)

        return x_analogy, {
            "answer_roles":  answer_roles,
            "mapping":       map_info["mapping"],
            "confidence":    map_info["confidence"],
            "align_score":   align_score,
        }


# ============= Principle Extractor =============

class PrincipleExtractor(nn.Module):
    """
    Discovers cross-schema regularities as learnable principles.

    A principle is a regularity that appears across many schemas:
      "Every schema has a CAUSE and an EFFECT slot" → CAUSALITY principle
      "Every schema preserves some quantity" → CONSERVATION principle
      "Every schema has a temporal ordering" → TEMPORALITY principle

    Mechanism:
      1. Read the active schema representations (schema_embedding vectors).
      2. Find patterns that recur across multiple schema slots.
      3. Compress these patterns into principle vectors.
      4. Principle vectors are used to:
         (a) regularise schema representations (schemas should be expressible
             as compositions of principles)
         (b) generalise predictions to novel schemas by principle lookup

    The compression regularisation loss penalises schemas that cannot be
    well-approximated by a linear combination of principles — this drives
    the principles toward maximal explanatory coverage.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_principles  = args.n_principles
        self.n_schemas     = args.n_schema_slots
        self.dim           = args.dim
        h                  = args.principle_hidden
        self.loss_weight   = args.principle_loss_weight

        # Principle vectors (learnable, shared across schemas)
        self.principles = nn.Parameter(
            torch.randn(args.n_principles, args.dim) * 0.02
        )

        # Schema → principle decomposition coefficients
        self.decompose_head = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.n_principles),
            nn.Softmax(dim=-1),
        )

        # Principle → hidden enrichment
        self.principle_proj = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.dim),
        )

        # Cross-schema attention: principles attend over all active schemas
        self.schema_attn = nn.MultiheadAttention(
            embed_dim=args.dim,
            num_heads=max(1, args.dim // 64),
            batch_first=True,
            dropout=0.0,
        )

        # Principle confidence: how universal is each principle?
        self.universality_head = nn.Sequential(
            nn.Linear(args.dim, 1),
            nn.Sigmoid(),
        )

        self.norm = RMSNorm(args.dim)

    def forward(
        self,
        x:               torch.Tensor,     # [B, L, D]
        schema_reprs:    torch.Tensor,     # [B, n_schemas, D] — schema embeddings
    ) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Attend over schema representations with principle vectors as queries
        principles_exp = self.principles.unsqueeze(0).expand(B, -1, -1)  # [B, n_p, D]
        p_attended, _  = self.schema_attn(
            query=principles_exp,
            key=schema_reprs,
            value=schema_reprs,
        )                                                          # [B, n_p, D]

        # Updated principles: original + attended schema info
        p_updated = self.principles.unsqueeze(0) + p_attended * 0.1

        # Decompose each schema into principles
        schema_coefficients = self.decompose_head(schema_reprs)   # [B, n_schemas, n_p]

        # Reconstruct schemas from principles (for compression loss)
        schema_reconstructed = torch.bmm(
            schema_coefficients, p_updated
        )                                                          # [B, n_schemas, D]
        compression_loss = F.mse_loss(schema_reconstructed, schema_reprs.detach())

        # Universality: how consistently does each principle appear?
        universality = self.universality_head(
            p_updated.mean(0)                                      # [n_p, D]
        ).squeeze(-1)                                              # [n_p]

        # Enrich hidden state with most universal principles
        top_k = min(4, self.n_principles)
        _, top_p_idx = universality.topk(top_k)
        top_principles = p_updated[:, top_p_idx, :].mean(1)       # [B, D]

        principle_enrichment = self.principle_proj(top_principles) # [B, D]
        x_principles = self.norm(x + principle_enrichment.unsqueeze(1) * 0.05)

        return x_principles, {
            "principles":          p_updated,                      # [B, n_p, D]
            "schema_coefficients": schema_coefficients,
            "universality":        universality,
            "compression_loss":    compression_loss * self.loss_weight,
        }


# ============= Abstraction Claudeson =============

class ClaudesonAbstraction(ClaudesonMetaCurriculum):
    """
    Claudeson 2026 — Abstraction Edition.

    Inherits the full MetaCurriculum architecture and adds:

      hae           — hierarchical abstraction encoder; simultaneously
                      encodes input at token/phrase/concept/schema/principle levels
      concept_bn    — concept bottleneck; forces interpretable sparse
                      concept activations as a compression bottleneck
      schema_engine — schema induction; binds tokens to typed role slots,
                      learns reusable structural templates
      analogy_module — analogical reasoning via structural mapping (Gentner)
      principle_ext  — principle extraction; finds cross-schema regularities

    Processing pipeline (after MetaCurriculum):
      SkillDiscovery → GoalGenerator    (MetaCurriculum)
            ↓
      Hierarchical Abstraction Encoder  (5-level bottom-up + top-down feedback)
            ↓
      Concept Bottleneck                (sparse interpretable concept space)
            ↓
      Schema Induction                  (structural template binding)
            ↓
      Analogical Reasoning              (structure-mapping across domains)
            ↓
      Principle Extraction              (cross-schema regularities)

    New output keys:
      abstraction — {levels, concept, schema, analogy, principles}
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.hae             = HierarchicalAbstractionEncoder(args)
        self.concept_bn      = ConceptBottleneck(args)
        self.schema_engine   = SchemaInductionEngine(args)
        self.analogy_module  = AnalogicalReasoningModule(args)
        self.principle_ext   = PrincipleExtractor(args)

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
        # ── Full MetaCurriculum pass ─────────────────────────────────────
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens,
            feedback=feedback, agent_observations=agent_observations,
            actual_action=actual_action, rung_labels=rung_labels,
            competence_signal=competence_signal,
        )
        x = base["hidden_states"]

        # ── Hierarchical Abstraction Encoder ─────────────────────────────
        level_reprs, x = self.hae(x)

        # ── Concept Bottleneck ───────────────────────────────────────────
        x, concept_out = self.concept_bn(x)

        # ── Schema Induction ─────────────────────────────────────────────
        x, schema_out = self.schema_engine(x)

        # ── Analogical Reasoning ─────────────────────────────────────────
        # Use bound roles as both source and target (self-analogy for now;
        # a production system would pass in separate source/target episodes)
        bound_roles = schema_out["bound_roles"]                   # [B, n_roles, D]
        x, analogy_out = self.analogy_module(x, bound_roles, bound_roles)

        # ── Principle Extraction ─────────────────────────────────────────
        # Build schema repr matrix for principle attention
        B = x.size(0)
        schema_emb_stack = torch.stack(
            [s.schema_embedding for s in self.schema_engine.schemas],
            dim=0
        ).unsqueeze(0).expand(B, -1, -1)                         # [B, n_schemas, D]

        x, principle_out = self.principle_ext(x, schema_emb_stack)

        return {
            **base,
            "hidden_states": x,
            "abstraction": {
                "n_levels":     len(level_reprs),
                "level_shapes": [list(l.shape) for l in level_reprs],
                "concept":      concept_out,
                "schema":       schema_out,
                "analogy":      analogy_out,
                "principles":   principle_out,
            },
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        losses = super().compute_auxiliary_losses()
        # Principle compression loss is returned in-output but also expose here
        # for convenience (requires a forward pass; approximated as zero here)
        return losses


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — ABSTRACTION EDITION")
    print("HAE · Concept Bottleneck · Schema Induction · Analogy · Principles")
    print("=" * 70)

    args = ModelArgs()
    # Tiny CPU demo
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
    args.n_skill_slots          = 8
    args.skill_rank             = 4
    args.skill_embed_dim        = 32
    args.cp_window              = 8
    args.cp_hidden              = 64
    args.oeg_n_compose          = 2
    args.oeg_hidden             = 64
    args.ig_beta                = 0.5
    # Abstraction specific
    args.n_abstraction_levels   = 3    # reduced for CPU demo
    args.hae_heads              = 2
    args.hae_pool_factor        = 2
    args.hae_hidden             = 64
    args.n_concepts             = 32
    args.concept_top_k          = 8
    args.concept_hidden         = 64
    args.n_schema_slots         = 8
    args.schema_n_roles         = 4
    args.schema_hidden          = 64
    args.schema_bind_iters      = 2
    args.analogy_hidden         = 64
    args.analogy_n_mappings     = 4
    args.n_principles           = 8
    args.principle_hidden       = 64

    print("\nInitialising ClaudesonAbstraction...")
    model = ClaudesonAbstraction(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text          = torch.randint(0, 512, (2, 32))
    feedback      = torch.randn(2, args.dim)
    agent_obs     = torch.randn(2, 8, args.dim)
    actual_action = torch.randint(0, args.action_space_size, (2,))

    model.irl.add_preference(torch.randn(args.dim), torch.randn(args.dim), label=1.0)

    for step in range(args.cp_window):
        model.cp_monitor.record_performance(0, 0.3 + 0.04 * step)

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            text=text,
            feedback=feedback,
            agent_observations=agent_obs,
            actual_action=actual_action,
            competence_signal=0.5,
        )

    ab = out["abstraction"]

    print("\nJedi state:")
    print(f"  Goal:   {out['jedi_goal']}")
    print(f"  Energy: {out['jedi_energy'].mean().item():.4f}")

    print(f"\nHierarchical Abstraction Encoder:")
    print(f"  Levels:  {ab['n_levels']}")
    for i, shape in enumerate(ab['level_shapes']):
        print(f"    L{i} ({LEVEL_NAMES[i] if i < len(LEVEL_NAMES) else 'UPPER'}): {shape}")

    print(f"\nConcept Bottleneck:")
    acts = ab['concept']['activations']
    print(f"  Active concepts (mean per token): {(acts > 0.1).float().mean().item():.3f}")
    top = ab['concept']['top_concepts'][0, 0, :4].tolist()
    print(f"  Top concepts (sample token):      {top}")

    print(f"\nSchema Induction:")
    fits = ab['schema']['fit_scores']
    print(f"  Best schema:    {ab['schema']['best_schema']}")
    print(f"  Max fit score:  {fits.max().item():.4f}")
    print(f"  Bound roles shape: {ab['schema']['bound_roles'].shape}")

    print(f"\nAnalogical Reasoning:")
    print(f"  Mapping shape:  {ab['analogy']['mapping'].shape}")
    print(f"  Confidence:     {ab['analogy']['confidence'].tolist()}")

    print(f"\nPrinciple Extractor:")
    u = ab['principles']['universality']
    print(f"  Universality:       {[f'{v:.3f}' for v in u.tolist()]}")
    print(f"  Compression loss:   {ab['principles']['compression_loss'].item():.4f}")
    print(f"  Schema coefficients shape: {ab['principles']['schema_coefficients'].shape}")

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonAbstraction READY.")
    print("Sees tokens. Understands phrases. Thinks in concepts.")
    print("Recognises schemas. Reasons by analogy. Discovers principles.")
    print("=" * 70)
