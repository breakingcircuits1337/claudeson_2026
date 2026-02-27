"""
Claudeson 2026 - Transcendent Edition
=======================================
Four frontiers beyond Sovereign:

  1. Global Workspace         Global Workspace Theory (Baars 1988, Dehaene 2001).
                              Specialised modules compete via a sparse attention
                              bottleneck to broadcast their signal to a shared
                              global workspace.  The winner's representation
                              propagates to every other module.  This is the
                              leading neuroscientific theory of how conscious
                              access works — not metaphysics, but a concrete
                              information-routing architecture.

  2. Compositional Program    A differentiable program inductor.  Reads the
     Synthesis                current reasoning trace and emits a latent
                              "program" as a sequence of discrete op-codes over
                              a register bank.  Programs are executed inside the
                              model; execution results feed back into the hidden
                              state.  Closes the loop between neural pattern
                              matching and symbolic computation.

  3. Inverse Reward Learning  Learns what the human *actually* values by
                              observing their choices, not from explicit labels.
                              Maintains a reward model updated via maximum
                              entropy IRL; the inferred reward shapes future
                              planning without needing a hand-crafted objective.

  4. Neuromorphic Event       Leaky Integrate-and-Fire dynamics over the hidden
     Processing               state.  Each "neuron" accumulates input until it
                              fires (threshold crossed), then resets.  Only fired
                              neurons propagate signal — sparse, asynchronous,
                              and time-aware.  Efficient like biology, without
                              losing the expressiveness of dense networks.

Architecture evolution:
  claudson → extended → infinite → pro → ultimate → jedi → grounded
          → sovereign → transcendent
                             ↑ you are here
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

from claudson_sovereign import (
    ModelArgs as SovereignArgs,
    ClaudesonSovereign,
)
from claudson_jedi import SwiGLU, RMSNorm


# ============= Configuration =============

@dataclass
class ModelArgs(SovereignArgs):
    # Global Workspace
    n_workspace_slots:  int = 16    # broadcast slots in the global workspace
    gw_competition_k:   int = 4     # top-k modules compete per step
    gw_broadcast_steps: int = 2     # iterations of global broadcast

    # Program Synthesis
    n_ops:             int   = 16   # number of primitive op-codes
    n_registers:       int   = 8    # register bank size (in model-dim vectors)
    prog_steps:        int   = 4    # execution steps per forward pass
    prog_hidden:       int   = 256  # hidden dim inside the synthesiser
    prog_inject_scale: float = 0.1  # scale of program output added back to hidden states

    # Inverse Reward Learning
    irl_hidden:         int = 256   # hidden dim for reward model
    irl_n_preferences:  int = 32    # trajectory preference pairs in buffer
    irl_update_every:   int = 16    # steps between IRL reward model updates

    # Neuromorphic
    lif_threshold:  float = 0.5    # membrane potential fire threshold
    lif_leak:       float = 0.9    # leak factor per step (< 1.0)
    lif_reset:      float = 0.0    # post-fire reset potential
    lif_steps:      int   = 4      # number of LIF simulation steps


# ============= Global Workspace Theory =============

class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory — Baars (1988), Dehaene (2001).

    In the neuroscientific theory, specialised unconscious processors
    (vision, language, memory, motor, ...) compete for access to a limited
    global workspace.  The winning broadcast ignites a 'global ignition' —
    the winning representation is simultaneously made available to all other
    processors.  This is what gives rise to the unified, reportable content
    of conscious experience.

    Implementation:
      workspace   — a bank of n_slots learnable slot vectors [n_slots, D]
      competition — each module projects its pooled representation into slot
                    space; sparse top-k softmax selects the winners
      broadcast   — the winning slot content is written back to every module
                    via cross-attention
      ignition    — a binary spike: did this token win global access? (1/0)

    Multiple forward passes let the workspace settle (gw_broadcast_steps).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim          = args.dim
        self.n_slots      = args.n_workspace_slots
        self.top_k        = args.gw_competition_k
        self.n_broadcasts = args.gw_broadcast_steps

        # The global workspace itself: persistent slot vectors
        self.workspace = nn.Parameter(torch.randn(args.n_workspace_slots, args.dim) * 0.02)

        # Competition: project hidden states → slot logits
        self.compete_proj = nn.Linear(args.dim, args.n_workspace_slots)

        # Write: winning modules write into the workspace
        self.write_proj = nn.Linear(args.dim, args.dim)

        # Broadcast: workspace → all positions via cross-attention
        self.broadcast_attn = nn.MultiheadAttention(
            embed_dim=args.dim,
            num_heads=max(1, args.dim // 64),
            batch_first=True,
            dropout=0.0,
        )

        # Ignition gate: how strongly does each position fire globally?
        self.ignition_head = nn.Sequential(
            nn.Linear(args.dim, 1),
            nn.Sigmoid(),
        )

        self.norm_pre  = RMSNorm(args.dim)
        self.norm_post = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        ws = self.workspace.unsqueeze(0).expand(B, -1, -1)   # [B, n_slots, D]

        ignition_history = []

        for _ in range(self.n_broadcasts):
            # Competition: each position bids for a workspace slot
            logits = self.compete_proj(self.norm_pre(x))      # [B, L, n_slots]

            # Sparse top-k competition: only top_k positions per slot win
            # We transpose to [B, n_slots, L] and take top-k over positions
            slot_logits = logits.transpose(1, 2)              # [B, n_slots, L]
            topk_vals, topk_idx = torch.topk(slot_logits, min(self.top_k, L), dim=-1)
            sparse_weights = torch.zeros_like(slot_logits)
            sparse_weights.scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
            # sparse_weights: [B, n_slots, L]  — each slot's winner distribution

            # Write: winning positions write to workspace slots
            written = torch.bmm(sparse_weights, self.write_proj(x))  # [B, n_slots, D]
            ws = self.norm_post(ws + written)

            # Broadcast: workspace content → all positions (global ignition)
            x_broadcast, _ = self.broadcast_attn(
                query=x,
                key=ws,
                value=ws,
            )                                                  # [B, L, D]
            x = self.norm_post(x + x_broadcast)

            # Ignition score per position
            ignition = self.ignition_head(x)                  # [B, L, 1]
            ignition_history.append(ignition)

        # Final ignition: mean across broadcast steps
        final_ignition = torch.stack(ignition_history, dim=0).mean(0)  # [B, L, 1]

        return x, {
            "workspace":  ws,                                 # [B, n_slots, D]
            "ignition":   final_ignition.squeeze(-1),         # [B, L]
            "peak_ignition": final_ignition.max().item(),
        }


# ============= Compositional Program Synthesis =============

class ProgramSynthesizer(nn.Module):
    """
    Differentiable program induction inside the forward pass.

    The idea: instead of generating tokens that *describe* a computation,
    generate a short executable program and *run* it.

    Architecture:
      encoder      — reads the hidden state, produces an initial register bank
      controller   — at each step, produces a soft distribution over op-codes
                     and over register addresses (read/write)
      ops          — a set of n_ops differentiable primitive operations
                     (add, gate, attend, norm, project, ...)
      executor     — applies the chosen op to the chosen registers
      decoder      — reads the final register bank back to hidden space

    Op-codes (all differentiable, soft-selected via gumbel-softmax):
      0  NOP        — no operation (identity)
      1  ADD        — register[dst] += register[src]
      2  GATE       — register[dst] = sigmoid(register[src]) * register[dst]
      3  NORM       — register[dst] = layernorm(register[src])
      4  PROJ       — register[dst] = linear(register[src])
      5  RESIDUAL   — register[dst] = register[src] + register[dst]
      6  ATTEND     — attend(register[query], register[key], register[value])
      7  NEGATE     — register[dst] = -register[src]
      8  SCALE      — register[dst] = register[src] * learned_scalar
      9  SWAP       — exchange register[a] and register[b]
      10 MAX        — register[dst] = max(register[a], register[b])
      11 MIN        — register[dst] = min(register[a], register[b])
      12 RELU       — register[dst] = relu(register[src])
      13 TANH       — register[dst] = tanh(register[src])
      14 OUTER      — register[dst] = outer product projection
      15 HALT       — stop execution (soft: weight remaining steps to zero)

    The program is synthesised conditioned on the current reasoning trace,
    executed inside the model, and the result is fed back into the hidden state.
    """

    N_OPS = 16

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim          = args.dim
        self.n_regs       = args.n_registers
        self.steps        = args.prog_steps
        self.inject_scale = args.prog_inject_scale
        h                 = args.prog_hidden

        # Encode hidden state → initial register bank
        self.encoder = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.n_registers * args.dim),
        )

        # Controller: (pooled hidden + register summary) → op + register addresses
        ctrl_in = args.dim + args.n_registers * args.dim
        self.controller = nn.GRUCell(ctrl_in, h)
        self.op_head    = nn.Linear(h, self.N_OPS)
        self.src_head   = nn.Linear(h, args.n_registers)  # source register
        self.dst_head   = nn.Linear(h, args.n_registers)  # destination register
        self.aux_head   = nn.Linear(h, args.n_registers)  # auxiliary (for 3-arg ops)

        # Per-op learned projections (used by PROJ, ATTEND, OUTER, SCALE)
        self.op_proj  = nn.Linear(args.dim, args.dim, bias=False)
        self.op_scale = nn.Parameter(torch.ones(1))
        self.op_attn  = nn.MultiheadAttention(args.dim, max(1, args.dim // 64),
                                               batch_first=True)

        # Decode final registers → hidden space
        self.decoder = nn.Sequential(
            nn.Linear(args.n_registers * args.dim, h),
            nn.GELU(),
            nn.Linear(h, args.dim),
        )

        self.norm = RMSNorm(args.dim)
        self.ctrl_h0 = nn.Parameter(torch.zeros(h))

    def _soft_read(self, regs: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        """Soft register read: addr is a [B, n_regs] weight distribution."""
        # regs: [B, n_regs, D], addr: [B, n_regs] → [B, D]
        return torch.einsum('br,brd->bd', addr, regs)

    def _soft_write(
        self, regs: torch.Tensor, addr: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Soft register write: distribute value to all registers by addr weight."""
        # addr: [B, n_regs], value: [B, D] → regs: [B, n_regs, D]
        return regs + addr.unsqueeze(-1) * value.unsqueeze(1)

    def _execute_op(
        self,
        op_weights: torch.Tensor,       # [B, N_OPS]
        src: torch.Tensor,              # [B, D] — soft-read source register
        dst: torch.Tensor,              # [B, D] — soft-read destination register
        aux: torch.Tensor,              # [B, D] — soft-read auxiliary register
    ) -> torch.Tensor:
        """Apply all ops softly, weighted by op_weights."""
        results = [
            dst,                                                    # 0  NOP
            dst + src,                                              # 1  ADD
            torch.sigmoid(src) * dst,                               # 2  GATE
            F.layer_norm(src, src.shape[-1:]),                      # 3  NORM
            self.op_proj(src),                                      # 4  PROJ
            src + dst,                                              # 5  RESIDUAL
            # 6 ATTEND: treat src as Q, dst as K, aux as V
            self.op_attn(
                src.unsqueeze(1), dst.unsqueeze(1), aux.unsqueeze(1)
            )[0].squeeze(1),
            -src,                                                   # 7  NEGATE
            src * self.op_scale,                                    # 8  SCALE
            src,                                                    # 9  SWAP (returns src)
            torch.maximum(src, dst),                                # 10 MAX
            torch.minimum(src, dst),                                # 11 MIN
            F.relu(src),                                            # 12 RELU
            torch.tanh(src),                                        # 13 TANH
            self.op_proj(src * dst),                                # 14 OUTER (approx)
            dst,                                                    # 15 HALT (NOP effect)
        ]
        stacked = torch.stack(results, dim=1)                       # [B, N_OPS, D]
        return (op_weights.unsqueeze(-1) * stacked).sum(1)          # [B, D]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape
        pooled = x.mean(1)                                          # [B, D]

        # Initialise register bank from hidden state
        regs = self.encoder(pooled).view(B, self.n_regs, D)        # [B, n_regs, D]

        ctrl_h = self.ctrl_h0.unsqueeze(0).expand(B, -1)           # [B, h]

        op_log = []
        halt_acc = torch.zeros(B, 1, device=x.device)

        for step in range(self.steps):
            # Controller input: pooled hidden + flattened registers
            ctrl_in = torch.cat([pooled, regs.view(B, -1)], dim=-1)
            ctrl_h  = self.controller(ctrl_in, ctrl_h)

            # Decode operation and addresses
            op_logits = self.op_head(ctrl_h)                        # [B, N_OPS]
            op_w      = F.gumbel_softmax(op_logits, tau=1.0, hard=False)

            src_w = F.softmax(self.src_head(ctrl_h), dim=-1)        # [B, n_regs]
            dst_w = F.softmax(self.dst_head(ctrl_h), dim=-1)
            aux_w = F.softmax(self.aux_head(ctrl_h), dim=-1)

            # Soft register reads
            src_val = self._soft_read(regs, src_w)                  # [B, D]
            dst_val = self._soft_read(regs, dst_w)
            aux_val = self._soft_read(regs, aux_w)

            # Execute op
            halt_weight = op_w[:, 15:16]                            # HALT weight
            halt_acc    = halt_acc + halt_weight
            live_weight = torch.clamp(1.0 - halt_acc, min=0.0)

            result = self._execute_op(op_w, src_val, dst_val, aux_val)

            # Soft write result to destination register, gated by liveness
            regs = self._soft_write(regs, dst_w, result * live_weight)

            op_log.append(op_w.argmax(-1))                          # hard op for logging

        # Decode final register state back to model dim
        prog_out = self.decoder(regs.view(B, -1))                   # [B, D]
        prog_out = self.norm(prog_out)

        # Inject back into sequence (add to all positions, gated by ignition-like score)
        x_prog = self.norm(x + prog_out.unsqueeze(1) * self.inject_scale)

        return x_prog, {
            "final_registers": regs,                                 # [B, n_regs, D]
            "program_output":  prog_out,                             # [B, D]
            "op_trace":        torch.stack(op_log, dim=1),           # [B, steps]
        }


# ============= Inverse Reward Learning =============

class InverseRewardLearner(nn.Module):
    """
    Learns what the agent *actually* values by observing choices.

    Standard RL needs a hand-crafted reward function.  IRL (Ng & Russell 2000)
    inverts this: given observed behaviour, recover the reward function that
    makes that behaviour optimal.

    Here we implement a neural approximation of Maximum Entropy IRL
    (Ziebart et al. 2008):
      reward_model  — a network R(s) → scalar that predicts reward from state
      preference    — given two trajectories (preferred, rejected), the model
                      learns to assign higher reward to the preferred one
      planning      — the inferred reward shapes action selection during forward

    The preference buffer stores (hidden_a, hidden_b, label) triples where
    label=1 means hidden_a was preferred.  The model trains R to satisfy
    R(a) > R(b) when label=1 via a Bradley-Terry preference model:
        P(a > b) = sigmoid(R(a) - R(b))

    During inference, R(current_hidden) is used as an intrinsic reward signal
    that steers the action loop toward value-aligned behaviour.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim          = args.dim
        h                 = args.irl_hidden
        self.buffer_size  = args.irl_n_preferences

        # Reward model R(s) → scalar
        self.reward_model = nn.Sequential(
            nn.Linear(args.dim, h),
            nn.GELU(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, 1),
        )

        # Feature extractor: compress hidden state to reward-relevant features
        self.feature_proj = nn.Sequential(
            nn.Linear(args.dim, h),
            RMSNorm(h),
            SwiGLU(h, h * 2),
            nn.Linear(h, args.dim),
        )

        # Preference update head: (R_a, R_b) → preference logit
        self.pref_head = nn.Linear(2, 1)

        # Preference buffer (stored as tensors; updated externally)
        self.register_buffer(
            'pref_buffer_a',
            torch.zeros(args.irl_n_preferences, args.dim)
        )
        self.register_buffer(
            'pref_buffer_b',
            torch.zeros(args.irl_n_preferences, args.dim)
        )
        self.register_buffer(
            'pref_labels',
            torch.zeros(args.irl_n_preferences)
        )
        self.register_buffer('buffer_ptr', torch.tensor(0))
        self.register_buffer('buffer_full', torch.tensor(False))

        self.norm = RMSNorm(args.dim)

    @torch.no_grad()
    def add_preference(
        self,
        state_a: torch.Tensor,  # [D] — preferred state
        state_b: torch.Tensor,  # [D] — rejected state
        label:   float = 1.0,   # 1.0 = a preferred; 0.0 = b preferred
    ) -> None:
        """Add a (preferred, rejected) pair to the preference buffer."""
        ptr = int(self.buffer_ptr.item())
        self.pref_buffer_a[ptr] = state_a.detach().mean(0) if state_a.dim() > 1 else state_a.detach()
        self.pref_buffer_b[ptr] = state_b.detach().mean(0) if state_b.dim() > 1 else state_b.detach()
        self.pref_labels[ptr]   = label
        self.buffer_ptr.data    = torch.tensor((ptr + 1) % self.buffer_size)
        if ptr + 1 >= self.buffer_size:
            self.buffer_full.data = torch.tensor(True)

    def preference_loss(self) -> torch.Tensor:
        """
        Bradley-Terry preference loss:
            L = -E[ label * log P(a>b) + (1-label) * log P(b>a) ]
        where P(a>b) = sigmoid(R(a) - R(b))
        """
        if not self.buffer_full.item() and self.buffer_ptr.item() == 0:
            return torch.tensor(0.0)

        n = self.buffer_size if self.buffer_full.item() else int(self.buffer_ptr.item())
        r_a = self.reward_model(
            self.feature_proj(self.pref_buffer_a[:n])
        ).squeeze(-1)                                               # [n]
        r_b = self.reward_model(
            self.feature_proj(self.pref_buffer_b[:n])
        ).squeeze(-1)

        labels = self.pref_labels[:n]
        logits  = r_a - r_b
        loss    = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Extract reward-relevant features
        features = self.feature_proj(x)                             # [B, L, D]

        # Score current hidden states
        reward = self.reward_model(features)                        # [B, L, 1]
        reward_norm = torch.sigmoid(reward)                         # [B, L, 1]

        # Intrinsic reward signal: up-weight high-value positions
        x_irl = self.norm(x + features * reward_norm)

        # Aggregate reward for planning signal
        value_signal = reward.mean(1)                               # [B, 1]

        return x_irl, {
            "reward":       reward.squeeze(-1),                     # [B, L]
            "value_signal": value_signal.squeeze(-1),               # [B]
            "pref_loss":    self.preference_loss(),
        }


# ============= Neuromorphic Event Processing =============

class NeuromorphicProcessor(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) dynamics over the hidden state.

    Biological neurons:
      1. Accumulate (integrate) incoming current into membrane potential.
      2. When potential crosses a threshold, emit a spike (fire).
      3. Reset to resting potential after firing.
      4. Leak (decay) potential between inputs — recent events matter more.

    This maps onto transformer hidden states as follows:
      membrane  — running membrane potential per hidden dimension
      input     — the current hidden state is treated as 'current input'
      fire      — positions where potential > threshold emit a spike mask
      reset     — fired positions reset potential to lif_reset
      leak      — potential decays by lif_leak each step

    Sparsity:  only *fired* positions propagate their signal.  Unfired positions
    pass through unchanged.  This creates natural sparsity without any explicit
    pruning — the dynamics self-regulate.

    Time-awareness:  because the membrane accumulates across LIF steps within a
    single forward pass, early inputs affect late outputs via the persistent
    membrane potential — an implicit temporal context without explicit RNN state.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim       = args.dim
        self.threshold = args.lif_threshold
        self.leak      = args.lif_leak
        self.reset_val = args.lif_reset
        self.steps     = args.lif_steps

        # Input projection: hidden → membrane current
        self.input_proj = nn.Linear(args.dim, args.dim, bias=False)

        # Threshold is learnable per-dimension (heterogeneous neurons)
        self.threshold_vec = nn.Parameter(
            torch.full((args.dim,), args.lif_threshold)
        )

        # Output projection: spikes → output hidden
        self.output_proj = nn.Linear(args.dim, args.dim, bias=False)

        # Lateral inhibition: fired neurons suppress neighbours
        self.inhibit_proj = nn.Linear(args.dim, args.dim, bias=False)

        self.norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, D = x.shape

        # Initialise membrane potential to zero for this forward pass
        membrane = torch.zeros_like(x)                              # [B, L, D]

        all_spikes   = []
        all_fire_rates = []

        for step in range(self.steps):
            # 1. Integrate: add input current to membrane
            current  = self.input_proj(x)                          # [B, L, D]
            membrane = membrane + current

            # 2. Lateral inhibition: high-firing neurons suppress neighbours
            if step > 0 and all_spikes:
                inhibition = self.inhibit_proj(all_spikes[-1])
                membrane   = membrane - 0.1 * inhibition

            # 3. Fire: positions where membrane > threshold
            fire_mask = (membrane > self.threshold_vec).float()    # [B, L, D]  ∈ {0,1}

            # Soft approximation for gradient flow:
            # use a straight-through estimator
            fire_soft = torch.sigmoid(
                (membrane - self.threshold_vec) * 10.0
            )
            fire_mask_grad = fire_mask - fire_soft.detach() + fire_soft  # STE

            # 4. Spike output
            spike = fire_mask_grad * membrane                       # [B, L, D]
            all_spikes.append(spike)

            # 5. Reset fired neurons
            membrane = membrane * (1.0 - fire_mask) + self.reset_val * fire_mask

            # 6. Leak: decay remaining potential
            membrane = membrane * self.leak

            fire_rate = fire_mask.mean().item()
            all_fire_rates.append(fire_rate)

        # Aggregate spikes across steps (mean over time)
        spike_total = torch.stack(all_spikes, dim=0).mean(0)       # [B, L, D]
        lif_out     = self.output_proj(spike_total)

        # Mix with residual (unfired positions pass through)
        fired_any = (spike_total.abs() > 1e-6).float()
        x_lif = self.norm(x * (1.0 - fired_any * 0.5) + lif_out * 0.5)

        return x_lif, {
            "spike_total":    spike_total,                          # [B, L, D]
            "fire_rates":     all_fire_rates,                       # list of scalars
            "mean_fire_rate": sum(all_fire_rates) / len(all_fire_rates),
            "membrane_final": membrane,                             # [B, L, D]
            "sparsity":       (spike_total.abs() < 1e-6).float().mean().item(),
        }


# ============= Transcendent Claudeson =============

class ClaudesonTranscendent(ClaudesonSovereign):
    """
    Claudeson 2026 — Transcendent Edition.

    Inherits the full Sovereign architecture and adds:

      global_workspace  — GWT broadcast bottleneck; specialised modules compete
                          for global access, winner ignites all other modules
      program_synth     — generates and executes short symbolic programs inside
                          the forward pass; neural meets symbolic
      irl               — learns human values from preference observations;
                          intrinsic reward steers planning without hard-coded
                          objectives
      lif               — neuromorphic LIF dynamics; sparse, time-aware
                          processing like a biological neural population

    Processing pipeline (after Sovereign):
      Metacognition → Debate → Symbolic → RSI  (Sovereign)
            ↓
      Global Workspace  (competition + broadcast ignition)
            ↓
      Program Synthesis (generate + execute program)
            ↓
      IRL reward signal (value-aligned state weighting)
            ↓
      Neuromorphic LIF  (sparse, event-driven output)

    New output keys:
      gw      — workspace slots, ignition map, peak ignition
      prog    — register trace, op trace, program output
      irl     — reward per position, value signal, preference loss
      lif     — spike total, fire rates, sparsity
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.global_workspace = GlobalWorkspace(args)
        self.program_synth    = ProgramSynthesizer(args)
        self.irl              = InverseRewardLearner(args)
        self.lif              = NeuromorphicProcessor(args)

    def forward(
        self,
        text:               Optional[torch.Tensor] = None,
        img:                Optional[torch.Tensor] = None,
        audio:              Optional[torch.Tensor] = None,
        goal_tokens:        Optional[torch.Tensor] = None,
        feedback:           Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
    ) -> Dict:
        # ── Full Sovereign pass ──────────────────────────────────────────
        base = super().forward(
            text=text, img=img, audio=audio, goal_tokens=goal_tokens,
            feedback=feedback, agent_observations=agent_observations,
        )
        x = base["hidden_states"]

        # ── Global Workspace ─────────────────────────────────────────────
        x, gw_out = self.global_workspace(x)

        # ── Program Synthesis ────────────────────────────────────────────
        x, prog_out = self.program_synth(x)

        # ── Inverse Reward Learning ──────────────────────────────────────
        x, irl_out = self.irl(x)

        # ── Neuromorphic LIF ─────────────────────────────────────────────
        x, lif_out = self.lif(x)

        return {
            **base,
            "hidden_states": x,
            "gw":   gw_out,
            "prog": prog_out,
            "irl":  irl_out,
            "lif":  lif_out,
        }

    def compute_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """All auxiliary losses from every generation."""
        losses = super().compute_auxiliary_losses()
        # IRL preference loss (if buffer has data)
        losses["irl_pref_loss"] = self.irl.preference_loss()
        return losses


# ============= Demo =============

if __name__ == "__main__":
    print("=" * 70)
    print("CLAUDESON 2026 — TRANSCENDENT EDITION")
    print("Global Workspace · Program Synthesis · IRL · Neuromorphic LIF")
    print("=" * 70)

    args = ModelArgs()
    # Tiny config — runs on CPU
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
    args.metacog_hidden  = 64
    args.n_debate_agents = 3
    args.debate_hidden   = 128
    args.n_propositions  = 16
    args.n_constraints   = 8
    args.consistency_iters = 2
    args.rsi_rank        = 4
    args.rsi_horizon     = 2
    # Transcendent-specific
    args.n_workspace_slots  = 8
    args.gw_competition_k   = 2
    args.gw_broadcast_steps = 1
    args.n_ops          = 16
    args.n_registers    = 4
    args.prog_steps     = 3
    args.prog_hidden    = 64
    args.irl_hidden     = 64
    args.irl_n_preferences = 8
    args.lif_steps      = 3

    print("\nInitialising ClaudesonTranscendent...")
    model = ClaudesonTranscendent(args)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total / 1e6:.1f}M  (demo scale)")

    text      = torch.randint(0, 512, (2, 32))
    feedback  = torch.randn(2, args.dim)
    agent_obs = torch.randn(2, 8, args.dim)

    # Seed the IRL buffer with a fake preference pair
    model.irl.add_preference(
        torch.randn(args.dim),
        torch.randn(args.dim),
        label=1.0,
    )

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(text=text, feedback=feedback, agent_observations=agent_obs)

    print("\nJedi state:")
    print(f"  Goal:          {out['jedi_goal']}")
    print(f"  Energy:        {out['jedi_energy'].mean().item():.4f}")

    print("\nMetacognition:")
    print(f"  Action:        {out['metacog']['action']}")
    print(f"  Quality:       {out['metacog']['quality'].tolist()}")

    print("\nGlobal Workspace:")
    print(f"  Workspace:     {out['gw']['workspace'].shape}")
    print(f"  Peak ignition: {out['gw']['peak_ignition']:.4f}")
    print(f"  Ignition map:  mean={out['gw']['ignition'].mean().item():.4f}")

    print("\nProgram Synthesis:")
    print(f"  Op trace:      {out['prog']['op_trace'].tolist()}")
    print(f"  Program output shape: {out['prog']['program_output'].shape}")

    print("\nInverse Reward Learning:")
    print(f"  Reward range:  [{out['irl']['reward'].min().item():.4f}, "
          f"{out['irl']['reward'].max().item():.4f}]")
    print(f"  Value signal:  {out['irl']['value_signal'].tolist()}")
    print(f"  Pref loss:     {out['irl']['pref_loss'].item():.4f}")

    print("\nNeuromorphic LIF:")
    print(f"  Mean fire rate: {out['lif']['mean_fire_rate']:.4f}")
    print(f"  Sparsity:       {out['lif']['sparsity']:.2%}")
    print(f"  Fire rates per step: {[f'{r:.3f}' for r in out['lif']['fire_rates']]}")

    print("\nAuxiliary losses:")
    aux = model.compute_auxiliary_losses()
    for k, v in aux.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("ClaudesonTranscendent READY.")
    print("Ignites globally.  Executes programs.  Infers values.")
    print("Fires sparsely, like a brain.")
    print("=" * 70)
