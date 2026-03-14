# SPDX-License-Identifier: LicenseRef-BreakingCircuits-Commercial
# Copyright (c) 2026 Breaking Circuits Research. All rights reserved.

"""
claudson_tracer.py — Reasoning Trace Extractor
===============================================

Provides an "MRI-like" view into a ClaudesonTranscendent (G9) forward pass
by capturing and formatting the key interpretability signals:

  1. GWT Ignition Map    — which token positions won the Global Workspace
                          broadcast bottleneck at each competition step
  2. Program Op Trace    — the symbolic operation sequence executed inside
                          ProgramSynthesizer
  3. Metacognitive State — quality score, uncertainty decomposition, and the
                          CONTINUE/ASK/BACKTRACK action decision
  4. Debate Dissent      — positions where the multi-agent debate was contested
  5. Logical Propositions — top active propositions at each position

Usage
-----
    from claudson_tracer import ReasoningTracer

    tracer = ReasoningTracer(model, top_k_tokens=8)
    out    = tracer.trace(text=tokens)

    print(tracer.summary(out))          # human-readable trace
    df     = tracer.to_dataframe(out)   # per-token DataFrame for analysis
    panels = tracer.grafana_payload(out) # dict ready for a Grafana annotations API call

The tracer is read-only and adds no overhead beyond the normal forward pass —
it just re-formats the signals that are already in the output dict.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# Op-code names match ProgramSynthesizer
OP_NAMES = [
    "NOP",
    "ADD",
    "GATE",
    "NORM",
    "PROJ",
    "RESIDUAL",
    "ATTEND",
    "NEGATE",
    "SCALE",
    "SWAP",
    "MAX",
    "MIN",
    "RELU",
    "TANH",
    "OUTER",
    "HALT",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenTrace:
    """Interpretability trace for a single token position."""

    position: int
    ignition_score: float  # how strongly this token won GWT broadcast
    won_broadcast: bool  # above median ignition threshold
    reward: float  # IRL inferred reward signal
    dissent: float  # cross-agent disagreement at this position
    contested: bool  # dissent exceeded threshold
    top_props: List[int]  # indices of top-3 active propositions
    prop_values: List[float]  # activation values of those propositions
    fire_rate: float  # LIF neuromorphic fire rate (sparsity proxy)


@dataclass
class ReasoningTrace:
    """Full trace for one forward pass."""

    # Metacognition
    metacog_action: List[str]  # ["CONTINUE"|"ASK"|"BACKTRACK", ...] per batch item
    metacog_quality: List[float]
    metacog_epistemic: List[float]
    metacog_aleatoric: List[float]

    # Global Workspace bottleneck
    peak_ignition: float
    ignition_map: torch.Tensor  # [B, L]  per-token ignition scores
    workspace_slots: torch.Tensor  # [B, n_slots, D]

    # Program execution
    op_trace: torch.Tensor  # [B, prog_steps]  hard op indices
    op_names: List[List[str]]  # human-readable per batch item

    # Per-token detail (batch item 0 by default)
    token_traces: List[TokenTrace]

    # IRL value signal
    value_signal: List[float]  # [B]

    # LIF sparsity
    mean_fire_rate: float
    sparsity: float

    # RSI self-edit decision
    rsi_accepted: bool
    rsi_acceptance_rate: float


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class ReasoningTracer:
    """
    Wraps a ClaudesonTranscendent model and extracts a structured
    ReasoningTrace from every forward pass.

    Parameters
    ----------
    model       : ClaudesonTranscendent instance
    top_k_tokens: number of "most ignited" tokens to highlight in summary
    """

    def __init__(self, model, top_k_tokens: int = 8):
        self.model = model
        self.top_k_tokens = top_k_tokens

    # ------------------------------------------------------------------
    # Core trace method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def trace(
        self,
        text: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        goal_tokens: Optional[torch.Tensor] = None,
        feedback: Optional[torch.Tensor] = None,
        agent_observations: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
    ) -> ReasoningTrace:
        """
        Run the model and extract the full reasoning trace.

        batch_idx selects which batch item to build per-token traces for.
        Returns a ReasoningTrace dataclass.
        """
        out = self.model(
            text=text,
            img=img,
            audio=audio,
            goal_tokens=goal_tokens,
            feedback=feedback,
            agent_observations=agent_observations,
        )
        return self._extract(out, batch_idx=batch_idx)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract(self, out: Dict, batch_idx: int = 0) -> ReasoningTrace:
        gw = out["gw"]
        prog = out["prog"]
        metacog = out["metacog"]
        irl = out["irl"]
        lif = out["lif"]
        debate = out["debate"]
        sym = out["symbolic"]

        B = gw["ignition"].shape[0]
        L = gw["ignition"].shape[1]
        bi = min(batch_idx, B - 1)

        # --- Metacognition ---
        actions = metacog["action"]  # list[str], len B
        quality = metacog["quality"].tolist()
        epistemic = metacog["epistemic"].tolist()
        aleatoric = metacog["aleatoric"].tolist()

        # --- GWT ignition ---
        ignition_map = gw["ignition"]  # [B, L]
        ign_b = ignition_map[bi]  # [L]
        ign_threshold = ign_b.median().item()

        # --- Program trace ---
        op_trace_idx = prog["op_trace"]  # [B, steps]
        op_names_all = [[OP_NAMES[i] for i in row.tolist()] for row in op_trace_idx]

        # --- IRL reward per position ---
        reward_b = irl["reward"][bi]  # [L]
        value_signal = irl["value_signal"].tolist()

        # --- Debate dissent ---
        dissent_b = debate["dissent"][bi]  # [L]
        contested_b = debate["contested"][bi]  # [L]

        # --- Propositions ---
        props_b = sym["propositions"][bi]  # [L, n_props]

        # --- LIF ---
        lif["fire_rates"]
        mean_fr = lif["mean_fire_rate"]
        sparsity = lif["sparsity"]

        # --- RSI ---
        rsi = out.get("rsi", {})
        rsi_acc = rsi.get("accepted", False)
        rsi_rate = rsi.get("acceptance_rate", 0.0)

        # --- Per-token traces (for batch_idx item) ---
        token_traces = []
        for pos in range(L):
            top_p = props_b[pos].topk(min(3, props_b.shape[-1]))
            token_traces.append(
                TokenTrace(
                    position=pos,
                    ignition_score=ign_b[pos].item(),
                    won_broadcast=ign_b[pos].item() > ign_threshold,
                    reward=reward_b[pos].item(),
                    dissent=dissent_b[pos].item(),
                    contested=bool(contested_b[pos].item()),
                    top_props=top_p.indices.tolist(),
                    prop_values=top_p.values.tolist(),
                    fire_rate=mean_fr,
                )
            )

        return ReasoningTrace(
            metacog_action=actions,
            metacog_quality=quality,
            metacog_epistemic=epistemic,
            metacog_aleatoric=aleatoric,
            peak_ignition=gw["peak_ignition"],
            ignition_map=ignition_map,
            workspace_slots=gw["workspace"],
            op_trace=op_trace_idx,
            op_names=op_names_all,
            token_traces=token_traces,
            value_signal=value_signal,
            mean_fire_rate=mean_fr,
            sparsity=sparsity,
            rsi_accepted=rsi_acc,
            rsi_acceptance_rate=rsi_rate,
        )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self, trace: ReasoningTrace, batch_idx: int = 0) -> str:
        lines = []
        sep = "─" * 60

        lines.append("╔══════════════════════════════════════════════════════════╗")
        lines.append("║       CLAUDESON REASONING TRACE — MRI VIEW               ║")
        lines.append("╚══════════════════════════════════════════════════════════╝")

        # Metacognition block
        lines.append(f"\n{sep}")
        lines.append("  METACOGNITIVE STATE")
        lines.append(sep)
        bi = min(batch_idx, len(trace.metacog_action) - 1)
        action = trace.metacog_action[bi]
        quality = trace.metacog_quality[bi]
        ep = trace.metacog_epistemic[bi]
        al = trace.metacog_aleatoric[bi]

        action_symbol = {"CONTINUE": "▶", "ASK": "?", "BACKTRACK": "↩"}.get(action, "·")
        lines.append(f"  Decision:          {action_symbol}  {action}")
        lines.append(f"  Reasoning quality: {quality:.3f}  {'█' * int(quality * 20)}")
        lines.append(f"  Epistemic (↓more data helps):  {ep:.4f}")
        lines.append(f"  Aleatoric (↓irreducible noise): {al:.4f}")

        # GWT bottleneck block
        lines.append(f"\n{sep}")
        lines.append("  GLOBAL WORKSPACE BOTTLENECK")
        lines.append(sep)
        lines.append(f"  Peak ignition score: {trace.peak_ignition:.4f}")

        # Top-k ignited tokens
        ign = trace.ignition_map[bi]
        topk_vals, topk_idx = ign.topk(min(self.top_k_tokens, len(ign)))
        lines.append(f"  Top-{self.top_k_tokens} broadcast winners (position → ignition):")
        for rank, (pos, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist()), 1):
            tt = trace.token_traces[pos]
            contested_tag = " [CONTESTED]" if tt.contested else ""
            lines.append(
                f"    #{rank:2d}  pos={pos:4d}  ignition={val:.4f}"
                f"  reward={tt.reward:+.3f}{contested_tag}"
            )

        # Ignition bar chart (compressed to 40 chars)
        bar_width = 40
        ign_norm = (ign - ign.min()) / (ign.max() - ign.min() + 1e-8)
        step = max(1, len(ign_norm) // bar_width)
        bar = "".join(
            "█"
            if ign_norm[i * step].item() > 0.6
            else "▓"
            if ign_norm[i * step].item() > 0.3
            else "░"
            for i in range(min(bar_width, len(ign_norm) // step))
        )
        lines.append(f"\n  Ignition map (seq→): [{bar}]")
        lines.append("  (█=high broadcast  ▓=medium  ░=low/suppressed)")

        # Program synthesis block
        lines.append(f"\n{sep}")
        lines.append("  PROGRAM SYNTHESIS TRACE")
        lines.append(sep)
        ops = trace.op_names[bi]
        lines.append("  Executed ops:  " + "  →  ".join(ops))

        # IRL value
        lines.append(f"\n{sep}")
        lines.append("  INVERSE REWARD LEARNING")
        lines.append(sep)
        lines.append(f"  Inferred value signal: {trace.value_signal[bi]:+.4f}")
        reward_vals = [tt.reward for tt in trace.token_traces]
        lines.append(f"  Reward range:  [{min(reward_vals):+.3f}, {max(reward_vals):+.3f}]")

        # LIF sparsity
        lines.append(f"\n{sep}")
        lines.append("  NEUROMORPHIC LIF")
        lines.append(sep)
        lines.append(f"  Mean fire rate:  {trace.mean_fire_rate:.4f}")
        lines.append(f"  Sparsity:        {trace.sparsity:.2%}  (higher = sparser = cheaper)")

        # RSI
        lines.append(f"\n{sep}")
        lines.append("  RECURSIVE SELF-IMPROVEMENT")
        lines.append(sep)
        edit_tag = "✓ COMMITTED" if trace.rsi_accepted else "✗ discarded"
        lines.append(f"  Self-edit this step:  {edit_tag}")
        lines.append(f"  Historical accept rate: {trace.rsi_acceptance_rate:.2%}")

        lines.append(f"\n{sep}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-token DataFrame (requires pandas)
    # ------------------------------------------------------------------

    def to_dataframe(self, trace: ReasoningTrace):
        """
        Convert per-token trace to a pandas DataFrame.
        Columns: position, ignition_score, won_broadcast, reward,
                 dissent, contested, fire_rate, top_prop_0, top_prop_1, top_prop_2
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). pip install pandas")

        rows = []
        for tt in trace.token_traces:
            row = {
                "position": tt.position,
                "ignition_score": tt.ignition_score,
                "won_broadcast": tt.won_broadcast,
                "reward": tt.reward,
                "dissent": tt.dissent,
                "contested": tt.contested,
                "fire_rate": tt.fire_rate,
            }
            for i, (p, v) in enumerate(zip(tt.top_props, tt.prop_values)):
                row[f"top_prop_{i}"] = p
                row[f"prop_val_{i}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Grafana annotations payload
    # ------------------------------------------------------------------

    def grafana_payload(
        self,
        trace: ReasoningTrace,
        dashboard_uid: str = "claudeson-gcp-001",
        batch_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of Grafana annotation dicts (one per contested token
        and one summary annotation), suitable for POST to:
            /api/annotations

        Each dict follows the Grafana HTTP API annotation schema.
        """
        import time

        now_ms = int(time.time() * 1000)
        bi = min(batch_idx, len(trace.metacog_action) - 1)
        annotations = []

        # Summary annotation
        annotations.append(
            {
                "dashboardUID": dashboard_uid,
                "time": now_ms,
                "tags": ["claudeson", "reasoning", trace.metacog_action[bi]],
                "text": (
                    f"<b>Metacog:</b> {trace.metacog_action[bi]}  "
                    f"quality={trace.metacog_quality[bi]:.3f}  "
                    f"peak_ignition={trace.peak_ignition:.4f}<br>"
                    f"<b>Ops:</b> {' → '.join(trace.op_names[bi])}<br>"
                    f"<b>LIF sparsity:</b> {trace.sparsity:.2%}  "
                    f"RSI: {'COMMITTED' if trace.rsi_accepted else 'discarded'}"
                ),
            }
        )

        # One annotation per contested token
        for tt in trace.token_traces:
            if tt.contested:
                annotations.append(
                    {
                        "dashboardUID": dashboard_uid,
                        "time": now_ms,
                        "tags": ["claudeson", "contested", "debate"],
                        "text": (
                            f"<b>Contested token pos={tt.position}</b>  "
                            f"dissent={tt.dissent:.4f}  "
                            f"ignition={tt.ignition_score:.4f}  "
                            f"reward={tt.reward:+.3f}"
                        ),
                    }
                )

        return annotations


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CLAUDESON REASONING TRACER — DEMO")
    print("=" * 60)

    try:
        from claudson_transcendent import ClaudesonTranscendent, ModelArgs
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

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

    print("\nLoading ClaudesonTranscendent (tiny demo config)...")
    model = ClaudesonTranscendent(args)
    tracer = ReasoningTracer(model, top_k_tokens=5)

    text = torch.randint(0, 512, (2, 32))
    feedback = torch.randn(2, args.dim)
    agent_obs = torch.randn(2, 8, args.dim)

    print("Running traced forward pass...")
    trace = tracer.trace(text=text, feedback=feedback, agent_observations=agent_obs)

    print(tracer.summary(trace))

    payload = tracer.grafana_payload(trace)
    print(f"Grafana annotations ready: {len(payload)} events")
    print(json.dumps(payload[0], indent=2))
