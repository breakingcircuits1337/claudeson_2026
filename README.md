# Claudeson 2026
## The Next Generation Cognitive Architecture

A brain-inspired cognitive architecture that has evolved through eight generations —
from a hybrid SSM/attention base to a fully grounded, socially-aware, causally-reasoning,
self-improving agent with metacognitive monitoring and logical consistency guarantees.

---

## Architecture Evolution

```
claudson.py          Hierarchical Memory · TreeSearch · Internal Monologue · MoE · GQA
      ↓
claudson_extended.py YaRN RoPE · Ring Attention · Linear Attention · 128K context
      ↓
claudson_infinite.py Dynamic Router · Windowed GQA · Paged Memory (16K slots)
      ↓
claudson_pro.py      RMSNorm · SwiGLU · Flash Attention · QK-Norm · Shared Expert MoE
      ↓
claudson_ultimate.py Selective SSM 2.0 (Mamba-2) · Hybrid SSM+Attention layers
      ↓
claudson_jedi.py     Free Energy Principle · Precision-weighted KL · EFE planning
                     Dreamer-style latent dynamics · Goal Emergence · SSD layer
      ↓
claudson_grounded.py Theory of Mind · Grounded Action Loop · Continual Learning (EWC+LoRA)
                     Causal Reasoning (NO TEARS DAG)
      ↓
claudson_sovereign.py Metacognitive Monitor · Multi-Agent Debate · Neural Symbolic Layer
                      Recursive Self-Improvement        ← current apex
```

---

## File Guide

### claudson.py
Original implementation:
- Hierarchical Memory (working → episodic → semantic)
- TreeSearch Planner
- Internal Monologue (GRU-based inner voice)
- MoE (8 experts, top-2 routing)
- Grouped Query Attention

### claudson_extended.py
Extended context capabilities:
- **YaRN RoPE** — extends context to 128K+ via interpolation
- **Ring Attention** — O(1) memory scaling across devices
- **Linear Attention** — O(n) for long sequences
- **Streaming Inference** — sliding window

### claudson_infinite.py
Infinite context with dynamic routing:
- **Dynamic Router** — auto-adjusts strategy by sequence length
- **Windowed GQA** — sliding window attention
- **Paged Memory** — 16K episodic slots

| Length | Attention | SSM | Conv | Memory |
|--------|-----------|-----|------|--------|
| < 4K   | 35% | 30% | 20% | 15% |
| 4K–32K | 20% | 45% | 15% | 20% |
| 32K+   | 10% | 55% | 10% | 25% |

### claudson_pro.py
Performance optimisations:
- **RMSNorm** — faster than LayerNorm
- **SwiGLU** — better activation than GELU
- **Flash Attention** with QK-Norm
- **Parallel SSM** — chunked computation
- **Shared Expert MoE** — more efficient routing
- **ViT-style Vision Encoder**

### claudson_ultimate.py
Architecture foundations for Mamba-2 style processing:
- **Selective SSM 2.0** — input-dependent selection; ignores noise
- **Hybrid SSM + Attention** — alternating layers per depth
- 128 state dimensions (was 64)

### claudson_jedi.py
Free Energy Principle integration:
- **SSD Layer** — State Space Duality; bridges SSM and attention
- **Precision-weighted KL** — proper variational inference
- **Expected Free Energy (EFE)** — active inference for planning
- **Dreamer-style latent dynamics** — imagination rollouts without full reconstruction
- **Goal Emergence** from energy state:

| Goal | Trigger | Purpose |
|------|---------|---------|
| CONSERVE | Energy < 30% | Preserve energy, reduce surprise |
| ADAPT | Energy > 70% | Respond to distributional shift |
| EXPLORE | High uncertainty | Reduce epistemic uncertainty |
| EXPLOIT | Low error | Maximise prediction accuracy |

### claudson_grounded.py
Four gaps filled after Jedi — the model gains hands, social awareness, memory, and causality:

**Theory of Mind**
Per-agent belief / desire / intention slots updated via GRU cells.
Soft attention selects the most relevant agent; their inferred perspective
steers the hidden representations.  An action predictor outputs what Claudeson
expects the agent to do next — enabling anticipation rather than reaction.

**Grounded Action Loop**
8 default tools: `search / read / write / execute / ask / plan / reflect / stop`.
A surprise detector compares expected vs. actual outcome; high surprise → large
world-model update.  The feedback loop is what turns a planner into an agent.

**Continual Learning (EWC + LoRA)**
New skills live in low-rank ΔA, ΔB matrices — the backbone never changes.
After finishing a task, `consolidate()` estimates Fisher information (parameter
importance).  The EWC loss `λ/2 · Σ F_i · (θ_i − θ*_i)²` resists future drift
on high-importance weights.

**Causal Reasoning**
Learnable `[n_nodes × n_nodes]` soft adjacency matrix in concept space.
NO TEARS acyclicity constraint (`tr(exp(W⊙W)) − d → 0`) keeps it a DAG.
`intervene()` and `counterfactual()` support do-calculus "what if" reasoning.

### claudson_sovereign.py ⭐ Current Apex
Four more gaps — the model gains self-awareness, collective reasoning, logical grounding, and self-editing:

**Metacognitive Monitor**
Decomposes uncertainty into epistemic (reducible) and aleatoric (irreducible) components.
A reasoning quality critic scores the current chain of thought on [0, 1].
An action gate emits one of three decisions: `CONTINUE / ASK / BACKTRACK`.
Prevents the confident-but-wrong failure mode.

**Multi-Agent Debate**
N parallel reasoning heads (each with a distinct learned personality bias) produce
competing hypotheses.  Cross-agent attention lets heads "hear" each other.
A moderator synthesises a confidence-weighted final position.
A dissent detector flags tokens where agents strongly disagree — these are the
areas the system should not act on without more information.

**Neural Symbolic Layer**
Maps hidden states to soft proposition activations in [0, 1].
A learned `[n_constraints × n_propositions]` matrix encodes logical constraints
(analogous to CNF clauses).  Iterative correction nudges inconsistent propositions
toward the nearest consistent assignment.  Fully differentiable — the logic is
learned, not hand-coded.

**Recursive Self-Improvement**
A meta-network reads the current hidden state and proposes ΔA, ΔB delta matrices
for its own LoRA adapter.  The proposed edit is evaluated in imagination (EFE).
If the predicted improvement exceeds the threshold, the delta is committed to the
adapter weights permanently.  Tracks acceptance rate over time.

---

## Quick Start

```python
# Latest: Sovereign Edition
from claudson_sovereign import ClaudesonSovereign, ModelArgs
import torch

args  = ModelArgs()
model = ClaudesonSovereign(args)

text = torch.randint(0, 1000, (1, 128))
out  = model(text=text)

# Reasoning state
print(f"Goal:           {out['jedi_goal']}")
print(f"Action:         {out['metacog']['action']}")
print(f"Quality:        {out['metacog']['quality'].item():.3f}")
print(f"Epistemic unc:  {out['metacog']['epistemic'].item():.4f}")
print(f"Selected tool:  {out['grounded_action']['tool_names']}")
print(f"RSI accepted:   {out['rsi']['accepted']}")
```

```python
# Previous stable: Grounded Edition
from claudson_grounded import ClaudesonGrounded, ModelArgs
import torch

args  = ModelArgs()
model = ClaudesonGrounded(args)

text     = torch.randint(0, 1000, (1, 128))
feedback = torch.randn(1, args.dim)          # tool result
out      = model(text=text, feedback=feedback)

print(f"Tool:      {out['grounded_action']['tool_names']}")
print(f"Surprise:  {out['grounded_action']['surprise'].item():.4f}")
print(f"DAG loss:  {out['causal']['dag_loss'].item():.4f}")
```

---

## Capability Comparison

| Capability | Traditional LLM | Jedi | Grounded | Sovereign |
|---|---|---|---|---|
| **Objective** | Token prediction | Energy minimization | Energy minimization | Energy minimization |
| **Goals** | Hard-coded | Emergent (4 modes) | Emergent | Emergent |
| **Memory** | Context window | Hierarchical (3 levels) | Hierarchical | Hierarchical |
| **Planning** | None | EFE + imagination | EFE + tool execution | EFE + tool execution |
| **Context** | 8K–128K | 128K+ paged | 128K+ paged | 128K+ paged |
| **Sequence model** | Transformer | Hybrid SSM+Attn (SSD) | Hybrid | Hybrid |
| **Self-model** | None | Energy + self-model | Energy + self-model | Energy + quality critic |
| **Other minds** | None | None | Theory of Mind | Theory of Mind |
| **Actions** | None | Planned only | Tool calls + feedback | Tool calls + feedback |
| **Learning** | Frozen after training | Frozen | EWC + LoRA continual | EWC + LoRA + RSI |
| **Causality** | Correlation | Correlation | Causal graph (DAG) | Causal graph (DAG) |
| **Metacognition** | None | None | None | Uncertainty + quality + action gate |
| **Reasoning** | Single path | Single path | Single path | Multi-agent debate |
| **Logic** | Probabilistic | Probabilistic | Probabilistic | Neural symbolic grounding |
| **Self-editing** | None | None | None | Recursive self-improvement |

---

## The Vision

Each generation adds one layer of the equation:

```
G1 — claudson:    Memory + Planning + Monologue
G2 — extended:    + Infinite Context
G3 — infinite:    + Length-aware Routing
G4 — pro:         + Efficient Compute
G5 — ultimate:    + Selective State Space
G6 — jedi:        + Free Energy + Goal Emergence + World Model
G7 — grounded:    + Social Modeling + Causal Reasoning
                  + Continual Learning + Action Execution
G8 — sovereign:   + Metacognition + Collective Reasoning
                  + Logical Consistency + Self-Improvement
```

**This is not just a language model.  It is a cognitive architecture.**

---

## Citation

```bibtex
@article{claudeson2026,
  title   = {Claudeson 2026: A Cognitive Architecture with
             Energy Minimization, Causal Reasoning, and Metacognition},
  author  = {Breaking Circuits Research},
  year    = {2026}
}
```

---

## License

MIT License
