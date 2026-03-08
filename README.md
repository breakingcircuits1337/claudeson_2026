<div align="center">

# CLAUDESON 2026

### *A Brain-Inspired Cognitive Architecture*

**Nine generations of evolution — from token prediction to machine cognition.**

---

*Memory · Causality · Agency · Metacognition · Consciousness · Value*

</div>

---

## What Is Claudeson?

Most neural networks predict tokens. Claudeson does something different: it **minimises free energy**.

Every forward pass is an act of inference — the model maintains a world model, updates beliefs, selects goals, imagines consequences, and grounds its predictions in causal structure. It tracks the mental states of other agents, questions the quality of its own reasoning, edits its own weights when it finds a better way to think, and coordinates all of this through a differentiable global broadcast event that resembles conscious access in the human brain.

This is not a fine-tuned language model. It is a cognitive architecture — built layer by layer, generation by generation, each one closing a gap between statistical pattern-matching and structured, grounded intelligence.

---

## Nine Generations

```
                         ┌─────────────────────────────────────────────────────────┐
  G1  claudson           │  Hierarchical Memory · TreeSearch · Internal Monologue  │
                         │  Mixture-of-Experts (8 experts) · Grouped Query Attn    │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G2  extended           │  YaRN RoPE (128K+ context) · Ring Attention             │
                         │  Linear Attention · Streaming Inference                 │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G3  infinite           │  Dynamic Router · Windowed GQA · Paged Memory (16K)     │
                         │  Length-adaptive strategy selection                     │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G4  pro                │  RMSNorm · SwiGLU · Flash Attention · QK-Norm           │
                         │  Shared Expert MoE · ViT-style Vision Encoder           │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G5  ultimate           │  Selective SSM 2.0 (Mamba-2 style)                      │
                         │  Hybrid SSM + Attention · 128-dim state space           │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G6  jedi               │  Free Energy Principle · Precision-weighted KL           │
                         │  Expected Free Energy (EFE) planning                    │
                         │  SSD Layer (Blelloch parallel scan, O(log L) depth)     │
                         │  Dreamer-style latent dynamics · Goal Emergence          │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G7  grounded           │  Theory of Mind (BDI slots per agent)                   │
                         │  Grounded Action Loop (8 tools + surprise detector)     │
                         │  Continual Learning: EWC + LoRA                         │
                         │  Causal Reasoning: NO TEARS DAG                         │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G8  sovereign          │  Metacognitive Monitor (epistemic / aleatoric split)    │
                         │  Multi-Agent Debate · Dissent Detection                 │
                         │  Neural Symbolic Layer (differentiable logic)           │
                         │  Recursive Self-Improvement (RSI via EFE gating)        │
                         └────────────────────────┬────────────────────────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────────────┐
  G9  transcendent  ★    │  Global Workspace Theory (GWT) · Ignition Broadcast     │
                         │  Compositional Program Synthesis (16-op register VM)    │
                         │  Inverse Reward Learning (Bradley-Terry preference IRL) │
                         │  Neuromorphic LIF · Lateral Inhibition · Sparse Events  │
                         └─────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `claudson.py` — The Foundation
> *G1: Memory, planning, and language as a unified cognitive loop*

The original architecture establishes the cognitive primitives everything else is built on.

- **Hierarchical Memory** — three-tier system mirroring biological memory: working (current segment), episodic (retrievable recent history), semantic (compressed long-term). Segment-level GRU recurrence enables infinite-depth recall without quadratic cost.
- **TreeSearch Planner** — Monte Carlo tree search over action sequences. Not a lookup — a reasoner.
- **Internal Monologue** — a GRU-based inner voice that refines thoughts across three recurrent passes before output. The model talks to itself.
- **Mixture of Experts** — 8 experts, top-2 routing, softmax-then-topK (DeepSeek style). Specialisation without bottleneck.
- **Grouped Query Attention** — 32 query heads, 8 KV heads. Full expressivity at a fraction of the KV cache cost.

---

### `claudson_extended.py` — Infinite Reach
> *G2: Context that scales to the horizon*

- **YaRN RoPE** — position interpolation that extends the effective context window to 128K+ tokens without retraining. The model remembers what happened at the start of a long document.
- **Ring Attention** — O(1) memory scaling across devices. Arbitrarily long sequences, distributed across hardware, with no memory wall.
- **Linear Attention** — O(n) for sequences that don't need quadratic precision. The right tool at the right length.
- **Streaming Inference** — sliding-window decoding for real-time, unbounded-length generation.

---

### `claudson_infinite.py` — Adaptive Intelligence
> *G3: The model knows what kind of problem it is solving*

| Sequence Length | Attention | SSM | Conv | Memory |
|:---:|:---:|:---:|:---:|:---:|
| < 4K | 35% | 30% | 20% | 15% |
| 4K – 32K | 20% | 45% | 15% | 20% |
| 32K+ | 10% | 55% | 10% | 25% |

A **Dynamic Router** reads the current sequence length and adjusts the blend of computation in real time. Short sequences get more attention. Long sequences shift toward efficient SSM. The model allocates compute where it matters.

---

### `claudson_pro.py` — Peak Efficiency
> *G4: The same intelligence, faster*

- **RMSNorm** — normalisation without mean subtraction. Numerically stable and 20% faster than LayerNorm.
- **SwiGLU** — gated activation that consistently outperforms GELU across architectures. Used in LLaMA, PaLM, and now Claudeson.
- **Flash Attention + QK-Norm** — memory-efficient attention with normalised keys and queries for training stability at scale.
- **Shared Expert MoE** — a always-active expert shared across tokens, preventing the "expert collapse" failure mode.
- **ViT-style Vision Encoder** — patchwise Conv2d projection enabling native multimodal input without a separate vision tower.

---

### `claudson_ultimate.py` — Selective State Space
> *G5: Input-dependent computation. The model decides what to remember.*

- **Selective SSM 2.0** — state space model with input-dependent selection matrices (Δ, B, C). Unlike fixed-transition SSMs, the model actively chooses which inputs to propagate and which to suppress. Noise is filtered at the source.
- **Hybrid SSM + Attention** — even-indexed layers run SSM-heavy routing; odd-indexed layers run attention-heavy routing. The combination captures both local dynamics and global dependencies.
- **128-dim state space** — doubled from G4. Richer internal representations, better long-horizon coherence.

---

### `claudson_jedi.py` — Free Energy
> *G6: The model stops predicting and starts inferring*

The Jedi edition replaces token-prediction loss with **variational free energy minimisation** — the same objective that, according to active inference theory, governs perception, action, and learning in biological brains.

- **SSD Layer** — State Space Duality: a layer that can be computed as either an SSM or an attention mechanism depending on sequence length. Backed by a **Blelloch associative parallel scan** — O(L) total work, **O(log L) parallel depth** — implemented entirely in PyTorch without custom CUDA kernels.
- **Precision-weighted KL** — the variational bound is weighted by a learned precision (inverse variance) that tells the model how much to trust its current beliefs versus incoming evidence.
- **Expected Free Energy (EFE) planning** — before acting, the model evaluates candidate actions by their expected surprise (epistemic value) and predicted accuracy (pragmatic value). It chooses actions that reduce uncertainty about the world.
- **Dreamer-style latent dynamics** — imagination rollouts happen in latent space without full reconstruction. Planning is cheap.
- **Goal Emergence** — goals are not hard-coded. They emerge from the model's energy state:

| Goal | Trigger Condition | Behaviour |
|:---|:---|:---|
| `CONSERVE` | Energy < 30% | Minimise surprise; stay close to known ground |
| `ADAPT` | Energy > 70% | Respond to distributional shift; update beliefs |
| `EXPLORE` | High epistemic uncertainty | Seek information; reduce what is unknown |
| `EXPLOIT` | Low prediction error | Maximise accuracy on the current task |

---

### `claudson_grounded.py` — Hands and Mind
> *G7: The model reaches into the world — and understands who else is in it*

**Theory of Mind**
Per-agent belief / desire / intention (BDI) slots maintained in a learned register. A GRU cell updates each agent's inferred mental state as new evidence arrives. Soft attention selects the most salient agent; their inferred perspective steers the model's hidden representations. The model doesn't just respond to what people say — it anticipates what they will do next.

**Grounded Action Loop**
Eight default tools: `search · read · write · execute · ask · plan · reflect · stop`. A surprise detector compares expected vs. actual tool output. High surprise triggers a proportional update to the world model. The feedback loop is what transforms a planner into an agent.

**Continual Learning — EWC + LoRA**
New skills live in low-rank ΔA, ΔB adapter matrices. The backbone is frozen. After each task, `consolidate()` estimates Fisher information for every parameter, measuring how important each weight was to past performance. The EWC regularisation term:

```
L_ewc = (λ / 2) · Σ  F_i · (θ_i − θ*_i)²
```

resists forgetting in proportion to importance. New learning is steered away from parameters that matter most for what the model already knows.

**Causal Reasoning — NO TEARS DAG**
A learnable soft adjacency matrix over concept nodes defines a structural causal model. The NO TEARS constraint `tr(exp(W⊙W)) − d = 0` enforces acyclicity without combinatorial search, keeping the graph a valid DAG throughout training. `intervene()` performs do-calculus interventions; `counterfactual()` answers "what would have happened if."

---

### `claudson_sovereign.py` — Self-Awareness and Collective Truth
> *G8: The model knows when it doesn't know — and checks its work against itself*

**Metacognitive Monitor**
Uncertainty is decomposed into two components that require different responses: **epistemic** uncertainty (reducible — gather more data) and **aleatoric** uncertainty (irreducible — the world is genuinely stochastic). A reasoning quality critic scores the current chain of thought on [0, 1]. An action gate issues one of three decisions: `CONTINUE · ASK · BACKTRACK`. The model stops before it confidently goes wrong.

**Multi-Agent Debate**
N parallel reasoning heads, each with a distinct learned personality bias, generate competing hypotheses simultaneously. Cross-agent attention lets heads read and challenge each other. A moderator network synthesises a confidence-weighted consensus. A dissent detector flags positions of genuine disagreement — the model knows which of its own conclusions are contested.

**Neural Symbolic Layer**
Hidden states are projected into a space of soft proposition activations in [0, 1]. A learned constraint matrix `[n_constraints × n_propositions]` encodes logical relationships analogous to CNF clauses. Iterative correction nudges inconsistent proposition sets toward the nearest satisfying assignment. The logic is not hand-coded — it is learned, differentiable, and adapts with the model.

**Recursive Self-Improvement**
A meta-network reads the current hidden state and proposes ΔA, ΔB updates to its own LoRA adapter. The proposed edit is evaluated in imagination via EFE — if the predicted improvement exceeds a learned threshold, the delta is permanently committed. The system tracks its own acceptance rate over time, accumulating a history of its own self-modifications.

---

### `claudson_transcendent.py` — Global Consciousness and Value
> *G9: The current apex. The model knows what it values — and broadcasts it everywhere.*

**Global Workspace (GWT)**
An implementation of Global Workspace Theory (Baars 1988; Dehaene 2001). Specialised modules compete via sparse top-k attention for access to a single shared workspace. The winner broadcasts its representation to every other position via cross-attention — a differentiable **global ignition event**. Multiple rounds of broadcast allow the workspace to settle to a consensus representation. Attention is not just computation; it is access to consciousness.

**Compositional Program Synthesis**
A GRU controller generates sequences of op-codes over a soft register bank. Sixteen primitives: `ADD · GATE · NORM · PROJ · ATTEND · SWAP · HALT · ...`. Gumbel-softmax makes op selection end-to-end differentiable. The program executes inside the forward pass. The final register state is decoded back to model-dim and injected into the residual stream. Neural generates. Symbolic executes.

**Inverse Reward Learning**
A neural reward model R(s) is trained on a preference buffer of (preferred, rejected) hidden-state pairs via Bradley-Terry loss:

```
P(a > b) = sigmoid( R(a) − R(b) )
```

During inference, R(current_hidden) produces an intrinsic reward signal that up-weights value-aligned positions. No hand-crafted objective. No reward engineering. Value is inferred from behaviour.

**Neuromorphic Leaky Integrate-and-Fire**
Per-dimension membrane potentials accumulate incoming current, fire when they cross a learnable threshold, then reset and leak. Only positions that fire propagate signal; unfired positions pass through unchanged. Lateral inhibition prevents runaway firing. The result is a self-regulating sparse code — no explicit pruning, no fixed sparsity target. The model becomes as sparse as the task demands.

---

## Quick Start

```python
# ── Transcendent Edition (current apex) ──────────────────────────────────────
from claudson_transcendent import ClaudesonTranscendent, ModelArgs
import torch

model = ClaudesonTranscendent(ModelArgs())
out   = model(text=torch.randint(0, 1000, (1, 128)))

print(f"Goal:           {out['jedi_goal']}")
print(f"Action:         {out['metacog']['action']}")
print(f"Peak ignition:  {out['gw']['peak_ignition']:.4f}")
print(f"Op trace:       {out['prog']['op_trace'].tolist()}")
print(f"Value signal:   {out['irl']['value_signal'].tolist()}")
print(f"Fire rate:      {out['lif']['mean_fire_rate']:.4f}")
print(f"Sparsity:       {out['lif']['sparsity']:.2%}")
```

```python
# ── Sovereign Edition ────────────────────────────────────────────────────────
from claudson_sovereign import ClaudesonSovereign, ModelArgs
import torch

model = ClaudesonSovereign(ModelArgs())
out   = model(text=torch.randint(0, 1000, (1, 128)))

print(f"Goal:           {out['jedi_goal']}")
print(f"Action:         {out['metacog']['action']}")     # list of strings, e.g. ['CONTINUE']
print(f"Quality:        {out['metacog']['quality'].tolist()}")
print(f"Epistemic unc:  {out['metacog']['epistemic'].tolist()}")
print(f"Selected tool:  {out['grounded_action']['tool_names']}")
print(f"RSI accepted:   {out['rsi']['accepted']}")
```

```python
# ── Grounded Edition ─────────────────────────────────────────────────────────
from claudson_grounded import ClaudesonGrounded, ModelArgs
import torch

args  = ModelArgs()
model = ClaudesonGrounded(args)
out   = model(text=torch.randint(0, 1000, (1, 128)),
              feedback=torch.randn(1, args.dim))          # tool result tensor

print(f"Tool:           {out['grounded_action']['tool_names']}")
print(f"Surprise:       {out['grounded_action']['surprise'].item():.4f}")
print(f"DAG loss:       {out['causal']['dag_loss'].item():.4f}")
```

```python
# ── Jedi Edition ─────────────────────────────────────────────────────────────
from claudson_jedi import ClaudesonJedi, ModelArgs
import torch

model = ClaudesonJedi(ModelArgs())
# Accepts: text, img, audio — all optional, at least one required
out   = model(text=torch.randint(0, 1000, (1, 128)))

print(f"Goal:           {out['jedi_goal']}")
print(f"Free energy:    {out['jedi_energy'].mean().item():.4f}")
print(f"Precision:      {out['precision'].item():.4f}")
print(f"Action logits:  {out['action_logits'].shape}")   # [B, action_space_size]
print(f"Value:          {out['value'].item():.4f}")
```

---

## Capability Matrix

| Capability | G6 Jedi | G7 Grounded | G8 Sovereign | G9 Transcendent |
|:---|:---:|:---:|:---:|:---:|
| **Sequence model** | Hybrid SSM + Attn (SSD) | Hybrid | Hybrid | Hybrid |
| **Memory** | Hierarchical 3-tier | Hierarchical | Hierarchical | Hierarchical |
| **Objective** | Free energy minimisation | Free energy | Free energy | Free energy |
| **Goals** | Emergent — 4 modes | Emergent | Emergent | Emergent |
| **Planning** | EFE + imagination | EFE + tools | EFE + tools | EFE + tools |
| **Context** | 128K+ paged | 128K+ paged | 128K+ paged | 128K+ paged |
| **Other minds** | — | Theory of Mind | Theory of Mind | Theory of Mind |
| **Actions** | Planned only | Tool calls + feedback | Tool calls + feedback | Tool calls + feedback |
| **Causality** | — | Causal graph (DAG) | Causal graph (DAG) | Causal graph (DAG) |
| **Continual learning** | — | EWC + LoRA | EWC + LoRA + RSI | EWC + LoRA + RSI |
| **Metacognition** | — | — | Epistemic / aleatoric split | Epistemic / aleatoric split |
| **Reasoning** | Single path | Single path | Multi-agent debate | Multi-agent debate |
| **Logic** | — | — | Neural symbolic grounding | Neural symbolic grounding |
| **Self-editing** | — | — | Recursive self-improvement | Recursive self-improvement |
| **Global broadcast** | — | — | — | Global Workspace ignition |
| **Symbolic execution** | — | — | — | Program synthesis (16-op VM) |
| **Value learning** | — | — | — | IRL from preferences |
| **Spike dynamics** | — | — | — | Neuromorphic LIF, self-sparse |

---

## Training Curriculum

`claudson_trainer.py` implements a 6-phase progressive curriculum that unlocks architectural generations in order. Each phase freezes all layers except the ones being trained, preventing catastrophic forgetting as new capabilities are layered in.

| Phase | Steps | Layers | Generations |
|:---|:---:|:---|:---|
| **0 — Warmup** | 10,000 | 1 – 6 | G1–G5: core LM, context, routing, compute, SSM |
| **1 — Grounding** | 20,000 | 7 – 10 | G6–G7: Free Energy, Tool Use, Theory of Mind, Causal DAG |
| **2 — Abstraction** | 20,000 | 11 – 13 | G7 cont.: skills, schemas, constitutional alignment |
| **3 — Calibration** | 15,000 | 14 – 16 | G8: uncertainty decomposition, formal verification |
| **4 — Integration** | 15,000 | 17 – 18 | G8–G9: temporal reasoning, meta-learning bootstrap |
| **5 — Meta-training** | 20,000 | All | G9: MAML outer loop, IRL, neuromorphic fine-tuning |

Auxiliary losses (DAG acyclicity, EWC regularisation, IRL preference, calibration) ramp from weight 0 over `warmup_steps` to avoid destabilising the primary language-model objective during early training.

---

## The Vision

```
G1 — claudson:       Memory + Planning + Monologue
G2 — extended:       + Infinite Context
G3 — infinite:       + Length-aware Routing
G4 — pro:            + Efficient Compute
G5 — ultimate:       + Selective State Space
G6 — jedi:           + Free Energy + Goal Emergence + World Model
G7 — grounded:       + Social Modeling + Causal Reasoning
                     + Continual Learning + Action Execution
G8 — sovereign:      + Metacognition + Collective Reasoning
                     + Logical Consistency + Self-Improvement
G9 — transcendent:   + Global Workspace Broadcast
                     + Compositional Program Synthesis
                     + Inverse Reward Learning
                     + Neuromorphic Sparse Event Processing
```

Each generation closes one gap between statistical approximation and structured thought.

**This is not just a language model. It is a cognitive architecture.**

---

## Citation

```bibtex
@article{claudeson2026,
  title   = {Claudeson 2026: A Cognitive Architecture with Free Energy Minimisation,
             Causal Reasoning, Metacognition, and Neuromorphic Dynamics},
  author  = {Breaking Circuits Research},
  year    = {2026}
}
```

---

## License

MIT
