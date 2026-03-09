# CLAUDE.md — Claudeson 2026 Codebase Guide

This file provides context for AI assistants working in this repository.

---

## Project Overview

**Claudeson 2026** is a PyTorch-based cognitive architecture research project implementing nine generations of brain-inspired neural network design. The architecture minimizes free energy rather than performing pure token prediction, and implements structured, grounded intelligence through hierarchical memory, causal reasoning, Theory of Mind, metacognitive monitoring, and Global Workspace Theory.

**Language:** Python 3
**Core dependency:** PyTorch (assumed pre-installed)
**Optional dependencies:** `wandb` (experiment tracking), `torch.utils.tensorboard` (visualization)
**Licensing:** Dual — AGPL-3.0 for Generations 1–5, Proprietary Commercial for Generations 6–9

---

## Repository Structure

All source files live at the root level (flat structure).

### Generations 1–5 (Open Core, AGPL-3.0)

| File | Generation | Key capability |
|------|-----------|---------------|
| `claudson.py` | G1 | Foundation — Hierarchical Memory, TreeSearch, MoE, GQA |
| `claudson_extended.py` | G2 | Infinite context — YaRN RoPE, Ring Attention, Streaming |
| `claudson_infinite.py` | G3 | Length-adaptive routing — Dynamic Router, paged memory |
| `claudson_pro.py` | G4 | Peak efficiency — RMSNorm, SwiGLU, Flash Attention, Vision |
| `claudson_ultimate.py` | G5 | Selective SSM 2.0 — Input-dependent computation |

### Generations 6–9 (Commercial License)

| File | Generation | Key capability |
|------|-----------|---------------|
| `claudson_jedi.py` | G6 | Free Energy Principle — EFE planning, goal emergence |
| `claudson_grounded.py` | G7 | Tools & Theory of Mind — Tool use, causal DAG, EWC |
| `claudson_sovereign.py` | G8 | Metacognition — Multi-agent debate, RSI, logic |
| `claudson_transcendent.py` | G9 | Global Workspace — GWT, program synthesis, IRL, LIF spikes |

### Companion / Variant Modules

| File | Purpose |
|------|---------|
| `claudson_abstraction.py` | Skill abstraction and schema learning |
| `claudson_causal_world.py` | Pearl-style causal world models |
| `claudson_formal_verification.py` | Formal verification integration |
| `claudson_grounded_language.py` | Grounded language understanding |
| `claudson_meta_learning.py` | MAML-based meta-learning |
| `claudson_metacurriculum.py` | Curriculum scheduling |
| `claudson_social_alignment.py` | Social norms and alignment |
| `claudson_temporal_reasoning.py` | Temporal reasoning and planning |
| `claudson_uncertainty.py` | Uncertainty estimation and calibration |
| `claudson_utils.py` | Shared utility: `RMSNorm` class |
| `claudson_trainer.py` | Unified 6-phase curriculum trainer |

### Tests

| File | Coverage |
|------|---------|
| `test_claudson_jedi.py` | RoPE mathematical properties (shape, norms, dot products) |
| `test_new_modules.py` | Smoke tests for module imports, forward passes, parallel scan correctness |

### Documentation

| File | Contents |
|------|---------|
| `README.md` | Full project overview, quick-start examples, capability matrix |
| `REVIEW_REPORT.md` | Code review findings and applied fixes |
| `LICENSE` | AGPL-3.0 (Generations 1–5) |
| `COMMERCIAL_LICENSE` | Proprietary terms (Generations 6–9) |

---

## Running Tests

```bash
pytest test_claudson_jedi.py -v
pytest test_new_modules.py -v
# Run a single parametrized test:
pytest test_new_modules.py::test_parallel_scan_matches_sequential -v
```

No build step is required — all modules are imported directly.

---

## Development Workflow

- **Branch naming:** `claude/<description>-<session-id>` (enforced by the git remote)
- **Commits:** Descriptive, present-tense messages (e.g., "Fix gradient checkpointing default")
- **PRs:** All changes merged via pull requests; 13 PRs merged as of March 2026
- **No CI/CD configuration** — tests are run manually before merging

### .gitignore

```
__pycache__/
*.pyc
*.pyo
*.pyd
checkpoints/
```

---

## Architecture Conventions

### 1. Configuration via Dataclasses

Every generation defines a `ModelArgs` dataclass at the top of its module. Later generations subclass the parent's `ModelArgs`:

```python
@dataclass
class ModelArgs(JediModelArgs):   # G7 inherits G6 config
    extra_param: int = 64
```

Key `ModelArgs` fields (G1 defaults):
- `dim=2048`, `n_layers=32`, `n_heads=32`, `n_kv_heads=8` (GQA)
- `vocab_size=128000`, `max_seq_len=8192`
- `num_experts=8`, `expert_top_k=2` (MoE)
- `gradient_checkpointing=False` (disabled by default — important, see REVIEW_REPORT.md)
- `mixed_precision=True`, `use_flash_attention=True`, `use_kv_cache=True`
- `constitutional_weight=0.01`, `constitutional_steer_scale=0.1` (alignment steering)

### 2. Module Structure Pattern

Each generation module follows this layout:
1. License/copyright header (SPDX identifier)
2. Imports (including parent generation imports)
3. `ModelArgs` dataclass (extends parent)
4. Utility/component classes (attention variants, memory, etc.)
5. Main model class inheriting from parent generation
6. `forward()` returning a `Dict` with feature-specific keys

### 3. Forward Pass Return Format

All models return a `Dict`. Key names are prefixed by generation:

```python
# G6 (Jedi):
{'jedi_goal': ..., 'jedi_energy': ..., 'precision': ..., 'value': ..., 'action_logits': ...}

# G7 (Grounded):
{'grounded_action': ..., 'causal_dag': ..., ...}

# G8 (Sovereign):
{'metacog': ..., ...}

# G9 (Transcendent):
{'gw': ..., 'prog': ..., 'irl': ..., 'lif': ..., ...}
```

### 4. Memory System

Three-tier hierarchy used across generations:
- **Working memory** — active context window (GRU-based segment recurrence)
- **Episodic memory** — recent experiences (`episodic_slots=2560`)
- **Semantic memory** — long-term compressed representations (`memory_slots=256`)

Paged memory: 16K active window + `position_offset` tracking for unbounded context.

### 5. Loss Computation

- Auxiliary losses are **ramped up from weight 0** over `warmup_steps` to avoid early destabilization
- Free energy minimization is the primary objective from G6 onward
- Expected Free Energy (EFE) is used for planning in G6+

---

## Trainer (claudson_trainer.py)

The unified trainer orchestrates a 6-phase curriculum:

| Phase | Layers | Focus |
|-------|--------|-------|
| 0 | 1–6 | Warmup — LM + world model |
| 1 | 7–10 | Grounding — ToM, causal, Pearl |
| 2 | 11–13 | Abstraction — skills, schemas, alignment |
| 3 | 14–16 | Calibration — uncertainty, verification |
| 4 | 17–18 | Integration — temporal + meta-learning |
| 5 | all | Meta-training — outer loop MAML |

**Key design decisions:**
- **Phase-gated training:** Only the current phase's layers are unfrozen; earlier layers remain frozen to prevent catastrophic forgetting
- **Gradient accumulation:** `grad_accum=8` micro-batches per optimizer step (effective batch = `batch_size × grad_accum`)
- **Layer-wise LR decay (LLRD):** Lower layers get smaller learning rates than upper layers
- **Checkpointing:** Saves `model`, `optimizer`, `scheduler`, and trainer metadata every `SAVE_EVERY` steps; auto-resumes if checkpoint exists in `CHECKPOINT_DIR`

### TrainerConfig defaults

```python
batch_size = 8
grad_accum = 8
max_lr = 3e-4
min_lr = 3e-5
mixed_precision = True  # BF16 preferred, FP16 fallback
```

---

## Distributed Training

The trainer auto-detects multi-GPU environments:

```bash
export RANK=0
export WORLD_SIZE=4
export LOCAL_RANK=0
```

The model is wrapped in `DistributedDataParallel` automatically. Checkpointing and logging only occur on `rank == 0`.

---

## Optional Experiment Tracking

```bash
export WANDB_PROJECT=claudeson-2026   # enables wandb logging
```

If `wandb` is not installed, falls back to TensorBoard, then stdout. No failures if neither is available.

---

## Quick-Start Examples

```python
# Generation 1 (Foundation)
from claudson import Claudson, ModelArgs
import torch
model = Claudson(ModelArgs())
out = model(text=torch.randint(0, 1000, (1, 128)))

# Generation 6 (Jedi — Free Energy)
from claudson_jedi import ClaudesonJedi, ModelArgs
model = ClaudesonJedi(ModelArgs())
out = model(text=torch.randint(0, 1000, (1, 128)))
print(f"Goal: {out['jedi_goal']}")

# Generation 9 (Transcendent — Global Workspace)
from claudson_transcendent import ClaudesonTranscendent, ModelArgs
model = ClaudesonTranscendent(ModelArgs())
out = model(text=torch.randint(0, 1000, (1, 128)))
print(f"Spikes: {out['lif']}")
```

---

## Key Things to Know When Modifying This Code

1. **`gradient_checkpointing` defaults to `False`** — enabling it requires care (see REVIEW_REPORT.md for the history of bugs here)
2. **Context limits** — earlier versions had hard context limits that were fixed (commits `f3dc210`, `b02ca37`); use `position_offset` tracking and YaRN RoPE for unbounded context
3. **Parallel scan** — the `parallel_scan` implementation was refactored (commit `3bcb307`); always verify correctness against sequential reference via `test_new_modules.py::test_parallel_scan_matches_sequential`
4. **Auxiliary loss weights** — never apply auxiliary losses at full weight from step 0; they must ramp up via the warmup schedule
5. **Licensing boundary** — G1–5 are AGPL-3.0 (open); G6–9 are proprietary commercial. Do not mix license headers
6. **`ModelArgs` inheritance chain** — each generation's `ModelArgs` must subclass the parent's to preserve compatibility with the trainer
7. **`RMSNorm`** — use the shared implementation from `claudson_utils.py` rather than reimplementing it in each module
8. **No `requirements.txt`** — PyTorch is assumed pre-installed; `wandb` and `tensorboard` are optional and imported with graceful fallback
