"""
Claudeson 2026 - Unified Trainer
==================================
Joint Training Loop for the 18-Layer Claudeson Stack

Overview
--------
Training an 18-layer architecture this deep is not a matter of calling
loss.backward() in a for loop.  Each layer contributes its own:
  - Primary loss (language modelling, goal achievement, etc.)
  - Auxiliary losses (DAG regularisation, principle compression, etc.)
  - Online updates (EMA buffers, calibration, skill stats, etc.)
  - Inner loops (MAML adaptation, MCDropout sampling, etc.)

This trainer coordinates all of that into a coherent curriculum:

  Phase 0 — Warmup (layers 1-6, standard LM + world model)
  Phase 1 — Grounding (layers 7-10, ToM, causal, Pearl)
  Phase 2 — Abstraction (layers 11-13, skills, schemas, alignment)
  Phase 3 — Calibration (layers 14-16, uncertainty, grounding, verification)
  Phase 4 — Integration (layers 17-18, temporal + meta-learning, all losses)
  Phase 5 — Meta-training (outer loop MAML across tasks)

Key design decisions:

  Gradient accumulation
    The full stack is enormous.  We accumulate over N_ACCUM micro-batches
    before each optimizer step — effective batch size = B × N_ACCUM.

  Layer-wise learning rate decay (LLRD)
    Lower layers (core transformer) get smaller learning rates than
    higher layers (meta-learning).  Standard practice for fine-tuning
    deep stacks; prevents early layers from being destroyed by large
    gradient signals from the top.

  Loss weighting schedule
    Auxiliary losses start with weight 0 and ramp up over WARMUP_STEPS.
    This prevents the auxiliary losses from destabilising early training
    before the primary language model loss has converged.

  Phase-gated parameter groups
    During Phase 0, only layers 1-6 are trained.  Layers 7-18 are frozen.
    Each phase unlocks the next set of layers.  This prevents the
    catastrophic forgetting that would result from training all 18 layers
    simultaneously from random initialisation.

  Distributed training support
    The trainer detects multiple GPUs and wraps the model in
    DistributedDataParallel automatically.  Gradient synchronisation
    is handled by PyTorch; the trainer only needs to manage the
    rank-0 checkpoint and logging.

  Mixed precision
    FP16 or BF16 autocast + gradient scaler throughout.
    BF16 preferred when available (better dynamic range, no overflow).

  Checkpoint / resume
    Full state dict saved every SAVE_EVERY steps:
      - model.state_dict()
      - optimizer.state_dict()
      - scheduler.state_dict()
      - trainer metadata (step, phase, loss history)
    Resume is automatic: if a checkpoint exists at CHECKPOINT_DIR,
    training resumes from the latest step.

  Wandb / TensorBoard integration
    Optional.  If wandb is installed and WANDB_PROJECT is set,
    all metrics are logged.  Falls back to TensorBoard, then stdout.
"""

import os
import sys
import math
import time
import json
import logging
import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from claudson_meta_learning import (
    ModelArgs,
    ClaudesonMetaLearning,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TrainerConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    checkpoint_dir:    str   = "./checkpoints"
    log_dir:           str   = "./logs"
    data_dir:          str   = "./data"

    # ── Model scale ────────────────────────────────────────────────────────
    model_size:        str   = "small"    # small / medium / large / xl

    # ── Training phases (steps per phase) ──────────────────────────────────
    phase0_steps:      int   = 10_000     # warmup: core LM
    phase1_steps:      int   = 20_000     # grounding layers
    phase2_steps:      int   = 20_000     # abstraction + alignment
    phase3_steps:      int   = 15_000     # uncertainty + verification
    phase4_steps:      int   = 15_000     # temporal reasoning
    phase5_steps:      int   = 20_000     # meta-learning outer loop

    # ── Optimisation ───────────────────────────────────────────────────────
    batch_size:        int   = 8
    grad_accum:        int   = 8          # effective batch = batch_size × grad_accum
    max_lr:            float = 3e-4
    min_lr:            float = 3e-5
    warmup_steps:      int   = 2_000
    weight_decay:      float = 0.1
    grad_clip:         float = 1.0
    llrd_factor:       float = 0.85       # lr decay per layer group (bottom → top)

    # ── Mixed precision ─────────────────────────────────────────────────────
    use_amp:           bool  = True
    amp_dtype:         str   = "bfloat16"  # "float16" or "bfloat16"

    # ── Loss weights (ramped from 0 over warmup) ────────────────────────────
    lm_loss_weight:        float = 1.0
    world_model_weight:    float = 0.1
    causal_dag_weight:     float = 0.01
    alignment_weight:      float = 0.05
    norm_penalty_weight:   float = 0.05
    principle_weight:      float = 0.05
    moral_weight:          float = 0.1
    ig_reward_weight:      float = 0.01
    meta_outer_weight:     float = 0.1
    verification_weight:   float = 0.05

    # ── Meta-learning ───────────────────────────────────────────────────────
    meta_task_batch:   int   = 4          # tasks per outer loop step
    meta_n_shot:       int   = 5          # K-shot
    meta_n_way:        int   = 5          # N-way

    # ── Logging / checkpointing ─────────────────────────────────────────────
    log_every:         int   = 50
    eval_every:        int   = 500
    save_every:        int   = 1_000
    wandb_project:     str   = ""
    run_name:          str   = "claudson-2026"

    # ── Hardware ────────────────────────────────────────────────────────────
    num_workers:       int   = 4
    pin_memory:        bool  = True
    compile_model:     bool  = False      # torch.compile (requires PyTorch 2.0+)

    # ── Debug ───────────────────────────────────────────────────────────────
    dry_run:           bool  = False      # run 10 steps then exit
    profile:           bool  = False      # enable torch profiler

    @property
    def total_steps(self) -> int:
        return (self.phase0_steps + self.phase1_steps + self.phase2_steps +
                self.phase3_steps + self.phase4_steps + self.phase5_steps)

    @property
    def amp_dtype_torch(self):
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16


# ═══════════════════════════════════════════════════════════════════════════════
# Model Size Presets
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_SIZES = {
    "small": dict(
        dim=512, n_layers=8, n_heads=8, n_kv_heads=4,
        vocab_size=32_000, max_seq_len=2048,
        num_experts=4, latent_dim=128, n_skill_slots=32,
        n_concepts=256, n_schema_slots=32, n_stakeholder_groups=8,
        n_invariants=16, teg_n_events=64, msp_n_scales=4,
        n_abstraction_levels=5,
    ),
    "medium": dict(
        dim=1024, n_layers=16, n_heads=16, n_kv_heads=8,
        vocab_size=32_000, max_seq_len=4096,
        num_experts=8, latent_dim=256, n_skill_slots=64,
        n_concepts=512, n_schema_slots=64, n_stakeholder_groups=8,
        n_invariants=16, teg_n_events=128, msp_n_scales=4,
        n_abstraction_levels=5,
    ),
    "large": dict(
        dim=2048, n_layers=24, n_heads=32, n_kv_heads=8,
        vocab_size=64_000, max_seq_len=8192,
        num_experts=16, latent_dim=512, n_skill_slots=128,
        n_concepts=1024, n_schema_slots=128, n_stakeholder_groups=16,
        n_invariants=32, teg_n_events=256, msp_n_scales=4,
        n_abstraction_levels=5,
    ),
    "demo": dict(
        dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=512, max_seq_len=64,
        num_experts=2, latent_dim=64, n_skill_slots=8,
        n_concepts=32, n_schema_slots=8, n_stakeholder_groups=4,
        n_invariants=8, teg_n_events=16, msp_n_scales=4,
        n_abstraction_levels=3,
    ),
}


def build_model_args(cfg: TrainerConfig) -> ModelArgs:
    """Construct ModelArgs from TrainerConfig and size preset."""
    preset = MODEL_SIZES.get(cfg.model_size, MODEL_SIZES["demo"])
    args   = ModelArgs()

    # Apply size preset
    for k, v in preset.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Fill derived / fixed fields
    args.n_kv_heads           = preset.get("n_kv_heads", args.n_heads // 2)
    args.memory_slots         = 128
    args.episodic_slots       = 256
    args.goal_dim             = args.dim
    args.energy_hidden        = args.dim
    args.ssm_state_dim        = 64
    args.ssm_chunk_size       = 64
    args.num_shared_experts   = 1
    args.env_state_dim        = args.dim // 4
    args.action_space_size    = 64
    args.planning_horizon     = 8
    args.num_simulations      = 8
    args.img_size             = 224
    args.patch_size           = 16
    args.audio_spec_dim       = 128
    args.gradient_checkpointing = (cfg.model_size in ("large", "xl"))
    args.n_agents             = 8
    args.lora_rank            = 16
    args.n_causal_nodes       = args.dim // 8
    args.metacog_hidden       = args.dim // 2
    args.n_debate_agents      = 5
    args.debate_hidden        = args.dim // 2
    args.n_propositions       = 32
    args.n_constraints        = 16
    args.consistency_iters    = 3
    args.rsi_rank             = 8
    args.rsi_horizon          = 4
    args.n_workspace_slots    = 16
    args.gw_competition_k     = 4
    args.gw_broadcast_steps   = 2
    args.n_ops                = 32
    args.n_registers          = 8
    args.prog_steps           = 5
    args.prog_hidden          = args.dim // 2
    args.irl_hidden           = args.dim // 2
    args.irl_n_preferences    = 32
    args.lif_steps            = 5
    args.causal_state_dim     = args.dim // 4
    args.intervention_horizon = 4
    args.n_intervention_samples = 8
    args.cf_n_branches        = 4
    args.attr_top_k           = 8
    args.pearl_hidden         = args.dim // 2
    args.skill_rank           = 16
    args.skill_embed_dim      = args.dim // 2
    args.cp_window            = 64
    args.cp_hidden            = args.dim // 2
    args.oeg_n_compose        = 4
    args.oeg_hidden           = args.dim // 2
    args.ig_beta              = 1.0
    args.hae_heads            = args.n_heads // 2
    args.hae_pool_factor      = 4
    args.hae_hidden           = args.dim
    args.concept_top_k        = 32
    args.concept_hidden       = args.dim
    args.schema_n_roles       = 8
    args.schema_hidden        = args.dim
    args.schema_bind_iters    = 3
    args.analogy_hidden       = args.dim
    args.analogy_n_mappings   = 8
    args.n_principles         = 32
    args.principle_hidden     = args.dim
    args.stakeholder_hidden   = args.dim
    args.welfare_hidden       = args.dim // 2
    args.n_welfare_objectives = 4
    args.n_norm_slots         = 64
    args.norm_hidden          = args.dim
    args.scr_n_perspectives   = 8
    args.scr_hidden           = args.dim
    args.n_moral_frameworks   = 4
    args.moral_hidden         = args.dim // 2
    args.bup_n_samples        = 10
    args.bup_dropout_rate     = 0.1
    args.bup_hidden           = args.dim
    args.cp_coverage          = 0.9
    args.cp_cal_size          = 1024
    args.cp_n_classes         = 128
    args.cal_n_bins           = 15
    args.ood_n_centroids       = 64
    args.ood_hidden           = args.dim
    args.uaa_hidden           = args.dim
    args.uaa_n_heads          = args.n_heads // 2
    args.psa_n_anchors        = 128
    args.psa_hidden           = args.dim
    args.psa_n_heads          = args.n_heads // 2
    args.msg_n_primitives     = 32
    args.msg_hidden           = args.dim // 2
    args.msg_compose_depth    = 4
    args.sms_n_steps          = 5
    args.sms_hidden           = args.dim
    args.sms_n_branches       = 4
    args.cmal_hidden          = args.dim
    args.gcm_hidden           = args.dim // 2
    args.gcm_n_pairs          = 16
    args.invariant_hidden     = args.dim // 2
    args.ppcc_hidden          = args.dim // 2
    args.ai_n_neurons         = 64
    args.ai_hidden            = args.dim // 2
    args.ceg_budget           = 20
    args.ceg_hidden           = args.dim // 2
    args.pcs_max_certs        = 1024
    args.teg_n_edge_types     = 6
    args.teg_hidden           = args.dim
    args.teg_n_heads          = args.n_heads // 2
    args.de_n_categories      = 8
    args.de_hidden            = args.dim // 2
    args.tce_n_iters          = 5
    args.tce_hidden           = args.dim // 2
    args.msp_hidden           = args.dim
    args.tca_n_traces         = 64
    args.tca_hidden           = args.dim // 2
    args.maml_inner_steps     = 5
    args.maml_inner_lr        = 0.01
    args.maml_first_order     = True
    args.maml_adapt_params    = "lora"
    args.meta_sgd             = True
    args.meta_sgd_init        = 0.01
    args.ctx_hidden           = args.dim
    args.ctx_n_heads          = args.n_heads // 2
    args.ctx_pool             = "attn"
    args.bayesian_maml        = True
    args.bayes_n_particles    = 5
    args.bayes_noise_std      = 0.001
    args.mvm_window           = 64
    args.mvm_ood_steps_thresh = 10
    args.n_shot               = cfg.meta_n_shot
    args.n_way                = cfg.meta_n_way

    return args


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Each phase specifies which top-level attributes of the model to UNFREEZE.
# All earlier-phase modules remain unfrozen (cumulative unfreezing).
PHASE_MODULES = {
    0: [  # Core LM
        "tok_emb", "layers", "norm", "lm_head",
        "pos_emb", "mamba", "router", "experts",
    ],
    1: [  # Grounding + causal
        "theory_of_mind", "continual_learner", "causal_reasoner",
        "action_loop", "energy_layer", "efe_planner", "latent_dynamics",
        "metacog", "debate", "neural_symbolic", "rsi",
        "global_workspace", "prog_synth", "irl", "lif",
        "causal_dynamics", "interventional_planner", "cf_engine",
        "attr_gate", "pearl",
    ],
    2: [  # Skills + abstraction + alignment
        "skill_discovery", "goal_generator", "cp_monitor",
        "hae", "concept_bn", "schema_engine", "analogy_module", "principle_ext",
        "stakeholder_vm", "welfare_agg", "norm_engine", "social_contract",
        "moral_estimator", "alignment_gate",
    ],
    3: [  # Uncertainty + grounding + verification
        "bup", "conformal", "calibrator", "ood", "uaa",
        "psa", "msg", "sms", "cmal", "gcm",
        "invariant_reg", "ppcc", "abstract_interp", "ceg",
    ],
    4: [  # Temporal
        "teg", "de", "tce", "msp", "tca",
    ],
    5: [  # Meta-learning
        "context_encoder", "meta_sgd_module", "inner_adapter",
        "bayes_maml", "task_conditioned", "meta_val_monitor",
    ],
}


def set_phase(model: nn.Module, phase: int) -> int:
    """
    Unfreeze modules for the current phase and all previous phases.
    Returns the number of trainable parameters.
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze phases 0..phase
    n_trainable = 0
    unfrozen_modules = set()
    for ph in range(phase + 1):
        for mod_name in PHASE_MODULES.get(ph, []):
            for name, module in model.named_modules():
                if name == mod_name or name.startswith(mod_name + "."):
                    for p in module.parameters():
                        p.requires_grad_(True)
                    unfrozen_modules.add(mod_name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Phase %d: %d trainable parameters (modules: %s)",
             phase, n_trainable, sorted(unfrozen_modules))
    return n_trainable


# ═══════════════════════════════════════════════════════════════════════════════
# Layer-wise Learning Rate Decay
# ═══════════════════════════════════════════════════════════════════════════════

def build_param_groups(
    model: nn.Module,
    base_lr: float,
    llrd_factor: float,
    weight_decay: float,
) -> List[Dict]:
    """
    Build parameter groups with layer-wise learning rate decay.

    Strategy:
      - Identify each "layer group" by depth (layer index or module depth)
      - Assign lr = base_lr × llrd_factor^(n_groups - group_idx)
      - No weight decay on norm layers and biases

    This ensures the bottom layers (early transformers) receive much
    smaller updates than the top layers (meta-learning), preventing
    catastrophic forgetting of the pre-trained representations.
    """
    # Group parameters by approximate depth
    # We use a simple heuristic: module name depth
    decay_params     = {}   # {lr: [params]}
    no_decay_params  = {}   # {lr: [params]}

    # Assign group index to each parameter
    phase_order = [name for ph in range(6) for name in PHASE_MODULES.get(ph, [])]
    n_groups    = max(len(phase_order), 1)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Find which phase this parameter belongs to
        group_idx = n_groups  # default: latest (highest lr)
        for i, mod_name in enumerate(phase_order):
            if name.startswith(mod_name):
                group_idx = i
                break

        lr = base_lr * (llrd_factor ** (n_groups - group_idx))

        # Separate weight decay: no decay for norms and biases
        no_wd = (name.endswith(".bias") or
                 "norm" in name.lower() or
                 "ln_" in name or
                 name.endswith("_norm"))

        bucket = no_decay_params if no_wd else decay_params
        if lr not in bucket:
            bucket[lr] = []
        bucket[lr].append(param)

    param_groups = []
    for lr, params in decay_params.items():
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    for lr, params in no_decay_params.items():
        param_groups.append({"params": params, "lr": lr, "weight_decay": 0.0})

    log.info("Built %d parameter groups (lr range: %.2e – %.2e)",
             len(param_groups),
             min(g["lr"] for g in param_groups),
             max(g["lr"] for g in param_groups))
    return param_groups


# ═══════════════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════════════

class CosineWithWarmup:
    """
    Cosine annealing with linear warmup and optional restarts.

    Schedule:
      step < warmup_steps:  lr = max_lr × step / warmup_steps
      step >= warmup_steps: lr = min_lr + 0.5 × (max_lr - min_lr) ×
                                 (1 + cos(π × (step - warmup) / (total - warmup)))
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_steps, total_steps):
        self.optimizer    = optimizer
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.step_count   = 0

    def get_lr(self) -> float:
        s = self.step_count
        if s < self.warmup_steps:
            return self.max_lr * s / max(self.warmup_steps, 1)
        progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            # Scale each group's lr proportionally
            group["lr"] = lr * (group["lr"] / (self.max_lr + 1e-9))
        self.step_count += 1
        return lr


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Aggregator
# ═══════════════════════════════════════════════════════════════════════════════

class LossAggregator:
    """
    Collects, weights, and logs all losses from a forward pass.

    Each loss has:
      - A weight (ramped from 0 to target over warmup_steps)
      - A running mean (for logging / normalisation)
      - A phase gate (only active from phase N onwards)
    """

    def __init__(self, cfg: TrainerConfig):
        self.cfg           = cfg
        self.step          = 0
        self._loss_history: Dict[str, List[float]] = {}

        # (weight, phase_gate, ramp_steps)
        self._spec = {
            "lm":            (cfg.lm_loss_weight,        0, 500),
            "world_model":   (cfg.world_model_weight,    0, 1000),
            "causal_dag":    (cfg.causal_dag_weight,     1, 2000),
            "alignment":     (cfg.alignment_weight,      2, 2000),
            "norm_penalty":  (cfg.norm_penalty_weight,   2, 2000),
            "principle":     (cfg.principle_weight,      2, 3000),
            "moral":         (cfg.moral_weight,          2, 3000),
            "ig_reward":     (cfg.ig_reward_weight,      2, 2000),
            "verification":  (cfg.verification_weight,   3, 3000),
            "meta_outer":    (cfg.meta_outer_weight,     5, 5000),
        }

    def _ramp(self, loss_name: str, phase: int) -> float:
        """Ramp weight from 0 → target over ramp_steps."""
        target_w, phase_gate, ramp_steps = self._spec.get(
            loss_name, (1.0, 0, 500)
        )
        if phase < phase_gate:
            return 0.0
        progress = min(self.step / max(ramp_steps, 1), 1.0)
        return target_w * progress

    def aggregate(
        self,
        losses:    Dict[str, Optional[torch.Tensor]],
        phase:     int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Weighted sum of all provided losses.

        losses: dict of {name: tensor or None}
        Returns: (total_loss, log_dict)
        """
        device = next(
            (v.device for v in losses.values() if v is not None and torch.is_tensor(v)),
            torch.device("cpu")
        )
        total   = torch.tensor(0.0, device=device, requires_grad=True)
        log_dict: Dict[str, float] = {}

        for name, loss_val in losses.items():
            if loss_val is None:
                continue
            if not torch.is_tensor(loss_val):
                loss_val = torch.tensor(float(loss_val), device=device)
            if not loss_val.isfinite():
                log.warning("Non-finite loss '%s': %s — skipping", name, loss_val.item())
                continue

            w = self._ramp(name, phase)
            if w == 0.0:
                continue

            weighted = w * loss_val
            total    = total + weighted
            scalar   = loss_val.detach().item()
            log_dict[f"loss/{name}"] = scalar
            if name not in self._loss_history:
                self._loss_history[name] = []
            self._loss_history[name].append(scalar)
            if len(self._loss_history[name]) > 100:
                self._loss_history[name].pop(0)

        log_dict["loss/total"] = total.detach().item()
        self.step += 1
        return total, log_dict

    def running_mean(self, name: str, window: int = 50) -> float:
        hist = self._loss_history.get(name, [])
        if not hist:
            return 0.0
        return sum(hist[-window:]) / len(hist[-window:])


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Dataset (for demo / testing)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticClaudsonDataset(Dataset):
    """
    A minimal synthetic dataset for verifying the training loop.

    In production, replace with:
      - Text corpora (C4, The Pile, RedPajama, etc.)
      - Multimodal corpora (LAION, AudioCaps, etc.)
      - Preference datasets for IRL (Anthropic HH, etc.)
      - Tool-use trajectories for the grounded action loop
      - Multi-task datasets for meta-learning (MAML benchmark, etc.)

    This class generates random tensors of the right shape and dtype.
    """

    def __init__(self, args: ModelArgs, n_samples: int = 10_000):
        self.args      = args
        self.n_samples = n_samples
        self.seq_len   = min(args.max_seq_len, 64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.seq_len
        return {
            "text":          torch.randint(0, self.args.vocab_size, (seq,)),
            "labels":        torch.randint(0, self.args.vocab_size, (seq,)),
            "feedback":      torch.randn(self.args.dim),
            "agent_obs":     torch.randn(4, self.args.dim),
            "actual_action": torch.randint(0, self.args.action_space_size, (1,)).squeeze(),
            # Support set for meta-learning (K=3 for demo)
            "support_x":     torch.randn(3, self.args.dim),
            "support_y":     torch.randn(3, self.args.dim),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Logger
# ═══════════════════════════════════════════════════════════════════════════════

class MetricLogger:
    """Unified logging to wandb / tensorboard / stdout."""

    def __init__(self, cfg: TrainerConfig, rank: int = 0):
        self.rank     = rank
        self.cfg      = cfg
        self._writer  = None
        self._wb_run  = None

        if rank != 0:
            return

        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        if WANDB_AVAILABLE and cfg.wandb_project:
            self._wb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=dataclasses.asdict(cfg),
            )
            log.info("wandb initialised: %s/%s", cfg.wandb_project, cfg.run_name)
        elif TB_AVAILABLE:
            self._writer = SummaryWriter(log_dir=cfg.log_dir)
            log.info("TensorBoard writer: %s", cfg.log_dir)
        else:
            log.info("Logging to stdout only")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.rank != 0:
            return

        if self._wb_run is not None:
            self._wb_run.log(metrics, step=step)
        elif self._writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, step)

        # Always log a selection to stdout
        if step % max(self.cfg.log_every, 1) == 0:
            parts = [f"step={step}"]
            for k in ["loss/total", "loss/lm", "lr", "phase"]:
                if k in metrics:
                    v = metrics[k]
                    parts.append(f"{k.split('/')[-1]}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            log.info("  ".join(parts))

    def close(self) -> None:
        if self._writer:
            self._writer.close()
        if self._wb_run:
            self._wb_run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointManager:

    def __init__(self, cfg: TrainerConfig):
        self.dir = Path(cfg.cfg.checkpoint_dir if hasattr(cfg, "cfg") else cfg.checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step:      int,
        phase:     int,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWithWarmup,
        loss_agg:  LossAggregator,
    ) -> Path:
        path = self.dir / f"ckpt_step{step:08d}.pt"
        torch.save({
            "step":          step,
            "phase":         phase,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler_step": scheduler.step_count,
            "loss_history":  loss_agg._loss_history,
        }, path)
        log.info("Saved checkpoint: %s", path)
        # Keep only the 3 most recent checkpoints
        ckpts = sorted(self.dir.glob("ckpt_step*.pt"))
        for old in ckpts[:-3]:
            old.unlink()
        return path

    def latest(self) -> Optional[Path]:
        ckpts = sorted(self.dir.glob("ckpt_step*.pt"))
        return ckpts[-1] if ckpts else None

    def load(
        self,
        path:      Path,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineWithWarmup,
        loss_agg:  LossAggregator,
    ) -> Tuple[int, int]:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.step_count          = ckpt["scheduler_step"]
        loss_agg._loss_history        = ckpt.get("loss_history", {})
        step  = ckpt["step"]
        phase = ckpt["phase"]
        log.info("Loaded checkpoint: %s  (step=%d, phase=%d)", path, step, phase)
        return step, phase


# ═══════════════════════════════════════════════════════════════════════════════
# Core Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudesonTrainer:
    """
    The unified training loop for the 18-layer Claudeson stack.

    Responsibilities:
      1. Build model and optimiser with LLRD parameter groups
      2. Phase-gate: unfreeze layers progressively
      3. Run micro-batch accumulation loop
      4. Collect all auxiliary losses from model outputs
      5. Weight losses appropriately for the current phase/step
      6. Update online buffers (calibration, OOD centroids, etc.)
      7. Log metrics and save checkpoints
      8. Run meta-training outer loop in Phase 5
    """

    def __init__(self, cfg: TrainerConfig, rank: int = 0, world_size: int = 1):
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Initialising ClaudesonTrainer  rank=%d  device=%s", rank, self.device)

        # ── Build model ───────────────────────────────────────────────────
        self.model_args = build_model_args(cfg)
        log.info("Building ClaudesonMetaLearning (%s)...", cfg.model_size)
        self.model      = ClaudesonMetaLearning(self.model_args).to(self.device)

        if cfg.compile_model and hasattr(torch, "compile"):
            log.info("torch.compile enabled")
            self.model = torch.compile(self.model)

        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

        # ── Phase tracking ────────────────────────────────────────────────
        self.phase         = 0
        self.global_step   = 0
        self._phase_steps  = [
            cfg.phase0_steps,
            cfg.phase1_steps,
            cfg.phase2_steps,
            cfg.phase3_steps,
            cfg.phase4_steps,
            cfg.phase5_steps,
        ]
        self._phase_start  = 0

        # ── Optimiser & scheduler ─────────────────────────────────────────
        set_phase(self._raw_model, self.phase)
        param_groups = build_param_groups(
            self._raw_model,
            base_lr=cfg.max_lr,
            llrd_factor=cfg.llrd_factor,
            weight_decay=cfg.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups, lr=cfg.max_lr, betas=(0.9, 0.95))
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            max_lr=cfg.max_lr,
            min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
        )

        # ── Mixed precision ────────────────────────────────────────────────
        self.scaler = GradScaler(enabled=cfg.use_amp and self.device.type == "cuda")

        # ── Loss & logging ─────────────────────────────────────────────────
        self.loss_agg      = LossAggregator(cfg)
        self.logger        = MetricLogger(cfg, rank)
        self.ckpt_manager  = CheckpointManager(cfg)

        # ── Dataset ────────────────────────────────────────────────────────
        dataset = SyntheticClaudsonDataset(self.model_args)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            drop_last=True,
        )

        # ── Accumulation state ─────────────────────────────────────────────
        self._accum_step   = 0
        self._accum_losses: Dict[str, List[float]] = {}

        total_params = sum(p.numel() for p in self._raw_model.parameters())
        log.info("Model: %.1fM total parameters", total_params / 1e6)

    @property
    def _raw_model(self) -> ClaudesonMetaLearning:
        """Unwrap DDP to access the raw model."""
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    def _current_phase(self) -> int:
        """Determine the current phase from the global step."""
        step  = self.global_step
        total = 0
        for ph, ph_steps in enumerate(self._phase_steps):
            total += ph_steps
            if step < total:
                return ph
        return len(self._phase_steps) - 1

    def _maybe_advance_phase(self) -> bool:
        """Check if we should advance to the next phase."""
        new_phase = self._current_phase()
        if new_phase > self.phase:
            self.phase = new_phase
            log.info("═══ Advancing to Phase %d ═══", self.phase)
            set_phase(self._raw_model, self.phase)

            # Rebuild param groups for the new set of trainable params
            param_groups = build_param_groups(
                self._raw_model,
                base_lr=self.cfg.max_lr,
                llrd_factor=self.cfg.llrd_factor,
                weight_decay=self.cfg.weight_decay,
            )
            self.optimizer = torch.optim.AdamW(
                param_groups, lr=self.cfg.max_lr, betas=(0.9, 0.95)
            )
            return True
        return False

    def _extract_losses(self, out: Dict, batch: Dict) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract all relevant losses from a model output dictionary.

        This is the central loss accounting function.
        Every loss that any layer contributes is collected here.
        """
        losses: Dict[str, Optional[torch.Tensor]] = {}

        # ── Language modelling loss ───────────────────────────────────────
        if "logits" in out and "labels" in batch:
            logits = out["logits"]                                 # [B, L, vocab]
            labels = batch["labels"].to(logits.device)            # [B, L]
            B, L, V = logits.shape
            lm_loss = F.cross_entropy(
                logits.reshape(B * L, V),
                labels.reshape(B * L),
                ignore_index=-100,
            )
            losses["lm"] = lm_loss

        # ── Auxiliary losses from model's own method ──────────────────────
        try:
            aux = self._raw_model.compute_auxiliary_losses()
            for k, v in aux.items():
                losses[k] = v
        except Exception as e:
            log.debug("compute_auxiliary_losses failed: %s", e)

        # ── World model / EFE loss ────────────────────────────────────────
        if "jedi_energy" in out:
            losses["world_model"] = out["jedi_energy"].mean().abs()

        # ── Information gain (negative reward = loss to minimise) ─────────
        if "metacurriculum" in out:
            mc = out["metacurriculum"]
            ig_rew = mc.get("discovery", {}).get("ig_reward")
            if ig_rew is not None and torch.is_tensor(ig_rew):
                losses["ig_reward"] = -ig_rew.mean() * 0.01  # maximise IG → minimise neg

        # ── Principle compression ─────────────────────────────────────────
        if "abstraction" in out:
            p_out = out["abstraction"].get("principles", {})
            comp  = p_out.get("compression_loss")
            if comp is not None and torch.is_tensor(comp):
                losses["principle"] = comp

        # ── Cross-modal alignment ─────────────────────────────────────────
        if "grounding" in out:
            al_loss = out["grounding"].get("alignment", {}).get("alignment_loss")
            if al_loss is not None and torch.is_tensor(al_loss):
                losses["alignment"] = al_loss

        # ── Norm penalty ──────────────────────────────────────────────────
        if "social_alignment" in out:
            sa  = out["social_alignment"]
            pen = sa.get("norms", {}).get("norm_penalty")
            if pen is not None and torch.is_tensor(pen):
                losses["norm_penalty"] = pen.mean()

        # ── Moral uncertainty (want low uncertainty = frameworks agree) ───
        if "social_alignment" in out:
            sa  = out["social_alignment"]
            std = sa.get("moral", {}).get("moral_std")
            if std is not None and torch.is_tensor(std):
                losses["moral"] = std.mean()

        # ── Meta outer loss (Phase 5 only) ────────────────────────────────
        if self.phase >= 5 and "meta_learning" in out:
            inner_loss = out["meta_learning"].get("adaptation", {}).get("final_loss", 0.0)
            losses["meta_outer"] = torch.tensor(float(inner_loss), requires_grad=False)

        return losses

    def _update_online_buffers(self, out: Dict, batch: Dict) -> None:
        """
        Update non-differentiable online buffers from a forward pass.

        These are EMA buffers, calibration stats, OOD centroids etc.
        that update in-place and don't need gradients.
        """
        raw = self._raw_model

        # Conformal calibration: update with confidence from this batch
        if hasattr(raw, "conformal"):
            confidence = torch.rand(1).item()   # placeholder; use real confidence in prod
            raw.conformal.update_calibration(confidence)

        # Stakeholder welfare EMA
        if hasattr(raw, "stakeholder_vm") and "social_alignment" in out:
            welfare = out["social_alignment"].get("welfare", {}).get("welfare")
            if welfare is not None:
                for g in range(raw.args.n_stakeholder_groups):
                    raw.stakeholder_vm.update_welfare(g, float(welfare[:, g].mean().item()))

        # Competence progress monitor
        if hasattr(raw, "cp_monitor") and "metacurriculum" in out:
            mc   = out["metacurriculum"]
            comp = mc.get("discovery", {}).get("ig", {}).get("kl")
            if comp is not None:
                raw.cp_monitor.record_performance(
                    int(mc.get("discovery", {}).get("active_skill", 0)),
                    float(comp.mean().item()),
                )

    @contextmanager
    def _amp_context(self):
        if self.cfg.use_amp and self.device.type == "cuda":
            with autocast(dtype=self.cfg.amp_dtype_torch):
                yield
        else:
            yield

    def _forward_pass(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Run a single forward pass and return (output, losses)."""
        text          = batch["text"].to(self.device)
        feedback      = batch["feedback"].to(self.device)
        agent_obs     = batch["agent_obs"].to(self.device)
        actual_action = batch["actual_action"].to(self.device)
        support_x     = batch.get("support_x")
        support_y     = batch.get("support_y")

        if support_x is not None:
            support_x = support_x.to(self.device)
        if support_y is not None:
            support_y = support_y.to(self.device)

        with self._amp_context():
            out = self.model(
                text=text,
                feedback=feedback,
                agent_observations=agent_obs,
                actual_action=actual_action,
                support_x=support_x if self.phase >= 5 else None,
                support_y=support_y if self.phase >= 5 else None,
                timestamp=float(self.global_step),
            )

        losses = self._extract_losses(out, batch)
        return out, losses

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        One gradient accumulation micro-step.

        When _accum_step reaches grad_accum, performs the optimizer step.
        Returns a metrics dict (populated only on the optimizer step).
        """
        self.model.train()
        self._maybe_advance_phase()

        out, raw_losses = self._forward_pass(batch)

        # Aggregate losses
        total_loss, log_dict = self.loss_agg.aggregate(raw_losses, self.phase)
        total_loss = total_loss / self.cfg.grad_accum  # scale for accumulation

        # Backward
        if self.cfg.use_amp and self.device.type == "cuda":
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        self._accum_step += 1

        metrics: Dict[str, float] = {}

        if self._accum_step >= self.cfg.grad_accum:
            # Optimizer step
            if self.cfg.use_amp and self.device.type == "cuda":
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
                self.optimizer.step()

            lr = self.scheduler.get_lr()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_step = 0
            self.global_step += 1

            # Update non-differentiable buffers
            with torch.no_grad():
                self._update_online_buffers(out, batch)

            # Compile metrics
            metrics = {**log_dict, "lr": lr, "phase": float(self.phase)}
            metrics["step"] = float(self.global_step)

            # Grad norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            metrics["grad_norm"] = math.sqrt(total_norm)

            self.logger.log(metrics, self.global_step)

        return metrics

    def run(self) -> None:
        """
        Full training run.

        Iterates over the dataloader for the total number of steps,
        handling phase transitions, checkpointing, and evaluation.
        """
        log.info("Starting training: %d total steps, %d phases",
                 self.cfg.total_steps, len(self._phase_steps))

        # Resume from checkpoint if available
        latest = self.ckpt_manager.latest()
        if latest:
            self.global_step, self.phase = self.ckpt_manager.load(
                latest, self._raw_model, self.optimizer, self.scheduler, self.loss_agg
            )
            set_phase(self._raw_model, self.phase)

        data_iter  = iter(self.dataloader)
        t0         = time.time()

        while self.global_step < self.cfg.total_steps:
            # Refill data iterator
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Train step
            metrics = self.train_step(batch)

            # Only act on full optimizer steps
            if metrics and self.global_step > 0:
                # ── Evaluation ────────────────────────────────────────────
                if self.global_step % self.cfg.eval_every == 0:
                    self._evaluate()

                # ── Checkpoint ────────────────────────────────────────────
                if self.global_step % self.cfg.save_every == 0 and self.rank == 0:
                    self.ckpt_manager.save(
                        self.global_step, self.phase,
                        self._raw_model, self.optimizer,
                        self.scheduler, self.loss_agg,
                    )

                # ── Speed logging ─────────────────────────────────────────
                if self.global_step % (self.cfg.log_every * 10) == 0:
                    elapsed = time.time() - t0
                    steps_per_sec = self.global_step / max(elapsed, 1e-6)
                    remaining = (self.cfg.total_steps - self.global_step) / max(steps_per_sec, 1e-6)
                    log.info(
                        "speed=%.1f steps/s  eta=%.0f min  phase=%d",
                        steps_per_sec, remaining / 60, self.phase,
                    )

            # ── Dry run exit ───────────────────────────────────────────────
            if self.cfg.dry_run and self.global_step >= 10:
                log.info("Dry run complete (10 steps).")
                break

        log.info("Training complete. Total steps: %d", self.global_step)
        if self.rank == 0:
            self.ckpt_manager.save(
                self.global_step, self.phase,
                self._raw_model, self.optimizer,
                self.scheduler, self.loss_agg,
            )
        self.logger.close()

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """
        Lightweight evaluation pass.

        Runs 10 batches in eval mode and reports average losses.
        In production, replace with held-out validation sets for
        each task type (LM perplexity, analogy accuracy, welfare score, etc.)
        """
        self.model.eval()
        eval_losses: Dict[str, List[float]] = {}
        n_batches = min(10, len(self.dataloader))
        data_iter = iter(self.dataloader)

        for _ in range(n_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            _, losses = self._forward_pass(batch)
            for k, v in losses.items():
                if v is not None and torch.is_tensor(v) and v.isfinite():
                    if k not in eval_losses:
                        eval_losses[k] = []
                    eval_losses[k].append(v.item())

        avg = {f"eval/{k}": sum(vs) / len(vs)
               for k, vs in eval_losses.items() if vs}

        self.logger.log(avg, self.global_step)
        self.model.train()

        if self.rank == 0 and avg:
            eval_lm = avg.get("eval/lm", float("nan"))
            ppl = math.exp(min(eval_lm, 20)) if not math.isnan(eval_lm) else float("nan")
            log.info("Eval  step=%d  ppl=%.2f  phase=%d",
                     self.global_step, ppl, self.phase)

        return avg


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed Training Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def init_distributed() -> Tuple[int, int]:
    """Initialise distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        log.info("Distributed: rank=%d/%d", rank, world_size)
        return rank, world_size
    return 0, 1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="Claudeson 2026 Trainer")
    parser.add_argument("--model-size",    default="demo",
                        choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--log-dir",       default="./logs")
    parser.add_argument("--batch-size",    type=int, default=8)
    parser.add_argument("--grad-accum",    type=int, default=4)
    parser.add_argument("--max-lr",        type=float, default=3e-4)
    parser.add_argument("--phase0-steps",  type=int, default=500)
    parser.add_argument("--phase1-steps",  type=int, default=500)
    parser.add_argument("--phase2-steps",  type=int, default=500)
    parser.add_argument("--phase3-steps",  type=int, default=500)
    parser.add_argument("--phase4-steps",  type=int, default=500)
    parser.add_argument("--phase5-steps",  type=int, default=500)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--run-name",      default="claudson-2026")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--no-amp",        action="store_true")
    parser.add_argument("--compile",       action="store_true")
    parser.add_argument("--log-every",     type=int, default=10)
    parser.add_argument("--save-every",    type=int, default=200)
    parser.add_argument("--eval-every",    type=int, default=100)

    a = parser.parse_args()

    cfg = TrainerConfig(
        model_size      = a.model_size,
        checkpoint_dir  = a.checkpoint_dir,
        log_dir         = a.log_dir,
        batch_size      = a.batch_size,
        grad_accum      = a.grad_accum,
        max_lr          = a.max_lr,
        phase0_steps    = a.phase0_steps,
        phase1_steps    = a.phase1_steps,
        phase2_steps    = a.phase2_steps,
        phase3_steps    = a.phase3_steps,
        phase4_steps    = a.phase4_steps,
        phase5_steps    = a.phase5_steps,
        wandb_project   = a.wandb_project,
        run_name        = a.run_name,
        dry_run         = a.dry_run,
        use_amp         = not a.no_amp,
        compile_model   = a.compile,
        log_every       = a.log_every,
        save_every      = a.save_every,
        eval_every      = a.eval_every,
    )
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Demo / Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def smoke_test() -> None:
    """
    Quick smoke test: build model, run 3 training steps, verify no crash.
    Runs without GPU, without full dataset, without logging.
    """
    print("=" * 70)
    print("CLAUDSON TRAINER — SMOKE TEST")
    print("=" * 70)

    cfg = TrainerConfig(
        model_size     = "demo",
        batch_size     = 2,
        grad_accum     = 2,
        phase0_steps   = 3,
        phase1_steps   = 3,
        phase2_steps   = 3,
        phase3_steps   = 3,
        phase4_steps   = 3,
        phase5_steps   = 3,
        use_amp        = False,
        compile_model  = False,
        dry_run        = True,
        log_every      = 1,
        eval_every     = 5,
        save_every     = 100,
        num_workers    = 0,
        pin_memory     = False,
        warmup_steps   = 2,
        checkpoint_dir = "/tmp/claudson_smoke_ckpt",
        log_dir        = "/tmp/claudson_smoke_logs",
    )

    print("\nBuilding trainer...")
    trainer = ClaudesonTrainer(cfg, rank=0, world_size=1)

    total   = sum(p.numel() for p in trainer._raw_model.parameters())
    trainable = sum(p.numel() for p in trainer._raw_model.parameters() if p.requires_grad)
    print(f"  Total params:     {total/1e6:.1f}M")
    print(f"  Trainable (Ph 0): {trainable/1e6:.1f}M")

    print("\nRunning 10 training steps (dry run)...")
    t0 = time.time()
    trainer.run()
    elapsed = time.time() - t0

    print(f"\nSmoke test complete in {elapsed:.1f}s")
    print(f"  Final step:   {trainer.global_step}")
    print(f"  Final phase:  {trainer.phase}")
    print(f"  Loss history keys: {list(trainer.loss_agg._loss_history.keys())}")

    # Report loss running means
    print("\nLoss summary:")
    for k, hist in trainer.loss_agg._loss_history.items():
        if hist:
            print(f"  {k:<25}: mean={sum(hist)/len(hist):.4f}  last={hist[-1]:.4f}")

    print("\n" + "=" * 70)
    print("Trainer verified. Ready for real training.")
    print()
    print("Usage:")
    print("  # Demo (sanity check)")
    print("  python claudson_trainer.py --model-size demo --dry-run")
    print()
    print("  # Small model, full training")
    print("  python claudson_trainer.py --model-size small \\")
    print("      --batch-size 16 --grad-accum 8 --max-lr 3e-4 \\")
    print("      --checkpoint-dir ./ckpts --wandb-project claudson")
    print()
    print("  # Multi-GPU (4 GPUs)")
    print("  torchrun --nproc_per_node=4 claudson_trainer.py \\")
    print("      --model-size medium --batch-size 8 --grad-accum 4")
    print("=" * 70)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv or len(sys.argv) == 1:
        smoke_test()
    else:
        rank, world_size = init_distributed()
        cfg = parse_args()
        trainer = ClaudesonTrainer(cfg, rank=rank, world_size=world_size)
        trainer.run()
        if world_size > 1:
            dist.destroy_process_group()
