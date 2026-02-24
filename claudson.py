import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# ============= Configuration =============
@dataclass
class ModelArgs:
    # Scaled up parameters
    dim: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8  # For GQA
    vocab_size: int = 128000
    patch_size: int = 16
    img_size: int = 224
    audio_spec_dim: int = 128
    max_seq_len: int = 8192
    
    # Memory configuration
    memory_slots: int = 256
    memory_dim: int = 2048
    episodic_slots: int = 2560
    memory_compression: int = 4
    
    # Agency & Planning
    action_space_size: int = 100
    planning_horizon: int = 8
    num_simulations: int = 8
    env_state_dim: int = 128
    goal_dim: int = 2048
    
    # MoE configuration
    num_experts: int = 8
    expert_top_k: int = 2
    
    # Training optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    use_kv_cache: bool = True

# ============= Position Embeddings =============
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, offset: int = 0):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) + offset
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ============= Modality Encoders =============
class ModalityEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(5, dim)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        tag = self.embeddings(torch.tensor([modality_id], device=x.device))
        return x + tag

class VisionEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.proj = nn.Conv2d(3, args.dim, kernel_size=args.patch_size, stride=args.patch_size)
        num_patches = (args.img_size // args.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, args.dim) * 0.02)
        self.norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x + self.pos_embed)

class AudioEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.proj = nn.Linear(args.audio_spec_dim, args.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, args.max_seq_len, args.dim) * 0.02)
        self.norm = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x + self.pos_embed[:, :x.size(1), :])

class GoalEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.proj = nn.Sequential(
            nn.Linear(args.dim, args.dim * 2),
            nn.GELU(),
            nn.Linear(args.dim * 2, args.goal_dim),
            nn.LayerNorm(args.goal_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g_emb = self.embed(x).mean(dim=1, keepdim=True)
        return self.proj(g_emb)

# ============= Memory Systems =============
class GlobalMemoryBank(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.slots = args.memory_slots
        self.register_buffer("memory", torch.randn(1, self.slots, self.dim) * 0.02)

        self.read_query = nn.Linear(args.dim, args.dim, bias=False)
        self.read_key = nn.Linear(args.dim, args.dim, bias=False)
        self.read_val = nn.Linear(args.dim, args.dim, bias=False)

        self.write_key = nn.Linear(args.dim, args.dim, bias=False)
        self.write_val = nn.Linear(args.dim, args.dim, bias=False)
        self.erase_gate = nn.Linear(args.dim, 1)
        self.write_gate = nn.Linear(args.dim, 1)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        B, S, D = query.shape
        q = self.read_query(query)
        k = self.read_key(self.memory).expand(B, -1, -1)
        v = self.read_val(self.memory).expand(B, -1, -1)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, v)

    def write(self, x: torch.Tensor, entropy: torch.Tensor):
        B, S, D = x.shape
        k_write = self.write_key(x)
        v_write = self.write_val(x)

        scores = torch.bmm(k_write, self.memory.expand(B, -1, -1).transpose(1, 2))
        attn = F.softmax(scores, dim=-1)
        
        update = torch.bmm(attn.transpose(1, 2), v_write).mean(0, keepdim=True)
        
        erase = torch.sigmoid(self.erase_gate(self.memory))
        write_strength = torch.sigmoid(self.write_gate(update))
        importance = torch.sigmoid(entropy.mean()).item()
        
        new_mem = self.memory * (1 - erase * 0.05) + update * write_strength * importance * 0.1
        self.memory.copy_(new_mem.detach())

class HierarchicalMemory(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.working_memory = GlobalMemoryBank(args)
        
        compressed_dim = args.dim // args.memory_compression
        self.episodic_slots = args.episodic_slots
        self.register_buffer(
            "episodic_memory", 
            torch.randn(1, self.episodic_slots, compressed_dim) * 0.02
        )
        
        self.semantic_memory = nn.Parameter(
            torch.randn(args.memory_slots, args.dim) * 0.02
        )
        
        self.consolidation_gate = nn.Linear(args.dim, 1)
        self.compressor = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, compressed_dim)
        )
        self.decompressor = nn.Sequential(
            nn.Linear(compressed_dim, args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, args.dim)
        )
        
        self.importance_scorer = nn.Linear(args.dim, 1)
        
    def consolidate(self, working_mem: torch.Tensor):
        importance = torch.sigmoid(self.importance_scorer(working_mem))
        threshold = importance.quantile(0.8)
        
        important = working_mem[importance.squeeze(-1) > threshold]
        if important.size(0) > 0:
            compressed = self.compressor(important)
            self.episodic_memory.data = (
                0.99 * self.episodic_memory + 
                0.01 * compressed.mean(0, keepdim=True)
            )
    
    def retrieve_contextual(self, query: torch.Tensor) -> torch.Tensor:
        B = query.size(0)
        working_out = self.working_memory.read(query)
        
        episodic_decompressed = self.decompressor(self.episodic_memory).expand(B, -1, -1)
        episodic_scores = torch.bmm(query, episodic_decompressed.transpose(1, 2)) / math.sqrt(query.size(-1))
        episodic_out = torch.bmm(F.softmax(episodic_scores, dim=-1), episodic_decompressed)
        
        semantic_expanded = self.semantic_memory.unsqueeze(0).expand(B, -1, -1)
        semantic_scores = torch.bmm(query, semantic_expanded.transpose(1, 2)) / math.sqrt(query.size(-1))
        semantic_out = torch.bmm(F.softmax(semantic_scores, dim=-1), semantic_expanded)
        
        return working_out + 0.5 * episodic_out + 0.3 * semantic_out

# ============= Enhanced SSM =============
class SelectiveSSM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.state_dim = 64
        self.dt_rank = math.ceil(dim / 16)

        self.norm = nn.LayerNorm(dim)
        self.x_proj = nn.Linear(dim, self.dt_rank + self.state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)
        
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.state_dim + 1).repeat(dim, 1).float())
        )
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_normed = self.norm(x)
        x_proj = self.x_proj(x_normed)
        delta, B_ssm, C_ssm = torch.split(x_proj, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())

        h = torch.zeros(B, D, self.state_dim, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt = delta[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt * A)
            
            bt = B_ssm[:, t, :].unsqueeze(1)
            xt = x_normed[:, t, :].unsqueeze(-1)
            dB_xt = (dt * bt) * xt
            
            h = dA * h + dB_xt
            
            ct = C_ssm[:, t, :].unsqueeze(-1)
            y = torch.bmm(h, ct).squeeze(-1)
            ys.append(y)

        y_stack = torch.stack(ys, dim=1)
        return self.out_proj(y_stack + x * self.D)

# ============= Grouped Query Attention =============
class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim
        
        self.q_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, args.max_seq_len)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        offset = 0
        if kv_cache is not None:
            offset = kv_cache[0].shape[2]

        cos, sin = self.rotary_emb(x, offset=offset)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)

        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), new_kv_cache

# ============= Mixture of Experts =============
class ExpertRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.expert_top_k
        self.dim = args.dim
        
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.dim, args.dim * 4, bias=False),
                nn.GELU(),
                nn.Linear(args.dim * 4, args.dim, bias=False)
            ) for _ in range(args.num_experts)
        ])
        
        self.register_buffer('expert_counts', torch.zeros(args.num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        gate_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        self.expert_counts += expert_mask.sum(dim=[0, 1, 2])
        
        outputs = torch.zeros_like(x)
        for i in range(self.top_k):
            for e in range(self.num_experts):
                mask = (top_k_indices[:, :, i] == e).unsqueeze(-1).float()
                if mask.sum() > 0:
                    gate_val = top_k_gates[:, :, i].unsqueeze(-1)
                    expert_out = self.experts[e](x)
                    outputs += mask * gate_val * expert_out
        
        load_balance_loss = self._compute_load_balance_loss()
        return outputs, load_balance_loss
    
    def _compute_load_balance_loss(self):
        if self.expert_counts.sum() == 0:
            return torch.tensor(0.0, device=self.expert_counts.device)
        counts_normalized = self.expert_counts / self.expert_counts.sum()
        target = torch.ones_like(counts_normalized) / self.num_experts
        return F.mse_loss(counts_normalized, target)

# ============= Hybrid Block =============
class HybridBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        
        self.attn = GroupedQueryAttention(args)
        self.ssm = SelectiveSSM(args.dim)
        self.conv = nn.Sequential(
            nn.Conv1d(args.dim, args.dim, 3, padding=1, groups=args.dim),
            nn.GELU(),
            nn.Conv1d(args.dim, args.dim, 1)
        )
        
        self.router_gate = nn.Linear(args.dim, 4, bias=False)
        
        self.norm1 = nn.LayerNorm(args.dim)
        self.moe = ExpertRouter(args)
        self.norm2 = nn.LayerNorm(args.dim)

    def forward(self, x: torch.Tensor, memory_bank: HierarchicalMemory, 
                goal_cond: Optional[torch.Tensor] = None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        res = x
        x = self.norm1(x)
        
        route_input = x if goal_cond is None else x + goal_cond
        weights = F.softmax(self.router_gate(route_input), dim=-1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(-1, keepdim=True)

        out_attn, new_kv_cache = self.attn(x, kv_cache=kv_cache)
        out_ssm = self.ssm(x)
        out_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out_mem = memory_bank.retrieve_contextual(x)

        experts = torch.stack([out_attn, out_ssm, out_conv, out_mem], dim=-1)
        mixed = (experts * weights.unsqueeze(2)).sum(-1)
        
        x = res + mixed
        moe_out, lb_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, entropy, lb_loss, new_kv_cache

# ============= World Model =============
class ImprovedWorldModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.action_space_size = args.action_space_size
        self.state_proj = nn.Linear(args.dim + args.action_space_size + args.env_state_dim, args.dim)
        self.dynamics = nn.GRUCell(args.dim, args.dim)
        self.state_decoder = nn.Linear(args.dim, args.env_state_dim)
        self.reward_head = nn.Linear(args.dim, 1)
        self.uncertainty_head = nn.Linear(args.dim, 1)

    def predict_step(self, h: torch.Tensor, action: torch.Tensor, env_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_onehot = F.one_hot(action, num_classes=action.size(-1) if action.dim() > 1 else self.action_space_size).float()
        combined = torch.cat([h, action_onehot, env_state], dim=-1)
        hidden = self.state_proj(combined)
        next_h = self.dynamics(hidden, h)
        next_state = self.state_decoder(next_h)
        reward = self.reward_head(next_h)
        return next_h, next_state, reward

# ============= Advanced Planning =============
class TreeSearchPlanner(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.horizon = args.planning_horizon
        self.num_simulations = args.num_simulations
        
        self.world_model = ImprovedWorldModel(args)
        self.value_net = nn.Sequential(
            nn.Linear(args.dim + args.goal_dim, args.dim),
            nn.LayerNorm(args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.dim // 2),
            nn.LayerNorm(args.dim // 2),
            nn.GELU(),
            nn.Linear(args.dim // 2, 1)
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(args.dim + args.goal_dim, args.dim),
            nn.LayerNorm(args.dim),
            nn.GELU(),
            nn.Linear(args.dim, args.action_space_size)
        )
        
        self.uncertainty_head = nn.Linear(args.dim, 1)
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor, env_state: torch.Tensor):
        best_actions = []
        best_value = float('-inf')
        
        for _ in range(self.num_simulations):
            actions, value = self._simulate_trajectory(state, goal, env_state)
            if value > best_value:
                best_value = value
                best_actions = actions
        
        first_action_logits = self.policy_net(torch.cat([state, goal.squeeze(1)], dim=-1))
        value = self.value_net(torch.cat([state, goal.squeeze(1)], dim=-1))
        
        return {
            "action_logits": first_action_logits,
            "value": value,
            "best_trajectory": best_actions
        }
    
    def _simulate_trajectory(self, state: torch.Tensor, goal: torch.Tensor, env_state: torch.Tensor):
        trajectory = []
        h = state
        curr_env = env_state
        total_value = 0
        
        for t in range(self.horizon):
            combined = torch.cat([h, goal.squeeze(1)], dim=-1)
            
            action_logits = self.policy_net(combined)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            
            h, curr_env, reward = self.world_model.predict_step(h, action, curr_env)
            
            value = self.value_net(torch.cat([h, goal.squeeze(1)], dim=-1))
            total_value += (0.99 ** t) * (value.mean() + reward.mean())
            trajectory.append(action)
            
        return trajectory, total_value.item()

# ============= Internal Monologue =============
class InternalMonologue(nn.Module):
    def __init__(self, dim: int, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.gru = nn.GRUCell(dim, dim)
        self.thought_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.reflection_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, goal_cond: Optional[torch.Tensor], prev_thought: Optional[torch.Tensor]):
        B, L, D = x.shape
        pooled = x.mean(dim=1)
        if goal_cond is not None:
            pooled = pooled + goal_cond.squeeze(1)
        
        h = prev_thought if prev_thought is not None else torch.zeros_like(pooled)
        
        thoughts = []
        for _ in range(self.steps):
            h_next = self.gru(pooled, h)
            h = self.thought_proj(torch.cat([pooled, h_next], dim=-1))
            thoughts.append(h)
        
        reflection = self.reflection_head(h)
        return h, thoughts, reflection

# ============= Main Model =============
class UniversalIntelligenceModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.text_enc = nn.Embedding(args.vocab_size, args.dim)
        self.vision_enc = VisionEncoder(args)
        self.audio_enc = AudioEncoder(args)
        self.goal_enc = GoalEncoder(args)
        self.tagger = ModalityEmbeddings(args.dim)
        
        self.memory_bank = HierarchicalMemory(args)
        self.layers = nn.ModuleList([HybridBlock(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.dim)
        
        self.monologue_core = InternalMonologue(args.dim)
        self.agency = TreeSearchPlanner(args)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        self.prev_thought = None
        self.gradient_checkpointing = args.gradient_checkpointing

    def forward(self, text: Optional[torch.Tensor] = None, img: Optional[torch.Tensor] = None, audio: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None, env_state: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        tokens = []
        B = 0

        if text is not None:
            B = text.size(0)
            tokens.append(self.tagger(self.text_enc(text), 0))
        if img is not None:
            B = img.size(0)
            tokens.append(self.tagger(self.vision_enc(img), 1))
        if audio is not None:
            B = audio.size(0)
            tokens.append(self.tagger(self.audio_enc(audio), 2))
            
        goal_cond = self.goal_enc(goal) if goal is not None else None
        x = torch.cat(tokens, dim=1)
        
        total_entropy = 0
        total_lb_loss = 0
        
        present_key_values = []

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                x, ent, lb, present_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, self.memory_bank, goal_cond, layer_past, use_reentrant=False
                )
            else:
                x, ent, lb, present_kv = layer(x, self.memory_bank, goal_cond, kv_cache=layer_past)

            if present_kv is not None:
                present_key_values.append(present_kv)

            total_entropy += ent.mean()
            total_lb_loss += lb
            
        x = self.norm(x)
        logits = self.lm_head(x)
        avg_ent = total_entropy / len(self.layers)
        avg_lb = total_lb_loss / len(self.layers)

        thought, thought_trace, reflection = self.monologue_core(x, goal_cond, self.prev_thought)
        self.prev_thought = thought.detach()
        
        self.memory_bank.consolidate(x)
        
        if env_state is None:
            env_state = torch.zeros(B, self.args.env_state_dim, device=x.device)
        
        last_state = x[:, -1, :] + thought
        agency_results = self.agency(
            last_state,
            goal_cond if goal_cond is not None else torch.zeros(B, 1, self.args.goal_dim, device=x.device),
            env_state
        )
        
        return {
            "thought": thought,
            "thought_trace": thought_trace,
            "reflection": reflection,
            "agency": agency_results,
            "entropy": avg_ent,
            "load_balance_loss": avg_lb,
            "hidden_states": x,
            "logits": logits,
            "past_key_values": present_key_values
        }

# ============= Training Utilities =============
class TrainingConfig:
    @staticmethod
    def get_optimizer(model, lr=1e-4, weight_decay=0.1):
        param_groups = []
        num_layers = len(model.layers)
        
        for layer_id, layer in enumerate(model.layers):
            decay_rate = 0.9 ** (num_layers - layer_id)
            param_groups.append({
                'params': layer.parameters(),
                'lr': lr * decay_rate,
                'weight_decay': weight_decay
            })
        
        param_groups.append({
            'params': [p for n, p in model.named_parameters() if 'layers' not in n],
            'lr': lr,
            'weight_decay': weight_decay
        })
        
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    @staticmethod
    def get_scheduler(optimizer, warmup_steps=2000, total_steps=100000):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def compute_losses(model_output, targets):
        losses = {}
        
        # Entropy regularization
        losses['entropy_reg'] = -0.01 * model_output['entropy'].mean()
        
        # Load balancing for MoE
        losses['load_balance'] = 0.01 * model_output['load_balance_loss']
        
        # Value prediction loss
        if 'returns' in targets:
            losses['value'] = F.mse_loss(
                model_output['agency']['value'].squeeze(-1),
                targets['returns']
            )
        
        # Action prediction loss
        if 'actions' in targets:
            losses['action'] = F.cross_entropy(
                model_output['agency']['action_logits'],
                targets['actions']
            )
        
        # Thought consistency (temporal smoothness)
        if 'prev_thought' in targets:
            losses['thought_consistency'] = 0.1 * F.mse_loss(
                model_output['thought'],
                targets['prev_thought']
            )
        
        # Language modeling loss
        if 'next_tokens' in targets and 'logits' in model_output:
            losses['lm'] = F.cross_entropy(
                model_output['logits'].view(-1, model_output['logits'].size(-1)),
                targets['next_tokens'].view(-1)
            )
        
        return losses

# ============= Data Loading =============
class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, max_seq_len: int = 1024, vocab_size: int = 128000):
        super().__init__()
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.samples = self._load_data()
    
    def _load_data(self):
        # Placeholder - implement based on your data format
        # Expected format: list of dicts with keys: text, image, audio, goal, actions, rewards
        samples = []
        # Example structure:
        # samples.append({
        #     'text': torch.randint(0, self.vocab_size, (seq_len,)),
        #     'image': torch.randn(3, 224, 224),
        #     'audio': torch.randn(audio_len, 128),
        #     'goal': torch.randint(0, self.vocab_size, (5,)),
        #     'actions': torch.randint(0, 100, (horizon,)),
        #     'rewards': torch.randn(horizon,)
        # })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class MultiModalCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        # Collate text
        text_batch = None
        if 'text' in batch[0]:
            max_len = max(item['text'].size(0) for item in batch)
            text_batch = torch.stack([
                F.pad(item['text'], (0, max_len - item['text'].size(0)), value=self.pad_token_id)
                for item in batch
            ])
        
        # Collate images
        img_batch = None
        if 'image' in batch[0]:
            img_batch = torch.stack([item['image'] for item in batch])
        
        # Collate audio
        audio_batch = None
        if 'audio' in batch[0]:
            max_len = max(item['audio'].size(0) for item in batch)
            audio_batch = torch.stack([
                F.pad(item['audio'], (0, 0, 0, max_len - item['audio'].size(0)))
                for item in batch
            ])
        
        # Collate goals
        goal_batch = None
        if 'goal' in batch[0]:
            goal_batch = torch.stack([item['goal'] for item in batch])
        
        # Collate targets
        targets = {}
        if 'actions' in batch[0]:
            targets['actions'] = torch.stack([item['actions'] for item in batch])
        if 'rewards' in batch[0]:
            targets['returns'] = torch.stack([item['rewards'] for item in batch])
        
        return {
            'text': text_batch,
            'img': img_batch,
            'audio': audio_batch,
            'goal': goal_batch,
            'targets': targets
        }

# ============= Training Loop =============
class Trainer:
    def __init__(
        self,
        model: UniversalIntelligenceModel,
        train_dataset: MultiModalDataset,
        val_dataset: Optional[MultiModalDataset] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=MultiModalCollator(),
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=MultiModalCollator(),
                pin_memory=True
            )
        
        # Optimizer and scheduler
        self.optimizer = TrainingConfig.get_optimizer(model, lr=1e-4)
        self.scheduler = TrainingConfig.get_scheduler(
            self.optimizer,
            warmup_steps=2000,
            total_steps=len(self.train_loader) * 100  # Assuming 100 epochs
        )
        
        # Mixed precision training
        self.use_amp = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Metrics tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        losses_dict = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            text = batch['text'].to(self.device) if batch['text'] is not None else None
            img = batch['img'].to(self.device) if batch['img'] is not None else None
            audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
            goal = batch['goal'].to(self.device) if batch['goal'] is not None else None
            
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(text=text, img=img, audio=audio, goal=goal)
                losses = TrainingConfig.compute_losses(outputs, targets)
                loss = sum(losses.values()) / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in losses.items():
                if k not in losses_dict:
                    losses_dict[k] = 0
                losses_dict[k] += v.item()
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / self.log_interval
                print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")
                for k, v in losses_dict.items():
                    print(f"  {k}: {v / self.log_interval:.4f}")
                total_loss = 0
                losses_dict = {}
            
            # Validation
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            text = batch['text'].to(self.device) if batch['text'] is not None else None
            img = batch['img'].to(self.device) if batch['img'] is not None else None
            audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
            goal = batch['goal'].to(self.device) if batch['goal'] is not None else None
            
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            outputs = self.model(text=text, img=img, audio=audio, goal=goal)
            losses = TrainingConfig.compute_losses(outputs, targets)
            loss = sum(losses.values())
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            self.train_epoch()
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = f"{self.checkpoint_dir}/{filename}"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        path = f"{self.checkpoint_dir}/{filename}"
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")

# ============= Inference Optimization =============
class InferenceEngine:
    def __init__(
        self,
        model: UniversalIntelligenceModel,
        device: str = 'cuda',
        compile_model: bool = True,
        use_kv_cache: bool = True
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_kv_cache = use_kv_cache
        
        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled successfully!")
            except:
                print("Model compilation not available, using eager mode")
        
        # Initialize KV cache
        self.kv_cache = {}
        
    @torch.no_grad()
    def generate(
        self,
        text: Optional[torch.Tensor] = None,
        img: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Generate tokens autoregressively with optimizations
        """
        self.model.eval()
        
        # Initial forward pass setup
        kv_cache = None
        generated_tokens = []
        
        for i in range(max_new_tokens):
            with torch.cuda.amp.autocast():
                # If first step, use full text. Else use just the last generated token.
                if i == 0:
                    current_text = text
                    current_img = img
                    current_audio = audio
                    current_goal = goal
                else:
                    current_text = generated_tokens[-1]
                    current_img = None
                    current_audio = None
                    current_goal = goal # Keep goal context if needed by model architecture

                outputs = self.model(
                    text=current_text,
                    img=current_img,
                    audio=current_audio,
                    goal=current_goal,
                    past_key_values=kv_cache
                )

            kv_cache = outputs['past_key_values']
            logits = outputs['logits'][:, -1, :]  # Last position

            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token)
            
        return generated_tokens
    
    @torch.no_grad()
    def batch_inference(
        self,
        batch_text: List[torch.Tensor],
        batch_img: Optional[List[torch.Tensor]] = None,
        batch_goal: Optional[List[torch.Tensor]] = None
    ):
        """
        Optimized batch inference
        """
        # Pad sequences to same length
        max_len = max(t.size(0) for t in batch_text)
        padded_text = torch.stack([
            F.pad(t, (0, max_len - t.size(0)))
            for t in batch_text
        ]).to(self.device)
        
        batch_imgs = None
        if batch_img is not None:
            batch_imgs = torch.stack(batch_img).to(self.device)
        
        batch_goals = None
        if batch_goal is not None:
            batch_goals = torch.stack(batch_goal).to(self.device)
        
        with torch.cuda.amp.autocast():
            outputs = self.model(
                text=padded_text,
                img=batch_imgs,
                goal=batch_goals
            )
        
        return outputs
    
    @torch.no_grad()
    def plan_action(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        env_state: torch.Tensor,
        num_candidates: int = 5
    ):
        """
        Optimized action planning with multiple candidate evaluation
        """
        with torch.cuda.amp.autocast():
            # Expand state for parallel candidate evaluation
            state_expanded = state.repeat(num_candidates, 1)
            goal_expanded = goal.repeat(num_candidates, 1, 1)
            env_expanded = env_state.repeat(num_candidates, 1)
            
            agency_output = self.model.agency(state_expanded, goal_expanded, env_expanded)
            
            # Select best action based on value
            best_idx = agency_output['value'].argmax()
            best_action = agency_output['action_logits'][best_idx]
            
        return best_action, agency_output['value'][best_idx]
    
    def benchmark(self, num_iterations: int = 100):
        """
        Benchmark inference speed
        """
        import time
        
        # Dummy inputs
        dummy_text = torch.randint(0, 1000, (1, 128)).to(self.device)
        dummy_goal = torch.randint(0, 1000, (1, 5)).to(self.device)
        
        # Warmup
        for _ in range(10):
            self.model(text=dummy_text, goal=dummy_goal)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iterations):
            self.model(text=dummy_text, goal=dummy_goal)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iterations
        throughput = 1.0 / avg_time
        
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        return avg_time, throughput

# ============= Example Usage =============
if __name__ == "__main__":
    # Initialize model
    args = ModelArgs()
    model = UniversalIntelligenceModel(args)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Example: Training
    print("\n=== Training Example ===")
    # train_dataset = MultiModalDataset('./data/train')
    # val_dataset = MultiModalDataset('./data/val')
    # trainer = Trainer(model, train_dataset, val_dataset)
    # trainer.train(num_epochs=10)
    
    # Example: Inference
    print("\n=== Inference Example ===")
    inference_engine = InferenceEngine(model, compile_model=False)
    
    # Mock inputs
    batch_size = 2
    dummy_text = torch.randint(0, 1000, (batch_size, 10))
    dummy_goal = torch.randint(0, 1000, (batch_size, 5))
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    
    # Run inference
    outputs = inference_engine.batch_inference(
        [dummy_text[i] for i in range(batch_size)],
        [dummy_img[i] for i in range(batch_size)],
        [dummy_goal[i] for i in range(batch_size)]
    )
    
    print(f"Thought shape: {outputs['thought'].shape}")
    print(f"Action logits shape: {outputs['agency']['action_logits'].shape}")
    
    # Benchmark
    print("\n=== Benchmarking ===")
    inference_engine.benchmark(num_iterations=50)
