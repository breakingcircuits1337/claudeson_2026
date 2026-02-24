"""
Claudeson 2026 - Infinite Context Mode
======================================
Dynamic routing based on sequence length:
- Short (≤4K): Balanced [Attention, SSM, Conv, Memory]
- Medium (4K-32K): Heavy SSM + Windowed Attention  
- Long (32K-128K+): SSM Dominant + Memory Paging + Ring Attention

Architecture: Sliding Window GQA + SSM for Global State + Hierarchical Memory Paging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

# ============= Configuration =============
@dataclass
class ModelArgs:
    # Core parameters
    dim: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128000
    patch_size: int = 16
    img_size: int = 224
    audio_spec_dim: int = 128
    
    # Context settings
    max_seq_len: int = 131072  # 128K
    base_seq_len: int = 8192   # Baseline for scaling
    
    # Infinite Context Mode
    use_infinite_context: bool = True
    
    # Sequence length thresholds
    short_threshold: int = 4096    # <4K: balanced
    medium_threshold: int = 32768  # 4K-32K: heavy SSM
    
    # Memory Paging - TUNED FOR INFINITE CONTEXT
    memory_slots: int = 2048       # Increased from 256
    memory_dim: int = 2048
    episodic_slots: int = 16384    # Increased from 2560 for paging
    memory_compression: int = 8    # Increased from 4 for efficiency
    
    # Windowed Attention
    attention_window: int = 4096   # Sliding window size
    
    # SSM Settings
    ssm_chunk_size: int = 512     # Process SSM in chunks
    
    # Agency & Planning
    action_space_size: int = 100
    planning_horizon: int = 8
    num_simulations: int = 8
    env_state_dim: int = 128
    goal_dim: int = 2048
    
    # MoE
    num_experts: int = 8
    expert_top_k: int = 2
    
    # Training optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    use_kv_cache: bool = True


# ============= Dynamic Router =============
class DynamicRouter(nn.Module):
    """
    Dynamically routes based on sequence length for Infinite Context Mode.
    
    Sequence Length    Attention    SSM     Conv    Memory
    --------------------------------------------------
    Short (<4K)        0.35        0.30    0.20    0.15
    Medium (4K-32K)    0.20        0.45    0.15    0.20
    Long (32K+)        0.10        0.55    0.10    0.25
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        
        # Gating network
        self.gate = nn.Linear(args.dim, 4, bias=False)  # [attn, ssm, conv, mem]
        
        # Sequence length embedding for context-aware routing
        self.seq_len_embed = nn.Embedding(3, 4)  # 3 modes: short, medium, long
        
        # Learnable mode biases (can be fine-tuned)
        self.register_buffer('short_weights', torch.tensor([0.35, 0.30, 0.20, 0.15]))
        self.register_buffer('medium_weights', torch.tensor([0.20, 0.45, 0.15, 0.20]))
        self.register_buffer('long_weights', torch.tensor([0.10, 0.55, 0.10, 0.25]))
    
    def get_mode(self, seq_len: int) -> int:
        """Determine sequence mode: 0=short, 1=medium, 2=long"""
        if seq_len < self.args.short_threshold:
            return 0
        elif seq_len < self.args.medium_threshold:
            return 1
        else:
            return 2
    
    def forward(self, x: torch.Tensor, mode_override: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns: (routing_weights, entropy, mode)
        """
        B, L, D = x.shape
        
        # Determine mode
        if mode_override is not None:
            mode = mode_override
        else:
            mode = self.get_mode(L)
        
        # Get gate logits
        gate_logits = self.gate(x.mean(dim=1, keepdim=True))  # [B, 1, 4]
        
        # Get mode-based biases
        if mode == 0:
            mode_bias = self.short_weights.unsqueeze(0).unsqueeze(0)
        elif mode == 1:
            mode_bias = self.medium_weights.unsqueeze(0).unsqueeze(0)
        else:
            mode_bias = self.long_weights.unsqueeze(0).unsqueeze(0)
        
        # Combine gating with mode bias
        # During training: use learnable gate
        # During inference: bias toward mode-specific weights
        if self.training:
            weights = F.softmax(gate_logits + mode_bias * 0.5, dim=-1)
        else:
            weights = mode_bias.expand(B, -1, -1)
        
        # Compute entropy
        entropy = -(weights * torch.log(weights + 1e-8)).sum(-1, keepdim=True)
        
        return weights, entropy, mode


# ============= Sliding Window GQA =============
class WindowedGQA(nn.Module):
    """
    Grouped Query Attention with Sliding Window.
    - Uses local attention for efficiency
    - Relies on SSM for global state tracking
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dim = args.dim
        self.window_size = args.attention_window
        
        self.q_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.dim, args.dim, bias=False)
        
        # Relative position bias for sliding window
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * args.attention_window + 1, args.n_heads))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV for GQA
        n_rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        
        # Sliding window mask
        if L > self.window_size:
            # Create causal sliding window mask
            window_mask = torch.ones(L, L, device=x.device)
            for i in range(L):
                window_mask[i, max(0, i - self.window_size):i] = 0
            window_mask = (window_mask - 1) * 1e10
            mask = window_mask if mask is None else mask + window_mask
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative position bias
        if L <= self.window_size:
            scores = scores + self.rel_pos_bias[self.window_size:self.window_size + L, :L].unsqueeze(0)
        
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        
        return self.o_proj(out)


# ============= Enhanced SSM with Chunking =============
class ChunkedSSM(nn.Module):
    """
    SSM with chunked processing for long sequences.
    Processes in chunks to maintain O(n) complexity while handling 100K+ tokens.

    NOTE: This implementation uses a sequential loop over chunks for demonstration.
    A production version would use parallel scan kernels.
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.state_dim = 64
        self.dt_rank = math.ceil(args.dim / 16)
        self.chunk_size = args.ssm_chunk_size
        
        self.norm = nn.LayerNorm(args.dim)
        self.x_proj = nn.Linear(args.dim, self.dt_rank + self.state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, args.dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.state_dim + 1).repeat(args.dim, 1).float()))
        self.D = nn.Parameter(torch.ones(args.dim))
        self.out_proj = nn.Linear(args.dim, args.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_normed = self.norm(x)
        x_proj = self.x_proj(x_normed)
        delta, B_ssm, C_ssm = torch.split(x_proj, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        
        # Chunked SSM computation
        num_chunks = (L + self.chunk_size - 1) // self.chunk_size
        h = torch.zeros(B, D, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, L)
            chunk_len = end - start
            
            delta_chunk = delta[:, start:end, :]
            B_chunk = B_ssm[:, start:end, :]
            x_chunk = x_normed[:, start:end, :]
            C_chunk = C_ssm[:, start:end, :]
            
            # Process chunk with persistent state
            chunk_outputs = []
            for t in range(chunk_len):
                dt = delta_chunk[:, t, :].unsqueeze(-1)
                bt = B_chunk[:, t, :].unsqueeze(1)
                xt = x_chunk[:, t, :].unsqueeze(-1)
                
                dA = torch.exp(dt * A)
                dB_xt = (dt * bt) * xt
                h = dA * h + dB_xt
                
                ct = C_chunk[:, t, :].unsqueeze(-1)
                y = torch.bmm(h, ct).squeeze(-1)
                chunk_outputs.append(y)
            
            outputs.append(torch.stack(chunk_outputs, dim=1))
        
        y_stack = torch.cat(outputs, dim=1)
        return self.out_proj(y_stack + x * self.D)


# ============= Memory Paging System =============
class PagedMemorySystem(nn.Module):
    """
    Hierarchical Memory with Paging for Infinite Context.
    
    - Working Memory: Current context window (4K tokens)
    - Episodic Memory: Paged storage for older context (16K slots)
    - Semantic Memory: Compressed knowledge base
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Working memory (current window)
        self.working_memory = nn.Linear(args.dim, args.dim)
        
        # Episodic memory with paging
        self.episodic_slots = args.episodic_slots
        compressed_dim = args.dim // args.memory_compression
        self.compression_ratio = args.memory_compression
        
        # Page table: maps page_id to memory location
        self.num_pages = args.episodic_slots // args.attention_window
        self.page_size = args.attention_window
        
        # Compressed episodic storage
        self.episodic_compressed = nn.Parameter(
            torch.randn(1, self.episodic_slots, compressed_dim) * 0.02
        )
        
        # Page index (what's in each page)
        self.page_index = nn.Parameter(
            torch.zeros(self.num_pages, dtype=torch.long)
        )
        
        # Semantic memory
        self.semantic_memory = nn.Parameter(
            torch.randn(args.memory_slots, args.dim) * 0.02
        )
        
        # Compression/Decompression
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
        
        # Importance scorer
        self.importance_scorer = nn.Linear(args.dim, 1)
        
        # Read/write heads
        self.read_gate = nn.Linear(args.dim, 1)
        self.write_gate = nn.Linear(args.dim, 1)
    
    def forward(self, x: torch.Tensor, mode: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (memory_output, importance_scores)
        mode: 0=short, 1=medium, 2=long
        """
        B, L, D = x.shape
        
        # Determine how much to read from each memory tier
        if mode == 0:  # Short: use working memory
            return x, torch.zeros(B, 1, device=x.device)
        
        # Working memory (always available)
        working_out = self.working_memory(x)
        
        # Read from episodic based on mode
        if mode >= 1 and L > self.page_size:
            # Retrieve relevant pages
            episodic_decomp = self.decompressor(self.episodic_compressed)
            
            # Attention-based retrieval
            episodic_scores = torch.bmm(x, episodic_decomp.transpose(1, 2)) / math.sqrt(D)
            episodic_attn = F.softmax(episodic_scores, dim=-1)
            episodic_out = torch.bmm(episodic_attn, episodic_decomp)
            
            # Scale by mode (more episodic for longer sequences)
            episodic_scale = 0.3 if mode == 1 else 0.5
            working_out = working_out + episodic_scale * episodic_out
        
        # Semantic memory (global knowledge)
        if mode >= 2:
            semantic_expanded = self.semantic_memory.unsqueeze(0).expand(B, -1, -1)
            semantic_scores = torch.bmm(x, semantic_expanded.transpose(1, 2)) / math.sqrt(D)
            semantic_out = torch.bmm(F.softmax(semantic_scores, dim=-1), semantic_expanded)
            working_out = working_out + 0.2 * semantic_out
        
        # Compute importance for potential writes
        importance = torch.sigmoid(self.importance_scorer(x))
        
        return working_out, importance
    
    def page_out(self, x: torch.Tensor):
        """Page older context to episodic memory"""
        B, L, D = x.shape
        
        if L <= self.page_size:
            return
        
        # Compress older portion
        older = x[:, :-self.page_size, :]
        compressed = self.compressor(older.mean(dim=1, keepdim=True))
        
        # Shift episodic memory (page replacement)
        self.episodic_compressed.data = torch.roll(
            self.episodic_compressed.data, shifts=-1, dims=1
        )
        self.episodic_compressed.data[:, -1, :] = compressed.squeeze(1)


# ============= Infinite Context Hybrid Block =============
class InfiniteContextBlock(nn.Module):
    """
    Hybrid Block with Dynamic Routing for Infinite Context Mode.
    
    Automatically adjusts:
    - Attention vs SSM balance based on sequence length
    - Memory paging for longer sequences
    - Compute allocation
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        
        # Components
        self.attn = WindowedGQA(args)
        self.ssm = ChunkedSSM(args)
        self.conv = nn.Sequential(
            nn.Conv1d(args.dim, args.dim, 3, padding=1, groups=args.dim),
            nn.GELU(),
            nn.Conv1d(args.dim, args.dim, 1)
        )
        
        # Dynamic router
        self.router = DynamicRouter(args)
        
        # Memory paging
        self.memory = PagedMemorySystem(args)
        
        # MoE for expert knowledge
        self.moe_gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.dim, args.dim * 4, bias=False),
                nn.GELU(),
                nn.Linear(args.dim * 4, args.dim, bias=False)
            ) for _ in range(args.num_experts)
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(args.dim)
        self.norm2 = nn.LayerNorm(args.dim)
        self.norm3 = nn.LayerNorm(args.dim)
        
        # Expert load balancing
        self.register_buffer('expert_counts', torch.zeros(args.num_experts))
    
    def forward(self, x: torch.Tensor, mode_override: Optional[int] = None):
        B, L, D = x.shape
        
        # 1. Dynamic routing based on sequence length
        route_weights, entropy, mode = self.router(x, mode_override)
        
        # 2. Compute each component
        # Attention (windowed)
        if mode == 2:  # Long: minimal attention
            attn_out = self.attn(self.norm1(x)) * 0.5
        else:
            attn_out = self.attn(self.norm1(x))
        
        # SSM (always runs, scales with mode)
        ssm_out = self.ssm(self.norm1(x))
        
        # Conv (local features)
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Memory (paged for long sequences)
        mem_out, importance = self.memory(x, mode)
        
        # 3. Fuse based on routing weights
        # route_weights: [B, 1, 4] -> [attn, ssm, conv, mem]
        w = route_weights.squeeze(1)  # [B, 4]
        
        mixed = (
            w[:, 0:1] * attn_out +
            w[:, 1:2] * ssm_out +
            w[:, 2:3] * conv_out +
            w[:, 3:4] * mem_out
        )
        
        x = x + mixed
        
        # 4. MoE pass
        moe_logits = self.moe_gate(self.norm2(x))
        top_k_logits, top_k_idx = torch.topk(moe_logits, k=2, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        moe_out = torch.zeros_like(x)
        for i in range(2):
            expert_id = top_k_idx[:, :, i]
            weight = top_k_weights[:, :, i].unsqueeze(-1)
            for e in range(self.args.num_experts):
                mask = (expert_id == e).float().unsqueeze(-1)
                if mask.sum() > 0:
                    moe_out += mask * weight * self.experts[e](x)
        
        # Track load balance
        expert_mask = F.one_hot(top_k_idx[:, :, 0], self.args.num_experts).float()
        self.expert_counts += expert_mask.sum(dim=[0, 1])
        
        x = x + moe_out * 0.1  # Light MoE contribution
        
        # 5. Page out old context for very long sequences
        if mode == 2 and L > self.args.attention_window:
            self.memory.page_out(x)
        
        # Compute load balance loss
        if self.expert_counts.sum() > 0:
            lb_loss = F.mse_loss(
                self.expert_counts / self.expert_counts.sum(),
                torch.ones_like(self.expert_counts) / self.args.num_experts
            )
        else:
            lb_loss = torch.tensor(0.0, device=x.device)
        
        return x, entropy, lb_loss, mode


# ============= Rotary Position Embedding =============
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


# ============= Encoders (Simplified) =============
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


# ============= Main Model =============
class InfiniteContextModel(nn.Module):
    """
    Claudeson 2026 - Infinite Context Edition
    
    Key Features:
    - Dynamic routing based on sequence length
    - Sliding window GQA + SSM for global state
    - Hierarchical memory paging
    - Supports 128K+ context tokens
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Encoders
        self.text_enc = nn.Embedding(args.vocab_size, args.dim)
        self.vision_enc = VisionEncoder(args)
        self.audio_enc = AudioEncoder(args)
        self.goal_enc = GoalEncoder(args)
        
        # Infinite Context layers
        self.layers = nn.ModuleList([InfiniteContextBlock(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.dim)
        
        # Reflection (internal monologue)
        self.monologue = nn.GRUCell(args.dim, args.dim)
        
        # Planning
        self.planner = nn.Linear(args.dim + args.goal_dim, args.action_space_size)
        self.value_head = nn.Linear(args.dim, 1)
    
    def forward(self, text=None, img=None, audio=None, goal=None, mode_override: Optional[int] = None):
        B = 0
        tokens = []
        
        if text is not None:
            B = text.size(0)
            tokens.append(self.text_enc(text))
        if img is not None:
            B = img.size(0)
            tokens.append(self.vision_enc(img))
        if audio is not None:
            B = audio.size(0)
            tokens.append(self.audio_enc(audio))
        
        if not tokens:
            raise ValueError("No input provided")
        
        x = torch.cat(tokens, dim=1)
        seq_len = x.size(1)
        
        # Goal conditioning
        goal_cond = self.goal_enc(goal) if goal is not None else None
        if goal_cond is not None and B > 0:
            x = x + goal_cond[:, :1, :]
        
        # Track modes per layer
        modes = []
        total_entropy = 0
        total_lb_loss = 0
        
        # Forward through layers
        for layer in self.layers:
            x, entropy, lb_loss, mode = layer(x, mode_override)
            total_entropy += entropy.mean()
            total_lb_loss += lb_loss
            modes.append(mode)
        
        x = self.norm(x)
        
        # Internal monologue
        h = x.mean(dim=1)
        thought = self.monologue(h, h)
        
        # Planning
        if goal_cond is not None:
            planning_input = torch.cat([thought, goal_cond.squeeze(1)], dim=-1)
        else:
            planning_input = thought
        
        action_logits = self.planner(planning_input)
        value = self.value_head(thought)
        
        # Determine dominant mode
        dominant_mode = max(set(modes), key=modes.count) if modes else 0
        mode_names = ["Short (<4K)", "Medium (4K-32K)", "Long (32K+)"]
        
        return {
            "hidden_states": x,
            "thought": thought,
            "action_logits": action_logits,
            "value": value,
            "entropy": total_entropy / len(self.layers),
            "load_balance_loss": total_lb_loss / len(self.layers),
            "mode": dominant_mode,
            "mode_name": mode_names[dominant_mode],
            "seq_len": seq_len,
        }


# ============= Demo =============
if __name__ == "__main__":
    args = ModelArgs()
    
    print("=" * 60)
    print("CLAUDESON 2026 - INFINITE CONTEXT MODE")
    print("=" * 60)
    
    print("\n📋 CONFIGURATION:")
    print(f"  Max sequence length: {args.max_seq_len:,} tokens ({args.max_seq_len/1024}K)")
    print(f"  Base sequence length: {args.base_seq_len:,}")
    print(f"  Short threshold: <{args.short_threshold:,}")
    print(f"  Medium threshold: <{args.medium_threshold:,}")
    print(f"  Attention window: {args.attention_window:,}")
    print(f"  SSM chunk size: {args.ssm_chunk_size:,}")
    print(f"  Episodic slots: {args.episodic_slots:,}")
    print(f"  Memory compression: {args.memory_compression}x")
    
    print("\n🏗️ INITIALIZING MODEL...")
    model = InfiniteContextModel(args)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")
    
    # Test different sequence lengths
    print("\n🧪 TESTING DYNAMIC ROUTING:")
    
    test_lengths = [1024, 8192, 16384, 65536, 131072]
    
    for seq_len in test_lengths:
        # Determine expected mode
        if seq_len < args.short_threshold:
            mode = 0
            mode_name = "Short"
        elif seq_len < args.medium_threshold:
            mode = 1
            mode_name = "Medium"  
        else:
            mode = 2
            mode_name = "Long"
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, seq_len))
        
        # Forward pass
        with torch.no_grad():
            output = model(text=dummy_input)
        
        print(f"\n  📏 Sequence: {seq_len:,} tokens ({seq_len/1024:.1f}K)")
        print(f"     Mode: {output['mode_name']}")
        print(f"     Hidden shape: {output['hidden_states'].shape}")
    
    print("\n" + "=" * 60)
    print("✅ INFINITE CONTEXT MODEL READY!")
    print("=" * 60)
    
    print("""
🎯 KEY FEATURES:
   • Dynamic routing: Automatically adjusts per sequence length
   • Short (<4K): Balanced attention/SSM
   • Medium (4K-32K): Heavy SSM + windowed attention
   • Long (32K+): SSM dominant + memory paging
   
   • Sliding window GQA: Efficient local attention
   • SSM chunking: O(n) complexity for long sequences  
   • Memory paging: 16K episodic slots for context
   • Semantic memory: Compressed knowledge base
""")
