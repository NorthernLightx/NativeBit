"""LLaMA-style causal transformer with NativeBit quantization.

Architecture: RoPE, RMSNorm, SwiGLU, value embeddings, logit soft-capping.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NativeBitLinear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(dim: int, max_len: int, theta: float = 10000.0) -> tuple:
    """Precompute cos/sin tables for Rotary Position Embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim/2)
    cos = freqs.cos()  # (max_len, dim/2)
    sin = freqs.sin()  # (max_len, dim/2)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Apply rotary embeddings to q and k tensors.

    Args:
        q, k: (B, n_heads, T, head_dim)
        cos, sin: (max_len, head_dim/2), will be sliced to T
    """
    T = q.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Split into pairs and rotate
    def rotate(x):
        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return rotated

    q_rot = q * cos.repeat(1, 1, 1, 2) + rotate(q) * sin.repeat(1, 1, 1, 2)
    k_rot = k * cos.repeat(1, 1, 1, 2) + rotate(k) * sin.repeat(1, 1, 1, 2)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional value embeddings."""

    def __init__(self, n_embd: int, n_head: int, context_len: int,
                 block_size: int = 64, n_entries: int = 8, use_nativebit: bool = True,
                 value_embed: nn.Embedding | None = None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        Linear = _make_linear(use_nativebit, block_size, n_entries)
        self.qkv = Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = Linear(n_embd, n_embd, bias=False)

        # Value embedding: float bypass for quantization-degraded attention.
        # Per-head gating: sigmoid(gate(x[:, :gate_channels])) * 2 → scale [0, 2]
        self.value_embed = value_embed
        if value_embed is not None:
            self.ve_gate_channels = min(32, n_embd)
            self.ve_gate = nn.Linear(self.ve_gate_channels, n_head, bias=False)

        # Precompute RoPE tables
        cos, sin = _precompute_rope(self.head_dim, context_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Value embedding: inject clean float signal into quantized values
        if self.value_embed is not None and idx is not None:
            ve = self.value_embed(idx)  # (B, T, n_embd) — float, not quantized
            ve = ve.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            # Input-dependent gate per head: sigmoid * 2 → neutral at init (gate=0 → scale=1)
            gate = 2.0 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            gate = gate.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
            v = v + gate * ve

        # Apply RoPE to Q and K
        q, k = _apply_rope(q, k, self.rope_cos, self.rope_sin)

        # Flash/fused attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: (Swish(x @ W_gate) * (x @ W_up)) @ W_down."""

    def __init__(self, n_embd: int, ffn_hidden: int,
                 block_size: int = 64, n_entries: int = 8, use_nativebit: bool = True):
        super().__init__()
        Linear = _make_linear(use_nativebit, block_size, n_entries)
        self.w_gate = Linear(n_embd, ffn_hidden, bias=False)
        self.w_up = Linear(n_embd, ffn_hidden, bias=False)
        self.w_down = Linear(ffn_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, ffn_hidden: int, context_len: int,
                 block_size: int = 64, n_entries: int = 8, use_nativebit: bool = True,
                 value_embed: nn.Embedding | None = None):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd, n_head, context_len, block_size, n_entries, use_nativebit,
            value_embed=value_embed,
        )
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, ffn_hidden, block_size, n_entries, use_nativebit)

    def forward(self, x: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), idx=idx)
        x = x + self.ffn(self.ln2(x))
        return x


class NativeBitGPT(nn.Module):
    """LLaMA-style GPT with NativeBitLinear layers.

    Architecture: RoPE + RMSNorm + SwiGLU + no bias.
    Embeddings, LM head, and norms stay in full precision.
    All other linear layers use NativeBitLinear (when use_nativebit=True).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layers: int = 4,
        n_embd: int = 128,
        n_head: int = 4,
        ffn_hidden: int = 512,
        context_len: int = 256,
        block_size: int = 64,
        n_entries: int = 8,
        use_nativebit: bool = True,
    ):
        super().__init__()
        self.context_len = context_len
        self.use_nativebit = use_nativebit

        # Embeddings — full precision, NOT quantized
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # Value embedding: reuse tok_emb weights (saves 10M params, shared gradients)
        self.value_embed = self.tok_emb
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            # Alternating layers get VE access
            ve = self.value_embed if (i % 2 == 1) else None
            self.blocks.append(
                TransformerBlock(n_embd, n_head, ffn_hidden, context_len,
                                 block_size, n_entries, use_nativebit,
                                 value_embed=ve)
            )

        self.ln_f = RMSNorm(n_embd)

        # LM head — full precision, NOT quantized
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        n_layers = len(self.blocks)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.normal_(m.weight, std=0.02)
        # Scale residual projections by 1/sqrt(2*n_layers) to prevent signal explosion
        residual_scale = (2 * n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.ffn.w_down.weight, std=0.02 * residual_scale)
        # Value embedding shares tok_emb weights — no separate init needed
        # Gate weights init to zero: sigmoid(0)=0.5, *2 = 1.0 = neutral start
        for block in self.blocks:
            if hasattr(block.attn, 've_gate') and block.attn.value_embed is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_len, f"Sequence length {T} > context_len {self.context_len}"

        x = self.tok_emb(idx)  # (B, T, C)

        for block in self.blocks:
            x = block(x, idx=idx)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)  # soft-capping
        return logits

    def count_parameters(self) -> dict:
        """Count parameters by category."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()
        layernorm = sum(
            p.numel() for m in self.modules() if isinstance(m, RMSNorm) for p in m.parameters()
        )
        codebook = sum(
            p.numel() for n, p in self.named_parameters() if "codebook" in n
        )
        main_weights = total - embedding - layernorm - codebook
        return {
            "total": total,
            "embedding": embedding,
            "layernorm": layernorm,
            "codebook": codebook,
            "main_weights": main_weights,
        }

    def get_nativebit_layers(self) -> list[NativeBitLinear]:
        return [m for m in self.modules() if isinstance(m, NativeBitLinear)]

    def update_all_utilization(self) -> None:
        for layer in self.get_nativebit_layers():
            layer.update_utilization_from_cache()

    def revive_all_dead_entries(self) -> int:
        total = 0
        for layer in self.get_nativebit_layers():
            total += layer.revive_dead_entries()
        return total

    def set_mode_inference(self) -> None:
        """Switch to inference mode: disable gradients and training."""
        self.requires_grad_(False)
        self.train(False)


def _make_linear(use_nativebit: bool, block_size: int, n_entries: int):
    """Return a constructor for NativeBitLinear or nn.Linear."""
    if use_nativebit:
        def make(in_f, out_f, bias=True):
            return NativeBitLinear(in_f, out_f, bias=bias,
                                   block_size=block_size, n_entries=n_entries)
        return make
    else:
        return nn.Linear


def build_model_from_config(config, use_nativebit: bool = True) -> NativeBitGPT:
    """Build a NativeBitGPT from a config object."""
    return NativeBitGPT(
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        n_embd=config.n_embd,
        n_head=config.n_head,
        ffn_hidden=config.ffn_hidden,
        context_len=config.context_len,
        block_size=config.block_size,
        n_entries=config.n_codebook,
        use_nativebit=use_nativebit,
    )
