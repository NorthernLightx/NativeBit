"""LLaMA-style causal transformer with NativeBit quantization — JAX/Flax port.

Architecture: RoPE, RMSNorm, SwiGLU, value embeddings, logit soft-capping.
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import NativeBitDense


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", nn.initializers.ones_init(), (x.shape[-1],))
        x_f32 = x.astype(jnp.float32)
        norm = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / norm).astype(x.dtype) * weight


def _precompute_rope(dim: int, max_len: int, theta: float = 10000.0):
    """Precompute cos/sin tables for RoPE."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # (max_len, dim/2)
    return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to q and k. Shapes: (B, H, T, D)."""
    T = q.shape[2]
    cos = cos[:T][None, None, :, :]  # (1, 1, T, dim/2)
    sin = sin[:T][None, None, :, :]

    def rotate(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = jnp.stack((-x2, x1), axis=-1).reshape(x.shape)
        return rotated

    cos2 = jnp.repeat(cos, 2, axis=-1)
    sin2 = jnp.repeat(sin, 2, axis=-1)
    q_rot = q * cos2 + rotate(q) * sin2
    k_rot = k * cos2 + rotate(k) * sin2
    return q_rot, k_rot


def _make_linear(use_nativebit: bool, block_size: int, n_entries: int,
                 compute_dtype: jnp.dtype, use_aqt: bool = False):
    """Return constructor for NativeBitDense or nn.Dense."""
    if use_nativebit:
        def make(features, use_bias=False):
            return NativeBitDense(
                features=features, use_bias=use_bias,
                block_size=block_size, n_entries=n_entries,
                compute_dtype=compute_dtype, use_aqt=use_aqt,
            )
        return make
    else:
        def make(features, use_bias=False):
            return nn.Dense(features=features, use_bias=use_bias,
                            dtype=compute_dtype, param_dtype=jnp.float32)
        return make


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""
    n_head: int
    n_embd: int
    context_len: int
    block_size: int = 64
    n_entries: int = 8
    use_nativebit: bool = True
    use_aqt: bool = False
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, kv_cache=None, pos_offset: int = 0):
        B, T, C = x.shape
        head_dim = self.n_embd // self.n_head
        Linear = _make_linear(self.use_nativebit, self.block_size, self.n_entries,
                              self.compute_dtype, self.use_aqt)

        qkv = Linear(3 * self.n_embd, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        # RoPE — pos_offset shifts positions for KV-cached inference
        cos, sin = _precompute_rope(head_dim, self.context_len)
        cos = jax.lax.dynamic_slice(cos, (pos_offset, 0), (T, cos.shape[1]))
        sin = jax.lax.dynamic_slice(sin, (pos_offset, 0), (T, sin.shape[1]))
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        cos2 = jnp.repeat(cos, 2, axis=-1)
        sin2 = jnp.repeat(sin, 2, axis=-1)
        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)
        q = q * cos2 + rotate(q) * sin2
        k = k * cos2 + rotate(k) * sin2

        # KV cache for inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = jnp.concatenate([cached_k, k], axis=2)
            v = jnp.concatenate([cached_v, v], axis=2)
        new_cache = (k, v)

        v = v.astype(q.dtype)

        # Attention — not causal when using cache (cache already has prior tokens)
        is_causal = kv_cache is None and T > 1
        out = jax.nn.dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        return Linear(self.n_embd, use_bias=False)(out), new_cache


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: (silu(x @ W_gate) * (x @ W_up)) @ W_down."""
    n_embd: int
    ffn_hidden: int
    block_size: int = 64
    n_entries: int = 8
    use_nativebit: bool = True
    use_aqt: bool = False
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        Linear = _make_linear(self.use_nativebit, self.block_size, self.n_entries,
                              self.compute_dtype, self.use_aqt)
        gate = nn.silu(Linear(self.ffn_hidden, use_bias=False)(x))
        up = Linear(self.ffn_hidden, use_bias=False)(x)
        return Linear(self.n_embd, use_bias=False)(gate * up)


class TransformerBlock(nn.Module):
    n_embd: int
    n_head: int
    ffn_hidden: int
    context_len: int
    block_size: int = 64
    n_entries: int = 8
    use_nativebit: bool = True
    use_aqt: bool = False
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, kv_cache=None, pos_offset: int = 0):
        attn_out, new_cache = CausalSelfAttention(
            n_head=self.n_head, n_embd=self.n_embd,
            context_len=self.context_len,
            block_size=self.block_size, n_entries=self.n_entries,
            use_nativebit=self.use_nativebit, use_aqt=self.use_aqt,
            compute_dtype=self.compute_dtype,
        )(RMSNorm()(x), kv_cache=kv_cache, pos_offset=pos_offset)
        x = x + attn_out

        x = x + SwiGLU(
            n_embd=self.n_embd, ffn_hidden=self.ffn_hidden,
            block_size=self.block_size, n_entries=self.n_entries,
            use_nativebit=self.use_nativebit, use_aqt=self.use_aqt,
            compute_dtype=self.compute_dtype,
        )(RMSNorm()(x))

        return x, new_cache


class NativeBitGPT(nn.Module):
    """LLaMA-style GPT with NativeBit quantization.

    Embeddings, LM head, and norms stay in full precision.
    """
    vocab_size: int = 50257
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    block_size: int = 64
    n_entries: int = 8
    use_nativebit: bool = True
    use_aqt: bool = False
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, idx: jnp.ndarray, kv_caches=None, pos_offset: int = 0):
        B, T = idx.shape

        # Embedding — full precision, NOT quantized
        embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )
        x = embedding[idx]  # (B, T, C)

        # Transformer blocks — remat for training, plain for inference
        BlockClass = nn.remat(TransformerBlock) if kv_caches is None else TransformerBlock
        new_caches = []
        for i in range(self.n_layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = BlockClass(
                n_embd=self.n_embd, n_head=self.n_head,
                ffn_hidden=self.ffn_hidden, context_len=self.context_len,
                block_size=self.block_size, n_entries=self.n_entries,
                use_nativebit=self.use_nativebit, use_aqt=self.use_aqt,
                compute_dtype=self.compute_dtype,
                name=f"block_{i}",
            )(x, kv_cache=layer_cache, pos_offset=pos_offset)
            new_caches.append(new_cache)

        x = RMSNorm(name="ln_f")(x)

        # LM head — weight-tied with embedding (full precision)
        logits = x.astype(jnp.float32) @ embedding.T

        # Soft-capping
        logits = 30.0 * jnp.tanh(logits / 30.0)

        if kv_caches is not None:
            return logits, new_caches
        return logits


def build_model(config, use_nativebit: bool = True,
                use_aqt: bool = False) -> NativeBitGPT:
    """Build model from config."""
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
        use_aqt=use_aqt,
    )


def apply_init_scaling(params, n_layers: int):
    """Scale residual projection weights by 1/sqrt(2*n_layers).

    Targets: out_proj (attention) and w_down (SwiGLU FFN) — the two
    projections that feed into residual additions. Prevents signal
    explosion in deep models.
    """
    scale = (2 * n_layers) ** -0.5

    def _scale_if_residual(path, param):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        # Match out_proj and w_down Dense/NativeBitDense layers
        # In Flax, the attention output is the 2nd Dense (index 1) in CausalSelfAttention
        # and w_down is the 3rd Dense (index 2) in SwiGLU
        # We identify by: last Dense/NativeBitDense in attention = out_proj,
        #                  last Dense/NativeBitDense in SwiGLU = w_down
        # Simpler: just match "Dense_1/weight" in attn and "Dense_2/weight" in SwiGLU
        if ("CausalSelfAttention_0/NativeBitDense_1/weight" in path_str or
            "CausalSelfAttention_0/Dense_1/weight" in path_str or
            "SwiGLU_0/NativeBitDense_2/weight" in path_str or
            "SwiGLU_0/Dense_2/weight" in path_str):
            return param * scale
        return param

    return jax.tree_util.tree_map_with_path(_scale_if_residual, params)
