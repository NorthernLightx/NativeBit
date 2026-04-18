"""PyTorch inference with Triton fused dequant kernel for packed NativeBit models.

Loads a .nbpack.npz checkpoint, builds a PyTorch transformer with PackedLinear
layers (Triton kernel), and benchmarks decode speed.

Usage:
    set CC=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/cl.exe
    python inference/generate_torch.py inference/2b_nb3.nbpack.npz --benchmark
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference.triton_kernel import PackedLinear, BS, NE


# ---------------------------------------------------------------------------
# Minimal LLaMA-style transformer in PyTorch (mirrors nativebit_jax/model.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        norm = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / norm * self.weight.float()).to(dtype)


def precompute_rope(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(q, k, cos, sin, pos_offset=0):
    dtype = q.dtype
    T = q.shape[2]
    cos = cos[pos_offset:pos_offset+T][None, None, :, :]
    sin = sin[pos_offset:pos_offset+T][None, None, :, :]
    cos2 = cos.repeat_interleave(2, dim=-1)
    sin2 = sin.repeat_interleave(2, dim=-1)
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    q_rot = (q.float() * cos2 + rotate(q.float()) * sin2).to(dtype)
    k_rot = (k.float() * cos2 + rotate(k.float()) * sin2).to(dtype)
    return q_rot, k_rot


class CausalAttention(nn.Module):
    def __init__(self, n_embd, n_head, context_len, make_linear):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = make_linear(3 * n_embd, n_embd)
        self.out_proj = make_linear(n_embd, 3 * n_embd)  # dummy in_features, replaced by loader
        self.cos, self.sin = precompute_rope(self.head_dim, context_len)

    def forward(self, x, kv_cache=None, pos_offset=0):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            pos_offset = kv_cache[2]
        cos = self.cos.to(x.device)
        sin = self.sin.to(x.device)
        q, k = apply_rope(q, k, cos, sin, pos_offset)

        if kv_cache is not None:
            k_buf, v_buf, cache_len = kv_cache
            k_buf[:, :, cache_len:cache_len+T] = k
            v_buf[:, :, cache_len:cache_len+T] = v
            new_len = cache_len + T
            scale = self.head_dim ** -0.5
            attn = torch.einsum('bhqd,bhkd->bhqk', q * scale, k_buf)
            q_pos = cache_len + torch.arange(T, device=x.device)
            k_pos = torch.arange(k_buf.shape[2], device=x.device)
            mask = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] < new_len)
            attn = attn.masked_fill(~mask[None, None], torch.finfo(attn.dtype).min)
            out = torch.einsum('bhqk,bhkd->bhqd', F.softmax(attn, dim=-1), v_buf)
            new_cache = (k_buf, v_buf, new_len)
        else:
            # Standard cross-position causal attention.
            # New 2.2B checkpoints (post-attention-fix) use this.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            new_cache = None

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, n_embd, ffn_hidden, make_linear):
        super().__init__()
        self.gate = make_linear(ffn_hidden, n_embd)
        self.up = make_linear(ffn_hidden, n_embd)
        self.down = make_linear(n_embd, ffn_hidden)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, ffn_hidden, context_len, make_linear):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalAttention(n_embd, n_head, context_len, make_linear)
        self.ln2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, ffn_hidden, make_linear)

    def forward(self, x, kv_cache=None, pos_offset=0):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache, pos_offset=pos_offset)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class PackedGPT(nn.Module):
    def __init__(self, vocab_size, n_layers, n_embd, n_head, ffn_hidden,
                 context_len, make_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.n_head = n_head
        self.context_len = context_len
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, ffn_hidden, context_len, make_linear)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(n_embd)

    def forward(self, idx, kv_caches=None, pos_offset=0):
        x = self.embedding(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache, pos_offset=pos_offset)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = x.to(self.embedding.weight.dtype) @ self.embedding.weight.T
        logits = 30.0 * torch.tanh(logits.float() / 30.0)
        if kv_caches is not None:
            return logits, new_caches
        return logits


def unpack_indices_3bit(packed, shape):
    n_groups = len(packed) // 3
    groups = packed.reshape(n_groups, 3).astype(np.uint32)
    bits24 = groups[:, 0] | (groups[:, 1] << 8) | (groups[:, 2] << 16)
    indices = np.zeros((n_groups, 8), dtype=np.uint8)
    for j in range(8):
        indices[:, j] = (bits24 >> (j * 3)) & 0x7
    total = shape[0] * shape[1]
    return indices.reshape(-1)[:total].reshape(shape)


def load_packed_model(packed_path, config, device='cuda'):
    """Load .nbpack.npz and build a PackedGPT with Triton kernels."""
    print(f"  Loading {packed_path}...")
    t0 = time.time()
    pack = np.load(packed_path)

    # Collect packed layers
    packed_layers = {}
    for key in pack.files:
        if not key.startswith("idx."):
            continue
        prefix = key[4:]
        cb = pack[f"cb.{prefix}"].astype(np.float32)
        idx_packed = pack[f"idx.{prefix}"]
        idx_shape = tuple(pack[f"idxshape.{prefix}"])
        w_shape = tuple(pack[f"shape.{prefix}"])
        indices = unpack_indices_3bit(idx_packed, idx_shape)
        packed_layers[prefix] = (indices, cb, w_shape)

    # Non-quantized params
    non_quant = {}
    for key in pack.files:
        if key.startswith("param."):
            path = key[6:].replace(".", "/")
            arr = pack[key]
            if arr.dtype.kind == 'V':
                arr = np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(arr.shape)
            non_quant[path] = arr

    # Factory: create PackedLinear from packed data
    layer_idx = [0]
    block_idx = [0]
    module_name = ['']

    def make_packed_linear(out_features, in_features):
        """Find and return PackedLinear for the current layer position."""
        # Match by shape — the layers are created in order
        return nn.Linear(in_features, out_features, bias=False)  # placeholder

    # Build model with placeholder linears, then replace
    model = PackedGPT(
        config.vocab_size, config.n_layers, config.n_embd, config.n_head,
        config.ffn_hidden, config.context_len,
        lambda out_f, in_f: nn.Linear(in_f, out_f, bias=False),
    ).to(device)

    # Replace Linear layers with PackedLinear using packed data.
    # Natural sort: block_2 before block_10 (alphabetical sort is wrong).
    import re
    def _nat_key(s):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    packed_keys = sorted(packed_layers.keys(), key=_nat_key)
    packed_iter = iter(packed_keys)

    def replace_linears(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                pk = next(packed_iter)
                indices, cb, w_shape = packed_layers[pk]
                packed_linear = PackedLinear.from_packed(indices, cb, w_shape, device)
                setattr(module, name, packed_linear)
            else:
                replace_linears(child, full_name)

    replace_linears(model)

    # Load non-quantized params (embedding, norms)
    emb_key = "params/embedding"
    if emb_key in non_quant:
        model.embedding.weight.data = torch.from_numpy(
            non_quant[emb_key].copy()).to(device)

    for i in range(config.n_layers):
        for j, norm_name in enumerate(["ln1", "ln2"]):
            norm_key = f"params/block_{i}/RMSNorm_{j}/weight"
            if norm_key in non_quant:
                norm = getattr(model.blocks[i], norm_name)
                norm.weight.data = torch.from_numpy(
                    non_quant[norm_key].copy()).to(device)

    ln_f_key = "params/ln_f/weight"
    if ln_f_key in non_quant:
        model.ln_f.weight.data = torch.from_numpy(
            non_quant[ln_f_key].copy()).to(device)

    # Report
    total_bytes = sum(p.nbytes for p in model.parameters()) + sum(
        b.nbytes for b in model.buffers())
    print(f"  Loaded in {time.time()-t0:.1f}s, VRAM: {total_bytes/1e9:.2f} GB")
    return model


def init_kv_cache(model, batch_size=1, device='cuda'):
    head_dim = model.n_embd // model.n_head
    caches = []
    for _ in range(model.n_layers):
        k = torch.zeros(batch_size, model.n_head, model.context_len, head_dim,
                        device=device, dtype=torch.float32)
        v = torch.zeros_like(k)
        caches.append((k, v, 0))
    return caches


def benchmark(model, prompt_tokens, n_generate=128, n_runs=3, device='cuda'):
    """Benchmark decode speed with KV cache."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Warmup
    print("  Compiling...", flush=True)
    caches = init_kv_cache(model, device=device)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, caches = model(x, kv_caches=caches)
    x1 = torch.tensor([[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, caches = model(x1, kv_caches=caches)
    torch.cuda.synchronize()

    # Decode benchmark
    gen_times = []
    for _ in range(n_runs):
        caches = init_kv_cache(model, device=device)
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, caches = model(x, kv_caches=caches)
        torch.cuda.synchronize()
        token = int(logits[0, -1].argmax())

        t0 = time.time()
        for i in range(n_generate):
            x1 = torch.tensor([[token]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, caches = model(x1, kv_caches=caches)
            token = int(logits[0, -1].argmax())
        torch.cuda.synchronize()
        gen_times.append(time.time() - t0)

    gen_time = sum(gen_times) / len(gen_times)
    tps = n_generate / gen_time
    ms_tok = gen_time / n_generate * 1000

    print(f"  Decode {n_generate} tok: {gen_time*1000:.0f}ms ({tps:.1f} tok/s, {ms_tok:.1f}ms/tok)")
    return {"generate_tps": round(tps, 1), "per_token_ms": round(ms_tok, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--n-generate", type=int, default=128)
    parser.add_argument("--prompt", default="The meaning of life is")
    args = parser.parse_args()

    from configs.tpu import TPU2BConfig
    config = TPU2BConfig()

    print(f"\n{'='*60}")
    print(f"  NativeBit 2.2B — PyTorch + Triton Kernel")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    model = load_packed_model(args.checkpoint, config)

    if args.benchmark:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        benchmark(model, prompt_tokens, n_generate=args.n_generate)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        print(f"Prompt: '{args.prompt}' ({len(prompt_tokens)} tokens)")

        caches = init_kv_cache(model)
        x = torch.tensor([prompt_tokens], dtype=torch.long, device='cuda')
        with torch.no_grad():
            logits, caches = model(x, kv_caches=caches)
        tokens = list(prompt_tokens)
        for _ in range(100):
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            x1 = torch.tensor([[token]], dtype=torch.long, device='cuda')
            with torch.no_grad():
                logits, caches = model(x1, kv_caches=caches)
        print(enc.decode(tokens))


if __name__ == "__main__":
    main()
