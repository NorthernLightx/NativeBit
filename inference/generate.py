"""Text generation and speed benchmark with static-shape KV cache.

Static cache: pre-allocated buffers + dynamic_update_slice.
No recompilation between tokens — JAX compiles prefill and step once each.

Usage:
    # From training checkpoint (float or NB)
    python inference/generate.py logs/gcs/2b_nb3_s42_params.npz --nativebit --prompt "The meaning of"

    # From packed checkpoint (codebook lookup, no latent weights)
    python inference/generate.py inference/2b_nb3.nbpack.npz --packed --benchmark --sweep

    # Speed benchmark (single prompt length)
    python inference/generate.py logs/gcs/2b_nb3_s42_params.npz --nativebit --benchmark

    # Full benchmark sweep (multiple prompt lengths, JSON output)
    python inference/generate.py logs/gcs/2b_nb3_s42_params.npz --nativebit --benchmark --sweep --out results.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from nativebit_jax.model import build_model
from configs.tpu import TPU2BConfig


def load_params(ckpt_path, model, config):
    """Load training checkpoint into model params."""
    print(f"  Loading checkpoint: {ckpt_path}")
    t0 = time.time()

    rng = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(rng, dummy)

    ckpt = np.load(ckpt_path)

    def _set_leaf(path, leaf):
        key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if key in ckpt:
            arr = ckpt[key]
            if arr.dtype.kind == 'V':  # bfloat16 saved as void
                return jnp.frombuffer(arr.tobytes(), dtype=jnp.bfloat16).reshape(arr.shape)
            return jnp.array(arr)
        return leaf

    params = jax.tree_util.tree_map_with_path(_set_leaf, params)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Loaded {n_params/1e6:.1f}M params in {time.time()-t0:.1f}s")
    return params


def _unpack_indices_3bit(packed, shape):
    """Unpack 3-bit packed bytes (uint8) to indices array."""
    n_groups = len(packed) // 3
    groups = packed.reshape(n_groups, 3).astype(np.uint32)
    bits24 = groups[:, 0] | (groups[:, 1] << 8) | (groups[:, 2] << 16)
    indices = np.zeros((n_groups, 8), dtype=np.uint8)
    for j in range(8):
        indices[:, j] = (bits24 >> (j * 3)) & 0x7
    total = shape[0] * shape[1]
    return indices.reshape(-1)[:total].reshape(shape)


def load_packed_params(packed_path, model, config):
    """Load packed .nbpack.npz into a PackedNativeBitDense model.

    Stores uint8 indices + fp32 codebooks as model params. The Pallas kernel
    in PackedNativeBitDense reads these directly — no weight materialization.
    HBM: ~3 GB (vs 8.76 GB float fp32).
    """
    print(f"  Loading packed checkpoint: {packed_path}")
    t0 = time.time()

    pack = np.load(packed_path)

    # Init model to get param tree structure
    rng = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(rng, dummy)

    # Build lookup: packed file prefix → (indices_uint8, codebook)
    # Reorder blocks to tiled layout for the fused Pallas kernel
    from nativebit_jax.packed_kernel import reorder_blocks_tiled
    packed_layers = {}
    for key in pack.files:
        if not key.startswith("idx."):
            continue
        prefix = key[4:]
        cb = pack[f"cb.{prefix}"].astype(np.float32)
        idx_packed = pack[f"idx.{prefix}"]
        idx_shape = tuple(pack[f"idxshape.{prefix}"])
        w_shape = tuple(pack[f"shape.{prefix}"])
        indices = _unpack_indices_3bit(idx_packed, idx_shape)

        # Reorder: flat weight order → tiled layout (one-time permutation)
        indices, cb = reorder_blocks_tiled(
            indices, cb, w_shape[0], w_shape[1], idx_shape[1])

        # Map NativeBitDense_N → PackedNativeBitDense_N
        param_path = prefix.replace(".", "/").replace(
            "NativeBitDense", "PackedNativeBitDense")
        packed_layers[param_path] = (indices, cb)

    # Collect non-quantized params
    non_quant = {}
    for key in pack.files:
        if not key.startswith("param."):
            continue
        param_path = key[6:].replace(".", "/")
        arr = pack[key]
        if arr.dtype.kind == 'V':
            arr = np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(arr.shape)
        non_quant[param_path] = arr

    # Map into model param tree
    def _set_leaf(path, leaf):
        key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if key in non_quant:
            return jnp.array(non_quant[key])
        for packed_path, (indices, cb) in packed_layers.items():
            if key == packed_path + "/indices":
                return jnp.array(indices, dtype=jnp.uint8)
            if key == packed_path + "/codebook":
                return jnp.array(cb, dtype=jnp.float32)
        return leaf

    params = jax.tree_util.tree_map_with_path(_set_leaf, params)

    total_bytes = sum(p.nbytes for p in jax.tree_util.tree_leaves(params))
    n_layers = len(packed_layers)
    print(f"  Loaded {n_layers} packed layers (uint8 indices + codebooks)")
    print(f"  HBM usage: {total_bytes / 1e9:.2f} GB (vs ~8.76 GB float fp32)")
    print(f"  Kernel mode: {os.environ.get('NATIVEBIT_KERNEL', 'pallas')}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    return params


def init_static_cache(model, batch_size=1, dtype=jnp.bfloat16):
    """Pre-allocate static-shape KV cache for all layers.

    Each layer cache: (k_buf, v_buf, cache_len) where buffers have shape
    (B, n_head, context_len, head_dim) and cache_len tracks filled positions.
    """
    head_dim = model.n_embd // model.n_head
    caches = []
    for _ in range(model.n_layers):
        k_buf = jnp.zeros((batch_size, model.n_head, model.context_len, head_dim), dtype=dtype)
        v_buf = jnp.zeros_like(k_buf)
        caches.append((k_buf, v_buf, jnp.array(0, dtype=jnp.int32)))
    return caches


def _make_forward(model):
    """Create jitted forward functions — prefill (full seq) and step (1 token).

    Both use static-shape cache. JAX compiles one trace per input shape,
    so prefill and step each compile once (different T dimension).
    """
    @jax.jit
    def prefill(params, x, caches):
        logits, new_caches = model.apply(params, x, kv_caches=caches)
        return logits[:, -1, :], new_caches

    @jax.jit
    def step(params, x, caches):
        logits, new_caches = model.apply(params, x, kv_caches=caches)
        return logits[:, -1, :], new_caches

    @jax.jit
    def forward_no_cache(params, x):
        logits = model.apply(params, x)
        return logits[:, -1, :]

    return prefill, step, forward_no_cache


def _sample(logits, temperature, top_k, rng_key):
    if temperature > 0:
        logits = logits / temperature
        if top_k > 0:
            topk_vals = jax.lax.top_k(logits, top_k)[0]
            logits = jnp.where(logits < topk_vals[-1], -1e10, logits)
        return int(jax.random.categorical(rng_key, logits))
    return int(jnp.argmax(logits))


def generate(prefill_fn, step_fn, params, prompt_tokens, model,
             max_new=100, temperature=0.8, top_k=40):
    """Autoregressive generation with static KV cache."""
    caches = init_static_cache(model)
    x = jnp.array([prompt_tokens], dtype=jnp.int32)
    logits, caches = prefill_fn(params, x, caches)
    next_token = _sample(logits[0], temperature, top_k,
                         jax.random.PRNGKey(len(prompt_tokens)))
    tokens = list(prompt_tokens) + [next_token]

    for i in range(max_new - 1):
        x = jnp.array([[next_token]], dtype=jnp.int32)
        logits, caches = step_fn(params, x, caches)
        next_token = _sample(logits[0], temperature, top_k,
                             jax.random.PRNGKey(len(tokens)))
        tokens.append(next_token)

    return tokens


def benchmark(prefill_fn, step_fn, params, prompt_tokens, model,
              n_generate=128, n_runs=3, label=""):
    """Measure tokens/sec with static KV cache. Returns dict of results."""
    prompt_len = len(prompt_tokens)

    # Warmup (compile both functions)
    print(f"  [{label}] Compiling (prompt={prompt_len} tokens)...", flush=True)
    caches = init_static_cache(model)
    x = jnp.array([prompt_tokens], dtype=jnp.int32)
    logits, caches = prefill_fn(params, x, caches)
    jax.block_until_ready(caches)
    x1 = jnp.array([[0]], dtype=jnp.int32)
    logits, caches = step_fn(params, x1, caches)
    jax.block_until_ready(caches)

    # Prefill benchmark (avg over n_runs)
    prefill_times = []
    for _ in range(n_runs):
        caches = init_static_cache(model)
        x = jnp.array([prompt_tokens], dtype=jnp.int32)
        t0 = time.time()
        logits, caches = prefill_fn(params, x, caches)
        jax.block_until_ready(caches)
        prefill_times.append(time.time() - t0)
    prefill_time = sum(prefill_times) / len(prefill_times)
    prefill_tps = prompt_len / prefill_time

    # Generation benchmark (avg over n_runs)
    gen_times = []
    for _ in range(n_runs):
        # Fresh prefill for each run
        caches = init_static_cache(model)
        x = jnp.array([prompt_tokens], dtype=jnp.int32)
        logits, caches = prefill_fn(params, x, caches)
        jax.block_until_ready(caches)
        token = int(jnp.argmax(logits[0]))

        t0 = time.time()
        for i in range(n_generate):
            x1 = jnp.array([[token]], dtype=jnp.int32)
            logits, caches = step_fn(params, x1, caches)
            token = int(jnp.argmax(logits[0]))
            jax.block_until_ready(logits)
        gen_times.append(time.time() - t0)

    gen_time = sum(gen_times) / len(gen_times)
    gen_tps = n_generate / gen_time
    per_tok_ms = gen_time / n_generate * 1000

    print(f"  [{label}] Prefill: {prefill_time*1000:.1f}ms ({prefill_tps:.0f} tok/s)")
    print(f"  [{label}] Decode {n_generate} tok: {gen_time*1000:.0f}ms "
          f"({gen_tps:.1f} tok/s, {per_tok_ms:.1f}ms/tok)")

    return {
        "label": label,
        "prompt_len": prompt_len,
        "n_generate": n_generate,
        "n_runs": n_runs,
        "prefill_ms": round(prefill_time * 1000, 1),
        "prefill_tps": round(prefill_tps, 1),
        "generate_ms": round(gen_time * 1000, 0),
        "generate_tps": round(gen_tps, 1),
        "per_token_ms": round(per_tok_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="NativeBit inference benchmark")
    parser.add_argument("checkpoint")
    parser.add_argument("--nativebit", action="store_true")
    parser.add_argument("--packed", action="store_true",
                        help="Load from packed .nbpack.npz (codebook lookup, no latent weights)")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--n-generate", type=int, default=128,
                        help="Number of tokens to generate in benchmark")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Benchmark multiple prompt lengths")
    parser.add_argument("--out", type=str, default=None,
                        help="Save benchmark results as JSON")
    args = parser.parse_args()

    config = TPU2BConfig()
    enc = tiktoken.get_encoding("gpt2")

    if args.packed:
        mode = "NB 3-bit (packed)"
    elif args.nativebit:
        mode = "NB 3-bit"
    else:
        mode = "Float"

    print(f"\n{'='*60}")
    print(f"  NativeBit 2.2B Inference — {mode}")
    print(f"  Device: {jax.devices()[0]} ({jax.device_count()} devices)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    if args.packed:
        # On-the-fly: keep uint8 indices + codebooks in HBM, reconstruct
        # per-layer via Pallas kernel. ~3x less HBM reads than float.
        # Fallback: NATIVEBIT_KERNEL=naive env var uses gather+matmul.
        # Fallback: --packed-bf16 pre-reconstructs to bf16 Dense params.
        model = build_model(config, use_nativebit=False, use_packed=True)
        params = load_packed_params(args.checkpoint, model, config)
    else:
        model = build_model(config, use_nativebit=args.nativebit)
        params = load_params(args.checkpoint, model, config)

    prefill_fn, step_fn, _ = _make_forward(model)

    if args.benchmark:
        results = []

        if args.sweep:
            # Sweep over prompt lengths
            prompt_lengths = [32, 128, 512]
            # Generate synthetic prompts of exact lengths
            long_prompt = enc.encode(
                "The meaning of life is a philosophical question concerning the "
                "significance of existence or consciousness. It has been the "
                "subject of much philosophical scientific and theological "
                "speculation throughout history. " * 50
            )
            for plen in prompt_lengths:
                tokens = long_prompt[:plen]
                r = benchmark(prefill_fn, step_fn, params, tokens, model,
                              n_generate=args.n_generate, label=f"{mode} prompt={plen}")
                results.append(r)
        else:
            prompt_tokens = enc.encode(args.prompt)
            r = benchmark(prefill_fn, step_fn, params, prompt_tokens, model,
                          n_generate=args.n_generate, label=mode)
            results.append(r)

        # Summary table
        print(f"\n{'='*60}")
        print(f"  RESULTS — {mode} (2.2B, {jax.devices()[0]})")
        print(f"{'='*60}")
        print(f"  {'Prompt':>8s}  {'Prefill':>10s}  {'Decode tok/s':>14s}  {'ms/tok':>8s}")
        print(f"  {'-'*44}")
        for r in results:
            print(f"  {r['prompt_len']:>6d}t  {r['prefill_tps']:>8.0f}t/s"
                  f"  {r['generate_tps']:>12.1f}t/s  {r['per_token_ms']:>6.1f}ms")

        # Save JSON
        meta = {
            "model": "NativeBitGPT-2.2B",
            "mode": mode,
            "device": str(jax.devices()[0]),
            "n_devices": jax.device_count(),
            "checkpoint": args.checkpoint,
            "context_len": config.context_len,
        }
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2)
            print(f"\n  Saved: {args.out}")

    else:
        prompt_tokens = enc.encode(args.prompt)
        print(f"Prompt: '{args.prompt}' ({len(prompt_tokens)} tokens)")
        tokens = generate(prefill_fn, step_fn, params, prompt_tokens, model,
                         max_new=args.max_new, temperature=args.temperature)
        text = enc.decode(tokens)
        print(f"\n--- Generated ---\n{text}\n---")


if __name__ == "__main__":
    main()
