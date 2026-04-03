"""Minimal text generation from a NativeBit packed or training checkpoint.

No KV cache — simple but correct. Good enough for speed benchmarks.

Usage:
    # From training checkpoint (float or NB)
    python inference/generate.py logs/gcs/2b_nb3_s42_params.npz --nativebit --prompt "The meaning of"

    # Speed benchmark
    python inference/generate.py logs/gcs/2b_nb3_s42_params.npz --nativebit --benchmark
"""
import argparse
import math
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


def load_params(ckpt_path, model, nativebit=False):
    """Load training checkpoint into model params."""
    rng = jax.random.PRNGKey(42)
    config = TPU2BConfig()
    dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(rng, dummy)

    ckpt = np.load(ckpt_path)

    def _set_leaf(path, leaf):
        key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if key in ckpt:
            arr = ckpt[key]
            if arr.dtype.kind == 'V':  # bfloat16 as void
                return jnp.frombuffer(arr.tobytes(), dtype=jnp.bfloat16).reshape(arr.shape)
            return jnp.array(arr)
        return leaf

    params = jax.tree_util.tree_map_with_path(_set_leaf, params)

    # No FSDP for inference — model fits on one chip (no optimizer state)
    return params, None


def _make_forward(model):
    """Create jitted forward functions — prefill (full seq) and step (1 token)."""
    @jax.jit
    def prefill(params, x):
        logits, caches = model.apply(params, x, kv_caches=[None] * model.n_layers, pos_offset=0)
        return logits[:, -1, :], caches

    @jax.jit
    def step(params, x, kv_caches, pos):
        logits, caches = model.apply(params, x, kv_caches=kv_caches, pos_offset=pos)
        return logits[:, -1, :], caches

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


def generate(prefill_fn, step_fn, params, prompt_tokens, max_new=100,
             temperature=0.8, top_k=40):
    """Autoregressive generation with KV cache."""
    # Prefill
    x = jnp.array([prompt_tokens], dtype=jnp.int32)
    logits, caches = prefill_fn(params, x)
    next_token = _sample(logits[0], temperature, top_k,
                         jax.random.PRNGKey(len(prompt_tokens)))
    tokens = list(prompt_tokens) + [next_token]

    # Generate with cache
    for i in range(max_new - 1):
        x = jnp.array([[next_token]], dtype=jnp.int32)
        logits, caches = step_fn(params, x, caches, len(tokens) - 1)
        next_token = _sample(logits[0], temperature, top_k,
                             jax.random.PRNGKey(len(tokens)))
        tokens.append(next_token)

    return tokens


def benchmark(prefill_fn, step_fn, params, prompt_tokens, n_generate=50):
    """Measure tokens/sec with KV cache."""
    # Warmup (compile both functions)
    x = jnp.array([prompt_tokens], dtype=jnp.int32)
    logits, caches = prefill_fn(params, x)
    jax.block_until_ready(caches)
    x1 = jnp.array([[0]], dtype=jnp.int32)
    logits, caches = step_fn(params, x1, caches, len(prompt_tokens))
    jax.block_until_ready(caches)

    # Prefill benchmark
    x = jnp.array([prompt_tokens], dtype=jnp.int32)
    t0 = time.time()
    logits, caches = prefill_fn(params, x)
    jax.block_until_ready(caches)
    prefill_time = time.time() - t0
    prefill_tps = len(prompt_tokens) / prefill_time

    # Generation benchmark (with KV cache)
    token = int(jnp.argmax(logits[0]))
    t0 = time.time()
    for i in range(n_generate):
        x1 = jnp.array([[token]], dtype=jnp.int32)
        logits, caches = step_fn(params, x1, caches, len(prompt_tokens) + i + 1)
        token = int(jnp.argmax(logits[0]))
        jax.block_until_ready(logits)
    gen_time = time.time() - t0
    gen_tps = n_generate / gen_time

    print(f"\n=== Inference Benchmark (KV cache) ===")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Prompt: {len(prompt_tokens)} tokens")
    print(f"  Prefill: {prefill_time*1000:.0f}ms ({prefill_tps:.0f} tok/s)")
    print(f"  Generate ({n_generate} tok): {gen_time*1000:.0f}ms ({gen_tps:.1f} tok/s)")
    print(f"  Per-token: {gen_time/n_generate*1000:.1f}ms")

    return {"prefill_tps": prefill_tps, "generate_tps": gen_tps,
            "per_token_ms": gen_time / n_generate * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--nativebit", action="store_true")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    config = TPU2BConfig()
    enc = tiktoken.get_encoding("gpt2")

    print(f"Device: {jax.devices()[0]} ({jax.device_count()} devices)")
    print(f"NativeBit: {args.nativebit}")

    model = build_model(config, use_nativebit=args.nativebit)
    params, mesh = load_params(args.checkpoint, model, args.nativebit)

    prompt_tokens = enc.encode(args.prompt)
    print(f"Prompt: '{args.prompt}' ({len(prompt_tokens)} tokens)")

    prefill_fn, step_fn, _ = _make_forward(model)

    if args.benchmark:
        benchmark(prefill_fn, step_fn, params, prompt_tokens)
    else:
        tokens = generate(prefill_fn, step_fn, params, prompt_tokens,
                         max_new=args.max_new, temperature=args.temperature)
        text = enc.decode(tokens)
        print(f"\n--- Generated ---\n{text}\n---")


if __name__ == "__main__":
    main()
