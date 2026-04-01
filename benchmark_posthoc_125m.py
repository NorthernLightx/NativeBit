"""Post-hoc quantization benchmark at 125M scale on TPU.

Trains a float 125M model (10K steps), then applies post-hoc methods and
compares against NativeBit TPU results.

Usage (on TPU):
    python benchmark_posthoc_125m.py
"""

import math
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from nativebit_jax.model import build_model, apply_init_scaling, NativeBitGPT
from nativebit_jax.train import (
    make_optimizer, make_train_step, make_eval_step,
    load_tokens, make_batches,
)
from configs.tpu import TPUMediumConfig


def train_float_model(config, train_tokens, test_tokens):
    """Train float model and return params + test PPL."""
    model = build_model(config, use_nativebit=False)
    rng = jax.random.PRNGKey(42)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    x_dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(init_rng, x_dummy)
    params = apply_init_scaling(params, config.n_layers)

    tx = make_optimizer(config)
    st = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    fast_step, _ = make_train_step(model, use_nativebit=False)

    # Compile
    batch_iter = make_batches(train_tokens, config.context_len, config.batch_size, data_rng)
    bx, by = next(batch_iter)
    st, _ = fast_step(st, bx, by)
    jax.block_until_ready(st)

    # Train
    print(f"  Training float 125M ({config.max_steps} steps)...", flush=True)
    step = 0
    t0 = time.time()
    while step < config.max_steps:
        data_rng, epoch_rng = jax.random.split(data_rng)
        for xb, yb in make_batches(train_tokens, config.context_len,
                                    config.batch_size, epoch_rng):
            st, loss = fast_step(st, xb, yb)
            step += 1
            if step % 1000 == 0:
                print(f"    step={step}, loss={float(loss):.4f}, "
                      f"sps={step/(time.time()-t0):.1f}", flush=True)
            if step >= config.max_steps:
                break

    # Eval
    eval_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    total_loss, n = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        total_loss += float(eval_fn(st.params, xb, yb))
        n += 1
        if n >= 50:
            break
    test_ppl = math.exp(min(total_loss / n, 20))
    print(f"  Float Test PPL: {test_ppl:.2f}")

    return model, st.params, test_ppl


def apply_posthoc_quantize(model, params, test_tokens, config,
                            method, n_entries, block_size):
    """Apply post-hoc quantization to float params and measure PPL.

    Quantizes in JAX: for each linear layer's weight, compute per-block
    codebook (percentile), assign indices, replace weights.
    """
    import copy

    def quantize_param(weight, n_entries, block_size):
        """Quantize a single weight matrix post-hoc."""
        shape = weight.shape
        w_flat = weight.reshape(-1)
        total = w_flat.size
        num_blocks = math.ceil(total / block_size)
        padded = num_blocks * block_size

        if padded > total:
            w_flat = jnp.pad(w_flat, (0, padded - total))

        w_blocks = w_flat.reshape(num_blocks, block_size)

        if method == "kmeans":
            # Per-block percentile codebook (same as NativeBit init)
            q = jnp.linspace(0, 1, n_entries)
            codebook = jnp.quantile(w_blocks.astype(jnp.float32), q, axis=1).T
        elif method == "uniform":
            # Per-block uniform grid between min and max
            w_min = w_blocks.min(axis=1, keepdims=True)
            w_max = w_blocks.max(axis=1, keepdims=True)
            levels = jnp.linspace(0, 1, n_entries).reshape(1, -1)
            codebook = w_min + levels * (w_max - w_min)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Assign each weight to nearest codebook entry
        dists = jnp.square(w_blocks[:, :, None] - codebook[:, None, :])
        indices = jnp.argmin(dists, axis=-1)

        # Reconstruct
        block_idx = jnp.arange(num_blocks)[:, None]
        w_quant = codebook[block_idx, indices].reshape(-1)[:total]
        return w_quant.reshape(shape)

    # Walk params and quantize all weight matrices (skip embedding)
    def walk_and_quantize(node, path=""):
        if isinstance(node, dict):
            result = {}
            for k, v in node.items():
                new_path = f"{path}/{k}" if path else k
                result[k] = walk_and_quantize(v, new_path)
            return result
        elif hasattr(node, 'shape') and len(node.shape) == 2:
            # Skip embedding
            if "embedding" in path:
                return node
            # Quantize weight matrices
            if node.shape[0] > 1 and node.shape[1] > 1:
                return quantize_param(node, n_entries, block_size)
        return node

    q_params = {"params": walk_and_quantize(params["params"])}

    # Eval with quantized params
    eval_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    total_loss, n = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        total_loss += float(eval_fn(q_params, xb, yb))
        n += 1
        if n >= 50:
            break
    return math.exp(min(total_loss / n, 20))


def main():
    config = TPUMediumConfig()
    config.max_steps = 10000
    config.batch_size = 16  # fit without remat

    print(f"\n{'='*60}")
    print(f"  Post-hoc Quantization Benchmark (125M, WikiText-103)")
    print(f"  Device: {jax.devices()[0]}")
    print(f"{'='*60}\n")

    train_tokens, valid_tokens, test_tokens = load_tokens(config.dataset, "data")

    # Train float model
    model, float_params, float_ppl = train_float_model(
        config, train_tokens, test_tokens)

    # Post-hoc methods
    results = [{"method": "Float", "ppl": float_ppl, "entries": "-", "block": "-"}]

    for name, method, n_entries, bs in [
        ("RTN 3-bit bs128", "uniform", 8, 128),
        ("RTN 3-bit bs64", "uniform", 8, 64),
        ("K-means 8-entry bs128", "kmeans", 8, 128),
        ("K-means 6-entry bs128", "kmeans", 6, 128),
        ("K-means 4-entry bs128", "kmeans", 4, 128),
    ]:
        print(f"\n  {name}...", flush=True)
        ppl = apply_posthoc_quantize(model, float_params, test_tokens, config,
                                      method, n_entries, bs)
        print(f"    PPL: {ppl:.2f} (Δ: +{(ppl/float_ppl-1)*100:.1f}%)")
        results.append({"method": name, "ppl": ppl, "entries": n_entries, "block": bs})

    # Print final table
    print(f"\n{'='*60}")
    print(f"  RESULTS (125M, WikiText-103, 10K steps)")
    print(f"{'='*60}")
    print(f"  {'Method':<30s} {'PPL':>8s} {'vs Float':>10s}")
    print(f"  {'-'*48}")
    for r in results:
        delta = f"+{(r['ppl']/float_ppl-1)*100:.1f}%" if r['ppl'] != float_ppl else "baseline"
        print(f"  {r['method']:<30s} {r['ppl']:>8.2f} {delta:>10s}")

    print(f"\n  NativeBit TPU results for comparison:")
    print(f"    NB cb=6 bs=128 (autoresearch best): PPL ~193.90 (+0.26%)")
    print(f"    NB cb=8 bs=128 (full requantize):   PPL ~185.66 (-0.03%)")

    # Save results
    os.makedirs("logs/jax", exist_ok=True)
    with open("logs/jax/posthoc_125m.json", "w") as f:
        json.dump({"float_ppl": float_ppl, "results": results}, f, indent=2)
    print(f"\n  Results saved to logs/jax/posthoc_125m.json")


if __name__ == "__main__":
    main()
