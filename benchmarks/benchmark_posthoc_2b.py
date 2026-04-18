"""Post-hoc quantization benchmark at 2.2B scale on TPU.

Loads a pre-trained float 2.2B checkpoint (.npz), applies post-hoc
quantization methods, and compares PPL against NativeBit co-trained.

Usage (on TPU):
    python benchmarks/benchmark_posthoc_2b.py
    python benchmarks/benchmark_posthoc_2b.py --ckpt logs/gcs/2b_float_s42_params.npz
"""

import argparse
import gc
import math
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import jax
import jax.numpy as jnp

from nativebit_jax.model import build_model, apply_init_scaling
from nativebit_jax.train import (
    make_eval_step, setup_fsdp,
    load_tokens, make_batches,
)
from jax.sharding import NamedSharding, PartitionSpec as P
from configs.tpu import TPU2BConfig


# NativeBit co-trained PPL must be supplied at runtime (--nb-ppl) — no default.
# Previously hardcoded to 179.30, which was from a pre-attention-fix run and
# misled comparisons. Pass the latest value from the NB eval log.


def load_float_npz(config, ckpt_path: str):
    """Build float model, load .npz checkpoint, setup FSDP."""
    model = build_model(config, use_nativebit=False)
    rng = jax.random.PRNGKey(config.seed)
    _, init_rng = jax.random.split(rng)

    # Init params to get tree structure
    x_dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(init_rng, x_dummy)
    params = apply_init_scaling(params, config.n_layers)

    # FSDP (no-op on single device)
    params, mesh = setup_fsdp(params)

    # Load .npz and rebuild params tree (same as train.py resume)
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt_data = np.load(ckpt_path)

    def _set_leaf(path, leaf):
        key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if key in ckpt_data:
            return jnp.array(ckpt_data[key])
        return leaf

    restored = jax.tree_util.tree_map_with_path(_set_leaf, params)

    # Re-shard to match FSDP layout
    if mesh is not None:
        def _reshard(old, new):
            if hasattr(old, 'sharding'):
                return jax.device_put(new, old.sharding)
            return new
        restored = jax.tree.map(_reshard, params, restored)

    n_keys = len(ckpt_data.files)
    del ckpt_data; gc.collect()
    print(f"  Loaded {n_keys} parameter arrays")

    return model, restored, mesh


def eval_ppl(eval_fn, params, test_tokens, config, mesh):
    """Evaluate perplexity on full test set."""
    eval_rng = jax.random.PRNGKey(0)
    data_sharding = NamedSharding(mesh, P('fsdp', None)) if mesh else None

    total_loss, n = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        if data_sharding is not None:
            xb = jax.device_put(xb, data_sharding)
            yb = jax.device_put(yb, data_sharding)
        total_loss += float(eval_fn(params, xb, yb))
        n += 1
    return math.exp(min(total_loss / n, 20)), n


def quantize_params(params, method: str, n_entries: int, block_size: int):
    """Apply post-hoc quantization to all weight matrices (skip embeddings)."""

    def quantize_weight(weight):
        shape = weight.shape
        w_flat = weight.reshape(-1)
        total = w_flat.size
        num_blocks = math.ceil(total / block_size)
        padded = num_blocks * block_size

        if padded > total:
            w_flat = jnp.pad(w_flat, (0, padded - total))

        w_blocks = w_flat.reshape(num_blocks, block_size)

        if method == "kmeans":
            q = jnp.linspace(0, 1, n_entries)
            codebook = jnp.quantile(w_blocks.astype(jnp.float32), q, axis=1).T
        elif method == "uniform":
            w_min = w_blocks.min(axis=1, keepdims=True)
            w_max = w_blocks.max(axis=1, keepdims=True)
            levels = jnp.linspace(0, 1, n_entries).reshape(1, -1)
            codebook = w_min + levels * (w_max - w_min)
        else:
            raise ValueError(f"Unknown method: {method}")

        dists = jnp.square(w_blocks[:, :, None] - codebook[:, None, :])
        indices = jnp.argmin(dists, axis=-1)

        block_idx = jnp.arange(num_blocks)[:, None]
        w_quant = codebook[block_idx, indices].reshape(-1)[:total]
        return w_quant.reshape(shape)

    def walk(node, path=""):
        if isinstance(node, dict):
            return {k: walk(v, f"{path}/{k}" if path else k)
                    for k, v in node.items()}
        elif hasattr(node, 'shape') and len(node.shape) == 2:
            if "embedding" in path:
                return node
            if node.shape[0] > 1 and node.shape[1] > 1:
                return quantize_weight(node)
        return node

    return {"params": walk(params["params"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None,
                        help=".npz checkpoint (default: logs/gcs/2b_float_s{seed}_params.npz)")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--nb-ppl", type=float, default=None,
                        help="NativeBit co-trained PPL to compare against "
                             "(read from the NB eval log 'type=eval' record). "
                             "Omit to skip NB comparison.")
    args = parser.parse_args()

    config = TPU2BConfig()
    config.seed = args.seed

    ckpt_path = args.ckpt or os.path.join(
        "logs", "gcs", f"2b_float_s{args.seed}_params.npz")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print(f"  Expected: {os.path.abspath(ckpt_path)}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Post-hoc Quantization Benchmark (2.2B, WikiText-103)")
    print(f"  Device: {jax.devices()[0]} ({jax.device_count()} devices)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")

    # Load data
    _, _, test_tokens = load_tokens(config.dataset, "data")

    # Load float checkpoint
    model, float_params, mesh = load_float_npz(config, ckpt_path)

    # Compile eval step once (reused for all methods)
    eval_fn = make_eval_step(model)

    # Float baseline
    print("\n  Evaluating float baseline...", flush=True)
    t0 = time.time()
    float_ppl, n_batches = eval_ppl(eval_fn, float_params, test_tokens, config, mesh)
    print(f"  Float baseline PPL: {float_ppl:.2f} ({n_batches} batches, {time.time()-t0:.0f}s)")

    results = [{"method": "Float (baseline)", "ppl": float_ppl}]

    # Post-hoc methods
    methods = [
        ("RTN 3-bit bs=128",               "uniform", 8,  128),
        ("RTN 3-bit bs=64",                "uniform", 8,  64),
        ("K-means 8-entry (3-bit) bs=128", "kmeans",  8,  128),
        ("K-means 6-entry (~2.6b) bs=128", "kmeans",  6,  128),
        ("K-means 4-entry (2-bit) bs=128", "kmeans",  4,  128),
    ]

    for name, method, n_entries, bs in methods:
        print(f"\n  {name}...", flush=True)
        t0 = time.time()
        q_params = quantize_params(float_params, method, n_entries, bs)

        # Re-shard quantized params for FSDP eval
        if mesh is not None:
            def _reshard(old, new):
                if hasattr(old, 'sharding'):
                    return jax.device_put(new, old.sharding)
                return new
            q_params = jax.tree.map(_reshard, float_params, q_params)

        ppl, _ = eval_ppl(eval_fn, q_params, test_tokens, config, mesh)
        dt = time.time() - t0
        delta = (ppl / float_ppl - 1) * 100
        sign = "+" if delta >= 0 else ""
        print(f"    PPL: {ppl:.2f} ({sign}{delta:.1f}%) [{dt:.0f}s]")
        results.append({"method": name, "ppl": round(ppl, 2),
                        "delta_pct": round(delta, 2)})
        del q_params; gc.collect()

    # Final table
    print(f"\n{'='*60}")
    print(f"  RESULTS — 2.2B Post-hoc vs NativeBit (WikiText-103)")
    print(f"  seed={args.seed}, {n_batches} test batches")
    print(f"{'='*60}")
    print(f"  {'Method':<36s} {'PPL':>8s} {'vs Float':>10s}")
    print(f"  {'-'*54}")
    for r in results:
        if "baseline" in r["method"]:
            print(f"  {r['method']:<36s} {r['ppl']:>8.2f} {'---':>10s}")
        else:
            d = r["delta_pct"]
            print(f"  {r['method']:<36s} {r['ppl']:>8.2f} {'+' if d >= 0 else ''}{d:.1f}%")
    print(f"  {'-'*54}")
    if args.nb_ppl is not None:
        nb_delta = (args.nb_ppl / float_ppl - 1) * 100
        nb_sign = "+" if nb_delta >= 0 else ""
        print(f"  {'NativeBit 3-bit (co-trained)':<36s} "
              f"{args.nb_ppl:>8.2f} {nb_sign}{nb_delta:.1f}%")
    else:
        print(f"  NativeBit comparison skipped — pass --nb-ppl to include.")
    print(f"  {'='*54}")

    # Save
    out_path = os.path.join(args.log_dir, f"posthoc_2b_s{args.seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "float_ppl": round(float_ppl, 2),
            "nb_ppl": args.nb_ppl,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
