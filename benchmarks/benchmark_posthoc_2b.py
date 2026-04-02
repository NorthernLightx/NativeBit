"""Post-hoc quantization benchmark at 2.2B scale on TPU.

Loads a pre-trained float 2.2B checkpoint (orbax), applies post-hoc
quantization methods, and compares PPL. Run AFTER float training completes.

Usage (on TPU):
    python benchmarks/benchmark_posthoc_2b.py --ckpt-dir logs/2b_float_s42_ckpt
    python benchmarks/benchmark_posthoc_2b.py --seed 42  # auto-find checkpoint
"""

import argparse
import math
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state

from nativebit_jax.model import build_model, apply_init_scaling
from nativebit_jax.train import (
    make_optimizer, make_eval_step, setup_fsdp,
    load_tokens, make_batches,
)
from jax.sharding import NamedSharding, PartitionSpec as P
from configs.tpu import TPU2BConfig


def load_float_model(config, ckpt_dir: str):
    """Build float model, init state with FSDP, restore from checkpoint."""
    model = build_model(config, use_nativebit=False)
    rng = jax.random.PRNGKey(config.seed)
    _, init_rng = jax.random.split(rng)

    x_dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(init_rng, x_dummy)
    params = apply_init_scaling(params, config.n_layers)

    # FSDP
    params, mesh = setup_fsdp(params)

    tx = make_optimizer(config)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

    # Restore checkpoint
    from pathlib import Path
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    latest = sorted(ckpt_path.iterdir())
    if not latest:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    latest_step = int(latest[-1].name)
    abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(str(latest[-1]), target=abstract_state)
    print(f"  Restored checkpoint step {latest_step} from {ckpt_dir}")

    return model, state.params, mesh


def apply_posthoc_quantize(model, params, test_tokens, config, mesh,
                            method: str, n_entries: int, block_size: int):
    """Apply post-hoc quantization to float params and measure PPL."""
    def quantize_param(weight, n_entries, block_size):
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

    def walk_and_quantize(node, path=""):
        if isinstance(node, dict):
            return {k: walk_and_quantize(v, f"{path}/{k}" if path else k)
                    for k, v in node.items()}
        elif hasattr(node, 'shape') and len(node.shape) == 2:
            if "embedding" in path:
                return node
            if node.shape[0] > 1 and node.shape[1] > 1:
                return quantize_param(node, n_entries, block_size)
        return node

    q_params = {"params": walk_and_quantize(params["params"])}

    # Eval
    eval_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    data_sharding = NamedSharding(mesh, P('fsdp', None)) if mesh else None

    total_loss, n = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        if data_sharding is not None:
            xb = jax.device_put(xb, data_sharding)
            yb = jax.device_put(yb, data_sharding)
        total_loss += float(eval_fn(q_params, xb, yb))
        n += 1
    return math.exp(min(total_loss / n, 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Orbax checkpoint dir (default: logs/2b_float_s{seed}_ckpt)")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    config = TPU2BConfig()
    config.seed = args.seed

    ckpt_dir = args.ckpt_dir or os.path.join(
        args.log_dir, f"2b_float_s{args.seed}_ckpt")

    print(f"\n{'='*60}")
    print(f"  Post-hoc Quantization Benchmark (2.2B, WikiText-103)")
    print(f"  Device: {jax.devices()[0]} ({jax.device_count()} devices)")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"{'='*60}\n")

    # Load data
    _, _, test_tokens = load_tokens(config.dataset, "data")

    # Load float checkpoint
    model, float_params, mesh = load_float_model(config, ckpt_dir)

    # Eval float baseline
    print("  Evaluating float baseline...", flush=True)
    float_ppl = apply_posthoc_quantize(
        model, float_params, test_tokens, config, mesh,
        "kmeans", 999, 64)  # dummy — won't actually quantize since we skip below

    # Actually eval float directly
    eval_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    data_sharding = NamedSharding(mesh, P('fsdp', None)) if mesh else None
    total_loss, n = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        if data_sharding is not None:
            xb = jax.device_put(xb, data_sharding)
            yb = jax.device_put(yb, data_sharding)
        total_loss += float(eval_fn(float_params, xb, yb))
        n += 1
    float_ppl = math.exp(min(total_loss / n, 20))
    print(f"  Float baseline PPL: {float_ppl:.2f}")

    # Post-hoc methods
    results = [{"method": "Float", "ppl": float_ppl}]

    methods = [
        ("RTN 3-bit bs128", "uniform", 8, 128),
        ("RTN 3-bit bs64", "uniform", 8, 64),
        ("K-means 8-entry bs128", "kmeans", 8, 128),
        ("K-means 6-entry bs128", "kmeans", 6, 128),
        ("K-means 4-entry (2-bit) bs128", "kmeans", 4, 128),
    ]

    for name, method, n_entries, bs in methods:
        print(f"\n  {name}...", flush=True)
        t0 = time.time()
        ppl = apply_posthoc_quantize(
            model, float_params, test_tokens, config, mesh,
            method, n_entries, bs)
        dt = time.time() - t0
        delta = (ppl / float_ppl - 1) * 100
        print(f"    PPL: {ppl:.2f} (+{delta:.1f}%) [{dt:.0f}s]")
        results.append({"method": name, "ppl": ppl, "delta_pct": round(delta, 2)})

    # Final table
    print(f"\n{'='*60}")
    print(f"  RESULTS (2.2B, WikiText-103, seed={args.seed})")
    print(f"{'='*60}")
    print(f"  {'Method':<32s} {'PPL':>8s} {'vs Float':>10s}")
    print(f"  {'-'*50}")
    for r in results:
        if r["method"] == "Float":
            print(f"  {r['method']:<32s} {r['ppl']:>8.2f} {'baseline':>10s}")
        else:
            print(f"  {r['method']:<32s} {r['ppl']:>8.2f} {'+' + str(r['delta_pct']) + '%':>10s}")

    # Save
    out_path = os.path.join(args.log_dir, f"posthoc_2b_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"seed": args.seed, "float_ppl": float_ppl, "results": results}, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
