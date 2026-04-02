"""JAX/Flax training loop for NativeBit — native TPU support.

Usage:
    # On TPU (auto-detected):
    python -m nativebit_jax.train --config tpu-small --name jax_bench

    # Quick local test:
    python -m nativebit_jax.train --max-steps 100 --name test
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Add parent to path so configs/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nativebit_jax.model import build_model, NativeBitGPT, apply_init_scaling
from nativebit_jax.codebook_utils import ema_update_codebooks, revive_dead_entries
from nativebit_jax.layers import NativeBitDense


# ---------------------------------------------------------------------------
# Data loading (reuse tiktoken tokenization from PyTorch version)
# ---------------------------------------------------------------------------

def load_tokens(dataset: str = "wikitext-2", data_dir: str = "data"):
    """Load and tokenize dataset via HuggingFace datasets + tiktoken.

    No torch dependency. Caches tokenized data as binary files.
    """
    import array
    import tiktoken

    cache_dir = os.path.join(data_dir, dataset.replace("-", ""))
    os.makedirs(cache_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    hf_name_map = {
        "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "openwebtext": ("openwebtext", "plain_text"),
    }
    hf_name, hf_config = hf_name_map.get(dataset, ("wikitext", "wikitext-2-raw-v1"))

    split_map = {"train": "train", "valid": "validation", "test": "test"}
    results = {}

    for our_name, hf_split in split_map.items():
        cache_path = os.path.join(cache_dir, f"{our_name}.tokens.bin")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                a = array.array("i")
                a.frombytes(f.read())
            results[our_name] = jnp.frombuffer(a, dtype=jnp.int32).copy()
            continue

        # Download via HuggingFace datasets
        print(f"  Downloading {dataset}/{hf_split}...", flush=True)
        from datasets import load_dataset
        ds = load_dataset(hf_name, hf_config, split=hf_split)
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
        tokens = enc.encode(text)

        # Cache
        a = array.array("i", tokens)
        with open(cache_path, "wb") as f:
            a.tofile(f)
        results[our_name] = jnp.array(tokens, dtype=jnp.int32)

    train, valid, test = results["train"], results["valid"], results["test"]
    print(f"  [{dataset}] Train: {len(train)} tok, Valid: {len(valid)}, Test: {len(test)}")
    return train, valid, test


def make_batches(tokens: jnp.ndarray, ctx_len: int, batch_size: int,
                 rng: jax.Array):
    """Yield (x, y) batches from token array, shuffled."""
    n_seq = (len(tokens) - 1) // ctx_len
    all_tokens = tokens[:n_seq * ctx_len + 1]
    all_x = all_tokens[:-1].reshape(n_seq, ctx_len)
    all_y = all_tokens[1:].reshape(n_seq, ctx_len)

    perm = jax.random.permutation(rng, n_seq)
    all_x = all_x[perm]
    all_y = all_y[perm]

    n_batches = n_seq // batch_size
    for i in range(n_batches):
        s = i * batch_size
        yield all_x[s:s + batch_size], all_y[s:s + batch_size]


# ---------------------------------------------------------------------------
# Optimizer with separate param groups
# ---------------------------------------------------------------------------

def make_optimizer(config):
    """Create optax optimizer with cosine schedule + warmup + param groups."""
    total = max(config.max_steps, config.warmup_steps + 1)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=total,
        end_value=config.lr * 0.1,
    )

    cb_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.codebook_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=total,
        end_value=config.codebook_lr * 0.1,
    )

    def label_fn(params):
        """Map each param to its optimizer group."""
        flat = jax.tree.map(lambda _: "decay", params)

        def relabel(path, _):
            path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            if "codebook" in path_str:
                return "codebook"
            if "embedding" in path_str or "RMSNorm" in path_str or "ln_f" in path_str:
                return "no_decay"
            return "decay"

        return jax.tree_util.tree_map_with_path(relabel, flat)

    tx = optax.multi_transform(
        {
            "decay": optax.chain(
                optax.clip_by_global_norm(config.grad_clip),
                optax.adamw(schedule, weight_decay=config.weight_decay),
            ),
            "no_decay": optax.chain(
                optax.clip_by_global_norm(config.grad_clip),
                optax.adamw(schedule, weight_decay=0.0),
            ),
            # Codebook: zero gradients — updated via EMA, not optimizer
            "codebook": optax.set_to_zero(),
        },
        label_fn,
    )
    return tx


# ---------------------------------------------------------------------------
# Train step (jit-compiled)
# ---------------------------------------------------------------------------

def _ema_update_params(params, intermediates, decay: float = 0.99):
    """Walk params tree, update codebooks using EMA from captured indices.

    For each NativeBitDense layer that emitted indices via sow(), compute
    the mean weight value assigned to each codebook entry and EMA-blend.
    """
    import math

    def _update_layer(layer_params, layer_intermediates):
        """Update one layer's codebook via EMA."""
        if "codebook" not in layer_params or "indices" not in layer_intermediates:
            return layer_params

        weight = layer_params["weight"]
        codebook = layer_params["codebook"]
        # sow stores as a tuple of values (one per call); take last
        indices = layer_intermediates["indices"]
        if isinstance(indices, (list, tuple)):
            indices = indices[-1]

        num_blocks, n_entries = codebook.shape
        total_weights = weight.size
        block_size_val = math.ceil(total_weights / num_blocks)
        # Reconstruct weight blocks
        padded_len = num_blocks * block_size_val
        w_flat = weight.reshape(-1)
        if padded_len > total_weights:
            w_flat = jnp.pad(w_flat, (0, padded_len - total_weights))
        w_blocks = w_flat.reshape(num_blocks, block_size_val)

        new_cb = ema_update_codebooks(codebook, indices, w_blocks, decay)
        return {**layer_params, "codebook": new_cb}

    # Walk the params pytree looking for layers with codebook + matching intermediates
    def _walk(params_node, inter_node):
        if isinstance(params_node, dict):
            if "codebook" in params_node and "weight" in params_node:
                # This is a NativeBitDense layer
                inter = inter_node if isinstance(inter_node, dict) else {}
                return _update_layer(params_node, inter)
            result = {}
            for k, v in params_node.items():
                inter_child = inter_node.get(k, {}) if isinstance(inter_node, dict) else {}
                result[k] = _walk(v, inter_child)
            return result
        return params_node

    inter_dict = intermediates.get("intermediates", {}) if isinstance(intermediates, dict) else {}
    new_params = _walk(params["params"], inter_dict.get("params", inter_dict))
    return {**params, "params": new_params}


def _revive_all(state, model, real_batch: jnp.ndarray):
    """Revive dead codebook entries across all NativeBitDense layers.

    Uses requantize_params to get current indices, then revives dead entries.
    """
    import math
    from nativebit_jax.layers import requantize_params

    # Get current indices via external requantize
    _, intermediates = requantize_params(state.params)
    # Wrap to match expected format
    updates = {"intermediates": intermediates}
    inter = updates.get("intermediates", {})

    def _walk_revive(params_node, inter_node):
        if isinstance(params_node, dict):
            if "codebook" in params_node and "weight" in params_node:
                codebook = params_node["codebook"]
                num_blocks, n_entries = codebook.shape
                inter = inter_node if isinstance(inter_node, dict) else {}
                indices = inter.get("indices", None)
                if indices is not None:
                    if isinstance(indices, (list, tuple)):
                        indices = indices[-1]
                    # Compute utilization from indices
                    one_hot = jax.nn.one_hot(indices, n_entries)
                    utilization = one_hot.sum(axis=1).astype(jnp.int32)
                    new_cb, n_dead = revive_dead_entries(codebook, utilization)
                    if n_dead > 0:
                        return {**params_node, "codebook": new_cb}
                return params_node
            result = {}
            for k, v in params_node.items():
                inter_child = inter_node.get(k, {}) if isinstance(inter_node, dict) else {}
                result[k] = _walk_revive(v, inter_child)
            return result
        return params_node

    inter_dict = inter.get("params", inter) if isinstance(inter, dict) else {}
    new_p = _walk_revive(state.params["params"], inter_dict)
    new_params = {**state.params, "params": new_p}
    return state.replace(params=new_params)


def make_train_step(model: NativeBitGPT):
    """Create a single jit-compiled train step.

    Same function for float and NativeBit — no branching. NativeBit layers
    read cached deltas; cache is updated externally by requantize_params().
    """
    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            # No mutable — cache is read-only during forward.
            # Cache updates happen externally via requantize_params().
            logits = model.apply(params, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
            ).mean()
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def make_eval_step(model):
    """Create jit-compiled eval step."""
    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply(params, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        ).mean()
        return loss
    return eval_step


def setup_fsdp(params):
    """Shard params across devices (FSDP). No-op on single device.

    Shards 2D+ params along first dim when evenly divisible, replicates otherwise.
    XLA automatically inserts allgather/reduce-scatter in jitted functions.
    """
    n_devices = jax.device_count()
    if n_devices <= 1:
        return params, None

    mesh = Mesh(jax.devices(), axis_names=('fsdp',))

    def _shard(x):
        if x.ndim >= 2 and x.shape[0] % n_devices == 0:
            spec = P('fsdp', *([None] * (x.ndim - 1)))
        else:
            spec = P(*([None] * x.ndim))
        return jax.device_put(x, NamedSharding(mesh, spec))

    sharded = jax.tree.map(_shard, params)
    print(f"  FSDP: sharded across {n_devices} devices")
    return sharded, mesh


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config, use_nativebit: bool = True, use_aqt: bool = False,
          experiment_name: str = "nativebit_jax",
          log_dir: str = "logs", data_dir: str = "data"):
    """Run training."""
    print(f"\n=== {experiment_name} (JAX/Flax) ===")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  NativeBit: {use_nativebit}")
    if use_aqt:
        from nativebit_jax.layers import _aqt_available
        if not _aqt_available:
            print("  WARNING: --use-aqt but aqt not installed. Falling back to bf16 matmul.")
        else:
            print("  AQT: INT8 matmuls enabled")

    # Model
    model = build_model(config, use_nativebit=use_nativebit, use_aqt=use_aqt)
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    # Init params (including cache collection for quantized weight caching)
    dummy_x = jnp.ones((1, config.context_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_x)

    # Scale residual projections to prevent signal explosion in deep models
    params = apply_init_scaling(params, config.n_layers)

    # Re-init codebooks from scaled weights (codebook was init'd from unscaled weights)
    if use_nativebit:
        from nativebit_jax.layers import _init_codebook_from_weight
        def _reinit_cb(path, param):
            path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            if path_str.endswith("/codebook"):
                # Find the matching weight param
                weight_path = path_str.rsplit("/codebook", 1)[0] + "/weight"
                # Walk params to find it
                node = params["params"]
                for key in weight_path.split("/"):
                    if key and key in node:
                        node = node[key]
                weight = node
                import math
                num_blocks, n_entries = param.shape
                total_weights = weight.size
                padded_len = num_blocks * (math.ceil(total_weights / num_blocks))
                return _init_codebook_from_weight(
                    weight, math.ceil(total_weights / num_blocks),
                    num_blocks, total_weights, padded_len, n_entries)
            return param
        params = jax.tree_util.tree_map_with_path(_reinit_cb, params)

        # Populate cache with initial quantized deltas
        from nativebit_jax.layers import requantize_params
        params, _ = requantize_params(params)

    # FSDP sharding (auto-enabled on multi-device)
    params, mesh = setup_fsdp(params)
    import gc; gc.collect()  # Free pre-shard allocations
    data_sharding = NamedSharding(mesh, P('fsdp', None)) if mesh else None

    def _to_device(x, y):
        if data_sharding is not None:
            return jax.device_put(x, data_sharding), jax.device_put(y, data_sharding)
        return x, y

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Params: {n_params / 1e6:.1f}M")
    print(f"  Steps: {config.max_steps}")

    # Optimizer + state
    tx = make_optimizer(config)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Data
    dataset = getattr(config, "dataset", "wikitext-2")
    train_tokens, valid_tokens, test_tokens = load_tokens(dataset, data_dir)

    # Compile train step (single function — no branching)
    ema_decay = getattr(config, "ema_decay", 0.999)
    requantize_every = getattr(config, "requantize_every", 10)
    train_step_fn = make_train_step(model)

    # Logger
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_path / f"{experiment_name}.jsonl"

    # Write header
    with open(jsonl_path, "a") as f:
        header = {
            "type": "header", "backend": "jax",
            "max_steps": config.max_steps,
            "use_nativebit": use_nativebit,
            "n_codebook": config.n_codebook,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "dataset": dataset,
        }
        f.write(json.dumps(header) + "\n")

    # --- Preflight: measure throughput on THROWAWAY state ---
    print("  Preflight: compiling + measuring throughput...", flush=True)
    rng, preflight_rng = jax.random.split(rng)
    batch_iter = make_batches(train_tokens, config.context_len, config.batch_size, preflight_rng)
    preflight_x, preflight_y = next(batch_iter)
    preflight_x, preflight_y = _to_device(preflight_x, preflight_y)

    # Use a copy — don't corrupt the real state
    pf_state = state
    pf_state, _ = train_step_fn(pf_state, preflight_x, preflight_y)  # compile
    jax.block_until_ready(pf_state)

    t0 = time.time()
    for _ in range(50):
        pf_state, _ = train_step_fn(pf_state, preflight_x, preflight_y)
    jax.block_until_ready(pf_state)
    t1 = time.time()
    preflight_sps = 50 / (t1 - t0)
    eta_h = config.max_steps / preflight_sps / 3600
    print(f"  Preflight: {preflight_sps:.1f} steps/s, ETA={eta_h:.1f}h")
    del pf_state  # discard — real state is untouched

    # --- Checkpointing setup ---
    ckpt_every = getattr(config, "checkpoint_every", 500)
    ckpt_dir = os.path.abspath(os.path.join(log_dir, f"{experiment_name}_ckpt"))
    checkpointer = ocp.StandardCheckpointer()
    resume_step = 0

    # Resume from checkpoint if available
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        latest = sorted(ckpt_path.iterdir())
        if latest:
            latest_step = int(latest[-1].name)
            abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
            state = checkpointer.restore(str(latest[-1]), target=abstract_state)
            resume_step = latest_step
            print(f"  Resumed from checkpoint at step {resume_step}")

    # --- Main loop ---
    step = resume_step
    start_time = time.time()
    last_log_time = start_time
    last_log_step = step
    epoch = 0
    last_batch = None  # Track last real batch for revival
    last_updates = None  # Cache intermediates for inter-requantize EMA

    while step < config.max_steps:
        epoch += 1
        data_rng, epoch_rng = jax.random.split(data_rng)

        for x_batch, y_batch in make_batches(train_tokens, config.context_len,
                                              config.batch_size, epoch_rng):
            x_batch, y_batch = _to_device(x_batch, y_batch)

            # External requantize: update cached deltas every N steps
            delay_quant = getattr(config, "delay_quant_steps", 0)
            quant_active = use_nativebit and step >= delay_quant
            need_requantize = (quant_active and
                               (step % requantize_every == 0 or step == delay_quant))

            if need_requantize:
                from nativebit_jax.layers import requantize_params
                new_params, _ = requantize_params(state.params, ema_decay)
                state = state.replace(params=new_params)
                jax.block_until_ready(state)  # force XLA to free old buffers

            state, loss = train_step_fn(state, x_batch, y_batch)

            last_batch = x_batch  # keep for revival

            # Dead entry revival (outside jit, every revive_every steps)
            if (use_nativebit and step > 0
                    and step % config.revive_every == 0):
                state = _revive_all(state, model, last_batch)

            if step % config.log_every == 0:
                loss_val = float(loss)
                ppl = math.exp(min(loss_val, 20))
                now = time.time()
                elapsed = now - start_time
                dt = now - last_log_time
                ds = step - last_log_step
                instant_sps = ds / dt if dt > 0 and ds > 0 else 0
                last_log_time = now
                last_log_step = step

                record = {
                    "step": step, "loss": round(loss_val, 6),
                    "perplexity": round(ppl, 2),
                    "elapsed_s": round(elapsed, 1),
                    "steps_per_sec": round(instant_sps, 1),
                }
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                print(f"  step={step:>5d}  loss={loss_val:.4f}  ppl={ppl:>10.2f}  "
                      f"sps={instant_sps:.1f}", flush=True)

                # Fast fail
                if math.isnan(loss_val) or loss_val > 1000:
                    raise RuntimeError(
                        f"Training diverged at step {step}: loss={loss_val}. "
                        f"Check LR, weight_decay, or init scaling."
                    )

            # Periodic checkpoint (numpy — orbax incompatible with FSDP mesh)
            if step > 0 and step % ckpt_every == 0:
                import numpy as np
                ckpt_file = os.path.join(log_dir, f"{experiment_name}_step{step}.npz")
                flat = {
                    "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path): val
                    for path, val in jax.tree_util.tree_leaves_with_path(state.params)
                }
                np.savez(ckpt_file, **{k: np.array(v) for k, v in flat.items()})
                print(f"  Checkpoint: {ckpt_file}", flush=True)
                # Keep only last 2
                import glob as glob_mod
                ckpts = sorted(glob_mod.glob(os.path.join(log_dir, f"{experiment_name}_step*.npz")))
                for old in ckpts[:-2]:
                    os.remove(old)

            step += 1
            if step >= config.max_steps:
                break

    # --- Eval ---
    print(f"\nEval...")
    eval_step_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    total_loss = 0.0
    n_batches = 0
    for x_batch, y_batch in make_batches(test_tokens, config.context_len,
                                          config.batch_size, eval_rng):
        x_batch, y_batch = _to_device(x_batch, y_batch)
        loss = eval_step_fn(state.params, x_batch, y_batch)
        total_loss += float(loss)
        n_batches += 1

    test_loss = total_loss / max(n_batches, 1)
    test_ppl = math.exp(min(test_loss, 20))
    print(f"  Test loss: {test_loss:.4f}  Test PPL: {test_ppl:.2f}")
    print(f"  Total time: {time.time() - start_time:.0f}s")

    # Save checkpoint (params as numpy arrays for portability)
    ckpt_path = os.path.join(log_dir, f"{experiment_name}_params.npz")
    flat_params = {
        "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path): val
        for path, val in jax.tree_util.tree_leaves_with_path(state.params)
    }
    import numpy as np
    np.savez(ckpt_path, **{k: np.array(v) for k, v in flat_params.items()})
    print(f"  Checkpoint: {ckpt_path}")

    return {"test_loss": test_loss, "test_ppl": test_ppl, "params": state.params}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NativeBit (JAX)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="nativebit_jax")
    parser.add_argument("--no-nativebit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "tpu-small", "tpu-medium", "tpu-large",
                                 "tpu-xl", "tpu-2b"])
    parser.add_argument("--use-aqt", action="store_true",
                        help="Use AQT INT8 matmuls (requires aqt package)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch_size (for memory-constrained TPUs)")
    args = parser.parse_args()

    from configs.default import DefaultConfig
    config_map = {"default": DefaultConfig}

    if args.config.startswith("tpu"):
        from configs.tpu import (TPUSmallConfig, TPUMediumConfig, TPULargeConfig,
                                  TPUXLConfig, TPU2BConfig)
        config_map.update({
            "tpu-small": TPUSmallConfig,
            "tpu-medium": TPUMediumConfig,
            "tpu-large": TPULargeConfig,
            "tpu-xl": TPUXLConfig,
            "tpu-2b": TPU2BConfig,
        })

    config = config_map[args.config]()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.seed = args.seed

    use_nativebit = not args.no_nativebit
    train(config, use_nativebit=use_nativebit, use_aqt=args.use_aqt,
          experiment_name=args.name, log_dir=args.log_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
