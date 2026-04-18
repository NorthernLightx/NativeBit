"""JAX/Flax training loop for NativeBit — native TPU support.

Usage:
    # On TPU (auto-detected):
    python -m nativebit_jax.train --config tpu-small --name jax_bench

    # Quick local test:
    python -m nativebit_jax.train --max-steps 100 --name test
"""

import argparse
import functools
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
from nativebit_jax.codebook_utils import ema_update_codebooks
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

        import numpy as np
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                a = array.array("i")
                a.frombytes(f.read())
            results[our_name] = np.frombuffer(a, dtype=np.int32).copy()
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
        results[our_name] = np.array(tokens, dtype=np.int32)

    train, valid, test = results["train"], results["valid"], results["test"]
    print(f"  [{dataset}] Train: {len(train)} tok, Valid: {len(valid)}, Test: {len(test)}")
    return train, valid, test


def make_batches(tokens, ctx_len: int, batch_size: int, rng: jax.Array):
    """Yield (x, y) batches from token array, shuffled.

    Tokens stay as numpy on host. Each batch is converted to jnp on yield.
    Shuffle uses numpy (seeded from JAX rng) to avoid device allocation.
    """
    import numpy as np
    tokens = np.asarray(tokens)
    n_seq = (len(tokens) - 1) // ctx_len
    all_tokens = tokens[:n_seq * ctx_len + 1]
    all_x = all_tokens[:-1].reshape(n_seq, ctx_len)
    all_y = all_tokens[1:].reshape(n_seq, ctx_len)

    # Use JAX rng to seed numpy shuffle (stays on host)
    seed = int(jax.random.bits(rng, (), dtype=jnp.uint32)) % (2**31)
    np_rng = np.random.RandomState(seed)
    perm = np_rng.permutation(n_seq)
    all_x = all_x[perm]
    all_y = all_y[perm]

    n_batches = n_seq // batch_size
    for i in range(n_batches):
        s = i * batch_size
        yield jnp.array(all_x[s:s + batch_size]), jnp.array(all_y[s:s + batch_size])


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


def make_train_step(model: NativeBitGPT, use_quant_reg: bool = False):
    """Create a single jit-compiled train step.

    Same function for float and NativeBit — no branching. NativeBit layers
    read cached deltas; cache is updated externally by requantize_params().

    If use_quant_reg=True, adds lam * mean(||w - sg(Q(w))||^2) to the loss.
    Lam is passed per-step so the schedule can vary during training.
    """
    from nativebit_jax.layers import compute_quant_reg

    if use_quant_reg:
        @jax.jit
        def train_step(state, x, y, lam):
            def loss_fn(params):
                logits = model.apply(params, x)
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                ).mean()
                reg = compute_quant_reg(params)
                return ce + lam * reg, (ce, reg)
            (loss, (ce, reg)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, ce, reg

        return train_step

    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
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

def _get_gcs_bucket():
    """Get or create a same-region GCS bucket for checkpoint sync.

    CRITICAL: bucket must be in the same region as the TPU to avoid
    cross-region transfer costs ($0.01-0.12/GB). Same-region is free.
    """
    import subprocess
    # Detect region from TPU VM metadata
    try:
        zone = subprocess.run(
            "curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google'",
            shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        region = "-".join(zone.split("/")[-1].split("-")[:-1])  # e.g. europe-west4
        project = subprocess.run(
            "curl -s http://metadata.google.internal/computeMetadata/v1/project/project-id -H 'Metadata-Flavor: Google'",
            shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None  # Not on GCP

    bucket = f"gs://{project}-nativebit-{region}"
    # Create if doesn't exist (idempotent)
    subprocess.run(f"gsutil mb -l {region} {bucket} 2>/dev/null; true",
                   shell=True, timeout=15)
    return bucket


def _get_git_hash():
    """Best-effort git HEAD sha; '' if not a git repo or git unavailable."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=3,
                           cwd=str(Path(__file__).resolve().parent.parent))
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _config_to_dict(config):
    """Dump config class attributes (supports dataclasses and plain classes)."""
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(config):
            return asdict(config)
    except Exception:
        pass
    out = {}
    for k in dir(config):
        if k.startswith("_"):
            continue
        v = getattr(config, k, None)
        if callable(v):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
    return out


def train(config, use_nativebit: bool = True, use_aqt: bool = False,
          experiment_name: str = "nativebit_jax",
          log_dir: str = "logs", data_dir: str = "data",
          init_from: str = None, val_every: int = 1000,
          argv: list = None):
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

        # If using canonical VQ-VAE EMA, seed `ema_s` with current codebook
        # values and `ema_N` with ones so that s/N = cb_init at step 0.
        if getattr(config, "use_canonical_ema", False):
            from nativebit_jax.layers import init_canonical_ema_state
            params = init_canonical_ema_state(params)

    # QAT: load weights from a pre-trained checkpoint BEFORE training starts.
    # Codebooks are then re-initialised from those loaded weights so the
    # quantizer is fitted to the real trained distribution, not random init.
    #
    # Handles the format mismatch between float-trained checkpoints (Flax
    # nn.Dense uses "Dense_N/kernel" with shape (in, out)) and NB-trained
    # checkpoints ("NativeBitDense_N/weight" with shape (out, in)).
    if init_from is not None:
        print(f"  QAT init: loading weights from {init_from}", flush=True)
        import numpy as np
        ckpt_data = np.load(init_from)
        ckpt_keys = set(ckpt_data.files)
        loaded_keys = set()

        def _candidate_keys(path_str: str, leaf_shape):
            """Yield (ckpt_key, needs_transpose) candidates for a param."""
            yield path_str, False
            if "/NativeBitDense_" in path_str and path_str.endswith("/weight"):
                translated = (path_str.replace("/NativeBitDense_", "/Dense_")
                              .replace("/weight", "/kernel"))
                yield translated, True

        def _load_leaf(path, leaf):
            key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            for cand, needs_transpose in _candidate_keys(key, leaf.shape):
                if cand in ckpt_keys:
                    arr = np.asarray(ckpt_data[cand])
                    if needs_transpose:
                        arr = arr.T
                    if arr.shape != leaf.shape:
                        print(f"    WARNING: shape mismatch for {key}: "
                              f"ckpt has {arr.shape}, expected {leaf.shape} "
                              f"(from {cand}). Skipping.", flush=True)
                        return leaf
                    loaded_keys.add(key)
                    return jnp.array(arr)
            return leaf

        params = jax.tree_util.tree_map_with_path(_load_leaf, params)

        total_leaves = sum(1 for _ in jax.tree_util.tree_leaves(params))
        print(f"  QAT init: loaded {len(loaded_keys)}/{total_leaves} leaves",
              flush=True)
        if len(loaded_keys) < total_leaves * 0.1:
            raise RuntimeError(
                f"QAT init loaded only {len(loaded_keys)}/{total_leaves} "
                f"leaves — checkpoint format probably doesn't match. "
                f"Sample ckpt keys: {list(ckpt_keys)[:5]}")
        del ckpt_data; import gc; gc.collect()

        # Re-init codebooks from LOADED weights, then repopulate cache.
        if use_nativebit:
            params = jax.tree_util.tree_map_with_path(_reinit_cb, params)
            params, _ = requantize_params(params)
            if getattr(config, "use_canonical_ema", False):
                from nativebit_jax.layers import init_canonical_ema_state
                params = init_canonical_ema_state(params)
        print(f"  QAT init: done.", flush=True)

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

    # GCS bucket (same-region, auto-created)
    gcs_bucket = _get_gcs_bucket()
    if gcs_bucket:
        print(f"  GCS: {gcs_bucket}")

    # Optimizer + state
    tx = make_optimizer(config)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Data
    dataset = getattr(config, "dataset", "wikitext-2")
    train_tokens, valid_tokens, test_tokens = load_tokens(dataset, data_dir)

    # Compile train step (single function — no branching)
    ema_decay = getattr(config, "ema_decay", 0.999)
    requantize_every = getattr(config, "requantize_every", 10)
    quant_reg_weight = getattr(config, "quant_reg_weight", 0.0) if use_nativebit else 0.0
    quant_reg_warmup_frac = getattr(config, "quant_reg_warmup_frac", 0.25)
    use_quant_reg = quant_reg_weight > 0.0
    use_canonical_ema = getattr(config, "use_canonical_ema", False) and use_nativebit
    train_step_fn = make_train_step(model, use_quant_reg=use_quant_reg)

    def lambda_schedule(step: int) -> float:
        """Linear warmup from 0 to quant_reg_weight over warmup_frac of training."""
        if not use_quant_reg:
            return 0.0
        warmup_steps = max(int(config.max_steps * quant_reg_warmup_frac), 1)
        return quant_reg_weight * min(1.0, step / warmup_steps)

    # Logger
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_path / f"{experiment_name}.jsonl"

    # Write header (schema v2: adds git_hash, argv, full config, init_from)
    with open(jsonl_path, "a") as f:
        header = {
            "type": "header", "schema_version": 2, "backend": "jax",
            "use_nativebit": use_nativebit,
            "dataset": dataset,
            "config": _config_to_dict(config),
            "git_hash": _get_git_hash(),
            "argv": argv or [],
            "init_from": init_from,
            # Redundant top-level copies for quick grep (kept for back-compat):
            "max_steps": config.max_steps,
            "n_codebook": config.n_codebook,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "quant_reg_weight": quant_reg_weight,
            "quant_reg_warmup_frac": quant_reg_warmup_frac,
            "use_canonical_ema": use_canonical_ema,
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
    if use_quant_reg:
        pf_state, *_ = train_step_fn(pf_state, preflight_x, preflight_y,
                                     jnp.float32(0.0))  # compile
    else:
        pf_state, _ = train_step_fn(pf_state, preflight_x, preflight_y)
    jax.block_until_ready(pf_state)

    t0 = time.time()
    for _ in range(50):
        if use_quant_reg:
            pf_state, *_ = train_step_fn(pf_state, preflight_x, preflight_y,
                                         jnp.float32(0.0))
        else:
            pf_state, _ = train_step_fn(pf_state, preflight_x, preflight_y)
    jax.block_until_ready(pf_state)
    t1 = time.time()
    preflight_sps = 50 / (t1 - t0)
    eta_h = config.max_steps / preflight_sps / 3600
    print(f"  Preflight: {preflight_sps:.1f} steps/s, ETA={eta_h:.1f}h")
    del pf_state  # discard — real state is untouched

    # --- Checkpointing setup ---
    ckpt_every = getattr(config, "checkpoint_every", 500)
    resume_step = 0

    # Resume from numpy checkpoint (local or GCS)
    import glob as glob_mod
    import subprocess
    # Try downloading latest checkpoint from GCS
    try:
        if gcs_bucket is None:
            raise RuntimeError("No GCS bucket")
        gcs_ckpts = subprocess.run(
            f"gsutil ls {gcs_bucket}/{experiment_name}_step*.npz",
            shell=True, capture_output=True, text=True, timeout=10)
        if gcs_ckpts.returncode == 0 and gcs_ckpts.stdout.strip():
            latest_gcs = sorted(gcs_ckpts.stdout.strip().split('\n'))[-1]
            local_ckpt = os.path.join(log_dir, os.path.basename(latest_gcs))
            if not os.path.exists(local_ckpt):
                subprocess.run(f"gsutil cp {latest_gcs} {local_ckpt}",
                               shell=True, timeout=120)
    except Exception:
        pass

    # Find latest local numpy checkpoint
    local_ckpts = sorted(glob_mod.glob(os.path.join(log_dir, f"{experiment_name}_step*.npz")))
    if local_ckpts:
        latest_ckpt = local_ckpts[-1]
        resume_step = int(latest_ckpt.split("_step")[-1].replace(".npz", ""))
        print(f"  Resuming from {latest_ckpt} (step {resume_step})...", flush=True)
        import numpy as np
        ckpt_data = np.load(latest_ckpt)
        # Rebuild params tree from flat dict
        def _set_leaf(path, leaf):
            key = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            if key in ckpt_data:
                return jnp.array(ckpt_data[key])
            return leaf
        restored_params = jax.tree_util.tree_map_with_path(_set_leaf, state.params)
        # Re-shard restored params
        if mesh is not None:
            def _reshard(old, new):
                if hasattr(old, 'sharding'):
                    return jax.device_put(new, old.sharding)
                return new
            restored_params = jax.tree.map(_reshard, state.params, restored_params)
        state = state.replace(params=restored_params)
        del ckpt_data, restored_params; gc.collect()
        print(f"  Resumed at step {resume_step}")

    # --- Initial eval (BEFORE any training step) -----------------------------
    # For QAT, this is the "post-hoc RTN" baseline at the loaded weights.
    # For from-scratch, it's a sanity check that the init is sensible.
    eval_step_fn = make_eval_step(model)

    def _eval_subset(tokens, max_batches: int = None):
        rng = jax.random.PRNGKey(0)
        total, n = 0.0, 0
        for xb, yb in make_batches(tokens, config.context_len,
                                   config.batch_size, rng):
            xb, yb = _to_device(xb, yb)
            total += float(eval_step_fn(state.params, xb, yb))
            n += 1
            if max_batches and n >= max_batches:
                break
        return (total / max(n, 1), n)

    if resume_step == 0:
        # Make sure cache reflects current weights before eval (for QAT from float).
        if use_nativebit:
            from nativebit_jax.layers import requantize_params
            fresh_params, _ = requantize_params(
                state.params, ema_decay, use_canonical_ema=use_canonical_ema)
            state = state.replace(params=fresh_params)

        init_loss, init_n = _eval_subset(valid_tokens, max_batches=64)
        init_ppl = math.exp(min(init_loss, 20))
        print(f"  Initial validation PPL (step 0, {init_n} batches): "
              f"{init_ppl:.2f}", flush=True)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({
                "type": "init_eval", "step": 0,
                "val_ppl": round(init_ppl, 2),
                "val_loss": round(init_loss, 6),
                "val_batches": init_n,
            }) + "\n")

    # --- Main loop ---
    from nativebit_jax.layers import compute_quant_diagnostics
    _diag_jitted = jax.jit(compute_quant_diagnostics) if use_nativebit else None

    step = resume_step
    start_time = time.time()
    last_log_time = start_time
    last_log_step = step
    epoch = 0

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
                new_params, _ = requantize_params(
                    state.params, ema_decay,
                    use_canonical_ema=use_canonical_ema)
                state = state.replace(params=new_params)
                jax.block_until_ready(state)  # force XLA to free old buffers

            if use_quant_reg:
                lam = jnp.float32(lambda_schedule(step))
                state, ce_loss, reg_loss = train_step_fn(state, x_batch, y_batch, lam)
                loss = ce_loss  # what we log as training loss / use for PPL
            else:
                state, loss = train_step_fn(state, x_batch, y_batch)
                ce_loss, reg_loss = loss, jnp.float32(0.0)

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
                if use_quant_reg:
                    record["quant_reg"] = round(float(reg_loss), 8)
                    record["lambda"] = round(float(lam), 6)
                # Quantization diagnostics — computed lazily, log every N steps
                # (cheap: one argmin pass over params, reuses jit cache).
                if _diag_jitted is not None:
                    diag = _diag_jitted(state.params)
                    record["quant_err_rms"] = round(float(diag["quant_error_rms"]), 6)
                    record["cb_utilization"] = round(float(diag["codebook_utilization"]), 4)
                    record["dead_frac"] = round(float(diag["dead_entries_frac"]), 4)
                # Periodic validation on held-out valid set (small subset for speed)
                if step > 0 and step % val_every == 0:
                    v_loss, v_n = _eval_subset(valid_tokens, max_batches=32)
                    record["val_loss"] = round(v_loss, 6)
                    record["val_ppl"] = round(math.exp(min(v_loss, 20)), 2)
                    record["val_batches"] = v_n
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                # Sync log to GCS every 500 steps
                if step % 500 == 0 and gcs_bucket:
                    import subprocess
                    subprocess.Popen(f"gsutil -q cp {jsonl_path} {gcs_bucket}/",
                                     shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                extra = ""
                if use_quant_reg:
                    extra += (f"  reg={float(reg_loss):.2e}"
                              f"  lam={float(lam):.4f}")
                if "quant_err_rms" in record:
                    extra += (f"  qerr={record['quant_err_rms']:.4f}"
                              f"  util={record['cb_utilization']:.3f}")
                if "val_ppl" in record:
                    extra += f"  val_ppl={record['val_ppl']:.2f}"
                print(f"  step={step:>5d}  loss={loss_val:.4f}  ppl={ppl:>10.2f}  "
                      f"sps={instant_sps:.1f}{extra}", flush=True)

                # Fast fail
                if math.isnan(loss_val) or loss_val > 1000:
                    raise RuntimeError(
                        f"Training diverged at step {step}: loss={loss_val}. "
                        f"Check LR, weight_decay, or init scaling."
                    )

            # Periodic checkpoint (numpy — orbax incompatible with FSDP mesh)
            if step > 0 and step % ckpt_every == 0:
                import numpy as np
                import subprocess
                ckpt_file = os.path.join(log_dir, f"{experiment_name}_step{step}.npz")
                flat = {
                    "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path): val
                    for path, val in jax.tree_util.tree_leaves_with_path(state.params)
                }
                np.savez(ckpt_file, **{k: np.array(v) for k, v in flat.items()})
                print(f"  Checkpoint: {ckpt_file}", flush=True)
                # Sync to GCS: upload new, delete old (keep only latest)
                if gcs_bucket:
                    gcs_base = f"{gcs_bucket}/"
                    subprocess.Popen(
                        f"gsutil -q cp {ckpt_file} {gcs_base} && "
                        f"gsutil -q rm {gcs_base}{experiment_name}_step{step - ckpt_every}.npz 2>/dev/null; true",
                        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Keep only last 2 locally
                import glob as glob_mod
                ckpts = sorted(glob_mod.glob(os.path.join(log_dir, f"{experiment_name}_step*.npz")))
                for old in ckpts[:-2]:
                    os.remove(old)

            step += 1
            if step >= config.max_steps:
                break

    # --- Eval ---
    # Refresh cached qw_delta from current latent weights so eval uses true
    # Q(w_final) instead of a stale delta up to `requantize_every` steps old.
    if use_nativebit:
        from nativebit_jax.layers import requantize_params
        fresh_params, _ = requantize_params(
            state.params, ema_decay, use_canonical_ema=use_canonical_ema)
        state = state.replace(params=fresh_params)

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
    elapsed_total = time.time() - start_time
    print(f"  Test loss: {test_loss:.4f}  Test PPL: {test_ppl:.2f}")
    print(f"  Total time: {elapsed_total:.0f}s")

    # Write eval results to JSONL (so they survive even if stdout is lost)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps({
            "type": "eval", "test_loss": round(test_loss, 6),
            "test_ppl": round(test_ppl, 2), "total_time_s": round(elapsed_total, 0),
        }) + "\n")

    # Save final checkpoint
    import numpy as np
    ckpt_path = os.path.join(log_dir, f"{experiment_name}_params.npz")
    flat_params = {
        "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path): val
        for path, val in jax.tree_util.tree_leaves_with_path(state.params)
    }
    np.savez(ckpt_path, **{k: np.array(v) for k, v in flat_params.items()})
    print(f"  Checkpoint: {ckpt_path}")

    # Sync final results to GCS (blocking — don't exit before upload finishes)
    if gcs_bucket:
        import subprocess
        for f in [str(jsonl_path), ckpt_path]:
            subprocess.run(f"gsutil -q cp {f} {gcs_bucket}/",
                           shell=True, timeout=300)
        # Clean up intermediate checkpoints from GCS
        subprocess.run(
            f"gsutil -q rm {gcs_bucket}/{experiment_name}_step*.npz 2>/dev/null; true",
            shell=True, timeout=60)

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
    parser.add_argument("--quant-reg-weight", type=float, default=None,
                        help="VQ-VAE-style commitment loss weight (pulls w "
                             "toward nearest codebook entry). 0 = disabled.")
    parser.add_argument("--quant-reg-warmup-frac", type=float, default=None,
                        help="Fraction of training to linearly warm lam from 0 "
                             "to quant_reg_weight (default from config).")
    parser.add_argument("--use-canonical-ema", action="store_true",
                        help="Canonical VQ-VAE EMA: EMA raw sums and counts, "
                             "then derive codebook = s/N. Default is "
                             "EMA-of-batch-means.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Path to an .npz checkpoint to initialise weights "
                             "from (e.g. a trained float baseline, for QAT). "
                             "Codebooks are re-initialised from the loaded "
                             "weight distribution.")
    parser.add_argument("--val-every", type=int, default=1000,
                        help="Run validation PPL every N steps (default 1000).")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Override LR warmup steps (default from config).")
    parser.add_argument("--delay-quant-steps", type=int, default=None,
                        help="Override delay_quant_steps. Set to 0 for QAT "
                             "from a trained float checkpoint so quantization "
                             "is active from step 0.")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="Override codebook EMA decay (default 0.999). "
                             "Lower values = faster adapt, useful for short "
                             "QAT runs.")
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
    if args.quant_reg_weight is not None:
        config.quant_reg_weight = args.quant_reg_weight
    if args.quant_reg_warmup_frac is not None:
        config.quant_reg_warmup_frac = args.quant_reg_warmup_frac
    if args.use_canonical_ema:
        config.use_canonical_ema = True
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.delay_quant_steps is not None:
        config.delay_quant_steps = args.delay_quant_steps
    if args.ema_decay is not None:
        config.ema_decay = args.ema_decay
    config.seed = args.seed

    use_nativebit = not args.no_nativebit
    train(config, use_nativebit=use_nativebit, use_aqt=args.use_aqt,
          experiment_name=args.name, log_dir=args.log_dir, data_dir=args.data_dir,
          init_from=args.init_from, val_every=args.val_every, argv=sys.argv)


if __name__ == "__main__":
    main()
