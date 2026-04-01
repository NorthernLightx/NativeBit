"""JAX/TPU evaluator for autoresearch — time-based budgets.

Replaces evaluator.py for TPU runs. Each trial gets a fixed wall-clock
budget instead of fixed steps. This naturally rewards faster configs.
"""

import math
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from nativebit_jax.model import build_model, apply_init_scaling
from nativebit_jax.train import (
    make_optimizer, make_train_step, make_eval_step,
    load_tokens, make_batches, _ema_update_params,
)
from nativebit_jax.layers import _init_codebook_from_weight

from .trial import Trial


# Time budgets (seconds)
SCREEN_TIME = 0         # Skip screen — go straight to validate
VALIDATE_TIME = 300     # 5 min validate
CONFIRM_SEEDS = [42, 123, 7]

# Acceptance thresholds
SCREEN_THRESHOLD = 1.05
VALIDATE_THRESHOLD = 1.00


class JaxTrialConfig:
    """Dynamic config built from autoresearch parameters.

    Scaled for TPUMediumConfig (125M params).
    """
    # Model (fixed — 125M)
    n_layers: int = 12
    n_embd: int = 768
    n_head: int = 12
    ffn_hidden: int = 3072
    context_len: int = 1024
    vocab_size: int = 50257

    # Training (fixed)
    batch_size: int = 8  # no remat, limited memory
    max_steps: int = 999999  # time-based, not step-based
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 50
    log_every: int = 100
    dataset: str = "wikitext-103"
    seed: int = 42

    # Searchable NativeBit params (set from autoresearch config)
    n_codebook: int = 8
    block_size: int = 128
    ema_decay: float = 0.999
    lr: float = 6e-4
    codebook_lr: float = 6e-5
    weight_decay: float = 0.1
    delay_quant_steps: int = 500
    requantize_every: int = 10

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Derive codebook_lr from lr
        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
            self.codebook_lr = self.lr * 0.1


def train_timed(config, time_budget: float, seed: int = 42,
                log_dir: str = "logs/autoresearch",
                data_dir: str = "data") -> dict:
    """Train for a fixed wall-clock budget. Returns metrics dict."""
    config.seed = seed

    model = build_model(config, use_nativebit=True)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    x_dummy = jnp.ones((1, config.context_len), dtype=jnp.int32)
    variables = model.init(init_rng, x_dummy, requantize=True)
    variables = apply_init_scaling(variables, config.n_layers)

    # Re-init codebooks from scaled weights
    import math as _math
    def _reinit_cb(path, param):
        path_str = "/".join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        if path_str.endswith("/codebook"):
            weight_path = path_str.rsplit("/codebook", 1)[0] + "/weight"
            node = variables["params"]
            for key in weight_path.split("/"):
                if key and key in node:
                    node = node[key]
            weight = node
            num_blocks, n_entries = param.shape
            total_weights = weight.size
            padded_len = num_blocks * (_math.ceil(total_weights / num_blocks))
            return _init_codebook_from_weight(
                weight, _math.ceil(total_weights / num_blocks),
                num_blocks, total_weights, padded_len, n_entries)
        return param
    variables = jax.tree_util.tree_map_with_path(_reinit_cb, variables)

    tx = make_optimizer(config)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables, tx=tx,
    )

    train_tokens, valid_tokens, test_tokens = load_tokens(config.dataset, data_dir)
    fast_step, rq_step = make_train_step(model, use_nativebit=True)

    ema_decay = config.ema_decay
    rq_every = config.requantize_every
    delay = config.delay_quant_steps

    # Compile
    rng, compile_rng = jax.random.split(rng)
    batch_iter = make_batches(train_tokens, config.context_len, config.batch_size, compile_rng)
    bx, by = next(batch_iter)
    state, _, _ = rq_step(state, bx, by)
    state, _ = fast_step(state, bx, by)
    jax.block_until_ready(state)

    # Train for time_budget seconds
    step = 0
    epoch = 0
    last_updates = None
    t_start = time.time()

    while time.time() - t_start < time_budget:
        epoch += 1
        data_rng, epoch_rng = jax.random.split(data_rng)

        for xb, yb in make_batches(train_tokens, config.context_len,
                                    config.batch_size, epoch_rng):
            if time.time() - t_start >= time_budget:
                break

            quant_active = step >= delay
            is_rq = quant_active and (step % rq_every == 0 or step == delay)

            if is_rq:
                state, loss, updates = rq_step(state, xb, yb)
                new_params = _ema_update_params(state.params, updates, ema_decay)
                state = state.replace(params=new_params)
                last_updates = updates
            else:
                state, loss = fast_step(state, xb, yb)
                if quant_active and last_updates is not None:
                    new_params = _ema_update_params(state.params, last_updates, ema_decay)
                    state = state.replace(params=new_params)

            step += 1

    elapsed = time.time() - t_start
    train_loss = float(loss)

    # Eval
    eval_step_fn = make_eval_step(model)
    eval_rng = jax.random.PRNGKey(0)
    total_loss, n_batches = 0.0, 0
    for xb, yb in make_batches(test_tokens, config.context_len,
                                config.batch_size, eval_rng):
        l = eval_step_fn(state.params, xb, yb)
        total_loss += float(l)
        n_batches += 1
        if n_batches >= 50:
            break

    test_loss = total_loss / max(n_batches, 1)
    test_ppl = math.exp(min(test_loss, 20))

    return {
        "test_ppl": test_ppl,
        "test_loss": test_loss,
        "train_loss": train_loss,
        "steps": step,
        "elapsed_s": elapsed,
        "steps_per_sec": step / elapsed,
    }


def evaluate_trial_jax(trial: Trial, best_ppl: float,
                       log_dir: str = "logs/autoresearch",
                       data_dir: str = "data",
                       best_screen_ppl: float = 0.0) -> Trial:
    """Run 3-phase evaluation: screen → validate → confirm."""
    config = JaxTrialConfig(**trial.config)

    # Phase 1: Validate (5 min) — skip screen, go straight to validation
    print(f"  Validate ({VALIDATE_TIME}s)...", flush=True)
    result = train_timed(config, VALIDATE_TIME, seed=42, log_dir=log_dir, data_dir=data_dir)
    trial.validate_ppl = result["test_ppl"]
    trial.validate_steps = result["steps"]
    trial.validate_sps = result["steps_per_sec"]
    trial.config["_steps_per_sec"] = round(result["steps_per_sec"], 1)
    print(f"    PPL={result['test_ppl']:.1f}, steps={result['steps']}, "
          f"sps={result['steps_per_sec']:.1f}")

    threshold = best_ppl * VALIDATE_THRESHOLD if best_ppl > 0 else float("inf")
    if result["test_ppl"] > threshold:
        trial.status = "rejected_validate"
        return trial

    # Phase 3: Confirm (3 seeds × 10 min)
    print(f"  Confirm (3 seeds × {VALIDATE_TIME}s)...", flush=True)
    ppls = []
    for seed in CONFIRM_SEEDS:
        r = train_timed(config, VALIDATE_TIME, seed=seed, log_dir=log_dir, data_dir=data_dir)
        ppls.append(r["test_ppl"])
        print(f"    seed={seed}: PPL={r['test_ppl']:.1f}")

    trial.confirm_ppl = sum(ppls) / len(ppls)
    trial.confirm_std = (sum((p - trial.confirm_ppl)**2 for p in ppls) / len(ppls)) ** 0.5
    trial.status = "accepted"
    return trial
