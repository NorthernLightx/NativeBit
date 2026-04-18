"""NativeBitDense — JAX/Flax port of NativeBitLinear.

Drop-in replacement for nn.Dense with learned per-block codebook quantization.
STE for weight gradients; codebook updated via EMA (not gradients).
Optional AQT (Accurate Quantized Training) for INT8 matmuls on TPU.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from .codebook_utils import ema_update_codebooks

from .codebook_utils import init_codebook_percentile

# Optional AQT import — gracefully degrade if not installed
_aqt_available = False
try:
    from aqt.jax.v2 import config as aqt_config
    from aqt.jax.v2.flax import AqtDotGeneral
    _aqt_available = True
except ImportError:
    pass


def _quantize(weight_flat: jnp.ndarray, codebook: jnp.ndarray,
              block_size: int, num_blocks: int,
              total_weights: int, padded_len: int,
              block_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Quantize weights to nearest codebook entries.

    Pure function — no side effects. Returns (quantized_flat, indices).
    """
    if padded_len > total_weights:
        w_padded = jnp.pad(weight_flat, (0, padded_len - total_weights))
    else:
        w_padded = weight_flat

    w_blocks = w_padded.reshape(num_blocks, block_size)

    dists = jnp.square(w_blocks[:, :, None] - codebook[:, None, :])
    indices = jnp.argmin(dists, axis=-1)

    quantized_blocks = codebook[block_idx, indices]
    quantized_flat = quantized_blocks.reshape(-1)[:total_weights]
    return quantized_flat, indices


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _jitted_requantize_and_ema(weight, codebook, block_size, num_blocks,
                                total_weights, padded_len, ema_decay):
    """Requantize one layer + EMA codebook update.

    Returns (new_delta_bf16, new_codebook, indices).
    """
    w_flat = weight.reshape(-1)
    w_padded = jnp.pad(w_flat, (0, padded_len - total_weights))
    w_blocks = w_padded.reshape(num_blocks, block_size)
    block_idx = jnp.arange(num_blocks)[:, None]

    # Quantize: distance + argmin + gather
    dists = jnp.square(w_blocks[:, :, None] - codebook[:, None, :])
    indices = jnp.argmin(dists, axis=-1)
    quantized_blocks = codebook[block_idx, indices]
    quantized_flat = quantized_blocks.reshape(-1)[:total_weights]

    # Delta for cached forward pass
    delta = (quantized_flat.reshape(weight.shape) - weight).astype(jnp.bfloat16)

    # EMA codebook update
    new_codebook = ema_update_codebooks(codebook, indices, w_blocks, ema_decay)

    return delta, new_codebook, indices


class NativeBitDense(nn.Module):
    """Dense layer with learned per-block codebook quantization.

    Supports cached quantization: when requantize=False, reuses previously
    cached quantized weights instead of recomputing distance+argmin+gather.
    This makes most training steps as fast as float — only paying the
    quantize cost every N steps.

    Attributes:
        features: output dimension.
        use_bias: whether to add a learnable bias.
        block_size: weights per codebook block.
        n_entries: codebook entries per block (8 = 3-bit).
        param_dtype: dtype for stored parameters.
        compute_dtype: dtype for matmul computation (bf16 on TPU).
        use_aqt: INT8 matmuls via AQT.
    """
    features: int
    use_bias: bool = False
    block_size: int = 64
    n_entries: int = 8
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.bfloat16
    use_aqt: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        out_features = self.features

        # Latent weights — gradients flow here via STE
        weight = self.param(
            "weight",
            nn.initializers.kaiming_normal(),
            (out_features, in_features),
            self.param_dtype,
        )

        # Per-block codebook (used by external requantize, not in forward)
        total_weights = out_features * in_features
        num_blocks = math.ceil(total_weights / self.block_size)
        padded_len = num_blocks * self.block_size

        self.param(
            "codebook",
            lambda rng, shape, dtype=None: _init_codebook_from_weight(
                weight, self.block_size, num_blocks, total_weights, padded_len, self.n_entries
            ),
            (num_blocks, self.n_entries),
            self.param_dtype,
        )

        # Cached quantized weight delta — stored in bf16 to save memory
        # (bf16 precision is sufficient for the quantization delta)
        cached_delta = self.variable(
            "cache", "qw_delta",
            lambda: jnp.zeros((out_features, in_features), dtype=jnp.bfloat16),
        )

        # Canonical VQ-VAE EMA state: running sum and count per codebook entry.
        # Used only when requantize_params(..., use_canonical_ema=True).
        # In EMA-of-means (default), these are unused and stay at init values.
        self.variable(
            "cache", "ema_N",
            lambda: jnp.ones((num_blocks, self.n_entries), dtype=jnp.float32),
        )
        self.variable(
            "cache", "ema_s",
            lambda: jnp.zeros((num_blocks, self.n_entries), dtype=jnp.float32),
        )

        # STE: forward uses cached quantized weights, backward flows to latent
        # No branching — remat-safe. Cache updated externally by requantize_params().
        quantized_w = weight + jax.lax.stop_gradient(cached_delta.value)

        # Matmul in compute_dtype
        x_compute = x.astype(self.compute_dtype)
        w_compute = quantized_w.astype(self.compute_dtype)

        if self.use_aqt and _aqt_available:
            aqt_dg = AqtDotGeneral(aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8))
            out = aqt_dg(
                x_compute, w_compute.T,
                dimension_numbers=(((x_compute.ndim - 1,), (0,)), ((), ())),
            )
        else:
            out = x_compute @ w_compute.T

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(),
                              (out_features,), self.param_dtype)
            out = out + bias.astype(self.compute_dtype)

        return out.astype(x.dtype)


class PackedNativeBitDense(nn.Module):
    """Dense layer with on-the-fly weight reconstruction from packed indices.

    Stores uint8 indices + fp32 codebook tables in HBM (~3x less than float).
    Forward: fused codebook gather + matmul via Pallas kernel on TPU,
    with naive fallback on CPU/GPU.
    No latent weights, no STE — inference only.

    Toggle: set NATIVEBIT_KERNEL=naive env var to force fallback.

    Memory per layer: indices (uint8, num_blocks × block_size) +
                      codebook (fp32, num_blocks × n_entries)
    vs float Dense:   kernel (fp32, in × out)
    Ratio: ~3x less for 3-bit (uint8 indices), ~5x with true 3-bit packing.
    """
    features: int
    use_bias: bool = False
    block_size: int = 128
    n_entries: int = 8
    compute_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        from .packed_kernel import packed_matmul

        in_features = x.shape[-1]
        out_features = self.features
        total = out_features * in_features
        num_blocks = math.ceil(total / self.block_size)

        indices = self.param("indices", nn.initializers.zeros_init(),
                             (num_blocks, self.block_size), jnp.uint8)
        codebook = self.param("codebook", nn.initializers.zeros_init(),
                              (num_blocks, self.n_entries), jnp.float32)

        # Flatten batch dims for matmul: (..., in_features) → (M, in_features)
        x_shape = x.shape
        x_2d = x.reshape(-1, in_features)

        out = packed_matmul(x_2d, indices, codebook,
                            out_features, in_features, self.block_size)

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(),
                              (out_features,), jnp.float32)
            out = out + bias.astype(out.dtype)

        # Restore batch dims
        return out.reshape(*x_shape[:-1], out_features).astype(x.dtype)


def _extract_layer_arrays(params_inner, cache):
    """Extract (weights, codebooks, deltas, ema_N, ema_s) + metadata for jit."""
    weights, codebooks, deltas = [], [], []
    ema_Ns, ema_ss, meta = [], [], []
    def _walk(node, cache_node, path=()):
        if isinstance(node, dict):
            if "codebook" in node and "weight" in node:
                weights.append(node["weight"])
                codebooks.append(node["codebook"])
                deltas.append(cache_node.get("qw_delta",
                    jnp.zeros_like(node["weight"], dtype=jnp.bfloat16)))
                nb, ne = node["codebook"].shape
                tw = node["weight"].size
                bs = math.ceil(tw / nb)
                # Canonical-EMA state (fall back to ones / codebook if missing)
                ema_Ns.append(cache_node.get(
                    "ema_N", jnp.ones((nb, ne), dtype=jnp.float32)))
                ema_ss.append(cache_node.get(
                    "ema_s", node["codebook"].astype(jnp.float32)))
                meta.append({"path": path, "block_size": bs, "num_blocks": nb,
                             "total_weights": tw, "padded_len": nb * bs})
            else:
                for k in node:
                    _walk(node[k], cache_node.get(k, {}), path + (k,))
    _walk(params_inner, cache)
    return weights, codebooks, deltas, ema_Ns, ema_ss, meta


def _rebuild_params(params_inner, cache, meta, new_codebooks, new_deltas,
                    new_ema_Ns=None, new_ema_ss=None):
    """Put updated codebooks, deltas, and optional EMA state back into the tree."""
    import copy
    p = copy.copy(params_inner)
    c = copy.copy(cache)
    for i, m in enumerate(meta):
        # Navigate to the right spot and update
        node = p
        cnode = c
        for k in m["path"][:-1]:
            if k not in node or not isinstance(node[k], dict):
                break
            node[k] = dict(node[k])
            if k not in cnode:
                cnode[k] = {}
            cnode[k] = dict(cnode[k])
            node = node[k]
            cnode = cnode[k]
        last = m["path"][-1]
        node[last] = dict(node[last])
        node[last]["codebook"] = new_codebooks[i]
        if last not in cnode:
            cnode[last] = {}
        cnode[last] = dict(cnode[last])
        cnode[last]["qw_delta"] = new_deltas[i]
        if new_ema_Ns is not None:
            cnode[last]["ema_N"] = new_ema_Ns[i]
            cnode[last]["ema_s"] = new_ema_ss[i]
    return p, c


# Single jitted function that processes ALL layers at once.
# Traced once (Python logic runs during tracing), compiled, then reuses buffers.
# Separate caches for the two EMA formulations.
_requantize_all_jitted = None
_requantize_all_jitted_canonical = None

def _make_requantize_all(meta, canonical: bool = False):
    """Build a jitted function for this specific model structure.

    canonical=False: e_new = α·e_old + (1−α)·batch_mean   (EMA of means)
    canonical=True:  N_new = α·N_old + (1−α)·batch_count
                     s_new = α·s_old + (1−α)·batch_sum
                     e_new = s_new / max(N_new, eps)        (canonical VQ-VAE)
    """
    @jax.jit
    def _fn_means(weights, codebooks, ema_decay):
        new_deltas = []
        new_codebooks = []
        for i, m in enumerate(meta):
            bs = m["block_size"]
            nb = m["num_blocks"]
            tw = m["total_weights"]
            pl = m["padded_len"]
            w = weights[i]
            cb = codebooks[i]

            w_flat = w.reshape(-1)
            w_padded = jnp.pad(w_flat, (0, pl - tw))
            w_blocks = w_padded.reshape(nb, bs)
            block_idx = jnp.arange(nb)[:, None]

            dists = jnp.square(w_blocks[:, :, None] - cb[:, None, :])
            indices = jnp.argmin(dists, axis=-1)
            quantized_blocks = cb[block_idx, indices]
            quantized_flat = quantized_blocks.reshape(-1)[:tw]
            delta = (quantized_flat.reshape(w.shape) - w).astype(jnp.bfloat16)

            n_entries = cb.shape[1]
            one_hot = jax.nn.one_hot(indices, n_entries)
            counts = one_hot.sum(axis=1)
            sums = jnp.einsum("bse,bs->be", one_hot, w_blocks.astype(jnp.float32))
            batch_means = sums / jnp.maximum(counts, 1)
            active = counts > 0
            new_cb = jnp.where(active, ema_decay * cb + (1 - ema_decay) * batch_means, cb)

            new_deltas.append(delta)
            new_codebooks.append(new_cb)
        return new_deltas, new_codebooks

    @jax.jit
    def _fn_canonical(weights, codebooks, ema_Ns, ema_ss, ema_decay):
        new_deltas = []
        new_codebooks = []
        new_ema_Ns = []
        new_ema_ss = []
        for i, m in enumerate(meta):
            bs = m["block_size"]
            nb = m["num_blocks"]
            tw = m["total_weights"]
            pl = m["padded_len"]
            w = weights[i]
            cb = codebooks[i]
            N_old = ema_Ns[i]
            s_old = ema_ss[i]

            w_flat = w.reshape(-1)
            w_padded = jnp.pad(w_flat, (0, pl - tw))
            w_blocks = w_padded.reshape(nb, bs)
            block_idx = jnp.arange(nb)[:, None]

            # Assignment against CURRENT codebook (cb, not the derived-from-s/N).
            dists = jnp.square(w_blocks[:, :, None] - cb[:, None, :])
            indices = jnp.argmin(dists, axis=-1)
            quantized_blocks = cb[block_idx, indices]
            quantized_flat = quantized_blocks.reshape(-1)[:tw]
            delta = (quantized_flat.reshape(w.shape) - w).astype(jnp.bfloat16)

            n_entries = cb.shape[1]
            one_hot = jax.nn.one_hot(indices, n_entries)
            batch_counts = one_hot.sum(axis=1)
            batch_sums = jnp.einsum("bse,bs->be", one_hot,
                                    w_blocks.astype(jnp.float32))

            # EMA the RAW sum and count — canonical VQ-VAE (van den Oord 2017).
            N_new = ema_decay * N_old + (1.0 - ema_decay) * batch_counts
            s_new = ema_decay * s_old + (1.0 - ema_decay) * batch_sums
            # Only derive codebook where we have actual statistics accumulated.
            # Floor on N avoids division blow-up for consistently-dead entries.
            cb_derived = s_new / jnp.maximum(N_new, 1e-5)
            # Keep old entry when N is near-zero (dead-for-a-long-time).
            have_stats = N_new > 1e-3
            new_cb = jnp.where(have_stats, cb_derived, cb)

            new_deltas.append(delta)
            new_codebooks.append(new_cb)
            new_ema_Ns.append(N_new)
            new_ema_ss.append(s_new)
        return new_deltas, new_codebooks, new_ema_Ns, new_ema_ss

    return _fn_canonical if canonical else _fn_means


def requantize_params(params, ema_decay=0.999, use_canonical_ema: bool = False):
    """Recompute cached deltas and EMA-update codebooks for all NB layers.

    Uses a single jit-compiled function for all layers — no per-call device
    array allocation.

    use_canonical_ema=True switches to canonical VQ-VAE EMA (EMA of raw
    sums and counts). Requires that `cache/ema_N` and `cache/ema_s` have
    been initialised (see `init_canonical_ema_state`).

    Returns (updated_params, intermediates_dict).
    """
    global _requantize_all_jitted, _requantize_all_jitted_canonical

    params_inner = params.get("params", params)
    cache = params.get("cache", {})
    weights, codebooks, deltas, ema_Ns, ema_ss, meta = \
        _extract_layer_arrays(params_inner, cache)

    if not weights:
        return params, {}

    if use_canonical_ema:
        if _requantize_all_jitted_canonical is None:
            _requantize_all_jitted_canonical = _make_requantize_all(
                meta, canonical=True)
        new_deltas, new_codebooks, new_ema_Ns, new_ema_ss = \
            _requantize_all_jitted_canonical(
                weights, codebooks, ema_Ns, ema_ss, jnp.float32(ema_decay))
        new_p, new_c = _rebuild_params(
            params_inner, cache, meta, new_codebooks, new_deltas,
            new_ema_Ns=new_ema_Ns, new_ema_ss=new_ema_ss)
    else:
        if _requantize_all_jitted is None:
            _requantize_all_jitted = _make_requantize_all(meta, canonical=False)
        new_deltas, new_codebooks = _requantize_all_jitted(
            weights, codebooks, jnp.float32(ema_decay))
        new_p, new_c = _rebuild_params(
            params_inner, cache, meta, new_codebooks, new_deltas)

    return {**params, "params": new_p, "cache": new_c}, {}


def init_canonical_ema_state(params):
    """Seed `cache/ema_s` with current codebook values and `cache/ema_N` with ones.

    Call this once after codebooks have been (re-)initialised from scaled
    weights, before enabling canonical EMA. Without this, the derived
    codebook `s/N` starts near zero and destroys the percentile init.
    """
    import copy
    params_inner = params.get("params", params)
    cache = params.get("cache", {})

    def _walk(p_node, c_node):
        if isinstance(p_node, dict):
            if "codebook" in p_node and "weight" in p_node:
                if "ema_N" not in c_node:
                    c_node["ema_N"] = jnp.ones_like(
                        p_node["codebook"], dtype=jnp.float32)
                else:
                    c_node["ema_N"] = jnp.ones_like(c_node["ema_N"])
                c_node["ema_s"] = p_node["codebook"].astype(jnp.float32)
            else:
                for k in p_node:
                    if k not in c_node:
                        c_node[k] = {}
                    else:
                        c_node[k] = dict(c_node[k])
                    _walk(p_node[k], c_node[k])

    new_cache = copy.deepcopy(cache)
    _walk(params_inner, new_cache)
    return {**params, "cache": new_cache}


def _init_codebook_from_weight(weight, block_size, num_blocks, total_weights,
                                padded_len, n_entries):
    """Initialize codebook from weight percentiles."""
    w_flat = weight.reshape(-1)
    if padded_len > total_weights:
        w_flat = jnp.pad(w_flat, (0, padded_len - total_weights))
    w_blocks = w_flat.reshape(num_blocks, block_size)
    q = jnp.linspace(0, 1, n_entries)
    return jnp.quantile(w_blocks.astype(jnp.float32), q, axis=1).T


def compute_quant_diagnostics(params):
    """Per-step quantization health metrics across all NativeBit layers.

    Pure function — call from anywhere (CPU/GPU/TPU) without side effects.
    Designed for experiment logs; no framework-specific state.

    Returns a plain dict of scalars:
      quant_error_rms     -- sqrt(mean ||w - Q(w)||^2) over all NB weights.
                             This is what STE pretends is zero; commitment
                             loss directly drives it down.
      codebook_utilization -- fraction of (block, entry) pairs that received
                              at least one weight assignment at the most
                              recent argmin. 1.0 = every entry used.
      dead_entries_frac   -- 1 - codebook_utilization (frozen entries).
      n_nb_layers         -- how many NB layers were scanned.

    All arithmetic is float32. Safe to call every log step.
    """
    params_inner = params.get("params", params)

    total_sq_err = jnp.float32(0.0)
    total_weights = 0
    total_active = 0
    total_entries = 0
    n_layers = 0

    def _walk(node):
        nonlocal total_sq_err, total_weights, total_active, total_entries, n_layers
        if isinstance(node, dict):
            if "codebook" in node and "weight" in node:
                w = node["weight"].astype(jnp.float32)
                cb = node["codebook"].astype(jnp.float32)

                num_blocks, n_entries = cb.shape
                tw = w.size
                bs = math.ceil(tw / num_blocks)
                padded = num_blocks * bs

                w_flat = w.reshape(-1)
                if padded > tw:
                    w_flat = jnp.pad(w_flat, (0, padded - tw))
                w_blocks = w_flat.reshape(num_blocks, bs)

                d_sq = (w_blocks[:, :, None] - cb[:, None, :]) ** 2
                indices = d_sq.argmin(axis=-1)
                min_d_sq = d_sq.min(axis=-1)

                if padded > tw:
                    flat_idx = jnp.arange(padded)
                    valid = (flat_idx < tw).reshape(num_blocks, bs)
                    min_d_sq = jnp.where(valid, min_d_sq, 0.0)

                total_sq_err = total_sq_err + min_d_sq.sum()
                total_weights += tw

                one_hot = jax.nn.one_hot(indices, n_entries)
                counts = one_hot.sum(axis=1)
                total_active = total_active + (counts > 0).sum()
                total_entries += num_blocks * n_entries
                n_layers += 1
            else:
                for k, v in node.items():
                    _walk(v)

    _walk(params_inner)

    if n_layers == 0:
        return {"quant_error_rms": 0.0, "codebook_utilization": 1.0,
                "dead_entries_frac": 0.0, "n_nb_layers": 0}

    rms = jnp.sqrt(total_sq_err / jnp.float32(max(total_weights, 1)))
    util = total_active.astype(jnp.float32) / jnp.float32(max(total_entries, 1))
    return {
        "quant_error_rms": rms,
        "codebook_utilization": util,
        "dead_entries_frac": 1.0 - util,
        "n_nb_layers": n_layers,
    }


def compute_quant_reg(params):
    """VQ-VAE-style commitment loss for NativeBit layers.

    Returns `sum_over_all_weights(min_j (w - sg(cb_j))^2) / n_nb_layers`.

    Normalizing by layer count (not weight count) gives per-weight gradient
    magnitude `2(w - Q(w)) / n_layers` — comparable to typical CE gradients
    at realistic model scales (~1e-4 to 1e-3), so λ ~ 1 is a sensible
    starting point rather than λ ~ 1e4 needed with per-weight mean.

    Gradient flows only to w (pulls toward nearest codebook entry).
    Codebook is stop_gradient — updated by EMA elsewhere.
    """
    params_inner = params.get("params", params)

    total_sq = jnp.float32(0.0)
    n_layers = 0  # static Python int — size is known at tracing time

    def _walk(node):
        nonlocal total_sq, n_layers
        if isinstance(node, dict):
            if "codebook" in node and "weight" in node:
                w = node["weight"].astype(jnp.float32)
                cb = jax.lax.stop_gradient(node["codebook"]).astype(jnp.float32)

                num_blocks, n_entries = cb.shape
                tw = w.size
                bs = math.ceil(tw / num_blocks)
                padded = num_blocks * bs

                w_flat = w.reshape(-1)
                if padded > tw:
                    w_flat = jnp.pad(w_flat, (0, padded - tw))
                w_blocks = w_flat.reshape(num_blocks, bs)

                d_sq = (w_blocks[:, :, None] - cb[:, None, :]) ** 2
                min_d_sq = d_sq.min(axis=-1)

                if padded > tw:
                    flat_idx = jnp.arange(padded)
                    valid = (flat_idx < tw).reshape(num_blocks, bs).astype(min_d_sq.dtype)
                    min_d_sq = min_d_sq * valid

                total_sq = total_sq + min_d_sq.sum()
                n_layers += 1
            else:
                for k, v in node.items():
                    _walk(v)

    _walk(params_inner)
    if n_layers == 0:
        return jnp.float32(0.0)
    return total_sq / jnp.float32(n_layers)
