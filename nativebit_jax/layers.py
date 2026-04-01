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


def _extract_layer_arrays(params_inner, cache):
    """Extract (weights, codebooks, deltas) as flat lists + metadata for jit."""
    weights, codebooks, deltas, meta = [], [], [], []
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
                meta.append({"path": path, "block_size": bs, "num_blocks": nb,
                             "total_weights": tw, "padded_len": nb * bs})
            else:
                for k in node:
                    _walk(node[k], cache_node.get(k, {}), path + (k,))
    _walk(params_inner, cache)
    return weights, codebooks, deltas, meta


def _rebuild_params(params_inner, cache, meta, new_codebooks, new_deltas):
    """Put updated codebooks and deltas back into the params tree."""
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
    return p, c


# Single jitted function that processes ALL layers at once.
# Traced once (Python logic runs during tracing), compiled, then reuses buffers.
_requantize_all_jitted = None

def _make_requantize_all(meta):
    """Build a jitted function for this specific model structure."""
    @jax.jit
    def _fn(weights, codebooks, ema_decay):
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

            # EMA update
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
    return _fn


def requantize_params(params, ema_decay=0.999):
    """Recompute cached deltas and EMA-update codebooks for all NB layers.

    Uses a single jit-compiled function for all layers — no per-call device
    array allocation, eliminating the TPU memory leak.

    Returns (updated_params, intermediates_dict).
    """
    global _requantize_all_jitted

    params_inner = params.get("params", params)
    cache = params.get("cache", {})
    weights, codebooks, deltas, meta = _extract_layer_arrays(params_inner, cache)

    if not weights:
        return params, {}

    if _requantize_all_jitted is None:
        _requantize_all_jitted = _make_requantize_all(meta)

    new_deltas, new_codebooks = _requantize_all_jitted(
        weights, codebooks, jnp.float32(ema_decay))

    new_p, new_c = _rebuild_params(params_inner, cache, meta, new_codebooks, new_deltas)
    return {**params, "params": new_p, "cache": new_c}, {}


def _init_codebook_from_weight(weight, block_size, num_blocks, total_weights,
                                padded_len, n_entries):
    """Initialize codebook from weight percentiles."""
    w_flat = weight.reshape(-1)
    if padded_len > total_weights:
        w_flat = jnp.pad(w_flat, (0, padded_len - total_weights))
    w_blocks = w_flat.reshape(num_blocks, block_size)
    q = jnp.linspace(0, 1, n_entries)
    return jnp.quantile(w_blocks.astype(jnp.float32), q, axis=1).T
