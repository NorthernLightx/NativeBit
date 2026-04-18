"""Codebook initialization and EMA update — JAX port."""

import jax
import jax.numpy as jnp


def init_codebook_percentile(weight: jnp.ndarray, n_entries: int) -> jnp.ndarray:
    """Initialize codebook from evenly spaced percentiles."""
    q = jnp.linspace(0, 1, n_entries)
    return jnp.quantile(weight.astype(jnp.float32), q)


def ema_update_codebooks(codebook: jnp.ndarray, indices: jnp.ndarray,
                         weight_blocks: jnp.ndarray,
                         decay: float = 0.99) -> jnp.ndarray:
    """EMA codebook update from quantization assignments.

    Args:
        codebook: (num_blocks, n_entries)
        indices: (num_blocks, block_size) — assignment indices from quantize
        weight_blocks: (num_blocks, block_size) — current weight values
        decay: EMA decay factor
    Returns:
        Updated codebook.
    """
    n_entries = codebook.shape[1]
    one_hot = jax.nn.one_hot(indices, n_entries)  # (N, B, E)
    counts = one_hot.sum(axis=1)  # (N, E)
    sums = jnp.einsum("bse,bs->be", one_hot, weight_blocks.astype(jnp.float32))
    batch_means = sums / jnp.maximum(counts, 1)
    active = counts > 0
    return jnp.where(active, decay * codebook + (1 - decay) * batch_means, codebook)
