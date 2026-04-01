"""Codebook initialization and collapse prevention — JAX port."""

import jax
import jax.numpy as jnp


def init_codebook_percentile(weight: jnp.ndarray, n_entries: int) -> jnp.ndarray:
    """Initialize codebook from evenly spaced percentiles."""
    q = jnp.linspace(0, 1, n_entries)
    return jnp.quantile(weight.astype(jnp.float32), q)


def init_codebook_kmeans_batch(weight_blocks: jnp.ndarray, n_entries: int,
                               n_iter: int = 10) -> jnp.ndarray:
    """Batched k-means init for all blocks — fully vectorized.

    Args:
        weight_blocks: (num_blocks, block_size)
        n_entries: codebook entries per block.
    Returns:
        (num_blocks, n_entries) codebook values.
    """
    w = weight_blocks.astype(jnp.float32)
    q = jnp.linspace(0, 1, n_entries)
    centroids = jnp.quantile(w, q, axis=1).T  # (num_blocks, n_entries)

    def step(centroids, _):
        dists = jnp.square(w[:, :, None] - centroids[:, None, :])
        assignments = dists.argmin(axis=-1)
        one_hot = jax.nn.one_hot(assignments, n_entries)
        counts = one_hot.sum(axis=1).clip(min=1)
        sums = jnp.einsum("bse,bs->be", one_hot, w)
        return sums / counts, None

    centroids, _ = jax.lax.scan(step, centroids, None, length=n_iter)
    return centroids.astype(weight_blocks.dtype)


def revive_dead_entries(codebook: jnp.ndarray, utilization: jnp.ndarray,
                        threshold: float = 0.01,
                        noise_scale: float = 0.01) -> tuple[jnp.ndarray, int]:
    """Reinitialize dead codebook entries by perturbing the most-used entry.

    Returns (new_codebook, num_dead).
    """
    total = utilization.sum(axis=-1, keepdims=True).clip(min=1).astype(jnp.float32)
    frac = utilization.astype(jnp.float32) / total
    dead_mask = frac < threshold
    num_dead = int(dead_mask.sum())

    if num_dead == 0:
        return codebook, 0

    best_idx = utilization.argmax(axis=-1)  # (num_blocks,)
    best_vals = jnp.take_along_axis(codebook, best_idx[:, None], axis=1)

    split_offset = jnp.abs(best_vals).clip(min=1e-4) * noise_scale
    n_entries = codebook.shape[1]
    signs = jnp.ones_like(codebook)
    signs = signs.at[:, ::2].set(-1.0)
    offsets = signs * split_offset

    replacement = jnp.broadcast_to(best_vals, codebook.shape) + offsets
    new_codebook = jnp.where(dead_mask, replacement, codebook)
    return new_codebook, num_dead


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
