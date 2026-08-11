"""Codebook initialization and collapse prevention utilities."""

import torch


def init_codebook_percentile(weight: torch.Tensor, n_entries: int) -> torch.Tensor:
    """Initialize codebook from evenly spaced percentiles of the weight distribution.

    Uses torch.quantile (pure PyTorch, no numpy needed).

    Args:
        weight: flat weight tensor to derive percentiles from.
        n_entries: number of codebook entries (e.g. 8 for 3-bit).

    Returns:
        Tensor of shape (n_entries,) with codebook values.
    """
    # quantile expects fractions in [0, 1]
    q = torch.linspace(0, 1, n_entries, device=weight.device)
    w_sorted = weight.detach().float().contiguous()
    values = torch.quantile(w_sorted, q)
    return values.to(dtype=weight.dtype)


def init_codebook_percentile_batch(weight_blocks: torch.Tensor, n_entries: int) -> torch.Tensor:
    """Percentile init for all blocks at once — vectorized.

    Mirrors the JAX `_init_codebook_from_weight`: evenly spaced quantiles
    per block. Used for QAT codebook re-init from trained float weights.

    Args:
        weight_blocks: (num_blocks, block_size) weight values per block.
        n_entries: number of codebook entries per block.

    Returns:
        Tensor of shape (num_blocks, n_entries) with codebook values.
    """
    q = torch.linspace(0, 1, n_entries, device=weight_blocks.device)
    values = torch.quantile(weight_blocks.detach().float(), q, dim=1).T
    return values.to(dtype=weight_blocks.dtype)


def ema_update_canonical(
    codebook: torch.Tensor,
    ema_N: torch.Tensor,
    ema_s: torch.Tensor,
    weight_blocks: torch.Tensor,
    decay: float = 0.99,
    valid_mask: torch.Tensor | None = None,
) -> None:
    """Canonical VQ-VAE EMA codebook update (van den Oord 2017). In-place.

    EMAs the raw per-entry sums and counts, then derives the codebook as
    s/N. Count-weighted: noisy low-count batches contribute proportionally,
    unlike EMA-of-means which weights every batch equally. Mirrors the JAX
    `_fn_canonical` in nativebit_jax/layers.py.

    Args:
        codebook: (num_blocks, n_entries) — updated in place.
        ema_N: (num_blocks, n_entries) running counts — updated in place.
        ema_s: (num_blocks, n_entries) running sums — updated in place.
        weight_blocks: (num_blocks, block_size) current weight values.
        decay: EMA decay factor (α).
        valid_mask: optional (num_blocks, block_size) bool; False marks
            padding positions excluded from the statistics.
    """
    with torch.no_grad():
        w = weight_blocks.detach().float()
        # Assignment against CURRENT codebook (not the derived s/N)
        dists = (w.unsqueeze(-1) - codebook.float().unsqueeze(1)).square()
        indices = dists.argmin(dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, codebook.shape[1]).float()
        if valid_mask is not None:
            one_hot = one_hot * valid_mask.unsqueeze(-1).float()
        batch_counts = one_hot.sum(dim=1)
        batch_sums = torch.einsum("bse,bs->be", one_hot, w)

        ema_N.mul_(decay).add_(batch_counts, alpha=1.0 - decay)
        ema_s.mul_(decay).add_(batch_sums, alpha=1.0 - decay)

        # Floor on N avoids division blow-up for consistently-dead entries;
        # keep the old entry where no statistics have accumulated.
        derived = ema_s / ema_N.clamp(min=1e-5)
        have_stats = ema_N > 1e-3
        codebook.data.copy_(
            torch.where(have_stats, derived, codebook.data.float()).to(codebook.dtype)
        )


def init_codebook_kmeans_batch(weight_blocks: torch.Tensor, n_entries: int, n_iter: int = 10) -> torch.Tensor:
    """Batched k-means init for all blocks at once — fully vectorized.

    Args:
        weight_blocks: (num_blocks, block_size) weight values per block.
        n_entries: number of codebook entries per block.
        n_iter: k-means iterations.

    Returns:
        Tensor of shape (num_blocks, n_entries) with codebook values.
    """
    B, S = weight_blocks.shape
    w = weight_blocks.detach().float()
    q = torch.linspace(0, 1, n_entries, device=w.device)
    centroids = torch.quantile(w, q, dim=1).T  # (B, n_entries)

    for _ in range(n_iter):
        dists = (w.unsqueeze(-1) - centroids.unsqueeze(1)).abs()
        assignments = dists.argmin(dim=-1)
        one_hot = torch.nn.functional.one_hot(assignments, n_entries).float()
        counts = one_hot.sum(dim=1).clamp(min=1)
        sums = (w.unsqueeze(-1) * one_hot).sum(dim=1)
        centroids = sums / counts

    return centroids.to(dtype=weight_blocks.dtype)


def revive_dead_entries(
    codebook: torch.Tensor,
    utilization: torch.Tensor,
    threshold: float = 0.01,
    noise_scale: float = 0.01,
) -> int:
    """Reinitialize dead codebook entries by perturbing the most-used entry.

    Fully vectorized — no Python loops over blocks.

    Args:
        codebook: (num_blocks, n_entries) learnable codebook values.
        utilization: (num_blocks, n_entries) usage counts (not normalized).
        threshold: fraction below which an entry is considered dead.
        noise_scale: std of noise added when reviving.

    Returns:
        Number of entries revived.
    """
    with torch.no_grad():
        total = utilization.sum(dim=-1, keepdim=True).clamp(min=1)
        frac = utilization.float() / total.float()

        # Dead mask: (num_blocks, n_entries)
        dead_mask = frac < threshold

        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return 0

        # Best entry per block: the most-used one
        best_idx = utilization.argmax(dim=-1)  # (num_blocks,)
        best_vals = codebook.data.gather(1, best_idx.unsqueeze(-1))  # (num_blocks, 1)

        # Split revival: offset dead entries symmetrically around the most-used entry.
        # Alternating +/- offset preserves the distribution better than random noise.
        split_offset = best_vals.abs().clamp(min=1e-4) * noise_scale
        signs = torch.ones_like(codebook.data)
        signs[:, ::2] = -1.0  # alternate signs across entries
        offsets = signs * split_offset

        codebook.data[dead_mask] = (best_vals.expand_as(codebook.data) + offsets)[dead_mask]

    return num_dead
