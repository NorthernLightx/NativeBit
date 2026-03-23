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
