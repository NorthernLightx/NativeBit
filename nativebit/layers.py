"""NativeBitLinear -- drop-in nn.Linear replacement with learned per-block codebooks.

Each weight is quantized to its nearest codebook entry during the forward pass.
Gradients flow through quantization via the straight-through estimator (STE).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_utils import init_codebook_percentile, revive_dead_entries


class NativeBitLinear(nn.Module):
    """Linear layer with learned per-block codebook quantization.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        bias: if True, adds a learnable bias (kept in full precision).
        block_size: number of weights sharing one codebook.
        n_entries: codebook entries per block (8 = 3-bit).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 block_size: int = 64, n_entries: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_entries = n_entries

        # Latent (full-precision) weights -- gradients flow here via STE
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Block layout
        total_weights = out_features * in_features
        self.num_blocks = math.ceil(total_weights / block_size)
        self.total_weights = total_weights
        self._padded_len = self.num_blocks * block_size

        # Per-block codebook: (num_blocks, n_entries)
        self.codebook = nn.Parameter(torch.empty(self.num_blocks, n_entries))

        # Utilization counter for dead entry detection
        self.register_buffer(
            "utilization", torch.zeros(self.num_blocks, n_entries, dtype=torch.long)
        )

        # Pre-computed block index for gather — avoid torch.arange every forward
        self.register_buffer(
            "_block_idx", torch.arange(self.num_blocks).unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming init for weights, percentile init for codebooks."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

        w_flat = self.weight.data.view(-1)
        for b in range(self.num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, self.total_weights)
            self.codebook.data[b] = init_codebook_percentile(w_flat[start:end], self.n_entries)

    def _quantize(self, w_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights to nearest codebook entries.

        Uses squared L2 distance and plain argmin.
        """
        if self._padded_len > self.total_weights:
            w_padded = F.pad(w_flat, (0, self._padded_len - self.total_weights))
        else:
            w_padded = w_flat

        w_blocks = w_padded.view(self.num_blocks, self.block_size)
        dists = (w_blocks.unsqueeze(-1) - self.codebook.unsqueeze(1)).square()
        indices = dists.argmin(dim=-1)

        quantized_blocks = self.codebook[self._block_idx, indices]

        return quantized_blocks.view(-1)[:self.total_weights], indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_flat = self.weight.view(-1)
        quantized_flat, indices = self._quantize(w_flat)
        self._last_indices = indices

        # Plain STE: forward uses quantized weights, backward flows to latent
        # weights only. Codebook is updated separately (EMA or dedicated loss),
        # not through the task loss gradient — this avoids expensive backward
        # through the gather and is compatible with EMA codebook updates.
        quantized_w = self.weight + (quantized_flat.view_as(self.weight) - self.weight).detach()

        return F.linear(x, quantized_w, self.bias)

    def update_utilization_from_cache(self) -> None:
        """Update utilization counters from the last forward pass."""
        if hasattr(self, '_last_indices'):
            with torch.no_grad():
                self.utilization.zero_()
                one_hot = F.one_hot(self._last_indices, self.n_entries)
                if self._padded_len > self.total_weights:
                    pad_count = self._padded_len - self.total_weights
                    one_hot[-1, self.block_size - pad_count:] = 0
                self.utilization.copy_(one_hot.sum(dim=1))

    def revive_dead_entries(self, threshold: float = 0.01, noise_scale: float = 0.01) -> int:
        """Revive dead codebook entries via split-based revival."""
        return revive_dead_entries(self.codebook, self.utilization, threshold, noise_scale)

    def get_utilization_stats(self) -> dict:
        """Utilization statistics for logging."""
        total = self.utilization.sum(dim=-1, keepdim=True).clamp(min=1).float()
        frac = self.utilization.float() / total
        dead = (frac < 0.01).sum().item()
        return {
            "dead_entries": int(dead),
            "total_entries": self.num_blocks * self.n_entries,
            "min_utilization": frac.min().item(),
            "max_utilization": frac.max().item(),
        }

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, block_size={self.block_size}, "
                f"n_entries={self.n_entries}, num_blocks={self.num_blocks}")
