"""Fast NativeBit inference — pre-computed indices, gather-only forward pass.

Converts a trained NativeBitLinear model to use cached indices for inference.
No distance computation, no argmin — just codebook[block_idx, indices] per layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NativeBitLinear


class NativeBitInferenceLinear(nn.Module):
    """Fast inference replacement for NativeBitLinear.

    Pre-computes indices once, then forward pass is just:
      weight = codebook[block_idx, cached_indices]
      output = x @ weight.T
    """

    def __init__(self, in_features: int, out_features: int,
                 codebook: torch.Tensor, indices: torch.Tensor,
                 block_size: int, bias: torch.Tensor = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.register_buffer("codebook", codebook)        # (num_blocks, n_entries)
        self.register_buffer("indices", indices)            # (num_blocks, block_size)
        num_blocks = codebook.shape[0]
        self.register_buffer("block_idx", torch.arange(num_blocks).unsqueeze(1))
        self.total_weights = out_features * in_features

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single gather — no distance computation
        weight_flat = self.codebook[self.block_idx, self.indices].reshape(-1)[:self.total_weights]
        weight = weight_flat.view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)


def convert_to_inference(model: nn.Module) -> nn.Module:
    """Convert all NativeBitLinear layers to NativeBitInferenceLinear.

    Pre-computes quantization indices once, then all subsequent forward
    passes use gather-only (no distance computation).
    """
    for name, module in model.named_modules():
        if isinstance(module, NativeBitLinear):
            # Compute indices one final time
            w_flat = module.weight.data.view(-1)
            if module._padded_len > module.total_weights:
                w_padded = F.pad(w_flat, (0, module._padded_len - module.total_weights))
            else:
                w_padded = w_flat

            w_blocks = w_padded.view(module.num_blocks, module.block_size)
            dists = (w_blocks.unsqueeze(-1) - module.codebook.data.unsqueeze(1)).square()
            indices = dists.argmin(dim=-1)  # (num_blocks, block_size)

            # Create fast inference layer
            fast = NativeBitInferenceLinear(
                module.in_features, module.out_features,
                module.codebook.data.clone(),
                indices,
                module.block_size,
                bias=module.bias.data.clone() if module.bias is not None else None,
            )

            # Replace in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], fast)

    model.requires_grad_(False)
    return model
