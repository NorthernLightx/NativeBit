"""Post-hoc quantization baselines for Phase 3 benchmarks.

All functions operate in-place on a model's nn.Linear weight tensors.
They skip the LM head (passed as exclude_modules).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_utils import init_codebook_percentile


# ---------------------------------------------------------------------------
# NF4 lookup table (from QLoRA / bitsandbytes)
# 16 values optimally placed for a standard normal distribution
# ---------------------------------------------------------------------------
NF4_TABLE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


def quantize_uniform(model: nn.Module, bits: int = 3,
                     block_size: int = 32,
                     exclude_modules: list[nn.Module] | None = None) -> dict:
    """Naive uniform integer quantization (post-hoc).

    Maps each block of weights to uniformly spaced levels between min and max.

    Returns:
        dict with total_bits, quantized_weights count.
    """
    exclude = set(id(m) for m in (exclude_modules or []))
    total_bits = 0
    total_weights = 0
    n_levels = 2 ** bits

    for module in model.modules():
        if not isinstance(module, nn.Linear) or id(module) in exclude:
            continue
        w = module.weight.data.view(-1)
        n = w.numel()
        num_blocks = math.ceil(n / block_size)
        padded = num_blocks * block_size
        w_padded = F.pad(w, (0, padded - n)) if padded > n else w
        w_blocks = w_padded.view(num_blocks, block_size)

        # Per-block min/max uniform quantization
        w_min = w_blocks.min(dim=1, keepdim=True).values
        w_max = w_blocks.max(dim=1, keepdim=True).values
        scale = (w_max - w_min).clamp(min=1e-8) / (n_levels - 1)

        # Quantize and dequantize
        q_idx = ((w_blocks - w_min) / scale).round().clamp(0, n_levels - 1)
        w_deq = w_min + q_idx * scale

        module.weight.data = w_deq.view(-1)[:n].view_as(module.weight.data)
        # Size: bits per weight + 2 float16 per block (scale, zero)
        total_bits += n * bits + num_blocks * 2 * 16
        total_weights += n

    return {"total_bits": total_bits, "total_weights": total_weights}


def quantize_kmeans(model: nn.Module, n_entries: int = 8,
                    block_size: int = 32,
                    exclude_modules: list[nn.Module] | None = None) -> dict:
    """K-means (percentile) quantization (post-hoc).

    Uses percentile-based codebook init (equivalent to 1-iteration k-means
    with sorted initialization). Same as our co-training codebook init.

    Returns:
        dict with total_bits, quantized_weights count.
    """
    exclude = set(id(m) for m in (exclude_modules or []))
    bits = math.ceil(math.log2(n_entries))
    total_bits = 0
    total_weights = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear) or id(module) in exclude:
            continue
        w = module.weight.data.view(-1)
        n = w.numel()
        num_blocks = math.ceil(n / block_size)
        padded = num_blocks * block_size
        w_padded = F.pad(w, (0, padded - n)) if padded > n else w
        w_blocks = w_padded.view(num_blocks, block_size)

        for b in range(num_blocks):
            cb = init_codebook_percentile(w_blocks[b], n_entries)
            dists = (w_blocks[b].unsqueeze(-1) - cb.unsqueeze(0)).abs()
            indices = dists.argmin(dim=-1)
            w_blocks[b] = cb[indices]

        module.weight.data = w_blocks.view(-1)[:n].view_as(module.weight.data)
        # Size: bits per weight + codebook entries * 16 bits per block
        total_bits += n * bits + num_blocks * n_entries * 16
        total_weights += n

    return {"total_bits": total_bits, "total_weights": total_weights}


def quantize_nf4(model: nn.Module, block_size: int = 32,
                 exclude_modules: list[nn.Module] | None = None) -> dict:
    """NF4 (Normal Float 4-bit) quantization (post-hoc).

    Per-block absmax scaling + lookup into a fixed 16-value table optimized
    for normally distributed weights (from QLoRA).

    Returns:
        dict with total_bits, quantized_weights count.
    """
    exclude = set(id(m) for m in (exclude_modules or []))
    nf4 = NF4_TABLE.clone()
    total_bits = 0
    total_weights = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear) or id(module) in exclude:
            continue
        w = module.weight.data.view(-1)
        n = w.numel()
        num_blocks = math.ceil(n / block_size)
        padded = num_blocks * block_size
        w_padded = F.pad(w, (0, padded - n)) if padded > n else w
        w_blocks = w_padded.view(num_blocks, block_size)

        nf4_dev = nf4.to(w.device, dtype=w.dtype)

        # Per-block absmax scaling
        absmax = w_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        w_norm = w_blocks / absmax  # normalize to [-1, 1]

        # Snap to nearest NF4 value
        dists = (w_norm.unsqueeze(-1) - nf4_dev.unsqueeze(0).unsqueeze(0)).abs()
        indices = dists.argmin(dim=-1)
        w_deq = nf4_dev[indices] * absmax

        module.weight.data = w_deq.view(-1)[:n].view_as(module.weight.data)
        # Size: 4 bits per weight + 1 float16 per block (absmax scale)
        total_bits += n * 4 + num_blocks * 16
        total_weights += n

    return {"total_bits": total_bits, "total_weights": total_weights}


def compute_model_size(model: nn.Module, method: str, bits: int = 3,
                       block_size: int = 32, n_entries: int = 8,
                       exclude_modules: list[nn.Module] | None = None) -> dict:
    """Compute actual model size in bits for a given quantization method.

    Does NOT modify the model. Just computes what the size would be.

    Args:
        method: "float16", "float32", "uniform", "kmeans", "nf4", "nativebit"

    Returns:
        dict with total_bits, weight_bits, embedding_bits, overhead_bits.
    """
    exclude = set(id(m) for m in (exclude_modules or []))
    weight_bits = 0
    overhead_bits = 0
    embedding_bits = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_bits += module.weight.numel() * 16  # always float16
        elif isinstance(module, nn.Linear):
            n = module.weight.numel()
            if id(module) in exclude:
                # LM head stays float16
                embedding_bits += n * 16
            elif method in ("float16",):
                weight_bits += n * 16
            elif method == "float32":
                weight_bits += n * 32
            elif method == "uniform":
                num_blocks = math.ceil(n / block_size)
                weight_bits += n * bits
                overhead_bits += num_blocks * 2 * 16  # scale + zero per block
            elif method == "kmeans":
                num_blocks = math.ceil(n / block_size)
                cb_bits = math.ceil(math.log2(n_entries))
                weight_bits += n * cb_bits
                overhead_bits += num_blocks * n_entries * 16  # codebook per block
            elif method == "nf4":
                num_blocks = math.ceil(n / block_size)
                weight_bits += n * 4
                overhead_bits += num_blocks * 16  # absmax per block
            elif method == "nativebit":
                num_blocks = math.ceil(n / block_size)
                cb_bits = math.ceil(math.log2(n_entries))
                weight_bits += n * cb_bits
                overhead_bits += num_blocks * n_entries * 16

    # LayerNorm params (always float16)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                overhead_bits += p.numel() * 16

    total_bits = weight_bits + overhead_bits + embedding_bits
    return {
        "total_bits": total_bits,
        "weight_bits": weight_bits,
        "overhead_bits": overhead_bits,
        "embedding_bits": embedding_bits,
        "total_bytes": total_bits / 8,
        "weight_bytes": weight_bits / 8,
    }


def measure_inference_speed(model: nn.Module, device: torch.device,
                            context_len: int = 512, n_tokens: int = 4096,
                            warmup: int = 3) -> float:
    """Measure inference speed in tokens/sec (batch forward pass, not autoregressive)."""
    import time
    model.to(device)
    model.requires_grad_(False)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randint(0, 1000, (1, context_len), device=device)
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated = 0
    with torch.no_grad():
        while generated < n_tokens:
            x = torch.randint(0, 1000, (1, context_len), device=device)
            model(x)
            generated += context_len
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return generated / elapsed
