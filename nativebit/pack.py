"""NativeBit model packing/export — convert training checkpoints to compressed format.

Packed format stores 3-bit (or N-bit) indices + codebook tables instead of
full float32 weights. Result is ~5x smaller than float16 and loads faster
(gather from codebook vs distance computation).
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Bit-packing: 3-bit (vectorized)
# ---------------------------------------------------------------------------

def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack a flat tensor of 3-bit indices (0-7) into bytes.

    Every 8 indices map to 3 bytes (24 bits).
    Returns a uint8 tensor.
    """
    indices = indices.to(torch.uint8).view(-1)
    # Pad to multiple of 8
    n = indices.numel()
    pad = (8 - n % 8) % 8
    if pad:
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=indices.device)])

    g = indices.view(-1, 8)  # (groups, 8)
    i0, i1, i2, i3, i4, i5, i6, i7 = g.unbind(dim=1)

    b0 = i0 | (i1 << 3) | ((i2 & 0x3) << 6)
    b1 = (i2 >> 2) | (i3 << 1) | (i4 << 4) | ((i5 & 0x1) << 7)
    b2 = (i5 >> 1) | (i6 << 2) | (i7 << 5)

    return torch.stack([b0, b1, b2], dim=1).view(-1).to(torch.uint8)


def unpack_3bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack a byte tensor into 3-bit indices.

    Args:
        packed: uint8 tensor from pack_3bit.
        count: original number of indices.

    Returns:
        uint8 tensor of shape (count,) with values in [0, 7].
    """
    packed = packed.to(torch.uint8).view(-1)
    g = packed.view(-1, 3)  # (groups, 3)
    b0, b1, b2 = g.unbind(dim=1)

    i0 = b0 & 0x7
    i1 = (b0 >> 3) & 0x7
    i2 = ((b0 >> 6) | (b1 << 2)) & 0x7
    i3 = (b1 >> 1) & 0x7
    i4 = (b1 >> 4) & 0x7
    i5 = ((b1 >> 7) | (b2 << 1)) & 0x7
    i6 = (b2 >> 2) & 0x7
    i7 = (b2 >> 5) & 0x7

    indices = torch.stack([i0, i1, i2, i3, i4, i5, i6, i7], dim=1).view(-1)
    return indices[:count]


# ---------------------------------------------------------------------------
# Bit-packing: 4-bit (nibble)
# ---------------------------------------------------------------------------

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) — 2 per byte."""
    indices = indices.to(torch.uint8).view(-1)
    n = indices.numel()
    if n % 2:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=indices.device)])
    g = indices.view(-1, 2)
    return (g[:, 0] | (g[:, 1] << 4)).to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack 4-bit indices."""
    packed = packed.to(torch.uint8).view(-1)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack([lo, hi], dim=1).view(-1)[:count]


# ---------------------------------------------------------------------------
# Generic N-bit (fallback)
# ---------------------------------------------------------------------------

def pack_nbits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Generic bit-packer. Slow fallback for non-3/4-bit."""
    if bits == 3:
        return pack_3bit(indices)
    if bits == 4:
        return pack_4bit(indices)

    indices = indices.to(torch.int32).view(-1)
    total_bits = indices.numel() * bits
    n_bytes = math.ceil(total_bits / 8)
    out = torch.zeros(n_bytes, dtype=torch.uint8, device=indices.device)

    bit_pos = 0
    for val in indices.tolist():
        for b in range(bits):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            out[byte_idx] |= ((val >> b) & 1) << bit_idx
            bit_pos += 1
    return out


def unpack_nbits(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Generic bit-unpacker."""
    if bits == 3:
        return unpack_3bit(packed, count)
    if bits == 4:
        return unpack_4bit(packed, count)

    packed = packed.to(torch.uint8).view(-1)
    out = torch.zeros(count, dtype=torch.uint8, device=packed.device)

    bit_pos = 0
    for i in range(count):
        val = 0
        for b in range(bits):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            val |= ((packed[byte_idx].item() >> bit_idx) & 1) << b
            bit_pos += 1
        out[i] = val
    return out


# ---------------------------------------------------------------------------
# Weight reconstruction (gather from codebook)
# ---------------------------------------------------------------------------

def reconstruct_weight(indices: torch.Tensor, codebook: torch.Tensor,
                       shape: tuple, block_size: int, total_weights: int) -> torch.Tensor:
    """Reconstruct a weight tensor from indices + codebook via gather.

    Args:
        indices: (num_blocks * block_size,) flat index tensor.
        codebook: (num_blocks, n_entries) codebook values.
        shape: (out_features, in_features) target weight shape.
        block_size: weights per block.
        total_weights: actual number of weights (before padding).

    Returns:
        Reconstructed weight tensor of given shape.
    """
    num_blocks = codebook.shape[0]
    idx_blocks = indices.view(num_blocks, block_size)

    weight_blocks = torch.gather(
        codebook.unsqueeze(1).expand(-1, block_size, -1),
        dim=2,
        index=idx_blocks.long().unsqueeze(-1),
    ).squeeze(-1)

    weight_flat = weight_blocks.reshape(-1)[:total_weights]
    return weight_flat.reshape(shape)


# ---------------------------------------------------------------------------
# Export: training checkpoint -> packed format
# ---------------------------------------------------------------------------

def _compute_indices(weight: torch.Tensor, codebook: torch.Tensor,
                     block_size: int, total_weights: int,
                     active_mask=None) -> torch.Tensor:
    """Compute quantization indices for a weight tensor (one-time, on CPU)."""
    import torch.nn.functional as F

    w_flat = weight.view(-1)
    num_blocks = codebook.shape[0]
    padded_len = num_blocks * block_size

    if padded_len > total_weights:
        w_padded = F.pad(w_flat, (0, padded_len - total_weights))
    else:
        w_padded = w_flat

    w_blocks = w_padded.view(num_blocks, block_size)
    dists = (w_blocks.unsqueeze(-1) - codebook.unsqueeze(1)).abs()

    if active_mask is not None:
        inactive = ~active_mask.unsqueeze(1)
        dists = dists.masked_fill(inactive, float('inf'))

    indices = dists.argmin(dim=-1)  # (num_blocks, block_size)
    return indices.view(-1)


def export_packed(ckpt_path: str, output_path: str, device: str = "cpu") -> dict:
    """Export a training checkpoint to packed NativeBit format.

    Args:
        ckpt_path: path to training checkpoint (.pt file).
        output_path: path for output packed file (.nbpack).
        device: device for computation (cpu recommended).

    Returns:
        Dict with size statistics.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from nativebit.model import build_model_from_config
    from nativebit.layers import NativeBitLinear
    from nativebit.generate import load_model_from_checkpoint

    # Load model from checkpoint
    model, config_dict = load_model_from_checkpoint(ckpt_path, torch.device(device))

    # Determine bit width from n_entries
    nb_layers = [m for m in model.modules() if isinstance(m, NativeBitLinear)]
    if not nb_layers:
        raise ValueError("No NativeBitLinear layers found — this is a float model, nothing to pack")

    n_entries = nb_layers[0].n_entries
    bits = math.ceil(math.log2(n_entries))

    # Extract full config from the loaded model (checkpoint config may be incomplete)
    config_dict = {
        "vocab_size": model.tok_emb.weight.shape[0],
        "n_embd": model.tok_emb.weight.shape[1],
        "n_layers": len(model.blocks),
        "n_head": model.blocks[0].attn.n_head,
        "ffn_hidden": model.blocks[0].ffn.w_gate.weight.shape[0]
            if hasattr(model.blocks[0].ffn.w_gate, 'weight')
            else model.blocks[0].ffn.w_gate.out_features,
        "context_len": model.context_len,
        "block_size": nb_layers[0].block_size,
        "n_codebook": n_entries,
    }

    # Select packer
    if bits == 3:
        packer = pack_3bit
    elif bits == 4:
        packer = pack_4bit
    else:
        def packer(idx):
            return pack_nbits(idx, bits)

    # Build packed state
    packed_state = {
        "format_version": FORMAT_VERSION,
        "config": config_dict,
        "bits": bits,
        "n_entries": n_entries,
        "quantized_layers": {},
        "float_params": {},
    }

    # Process quantized layers
    quantized_bytes = 0
    codebook_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, NativeBitLinear):
            continue

        active_mask = module.active_mask if hasattr(module, 'active_mask') else None
        indices = _compute_indices(
            module.weight.data, module.codebook.data,
            module.block_size, module.total_weights, active_mask,
        )
        packed_indices = packer(indices)
        codebook_f16 = module.codebook.data.half()

        layer_data = {
            "packed_indices": packed_indices,
            "codebook": codebook_f16,
            "shape": (module.out_features, module.in_features),
            "block_size": module.block_size,
            "n_entries": module.n_entries,
            "num_blocks": module.num_blocks,
            "total_weights": module.total_weights,
        }
        if module.bias is not None:
            layer_data["bias"] = module.bias.data.half()

        packed_state["quantized_layers"][name] = layer_data
        quantized_bytes += packed_indices.numel()
        codebook_bytes += codebook_f16.numel() * 2  # float16 = 2 bytes

    # Process float params (embeddings, norms, lm_head)
    float_bytes = 0
    nb_prefixes = set()
    for nb_name, nb_mod in model.named_modules():
        if isinstance(nb_mod, NativeBitLinear):
            nb_prefixes.add(nb_name + ".")

    for name, param in model.named_parameters():
        # Skip params that belong to NativeBitLinear layers
        is_nb = any(name.startswith(prefix) for prefix in nb_prefixes)
        if is_nb:
            continue

        # Skip lm_head (weight-tied with tok_emb)
        if name == "lm_head.weight":
            continue

        packed_state["float_params"][name] = param.data.half()
        float_bytes += param.numel() * 2

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed_state, output_path)

    # Compute stats
    file_size = output_path.stat().st_size
    ckpt_size = Path(ckpt_path).stat().st_size

    stats = {
        "packed_indices_bytes": quantized_bytes,
        "codebook_bytes": codebook_bytes,
        "float_param_bytes": float_bytes,
        "file_size_bytes": file_size,
        "original_size_bytes": ckpt_size,
        "compression_ratio": ckpt_size / file_size,
        "bits": bits,
        "n_quantized_layers": len(packed_state["quantized_layers"]),
    }
    return stats


# ---------------------------------------------------------------------------
# Load: packed format -> model ready for inference
# ---------------------------------------------------------------------------

def load_packed(path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a packed NativeBit model for fast inference.

    Returns a standard NativeBitGPT with use_nativebit=False (all nn.Linear),
    with weights reconstructed from packed indices + codebooks.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from nativebit.model import build_model_from_config

    packed = torch.load(path, map_location="cpu", weights_only=False)

    # Build config
    config_dict = packed["config"]
    bits = packed["bits"]

    if bits == 3:
        unpacker = unpack_3bit
    elif bits == 4:
        unpacker = unpack_4bit
    else:
        def unpacker(data, count):
            return unpack_nbits(data, count, bits)

    class Config:
        pass
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Build float model (no NativeBitLinear — all nn.Linear)
    model = build_model_from_config(config, use_nativebit=False)

    # Load float params
    for name, tensor in packed["float_params"].items():
        parts = name.split(".")
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        setattr(obj, parts[-1], nn.Parameter(tensor.float()))

    # Reconstruct quantized layer weights
    for layer_name, layer_data in packed["quantized_layers"].items():
        n_indices = layer_data["num_blocks"] * layer_data["block_size"]
        indices = unpacker(layer_data["packed_indices"], n_indices)
        codebook = layer_data["codebook"].float()

        weight = reconstruct_weight(
            indices, codebook,
            layer_data["shape"],
            layer_data["block_size"],
            layer_data["total_weights"],
        )

        # Navigate to the module and set weight
        parts = layer_name.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
        obj.weight = nn.Parameter(weight)

        if "bias" in layer_data:
            obj.bias = nn.Parameter(layer_data["bias"].float())

    # Re-tie lm_head weight
    model.lm_head.weight = model.tok_emb.weight

    model = model.to(device)
    model.requires_grad_(False)
    model.train(False)
    return model


# ---------------------------------------------------------------------------
# Verify: compare original vs packed model outputs
# ---------------------------------------------------------------------------

def verify_packed(ckpt_path: str, packed_path: str, device: str = "cpu") -> dict:
    """Verify packed model produces same outputs as original.

    Returns dict with max_diff, mean_diff, and pass/fail.
    """
    from nativebit.generate import load_model_from_checkpoint

    dev = torch.device(device)

    # Load original
    orig_model, orig_config = load_model_from_checkpoint(ckpt_path, dev)
    orig_model.requires_grad_(False)
    orig_model.train(False)

    # Load packed
    packed_model = load_packed(packed_path, device)

    # Test input — use model's vocab size
    vocab_size = orig_config.get("vocab_size", 50257)
    torch.manual_seed(42)
    test_input = torch.randint(0, vocab_size, (1, 32), device=dev)

    with torch.no_grad():
        orig_logits = orig_model(test_input).float()
        packed_logits = packed_model(test_input).float()

    diff = (orig_logits - packed_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # float16 codebooks introduce small rounding — allow tolerance
    passed = max_diff < 0.1

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "passed": passed,
    }
