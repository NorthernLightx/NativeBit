"""Pack a NativeBit training checkpoint into minimal inference format.

Training checkpoint stores: latent weights (fp32) + codebooks + cached deltas.
Packed format stores only: codebook indices (3-bit) + codebook tables + non-quantized params.

Usage:
    python inference/pack.py logs/gcs/2b_nb3_s42_params.npz --out inference/2b_nb3.nbpack
"""
import argparse
import json
import math
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def quantize_weight(weight, codebook):
    """Quantize weight to codebook indices."""
    num_blocks, n_entries = codebook.shape
    total = weight.size
    block_size = math.ceil(total / num_blocks)
    padded = num_blocks * block_size

    w_flat = weight.reshape(-1)
    if padded > total:
        w_flat = np.pad(w_flat, (0, padded - total))

    w_blocks = w_flat.reshape(num_blocks, block_size)
    dists = (w_blocks[:, :, None] - codebook[:, None, :]) ** 2
    indices = np.argmin(dists, axis=-1).astype(np.uint8)
    return indices


def pack_indices_3bit(indices):
    """Pack uint8 indices (0-7) into 3-bit packed bytes.

    8 indices → 3 bytes (24 bits = 8 × 3 bits). Vectorized with numpy.
    """
    flat = indices.reshape(-1)
    # Pad to multiple of 8
    pad = (8 - len(flat) % 8) % 8
    if pad:
        flat = np.pad(flat, (0, pad))

    groups = flat.reshape(-1, 8).astype(np.uint32)
    bits24 = np.zeros(len(groups), dtype=np.uint32)
    for j in range(8):
        bits24 |= (groups[:, j] & 0x7) << (j * 3)
    packed = np.zeros((len(groups), 3), dtype=np.uint8)
    packed[:, 0] = bits24 & 0xFF
    packed[:, 1] = (bits24 >> 8) & 0xFF
    packed[:, 2] = (bits24 >> 16) & 0xFF
    return packed.tobytes()


def pack_checkpoint(ckpt_path, out_path, block_size=128, n_entries=8):
    """Convert training checkpoint to packed inference format."""
    print(f"Loading {ckpt_path}...")
    ckpt = np.load(ckpt_path)
    keys = list(ckpt.keys())

    # Separate quantized layers from non-quantized params
    # Quantized layers have both "weight" and "codebook" keys
    weight_keys = [k for k in keys if k.endswith("/weight")]
    codebook_keys = [k for k in keys if k.endswith("/codebook")]

    # Match weights to codebooks
    quantized = {}
    for wk in weight_keys:
        prefix = wk.rsplit("/weight", 1)[0]
        ck = prefix + "/codebook"
        if ck in codebook_keys:
            quantized[prefix] = (wk, ck)

    # Non-quantized params (embedding, norms, biases)
    non_quant_keys = [k for k in keys
                      if not any(k.startswith(p) for p in quantized)
                      or k.endswith("/codebook")]
    # Actually: keep everything that's NOT a weight in a quantized layer
    non_quant_keys = []
    for k in keys:
        prefix = k.rsplit("/", 1)[0] if "/" in k else ""
        if prefix in quantized and k.endswith("/weight"):
            continue  # Skip quantized weights (replaced by indices)
        if k.endswith("/codebook"):
            continue  # Codebooks stored separately
        # Skip cached deltas
        if "/qw_delta" in k:
            continue
        non_quant_keys.append(k)

    print(f"  Quantized layers: {len(quantized)}")
    print(f"  Non-quantized params: {len(non_quant_keys)}")

    # Build packed format
    packed_data = {
        "format": "nbpack_v1",
        "block_size": block_size,
        "n_entries": n_entries,
        "layers": {},
        "params": {},
    }

    # Quantize all layers once (avoid double computation)
    save_dict = {}
    total_indices_bytes = 0
    total_codebook_bytes = 0

    for idx, (prefix, (wk, ck)) in enumerate(sorted(quantized.items())):
        weight = ckpt[wk].astype(np.float32)
        codebook = ckpt[ck].astype(np.float32)

        indices = quantize_weight(weight, codebook)
        packed_idx = pack_indices_3bit(indices)

        safe_prefix = prefix.replace("/", ".")
        save_dict[f"idx.{safe_prefix}"] = np.frombuffer(packed_idx, dtype=np.uint8)
        save_dict[f"cb.{safe_prefix}"] = codebook
        save_dict[f"shape.{safe_prefix}"] = np.array(weight.shape)
        save_dict[f"idxshape.{safe_prefix}"] = np.array(indices.shape)

        total_indices_bytes += len(packed_idx)
        total_codebook_bytes += codebook.nbytes
        print(f"    [{idx+1}/{len(quantized)}] {prefix}: "
              f"{weight.shape} -> {len(packed_idx)/1e6:.1f} MB", flush=True)

    total_nonquant_bytes = sum(ckpt[k].nbytes for k in non_quant_keys)

    print(f"\n  Packed size breakdown:")
    print(f"    Indices (3-bit):    {total_indices_bytes / 1e9:.2f} GB")
    print(f"    Codebooks (fp32):   {total_codebook_bytes / 1e9:.2f} GB")
    print(f"    Non-quantized:      {total_nonquant_bytes / 1e9:.2f} GB")
    total = total_indices_bytes + total_codebook_bytes + total_nonquant_bytes
    print(f"    TOTAL:              {total / 1e9:.2f} GB")

    # Non-quantized params
    for k in non_quant_keys:
        arr = ckpt[k]
        if arr.dtype.kind == 'V':  # bfloat16 as void
            arr = np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(arr.shape)
        save_dict[f"param.{k.replace('/', '.')}"] = arr

    # Metadata
    save_dict["__metadata__"] = np.array([block_size, n_entries, len(quantized), len(non_quant_keys)])

    np.savez_compressed(out_path, **save_dict)
    actual_size = os.path.getsize(out_path)
    print(f"\n  Saved: {out_path} ({actual_size / 1e9:.2f} GB on disk)")
    print(f"  Compression vs fp32: {sum(ckpt[k].nbytes for k in keys) / actual_size:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-entries", type=int, default=8)
    args = parser.parse_args()

    out = args.out or args.checkpoint.replace("_params.npz", ".nbpack.npz")
    pack_checkpoint(args.checkpoint, out, args.block_size, args.n_entries)
