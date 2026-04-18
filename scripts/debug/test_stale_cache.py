"""Diagnostic: how stale is the NB 2.2B checkpoint's cached delta?

Compares the loaded `qw_delta` (stale cache) against a freshly recomputed
`Q(w) - w`. If drift is large relative to codebook spacing, Bug 2 matters.

Cheap — only loads a few layers at a time, no forward pass.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
try:
    from ml_dtypes import bfloat16  # JAX ships this transitively
except ImportError:
    import jax.numpy as jnp_tmp
    bfloat16 = jnp_tmp.bfloat16


CKPT = "logs/gcs/2b_nb_fixed_params.npz"
BLOCK_SIZE = 128  # TPU2BConfig.block_size
N_ENTRIES = 8


def fresh_quantize(w: np.ndarray, codebook: np.ndarray, block_size: int):
    """Recompute nearest-codebook quantized weights."""
    shape = w.shape
    flat = w.reshape(-1)
    total = flat.size
    num_blocks = codebook.shape[0]
    padded = num_blocks * block_size
    if padded > total:
        flat = np.concatenate([flat, np.zeros(padded - total, dtype=flat.dtype)])
    blocks = flat.reshape(num_blocks, block_size)
    # distances: (num_blocks, block_size, n_entries)
    d = (blocks[:, :, None] - codebook[:, None, :]) ** 2
    idx = d.argmin(axis=-1)  # (num_blocks, block_size)
    block_idx = np.arange(num_blocks)[:, None]
    q = codebook[block_idx, idx].reshape(-1)[:total]
    return q.reshape(shape), idx


def main():
    print(f"Loading {CKPT} (mmap)...")
    data = np.load(CKPT, mmap_mode="r")

    # Find NB layer keys (weight + codebook + cache/qw_delta triples)
    all_keys = list(data.keys())
    nb_layers = []
    for k in all_keys:
        if k.endswith("/weight") and "NativeBitDense" in k:
            base = k.rsplit("/weight", 1)[0]
            cb_key = base + "/codebook"
            cache_key = "cache/" + base.replace("params/", "", 1) + "/qw_delta"
            if cb_key in data and cache_key in data:
                nb_layers.append((base, k, cb_key, cache_key))
    print(f"Found {len(nb_layers)} NativeBit layers.\n")

    # Analyze all layers (summary) + show a few per-layer details
    drifts = []
    spacings = []
    quant_distances = []
    fwd_diffs = []
    print(f"{'layer':<70s} {'drift_rms':>10s} {'spacing':>10s} {'ratio':>6s} {'fwd_diff':>10s}")
    print("-" * 120)

    for i, (base, wk, ck, cachek) in enumerate(nb_layers):
        w = np.asarray(data[wk]).astype(np.float32)
        cb = np.asarray(data[ck]).astype(np.float32)
        # Cache is bf16 stored as |V2 — view as bf16 then cast
        raw = np.asarray(data[cachek])
        if raw.dtype.kind == "V":
            stale_delta = raw.view(bfloat16).astype(np.float32)
        else:
            stale_delta = raw.astype(np.float32)

        q_fresh, idx_fresh = fresh_quantize(w, cb, BLOCK_SIZE)
        fresh_delta = (q_fresh - w).astype(np.float32)

        # Implied STALE quantization: stale_forward was w_old + fresh_Q(w_old).
        # We don't have w_old, but we can recover the stale index by finding which
        # codebook entry `w + stale_delta` is closest to.
        stale_q = w + stale_delta  # what the forward actually computed
        # For each weight, find nearest codebook entry
        total = w.size
        num_blocks = cb.shape[0]
        padded = num_blocks * BLOCK_SIZE
        sqf = stale_q.reshape(-1)
        if padded > total:
            sqf = np.concatenate([sqf, np.zeros(padded - total, dtype=sqf.dtype)])
        sq_blocks = sqf.reshape(num_blocks, BLOCK_SIZE)
        d_stale = (sq_blocks[:, :, None] - cb[:, None, :]) ** 2
        idx_stale = d_stale.argmin(axis=-1)
        boundary_crossings = float((idx_stale != idx_fresh).mean())

        # Stale forward: w + stale_delta. Fresh forward: Q(w) = w + fresh_delta.
        # Difference between stale and fresh forward values:
        fwd_diff = stale_delta - fresh_delta  # (this is what the forward is off by)

        drift_rms = float(np.sqrt((fwd_diff ** 2).mean()))
        # Codebook spacing per block (gap between sorted entries)
        cb_sorted = np.sort(cb, axis=1)
        spacing = float(np.mean(cb_sorted[:, 1:] - cb_sorted[:, :-1]))

        # How far each weight is from its assigned codebook entry (the quantization error)
        quant_err = float(np.sqrt((fresh_delta ** 2).mean()))

        drifts.append(drift_rms)
        spacings.append(spacing)
        quant_distances.append(quant_err)
        fwd_diffs.append(drift_rms)
        if i == 0:
            all_crossings = []
        all_crossings.append(boundary_crossings)

        if i < 8 or i >= len(nb_layers) - 4:
            short = base.replace("params/", "")
            ratio = drift_rms / spacing if spacing > 0 else 0
            print(f"{short:<70s} {drift_rms:>10.5f} {spacing:>10.5f} {ratio:>6.2f} {drift_rms:>10.5f}")
        elif i == 8:
            print("  ...")

    drifts = np.array(drifts)
    spacings = np.array(spacings)
    quant_distances = np.array(quant_distances)

    print("\n=== Summary ===")
    print(f"  Layers                       : {len(drifts)}")
    print(f"  Mean drift RMS (stale-fresh) : {drifts.mean():.5f}")
    print(f"  Mean codebook spacing        : {spacings.mean():.5f}")
    print(f"  Drift / spacing ratio (mean) : {(drifts/spacings).mean():.3f}")
    print(f"  Mean fresh quantization error: {quant_distances.mean():.5f}")
    print(f"  Drift / quant_err ratio      : {(drifts/quant_distances).mean():.3f}")
    crossings = np.array(all_crossings)
    print(f"  Weights crossing boundary    : {crossings.mean()*100:.2f}%  "
          f"(min={crossings.min()*100:.2f}%, max={crossings.max()*100:.2f}%)")
    print()
    if (drifts / spacings).mean() > 0.5:
        print("  >>> Stale cache drift is LARGE relative to codebook spacing.")
        print("  >>> Bug 2 (no requantize before eval) is REAL — eval is comparing")
        print("      stale quantized weights against post-hoc's fresh quantized weights.")
    elif (drifts / spacings).mean() > 0.1:
        print("  >>> Drift is moderate. Bug 2 probably contributes but isn't dominant.")
    else:
        print("  >>> Drift is small. Bug 2 is minor — cache is basically fresh.")


if __name__ == "__main__":
    main()
