"""Fused dequantize-matmul kernel for packed NativeBit inference.

Reads packed uint8 indices + codebook tables from HBM, reconstructs weight
tiles in VMEM, and multiplies — reading ~3x less HBM than float matmul.

Key: blocks are reordered at load time so each (TILE_N, block_size) sub-array
maps directly to a matmul operand. No reshape in the kernel.

Implementations:
  1. "tiled"  — Pallas kernel for TPU, tiled over N with unrolled K-chunks
  2. "naive"  — gather then matmul (works everywhere, slow to compile at scale)

Usage:
    from nativebit_jax.packed_kernel import packed_matmul, reorder_blocks_tiled
    # At load time: reorder once
    idx, cb = reorder_blocks_tiled(idx, cb, N, K, block_size)
    # At inference: fused dequant+matmul
    y = packed_matmul(x, idx, cb, N, K, block_size)

Toggle: NATIVEBIT_KERNEL=naive env var forces fallback.
"""

import math
import os

import jax
import jax.numpy as jnp

_KERNEL_MODE = os.environ.get("NATIVEBIT_KERNEL", "tiled")

TILE_N = 128


def reorder_blocks_tiled(indices, codebook, out_features, in_features,
                         block_size, tile_n=TILE_N):
    """Reorder blocks from flat weight order to tiled layout.

    Flat order: block b covers W[b // bpr, (b % bpr)*bs : (b % bpr + 1)*bs]
    Tiled order: blocks grouped by (tile, k_chunk, row_in_tile) so that
    each K-chunk's TILE_N blocks are contiguous — maps directly to matmul.

    Call once at load time. O(num_blocks) permutation, no copy of values.
    """
    import numpy as np
    N, K, bs = out_features, in_features, block_size
    bpr = K // bs           # blocks per row
    n_tiles = N // tile_n
    ne = codebook.shape[1]

    # (num_blocks, bs) → (N, bpr, bs) → (n_tiles, tile_n, bpr, bs)
    #   → transpose to (n_tiles, bpr, tile_n, bs) → flatten back
    idx = indices.reshape(N, bpr, bs)
    idx = idx.reshape(n_tiles, tile_n, bpr, bs).transpose(0, 2, 1, 3)
    idx = idx.reshape(-1, bs)

    cb = codebook.reshape(N, bpr, ne)
    cb = cb.reshape(n_tiles, tile_n, bpr, ne).transpose(0, 2, 1, 3)
    cb = cb.reshape(-1, ne)

    return idx, cb


def packed_matmul(x, indices, codebook, out_features, in_features, block_size):
    """Fused dequantize + matmul: y = x @ reconstruct(indices, codebook).T

    Expects indices/codebook in TILED layout (from reorder_blocks_tiled).

    Args:
        x: (M, in_features) input activations
        indices: (num_blocks, block_size) uint8 — tiled layout
        codebook: (num_blocks, n_entries) float32 — tiled layout
        out_features: N
        in_features: K
        block_size: bs

    Returns:
        y: (M, out_features) bfloat16
    """
    if _KERNEL_MODE == "tiled" and _pallas_available():
        return _packed_matmul_tiled(
            x, indices, codebook, out_features, in_features, block_size)
    return _packed_matmul_naive(
        x, indices, codebook, out_features, in_features, block_size)


def _packed_matmul_naive(x, indices, codebook, out_features, in_features, block_size):
    """Fallback: materialize full weight matrix, then matmul.

    Works with EITHER flat or tiled layout (gather doesn't care about order,
    as long as each block's codebook matches its indices).
    """
    num_blocks = indices.shape[0]
    block_idx = jnp.arange(num_blocks)[:, None]
    total = out_features * in_features
    w_flat = codebook[block_idx, indices.astype(jnp.int32)].reshape(-1)[:total]
    weight = w_flat.reshape(out_features, in_features)
    return (x.astype(jnp.bfloat16) @ weight.astype(jnp.bfloat16).T)


def _pallas_available():
    try:
        from jax.experimental import pallas as pl  # noqa: F401
        return True
    except ImportError:
        return False


def _packed_matmul_tiled(x, indices, codebook, out_features, in_features, block_size):
    """Pallas kernel: tiled dequant+matmul for TPU.

    Expects TILED block layout (from reorder_blocks_tiled). For each output
    tile of TILE_N rows, loads all K-chunks' blocks from HBM into VMEM,
    then unrolls over K-chunks: reconstruct (TILE_N, block_size) weight
    sub-matrix via compare+select, partial matmul, accumulate.

    No reshape inside kernel — each (TILE_N, block_size) block is already
    the right shape for matmul.

    HBM reads per layer (M=1 decode):
      indices: N * K * 1 byte (uint8, cast to int32 outside)
      codebook: N * (K/bs) * ne * 4 bytes
      x: K * 2 bytes
      With bs=128, ne=8: ~1.25 bytes/weight vs 4 bytes/weight (float)
    """
    from jax.experimental import pallas as pl

    M = x.shape[0]
    N = out_features
    K = in_features
    bs = block_size
    ne = codebook.shape[1]
    bpr = K // bs                          # blocks per row (K-chunks)
    blocks_per_tile = bpr * TILE_N         # total blocks loaded per output tile

    def kernel(x_ref, idx_ref, cb_ref, y_ref):
        x_val = x_ref[...]                # (M, K) — full input
        idx_all = idx_ref[...]             # (bpr * TILE_N, bs) — this tile's blocks
        cb_all = cb_ref[...]               # (bpr * TILE_N, ne) — this tile's codebooks

        # Accumulate partial matmuls over K-chunks (unrolled at trace time)
        y_acc = jnp.zeros((M, TILE_N), dtype=jnp.float32)

        for kb in range(bpr):
            # Static slice: blocks for this K-chunk (contiguous in tiled layout)
            s = kb * TILE_N
            idx_k = idx_all[s:s + TILE_N, :]        # (TILE_N, bs) int32
            cb_k = cb_all[s:s + TILE_N, :]           # (TILE_N, ne) f32
            x_k = x_val[:, kb * bs:(kb + 1) * bs]   # (M, bs)

            # Reconstruct weight chunk via compare+select (8 iterations, unrolled)
            w_k = jnp.zeros((TILE_N, bs), dtype=jnp.bfloat16)
            for e in range(ne):
                mask = (idx_k == e).astype(jnp.bfloat16)
                w_k = w_k + mask * cb_k[:, e:e + 1].astype(jnp.bfloat16)

            # Partial matmul: (M, bs) @ (bs, TILE_N) → (M, TILE_N)
            y_acc = y_acc + jax.lax.dot(
                x_k.astype(jnp.bfloat16), w_k.T,
                preferred_element_type=jnp.float32)

        y_ref[...] = y_acc

    num_tiles = N // TILE_N
    indices_i32 = indices.astype(jnp.int32)

    y = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((M, K), lambda i: (0, 0)),
            pl.BlockSpec((blocks_per_tile, bs), lambda i: (i * blocks_per_tile, 0)),
            pl.BlockSpec((blocks_per_tile, ne), lambda i: (i * blocks_per_tile, 0)),
        ],
        out_specs=pl.BlockSpec((M, TILE_N), lambda i: (0, i * TILE_N)),
    )(x, indices_i32, codebook)

    return y.astype(jnp.bfloat16)
