"""Triton fused dequant-matvec kernel for NativeBit packed inference.

For decode (M=1), the matmul is bandwidth-bound. This kernel reads uint8
codebook indices (~1 byte/weight) instead of fp16 weights (2 bytes/weight),
achieving ~1.6x less bandwidth with near-peak utilization via coalesced
memory access and register-level codebook select.

Matches or beats cuBLAS fp16 matvec on RTX 3070 (Ampere).

Usage:
    from inference.triton_kernel import dequant_matvec
    y = dequant_matvec(x, indices, codebook, N, K)  # all torch.Tensor on CUDA
"""
import torch
import triton
import triton.language as tl

BS = 128   # block_size (weights per codebook block)
NE = 8     # n_entries (codebook size per block, 3-bit = 8)
TILE_N = 32  # output rows per thread block (tuned for RTX 3070)


@triton.jit
def _dequant_matvec_kernel(
    x_ptr, idx_ptr, cb_ptr, y_ptr,
    BPR: tl.constexpr,
    TILE_N: tl.constexpr,
    BS: tl.constexpr,
    NE: tl.constexpr,
):
    """Fused codebook dequant + vector-matrix multiply.

    Each thread block computes TILE_N output values. For each K-chunk:
    1. Load x chunk (BS floats, shared across rows) — coalesced
    2. Load uint8 indices (TILE_N * BS bytes) — coalesced
    3. Load codebook entries per-entry (8 coalesced loads of TILE_N floats)
    4. Reconstruct weights via compare+select in registers
    5. Element-wise multiply + reduce -> TILE_N partial outputs
    """
    pid = tl.program_id(0)
    n_offs = pid * TILE_N + tl.arange(0, TILE_N)

    y_acc = tl.zeros((TILE_N,), dtype=tl.float32)

    for kb in range(BPR):
        x_k = tl.load(x_ptr + kb * BS + tl.arange(0, BS)).to(tl.float32)
        block_ids = n_offs * BPR + kb

        # Coalesced index load: sequential block_ids in memory
        idx_ptrs = block_ids[:, None] * BS + tl.arange(0, BS)[None, :]
        indices = tl.load(idx_ptr + idx_ptrs).to(tl.int32)

        # Coalesced codebook loads + register select (8 iterations)
        weights = tl.zeros((TILE_N, BS), dtype=tl.float32)
        for e in range(NE):
            cb_e = tl.load(cb_ptr + block_ids * NE + e)
            mask = (indices == e)
            weights += tl.where(mask, cb_e[:, None], tl.zeros((TILE_N, BS), tl.float32))

        y_acc += tl.sum(weights * x_k[None, :], axis=1)

    tl.store(y_ptr + n_offs, y_acc)


@triton.jit
def _dequant_matvec_3bit_kernel(
    x_ptr, packed_ptr, cb_ptr, y_ptr,
    BPR: tl.constexpr,
    TILE_N: tl.constexpr,
    BS: tl.constexpr,
    NE: tl.constexpr,
    PPB: tl.constexpr,   # packed bytes per block = BS * 3 // 8
    N_GROUPS: tl.constexpr,  # groups per block = BS // 8
):
    """Fused 3-bit unpack + codebook dequant + matvec.

    Reads 3-bit packed indices from VRAM (0.375 bytes/weight).
    Unpacks 3 bytes → 8 indices via shift+mask in registers.
    Then codebook select + dot as in the uint8 kernel.

    VRAM per layer: N*K*3/8 (packed) + N*(K/BS)*NE*4 (codebook) + K*4 (x)
    For 7680x2560: 7.4 MB + 4.8 MB = 12.2 MB (vs 24.5 MB uint8, 39.3 MB fp16)
    """
    pid = tl.program_id(0)
    n_offs = pid * TILE_N + tl.arange(0, TILE_N)
    y_acc = tl.zeros((TILE_N,), dtype=tl.float32)

    shifts = tl.arange(0, 8) * 3  # [0, 3, 6, 9, 12, 15, 18, 21]

    for kb in range(BPR):
        block_ids = n_offs * BPR + kb

        for g in range(N_GROUPS):
            # Load 3 packed bytes per block → 24-bit int
            base = block_ids * PPB + g * 3
            b0 = tl.load(packed_ptr + base).to(tl.int32)
            b1 = tl.load(packed_ptr + base + 1).to(tl.int32)
            b2 = tl.load(packed_ptr + base + 2).to(tl.int32)
            bits24 = b0 | (b1 << 8) | (b2 << 16)  # (TILE_N,)

            # Unpack 8 indices: (TILE_N, 8)
            indices_g = (bits24[:, None] >> shifts[None, :]) & 0x7

            # x values for this group of 8
            x_g = tl.load(x_ptr + kb * BS + g * 8 + tl.arange(0, 8)).to(tl.float32)

            # Codebook select on (TILE_N, 8)
            weights_g = tl.zeros((TILE_N, 8), dtype=tl.float32)
            for e in range(NE):
                cb_e = tl.load(cb_ptr + block_ids * NE + e)
                mask = (indices_g == e)
                weights_g += tl.where(mask, cb_e[:, None], tl.zeros((TILE_N, 8), tl.float32))

            y_acc += tl.sum(weights_g * x_g[None, :], axis=1)

    tl.store(y_ptr + n_offs, y_acc)


def dequant_matvec(x, indices, codebook, N, K, tile_n=TILE_N):
    """Fused dequant + matvec: y = reconstruct(indices, codebook) @ x.

    Args:
        x: (K,) float32 input vector
        indices: (num_blocks, BS) uint8 codebook indices (flat layout)
        codebook: (num_blocks, NE) float32 codebook values
        N: number of output features
        K: number of input features
        tile_n: output tile size (32 or 64 work best on Ampere)

    Returns:
        y: (N,) float32 output vector
    """
    bpr = int(K) // BS
    y = torch.empty(int(N), dtype=torch.float32, device=x.device)
    grid = (int(N) // tile_n,)
    _dequant_matvec_kernel[grid](
        x, indices.view(-1), codebook.view(-1), y,
        BPR=bpr, TILE_N=tile_n, BS=BS, NE=NE,
    )
    return y


@triton.jit
def _dequant_3bit_fast_kernel(
    x_ptr, packed_ptr, cb_ptr, y_ptr,
    BPR: tl.constexpr,
    TILE_N: tl.constexpr,
    BS: tl.constexpr,
    NE: tl.constexpr,
    PPB: tl.constexpr,
):
    """V2 3-bit: load all 48 packed bytes at once, unpack group-by-group,
    accumulate dot products. Uses (TILE_N, 48) bulk load for coalescing."""
    pid = tl.program_id(0)
    n_offs = pid * TILE_N + tl.arange(0, TILE_N)
    y_acc = tl.zeros((TILE_N,), dtype=tl.float32)
    shifts = tl.arange(0, 8) * 3

    for kb in range(BPR):
        block_ids = n_offs * BPR + kb

        # Bulk load packed bytes — pad to 64 (power of 2) for tl.arange
        packed_ptrs = block_ids[:, None] * PPB + tl.arange(0, 64)[None, :]
        mask = tl.arange(0, 64)[None, :] < PPB
        packed = tl.load(packed_ptr + packed_ptrs, mask=mask, other=0).to(tl.int32)

        for g in range(BS // 8):
            b0 = packed[:, g * 3]
            b1 = packed[:, g * 3 + 1]
            b2 = packed[:, g * 3 + 2]
            bits24 = b0 | (b1 << 8) | (b2 << 16)

            idx_g = (bits24[:, None] >> shifts[None, :]) & 0x7  # (TILE_N, 8)
            x_g = tl.load(x_ptr + kb * BS + g * 8 + tl.arange(0, 8)).to(tl.float32)

            w_g = tl.zeros((TILE_N, 8), dtype=tl.float32)
            for e in range(NE):
                cb_e = tl.load(cb_ptr + block_ids * NE + e)
                w_g += tl.where(idx_g == e, cb_e[:, None], tl.zeros((TILE_N, 8), tl.float32))

            y_acc += tl.sum(w_g * x_g[None, :], axis=1)

    tl.store(y_ptr + n_offs, y_acc)


def dequant_matvec_3bit(x, packed_indices, codebook, N, K, tile_n=TILE_N):
    """Fused dequant + matvec with 3-bit packed indices in VRAM."""
    bpr = int(K) // BS
    ppb = BS * 3 // 8
    y = torch.empty(int(N), dtype=torch.float32, device=x.device)
    grid = (int(N) // tile_n,)
    _dequant_3bit_fast_kernel[grid](
        x, packed_indices, codebook.view(-1), y,
        BPR=bpr, TILE_N=tile_n, BS=BS, NE=NE, PPB=ppb,
    )
    return y


def pack_indices_to_3bit(indices_uint8):
    """Pack uint8 indices (0-7) to 3-bit format on GPU.

    8 indices → 3 bytes. Returns flat uint8 tensor.
    """
    flat = indices_uint8.reshape(-1)
    # Pad to multiple of 8
    pad = (8 - flat.shape[0] % 8) % 8
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))

    groups = flat.reshape(-1, 8).to(torch.int32)
    # Pack: 8 values × 3 bits = 24 bits = 3 bytes
    bits24 = torch.zeros(groups.shape[0], dtype=torch.int32, device=flat.device)
    for j in range(8):
        bits24 |= (groups[:, j] & 0x7) << (j * 3)

    # Split 24-bit int into 3 bytes
    b0 = (bits24 & 0xFF).to(torch.uint8)
    b1 = ((bits24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((bits24 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=1).reshape(-1)


class PackedLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear using packed NativeBit weights.

    Stores 3-bit packed indices + fp32 codebook in VRAM.
    Forward uses the fused Triton dequant-matvec kernel with in-kernel unpacking.

    VRAM: ~0.375 bytes/weight (indices) + ~0.25 bytes/weight (codebook)
    vs fp16 nn.Linear: 2 bytes/weight → ~3.2x less VRAM.
    """

    def __init__(self, packed_indices, codebook, out_features, in_features):
        super().__init__()
        self.register_buffer('packed_indices', packed_indices)  # flat uint8, 3-bit packed
        self.register_buffer('codebook', codebook)              # (num_blocks, NE) fp32
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        if x_flat.shape[0] == 1:
            # Try CUDA kernel (register-indexed lookup), fall back to Triton
            try:
                from inference.cuda_kernel import dequant_matvec_3bit_cuda
                y = dequant_matvec_3bit_cuda(
                    x_flat[0], self.packed_indices, self.codebook,
                    self.out_features, self.in_features,
                )
            except Exception:
                y = dequant_matvec_3bit(
                    x_flat[0], self.packed_indices, self.codebook,
                    self.out_features, self.in_features,
                )
            return y.reshape(*shape[:-1], self.out_features)
        else:
            # Batched: unpack + reconstruct + matmul
            num_blocks = self.codebook.shape[0]
            # Unpack 3-bit to uint8 for gather
            n_groups = self.packed_indices.shape[0] // 3
            packed = self.packed_indices.reshape(n_groups, 3).to(torch.int32)
            bits24 = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            indices = torch.zeros(n_groups, 8, dtype=torch.int64, device=x.device)
            for j in range(8):
                indices[:, j] = (bits24 >> (j * 3)) & 0x7
            total_idx = num_blocks * BS
            indices = indices.reshape(-1)[:total_idx].reshape(num_blocks, BS)
            block_idx = torch.arange(num_blocks, device=x.device).unsqueeze(1)
            total = self.out_features * self.in_features
            w = self.codebook[block_idx, indices].reshape(-1)[:total]
            w = w.reshape(self.out_features, self.in_features)
            y = x_flat @ w.T
            return y.reshape(*shape[:-1], self.out_features)

    @staticmethod
    def from_packed(indices_np, codebook_np, weight_shape, device='cuda'):
        """Create from unpacked uint8 numpy arrays (from .nbpack.npz).

        Packs indices to 3-bit on GPU for minimal VRAM.
        """
        indices_uint8 = torch.from_numpy(indices_np.copy()).to(device)
        packed = pack_indices_to_3bit(indices_uint8)
        codebook = torch.from_numpy(codebook_np.copy()).to(device)
        return PackedLinear(packed, codebook, weight_shape[0], weight_shape[1])
