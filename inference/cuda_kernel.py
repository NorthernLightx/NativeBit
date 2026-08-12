"""CUDA fused 3-bit dequant-matvec kernel for NativeBit packed inference.

One warp per output row; lanes stride across 3-byte index groups so packed
reads coalesce. Codebook dequant is a dynamically-indexed local array
(L1-resident, one hot 32-byte line per block); warp shuffle reduces the
partial sums. Supports block_size 128 (TPU configs) and 64 (local configs).

Three variants, auto-dispatched: shuffle-LUT (default when K % 256 == 0),
byte-load, uint32-load. The shuffle-LUT kernel holds each block's 8 codebook
entries in warp lanes and dequantizes via __shfl_sync with a variable source
lane — a pure ALU lookup with no local-memory array — plus float4 x loads
and a 2-step / 2-accumulator unroll.

Measured on RTX 3070 (interleaved CUDA-event timing, warm clocks): the
shuffle-LUT variant runs 2.2B decode shapes in ~39-54us vs cuBLAS fp16
matvec ~96-107us — 2.0-2.5x, 250-300 GB/s effective (~2/3 of DRAM peak).
The byte variant (no K%256 requirement) is ~1.2-1.4x cuBLAS.
"""
import os
import torch

# Ensure ninja + cl.exe on PATH for JIT compilation
try:
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

# Stable per-user build dir. The default lives under AppData, which Windows
# Store Python virtualizes per-app — builds from different interpreters or
# an elevated process poison it and later rebuilds fail with
# "ninja: error: loading 'build.ninja'".
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR", os.path.join(os.path.expanduser("~"), ".nativebit_ext"))

# Find MSVC cl.exe if not on PATH
import glob
_msvc_paths = glob.glob(
    r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64")
for p in sorted(_msvc_paths, reverse=True):
    if os.path.exists(os.path.join(p, "cl.exe")):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        break

from torch.utils.cpp_extension import load_inline

BS = 128
NE = 8

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// One WARP per output row, lanes stride across the row's 8-index groups.
// Byte-load variant: adjacent lanes read adjacent 3-byte chunks (coalesced).
// Wins at large K where the uint32 variant's register pressure hurts.
__global__ void dequant_matvec_3bit_bytes_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int groups_per_row, int gpb
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    const uint8_t* prow = packed + (size_t)warp_id * groups_per_row * 3;
    const size_t block_row_base = (size_t)warp_id * (groups_per_row / gpb);

    float acc = 0.0f;
    for (int g = lane; g < groups_per_row; g += 32) {
        uint32_t b0 = prow[g * 3];
        uint32_t b1 = prow[g * 3 + 1];
        uint32_t b2 = prow[g * 3 + 2];
        uint32_t bits24 = b0 | (b1 << 8) | (b2 << 16);

        const float* cbp = codebook + (block_row_base + g / gpb) * 8;
        float cb[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) cb[e] = cbp[e];

        const float* xg = x + (size_t)g * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += cb[(bits24 >> (j * 3)) & 0x7] * xg[j];
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// Shuffle-LUT variant: the 8 codebook entries of each block live in warp
// lanes, and __shfl_sync with a variable source lane performs the dequant
// lookup as a pure ALU op — no local-memory array, no select tree. With
// float4 x loads, LSU pressure drops ~3x vs the byte kernel. Requires
// K % 256 == 0 (32 groups per warp-step).
template<int GPB>  // groups per codebook block = block_size / 8 (8 or 16)
__global__ void dequant_matvec_3bit_shfl_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int groups_per_row
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    const uint8_t* prow = packed + (size_t)warp_id * groups_per_row * 3;
    const float* cbrow = codebook + (size_t)warp_id * (groups_per_row / GPB) * 8;

    const int sub = lane % GPB;             // group index within block
    const int blk = lane / GPB;             // block slot within warp-step
    const int BLOCKS_PER_STEP = 32 / GPB;
    const int lut_base = blk * GPB;         // shuffle source base for this block

    // Two accumulators + 2-step unroll: breaks the FMA dependency chain and
    // overlaps the next step's loads with this step's shuffles.
    float acc0 = 0.0f, acc1 = 0.0f;
    int steps = groups_per_row / 32;
    int s = 0;
    for (; s + 2 <= steps; s += 2) {
        int g0 = s * 32 + lane;
        int g1 = g0 + 32;
        int cb_block0 = s * BLOCKS_PER_STEP + blk;
        int cb_block1 = cb_block0 + BLOCKS_PER_STEP;
        float cb_reg0 = (sub < 8) ? cbrow[cb_block0 * 8 + sub] : 0.0f;
        float cb_reg1 = (sub < 8) ? cbrow[cb_block1 * 8 + sub] : 0.0f;

        uint32_t bitsA = (uint32_t)prow[g0 * 3]
                       | ((uint32_t)prow[g0 * 3 + 1] << 8)
                       | ((uint32_t)prow[g0 * 3 + 2] << 16);
        uint32_t bitsB = (uint32_t)prow[g1 * 3]
                       | ((uint32_t)prow[g1 * 3 + 1] << 8)
                       | ((uint32_t)prow[g1 * 3 + 2] << 16);

        const float4* x4a = reinterpret_cast<const float4*>(x + (size_t)g0 * 8);
        const float4* x4b = reinterpret_cast<const float4*>(x + (size_t)g1 * 8);
        float4 a0 = x4a[0], a1 = x4a[1], b0v = x4b[0], b1v = x4b[1];
        float xsA[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float xsB[8] = {b0v.x, b0v.y, b0v.z, b0v.w, b1v.x, b1v.y, b1v.z, b1v.w};

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int iA = (bitsA >> (j * 3)) & 0x7;
            int iB = (bitsB >> (j * 3)) & 0x7;
            acc0 += __shfl_sync(0xffffffff, cb_reg0, lut_base + iA) * xsA[j];
            acc1 += __shfl_sync(0xffffffff, cb_reg1, lut_base + iB) * xsB[j];
        }
    }
    for (; s < steps; s++) {
        int g = s * 32 + lane;
        int cb_block = s * BLOCKS_PER_STEP + blk;
        float cb_reg = (sub < 8) ? cbrow[cb_block * 8 + sub] : 0.0f;
        uint32_t bits = (uint32_t)prow[g * 3]
                      | ((uint32_t)prow[g * 3 + 1] << 8)
                      | ((uint32_t)prow[g * 3 + 2] << 16);
        const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)g * 8);
        float4 xa = x4[0], xb = x4[1];
        float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int idx = (bits >> (j * 3)) & 0x7;
            acc0 += __shfl_sync(0xffffffff, cb_reg, lut_base + idx) * xs[j];
        }
    }

    float acc = acc0 + acc1;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// Multi-row shuffle-LUT: one warp computes 4 output rows, sharing the x
// loads across them. x re-reads from L2 were a hidden co-limiter (~76MB per
// call at 2.2B qkv); 4 rows per warp cuts that 4x. Per row a separate
// cb register serves as the 32-lane shuffle LUT.
template<int GPB>
__global__ void dequant_matvec_3bit_shfl4_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int groups_per_row
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int row0 = warp_id * 4;
    if (row0 >= N) return;

    const int sub = lane % GPB;
    const int blk = lane / GPB;
    const int BLOCKS_PER_STEP = 32 / GPB;
    const int lut_base = blk * GPB;
    const int blocks_per_row = groups_per_row / GPB;

    const uint8_t* prow0 = packed + (size_t)row0 * groups_per_row * 3;
    const float* cbrow0 = codebook + (size_t)row0 * blocks_per_row * 8;
    const size_t row_pstride = (size_t)groups_per_row * 3;
    const size_t row_cstride = (size_t)blocks_per_row * 8;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int steps = groups_per_row / 32;
    for (int s = 0; s < steps; s++) {
        int g = s * 32 + lane;
        int cb_block = s * BLOCKS_PER_STEP + blk;

        // Shared x for all 4 rows
        const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)g * 8);
        float4 xa = x4[0], xb = x4[1];
        float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};

        // Per-row packed bits + codebook LUT register
        uint32_t bits[4];
        float cb_reg[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const uint8_t* pr = prow0 + r * row_pstride;
            bits[r] = (uint32_t)pr[g * 3]
                    | ((uint32_t)pr[g * 3 + 1] << 8)
                    | ((uint32_t)pr[g * 3 + 2] << 16);
            cb_reg[r] = (sub < 8)
                ? cbrow0[r * row_cstride + cb_block * 8 + sub] : 0.0f;
        }

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xv = xs[j];
            acc0 += __shfl_sync(0xffffffff, cb_reg[0],
                                lut_base + ((bits[0] >> (j * 3)) & 0x7)) * xv;
            acc1 += __shfl_sync(0xffffffff, cb_reg[1],
                                lut_base + ((bits[1] >> (j * 3)) & 0x7)) * xv;
            acc2 += __shfl_sync(0xffffffff, cb_reg[2],
                                lut_base + ((bits[2] >> (j * 3)) & 0x7)) * xv;
            acc3 += __shfl_sync(0xffffffff, cb_reg[3],
                                lut_base + ((bits[3] >> (j * 3)) & 0x7)) * xv;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (lane == 0) {
        y[row0] = acc0;
        y[row0 + 1] = acc1;
        y[row0 + 2] = acc2;
        y[row0 + 3] = acc3;
    }
}

// Select-tree variant: uint32-chunk loads (4 groups = 12B = 3 aligned u32)
// with the codebook in 8 named registers and a 7-select tree per weight.
// The uint32 kernel's flaw was its dynamically-indexed local array (every
// access an LSU op); selects run on the wide FP pipe instead, and the 8
// codebook loads amortize over a full 32-weight chunk. ~0.6 LSU/weight.
__global__ void dequant_matvec_3bit_seltree_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int groups_per_row, int gpb
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    const uint8_t* prow = packed + (size_t)warp_id * groups_per_row * 3;
    const size_t block_row_base = (size_t)warp_id * (groups_per_row / gpb);

    int chunks_per_row = groups_per_row >> 2;  // 4-group chunks
    float acc = 0.0f;
    for (int c = lane; c < chunks_per_row; c += 32) {
        int g0 = c * 4;
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(prow + (size_t)g0 * 3);
        uint32_t w0 = p32[0], w1 = p32[1], w2 = p32[2];

        uint32_t bits[4];
        bits[0] = w0 & 0xFFFFFFu;
        bits[1] = (w0 >> 24) | ((w1 & 0xFFFFu) << 8);
        bits[2] = (w1 >> 16) | ((w2 & 0xFFu) << 16);
        bits[3] = w2 >> 8;

        // 4-group chunks never straddle a block (gpb is 8 or 16)
        const float* cbp = codebook + (block_row_base + g0 / gpb) * 8;
        float c0 = cbp[0], c1 = cbp[1], c2 = cbp[2], c3 = cbp[3];
        float c4 = cbp[4], c5 = cbp[5], c6 = cbp[6], c7 = cbp[7];

        const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)g0 * 8);

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            float4 xa = x4[q * 2], xb = x4[q * 2 + 1];
            float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            uint32_t b = bits[q];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t idx = (b >> (j * 3)) & 0x7;
                float lo = (idx & 2) ? ((idx & 1) ? c3 : c2)
                                     : ((idx & 1) ? c1 : c0);
                float hi = (idx & 2) ? ((idx & 1) ? c7 : c6)
                                     : ((idx & 1) ? c5 : c4);
                acc += ((idx & 4) ? hi : lo) * xs[j];
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// uint32-load variant: 4 groups (12B) per lane iteration. Wins at small K.
__global__ void dequant_matvec_3bit_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int groups_per_row, int gpb  // gpb = block_size / 8
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    const uint8_t* prow = packed + (size_t)warp_id * groups_per_row * 3;
    const size_t block_row_base = (size_t)warp_id * (groups_per_row / gpb);

    // Each lane iteration takes 4 consecutive groups (12 bytes = 3 aligned
    // uint32 loads = 32 weights). 4-group chunks never straddle a codebook
    // block: gpb is 8 or 16, both multiples of 4. Requires K % 32 == 0.
    int chunks_per_row = groups_per_row >> 2;
    float acc = 0.0f;
    for (int c = lane; c < chunks_per_row; c += 32) {
        int g0 = c * 4;
        const uint32_t* p32 = reinterpret_cast<const uint32_t*>(prow + (size_t)g0 * 3);
        uint32_t w0 = p32[0], w1 = p32[1], w2 = p32[2];

        uint32_t bits[4];
        bits[0] = w0 & 0xFFFFFFu;
        bits[1] = (w0 >> 24) | ((w1 & 0xFFFFu) << 8);
        bits[2] = (w1 >> 16) | ((w2 & 0xFFu) << 16);
        bits[3] = w2 >> 8;

        const float* cbp = codebook + (block_row_base + g0 / gpb) * 8;
        float cb[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) cb[e] = cbp[e];

        const float* xg = x + (size_t)g0 * 8;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc += cb[(bits[q] >> (j * 3)) & 0x7] * xg[q * 8 + j];
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

torch::Tensor dequant_matvec_3bit(
    torch::Tensor x,        // (K,) float32
    torch::Tensor packed,    // flat uint8, 3-bit packed indices
    torch::Tensor codebook,  // (num_blocks, NE) float32
    int N, int K, int block_size, int variant  // 0 = auto, 1 = bytes, 2 = u32
) {
    int groups_per_row = K / 8;
    int gpb = block_size / 8;
    auto y = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;

    // variant: 0 auto (shuffle-LUT when K%256==0, else bytes),
    //          1 bytes, 2 u32, 3 shuffle-LUT, 4 shuffle-LUT 4 rows/warp,
    //          5 select-tree u32
    // 4-row measured equal to single-row (x re-reads were not a limiter);
    // kept selectable, not default.
    bool can_shfl = (K % 256 == 0) && (gpb == 8 || gpb == 16);
    int v = variant;
    if (v == 0) v = can_shfl ? 3 : 1;

    if (v == 5) {
        dequant_matvec_3bit_seltree_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            codebook.data_ptr<float>(), y.data_ptr<float>(),
            N, groups_per_row, gpb);
    } else if (v == 4) {
        int warps4 = N / 4;
        int blocks4 = (warps4 + warps_per_block - 1) / warps_per_block;
        if (gpb == 16) {
            dequant_matvec_3bit_shfl4_kernel<16><<<blocks4, threads, 0, stream>>>(
                x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
                codebook.data_ptr<float>(), y.data_ptr<float>(),
                N, groups_per_row);
        } else {
            dequant_matvec_3bit_shfl4_kernel<8><<<blocks4, threads, 0, stream>>>(
                x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
                codebook.data_ptr<float>(), y.data_ptr<float>(),
                N, groups_per_row);
        }
    } else if (v == 3) {
        if (gpb == 16) {
            dequant_matvec_3bit_shfl_kernel<16><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
                codebook.data_ptr<float>(), y.data_ptr<float>(),
                N, groups_per_row);
        } else {
            dequant_matvec_3bit_shfl_kernel<8><<<blocks, threads, 0, stream>>>(
                x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
                codebook.data_ptr<float>(), y.data_ptr<float>(),
                N, groups_per_row);
        }
    } else if (v == 2) {
        dequant_matvec_3bit_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            codebook.data_ptr<float>(), y.data_ptr<float>(),
            N, groups_per_row, gpb);
    } else {
        dequant_matvec_3bit_bytes_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), packed.data_ptr<uint8_t>(),
            codebook.data_ptr<float>(), y.data_ptr<float>(),
            N, groups_per_row, gpb);
    }
    return y;
}
"""

_CPP_SRC = r"""
torch::Tensor dequant_matvec_3bit(
    torch::Tensor x, torch::Tensor packed, torch::Tensor codebook,
    int N, int K, int block_size, int variant);
"""

_module = None

def get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="nativebit_cuda",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["dequant_matvec_3bit"],
            verbose=False,
            extra_cuda_cflags=[
                "-allow-unsupported-compiler",
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
            ],
            extra_cflags=["-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
        )
    return _module


def dequant_matvec_3bit_cuda(x, packed, codebook, N, K, block_size=BS,
                             variant=0):
    """Fused 3-bit dequant + matvec via custom CUDA kernel (warp per row).

    variant: 0 auto-dispatch by K, 1 byte-load kernel, 2 uint32-load kernel.
    """
    m = get_module()
    return m.dequant_matvec_3bit(x, packed.view(-1), codebook.view(-1),
                                 int(N), int(K), int(block_size), int(variant))
