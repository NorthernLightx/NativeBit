"""CUDA fused 3-bit dequant-matvec kernel for NativeBit packed inference.

Reads 3-bit packed indices from VRAM, unpacks via bit shifts, looks up
codebook values from registers (not compare+select), and accumulates
the dot product. One thread per output row.

The codebook (8 floats = 32 bytes) fits entirely in registers.
cb[idx] compiles to a single indexed register access — no branching.
"""
import os
import torch

# Ensure ninja + cl.exe on PATH for JIT compilation
try:
    import ninja
    os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

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

__global__ void dequant_matvec_3bit_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float* __restrict__ codebook,
    float* __restrict__ y,
    int N, int bpr
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const int BS = 128;
    const int PPB = 48;  // BS * 3 / 8

    float acc = 0.0f;

    for (int kb = 0; kb < bpr; kb++) {
        int block_id = row * bpr + kb;

        // Load codebook into registers (8 floats = 32 bytes)
        float cb[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            cb[e] = codebook[block_id * 8 + e];
        }

        // Process 16 groups of 3 packed bytes = 8 indices each
        const uint8_t* pb = packed + block_id * PPB;
        int x_base = kb * BS;

        #pragma unroll
        for (int g = 0; g < 16; g++) {
            uint32_t bits24 = pb[g*3]
                            | (uint32_t(pb[g*3+1]) << 8)
                            | (uint32_t(pb[g*3+2]) << 16);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int idx = (bits24 >> (j * 3)) & 0x7;
                acc += cb[idx] * x[x_base + g * 8 + j];
            }
        }
    }

    y[row] = acc;
}

torch::Tensor dequant_matvec_3bit(
    torch::Tensor x,        // (K,) float32
    torch::Tensor packed,    // flat uint8, 3-bit packed indices
    torch::Tensor codebook,  // (num_blocks, 8) float32
    int N, int K
) {
    int bpr = K / 128;
    auto y = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    dequant_matvec_3bit_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        y.data_ptr<float>(),
        N, bpr
    );
    return y;
}
"""

_CPP_SRC = r"""
torch::Tensor dequant_matvec_3bit(
    torch::Tensor x, torch::Tensor packed, torch::Tensor codebook,
    int N, int K);
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


def dequant_matvec_3bit_cuda(x, packed, codebook, N, K):
    """Fused 3-bit dequant + matvec via custom CUDA kernel."""
    m = get_module()
    return m.dequant_matvec_3bit(x, packed.view(-1), codebook.view(-1), int(N), int(K))
