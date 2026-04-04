# NativeBit

What if neural networks were never float? NativeBit trains models in their quantized representation from the start. Instead of the usual train-then-compress pipeline, each weight block learns its own codebook during training via straight-through estimation + EMA updates.

The result: models trained at 2-4 bit precision that match float quality, with no training overhead.

## Results

Perplexity on WikiText-103 (lower is better):

| Scale | Float | NativeBit 3-bit | Gap |
|-------|-------|----------------|-----|
| 22M | 101.21 | 96.46 | -4.7% (NB is better) |
| 125M | 185.74 | 185.95 | +0.11% |
| 350M | 192.06 | 193.68 | +0.84% |
| **2.2B** | **180.64** | **178.98** | **-0.92% (NB is better)** |

At 2.2B (matching BitNet b1.58 scale: 26 layers, 2560 hidden, 6912 FFN), NB 3-bit beats float. The packed model is 1.70 GB vs 8.76 GB float (5.1x compression).

The bit width is just a config parameter (codebook size K). At 125M on WikiText-103:

| Bits | K | PPL | vs Float |
|------|---|-----|----------|
| 2 | 4 | 193.71 | +4.3% |
| ~2.6 | 6 | 192.88 | +3.8% |
| 3 | 8 | 185.95 | +0.11% |
| 4 | 16 | 192.65 | +3.7% |

3-bit is the sweet spot where quality matches float. 2-bit still holds much better than post-hoc methods at the same width.

Compared to post-hoc methods (applied to the same trained float model):

| Method | 125M PPL | vs Float |
|--------|----------|----------|
| NativeBit 3-bit | 185.95 | +0.11% |
| RTN 3-bit | 186.35 | +0.3% |
| K-means 8-entry | 188.36 | +1.4% |
| NativeBit 2-bit | 193.71 | +4.3% |
| K-means 4-entry | 211.44 | +13.8% |

The advantage grows at lower bit widths. At 2-bit, post-hoc k-means degrades by 13.8% while NativeBit holds at 4.3%.

## How it works

The core idea is simple: quantize weights every forward pass, but let gradients flow through via STE. The codebook entries (the allowed weight values) update via EMA rather than gradients, which eliminates the codebook learning rate and works better in practice (+4.6% over gradient-based).

The expensive part is the quantization itself (distance computation + argmin over all weights). We cache the quantization assignments and reuse them for N steps. A factorial experiment across learning rates, bit widths, and model scales showed that N can go up to 500 with no quality impact, which makes the training overhead effectively zero.

Training speed on TPU v6e-8:

| Requantize interval | Overhead |
|--------------------|----------|
| Every step | 10x |
| Every 10 steps | 1.27x |
| Every 200 steps | 1.0x |

## Project structure

Two backends: PyTorch for local GPU development, JAX/Flax for TPU training at scale.

```
nativebit/           PyTorch backend
nativebit_jax/       JAX/Flax backend (TPU)
inference/           Packed inference (Triton + CUDA kernels)
configs/             Model configs (48M to 2.2B)
autoresearch/        Autonomous hyperparameter search
benchmarks/          Post-hoc quantization comparisons
experiments/         Ablations, sweeps, and run scripts
```

`nativebit/layers.py` and `nativebit_jax/layers.py` are the core files. NativeBitLinear/NativeBitDense are drop-in replacements for nn.Linear/nn.Dense. `inference/triton_kernel.py` and `inference/cuda_kernel.py` provide fused dequant-matvec kernels for deployment.

## Quick start

PyTorch (any GPU):
```bash
pip install torch tiktoken matplotlib tqdm
python train.py --name test --max-steps 50          # quick test
python train.py --name float_baseline --no-nativebit # float baseline
python train.py --name nb_3bit                       # NativeBit 3-bit
```

JAX (Cloud TPU):
```bash
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tiktoken tqdm datasets
python -m nativebit_jax.train --config tpu-medium --name nb_125m
```

## Dashboard

There's a live training dashboard that plots loss, perplexity, codebook utilization, and gradient stats in real time. Works with local logs or can sync from a TPU VM.

```bash
python analysis/dashboard.py --log-dir logs          # local logs
python analysis/dashboard.py --tpu-name my-tpu \     # sync from TPU
    --tpu-zone europe-west4-a
```

## Autoresearch

There's an autonomous experiment runner that searches hyperparameters by analyzing training logs for problems (dead codebook entries, gradient issues, plateaus) and picking the next config accordingly.

```bash
python autoresearch/autoresearch_run.py --resume --max-hours 4
python autoresearch/autoresearch_run.py --report
```

## Design choices

- Per-block codebooks (block_size=128), not per-layer or global
- EMA codebook updates, not gradient-based
- Embeddings, LM head, and norms stay in float
- Percentile-based codebook initialization
- LLaMA-style architecture: RoPE, RMSNorm, SwiGLU, weight tying
- Self-contained, no HuggingFace/GPTQ/AWQ dependencies

## Inference

Training checkpoints pack into a minimal format: 3-bit codebook indices + codebook tables + unquantized params. The 2.2B model goes from 8.76 GB to 1.70 GB on disk.

For single-token decode (the bottleneck in generation), we have custom kernels that read packed indices directly from VRAM instead of materialized weight matrices:

```bash
# JAX (TPU/CPU) — static KV cache, no recompilation per token
python inference/generate.py inference/2b_nb3.nbpack.npz --packed --benchmark

# PyTorch + Triton/CUDA kernels (GPU)
python inference/generate_torch.py inference/2b_nb3.nbpack.npz --benchmark
```

RTX 3070 (2.2B model, same unoptimized model code for both):

| Method | VRAM | Decode tok/s |
|--------|------|-------------|
| Float fp16 (native) | 4.38 GB | 12.2 |
| NB 3-bit (CUDA kernel) | 1.80 GB | 8.1 |

NB trades ~1.5x speed for 2.4x VRAM savings. On 4 GB GPUs, fp16 doesn't fit but NB does. Per-layer, the Triton uint8 kernel matches cuBLAS fp16 (0.113 ms vs 0.115 ms); the end-to-end gap is from the CUDA 3-bit kernel doing fp32 compute while fp16 uses half-precision throughout.

## What's next

- Downstream benchmarks (HellaSwag, PIQA, ARC) instead of just perplexity
- Warp-level CUDA kernel (32 threads per row with coalesced reads) to close the per-layer 1.37x gap
- Training on larger datasets (The Pile, RedPajama) where the regularization effect might behave differently
- Combining weight quantization with activation quantization
- Progressive bit-width schedules where the model starts at higher precision and compresses during training

## License

MIT
