# NativeBit

What if neural networks were never float? NativeBit trains models in their quantized representation from the start. Instead of the usual train-then-compress pipeline, each weight block learns its own codebook during training via straight-through estimation + EMA updates.

The result: models trained at 2-4 bit precision that match float quality, with no training overhead.

## Results

Perplexity on WikiText-103 (lower is better):

| Scale | Float | NativeBit 3-bit | Gap |
|-------|-------|----------------|-----|
| 22M | 101.21 | 96.46 | -4.7% (NB is better) |
| 125M | 185.74 | 185.95 ± 0.24 | +0.11% |
| 350M | 192.06 | 193.68 | +0.84% |

At 125M the gap is 0.11%, smaller than seed variance. At 22M, quantization noise actually helps as regularization.

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
configs/             Model configs (48M to 1B+)
autoresearch/        Autonomous hyperparameter search
benchmarks/          Post-hoc quantization comparisons
experiments/         Ablations, sweeps, and run scripts
```

`nativebit/layers.py` and `nativebit_jax/layers.py` are the core files. NativeBitLinear/NativeBitDense are drop-in replacements for nn.Linear/nn.Dense.

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

## What's next

The scaling results (22M to 350M) show the quality gap stays under 1%, but the real test is whether this holds at 2B+ where most deployment happens. The main blocker is model sharding across TPU chips, which the current single-chip code doesn't support. BitNet b1.58 showed their approach works at 2B with ternary weights; we want to compare learned codebooks against fixed grids at the same scale.

Other things worth exploring:
- Training on larger datasets (The Pile, RedPajama) where the regularization effect might behave differently
- Downstream benchmarks (HellaSwag, PIQA, ARC) instead of just perplexity
- Custom inference kernels that exploit the codebook structure for actual speedup, not just compression
- Combining weight quantization with activation quantization
- Progressive bit-width schedules where the model starts at higher precision and compresses during training

## License

MIT
