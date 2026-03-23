# NativeBit

**Train neural networks in their quantized representation from birth.**

Instead of the standard train-in-float-then-compress pipeline, NativeBit co-discovers optimal quantization values during training via per-block learned codebooks. The model is *native* to its bit representation from the first gradient step.

## Key Insight

3-bit NativeBit matches or beats float baselines. Quantization acts as regularization -- the discrete codebook constraint prevents overfitting, similar to how dropout or weight decay regularize continuous weights.

## How It Works

1. **Per-block learned codebooks** -- every 64 weights share a codebook of 8 entries (3-bit). Codebook values are learned jointly with weights via gradient descent.

2. **Gradient-scaled STE** -- the straight-through estimator passes gradients through quantization. We scale gradients by proximity to the codebook entry: weights close to their entry get full gradient, distant weights get reduced gradient to prevent oscillation.

3. **Stochastic rounding** -- during training, each weight occasionally snaps to its 2nd-nearest codebook entry (with probability proportional to proximity). This adds exploration and prevents codebook assignments from getting stuck.

4. **Split-based revival** -- dead codebook entries are revived by splitting the most-used entry symmetrically, preserving the weight distribution.

5. **Value embeddings** -- float-precision token embeddings are injected into attention values on alternating layers, providing a clean bypass for quantization-degraded attention. Per-head gating learns how much to rely on this bypass.

6. **Logit soft-capping** -- `logits = 30 * tanh(logits / 30)` prevents extreme logits and stabilizes late training.

## Architecture

LLaMA-style transformer: RoPE, RMSNorm, SwiGLU, no bias, weight-tied embeddings.

- Embeddings, LM head, and layer norms stay in full precision
- All other linear layers use `NativeBitLinear` (3-bit codebook quantization)
- Value embeddings shared with token embeddings (weight-tied)

## Quick Start

```bash
# Install
pip install -e .

# Train NativeBit model (3-bit, WikiText-2)
python train.py --name nativebit --max-steps 5000

# Train float baseline for comparison
python train.py --name float --no-nativebit --max-steps 5000

# Export to packed 3-bit format
python export.py --checkpoint logs/nativebit_final.pt --output model.nbpack
```

## Project Structure

```
nativebit/
  layers.py          NativeBitLinear (the core quantized layer)
  model.py           LLaMA-style GPT with value embeddings
  codebook_utils.py  Codebook initialization and revival
  data.py            WikiText-2 data loading + BPB metric
  pack.py            3-bit packing for inference
  generate.py        Text generation from checkpoints
  logging.py         Training metrics (JSONL)
  seed.py            Reproducibility

configs/
  default.py         Default training configuration

train.py             Main training entry point
export.py            Pack models to 3-bit format
tests/               Automated test suite
```

## Design Principles

- **No post-hoc compression.** The model trains in its quantized format from step 1.
- **Per-block codebooks.** Each block of 64 weights learns its own 8-entry codebook. This is more flexible than global quantization (different layers need different value ranges).
- **Gradient-based codebook learning.** Codebook entries are `nn.Parameter`s that receive gradients and are updated by the optimizer, just like weights.
- **Everything stays simple.** Pure PyTorch, no custom CUDA kernels, no external quantization libraries.

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- tiktoken
- Single GPU (tested on RTX 3070 8GB)

## License

MIT
