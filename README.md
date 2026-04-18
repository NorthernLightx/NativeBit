# NativeBit

Quantization-aware training for LLMs with per-block learned codebooks. At 2.2B parameters and 3-bit precision, a fine-tuned NativeBit model matches its float counterpart on WikiText-103 (30.50 vs 30.51 perplexity) while beating post-hoc RTN quantization by 2.7%.

## Result

WikiText-103 perplexity, 2.2B model (26 layers, 2560 hidden, 6912 FFN — same shape as BitNet b1.58-2B-4T).

| Method | 3-bit PPL | vs Float |
|--------|-----------|----------|
| Float baseline | 30.51 | 0% |
| Post-hoc RTN | 31.33 | +2.7% |
| NativeBit from-scratch (20K steps) | 34.23 | +12.2% |
| **NativeBit via QAT (5K steps)** | **30.50** | **−0.03%** |

The QAT recipe — load a trained float checkpoint, then fine-tune with NativeBit active for 5K steps using a commitment loss and canonical VQ-VAE EMA — is what closes the gap. Training NativeBit from scratch at this scale leaves a sizeable gap to float that post-hoc RTN doesn't have.

Compression: packed NativeBit 2.2B is 1.70 GB vs 8.76 GB float — 5.1× smaller on disk.

## What NativeBit does

Weights are quantized to one of `K` learned values per block (default `K=8`, `block_size=128`, i.e. 3-bit). The codebook entries are trainable and updated via EMA of raw sums and counts (canonical VQ-VAE style). Forward pass uses the quantized values; gradients flow to latent floats via straight-through estimation.

Three ingredients make the training converge to float quality:

1. **Commitment loss** `λ·E[‖w − Q(w).sg‖²]` pulls latent weights onto codebook entries, reducing STE bias so `forward ≈ backward` near the fixed point.
2. **Canonical EMA** — `N ← α·N + (1−α)·count_batch`, `s ← α·s + (1−α)·sum_batch`, `e = s/N` — tracks raw per-entry statistics instead of EMA-ing batch means. Gives proper count-weighting for noisy low-population clusters.
3. **QAT from trained float** instead of from-scratch. Decouples "learn the task" from "adapt to quantization constraint."

Embeddings, LM head, and RMSNorm parameters stay in float. Only the dense matmuls in attention and SwiGLU are quantized.

## Code layout

```
nativebit_jax/         JAX/Flax backend (TPU training at 2.2B scale)
  layers.py            NativeBitDense + compute_quant_reg + canonical EMA
  model.py             LLaMA-style transformer (RoPE, RMSNorm, SwiGLU)
  train.py             Training loop with QAT init, commitment loss,
                       periodic validation, full-config JSONL logging
  codebook_utils.py    Codebook init + EMA helpers
nativebit/             PyTorch backend (local GPU development)
inference/             Packed inference (Triton + CUDA dequant kernels)
configs/tpu.py         Model configs (25M → 2.2B)
benchmarks/            Post-hoc quantization baselines for comparison
tests/                 Unit tests (attention, quant-reg, canonical EMA, QAT init)
infra/                 TPU provisioning + training launch scripts
```

The two key files are `nativebit_jax/layers.py` (NativeBitDense + `requantize_params` + `compute_quant_reg` + `compute_quant_diagnostics`) and `nativebit_jax/train.py` (the training loop). The JAX backend is the primary one — it's what produced the paper results on v6e-8 TPU. The PyTorch backend is maintained for local GPU iteration.

## Reproducing

### Float baseline (2.2B, ~7h on v6e-8)

```bash
python -m nativebit_jax.train \
    --config tpu-2b --dataset openwebtext \
    --no-nativebit --name 2b_float
```

### NativeBit from-scratch (2.2B, ~7h on v6e-8)

```bash
python -m nativebit_jax.train \
    --config tpu-2b --dataset openwebtext --name 2b_nb
```

Lands at +12.2% above float on WikiText-103 cross-eval at 3-bit — not competitive with post-hoc RTN. Used here as an ablation; for real quality use QAT.

### NativeBit via QAT (2.2B, ~2h on v6e-8)

```bash
python -m nativebit_jax.train \
    --config tpu-2b --dataset openwebtext --name 2b_nb_qat \
    --max-steps 5000 \
    --init-from logs/2b_float_params.npz \
    --lr 1e-4 --weight-decay 0.01 \
    --warmup-steps 200 --delay-quant-steps 0 --ema-decay 0.99 \
    --quant-reg-weight 1.0 --use-canonical-ema \
    --val-every 500
```

Matches float PPL on WikiText-103.

### Post-hoc baseline comparison

```bash
python benchmarks/benchmark_posthoc_2b.py --ckpt logs/2b_float_params.npz
```

## Training logs

Each run writes a JSONL log with:

- **Header** (`schema_version=2`): full config dump, `git_hash`, `argv`, `init_from`.
- **`init_eval`**: validation PPL at step 0, before any training.
- **Per-step** (every `log_every`): loss, perplexity, `quant_err_rms`, `cb_utilization`, `dead_frac`, optionally `quant_reg` and `lambda`.
- **Periodic validation** (every `val_every`): `val_loss`, `val_ppl`.
- **Final `eval`**: held-out test PPL on the training dataset + cross-eval on WikiText-103.

`compute_quant_diagnostics(params)` in `layers.py` is a pure function you can call from anywhere to get `{quant_error_rms, codebook_utilization, dead_entries_frac}` — useful for experiment monitoring.

## Inference

Training checkpoints pack into a minimal format: per-block uint8 indices + fp32 codebook tables + unquantized embeddings/norms. The 2.2B model is 1.70 GB on disk (5.1× float).

```bash
# Pack a trained NativeBit checkpoint
python inference/pack.py logs/2b_nb_qat_params.npz inference/2b_nb.nbpack.npz

# Generate (JAX / TPU / CPU)
python inference/generate.py inference/2b_nb.nbpack.npz --packed --benchmark

# Generate (PyTorch + Triton/CUDA dequant kernels on GPU)
python inference/generate_torch.py inference/2b_nb.nbpack.npz --benchmark
```

The packed `generate_torch.py` path uses a fused dequant-matvec kernel that reads uint8 codebook indices directly from VRAM, avoiding the materialized-weight-matrix bottleneck of single-token decode.

## Architecture notes

A few design choices worth calling out:

- **Cross-position (not cross-head) attention.** The JAX `jax.nn.dot_product_attention(q, k, v, is_causal=True)` call with input layout `(B, H, T, D)` computes attention across heads, not positions. The current code uses an explicit `einsum` with an fp32 softmax for stability under NativeBit quantization. If you swap attention, add a probe: `KL(predict(full_context), predict(single_token)) > 0.1` confirms the model actually uses context.
- **STE through cached delta.** Forward computes `w + stop_gradient(Q(w_cached) − w_cached)`. Gradients flow cleanly to the latent `w`. The cache is refreshed every `requantize_every` steps (default 10). Commitment loss keeps the "stale cache drift" small by pulling `w` onto codebook entries.
- **Canonical VQ-VAE EMA vs EMA-of-means.** We EMA the raw per-entry sums `s` and counts `N` (then `e = s/N`) rather than EMA-ing batch means. This correctly count-weights updates when clusters have uneven population — important for low-bit regimes where some codebook entries get fewer samples per batch.
- **Embeddings, LM head, and norms stay float.** Only `NativeBitDense` layers (the dense matmuls in attention and SwiGLU) are quantized. Quantizing the tied embedding / LM head would require a different treatment.

## Caveats

- **Validated at 2.2B only.** Smaller-scale results (125M, 350M) need re-validation with the fixed attention kernel — they were produced with an earlier JAX implementation that had a cross-head/cross-position attention bug.
- **QAT is the recommended recipe.** NativeBit from-scratch at 2.2B lands +12.2% above float at 3-bit — worse than post-hoc RTN. The NB story is "trainable quantization that matches float via short fine-tuning," not "matches float from random init."
- **2-bit regime untested with fixed attention.** The earlier 2-bit wins over k-means post-hoc came from pre-fix runs.
- **Single dataset.** Trained and evaluated on OpenWebText + cross-evaluated on WikiText-103. Other domains (code, multilingual) unknown.

## License

MIT
