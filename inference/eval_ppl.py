"""Evaluate perplexity of float and NB 2.2B models on WikiText-103 test set.

Runs both models sequentially on RTX 3070 (8 GB VRAM), producing a
comparison table with PPL, model size, and token-level agreement.

Usage:
    python inference/eval_ppl.py
    python inference/eval_ppl.py --max-batches 50   # quick sanity check
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from configs.tpu import TPU2BConfig
from inference.generate_torch import load_packed_model
from inference.compare import load_float_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_tokens(data_dir="data"):
    """Load WikiText-103 test tokens using the JAX tokenization path.

    Must match the tokenization used during training (nativebit_jax/train.py).
    """
    import array
    import numpy as np
    cache_path = os.path.join(data_dir, "wikitext103", "test.tokens.bin")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found. Run the JAX tokenization first:\n"
            "  python -c \"from nativebit_jax.train import load_tokens; "
            "load_tokens('wikitext-103')\"")
    with open(cache_path, "rb") as f:
        a = array.array("i")
        a.frombytes(f.read())
    tokens = np.frombuffer(a, dtype=np.int32).copy()
    return torch.from_numpy(tokens).long()


def make_batches(tokens, context_len=1024):
    """Yield (x, y) pairs of shape (1, context_len) from token array."""
    n = len(tokens) - 1
    for i in range(0, n - context_len + 1, context_len):
        x = tokens[i:i + context_len].unsqueeze(0)
        y = tokens[i + 1:i + context_len + 1].unsqueeze(0)
        yield x, y


@torch.no_grad()
def compute_ppl(model, tokens, context_len=1024, max_batches=0, device=DEVICE,
                label=""):
    """Compute perplexity on token sequence. Returns (ppl, loss, n_batches)."""
    model.eval()  # noop for already-eval models
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    t0 = time.time()

    for x, y in make_batches(tokens, context_len):
        if max_batches > 0 and n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        n_batches += 1

        if n_batches % 200 == 0:
            running_ppl = math.exp(min(total_loss / total_tokens, 20))
            elapsed = time.time() - t0
            print(f"  [{label}] batch {n_batches}: "
                  f"PPL={running_ppl:.2f}, "
                  f"{elapsed:.0f}s", flush=True)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    elapsed = time.time() - t0
    print(f"  [{label}] Final: PPL={ppl:.2f} "
          f"({n_batches} batches, {total_tokens} tokens, {elapsed:.0f}s)")
    return ppl, avg_loss, n_batches


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--float-ckpt", default="logs/gcs/2b_float_s42_params.npz")
    parser.add_argument("--nb-ckpt", default="inference/2b_nb3.nbpack.npz")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit batches (0=full test set)")
    args = parser.parse_args()

    config = TPU2BConfig()

    print(f"\n{'='*70}")
    print(f"  NativeBit 2.2B PPL Comparison (WikiText-103 test)")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    # Load test tokens
    print("Loading WikiText-103 test set...")
    test_tokens = load_test_tokens()
    n_sequences = (len(test_tokens) - 1) // config.context_len
    print(f"  {len(test_tokens)} tokens, {n_sequences} sequences "
          f"(context={config.context_len})\n")

    # --- Float model ---
    print("Phase 1: Float fp16")
    print("-" * 40)
    float_model = load_float_model(args.float_ckpt, config)
    float_ppl, float_loss, float_n = compute_ppl(
        float_model, test_tokens, config.context_len,
        max_batches=args.max_batches, label="Float fp16")
    float_size = sum(p.nbytes for p in float_model.parameters()) / 1e9
    del float_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  Float model freed. VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB\n")

    # --- NB packed model ---
    print("Phase 2: NB 3-bit packed")
    print("-" * 40)
    # Load on CPU, materialize weights, convert to fp16, then move to GPU.
    nb_model = load_packed_model(args.nb_ckpt, config, device="cpu")
    nb_model.eval()

    print("  Materializing packed weights on CPU...")
    from inference.triton_kernel import PackedLinear, BS
    for name, module in list(nb_model.named_modules()):
        if isinstance(module, PackedLinear):
            # Move packed data to CPU if not already
            packed_idx = module.packed_indices.cpu()
            codebook = module.codebook.cpu()
            n_groups = packed_idx.shape[0] // 3
            packed = packed_idx.reshape(n_groups, 3).to(torch.int32)
            bits24 = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            indices = torch.zeros(n_groups, 8, dtype=torch.int64)
            for j in range(8):
                indices[:, j] = (bits24 >> (j * 3)) & 0x7
            num_blocks = codebook.shape[0]
            total_idx = num_blocks * BS
            indices = indices.reshape(-1)[:total_idx].reshape(num_blocks, BS)
            block_idx = torch.arange(num_blocks).unsqueeze(1)
            total = module.out_features * module.in_features
            w = codebook[block_idx, indices].reshape(-1)[:total]
            w = w.reshape(module.out_features, module.in_features).to(torch.bfloat16)
            linear = torch.nn.Linear(module.in_features, module.out_features, bias=False)
            linear.weight.data = w
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(nb_model.named_modules())[parts[0]]
                setattr(parent, parts[1], linear)
    nb_model.to(torch.bfloat16)
    nb_model.to(DEVICE)
    vram_mb = torch.cuda.memory_allocated() / 1e6
    print(f"  Done. VRAM: {vram_mb:.0f} MB")
    nb_ppl, nb_loss, nb_n = compute_ppl(
        nb_model, test_tokens, config.context_len,
        max_batches=args.max_batches, label="NB 3-bit")
    nb_size_params = sum(p.nbytes for p in nb_model.parameters())
    nb_size_bufs = sum(b.nbytes for b in nb_model.buffers())
    nb_size = (nb_size_params + nb_size_bufs) / 1e9
    free_model(nb_model)
    print()

    # --- Results table ---
    gap = (nb_ppl - float_ppl) / float_ppl * 100
    compression = float_size / nb_size

    print(f"{'='*70}")
    print(f"  RESULTS: WikiText-103 Test Perplexity")
    print(f"{'='*70}")
    print(f"  {'Model':<25s}  {'PPL':>8s}  {'Loss':>8s}  {'Size':>8s}  {'Notes'}")
    print(f"  {'-'*65}")
    print(f"  {'Float fp16':<25s}  {float_ppl:>8.2f}  {float_loss:>8.4f}  "
          f"{float_size:>6.2f}GB  baseline")
    print(f"  {'NativeBit 3-bit':<25s}  {nb_ppl:>8.2f}  {nb_loss:>8.4f}  "
          f"{nb_size:>6.2f}GB  {gap:+.2f}%")
    print(f"  {'-'*65}")
    print(f"  Compression: {compression:.1f}x")
    print(f"  PPL gap: {gap:+.2f}%")
    print(f"{'='*70}\n")

    # Save results
    results = {
        "dataset": "wikitext-103",
        "split": "test",
        "context_len": config.context_len,
        "float": {
            "ppl": round(float_ppl, 2),
            "loss": round(float_loss, 4),
            "size_gb": round(float_size, 2),
            "n_batches": float_n,
        },
        "nb_3bit": {
            "ppl": round(nb_ppl, 2),
            "loss": round(nb_loss, 4),
            "size_gb": round(nb_size, 2),
            "n_batches": nb_n,
        },
        "ppl_gap_pct": round(gap, 2),
        "compression": round(compression, 1),
    }
    out_path = "logs/bench/ppl_comparison_2b.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
