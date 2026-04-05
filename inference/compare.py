"""Side-by-side text generation: float fp16 vs NativeBit 3-bit (2.2B).

Loads each model sequentially (8 GB VRAM can't hold both), generates from
the same prompts with identical sampling, prints a comparison.

Usage:
    python inference/compare.py
    python inference/compare.py --prompts "Once upon a time" "The president said"
    python inference/compare.py --max-new 200 --temperature 0.9 --top-k 50
"""
import argparse
import gc
import math
import os
import sys
import time

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from configs.tpu import TPU2BConfig
from inference.generate_torch import (
    PackedGPT, RMSNorm, load_packed_model, init_kv_cache,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Float model loader (JAX .npz -> PyTorch fp16)
# ---------------------------------------------------------------------------

def load_float_model(ckpt_path, config, device=DEVICE, dtype=DTYPE):
    """Load JAX float checkpoint into PackedGPT with nn.Linear layers."""
    print(f"  Loading float checkpoint: {ckpt_path}")
    t0 = time.time()

    ckpt = np.load(ckpt_path)

    model = PackedGPT(
        config.vocab_size, config.n_layers, config.n_embd, config.n_head,
        config.ffn_hidden, config.context_len,
        lambda out_f, in_f: nn.Linear(in_f, out_f, bias=False),
    )

    # Embedding
    model.embedding.weight.data = torch.from_numpy(
        ckpt["params/embedding"].copy()).to(dtype)

    # Transformer blocks
    for i in range(config.n_layers):
        block = model.blocks[i]

        # Attention QKV: JAX kernel is (in, out), PyTorch weight is (out, in)
        qkv_w = ckpt[f"params/block_{i}/CausalSelfAttention_0/Dense_0/kernel"]
        block.attn.qkv.weight.data = torch.from_numpy(qkv_w.T.copy()).to(dtype)

        # Attention output projection
        out_w = ckpt[f"params/block_{i}/CausalSelfAttention_0/Dense_1/kernel"]
        block.attn.out_proj.weight.data = torch.from_numpy(out_w.T.copy()).to(dtype)

        # RMSNorms
        block.ln1.weight.data = torch.from_numpy(
            ckpt[f"params/block_{i}/RMSNorm_0/weight"].copy()).to(dtype)
        block.ln2.weight.data = torch.from_numpy(
            ckpt[f"params/block_{i}/RMSNorm_1/weight"].copy()).to(dtype)

        # SwiGLU: gate, up, down
        gate_w = ckpt[f"params/block_{i}/SwiGLU_0/Dense_0/kernel"]
        block.ffn.gate.weight.data = torch.from_numpy(gate_w.T.copy()).to(dtype)

        up_w = ckpt[f"params/block_{i}/SwiGLU_0/Dense_1/kernel"]
        block.ffn.up.weight.data = torch.from_numpy(up_w.T.copy()).to(dtype)

        down_w = ckpt[f"params/block_{i}/SwiGLU_0/Dense_2/kernel"]
        block.ffn.down.weight.data = torch.from_numpy(down_w.T.copy()).to(dtype)

    # Final layer norm
    model.ln_f.weight.data = torch.from_numpy(
        ckpt["params/ln_f/weight"].copy()).to(dtype)

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    vram = sum(p.nbytes for p in model.parameters())
    print(f"  Loaded {n_params/1e9:.2f}B params in {time.time()-t0:.1f}s")
    print(f"  VRAM: {vram/1e9:.2f} GB ({dtype})")
    return model


# ---------------------------------------------------------------------------
# Generation (shared by both models)
# ---------------------------------------------------------------------------

def init_kv_cache_typed(model, batch_size=1, device=DEVICE, dtype=torch.float32):
    """KV cache with configurable dtype (fp16 for float model, fp32 for packed)."""
    head_dim = model.n_embd // model.n_head
    caches = []
    for _ in range(model.n_layers):
        k = torch.zeros(batch_size, model.n_head, model.context_len, head_dim,
                        device=device, dtype=dtype)
        v = torch.zeros_like(k)
        caches.append((k, v, 0))
    return caches


@torch.no_grad()
def generate(model, prompt_tokens, max_new=100, temperature=0.8, top_k=40,
             rep_penalty=1.3, seed=42, device=DEVICE, cache_dtype=torch.float32):
    """Autoregressive generation with KV cache and repetition penalty."""
    torch.manual_seed(seed)
    caches = init_kv_cache_typed(model, device=device, dtype=cache_dtype)

    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    logits, caches = model(x, kv_caches=caches)

    tokens = list(prompt_tokens)

    for i in range(max_new):
        logits_last = logits[0, -1].float()

        # Repetition penalty: reduce logits of tokens already generated
        if rep_penalty != 1.0:
            seen = set(tokens)
            for tok_id in seen:
                if logits_last[tok_id] > 0:
                    logits_last[tok_id] /= rep_penalty
                else:
                    logits_last[tok_id] *= rep_penalty

        if temperature > 0:
            logits_last = logits_last / temperature
            if top_k > 0:
                v, _ = torch.topk(logits_last, top_k)
                logits_last[logits_last < v[-1]] = float('-inf')
            probs = F.softmax(logits_last, dim=-1)
            token = int(torch.multinomial(probs, 1))
        else:
            token = int(logits_last.argmax())

        tokens.append(token)
        x1 = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, caches = model(x1, kv_caches=caches)

    return tokens


def free_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "The meaning of life is",
    "In a shocking finding, scientists discovered that",
    "The European Union announced today that",
    "Once upon a time, in a land far away,",
    "The key difference between machine learning and",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare float fp16 vs NativeBit 3-bit generation (2.2B)")
    parser.add_argument("--float-ckpt", default="logs/gcs/2b_float_s42_params.npz")
    parser.add_argument("--nb-ckpt", default="inference/2b_nb3.nbpack.npz")
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rep-penalty", type=float, default=1.3,
                        help="Repetition penalty (1.0=off, 1.3=default)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    config = TPU2BConfig()
    enc = tiktoken.get_encoding("gpt2")

    print(f"\n{'='*70}")
    print(f"  NativeBit 2.2B: Float fp16 vs NB 3-bit Comparison")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Prompts: {len(prompts)}, max_new={args.max_new}, "
          f"temp={args.temperature}, top_k={args.top_k}, "
          f"rep_penalty={args.rep_penalty}, seed={args.seed}")
    print(f"{'='*70}")

    # --- Phase 1: Float model ---
    print(f"\n{'_'*70}")
    print("  Phase 1: Loading float fp16 model...")
    print(f"{'_'*70}")
    float_model = load_float_model(args.float_ckpt, config)

    float_outputs = {}
    for prompt in prompts:
        prompt_tokens = enc.encode(prompt)
        t0 = time.time()
        tokens = generate(float_model, prompt_tokens,
                          max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, rep_penalty=args.rep_penalty,
                          seed=args.seed, cache_dtype=DTYPE)
        elapsed = time.time() - t0
        text = enc.decode(tokens)
        float_outputs[prompt] = (text, elapsed)
        print(f"  Generated {len(tokens)} tokens in {elapsed:.1f}s")

    free_model(float_model)
    print("  Float model freed.")

    # --- Phase 2: NB packed model ---
    print(f"\n{'_'*70}")
    print("  Phase 2: Loading NativeBit 3-bit packed model...")
    print(f"{'_'*70}")

    if not os.path.exists(args.nb_ckpt):
        print(f"\n  ERROR: Packed checkpoint not found: {args.nb_ckpt}")
        print(f"  Run: python inference/pack.py logs/gcs/2b_nb3_s42_params.npz "
              f"--out {args.nb_ckpt}")
        sys.exit(1)

    nb_model = load_packed_model(args.nb_ckpt, config, device=DEVICE)
    nb_model.eval()

    nb_outputs = {}
    for prompt in prompts:
        prompt_tokens = enc.encode(prompt)
        t0 = time.time()
        tokens = generate(nb_model, prompt_tokens,
                          max_new=args.max_new, temperature=args.temperature,
                          top_k=args.top_k, rep_penalty=args.rep_penalty,
                          seed=args.seed, cache_dtype=torch.float32)
        elapsed = time.time() - t0
        text = enc.decode(tokens)
        nb_outputs[prompt] = (text, elapsed)
        print(f"  Generated {len(tokens)} tokens in {elapsed:.1f}s")

    free_model(nb_model)
    print("  NB model freed.")

    # --- Comparison ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Float fp16 vs NativeBit 3-bit (2.2B)")
    print(f"{'='*70}")

    for prompt in prompts:
        float_text, float_time = float_outputs[prompt]
        nb_text, nb_time = nb_outputs[prompt]

        # Split into prompt + generated
        prompt_len = len(prompt)
        float_gen = float_text[prompt_len:]
        nb_gen = nb_text[prompt_len:]

        print(f"\n{'_'*70}")
        print(f"  PROMPT: {prompt}")
        print(f"{'_'*70}")
        print(f"\n  [Float fp16] ({float_time:.1f}s)")
        print(f"  {float_gen.strip()}")
        print(f"\n  [NB 3-bit]   ({nb_time:.1f}s)")
        print(f"  {nb_gen.strip()}")

    # Summary
    float_total = sum(t for _, t in float_outputs.values())
    nb_total = sum(t for _, t in nb_outputs.values())
    print(f"\n{'='*70}")
    print(f"  TIMING SUMMARY")
    print(f"  Float fp16 total: {float_total:.1f}s")
    print(f"  NB 3-bit total:   {nb_total:.1f}s")
    print(f"  Speedup:          {float_total/nb_total:.2f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
