"""Downstream task evaluation: LAMBADA and HellaSwag.

Evaluates float fp16 and NB 3-bit 2.2B models on standard benchmarks.
No training — forward passes only.

Usage:
    python inference/eval_downstream.py
    python inference/eval_downstream.py --tasks lambada hellaswag
    python inference/eval_downstream.py --float-ckpt logs/gcs/2b_float_owt100k_params.npz
    python inference/eval_downstream.py --max-examples 200   # quick test
"""
import argparse
import gc
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from configs.tpu import TPU2BConfig
from inference.generate_torch import load_packed_model
from inference.compare import load_float_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "eval")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def download_if_needed(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved: {path}")


def load_lambada(max_examples=0):
    """Load LAMBADA test set. Each example: passage where last word is target."""
    path = os.path.join(DATA_DIR, "lambada_test.jsonl")
    download_if_needed(
        "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl",
        path)

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line.strip())["text"]
            examples.append(text)
    if max_examples > 0:
        examples = examples[:max_examples]
    print(f"  LAMBADA: {len(examples)} examples")
    return examples


def load_hellaswag(max_examples=0):
    """Load HellaSwag validation set. 4-way multiple choice."""
    path = os.path.join(DATA_DIR, "hellaswag_val.jsonl")
    download_if_needed(
        "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
        path)

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            examples.append({
                "context": obj["ctx"],
                "endings": obj["endings"],
                "label": int(obj["label"]),
            })
    if max_examples > 0:
        examples = examples[:max_examples]
    print(f"  HellaSwag: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_lambada(model, examples, enc, context_len=1024, device=DEVICE, label=""):
    """LAMBADA: predict the last word of each passage. Report accuracy."""
    correct = 0
    total = 0
    t0 = time.time()

    for i, text in enumerate(examples):
        # Last word = everything after the last space
        last_space = text.rfind(" ")
        if last_space < 0:
            continue

        context = text[:last_space]
        target_word = text[last_space:]  # includes leading space

        ctx_tokens = enc.encode(context)
        target_tokens = enc.encode(target_word)

        if len(ctx_tokens) + len(target_tokens) > context_len:
            # Truncate context from the left to fit
            max_ctx = context_len - len(target_tokens)
            ctx_tokens = ctx_tokens[-max_ctx:]

        # Feed context, check if model greedily predicts target tokens
        all_tokens = ctx_tokens + target_tokens
        x = torch.tensor([all_tokens], dtype=torch.long, device=device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Check each target position
        match = True
        for j, target_tok in enumerate(target_tokens):
            pos = len(ctx_tokens) - 1 + j  # position whose logits predict next token
            pred = logits[0, pos].argmax().item()
            if pred != target_tok:
                match = False
                break

        if match:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            acc = correct / total * 100
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(examples)}: "
                  f"acc={acc:.1f}%, {elapsed:.0f}s", flush=True)

    acc = correct / max(total, 1) * 100
    elapsed = time.time() - t0
    print(f"  [{label}] LAMBADA: {acc:.2f}% ({correct}/{total}), {elapsed:.0f}s")
    return {"accuracy": round(acc, 2), "correct": correct, "total": total,
            "time_s": round(elapsed, 1)}


@torch.no_grad()
def run_hellaswag(model, examples, enc, context_len=1024, device=DEVICE, label=""):
    """HellaSwag: pick the most likely ending from 4 choices. Report accuracy."""
    correct = 0
    total = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        ctx_tokens = enc.encode(ex["context"])

        best_score = float("-inf")
        best_idx = -1

        for j, ending in enumerate(ex["endings"]):
            end_tokens = enc.encode(" " + ending)

            # Truncate context from left if needed
            max_ctx = context_len - len(end_tokens) - 1
            if max_ctx < 1:
                continue
            ctx = ctx_tokens[-max_ctx:] if len(ctx_tokens) > max_ctx else ctx_tokens

            all_tokens = ctx + end_tokens
            x = torch.tensor([all_tokens], dtype=torch.long, device=device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Average log-prob of ending tokens
            log_probs = F.log_softmax(logits[0].float(), dim=-1)
            score = 0.0
            for k, tok in enumerate(end_tokens):
                pos = len(ctx) - 1 + k
                score += log_probs[pos, tok].item()
            score /= len(end_tokens)  # length-normalize

            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx == ex["label"]:
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            acc = correct / total * 100
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(examples)}: "
                  f"acc={acc:.1f}%, {elapsed:.0f}s", flush=True)

    acc = correct / max(total, 1) * 100
    elapsed = time.time() - t0
    print(f"  [{label}] HellaSwag: {acc:.2f}% ({correct}/{total}), {elapsed:.0f}s")
    return {"accuracy": round(acc, 2), "correct": correct, "total": total,
            "time_s": round(elapsed, 1)}


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Downstream evaluation: LAMBADA & HellaSwag")
    parser.add_argument("--float-ckpt", default="logs/gcs/2b_float_owt100k_params.npz")
    parser.add_argument("--nb-ckpt", default="inference/2b_nb3_owt100k.nbpack.npz")
    parser.add_argument("--tasks", nargs="+", default=["lambada", "hellaswag"],
                        choices=["lambada", "hellaswag"])
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit examples per task (0=all)")
    parser.add_argument("--output", type=str, default="logs/bench/downstream_2b.json")
    args = parser.parse_args()

    config = TPU2BConfig()
    enc = tiktoken.get_encoding("gpt2")

    print(f"\n{'='*70}")
    print(f"  NativeBit 2.2B Downstream Evaluation")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Tasks: {', '.join(args.tasks)}")
    print(f"{'='*70}\n")

    # Load datasets
    print("Loading datasets...")
    datasets = {}
    if "lambada" in args.tasks:
        datasets["lambada"] = load_lambada(args.max_examples)
    if "hellaswag" in args.tasks:
        datasets["hellaswag"] = load_hellaswag(args.max_examples)
    print()

    results = {}

    # --- Float model ---
    print(f"{'_'*70}")
    print(f"  Phase 1: Float fp16")
    print(f"{'_'*70}")
    float_model = load_float_model(args.float_ckpt, config)

    results["float"] = {}
    if "lambada" in datasets:
        results["float"]["lambada"] = run_lambada(
            float_model, datasets["lambada"], enc,
            config.context_len, label="Float fp16")
    if "hellaswag" in datasets:
        results["float"]["hellaswag"] = run_hellaswag(
            float_model, datasets["hellaswag"], enc,
            config.context_len, label="Float fp16")
    free_model(float_model)
    print()

    # --- NB packed model ---
    print(f"{'_'*70}")
    print(f"  Phase 2: NativeBit 3-bit")
    print(f"{'_'*70}")
    if not os.path.exists(args.nb_ckpt):
        print(f"\n  ERROR: Packed checkpoint not found: {args.nb_ckpt}")
        print(f"  Run: python inference/pack.py <nb_checkpoint> --out {args.nb_ckpt}")
        sys.exit(1)

    # Load packed, materialize weights to fp16 for standard forward pass
    nb_model = load_packed_model(args.nb_ckpt, config, device="cpu")
    nb_model.eval()
    print("  Materializing packed weights to fp16...")
    from inference.triton_kernel import PackedLinear, BS
    for name, module in list(nb_model.named_modules()):
        if isinstance(module, PackedLinear):
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
            w = w.reshape(module.out_features, module.in_features).to(DTYPE)
            linear = torch.nn.Linear(module.in_features, module.out_features, bias=False)
            linear.weight.data = w
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(nb_model.named_modules())[parts[0]]
                setattr(parent, parts[1], linear)
    nb_model.to(DTYPE)
    nb_model.to(DEVICE)
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    results["nb_3bit"] = {}
    if "lambada" in datasets:
        results["nb_3bit"]["lambada"] = run_lambada(
            nb_model, datasets["lambada"], enc,
            config.context_len, label="NB 3-bit")
    if "hellaswag" in datasets:
        results["nb_3bit"]["hellaswag"] = run_hellaswag(
            nb_model, datasets["hellaswag"], enc,
            config.context_len, label="NB 3-bit")
    free_model(nb_model)
    print()

    # --- Summary ---
    print(f"{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Task':<15s}  {'Float fp16':>12s}  {'NB 3-bit':>12s}  {'Gap':>8s}")
    print(f"  {'-'*50}")
    for task in args.tasks:
        if task in results["float"] and task in results["nb_3bit"]:
            f_acc = results["float"][task]["accuracy"]
            n_acc = results["nb_3bit"][task]["accuracy"]
            gap = n_acc - f_acc
            print(f"  {task:<15s}  {f_acc:>10.2f}%  {n_acc:>10.2f}%  {gap:>+7.2f}%")
    print(f"{'='*70}\n")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
