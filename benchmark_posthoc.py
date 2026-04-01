"""Post-hoc quantization benchmark — compare NativeBit vs quantize-after-training.

Trains a float model, then applies post-hoc quantization methods and measures PPL.
Uses PyTorch on local GPU (RTX 3070).

Usage:
    python benchmark_posthoc.py --max-steps 10000
"""

import argparse
import copy
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from nativebit.device import get_device, amp_context
from nativebit.baselines import (
    quantize_uniform, quantize_kmeans, quantize_nf4,
    compute_model_size,
)
from configs.default import DefaultConfig


@torch.no_grad()
def run_eval(model, loader, device, max_batches=0):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss * y.numel()
        total_tokens += y.numel()
    model.train()
    return total_loss.item() / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    config = DefaultConfig()
    config.max_steps = args.max_steps

    print(f"\n{'='*60}")
    print(f"  Post-hoc Quantization Benchmark")
    print(f"  Device: {device}, Steps: {args.max_steps}")
    print(f"{'='*60}\n")

    train_loader, valid_loader, test_loader = get_dataloaders(
        config.context_len, config.batch_size, args.data_dir,
    )

    # Train float model
    print("Training float model...", flush=True)
    model = build_model_from_config(config, use_nativebit=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_iter = iter(train_loader)

    t0 = time.time()
    for step in range(args.max_steps):
        optimizer.zero_grad()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % 500 == 0:
            print(f"  step={step}, loss={loss.item():.4f}", flush=True)

    print(f"  Training done: {time.time()-t0:.0f}s")

    float_state = copy.deepcopy(model.state_dict())

    # Float baseline
    print("\nFloat baseline...", flush=True)
    test_loss = run_eval(model, test_loader, device)
    float_ppl = math.exp(min(test_loss, 20))
    print(f"  Float PPL: {float_ppl:.2f}")

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    exclude = [raw_model.lm_head] if hasattr(raw_model, 'lm_head') else []

    float16_size = compute_model_size(raw_model, "float16", exclude_modules=exclude)
    float32_size = compute_model_size(raw_model, "float32", exclude_modules=exclude)

    results = [
        {"method": "Float32", "ppl": float_ppl, "mb": float32_size["total_bytes"]/1e6, "bpw": 32},
        {"method": "Float16", "ppl": float_ppl, "mb": float16_size["total_bytes"]/1e6, "bpw": 16},
    ]

    # Post-hoc methods
    methods = [
        ("RTN 3-bit bs128", quantize_uniform, {"bits": 3, "block_size": 128}),
        ("RTN 3-bit bs64", quantize_uniform, {"bits": 3, "block_size": 64}),
        ("K-means 8-entry bs128", quantize_kmeans, {"n_entries": 8, "block_size": 128}),
        ("K-means 6-entry bs128", quantize_kmeans, {"n_entries": 6, "block_size": 128}),
        ("NF4 4-bit bs128", quantize_nf4, {"block_size": 128}),
        ("NF4 4-bit bs64", quantize_nf4, {"block_size": 64}),
    ]

    for name, fn, kwargs in methods:
        print(f"\n{name}...", flush=True)
        model.load_state_dict(copy.deepcopy(float_state))
        fn(raw_model, exclude_modules=exclude, **kwargs)
        test_loss = run_eval(model, test_loader, device)
        ppl = math.exp(min(test_loss, 20))

        if "RTN" in name:
            size = compute_model_size(raw_model, "uniform", bits=kwargs["bits"],
                                       block_size=kwargs["block_size"], exclude_modules=exclude)
            bpw = kwargs["bits"]
        elif "K-means" in name:
            ne = kwargs["n_entries"]
            size = compute_model_size(raw_model, "kmeans", n_entries=ne,
                                       block_size=kwargs["block_size"], exclude_modules=exclude)
            bpw = math.ceil(math.log2(ne))
        else:
            size = compute_model_size(raw_model, "nf4", block_size=kwargs["block_size"],
                                       exclude_modules=exclude)
            bpw = 4

        print(f"  PPL={ppl:.2f}, Size={size['total_bytes']/1e6:.1f} MB")
        results.append({"method": name, "ppl": ppl, "mb": size["total_bytes"]/1e6, "bpw": bpw})

    # NativeBit theoretical sizes
    nb6 = compute_model_size(raw_model, "nativebit", n_entries=6, block_size=128,
                              exclude_modules=exclude)
    nb8 = compute_model_size(raw_model, "nativebit", n_entries=8, block_size=128,
                              exclude_modules=exclude)

    # Print table
    print(f"\n{'='*70}")
    print(f"  RESULTS ({args.max_steps} steps, WikiText-2, {config.n_embd}d)")
    print(f"{'='*70}")
    print(f"  {'Method':<25s} {'PPL':>8s} {'Size MB':>8s} {'BPW':>5s} {'vs Float':>10s}")
    print(f"  {'-'*56}")
    for r in results:
        delta = f"+{(r['ppl']/float_ppl-1)*100:.1f}%" if r['ppl'] != float_ppl else "baseline"
        print(f"  {r['method']:<25s} {r['ppl']:>8.2f} {r['mb']:>8.1f} {r['bpw']:>5} {delta:>10s}")

    print(f"\n  NativeBit packed (theoretical):")
    print(f"    cb=6 bs=128: {nb6['total_bytes']/1e6:.1f} MB (TPU PPL: ~193.90)")
    print(f"    cb=8 bs=128: {nb8['total_bytes']/1e6:.1f} MB (TPU PPL: ~185.66)")


if __name__ == "__main__":
    main()
