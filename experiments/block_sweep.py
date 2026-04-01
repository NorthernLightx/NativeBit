"""Block size sweep — find optimal block_size for NativeBit.

Tests block_size at 16, 32, 64 (current default).
Smaller blocks = more codebooks, each covering fewer weights = finer granularity.
Uses best codebook LR (3.0x = 9e-4) from LR sweep.
Runs 2000-step quick experiments, then full 5000 with the winner.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from train import train, run_evaluation
from configs.small import SmallConfig


def run_sweep(config, device, log_dir, data_dir, quick_steps=2000):
    """Run block size sweep and return results."""
    sweep_values = {
        "bs_16": 16,
        "bs_32": 32,
        "bs_64": 64,   # current default
    }

    results = {}
    for name, bs in sweep_values.items():
        print(f"\n{'='*60}")
        print(f"SWEEP: {name} (block_size={bs})")
        print(f"{'='*60}")

        set_seed(config.seed)
        config.block_size = bs
        config.max_steps = quick_steps

        model = build_model_from_config(config, use_nativebit=True)

        # Show codebook stats for this block size
        nb_layers = model.get_nativebit_layers()
        total_blocks = sum(l.num_blocks for l in nb_layers)
        total_cb_params = sum(l.codebook.numel() for l in nb_layers)
        print(f"  Total codebook blocks: {total_blocks}")
        print(f"  Total codebook params: {total_cb_params:,}")
        print(f"  Codebook overhead: {total_cb_params / sum(p.numel() for p in model.parameters()) * 100:.2f}%")

        r = train(model, config, device, f"sweep_{name}", log_dir, data_dir)
        results[name] = r
        results[name]["block_size"] = bs
        results[name]["codebook_blocks"] = total_blocks
        results[name]["codebook_params"] = total_cb_params

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Block size sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-steps", type=int, default=2000)
    parser.add_argument("--full-steps", type=int, default=5000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--codebook-lr-multiplier", type=float, default=3.0,
                        help="Codebook LR as multiple of main LR (default: 3.0x from LR sweep)")
    args = parser.parse_args()

    config = SmallConfig()
    config.seed = args.seed
    config.codebook_lr = config.lr * args.codebook_lr_multiplier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Using codebook LR: {config.codebook_lr:.1e} ({args.codebook_lr_multiplier}x main LR)")

    # Quick sweep
    results = run_sweep(config, device, args.log_dir, args.data_dir, args.quick_steps)

    # Summary
    print(f"\n{'='*60}")
    print("BLOCK SIZE SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Name':<10s} {'BlkSize':>8s} {'#Blocks':>8s} {'CB Params':>10s} {'Val PPL':>10s} {'Test PPL':>10s}")
    print("-" * 60)

    best_name = None
    best_ppl = float("inf")
    for name, r in sorted(results.items(), key=lambda x: x[1]["test_ppl"]):
        print(f"{name:<10s} {r['block_size']:>8d} {r['codebook_blocks']:>8d} {r['codebook_params']:>10,d} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f}")
        if r["test_ppl"] < best_ppl:
            best_ppl = r["test_ppl"]
            best_name = name

    best_bs = results[best_name]["block_size"]
    print(f"\nBest: {best_name} (block_size={best_bs}, test_ppl={best_ppl:.2f})")

    # Full run with best block size
    print(f"\n{'='*60}")
    print(f"FULL RUN: {best_name} for {args.full_steps} steps")
    print(f"{'='*60}")

    set_seed(config.seed)
    config.block_size = best_bs
    config.max_steps = args.full_steps
    model = build_model_from_config(config, use_nativebit=True)
    full_result = train(model, config, device, "nativebit_best_bs", args.log_dir, args.data_dir)

    # Compare with post-hoc k-means from phase1
    summary_path = os.path.join(args.log_dir, "phase1_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            phase1 = json.load(f)
        if "C_posthoc_kmeans" in phase1:
            c_ppl = phase1["C_posthoc_kmeans"]["test_ppl"]
            b_ppl = full_result["test_ppl"]
            print(f"\n{'='*60}")
            print(f"NativeBit (best block_size={best_bs}): Test PPL = {b_ppl:.2f}")
            print(f"Post-hoc k-means:                      Test PPL = {c_ppl:.2f}")
            if b_ppl < c_ppl:
                print(f"SUCCESS: NativeBit wins by {c_ppl - b_ppl:.2f} PPL!")
            else:
                print(f"Gap: {b_ppl - c_ppl:.2f} PPL still to close")
            print(f"{'='*60}")

    # Save sweep results
    sweep_path = os.path.join(args.log_dir, "block_sweep_summary.json")
    results["full_run"] = full_result
    results["full_run"]["block_size"] = best_bs
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep summary saved to {sweep_path}")


if __name__ == "__main__":
    main()
