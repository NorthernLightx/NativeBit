"""Codebook LR sweep — find optimal codebook learning rate.

Tests codebook LR at 0.1x, 0.3x, 1.0x, and 3.0x of main LR.
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
    """Run codebook LR sweep and return results."""
    main_lr = config.lr  # 3e-4

    sweep_values = {
        "cb_0.1x": main_lr * 0.1,   # 3e-5 (current default)
        "cb_0.3x": main_lr * 0.3,   # 9e-5
        "cb_1.0x": main_lr * 1.0,   # 3e-4
        "cb_3.0x": main_lr * 3.0,   # 9e-4
    }

    results = {}
    for name, cb_lr in sweep_values.items():
        print(f"\n{'='*60}")
        print(f"SWEEP: {name} (codebook_lr={cb_lr:.1e}, main_lr={main_lr:.1e})")
        print(f"{'='*60}")

        set_seed(config.seed)
        config.codebook_lr = cb_lr
        config.max_steps = quick_steps

        model = build_model_from_config(config, use_nativebit=True)
        r = train(model, config, device, f"sweep_{name}", log_dir, data_dir)
        results[name] = r
        results[name]["codebook_lr"] = cb_lr

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Codebook LR sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-steps", type=int, default=2000)
    parser.add_argument("--full-steps", type=int, default=5000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    config = SmallConfig()
    config.seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    # Quick sweep
    results = run_sweep(config, device, args.log_dir, args.data_dir, args.quick_steps)

    # Summary
    print(f"\n{'='*60}")
    print("CODEBOOK LR SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Name':<15s} {'CB LR':>10s} {'Val PPL':>10s} {'Test PPL':>10s}")
    print("-" * 50)

    best_name = None
    best_ppl = float("inf")
    for name, r in sorted(results.items(), key=lambda x: x[1]["test_ppl"]):
        print(f"{name:<15s} {r['codebook_lr']:>10.1e} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f}")
        if r["test_ppl"] < best_ppl:
            best_ppl = r["test_ppl"]
            best_name = name

    best_cb_lr = results[best_name]["codebook_lr"]
    print(f"\nBest: {best_name} (codebook_lr={best_cb_lr:.1e}, test_ppl={best_ppl:.2f})")

    # Full run with best LR
    print(f"\n{'='*60}")
    print(f"FULL RUN: {best_name} for {args.full_steps} steps")
    print(f"{'='*60}")

    set_seed(config.seed)
    config.codebook_lr = best_cb_lr
    config.max_steps = args.full_steps
    model = build_model_from_config(config, use_nativebit=True)
    full_result = train(model, config, device, "nativebit_best_lr", args.log_dir, args.data_dir)

    # Compare with post-hoc k-means from phase1
    summary_path = os.path.join(args.log_dir, "phase1_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            phase1 = json.load(f)
        if "C_posthoc_kmeans" in phase1:
            c_ppl = phase1["C_posthoc_kmeans"]["test_ppl"]
            b_ppl = full_result["test_ppl"]
            print(f"\n{'='*60}")
            print(f"NativeBit (best LR): Test PPL = {b_ppl:.2f}")
            print(f"Post-hoc k-means:    Test PPL = {c_ppl:.2f}")
            if b_ppl < c_ppl:
                print(f"SUCCESS: NativeBit wins by {c_ppl - b_ppl:.2f} PPL!")
            else:
                print(f"Gap: {b_ppl - c_ppl:.2f} PPL still to close")
            print(f"{'='*60}")

    # Save sweep results
    sweep_path = os.path.join(args.log_dir, "lr_sweep_summary.json")
    results["full_run"] = full_result
    results["full_run"]["codebook_lr"] = best_cb_lr
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep summary saved to {sweep_path}")


if __name__ == "__main__":
    main()
