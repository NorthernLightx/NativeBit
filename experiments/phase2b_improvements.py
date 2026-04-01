"""Phase 2b — targeted improvements to close the quantization gap.

Validation results showed:
  Float @ 10k:     62.95 PPL  (baseline)
  NativeBit 3-bit: 68.29 PPL  (8.5% gap)
  NativeBit 4-bit: 69.65 PPL  (10.6% gap — more entries doesn't help!)

This script tests targeted improvements:

G1: block_size=32 + codebook_lr=1e-3  — more expressive codebooks + faster LR
G2: Soft quantization annealing (tau 1.0->0.01) + block_size=32 + codebook_lr=1e-3
G3: Float baseline @ 10k (control, should match F1=62.95)
"""

import argparse
import json
import math
import os
import sys

if sys.platform == "win32":
    if "CC" not in os.environ:
        _cl = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe"
        if os.path.isfile(_cl):
            os.environ["CC"] = _cl
    if "TRITON_CACHE_DIR" not in os.environ:
        os.environ["TRITON_CACHE_DIR"] = r"C:\tmp\triton"
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"C:\tmp\inductor"
        os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from train import train
from configs.small import SmallConfig


def run_g1(config, device, log_dir, data_dir) -> dict:
    """G1: block_size=32 + codebook_lr=1e-3 @ 10k steps."""
    print("\n" + "=" * 60)
    print(f"G1: block_size={config.block_size}, codebook_lr={config.codebook_lr}")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    model = torch.compile(model)
    return train(model, config, device, "exp_g1_bs32_cblr", log_dir, data_dir)


def run_g2(config, device, log_dir, data_dir) -> dict:
    """G2: Soft quantization annealing + block_size=32 + codebook_lr=1e-3."""
    print("\n" + "=" * 60)
    print(f"G2: Soft annealing (tau {config.tau_start}->{config.tau_end} over {config.tau_anneal_steps} steps)")
    print(f"    block_size={config.block_size}, codebook_lr={config.codebook_lr}")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    # Skip torch.compile for soft annealing — tau changes cause recompilation issues
    return train(model, config, device, "exp_g2_soft_anneal", log_dir, data_dir)


def run_g3(config, device, log_dir, data_dir) -> dict:
    """G3: Float baseline @ 10k (control)."""
    print("\n" + "=" * 60)
    print("G3: Float Baseline @ 10k steps (control)")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=False)
    model = torch.compile(model)
    return train(model, config, device, "exp_g3_float_ctrl", log_dir, data_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 2b improvement experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=["g1", "g2", "g3"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    results = {}

    # G1: block_size=32 + higher codebook LR
    if args.only is None or args.only == "g1":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        cfg.block_size = 32          # halved from 64
        cfg.codebook_lr = 1e-3      # 33x higher than default 3e-5
        results["G1_bs32_cblr1e3"] = run_g1(cfg, device, args.log_dir, args.data_dir)

    # G2: Soft annealing + block_size=32 + higher codebook LR
    if args.only is None or args.only == "g2":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        cfg.block_size = 32
        cfg.codebook_lr = 1e-3
        # Soft quantization annealing
        cfg.tau_start = 1.0
        cfg.tau_end = 0.01
        cfg.tau_anneal_steps = 3000
        results["G2_soft_anneal"] = run_g2(cfg, device, args.log_dir, args.data_dir)

    # G3: Float control
    if args.only is None or args.only == "g3":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        results["G3_float_ctrl"] = run_g3(cfg, device, args.log_dir, args.data_dir)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2b IMPROVEMENT RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<30s} {'Val PPL':>10s} {'Test PPL':>10s} {'Val Loss':>10s}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f} {r['val_loss']:>10.4f}")

    # Load prior results
    for summary_file in ["validation_summary.json", "phase1_summary.json", "phase2_summary.json"]:
        path = os.path.join(args.log_dir, summary_file)
        if os.path.exists(path):
            with open(path) as f:
                prior = json.load(f)
            print(f"\n--- {summary_file} ---")
            for name, r in prior.items():
                if 'test_ppl' in r:
                    print(f"{name:<30s} {r.get('val_ppl', 0):>10.2f} {r['test_ppl']:>10.2f} {r.get('val_loss', 0):>10.4f}")

    summary_path = os.path.join(args.log_dir, "phase2b_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
