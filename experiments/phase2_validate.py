"""Phase 2 validation — isolate what drives Experiment E's improvement.

Experiment E got 68.08 PPL but had two confounds vs Phase 1:
  1. 10k steps (vs 5k)
  2. 64 codebook entries (vs 8), even after merging ~16 remain active

This script runs controlled comparisons:

F1: Float baseline @ 10k steps    — does more training help float too?
F2: Fixed 4-bit (16 entries) @ 10k — is 4-bit just the right capacity?
F3: Fixed 3-bit (8 entries) @ 10k  — does 3-bit just need more training?

These isolate: training budget vs codebook capacity vs progressive compression.
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


def run_f1(config, device, log_dir, data_dir) -> dict:
    """F1: Float baseline @ 10k steps."""
    print("\n" + "=" * 60)
    print("F1: Float Baseline @ 10k steps")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=False)
    model = torch.compile(model)
    return train(model, config, device, "val_f1_float_10k", log_dir, data_dir)


def run_f2(config, device, log_dir, data_dir) -> dict:
    """F2: Fixed 4-bit (16 entries) @ 10k steps."""
    print("\n" + "=" * 60)
    print(f"F2: Fixed 4-bit ({config.n_codebook} entries) @ {config.max_steps} steps")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    model = torch.compile(model)
    return train(model, config, device, "val_f2_4bit_10k", log_dir, data_dir)


def run_f3(config, device, log_dir, data_dir) -> dict:
    """F3: Fixed 3-bit (8 entries) @ 10k steps."""
    print("\n" + "=" * 60)
    print(f"F3: Fixed 3-bit ({config.n_codebook} entries) @ {config.max_steps} steps")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    model = torch.compile(model)
    return train(model, config, device, "val_f3_3bit_10k", log_dir, data_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 validation experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=["f1", "f2", "f3"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    results = {}

    # F1: Float @ 10k
    if args.only is None or args.only == "f1":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        results["F1_float_10k"] = run_f1(cfg, device, args.log_dir, args.data_dir)

    # F2: Fixed 4-bit @ 10k
    if args.only is None or args.only == "f2":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        cfg.n_codebook = 16  # 4-bit
        results["F2_4bit_10k"] = run_f2(cfg, device, args.log_dir, args.data_dir)

    # F3: Fixed 3-bit @ 10k
    if args.only is None or args.only == "f3":
        cfg = SmallConfig()
        cfg.seed = args.seed
        cfg.max_steps = args.max_steps
        cfg.n_codebook = 8  # 3-bit
        results["F3_3bit_10k"] = run_f3(cfg, device, args.log_dir, args.data_dir)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<30s} {'Val PPL':>10s} {'Test PPL':>10s} {'Val Loss':>10s}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f} {r['val_loss']:>10.4f}")

    # Load prior results for full comparison
    for summary_file in ["phase1_summary.json", "phase2_summary.json"]:
        path = os.path.join(args.log_dir, summary_file)
        if os.path.exists(path):
            with open(path) as f:
                prior = json.load(f)
            print(f"\n--- {summary_file} ---")
            for name, r in prior.items():
                ppl_keys = [k for k in r if 'ppl' in k.lower()]
                if 'test_ppl' in r:
                    print(f"{name:<30s} {r.get('val_ppl', 0):>10.2f} {r['test_ppl']:>10.2f} {r.get('val_loss', 0):>10.4f}")

    summary_path = os.path.join(args.log_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
