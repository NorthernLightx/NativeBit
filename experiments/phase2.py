"""Phase 2 experiments — entropy regularization & adaptive bit width.

Experiment D: NativeBit with entropy regularization (fixed 3-bit).
    Same as Phase 1 Experiment B, but with entropy loss term.
    Compare perplexity and utilization patterns vs Experiment B.

Experiment E: NativeBit with progressive compression.
    Start at 6-bit (64 entries), let model compress via merges.
    Record final bit width distribution, total size, final PPL.

Success metric: different blocks converge to different bit widths
(not everything settles at the same width).
"""

import argparse
import json
import math
import os
import sys

# torch.compile on Windows: MSVC compiler + short cache paths + disable FX graph cache
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
from nativebit.data import get_dataloaders
from train import train, run_evaluation
from configs.small import SmallConfig


def run_experiment_d(config, device, log_dir, data_dir) -> dict:
    """Experiment D — NativeBit 3-bit with entropy regularization."""
    print("\n" + "=" * 60)
    print("EXPERIMENT D: NativeBit 3-bit + Entropy Regularization")
    print(f"  entropy_lambda = {config.entropy_lambda}")
    print(f"  entropy_temperature = {config.entropy_temperature}")
    print("=" * 60)

    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    model = torch.compile(model)
    return train(model, config, device, "exp_d_entropy_3bit", log_dir, data_dir)


def run_experiment_e(config, device, log_dir, data_dir) -> dict:
    """Experiment E — Progressive compression from 6-bit.

    Starts with 64 codebook entries (6-bit) and progressively merges
    underutilized entries at scheduled steps.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT E: Progressive Compression (6-bit -> adaptive)")
    print(f"  Initial entries: {config.n_codebook}")
    print(f"  Merge steps: {config.merge_steps}")
    print(f"  Merge util threshold: {config.merge_util_threshold}")
    print(f"  entropy_lambda = {config.entropy_lambda}")
    print("=" * 60)

    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    # Initialize active masks from the start for progressive compression
    model.init_all_active_masks()
    model = torch.compile(model)
    results = train(model, config, device, "exp_e_progressive", log_dir, data_dir)

    # After training, report bit width distribution
    if hasattr(model, 'get_bit_width_summary'):
        bw = model.get_bit_width_summary()
        print(f"\n  Final bit width distribution:")
        print(f"    Mean: {bw['global_mean_bits']} bits")
        print(f"    Min:  {bw['global_min_bits']} bits")
        print(f"    Max:  {bw['global_max_bits']} bits")
        print(f"    Histogram: {bw['bit_width_histogram']}")
        results["bit_width_summary"] = bw

    if hasattr(model, 'compute_model_size_bits'):
        sz = model.compute_model_size_bits()
        print(f"\n  Model size:")
        print(f"    Quantized weights: {sz['quantized_weight_bytes']:,} bytes")
        print(f"    Codebook overhead: {sz['codebook_bits'] // 8:,} bytes")
        print(f"    Float params: {sz['float_bits'] // 8:,} bytes")
        print(f"    Total: {sz['total_bytes']:,} bytes")
        results["model_size"] = sz

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2 experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=["d", "e"],
                        help="Run only one experiment")
    parser.add_argument("--entropy-lambda", type=float, default=None,
                        help="Override entropy regularization strength")
    parser.add_argument("--entropy-temp", type=float, default=None,
                        help="Override entropy temperature")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    results = {}

    # ── Experiment D: fixed 3-bit + entropy reg ──
    if args.only is None or args.only == "d":
        config_d = SmallConfig()
        if args.max_steps is not None:
            config_d.max_steps = args.max_steps
        config_d.seed = args.seed
        config_d.entropy_lambda = args.entropy_lambda if args.entropy_lambda is not None else 0.01
        if args.entropy_temp is not None:
            config_d.entropy_temperature = args.entropy_temp
        results["D_entropy_3bit"] = run_experiment_d(config_d, device, args.log_dir, args.data_dir)

    # ── Experiment E: progressive compression from 6-bit ──
    if args.only is None or args.only == "e":
        config_e = SmallConfig()
        config_e.seed = args.seed
        # Start at 6-bit (64 entries)
        config_e.n_codebook = 64
        # Longer training for progressive compression
        config_e.max_steps = args.max_steps if args.max_steps is not None else 10000
        # Enable progressive compression
        config_e.progressive = True
        config_e.merge_steps = [2000, 4000, 6000, 8000]
        config_e.merge_util_threshold = 0.02
        # Entropy reg helps guide the merging
        config_e.entropy_lambda = args.entropy_lambda if args.entropy_lambda is not None else 0.01
        if args.entropy_temp is not None:
            config_e.entropy_temperature = args.entropy_temp
        results["E_progressive"] = run_experiment_e(config_e, device, args.log_dir, args.data_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<30s} {'Val PPL':>10s} {'Test PPL':>10s} {'Val Loss':>10s}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<30s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f} {r['val_loss']:>10.4f}")

    # Compare with Phase 1 if available
    p1_path = os.path.join(args.log_dir, "phase1_summary.json")
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        print("\n--- Phase 1 comparison ---")
        for name, r in p1.items():
            print(f"{name:<30s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f} {r['val_loss']:>10.4f}")

    # Save summary
    # Convert non-serializable items
    serializable = {}
    for k, v in results.items():
        serializable[k] = {sk: sv for sk, sv in v.items()
                          if isinstance(sv, (int, float, str, dict, list))}

    summary_path = os.path.join(args.log_dir, "phase2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
