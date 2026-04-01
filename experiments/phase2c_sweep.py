"""Phase 2c — sweep block_size x codebook_lr x n_entries with health checks.

Best results so far:
  Float @ 10k:     62.95 PPL
  G1 (3b, bs32):   67.63 PPL  (post-hoc floor: 67.71)
  G5 (3b, bs16):   67.52 PPL  (post-hoc floor: 66.18)

Post-hoc quantization error floors:
  3-bit bs=16: 66.18 | 3-bit bs=32: 67.71 | 3-bit bs=64: 72.58
  4-bit bs=16: 62.95 | 4-bit bs=32: 64.13 | 4-bit bs=64: 65.06

This script sweeps configs to find the best co-trained result.
Health checks abort runs early if they look doomed (PPL or dead% too high).
"""

import argparse
import json
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


SWEEP_CONFIGS = {
    # 4-bit experiments — try to approach float (post-hoc floor: 64.13 @ bs=32)
    "H1_4bit_bs32_cb3e4": dict(n_codebook=16, block_size=32, codebook_lr=3e-4),
    "H2_4bit_bs32_cb1e4": dict(n_codebook=16, block_size=32, codebook_lr=1e-4),
    "H3_4bit_bs16_cb3e4": dict(n_codebook=16, block_size=16, codebook_lr=3e-4),
    # 3-bit with lower codebook LR at bs=16 (G5 had too many dead entries)
    "H4_3bit_bs16_cb3e4": dict(n_codebook=8, block_size=16, codebook_lr=3e-4),
}


def run_experiment(name, overrides, device, args):
    print(f"\n{'='*60}")
    print(f"{name}: {overrides}")
    print(f"{'='*60}")

    cfg = SmallConfig()
    cfg.seed = args.seed
    cfg.max_steps = args.max_steps
    cfg.health_check_steps = [1000, 2000]
    cfg.health_max_ppl = {1000: 280, 2000: 200}
    cfg.health_max_dead_pct = 15.0

    for k, v in overrides.items():
        setattr(cfg, k, v)

    set_seed(cfg.seed)
    model = build_model_from_config(cfg, use_nativebit=True)
    model = torch.compile(model)
    return train(model, cfg, device, f"exp_{name.lower()}", args.log_dir, args.data_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 2c sweep experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment (e.g. H1_4bit_bs32_cb3e4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    configs = SWEEP_CONFIGS
    if args.only:
        key = args.only
        if key not in configs:
            # Try case-insensitive match
            for k in configs:
                if k.lower() == key.lower():
                    key = k
                    break
        configs = {key: configs[key]}

    results = {}
    for name, overrides in configs.items():
        result = run_experiment(name, overrides, device, args)
        results[name] = result
        aborted = result.get("aborted_at_step")
        if aborted:
            print(f"  ** {name} ABORTED at step {aborted}: {result.get('abort_reason', '')}")
        else:
            print(f"  ** {name}: Val PPL={result['val_ppl']:.2f}, Test PPL={result['test_ppl']:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 2c SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'Experiment':<30s} {'Val PPL':>10s} {'Test PPL':>10s} {'Note':>15s}")
    print("-" * 70)
    for name, r in results.items():
        if r.get("aborted_at_step"):
            print(f"{name:<30s} {'---':>10s} {'---':>10s} {'ABORTED @'+str(r['aborted_at_step']):>15s}")
        else:
            print(f"{name:<30s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f}")

    # Load prior best results for comparison
    for sf in ["validation_summary.json", "phase2b_summary.json"]:
        path = os.path.join(args.log_dir, sf)
        if os.path.exists(path):
            with open(path) as f:
                prior = json.load(f)
            print(f"\n--- {sf} ---")
            for nm, r in prior.items():
                if "test_ppl" in r and r["test_ppl"] != float("inf"):
                    print(f"{nm:<30s} {r.get('val_ppl', 0):>10.2f} {r['test_ppl']:>10.2f}")

    summary_path = os.path.join(args.log_dir, "phase2c_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
