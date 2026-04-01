"""125M benchmark: Float vs NativeBit 2-bit vs Progressive on WikiText-103.

Usage:
  python -u experiments/bench_125m.py 2>&1 | tee logs/bench125m.log
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from configs.large import LargeConfig, LargeFloatConfig, LargeProgressiveConfig
from train import train


EXPERIMENTS = {
    "float":       ("bench_float_125m",       LargeFloatConfig,        False),
    "2bit":        ("bench_2bit_125m",         LargeConfig,             True),
    "progressive": ("bench_progressive_125m",  LargeProgressiveConfig,  True),
}

LOG_DIR = "logs/bench125m"
DATA_DIR = "data"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_run = {args.only: EXPERIMENTS[args.only]} if args.only else EXPERIMENTS
    results = []

    for key, (exp_name, ConfigClass, use_nativebit) in to_run.items():
        config = ConfigClass()

        label = f"{key} ({'NativeBit' if use_nativebit else 'Float'})"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  Name: {exp_name}")
        print(f"  Steps: {config.max_steps}, LR: {config.lr}, WD: {config.weight_decay}")
        if use_nativebit:
            print(f"  n_codebook={config.n_codebook}, block_size={config.block_size}, EMA={config.use_ema}")
        print(f"{'='*70}\n")

        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=use_nativebit)

        t0 = time.time()
        try:
            res = train(model, config, device, exp_name, LOG_DIR, DATA_DIR)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM with batch_size={config.batch_size}, retrying with {config.batch_size // 2}")
                del model
                torch.cuda.empty_cache()
                config.batch_size //= 2
                set_seed(config.seed)
                model = build_model_from_config(config, use_nativebit=use_nativebit)
                t0 = time.time()
                res = train(model, config, device, exp_name, LOG_DIR, DATA_DIR)
            else:
                raise
        elapsed = time.time() - t0

        results.append({
            "name": key, "test_ppl": res["test_ppl"],
            "val_ppl": res["val_ppl"], "time_min": elapsed / 60,
        })
        print(f"\n  {key} done: test_ppl={res['test_ppl']:.2f}, "
              f"val_ppl={res['val_ppl']:.2f}, {elapsed/60:.1f} min")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  125M BENCHMARK RESULTS (WikiText-103, 30k steps)")
    print(f"{'='*70}")
    print(f"  {'Config':<16} {'Test PPL':>10} {'Val PPL':>10} {'Time':>10}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['name']:<16} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['time_min']:>8.1f} min")


if __name__ == "__main__":
    main()
