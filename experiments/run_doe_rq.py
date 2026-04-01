"""DoE: Requantize interval sensitivity study.

Phase 1 screening — 3K steps each, 2 seeds.
Block A: rq × LR (125M, 3-bit)
Block B: rq × bits (125M, LR=6e-4)
Block C: rq × scale (3-bit, LR=6e-4)

Each config is independent — run on separate TPUs for parallelism.
This script runs ONE config (specified by index or args).

Usage:
  python run_doe_rq.py --index N        # run config N (0-41)
  python run_doe_rq.py --list           # print all configs
  python run_doe_rq.py --range 0 7      # run configs 0-7 sequentially
"""
import sys, os, json, time, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nativebit_jax.train import train
from configs.tpu import TPUMediumConfig, TPULargeConfig

# ── Build config table ──────────────────────────────────────────────
CONFIGS = []

RQ_LEVELS = [10, 100, 500]
SEEDS = [42, 123]

# Block A: rq × LR (125M, 3-bit cb=8)
for rq in RQ_LEVELS:
    for lr in [3e-4, 6e-4, 1.2e-3]:
        for seed in SEEDS:
            CONFIGS.append({
                "block": "A", "scale": "125M", "rq": rq, "lr": lr,
                "cb": 8, "seed": seed,
                "name": f"A_rq{rq}_lr{lr:.0e}_s{seed}",
            })

# Block B: rq × bits (125M, LR=6e-4)
for rq in RQ_LEVELS:
    for cb in [4, 8, 16]:
        for seed in SEEDS:
            key = f"B_rq{rq}_cb{cb}_s{seed}"
            # Skip duplicates with Block A (cb=8, lr=6e-4)
            dup = f"A_rq{rq}_lr6e-04_s{seed}"
            if any(c["name"] == dup for c in CONFIGS) and cb == 8:
                CONFIGS.append({
                    "block": "B", "scale": "125M", "rq": rq, "lr": 6e-4,
                    "cb": cb, "seed": seed, "name": key, "alias": dup,
                })
            else:
                CONFIGS.append({
                    "block": "B", "scale": "125M", "rq": rq, "lr": 6e-4,
                    "cb": cb, "seed": seed, "name": key,
                })

# Block C: rq × scale (3-bit cb=8, LR=6e-4)
for rq in RQ_LEVELS:
    for scale in ["125M", "350M"]:
        for seed in SEEDS:
            key = f"C_rq{rq}_{scale}_s{seed}"
            # Skip 125M duplicates with Block A
            if scale == "125M":
                dup = f"A_rq{rq}_lr6e-04_s{seed}"
                CONFIGS.append({
                    "block": "C", "scale": scale, "rq": rq, "lr": 6e-4,
                    "cb": 8, "seed": seed, "name": key, "alias": dup,
                })
            else:
                CONFIGS.append({
                    "block": "C", "scale": scale, "rq": rq, "lr": 6e-4,
                    "cb": 8, "seed": seed, "name": key,
                })

# Deduplicate: only run configs without "alias"
UNIQUE = [c for c in CONFIGS if "alias" not in c]
ALL = CONFIGS  # for result collection


def make_config(spec):
    """Create a training config from a DoE spec."""
    if spec["scale"] == "125M":
        cfg = TPUMediumConfig()
        cfg.batch_size = 4  # fits on v6e-8 with NB codebook+cache overhead
    else:
        cfg = TPULargeConfig()
        cfg.batch_size = 8  # fits on v6e-8

    cfg.max_steps = 3000
    cfg.requantize_every = spec["rq"]
    cfg.revive_every = max(spec["rq"] * 5, 500)
    cfg.lr = spec["lr"]
    cfg.n_codebook = spec["cb"]
    cfg.seed = spec["seed"]
    cfg.log_every = 100
    return cfg


def run_one(idx):
    """Run a single config by index into UNIQUE."""
    spec = UNIQUE[idx]
    os.makedirs("logs/jax/doe_rq", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DoE Config {idx}/{len(UNIQUE)}: {spec['name']}")
    print(f"  Block={spec['block']} scale={spec['scale']} rq={spec['rq']}")
    print(f"  lr={spec['lr']} cb={spec['cb']} seed={spec['seed']}")
    print(f"{'='*60}\n")

    cfg = make_config(spec)

    t0 = time.time()
    r = train(cfg, use_nativebit=True, experiment_name=f"doe_{spec['name']}",
              log_dir="logs/jax/doe_rq", data_dir="data")
    elapsed = round(time.time() - t0, 1)

    result = {
        **spec,
        "ppl": r["test_ppl"],
        "elapsed_s": elapsed,
    }

    outpath = f"logs/jax/doe_rq/{spec['name']}.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Result: PPL={r['test_ppl']:.2f}, time={elapsed:.0f}s")
    print(f"  Saved: {outpath}")

    del r; gc.collect()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all configs")
    parser.add_argument("--list-unique", action="store_true", help="List unique (non-alias) configs")
    parser.add_argument("--index", type=int, help="Run config at index")
    parser.add_argument("--range", type=int, nargs=2, help="Run configs in range [start, end]")
    args = parser.parse_args()

    if args.list:
        print(f"All configs: {len(ALL)} ({len(UNIQUE)} unique)")
        for i, c in enumerate(ALL):
            alias = f" -> {c['alias']}" if "alias" in c else ""
            print(f"  {i:>3d} {c['name']:<35s} block={c['block']} rq={c['rq']:>4d} "
                  f"lr={c['lr']:.0e} cb={c['cb']:>2d} {c['scale']}{alias}")
    elif args.list_unique:
        print(f"Unique configs to run: {len(UNIQUE)}")
        for i, c in enumerate(UNIQUE):
            print(f"  {i:>3d} {c['name']:<35s} block={c['block']} rq={c['rq']:>4d} "
                  f"lr={c['lr']:.0e} cb={c['cb']:>2d} {c['scale']}")
    elif args.index is not None:
        run_one(args.index)
    elif args.range is not None:
        import subprocess
        start, end = args.range
        results = []
        for i in range(start, min(end + 1, len(UNIQUE))):
            spec = UNIQUE[i]
            print(f"\n=== Launching config {i}: {spec['name']} as subprocess ===")
            # Run each config in a SEPARATE Python process so TPU memory is fully
            # released between configs (JAX device memory is not freed by gc.collect)
            ret = subprocess.run(
                [sys.executable, __file__, "--index", str(i)],
                capture_output=False,
            )
            if ret.returncode != 0:
                print(f"  WARNING: config {i} failed (exit code {ret.returncode})")
            # Collect result if saved
            outpath = f"logs/jax/doe_rq/{spec['name']}.json"
            if os.path.exists(outpath):
                with open(outpath) as f:
                    results.append(json.load(f))
        print(f"\n{'='*60}")
        print(f"  BATCH DONE: configs {start}-{end}")
        for r in results:
            print(f"  {r['name']}: PPL={r['ppl']:.2f}")
    else:
        parser.print_help()
