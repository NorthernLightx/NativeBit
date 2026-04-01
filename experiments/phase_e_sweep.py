"""Phase E: Comprehensive sweep of all research ideas, combinations, and edge cases.

Sweep plan (~20 experiments, ~5 min each, ~2 hours total):

Phase 1 — Bit Width Sweep (EMA baseline)
  a. EMA 2-bit (4 entries, bs=64)
  b. EMA 3-bit (8 entries, bs=64) — reference baseline
  c. EMA 4-bit (16 entries, bs=32)
  d. EMA 5-bit (32 entries, bs=32)

Phase 2 — Block Size Sweep (EMA 3-bit)
  e. EMA 3-bit bs=16
  f. EMA 3-bit bs=32
  g. EMA 3-bit bs=128

Phase 3 — Research Ideas
  h. EMA + importance-aware (attn bs=32, ffn bs=64)
  i. EMA + learned distance
  j. STE factored + learned distance
  k. Delayed quantization (float 500 steps, then EMA 3-bit)
  l. Codebook distillation (float teacher + EMA 3-bit student)

Phase 4 — Combinations & Tuning
  m. EMA + factored init
  n. EMA + entropy reg (lambda=0.05)
  o. EMA decay=0.995
  p. EMA decay=0.999
  q. EMA + higher main LR (6e-4)
  r. Weight decay=0.05
  s. Weight decay=0.0

Phase 5 — Best Combo Candidates
  t. EMA 3-bit bs=32 decay=0.995
"""

import os
import sys
import time

# torch.compile on Windows
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

import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from train import train


class SweepConfig:
    """Base config for sweep experiments (SmallConfig-sized, WikiText-2)."""
    # Model
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    vocab_size: int = 50257

    # NativeBit defaults
    block_size: int = 64
    n_codebook: int = 8

    # Training
    batch_size: int = 16
    lr: float = 3e-4
    codebook_lr: float = 3e-5
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 100
    log_every: int = 50

    # Data
    dataset: str = "wikitext-2"
    seed: int = 42

    # Disabled features (safe defaults)
    entropy_lambda: float = 0.0
    entropy_temperature: float = 0.01
    progressive: bool = False
    merge_util_threshold: float = 0.02
    merge_dist_threshold = None
    merge_steps = None
    tau_start: float = 0.0
    tau_end: float = 0.01
    tau_anneal_steps: int = 3000
    quantize_mode: str = "ste"
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.1
    gumbel_tau_anneal_steps: int = 0
    diversity_lambda: float = 0.0
    n_codebooks: int = 1

    # EMA (overridden per experiment)
    use_ema: bool = False
    ema_decay: float = 0.99

    # Features (overridden per experiment)
    factored_codebook: bool = False
    learned_distance: bool = False
    factored_init: bool = False
    block_size_attn = None
    block_size_ffn = None
    weight_decay: float = 0.01
    delay_quant_steps: int = 0
    distill_alpha: float = 0.0
    distill_temp: float = 2.0

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 15.0


# ── Experiment definitions ──────────────────────────────────────────────────

EXPERIMENTS = {
    # Phase 1: Bit Width Sweep
    "a": {
        "name": "sweepE_ema_2bit",
        "desc": "EMA 2-bit (4 entries, bs=64)",
        "use_ema": True, "n_codebook": 4, "block_size": 64,
    },
    "b": {
        "name": "sweepE_ema_3bit",
        "desc": "EMA 3-bit (8 entries, bs=64) — baseline",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
    },
    "c": {
        "name": "sweepE_ema_4bit_bs32",
        "desc": "EMA 4-bit (16 entries, bs=32)",
        "use_ema": True, "n_codebook": 16, "block_size": 32,
    },
    "d": {
        "name": "sweepE_ema_5bit_bs32",
        "desc": "EMA 5-bit (32 entries, bs=32)",
        "use_ema": True, "n_codebook": 32, "block_size": 32,
    },

    # Phase 2: Block Size Sweep
    "e": {
        "name": "sweepE_ema_3bit_bs16",
        "desc": "EMA 3-bit bs=16 (very fine)",
        "use_ema": True, "n_codebook": 8, "block_size": 16,
    },
    "f": {
        "name": "sweepE_ema_3bit_bs32",
        "desc": "EMA 3-bit bs=32",
        "use_ema": True, "n_codebook": 8, "block_size": 32,
    },
    "g": {
        "name": "sweepE_ema_3bit_bs128",
        "desc": "EMA 3-bit bs=128 (coarse)",
        "use_ema": True, "n_codebook": 8, "block_size": 128,
    },

    # Phase 3: Research Ideas
    "h": {
        "name": "sweepE_ema_importance_aware",
        "desc": "EMA importance-aware (attn bs=32, ffn bs=64)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "block_size_attn": 32, "block_size_ffn": 64,
    },
    "i": {
        "name": "sweepE_ema_learned_dist",
        "desc": "EMA + learned distance metric",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "learned_distance": True,
    },
    "j": {
        "name": "sweepE_ste_factored_learned_dist",
        "desc": "STE factored + learned distance",
        "use_ema": False, "n_codebook": 8, "block_size": 64,
        "factored_codebook": True, "learned_distance": True,
        "codebook_lr": 3e-5,
    },
    "k": {
        "name": "sweepE_delayed_quant_ema",
        "desc": "Delayed quantization (500 float steps, then EMA)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "delay_quant_steps": 500,
    },
    "l": {
        "name": "sweepE_distillation_ema",
        "desc": "Codebook distillation (float teacher + EMA student)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "distill_alpha": 0.5, "distill_temp": 2.0,
    },

    # Phase 4: Combinations & Tuning
    "m": {
        "name": "sweepE_ema_factored_init",
        "desc": "EMA + factored init",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "factored_init": True,
    },
    "n": {
        "name": "sweepE_ema_entropy",
        "desc": "EMA + entropy reg (lambda=0.05)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "entropy_lambda": 0.05,
    },
    "o": {
        "name": "sweepE_ema_decay995",
        "desc": "EMA decay=0.995 (slower)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "ema_decay": 0.995,
    },
    "p": {
        "name": "sweepE_ema_decay999",
        "desc": "EMA decay=0.999 (very slow)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "ema_decay": 0.999,
    },
    "q": {
        "name": "sweepE_ema_hilr",
        "desc": "EMA + higher main LR (6e-4)",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "lr": 6e-4,
    },
    "r": {
        "name": "sweepE_ema_wd05",
        "desc": "EMA + weight decay=0.05",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "weight_decay": 0.05,
    },
    "s": {
        "name": "sweepE_ema_wd0",
        "desc": "EMA + weight decay=0.0",
        "use_ema": True, "n_codebook": 8, "block_size": 64,
        "weight_decay": 0.0,
    },

    # Phase 5: Best Combo Candidates
    "t": {
        "name": "sweepE_ema_bs32_decay995",
        "desc": "EMA 3-bit bs=32 decay=0.995 (best combo?)",
        "use_ema": True, "n_codebook": 8, "block_size": 32,
        "ema_decay": 0.995,
    },
}


def apply_config(config, overrides):
    """Apply experiment overrides to config object."""
    for key, value in overrides.items():
        if key in ("name", "desc"):
            continue
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(description="Phase E: Comprehensive Sweep")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only one experiment (a-t)")
    parser.add_argument("--from-exp", type=str, default=None,
                        help="Start from this experiment (skip earlier ones)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help="Run only experiments from this phase")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase groupings
    phases = {
        1: list("abcd"),
        2: list("efg"),
        3: list("hijkl"),
        4: list("mnopqrs"),
        5: list("t"),
    }

    # Determine which experiments to run
    if args.only:
        to_run = [args.only]
    elif args.phase:
        to_run = phases[args.phase]
    else:
        to_run = list(EXPERIMENTS.keys())

    # Skip experiments before --from-exp
    if args.from_exp:
        all_keys = list(EXPERIMENTS.keys())
        start_idx = all_keys.index(args.from_exp)
        to_run = [k for k in to_run if all_keys.index(k) >= start_idx]

    results_all = {}
    total_start = time.time()

    for key in to_run:
        exp = EXPERIMENTS[key]
        config = SweepConfig()
        if args.max_steps:
            config.max_steps = args.max_steps
        apply_config(config, exp)

        name = exp["name"]

        # Build model
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)
        cb_params = sum(p.numel() for n, p in model.named_parameters() if "codebook" in n)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"\n{'='*70}")
        print(f"  Experiment {key.upper()}: {exp['desc']}")
        print(f"  EMA: {config.use_ema}, n_codebook: {config.n_codebook}, "
              f"block_size: {config.block_size}")
        if config.block_size_attn:
            print(f"  block_size_attn: {config.block_size_attn}, "
                  f"block_size_ffn: {config.block_size_ffn}")
        if config.learned_distance:
            print(f"  learned_distance: True")
        if config.factored_codebook:
            print(f"  factored_codebook: True")
        if config.factored_init:
            print(f"  factored_init: True")
        if config.delay_quant_steps > 0:
            print(f"  delay_quant_steps: {config.delay_quant_steps}")
        if config.distill_alpha > 0:
            print(f"  distill_alpha: {config.distill_alpha}, "
                  f"distill_temp: {config.distill_temp}")
        print(f"  Codebook params: {cb_params:,}, Total params: {total_params:,}")
        print(f"  Steps: {config.max_steps}")
        print(f"{'='*70}\n")

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        results["elapsed_min"] = round(elapsed / 60, 1)
        results["cb_params"] = cb_params
        results["total_params"] = total_params
        results_all[key] = results

        print(f"\n  {name} finished in {elapsed/60:.1f} min")
        print(f"  Test PPL: {results['test_ppl']:.2f}")
        print(f"  Val PPL:  {results['val_ppl']:.2f}")
        if "aborted_at_step" in results:
            print(f"  ABORTED at step {results['aborted_at_step']}: "
                  f"{results['abort_reason']}")

        del model
        torch.cuda.empty_cache()

    # Summary
    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  PHASE E SWEEP RESULTS ({len(results_all)} experiments, {total_elapsed:.0f} min)")
    print(f"{'='*70}")
    print(f"  {'Exp':<3s} {'Description':<50s} {'Test PPL':>10s} {'Val PPL':>10s} {'CB Params':>10s} {'Time':>7s}")
    print(f"  {'-'*90}")

    # Sort by test PPL (best first)
    sorted_keys = sorted(results_all.keys(),
                         key=lambda k: results_all[k]["test_ppl"])
    for key in sorted_keys:
        exp = EXPERIMENTS[key]
        r = results_all[key]
        aborted = " ABORT" if "aborted_at_step" in r else ""
        print(f"  {key.upper():<3s} {exp['desc']:<50s} "
              f"{r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} "
              f"{r['cb_params']:>10,d} {r['elapsed_min']:>5.1f}m{aborted}")

    print(f"\n  Total time: {total_elapsed:.0f} minutes")
    print()


if __name__ == "__main__":
    main()
