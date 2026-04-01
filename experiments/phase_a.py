"""Phase A: Fix dead entries with Gumbel-Softmax + Diversity Loss.

Experiments:
  A. STE baseline (3-bit) — current approach, for comparison
  B. Gumbel-Softmax (3-bit) — should reduce dead entries
  C. Gumbel-Softmax + Diversity Loss (3-bit) — should eliminate dead entries
  D. Gumbel-Softmax + Diversity Loss (4-bit) — the key test, previously broken

Success criteria:
  - Dead entries < 1% throughout training (esp. for 4-bit)
  - 4-bit PPL beats 3-bit PPL (unlocking the extra bit width)
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


class PhaseAConfig:
    """WikiText-2, SmallConfig-sized, quick iteration."""
    # Model
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    vocab_size: int = 50257

    # NativeBit (overridden per experiment)
    block_size: int = 64
    n_codebook: int = 8  # 3-bit default

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

    # Seed
    seed: int = 42

    # Phase 2 fields (disabled)
    entropy_lambda: float = 0.0
    entropy_temperature: float = 0.01
    progressive: bool = False
    merge_util_threshold: float = 0.02
    merge_dist_threshold = None
    merge_steps = None
    tau_start: float = 0.0
    tau_end: float = 0.01
    tau_anneal_steps: int = 3000

    # Phase A defaults (overridden per experiment)
    quantize_mode: str = "ste"
    gumbel_tau_start: float = 1.0
    gumbel_tau_end: float = 0.1
    gumbel_tau_anneal_steps: int = 0  # 0 = use max_steps
    diversity_lambda: float = 0.0

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 15.0


EXPERIMENTS = {
    "a": {
        "name": "phaseA_ste_3bit",
        "desc": "STE baseline (3-bit)",
        "quantize_mode": "ste",
        "n_codebook": 8,
        "diversity_lambda": 0.0,
    },
    "b": {
        "name": "phaseA_ste_div_3bit",
        "desc": "STE + Diversity Loss (3-bit)",
        "quantize_mode": "ste",
        "n_codebook": 8,
        "diversity_lambda": 0.5,
    },
    "c": {
        "name": "phaseA_gumbel_low_3bit",
        "desc": "Gumbel-Softmax low-tau (3-bit)",
        "quantize_mode": "gumbel",
        "n_codebook": 8,
        "diversity_lambda": 0.0,
        "gumbel_tau_start": 0.3,
        "gumbel_tau_end": 0.05,
    },
    "d": {
        "name": "phaseA_gumbel_low_div_3bit",
        "desc": "Gumbel low-tau + Diversity (3-bit)",
        "quantize_mode": "gumbel",
        "n_codebook": 8,
        "diversity_lambda": 0.5,
        "gumbel_tau_start": 0.3,
        "gumbel_tau_end": 0.05,
    },
    "e": {
        "name": "phaseA_ste_div_4bit",
        "desc": "STE + Diversity Loss (4-bit)",
        "quantize_mode": "ste",
        "n_codebook": 16,
        "diversity_lambda": 0.5,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase A: Gumbel-Softmax + Diversity Loss")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=list(EXPERIMENTS.keys()),
                        help="Run only one experiment (a/b/c/d)")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_run = [args.only] if args.only else list(EXPERIMENTS.keys())
    results_all = {}

    for key in to_run:
        exp = EXPERIMENTS[key]
        config = PhaseAConfig()
        if args.max_steps:
            config.max_steps = args.max_steps

        # Apply experiment-specific overrides
        config.quantize_mode = exp["quantize_mode"]
        config.n_codebook = exp["n_codebook"]
        config.diversity_lambda = exp["diversity_lambda"]
        if "gumbel_tau_start" in exp:
            config.gumbel_tau_start = exp["gumbel_tau_start"]
        if "gumbel_tau_end" in exp:
            config.gumbel_tau_end = exp["gumbel_tau_end"]
        if config.gumbel_tau_anneal_steps == 0:
            config.gumbel_tau_anneal_steps = config.max_steps

        name = exp["name"]
        print(f"\n{'='*70}")
        print(f"  Experiment {key.upper()}: {exp['desc']}")
        print(f"  Mode: {config.quantize_mode}, Entries: {config.n_codebook}, "
              f"Diversity: {config.diversity_lambda}")
        print(f"  Steps: {config.max_steps}")
        print(f"{'='*70}\n")

        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        results["elapsed_min"] = round(elapsed / 60, 1)
        results_all[key] = results

        print(f"\n  {name} finished in {elapsed/60:.1f} min")
        print(f"  Test PPL: {results['test_ppl']:.2f}")
        print(f"  Val PPL:  {results['val_ppl']:.2f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  PHASE A RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<40s} {'Test PPL':>10s} {'Val PPL':>10s} {'Time':>8s}")
    print(f"  {'-'*68}")
    for key in to_run:
        exp = EXPERIMENTS[key]
        r = results_all[key]
        print(f"  {exp['desc']:<40s} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['elapsed_min']:>6.1f}m")
    print()


if __name__ == "__main__":
    main()
