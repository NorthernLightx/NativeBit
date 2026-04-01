"""Phase C: Residual/Additive Quantization — weight ≈ cb1[i] + cb2[j].

Two small codebooks give multiplicative precision:
  8×8 = 64 effective values from two 3-bit codebooks (6-bit effective).

Experiments:
  A. STE + single codebook, 8 entries (3-bit) — baseline
  B. STE + residual (2 codebooks), 8+8 entries — 6-bit effective
  C. EMA + single codebook, 8 entries (3-bit) — EMA baseline
  D. EMA + residual (2 codebooks), 8+8 entries — KEY TEST: EMA + residual
  E. EMA + residual, 4+4 entries — 4-bit effective from two 2-bit codebooks

Success criteria:
  - Residual quantization improves PPL over single codebook (same total entries)
  - EMA + residual is best combination
  - Small codebooks (4+4) are viable
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


class PhaseCConfig:
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

    # Disabled features
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

    # EMA (overridden per experiment)
    use_ema: bool = False
    ema_decay: float = 0.99

    # Residual (overridden per experiment)
    n_codebooks: int = 1

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 15.0


EXPERIMENTS = {
    "a": {
        "name": "phaseC_ste_single_3bit",
        "desc": "STE single codebook 8-entry (3-bit) — baseline",
        "use_ema": False,
        "n_codebook": 8,
        "n_codebooks": 1,
    },
    "b": {
        "name": "phaseC_ste_residual_3bit",
        "desc": "STE residual 8+8 entries (6-bit effective)",
        "use_ema": False,
        "n_codebook": 8,
        "n_codebooks": 2,
    },
    "c": {
        "name": "phaseC_ema_single_3bit",
        "desc": "EMA single codebook 8-entry (3-bit) — EMA baseline",
        "use_ema": True,
        "n_codebook": 8,
        "n_codebooks": 1,
    },
    "d": {
        "name": "phaseC_ema_residual_3bit",
        "desc": "EMA residual 8+8 (6-bit effective) — KEY TEST",
        "use_ema": True,
        "n_codebook": 8,
        "n_codebooks": 2,
    },
    "e": {
        "name": "phaseC_ema_residual_2bit",
        "desc": "EMA residual 4+4 (4-bit effective from 2-bit cbs)",
        "use_ema": True,
        "n_codebook": 4,
        "n_codebooks": 2,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase C: Residual Quantization")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=list(EXPERIMENTS.keys()),
                        help="Run only one experiment (a/b/c/d/e)")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_run = [args.only] if args.only else list(EXPERIMENTS.keys())
    results_all = {}

    for key in to_run:
        exp = EXPERIMENTS[key]
        config = PhaseCConfig()
        if args.max_steps:
            config.max_steps = args.max_steps

        # Apply experiment overrides
        config.use_ema = exp["use_ema"]
        config.n_codebook = exp["n_codebook"]
        config.n_codebooks = exp["n_codebooks"]

        name = exp["name"]
        n_eff = config.n_codebook ** config.n_codebooks
        eff_bits = f"{n_eff} effective values"
        print(f"\n{'='*70}")
        print(f"  Experiment {key.upper()}: {exp['desc']}")
        print(f"  EMA: {config.use_ema}, Codebooks: {config.n_codebooks}, "
              f"Entries/cb: {config.n_codebook}, {eff_bits}")
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
    print("  PHASE C RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<50s} {'Test PPL':>10s} {'Val PPL':>10s} {'Time':>8s}")
    print(f"  {'-'*78}")
    for key in to_run:
        exp = EXPERIMENTS[key]
        r = results_all[key]
        print(f"  {exp['desc']:<50s} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['elapsed_min']:>6.1f}m")
    print()


if __name__ == "__main__":
    main()
