"""Phase D: Factored Codebooks — cb[b,e] = scale[b] * global_cb[e] + offset[b].

Shares a global codebook pattern across all blocks with per-block affine adjustment.
Reduces codebook parameters ~4x while (hopefully) preserving quality.

Experiments:
  A. STE full codebook (baseline) — standard per-block codebook
  B. STE factored codebook — same training, 4x fewer codebook params
  C. EMA full codebook — EMA baseline (best known method)
  D. STE factored + lower codebook LR — factored may need different LR

Success criteria:
  - Factored PPL within 5% of full codebook at same bit width
  - Codebook param count reduced ~4x
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


class PhaseDConfig:
    """WikiText-2, SmallConfig-sized, quick iteration."""
    # Model
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    vocab_size: int = 50257

    # NativeBit
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
    n_codebooks: int = 1

    # EMA (overridden per experiment)
    use_ema: bool = False
    ema_decay: float = 0.99

    # Factored (overridden per experiment)
    factored_codebook: bool = False

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 15.0


EXPERIMENTS = {
    "a": {
        "name": "phaseD_ste_full",
        "desc": "STE full codebook (baseline)",
        "use_ema": False,
        "factored_codebook": False,
        "codebook_lr": 3e-5,
    },
    "b": {
        "name": "phaseD_ste_factored",
        "desc": "STE factored codebook (4x fewer cb params)",
        "use_ema": False,
        "factored_codebook": True,
        "codebook_lr": 3e-5,
    },
    "c": {
        "name": "phaseD_ema_full",
        "desc": "EMA full codebook (best known method)",
        "use_ema": True,
        "factored_codebook": False,
        "codebook_lr": 3e-5,
    },
    "d": {
        "name": "phaseD_ste_factored_hicblr",
        "desc": "STE factored + higher codebook LR (1e-4)",
        "use_ema": False,
        "factored_codebook": True,
        "codebook_lr": 1e-4,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase D: Factored Codebooks")
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
        config = PhaseDConfig()
        if args.max_steps:
            config.max_steps = args.max_steps

        config.use_ema = exp["use_ema"]
        config.factored_codebook = exp["factored_codebook"]
        config.codebook_lr = exp["codebook_lr"]

        name = exp["name"]

        # Count codebook params for display
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)
        cb_params = sum(p.numel() for n, p in model.named_parameters() if "codebook" in n)

        print(f"\n{'='*70}")
        print(f"  Experiment {key.upper()}: {exp['desc']}")
        print(f"  EMA: {config.use_ema}, Factored: {config.factored_codebook}, "
              f"CB LR: {config.codebook_lr}")
        print(f"  Codebook params: {cb_params:,}")
        print(f"  Steps: {config.max_steps}")
        print(f"{'='*70}\n")

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        results["elapsed_min"] = round(elapsed / 60, 1)
        results["cb_params"] = cb_params
        results_all[key] = results

        print(f"\n  {name} finished in {elapsed/60:.1f} min")
        print(f"  Test PPL: {results['test_ppl']:.2f}")
        print(f"  Val PPL:  {results['val_ppl']:.2f}")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  PHASE D RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<45s} {'Test PPL':>10s} {'Val PPL':>10s} {'CB Params':>10s} {'Time':>8s}")
    print(f"  {'-'*83}")
    for key in to_run:
        exp = EXPERIMENTS[key]
        r = results_all[key]
        print(f"  {exp['desc']:<45s} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['cb_params']:>10,d} {r['elapsed_min']:>6.1f}m")
    print()


if __name__ == "__main__":
    main()
