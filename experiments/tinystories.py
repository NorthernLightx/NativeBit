"""Train float baseline + NativeBit 3-bit on TinyStories for readable text generation."""

import os
import sys
import math
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


class TinyStoriesConfig:
    """MediumConfig tuned for TinyStories dataset."""
    # Model (same as MediumConfig)
    n_layers: int = 6
    n_embd: int = 256
    n_head: int = 4
    ffn_hidden: int = 1024
    context_len: int = 512
    vocab_size: int = 50257

    # NativeBit (best from Phase 2b)
    block_size: int = 16
    n_codebook: int = 8     # 3-bit

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 4  # effective batch = 64
    lr: float = 3e-4
    codebook_lr: float = 1e-3
    max_steps: int = 15000   # More steps — more data to learn from
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    requant_every: int = 1  # recompute indices every N steps (1=every step)
    revive_every: int = 100
    log_every: int = 100
    checkpoint_every: int = 2000

    # Data
    dataset: str = "tinystories"

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

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 10.0


def main():
    parser = argparse.ArgumentParser(description="Train on TinyStories")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=["float", "nativebit"],
                        help="Run only one experiment")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TinyStoriesConfig()
    if args.max_steps:
        config.max_steps = args.max_steps

    experiments = []
    if args.only != "nativebit":
        experiments.append(("ts_float", False))
    if args.only != "float":
        experiments.append(("ts_nativebit_3bit", True))

    for name, use_nativebit in experiments:
        print(f"\n{'='*70}")
        print(f"  {name} ({'NativeBit 3-bit' if use_nativebit else 'Float baseline'})")
        print(f"  Dataset: TinyStories (~12M tokens)")
        print(f"  Steps: {config.max_steps}")
        print(f"{'='*70}\n")

        set_seed(config.seed)

        # NativeBit is already slow — skip grad accum to keep wall time reasonable
        if use_nativebit:
            config.grad_accum_steps = 1

        model = build_model_from_config(config, use_nativebit=use_nativebit)

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        print(f"\n  {name} finished in {elapsed/60:.1f} min")
        print(f"  Test PPL: {results['test_ppl']:.2f}")
        print(f"  Val PPL:  {results['val_ppl']:.2f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    print("\nDone! Models available in Arena at http://localhost:8765")


if __name__ == "__main__":
    main()
