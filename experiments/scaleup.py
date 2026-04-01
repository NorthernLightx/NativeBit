"""Scale-up experiment: MediumConfig + TinyStories with autoresearch findings.

Compares:
  A) Float baseline (upper bound)
  B) NativeBit 3-bit old (block_size=16, n_codebook=8, gradient codebooks)
  C) NativeBit 2-bit champion (autoresearch: n_codebook=4, block_size=128, EMA)
  D) NativeBit 3-bit EMA (autoresearch settings but 8 entries)
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


class BaseConfig:
    """MediumConfig architecture on TinyStories."""
    # Model (MediumConfig)
    n_layers: int = 6
    n_embd: int = 256
    n_head: int = 4
    ffn_hidden: int = 1024
    context_len: int = 512
    vocab_size: int = 50257

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 3e-4
    codebook_lr: float = 1e-3
    max_steps: int = 15000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 100
    log_every: int = 100
    checkpoint_every: int = 2000
    weight_decay: float = 0.05

    # Data
    dataset: str = "tinystories"
    seed: int = 42

    # NativeBit defaults
    block_size: int = 64
    n_codebook: int = 8
    use_ema: bool = False
    ema_decay: float = 0.99
    n_codebooks: int = 1
    factored_codebook: bool = False
    factored_init: bool = False
    learned_distance: bool = False

    # Gradient checkpointing — reduces VRAM at cost of ~30% more compute
    grad_checkpoint: bool = True

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
    diversity_lambda: float = 0.0
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 10.0
    delay_quant_steps: int = 0
    distill_alpha: float = 0.0
    distill_temp: float = 2.0
    quant_warmup_steps: int = 0
    quant_dropout: float = 0.0


class FloatConfig(BaseConfig):
    """A: Float baseline — no quantization."""
    grad_accum_steps: int = 4  # effective batch = 64


class NativeBit3bitOld(BaseConfig):
    """B: NativeBit 3-bit with old settings (previous TinyStories experiment)."""
    block_size: int = 16
    n_codebook: int = 8
    codebook_lr: float = 1e-3
    use_ema: bool = False


class NativeBit2bitChampion(BaseConfig):
    """C: NativeBit 2-bit — autoresearch champion config scaled up."""
    block_size: int = 128
    block_size_attn: int = 32
    n_codebook: int = 4       # 2-bit (only 4 entries!)
    use_ema: bool = True
    ema_decay: float = 0.9974  # best from autoresearch
    factored_init: bool = True
    lr: float = 6e-4           # higher LR worked better
    requant_every: int = 10    # cache quantized weights, requantize every 10 steps


class NativeBit3bitEMA(BaseConfig):
    """D: NativeBit 3-bit with autoresearch-style EMA settings."""
    block_size: int = 128
    block_size_attn: int = 32
    n_codebook: int = 8       # 3-bit (8 entries)
    use_ema: bool = True
    ema_decay: float = 0.9974
    factored_init: bool = True
    lr: float = 6e-4
    requant_every: int = 10


class NativeBitProgressive(BaseConfig):
    """E: Progressive merge — start 16 entries (4-bit), merge down to 4 (2-bit).

    Schedule over 30k steps:
      0-5000:     16 entries, free exploration
      5000-17000: merge one pair every 1000 steps (12 merges: 16->4)
      17000-30000: fine-tune at 4 entries (2-bit)
    """
    block_size: int = 128
    block_size_attn: int = 32
    n_codebook: int = 16      # Start at 4-bit (16 entries)
    use_ema: bool = True
    ema_decay: float = 0.9974
    factored_init: bool = True
    lr: float = 6e-4
    max_steps: int = 30000
    requant_every: int = 10

    # 12 merges from 16 down to 4 entries, evenly spaced 5000-17000
    merge_schedule: list = [5000, 6000, 7000, 8000, 9000, 10000,
                            11000, 12000, 13000, 14000, 15000, 16000]
    merge_min_active: int = 4  # Never go below 4 entries (2-bit floor)


EXPERIMENTS = {
    "float":        ("su_float_baseline",      FloatConfig,           False),
    "3bit-old":     ("su_nativebit_3bit_old",  NativeBit3bitOld,      True),
    "2bit-champ":   ("su_nativebit_2bit_champ", NativeBit2bitChampion, True),
    "3bit-ema":     ("su_nativebit_3bit_ema",  NativeBit3bitEMA,      True),
    "progressive":  ("su_nativebit_progressive", NativeBitProgressive, True),
}


def main():
    parser = argparse.ArgumentParser(description="Scale-up experiment: MediumConfig + TinyStories")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help="Run only one experiment")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_run = [args.only] if args.only else list(EXPERIMENTS.keys())
    results_table = []

    for key in to_run:
        exp_name, ConfigClass, use_nativebit = EXPERIMENTS[key]
        config = ConfigClass()
        if args.max_steps:
            config.max_steps = args.max_steps

        label = f"{key} ({'NativeBit' if use_nativebit else 'Float'})"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  Name: {exp_name}")
        if use_nativebit:
            bs_attn = getattr(config, 'block_size_attn', None)
            print(f"  n_codebook={config.n_codebook}, block_size={config.block_size}"
                  f"{f', bs_attn={bs_attn}' if bs_attn else ''}"
                  f", EMA={config.use_ema}"
                  f"{f', decay={config.ema_decay}' if config.use_ema else ''}"
                  f", factored_init={config.factored_init}")
        print(f"  lr={config.lr}, wd={config.weight_decay}, steps={config.max_steps}")
        print(f"{'='*70}\n")

        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=use_nativebit)

        # torch.compile: safe for float, skip for NativeBit (graph breaks)
        if not use_nativebit:
            try:
                model = torch.compile(model)
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile failed: {e}")

        t0 = time.time()
        try:
            res = train(model, config, device, exp_name, args.log_dir, args.data_dir)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM with batch_size={config.batch_size}, retrying with {config.batch_size // 2}")
                del model
                torch.cuda.empty_cache()
                config.batch_size //= 2
                set_seed(config.seed)
                model = build_model_from_config(config, use_nativebit=use_nativebit)
                t0 = time.time()
                res = train(model, config, device, exp_name, args.log_dir, args.data_dir)
            else:
                raise
        elapsed = time.time() - t0

        results_table.append({
            "name": key,
            "test_ppl": res["test_ppl"],
            "val_ppl": res["val_ppl"],
            "time_min": elapsed / 60,
        })

        print(f"\n  {key} done: test_ppl={res['test_ppl']:.2f}, val_ppl={res['val_ppl']:.2f}, {elapsed/60:.1f} min")

        del model
        torch.cuda.empty_cache()

    # Summary table
    if len(results_table) > 1:
        print(f"\n{'='*70}")
        print(f"  SCALE-UP RESULTS (MediumConfig + TinyStories, {config.max_steps} steps)")
        print(f"{'='*70}")
        print(f"  {'Config':<16} {'Test PPL':>10} {'Val PPL':>10} {'Time':>10}")
        print(f"  {'-'*50}")
        for r in results_table:
            print(f"  {r['name']:<16} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} {r['time_min']:>8.1f} min")
        print()


if __name__ == "__main__":
    main()
