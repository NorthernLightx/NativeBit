"""3-phase acceptance protocol: screen → validate → confirm."""

import math
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

import torch

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from train import train

from .trial import Trial


# Phase step counts
SCREEN_STEPS = 1500
VALIDATE_STEPS = 5000
CONFIRM_SEEDS = [42, 123, 7]

# Acceptance thresholds (multiplier on current best PPL)
SCREEN_THRESHOLD = 1.05    # 5% slack for quick screen
VALIDATE_THRESHOLD = 1.00  # must beat current best


class TrialConfig:
    """Dynamic config object built from autoresearch parameters.

    Mirrors DoEConfig from doe_sweep.py — train.py reads attributes from this.
    """

    # Model (fixed)
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    vocab_size: int = 50257

    # NativeBit (varied)
    block_size: int = 64
    n_codebook: int = 8

    # Training
    batch_size: int = 32
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

    # EMA (always on)
    use_ema: bool = True
    ema_decay: float = 0.99

    # Defaults for features we may toggle
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
    factored_codebook: bool = False
    learned_distance: bool = False
    factored_init: bool = False
    block_size_attn = None
    block_size_ffn = None
    weight_decay: float = 0.01
    delay_quant_steps: int = 0
    distill_alpha: float = 0.0
    distill_temp: float = 2.0

    # Per-layer codebook precision
    n_codebook_attn = None  # None = use n_codebook
    n_codebook_ffn = None   # None = use n_codebook

    # Smooth quantization warmup
    quant_warmup_steps: int = 0

    # Stochastic quantization dropout
    quant_dropout: float = 0.0

    # Health checks (disabled for autoresearch — we use phase thresholds instead)
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 50.0  # lenient — let the phase protocol decide


def build_trial_config(params: dict, max_steps: int, seed: int = 42) -> TrialConfig:
    """Build a TrialConfig from autoresearch parameter dict."""
    cfg = TrialConfig()
    cfg.max_steps = max_steps
    cfg.seed = seed

    # Map autoresearch params to config attributes
    cfg.n_codebook = params.get("n_codebook", 8)
    cfg.block_size = params.get("block_size", 64)
    cfg.ema_decay = params.get("ema_decay", 0.99)
    cfg.lr = params.get("learning_rate", 3e-4)
    cfg.weight_decay = params.get("weight_decay", 0.1)
    # Cap delay to 20% of training — higher causes loss spikes when quant kicks in
    raw_delay = params.get("delay_quant_steps", 0)
    cfg.delay_quant_steps = min(raw_delay, max_steps // 5)
    cfg.entropy_lambda = params.get("entropy_lambda", 0.0)
    cfg.factored_init = params.get("factored_init", False)
    cfg.distill_alpha = params.get("distill_alpha", 0.0)
    cfg.distill_temp = params.get("distill_temp", 2.0)

    cfg.learned_distance = params.get("learned_distance", False)
    cfg.n_codebooks = params.get("n_codebooks", 1)

    # Per-layer codebook precision
    cfg.n_codebook_attn = params.get("n_codebook_attn", None)
    cfg.n_codebook_ffn = params.get("n_codebook_ffn", None)

    # Smooth quantization warmup
    cfg.quant_warmup_steps = params.get("quant_warmup_steps", 0)
    cfg.quant_dropout = params.get("quant_dropout", 0.0)

    # block_size_attn: None means use block_size
    bs_attn = params.get("block_size_attn", None)
    if bs_attn is not None:
        cfg.block_size_attn = bs_attn
        cfg.block_size_ffn = cfg.block_size
    else:
        cfg.block_size_attn = None
        cfg.block_size_ffn = None

    # Reduce batch size for distillation or residual quantization (more VRAM)
    if cfg.distill_alpha > 0:
        cfg.batch_size = 16
    elif cfg.n_codebooks > 1:
        cfg.batch_size = 24

    return cfg


def _run_training(params: dict, max_steps: int, seed: int,
                  log_dir: str, data_dir: str, name: str) -> dict:
    """Run a single training trial. Returns results dict from train().

    Handles OOM by retrying with smaller batch size.
    """
    cfg = build_trial_config(params, max_steps, seed)

    set_seed(seed)
    model = build_model_from_config(cfg, use_nativebit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        results = train(model, cfg, device, name, log_dir, data_dir)
    except torch.cuda.OutOfMemoryError:
        # Retry with half batch size
        del model
        torch.cuda.empty_cache()
        print(f"  OOM with batch_size={cfg.batch_size}, retrying with {cfg.batch_size // 2}")
        cfg.batch_size = max(8, cfg.batch_size // 2)
        set_seed(seed)
        model = build_model_from_config(cfg, use_nativebit=True)
        try:
            results = train(model, cfg, device, name, log_dir, data_dir)
        except torch.cuda.OutOfMemoryError:
            del model
            torch.cuda.empty_cache()
            return {
                "test_ppl": float("inf"),
                "val_ppl": float("inf"),
                "train_loss": float("inf"),
                "aborted_at_step": 0,
                "abort_reason": "OOM even at batch_size=8",
            }
    finally:
        # Always clean up GPU memory
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()

    return results


def evaluate_trial(trial: Trial, best_ppl: float,
                   log_dir: str, data_dir: str,
                   best_screen_ppl: float = None) -> Trial:
    """Run the 3-phase acceptance protocol on a trial.

    Args:
        best_ppl: Best confirmed PPL (used for validate and confirm thresholds).
        best_screen_ppl: Champion's screen-phase PPL (used for screen threshold).
            If None, falls back to best_ppl (original behavior).

    Modifies trial in-place and returns it.
    """
    params = trial.config
    base_name = f"ar_trial{trial.trial_id:04d}"

    # ── Phase 1: Screen (1500 steps) ──────────────────────────────────────
    trial.status = "screening"
    print(f"\n  Phase 1/3: SCREEN ({SCREEN_STEPS} steps)")

    t0 = time.time()
    results = _run_training(
        params, SCREEN_STEPS, trial.seed,
        log_dir, data_dir, f"{base_name}_screen",
    )
    trial.screen_time = time.time() - t0
    trial.screen_ppl = results.get("test_ppl", float("inf"))

    if results.get("aborted_at_step") is not None:
        trial.aborted = True
        trial.abort_reason = results.get("abort_reason", "aborted")

    # If screen PPL is infinite (OOM, diverged), always reject — unless this
    # is the very first trial (best_ppl == inf), in which case skip screen
    # and let validate decide
    if math.isinf(trial.screen_ppl) and not math.isinf(best_ppl):
        trial.status = "rejected"
        trial.reject_phase = "screen"
        print(f"  Screen PPL: inf — REJECTED (no valid result)")
        return trial

    # Use champion's screen PPL for screen threshold (not confirm PPL, which is
    # from 5000 steps and thus much lower — comparing 1500-step PPL against
    # 5000-step PPL would reject every config including the champion itself)
    screen_base = best_screen_ppl if best_screen_ppl is not None and not math.isinf(best_screen_ppl) else best_ppl
    threshold = screen_base * SCREEN_THRESHOLD if not math.isinf(screen_base) else float("inf")
    print(f"  Screen PPL: {trial.screen_ppl:.2f} (threshold: {threshold:.2f}, base: {screen_base:.2f})")

    if trial.screen_ppl > threshold:
        trial.status = "rejected"
        trial.reject_phase = "screen"
        print(f"  REJECTED at screen phase")
        return trial

    # ── Phase 2: Validate (5000 steps) ────────────────────────────────────
    trial.status = "validating"
    print(f"\n  Phase 2/3: VALIDATE ({VALIDATE_STEPS} steps)")

    t0 = time.time()
    results = _run_training(
        params, VALIDATE_STEPS, trial.seed,
        log_dir, data_dir, f"{base_name}_validate",
    )
    trial.validate_time = time.time() - t0
    trial.validate_ppl = results.get("test_ppl", float("inf"))

    threshold = best_ppl * VALIDATE_THRESHOLD
    print(f"  Validate PPL: {trial.validate_ppl:.2f} (threshold: {threshold:.2f})")

    if trial.validate_ppl > threshold:
        trial.status = "rejected"
        trial.reject_phase = "validate"
        print(f"  REJECTED at validate phase")
        return trial

    # ── Phase 3: Confirm (5000 steps × 3 seeds) ──────────────────────────
    trial.status = "confirming"
    print(f"\n  Phase 3/3: CONFIRM ({VALIDATE_STEPS} steps × {len(CONFIRM_SEEDS)} seeds)")

    seed_ppls = []
    t0 = time.time()
    for i, seed in enumerate(CONFIRM_SEEDS):
        print(f"    Seed {seed} ({i+1}/{len(CONFIRM_SEEDS)})...")
        results = _run_training(
            params, VALIDATE_STEPS, seed,
            log_dir, data_dir, f"{base_name}_confirm_s{seed}",
        )
        ppl = results.get("test_ppl", float("inf"))
        seed_ppls.append(ppl)
        print(f"    Seed {seed}: PPL={ppl:.2f}")
    trial.confirm_time = time.time() - t0

    trial.confirm_ppls = seed_ppls
    trial.confirm_ppl = sum(seed_ppls) / len(seed_ppls)
    trial.confirm_std = (sum((p - trial.confirm_ppl) ** 2 for p in seed_ppls) / len(seed_ppls)) ** 0.5

    # Check: mean < best AND std < 5% of mean
    std_ok = trial.confirm_std < 0.05 * trial.confirm_ppl
    mean_ok = trial.confirm_ppl < best_ppl

    print(f"  Confirm: mean={trial.confirm_ppl:.2f}, std={trial.confirm_std:.2f}")
    print(f"    mean < best ({best_ppl:.2f})? {'YES' if mean_ok else 'NO'}")
    print(f"    std < 5% of mean? {'YES' if std_ok else 'NO'}")

    if mean_ok and std_ok:
        trial.status = "accepted"
        print(f"  ACCEPTED — new champion!")
    else:
        trial.status = "rejected"
        trial.reject_phase = "confirm"
        print(f"  REJECTED at confirm phase")

    return trial
