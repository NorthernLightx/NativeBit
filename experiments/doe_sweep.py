"""Design of Experiments (DoE) sweep for NativeBit hyperparameters.

Uses a two-phase approach:
  Phase 1 — Plackett-Burman screening (12 runs) to identify which factors matter
  Phase 2 — Full factorial on significant factors (auto-generated after Phase 1)

Factors (all with EMA=True, our best method):
  A. n_codebook       : 4 (2-bit) vs 16 (4-bit)       center=8 (3-bit)
  B. block_size       : 32 vs 128                      center=64
  C. ema_decay        : 0.99 vs 0.999                  center=0.995
  D. learned_distance : False vs True
  E. factored_init    : False vs True
  F. weight_decay     : 0.0 vs 0.05                    center=0.01
  G. lr               : 3e-4 vs 6e-4                   center=4.5e-4
  H. delay_quant_steps: 0 vs 500                       center=250
  I. distill_alpha    : 0.0 vs 0.5                     center=0.25
  J. entropy_lambda   : 0.0 vs 0.05                    center=0.025
  K. block_size_attn  : None(=block_size) vs 32        (importance-aware)

Response: Test PPL (lower is better)
"""

import os
import sys
import time
import json
import math

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


# ── Factor definitions ──────────────────────────────────────────────────────

FACTORS = {
    "A_n_codebook":       {"low": 4,     "high": 16,    "center": 8},
    "B_block_size":       {"low": 32,    "high": 128,   "center": 64},
    "C_ema_decay":        {"low": 0.99,  "high": 0.999, "center": 0.995},
    "D_learned_distance": {"low": False, "high": True,  "center": False},
    "E_factored_init":    {"low": False, "high": True,  "center": False},
    "F_weight_decay":     {"low": 0.0,   "high": 0.05,  "center": 0.01},
    "G_lr":               {"low": 3e-4,  "high": 6e-4,  "center": 4.5e-4},
    "H_delay_quant":      {"low": 0,     "high": 500,   "center": 250},
    "I_distill_alpha":    {"low": 0.0,   "high": 0.5,   "center": 0.25},
    "J_entropy_lambda":   {"low": 0.0,   "high": 0.05,  "center": 0.025},
    "K_block_size_attn":  {"low": 0,     "high": 32,    "center": 0},
    # low=0 means block_size_attn=None (use default block_size)
}

# Plackett-Burman design matrix for 12 runs, 11 factors
# Standard PB12 generator row: + + - + + + - - - + -
# Each row is a run, each column is a factor: +1 = high, -1 = low
PB12_MATRIX = [
    # A   B   C   D   E   F   G   H   I   J   K
    [+1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1],  # run 1
    [-1, +1, +1, -1, +1, +1, +1, -1, -1, -1, +1],  # run 2
    [+1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1],  # run 3
    [-1, +1, -1, +1, +1, -1, +1, +1, +1, -1, -1],  # run 4
    [-1, -1, +1, -1, +1, +1, -1, +1, +1, +1, -1],  # run 5
    [-1, -1, -1, +1, -1, +1, +1, -1, +1, +1, +1],  # run 6
    [+1, -1, -1, -1, +1, -1, +1, +1, -1, +1, +1],  # run 7
    [+1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1],  # run 8
    [+1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1],  # run 9 (fixed)
    [-1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1],  # run 10
    [+1, -1, +1, +1, +1, -1, -1, -1, +1, -1, +1],  # run 11
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # run 12 (all low)
]


class DoEConfig:
    """Base config — modified per run by DoE factor levels."""
    # Model (fixed)
    n_layers: int = 4
    n_embd: int = 128
    n_head: int = 4
    ffn_hidden: int = 512
    context_len: int = 256
    vocab_size: int = 50257

    # NativeBit (varied by DoE)
    block_size: int = 64
    n_codebook: int = 8

    # Training (some varied by DoE)
    batch_size: int = 32  # optimal for RTX 3070 (1.55x faster than bs=16)
    lr: float = 3e-4
    codebook_lr: float = 3e-5  # unused with EMA but kept for param groups
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 100
    log_every: int = 50

    # Data
    dataset: str = "wikitext-2"
    seed: int = 42

    # Fixed: EMA is our best method
    use_ema: bool = True
    ema_decay: float = 0.99

    # Disabled defaults (overridden per run)
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

    # Health checks
    health_check_steps = None
    health_max_ppl = None
    health_max_dead_pct: float = 20.0


def decode_run(run_vector):
    """Convert a PB design row (+1/-1 per factor) into config overrides."""
    factor_names = list(FACTORS.keys())
    overrides = {}
    for i, level in enumerate(run_vector):
        factor = FACTORS[factor_names[i]]
        value = factor["high"] if level > 0 else factor["low"]
        overrides[factor_names[i]] = value
    return overrides


def apply_overrides(config, overrides):
    """Apply DoE factor values to a config object."""
    config.n_codebook = overrides["A_n_codebook"]
    config.block_size = overrides["B_block_size"]
    config.ema_decay = overrides["C_ema_decay"]
    config.learned_distance = overrides["D_learned_distance"]
    config.factored_init = overrides["E_factored_init"]
    config.weight_decay = overrides["F_weight_decay"]
    config.lr = overrides["G_lr"]
    config.delay_quant_steps = overrides["H_delay_quant"]
    config.distill_alpha = overrides["I_distill_alpha"]
    config.entropy_lambda = overrides["J_entropy_lambda"]

    bs_attn_val = overrides["K_block_size_attn"]
    if bs_attn_val > 0:
        config.block_size_attn = bs_attn_val
        config.block_size_ffn = config.block_size  # FFN uses default
    else:
        config.block_size_attn = None
        config.block_size_ffn = None


def run_desc(overrides):
    """Human-readable description of a DoE run."""
    parts = []
    parts.append(f"cb={overrides['A_n_codebook']}")
    parts.append(f"bs={overrides['B_block_size']}")
    parts.append(f"decay={overrides['C_ema_decay']}")
    if overrides["D_learned_distance"]:
        parts.append("lrn_dist")
    if overrides["E_factored_init"]:
        parts.append("fac_init")
    parts.append(f"wd={overrides['F_weight_decay']}")
    parts.append(f"lr={overrides['G_lr']:.0e}")
    if overrides["H_delay_quant"] > 0:
        parts.append(f"delay={overrides['H_delay_quant']}")
    if overrides["I_distill_alpha"] > 0:
        parts.append(f"distill={overrides['I_distill_alpha']}")
    if overrides["J_entropy_lambda"] > 0:
        parts.append(f"ent={overrides['J_entropy_lambda']}")
    if overrides["K_block_size_attn"] > 0:
        parts.append(f"bs_attn={overrides['K_block_size_attn']}")
    return " | ".join(parts)


def analyze_main_effects(results, design_matrix):
    """Compute main effect of each factor on test PPL.

    Main effect = mean(response at high) - mean(response at low).
    Negative = high level is better (lower PPL).
    """
    factor_names = list(FACTORS.keys())
    n_factors = len(factor_names)
    effects = {}

    ppls = [r["test_ppl"] for r in results]

    for j in range(n_factors):
        high_ppls = [ppls[i] for i in range(len(ppls)) if design_matrix[i][j] > 0]
        low_ppls = [ppls[i] for i in range(len(ppls)) if design_matrix[i][j] < 0]
        effect = sum(high_ppls) / len(high_ppls) - sum(low_ppls) / len(low_ppls)
        effects[factor_names[j]] = round(effect, 2)

    return effects


def main():
    parser = argparse.ArgumentParser(description="DoE Sweep for NativeBit")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1=screening (PB12), 2=optimization (auto from phase 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this run number (1-indexed)")
    parser.add_argument("--from-run", type=int, default=None,
                        help="Resume from this run number (1-indexed)")
    parser.add_argument("--center-points", type=int, default=2,
                        help="Number of center-point replicates (for curvature)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.phase == 1:
        run_screening(args, device)
    elif args.phase == 2:
        run_optimization(args, device)


def run_screening(args, device):
    """Phase 1: Plackett-Burman screening design + center points."""
    design = list(PB12_MATRIX)

    # Add center points for curvature detection
    n_center = args.center_points
    center_vector = [0] * len(FACTORS)  # 0 = center level
    for _ in range(n_center):
        design.append(center_vector)

    total_runs = len(design)

    # Determine which runs to execute
    if args.only:
        run_indices = [args.only - 1]
    elif args.from_run:
        run_indices = list(range(args.from_run - 1, total_runs))
    else:
        run_indices = list(range(total_runs))

    results_all = []
    results_file = os.path.join(args.log_dir, "doe_screening_results.json")

    # Load existing results if resuming
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_all = json.load(f)
        print(f"Loaded {len(results_all)} existing results from {results_file}")

    total_start = time.time()

    for run_idx in run_indices:
        run_num = run_idx + 1
        row = design[run_idx]
        is_center = all(v == 0 for v in row)

        # Decode factor levels
        if is_center:
            factor_names = list(FACTORS.keys())
            overrides = {name: FACTORS[name]["center"] for name in factor_names}
            name = f"doe_center_{run_num}"
            desc = "CENTER POINT"
        else:
            overrides = decode_run(row)
            name = f"doe_run{run_num:02d}"
            desc = run_desc(overrides)

        # Skip if already completed
        already_done = any(r["run_num"] == run_num for r in results_all)
        if already_done:
            print(f"  Run {run_num}/{total_runs} already completed, skipping")
            continue

        # Build config
        config = DoEConfig()
        if args.max_steps:
            config.max_steps = args.max_steps
        apply_overrides(config, overrides)

        # Build model
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)
        cb_params = sum(p.numel() for n, p in model.named_parameters() if "codebook" in n)

        print(f"\n{'='*70}")
        print(f"  DoE Run {run_num}/{total_runs}: {desc}")
        print(f"  {'CENTER POINT' if is_center else 'PB Design Row: ' + str(row)}")
        print(f"  n_codebook={config.n_codebook}, block_size={config.block_size}, "
              f"ema_decay={config.ema_decay}")
        print(f"  lr={config.lr:.0e}, wd={config.weight_decay}, "
              f"delay={config.delay_quant_steps}")
        if config.learned_distance:
            print(f"  learned_distance=True")
        if config.factored_init:
            print(f"  factored_init=True")
        if config.distill_alpha > 0:
            print(f"  distill_alpha={config.distill_alpha}")
        if config.entropy_lambda > 0:
            print(f"  entropy_lambda={config.entropy_lambda}")
        if config.block_size_attn:
            print(f"  block_size_attn={config.block_size_attn}")
        print(f"  Codebook params: {cb_params:,}")
        print(f"  Steps: {config.max_steps}")
        print(f"{'='*70}\n")

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        result_record = {
            "run_num": run_num,
            "name": name,
            "desc": desc,
            "is_center": is_center,
            "design_row": row,
            "overrides": {k: v if not isinstance(v, bool) else int(v) for k, v in overrides.items()},
            "test_ppl": results["test_ppl"],
            "val_ppl": results["val_ppl"],
            "train_loss": results["train_loss"],
            "elapsed_min": round(elapsed / 60, 1),
            "cb_params": cb_params,
        }
        if "aborted_at_step" in results:
            result_record["aborted"] = True
            result_record["abort_reason"] = results["abort_reason"]

        results_all.append(result_record)

        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(results_all, f, indent=2)

        print(f"\n  Run {run_num} finished in {elapsed/60:.1f} min")
        print(f"  Test PPL: {results['test_ppl']:.2f}, Val PPL: {results['val_ppl']:.2f}")

        del model
        torch.cuda.empty_cache()

    # ── Analysis ────────────────────────────────────────────────────────────
    total_elapsed = (time.time() - total_start) / 60

    # Separate PB runs from center points
    pb_results = [r for r in results_all if not r["is_center"]]
    center_results = [r for r in results_all if r["is_center"]]

    if len(pb_results) == 12:
        pb_design = [r["design_row"] for r in pb_results]
        effects = analyze_main_effects(pb_results, pb_design)

        print(f"\n{'='*70}")
        print(f"  DoE SCREENING RESULTS ({len(results_all)} runs, {total_elapsed:.0f} min)")
        print(f"{'='*70}")

        # Results table
        print(f"\n  {'Run':>4s} {'Description':<55s} {'Test PPL':>10s} {'Time':>7s}")
        print(f"  {'-'*80}")
        sorted_results = sorted(results_all, key=lambda r: r["test_ppl"])
        for r in sorted_results:
            tag = " *" if r["is_center"] else ""
            aborted = " ABORT" if r.get("aborted") else ""
            print(f"  {r['run_num']:>4d} {r['desc']:<55s} "
                  f"{r['test_ppl']:>10.2f} {r['elapsed_min']:>5.1f}m{tag}{aborted}")

        # Main effects (sorted by absolute magnitude)
        print(f"\n  MAIN EFFECTS (negative = high level is better):")
        print(f"  {'Factor':<25s} {'Effect':>10s} {'|Effect|':>10s} {'Significant?':>14s}")
        print(f"  {'-'*62}")

        # Estimate noise from center points or residuals
        if center_results:
            center_ppls = [r["test_ppl"] for r in center_results]
            noise_est = max(abs(center_ppls[0] - center_ppls[-1]), 1.0) if len(center_ppls) > 1 else 5.0
        else:
            noise_est = 5.0

        sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, effect in sorted_effects:
            abs_eff = abs(effect)
            # Rough significance: |effect| > 2 * noise estimate
            sig = "YES ***" if abs_eff > 3 * noise_est else ("yes *" if abs_eff > 2 * noise_est else ("maybe" if abs_eff > noise_est else "no"))
            direction = "high better" if effect < 0 else "low better"
            print(f"  {name:<25s} {effect:>+10.2f} {abs_eff:>10.2f} {sig:>14s}  ({direction})")

        # Center point curvature
        if center_results:
            center_mean = sum(r["test_ppl"] for r in center_results) / len(center_results)
            pb_mean = sum(r["test_ppl"] for r in pb_results) / len(pb_results)
            curvature = center_mean - pb_mean
            print(f"\n  Center point PPL: {center_mean:.2f} (mean of {len(center_results)} replicates)")
            print(f"  PB mean PPL: {pb_mean:.2f}")
            print(f"  Curvature: {curvature:+.2f} ({'significant' if abs(curvature) > 2 * noise_est else 'not significant'})")

        # Recommendations
        print(f"\n  RECOMMENDATIONS FOR PHASE 2:")
        sig_factors = [(n, e) for n, e in sorted_effects if abs(e) > noise_est]
        if sig_factors:
            print(f"  Significant factors to optimize:")
            for name, effect in sig_factors:
                better = "HIGH" if effect < 0 else "LOW"
                print(f"    - {name}: set to {better} (effect={effect:+.2f})")
        else:
            print(f"  No clearly significant factors found. Consider longer runs or wider factor ranges.")

        # Save analysis
        analysis = {
            "effects": effects,
            "sorted_effects": sorted_effects,
            "significant_factors": [(n, e) for n, e in sorted_effects if abs(e) > noise_est],
            "noise_estimate": noise_est,
            "center_ppls": [r["test_ppl"] for r in center_results] if center_results else [],
        }
        analysis_file = os.path.join(args.log_dir, "doe_screening_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n  Analysis saved to {analysis_file}")

    else:
        print(f"\n  Only {len(pb_results)}/12 PB runs completed. "
              f"Run remaining with --from-run to complete analysis.")
    print()


def run_optimization(args, device):
    """Phase 2: Full factorial on significant factors from Phase 1."""
    analysis_file = os.path.join(args.log_dir, "doe_screening_analysis.json")
    if not os.path.exists(analysis_file):
        print("ERROR: Run phase 1 first (--phase 1)")
        return

    with open(analysis_file) as f:
        analysis = json.load(f)

    sig_factors = analysis["significant_factors"]
    if not sig_factors:
        print("No significant factors found in Phase 1. Nothing to optimize.")
        return

    print(f"\n{'='*70}")
    print(f"  DoE PHASE 2: OPTIMIZATION")
    print(f"  Significant factors from screening:")
    for name, effect in sig_factors:
        print(f"    {name}: effect={effect:+.2f}")
    print(f"{'='*70}\n")

    # Build full factorial on significant factors (up to 4 factors = 16 runs)
    # For >4 significant factors, take top 4 by |effect|
    opt_factors = sig_factors[:4]
    n_opt = len(opt_factors)
    n_runs = 2 ** n_opt

    print(f"  Full {2}^{n_opt} factorial = {n_runs} runs")

    # Generate factorial design
    import itertools
    levels = list(itertools.product([-1, +1], repeat=n_opt))

    # Non-significant factors fixed at their best level from screening
    all_effects = dict(analysis["sorted_effects"])
    opt_factor_names = {name for name, _ in opt_factors}

    results_all = []
    results_file = os.path.join(args.log_dir, "doe_optimization_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results_all = json.load(f)

    total_start = time.time()

    for run_idx, level_combo in enumerate(levels):
        run_num = run_idx + 1

        # Skip if done
        if any(r["run_num"] == run_num for r in results_all):
            print(f"  Run {run_num}/{n_runs} already completed, skipping")
            continue

        # Build overrides: significant factors varied, others fixed at best level
        factor_names = list(FACTORS.keys())
        overrides = {}
        for fname in factor_names:
            factor = FACTORS[fname]
            if fname in opt_factor_names:
                # Varied factor
                idx = [n for n, _ in opt_factors].index(fname)
                level = level_combo[idx]
                overrides[fname] = factor["high"] if level > 0 else factor["low"]
            else:
                # Fixed at best level from screening
                effect = all_effects.get(fname, 0)
                overrides[fname] = factor["low"] if effect > 0 else factor["high"]

        config = DoEConfig()
        if args.max_steps:
            config.max_steps = args.max_steps
        apply_overrides(config, overrides)

        desc = run_desc(overrides)
        name = f"doe_opt{run_num:02d}"

        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)
        cb_params = sum(p.numel() for n, p in model.named_parameters() if "codebook" in n)

        print(f"\n{'='*70}")
        print(f"  Optimization Run {run_num}/{n_runs}: {desc}")
        print(f"  Factor levels: {dict(zip([n for n,_ in opt_factors], level_combo))}")
        print(f"{'='*70}\n")

        t0 = time.time()
        results = train(model, config, device, name, args.log_dir, args.data_dir)
        elapsed = time.time() - t0

        result_record = {
            "run_num": run_num,
            "name": name,
            "desc": desc,
            "level_combo": list(level_combo),
            "overrides": {k: v if not isinstance(v, bool) else int(v) for k, v in overrides.items()},
            "test_ppl": results["test_ppl"],
            "val_ppl": results["val_ppl"],
            "elapsed_min": round(elapsed / 60, 1),
            "cb_params": cb_params,
        }
        results_all.append(result_record)

        with open(results_file, "w") as f:
            json.dump(results_all, f, indent=2)

        print(f"\n  Run {run_num}: Test PPL={results['test_ppl']:.2f} ({elapsed/60:.1f} min)")

        del model
        torch.cuda.empty_cache()

    # Summary
    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION RESULTS (sorted by Test PPL)")
    print(f"{'='*70}")
    print(f"  {'Run':>4s} {'Description':<55s} {'Test PPL':>10s}")
    print(f"  {'-'*72}")
    for r in sorted(results_all, key=lambda x: x["test_ppl"]):
        print(f"  {r['run_num']:>4d} {r['desc']:<55s} {r['test_ppl']:>10.2f}")

    best = min(results_all, key=lambda x: x["test_ppl"])
    print(f"\n  BEST CONFIG: Run {best['run_num']}, Test PPL={best['test_ppl']:.2f}")
    print(f"  {best['desc']}")
    print(f"\n  Total time: {total_elapsed:.0f} minutes")


if __name__ == "__main__":
    main()
