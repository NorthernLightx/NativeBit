"""Phase 3 ablation studies — systematic sweeps at medium scale.

Ablation groups:
  A) Block size sweep:  16, 32, 64, 128  (3-bit, cb_lr=1e-3)
  B) Bit width sweep:   2, 3, 4, 5 bit   (bs=32, cb_lr=1e-3 for 2-3bit, 3e-4 for 4-5bit)
  C) Codebook LR sweep: 3e-5, 1e-4, 3e-4, 1e-3  (3-bit, bs=32)
  D) Steps sweep:       5k, 10k, 20k     (3-bit, bs=32, cb_lr=1e-3)

Each ablation trains a NativeBit model and records test PPL + model size.
Also computes the post-hoc k-means floor for comparison.
"""

import argparse
import copy
import json
import math
import os
import sys

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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from nativebit.baselines import quantize_kmeans, compute_model_size
from train import train, run_evaluation
from configs.medium import MediumConfig


def eval_model(model, config, device, data_dir):
    """Quick test PPL measurement."""
    _, valid_loader, test_loader = get_dataloaders(
        config.context_len, config.batch_size, data_dir,
        dataset=getattr(config, "dataset", "wikitext-2"),
    )
    val_loss = run_evaluation(model, valid_loader, device)
    test_loss = run_evaluation(model, test_loader, device)
    return {
        "val_ppl": math.exp(min(val_loss, 20)),
        "test_ppl": math.exp(min(test_loss, 20)),
        "val_loss": val_loss,
        "test_loss": test_loss,
    }


def posthoc_floor(float_model, config, device, data_dir, n_entries, block_size):
    """Compute post-hoc k-means PPL floor for given settings."""
    model_copy = copy.deepcopy(float_model)
    quantize_kmeans(model_copy, n_entries=n_entries, block_size=block_size,
                    exclude_modules=[model_copy.lm_head])
    metrics = eval_model(model_copy, config, device, data_dir)
    del model_copy
    return metrics["test_ppl"]


def train_ablation(config, device, name, log_dir, data_dir,
                   block_size, n_codebook, codebook_lr, max_steps, retrain=False):
    """Train a single NativeBit ablation variant."""
    # Override config
    config.block_size = block_size
    config.n_codebook = n_codebook
    config.codebook_lr = codebook_lr
    config.max_steps = max_steps

    ckpt_path = os.path.join(log_dir, f"{name}_final.pt")
    if os.path.exists(ckpt_path) and not retrain:
        print(f"  Loading checkpoint: {ckpt_path}")
        model = build_model_from_config(config, use_nativebit=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)
    else:
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=True)
        # Don't torch.compile NativeBit — compiler fights codebook ops, 2.4x slower
        train(model, config, device, name, log_dir, data_dir)

    metrics = eval_model(model, config, device, data_dir)
    size_info = compute_model_size(model, "nativebit", n_entries=n_codebook,
                                   block_size=block_size,
                                   exclude_modules=[model.lm_head])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {**metrics, "size_bytes": size_info["total_bytes"],
            "size_bits": size_info["total_bits"]}


def get_float_model(config, device, log_dir, data_dir, retrain=False):
    """Get trained float baseline (train or load)."""
    name = "phase3_float_wikitext-2"
    ckpt_path = os.path.join(log_dir, f"{name}_final.pt")
    if os.path.exists(ckpt_path) and not retrain:
        print(f"  Loading float baseline: {ckpt_path}")
        model = build_model_from_config(config, use_nativebit=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)
    else:
        print("  Training float baseline first...")
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=False)
        model = torch.compile(model)
        train(model, config, device, name, log_dir, data_dir)
    return model


def run_ablation_group(group_name, variants, base_config, device, log_dir, data_dir,
                       float_model, retrain=False):
    """Run a group of ablation experiments and print results."""
    print(f"\n{'='*70}")
    print(f"  ABLATION GROUP: {group_name}")
    print(f"{'='*70}")

    results = {}
    for var_name, params in variants.items():
        print(f"\n--- {var_name} ---")
        config = copy.copy(base_config)
        config.dataset = "wikitext-2"

        bs = params.get("block_size", 32)
        nc = params.get("n_codebook", 8)
        cb_lr = params.get("codebook_lr", 1e-3)
        steps = params.get("max_steps", base_config.max_steps)

        # Health checks for risky configs
        if nc >= 16:
            config.health_check_steps = [1000, 2000]
            config.health_max_ppl = {1000: 500, 2000: 300}
            config.health_max_dead_pct = 15.0
        else:
            config.health_check_steps = [1000]
            config.health_max_ppl = {1000: 500}
            config.health_max_dead_pct = 12.0

        exp_name = f"abl_{group_name}_{var_name}"
        r = train_ablation(config, device, exp_name, log_dir, data_dir,
                           bs, nc, cb_lr, steps, retrain=retrain)

        # Post-hoc floor
        floor_ppl = posthoc_floor(float_model, config, device, data_dir, nc, bs)
        r["posthoc_floor"] = floor_ppl
        r["gap_to_floor"] = r["test_ppl"] - floor_ppl
        r["params"] = params

        results[var_name] = r
        print(f"  {var_name}: test_ppl={r['test_ppl']:.2f}  floor={floor_ppl:.2f}  "
              f"gap={r['gap_to_floor']:+.2f}  size={r['size_bytes']/1024:.1f}KB")

    # Summary table
    print(f"\n  {'Variant':<20s} {'Test PPL':>10s} {'Floor':>10s} {'Gap':>8s} {'Size(KB)':>10s}")
    print(f"  {'-'*58}")
    for var_name, r in results.items():
        print(f"  {var_name:<20s} {r['test_ppl']:>10.2f} {r['posthoc_floor']:>10.2f} "
              f"{r['gap_to_floor']:>+8.2f} {r['size_bytes']/1024:>10.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 ablation studies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--group", type=str, default=None,
                        choices=["block_size", "bit_width", "codebook_lr", "steps", "all"],
                        help="Run only one ablation group (default: all)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps for all ablations (useful for quick tests)")
    args = parser.parse_args()

    base_config = MediumConfig()
    if args.max_steps is not None:
        base_config.max_steps = args.max_steps
    base_config.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    # Get float baseline (needed for post-hoc floors)
    print("Loading float baseline for post-hoc floor computation...")
    float_model = get_float_model(base_config, device, args.log_dir, args.data_dir)
    float_metrics = eval_model(float_model, base_config, device, args.data_dir)
    print(f"Float baseline: test_ppl={float_metrics['test_ppl']:.2f}")

    all_results = {"float_baseline": float_metrics}
    run_group = args.group or "all"

    # --- A) Block size sweep ---
    if run_group in ("all", "block_size"):
        block_variants = {
            "bs16":  {"block_size": 16, "n_codebook": 8, "codebook_lr": 1e-3},
            "bs32":  {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-3},
            "bs64":  {"block_size": 64, "n_codebook": 8, "codebook_lr": 1e-3},
            "bs128": {"block_size": 128, "n_codebook": 8, "codebook_lr": 1e-3},
        }
        all_results["block_size"] = run_ablation_group(
            "block_size", block_variants, base_config, device,
            args.log_dir, args.data_dir, float_model, args.retrain)

    # --- B) Bit width sweep ---
    if run_group in ("all", "bit_width"):
        bit_variants = {
            "2bit":  {"block_size": 32, "n_codebook": 4,  "codebook_lr": 1e-3},
            "3bit":  {"block_size": 32, "n_codebook": 8,  "codebook_lr": 1e-3},
            "4bit":  {"block_size": 32, "n_codebook": 16, "codebook_lr": 3e-4},
            "5bit":  {"block_size": 32, "n_codebook": 32, "codebook_lr": 1e-4},
        }
        all_results["bit_width"] = run_ablation_group(
            "bit_width", bit_variants, base_config, device,
            args.log_dir, args.data_dir, float_model, args.retrain)

    # --- C) Codebook LR sweep ---
    if run_group in ("all", "codebook_lr"):
        lr_variants = {
            "lr_3e-5": {"block_size": 32, "n_codebook": 8, "codebook_lr": 3e-5},
            "lr_1e-4": {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-4},
            "lr_3e-4": {"block_size": 32, "n_codebook": 8, "codebook_lr": 3e-4},
            "lr_1e-3": {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-3},
        }
        all_results["codebook_lr"] = run_ablation_group(
            "codebook_lr", lr_variants, base_config, device,
            args.log_dir, args.data_dir, float_model, args.retrain)

    # --- D) Steps sweep ---
    if run_group in ("all", "steps"):
        step_variants = {
            "5k":  {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-3, "max_steps": 5000},
            "10k": {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-3, "max_steps": 10000},
            "20k": {"block_size": 32, "n_codebook": 8, "codebook_lr": 1e-3, "max_steps": 20000},
        }
        all_results["steps"] = run_ablation_group(
            "steps", step_variants, base_config, device,
            args.log_dir, args.data_dir, float_model, args.retrain)

    # --- Final summary ---
    print(f"\n{'='*70}")
    print("PHASE 3 ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"Float baseline: test_ppl={float_metrics['test_ppl']:.2f}")

    for group_name, group_results in all_results.items():
        if group_name == "float_baseline":
            continue
        print(f"\n  {group_name}:")
        if isinstance(group_results, dict) and "test_ppl" not in group_results:
            for var_name, r in group_results.items():
                gap_str = f"gap={r['gap_to_floor']:+.2f}" if "gap_to_floor" in r else ""
                print(f"    {var_name:<20s} ppl={r['test_ppl']:.2f}  {gap_str}")

    # Save
    summary_path = os.path.join(args.log_dir, "phase3_ablations.json")
    # Convert non-serializable values
    def _clean(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
        return obj

    def _clean_dict(d):
        if isinstance(d, dict):
            return {k: _clean_dict(v) for k, v in d.items()}
        return _clean(d)

    with open(summary_path, "w") as f:
        json.dump(_clean_dict(all_results), f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
