"""Phase 1 experiments — prove co-training beats post-hoc compression.

Experiment A: Float16 baseline (no quantization) — quality upper bound.
Experiment B: NativeBit 3-bit (co-trained from scratch).
Experiment C: Post-hoc k-means 3-bit (train float16, then quantize).

Success metric: Experiment B achieves lower perplexity than Experiment C.
"""

import argparse
import json
import math
import os
import sys

# torch.compile on Windows: MSVC compiler + short cache paths + disable FX graph cache
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from train import train, run_evaluation
from configs.small import SmallConfig


def posthoc_kmeans_quantize(model: torch.nn.Module, block_size: int = 64, n_clusters: int = 8):
    """Apply post-hoc k-means quantization to all applicable linear layers.

    For each block of `block_size` weights, run k-means with `n_clusters`
    and replace each weight with its nearest cluster center.

    Modifies the model in-place.
    """
    for name, module in model.named_modules():
        # Only quantize regular nn.Linear (not embeddings, LM head handled by caller)
        if not isinstance(module, torch.nn.Linear):
            continue
        # Skip embedding/LM head (tied weights) and layernorm
        if "tok_emb" in name or "lm_head" in name or "ln" in name:
            continue

        w = module.weight.data
        w_flat = w.view(-1).clone()
        total = w_flat.numel()
        num_blocks = math.ceil(total / block_size)

        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, total)
            block = w_flat[start:end].float()

            centers = _kmeans_torch(block, n_clusters, max_iter=50)

            # Assign each weight to nearest center
            dists = (block.unsqueeze(-1) - centers.unsqueeze(0)).abs()
            indices = dists.argmin(dim=-1)
            w_flat[start:end] = centers[indices].to(w_flat.dtype)

        module.weight.data = w_flat.view_as(w)


def _kmeans_torch(data: torch.Tensor, k: int, max_iter: int = 50) -> torch.Tensor:
    """Simple k-means clustering on 1-D tensor. Returns k cluster centers."""
    # Initialize from evenly spaced quantiles (same as codebook init)
    q = torch.linspace(0, 1, k, device=data.device)
    centers = torch.quantile(data.contiguous(), q)

    for _ in range(max_iter):
        # Assign
        dists = (data.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        labels = dists.argmin(dim=-1)

        # Update
        new_centers = torch.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = data[mask].mean()
            else:
                new_centers[c] = centers[c]

        if torch.allclose(centers, new_centers, atol=1e-7):
            break
        centers = new_centers

    return centers


def run_experiment_a(config, device, log_dir, data_dir) -> dict:
    """Experiment A — Float16 baseline (no quantization)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Float16 Baseline")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=False)
    model = torch.compile(model)
    return train(model, config, device, "exp_a_float_baseline", log_dir, data_dir)


def run_experiment_b(config, device, log_dir, data_dir) -> dict:
    """Experiment B — NativeBit 3-bit co-trained from scratch."""
    print("\n" + "=" * 60)
    print("EXPERIMENT B: NativeBit 3-bit (co-trained)")
    print("=" * 60)
    set_seed(config.seed)
    model = build_model_from_config(config, use_nativebit=True)
    model = torch.compile(model)
    return train(model, config, device, "exp_b_nativebit_3bit", log_dir, data_dir)


def run_experiment_c(config, device, log_dir, data_dir) -> dict:
    """Experiment C — Post-hoc k-means 3-bit.

    Train a float16 model (reuse Experiment A architecture), then quantize.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Post-hoc K-means 3-bit")
    print("=" * 60)

    # Try to load Experiment A checkpoint
    ckpt_path = os.path.join(log_dir, "exp_a_float_baseline_final.pt")
    if os.path.exists(ckpt_path):
        print("  Loading Experiment A checkpoint for post-hoc quantization...")
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Train from scratch if no checkpoint
        print("  No Experiment A checkpoint found, training float baseline first...")
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=False)
        model = torch.compile(model)
        train(model, config, device, "exp_c_float_for_kmeans", log_dir, data_dir)

    model = model.to(device)

    # Apply post-hoc k-means quantization
    print("  Applying post-hoc k-means quantization (k=8, block_size=64)...")
    posthoc_kmeans_quantize(model, block_size=config.block_size, n_clusters=config.n_codebook)

    # Re-measure performance
    _, valid_loader, test_loader = get_dataloaders(config.context_len, config.batch_size, data_dir)

    val_loss = run_evaluation(model, valid_loader, device)
    test_loss = run_evaluation(model, test_loader, device)
    val_ppl = math.exp(min(val_loss, 20))
    test_ppl = math.exp(min(test_loss, 20))

    print(f"  Post-hoc k-means: Val loss: {val_loss:.4f}  Val PPL: {val_ppl:.2f}")
    print(f"  Post-hoc k-means: Test loss: {test_loss:.4f}  Test PPL: {test_ppl:.2f}")

    return {
        "train_loss": 0.0,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--only", type=str, default=None, choices=["a", "b", "c"],
                        help="Run only one experiment")
    args = parser.parse_args()

    config = SmallConfig()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    config.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    results = {}

    if args.only is None or args.only == "a":
        results["A_float_baseline"] = run_experiment_a(config, device, args.log_dir, args.data_dir)
    if args.only is None or args.only == "b":
        results["B_nativebit_3bit"] = run_experiment_b(config, device, args.log_dir, args.data_dir)
    if args.only is None or args.only == "c":
        results["C_posthoc_kmeans"] = run_experiment_c(config, device, args.log_dir, args.data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25s} {'Val PPL':>10s} {'Test PPL':>10s} {'Val Loss':>10s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25s} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f} {r['val_loss']:>10.4f}")

    if "B_nativebit_3bit" in results and "C_posthoc_kmeans" in results:
        b_ppl = results["B_nativebit_3bit"]["test_ppl"]
        c_ppl = results["C_posthoc_kmeans"]["test_ppl"]
        if b_ppl < c_ppl:
            print(f"\nSUCCESS: NativeBit ({b_ppl:.2f}) < Post-hoc k-means ({c_ppl:.2f})")
            print("Co-training beats post-hoc compression!")
        else:
            print(f"\nNOT YET: NativeBit ({b_ppl:.2f}) >= Post-hoc k-means ({c_ppl:.2f})")
            print("Co-training hasn't beaten post-hoc. Check codebook utilization logs.")

    # Save summary
    summary_path = os.path.join(args.log_dir, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Generate plots
    try:
        from analysis.visualizations import plot_single_run, plot_comparison, load_jsonl

        plot_dir = os.path.join(args.log_dir, "plots")
        log_files = {
            "exp_a_float_baseline": os.path.join(args.log_dir, "exp_a_float_baseline.jsonl"),
            "exp_b_nativebit_3bit": os.path.join(args.log_dir, "exp_b_nativebit_3bit.jsonl"),
        }

        all_runs = {}
        for name, path in log_files.items():
            if os.path.exists(path):
                records = load_jsonl(path)
                all_runs[name] = records
                saved = plot_single_run(records, name, plot_dir)
                for p in saved:
                    print(f"  Plot: {p}")

        if len(all_runs) > 1:
            saved = plot_comparison(all_runs, plot_dir)
            for p in saved:
                print(f"  Plot: {p}")

        print(f"\nAll plots saved to {plot_dir}")
    except Exception as e:
        print(f"\nWarning: could not generate plots: {e}")


if __name__ == "__main__":
    main()
