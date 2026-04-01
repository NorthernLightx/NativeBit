"""Phase 3 benchmarks — rigorous comparison of NativeBit vs baselines at medium scale.

Methods compared:
  1. Float16 baseline (upper bound)
  2. Uniform 3-bit (post-hoc)
  3. K-means 3-bit (post-hoc)
  4. NF4 4-bit (post-hoc, QLoRA-style)
  5. NativeBit 3-bit (co-trained)

Each method is tested on WikiText-2 (and optionally TinyStories).
Reports: test PPL, model size (bytes), inference speed (tok/s), quality-per-bit.
"""

import argparse
import copy
import json
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.data import get_dataloaders
from nativebit.baselines import (
    quantize_uniform,
    quantize_kmeans,
    quantize_nf4,
    compute_model_size,
    measure_inference_speed,
)
from train import train, run_evaluation
from configs.medium import MediumConfig


def eval_on_dataset(model, config, device, dataset, data_dir):
    """Run test-set perplexity for a model on a given dataset."""
    _, valid_loader, test_loader = get_dataloaders(
        config.context_len, config.batch_size, data_dir, dataset=dataset,
    )
    val_loss = run_evaluation(model, valid_loader, device)
    test_loss = run_evaluation(model, test_loader, device)
    val_ppl = math.exp(min(val_loss, 20))
    test_ppl = math.exp(min(test_loss, 20))
    return {"val_loss": val_loss, "val_ppl": val_ppl,
            "test_loss": test_loss, "test_ppl": test_ppl}


def train_or_load(config, device, name, log_dir, data_dir, use_nativebit, retrain=False):
    """Train a model or load from checkpoint."""
    ckpt_path = os.path.join(log_dir, f"{name}_final.pt")
    if os.path.exists(ckpt_path) and not retrain:
        print(f"\n  Loading checkpoint: {ckpt_path}")
        model = build_model_from_config(config, use_nativebit=use_nativebit)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix from torch.compile checkpoints
        sd = ckpt["model_state_dict"]
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)
        return model, ckpt
    else:
        print(f"\n  Training {name} from scratch...")
        set_seed(config.seed)
        model = build_model_from_config(config, use_nativebit=use_nativebit)
        if not use_nativebit:
            model = torch.compile(model)
        results = train(model, config, device, name, log_dir, data_dir)
        return model, results


def run_benchmarks(args):
    config = MediumConfig()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    config.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",")]
    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*70}")

        config.dataset = dataset
        results = {}

        # --- 1. Float16 Baseline ---
        print(f"\n--- [1/5] Float16 Baseline ---")
        float_name = f"phase3_float_{dataset}"
        float_model, float_info = train_or_load(
            config, device, float_name, args.log_dir, args.data_dir,
            use_nativebit=False, retrain=args.retrain,
        )
        # Always re-run test PPL (fast)
        float_metrics = eval_on_dataset(float_model, config, device, dataset, args.data_dir)
        float_size = compute_model_size(float_model, "float16",
                                        exclude_modules=[float_model.lm_head])
        print(f"  Float16: test PPL={float_metrics['test_ppl']:.2f}, "
              f"size={float_size['total_bytes']/1024:.1f} KB")
        results["float16"] = {
            **float_metrics,
            "size_bytes": float_size["total_bytes"],
            "size_bits": float_size["total_bits"],
            "method": "float16",
        }

        # --- 2. Uniform 3-bit (post-hoc) ---
        print(f"\n--- [2/5] Uniform 3-bit (post-hoc) ---")
        uni_model = copy.deepcopy(float_model)
        quantize_uniform(uni_model, bits=3, block_size=config.block_size,
                         exclude_modules=[uni_model.lm_head])
        uni_metrics = eval_on_dataset(uni_model, config, device, dataset, args.data_dir)
        uni_size = compute_model_size(float_model, "uniform", bits=3,
                                      block_size=config.block_size,
                                      exclude_modules=[float_model.lm_head])
        print(f"  Uniform 3-bit: test PPL={uni_metrics['test_ppl']:.2f}, "
              f"size={uni_size['total_bytes']/1024:.1f} KB")
        results["uniform_3bit"] = {
            **uni_metrics,
            "size_bytes": uni_size["total_bytes"],
            "size_bits": uni_size["total_bits"],
            "method": "uniform_3bit",
        }
        del uni_model

        # --- 3. K-means 3-bit (post-hoc) ---
        print(f"\n--- [3/5] K-means 3-bit (post-hoc) ---")
        km_model = copy.deepcopy(float_model)
        quantize_kmeans(km_model, n_entries=8, block_size=config.block_size,
                        exclude_modules=[km_model.lm_head])
        km_metrics = eval_on_dataset(km_model, config, device, dataset, args.data_dir)
        km_size = compute_model_size(float_model, "kmeans", n_entries=8,
                                     block_size=config.block_size,
                                     exclude_modules=[float_model.lm_head])
        print(f"  K-means 3-bit: test PPL={km_metrics['test_ppl']:.2f}, "
              f"size={km_size['total_bytes']/1024:.1f} KB")
        results["kmeans_3bit"] = {
            **km_metrics,
            "size_bytes": km_size["total_bytes"],
            "size_bits": km_size["total_bits"],
            "method": "kmeans_3bit",
        }
        del km_model

        # --- 4. NF4 4-bit (post-hoc) ---
        print(f"\n--- [4/5] NF4 4-bit (post-hoc) ---")
        nf4_model = copy.deepcopy(float_model)
        quantize_nf4(nf4_model, block_size=config.block_size,
                     exclude_modules=[nf4_model.lm_head])
        nf4_metrics = eval_on_dataset(nf4_model, config, device, dataset, args.data_dir)
        nf4_size = compute_model_size(float_model, "nf4",
                                      block_size=config.block_size,
                                      exclude_modules=[float_model.lm_head])
        print(f"  NF4 4-bit: test PPL={nf4_metrics['test_ppl']:.2f}, "
              f"size={nf4_size['total_bytes']/1024:.1f} KB")
        results["nf4_4bit"] = {
            **nf4_metrics,
            "size_bytes": nf4_size["total_bytes"],
            "size_bits": nf4_size["total_bits"],
            "method": "nf4_4bit",
        }
        del nf4_model

        # Free float model memory before NativeBit training
        del float_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- 5. NativeBit 3-bit (co-trained) ---
        print(f"\n--- [5/5] NativeBit 3-bit (co-trained) ---")
        nb_name = f"phase3_nativebit_{dataset}"
        nb_model, nb_info = train_or_load(
            config, device, nb_name, args.log_dir, args.data_dir,
            use_nativebit=True, retrain=args.retrain,
        )
        nb_metrics = eval_on_dataset(nb_model, config, device, dataset, args.data_dir)
        nb_size = compute_model_size(nb_model, "nativebit", n_entries=config.n_codebook,
                                     block_size=config.block_size,
                                     exclude_modules=[nb_model.lm_head])
        print(f"  NativeBit 3-bit: test PPL={nb_metrics['test_ppl']:.2f}, "
              f"size={nb_size['total_bytes']/1024:.1f} KB")
        results["nativebit_3bit"] = {
            **nb_metrics,
            "size_bytes": nb_size["total_bytes"],
            "size_bits": nb_size["total_bits"],
            "method": "nativebit_3bit",
        }

        # Inference speed (optional, slow)
        if args.speed:
            print(f"\n  Measuring inference speed...")
            for method_name in results:
                # Re-load or use existing model for speed test
                if method_name == "nativebit_3bit":
                    speed = measure_inference_speed(nb_model, device, config.context_len)
                elif method_name == "float16":
                    tmp_model = build_model_from_config(config, use_nativebit=False)
                    ckpt_path = os.path.join(args.log_dir, f"{float_name}_final.pt")
                    if os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                        tmp_model.load_state_dict(ckpt["model_state_dict"])
                    speed = measure_inference_speed(tmp_model, device, config.context_len)
                    del tmp_model
                else:
                    speed = 0.0  # post-hoc methods have same speed as float
                results[method_name]["tok_per_sec"] = round(speed, 1)
                if speed > 0:
                    print(f"    {method_name}: {speed:.1f} tok/s")

        del nb_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_results[dataset] = results

    # --- Summary ---
    print(f"\n{'='*70}")
    print("PHASE 3 BENCHMARK RESULTS")
    print(f"{'='*70}")

    for dataset, results in all_results.items():
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Method':<20s} {'Test PPL':>10s} {'Val PPL':>10s} "
              f"{'Size(KB)':>10s} {'Bits/w':>8s} {'PPL/bit':>8s}")
        print(f"  {'-'*66}")

        float_ppl = results.get("float16", {}).get("test_ppl", 0)
        for name, r in results.items():
            size_kb = r["size_bytes"] / 1024
            # Compute effective bits per weight (excluding embeddings)
            weight_bits = r["size_bits"] - results["float16"]["size_bits"] * 0  # total bits
            total_weights = 0
            # Rough: non-embedding params
            ppl_per_bit = 0
            if r["size_bits"] > 0:
                ppl_per_bit = r["test_ppl"] / (r["size_bits"] / 1_000_000)
            speed_str = f"{r.get('tok_per_sec', 0):.0f}" if r.get("tok_per_sec") else ""
            print(f"  {name:<20s} {r['test_ppl']:>10.2f} {r['val_ppl']:>10.2f} "
                  f"{size_kb:>10.1f} {r['size_bits']/1e6:>8.2f}M {ppl_per_bit:>8.4f}")

        # Delta table
        print(f"\n  Deltas vs Float16 (PPL):")
        for name, r in results.items():
            if name == "float16":
                continue
            delta = r["test_ppl"] - float_ppl
            pct = delta / float_ppl * 100 if float_ppl > 0 else 0
            size_ratio = r["size_bytes"] / results["float16"]["size_bytes"] if results["float16"]["size_bytes"] > 0 else 0
            print(f"    {name:<20s} {delta:>+8.2f} ({pct:>+5.1f}%)  "
                  f"size={size_ratio:.2f}x float")

    # Save summary JSON
    summary_path = os.path.join(args.log_dir, "phase3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="wikitext-2",
                        help="Comma-separated list: wikitext-2,tinystories")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain even if checkpoints exist")
    parser.add_argument("--speed", action="store_true",
                        help="Measure inference speed (slow)")
    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
