"""TPU overnight experiment suite — scale validation for NativeBit paper.

Runs on Cloud TPU v6e. Answers key questions:
1. Does NativeBit work at 48M+ params?
2. Does quantization=regularization hold at scale?
3. What's the optimal block_size at scale?
4. Does 4-bit work at larger model sizes?

Usage:
    python experiments/tpu_overnight.py --log-dir logs/tpu --data-dir data
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nativebit.seed import set_seed
from nativebit.model import build_model_from_config
from nativebit.device import get_device, is_cuda
from train import train


class BaseTPUConfig:
    """32M param config — proven stable on TPU v6e XLA.
    Bumped batch_size to 64 for better TPU utilization."""
    n_layers: int = 20
    n_embd: int = 192
    n_head: int = 4
    ffn_hidden: int = 768
    context_len: int = 256
    vocab_size: int = 50257

    block_size: int = 64
    n_codebook: int = 8

    batch_size: int = 64
    lr: float = 1.5e-3
    codebook_lr: float = 1.5e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 100
    log_every: int = 50
    weight_decay: float = 0.01
    dataset: str = "wikitext-103"
    seed: int = 42


class LargerTPUConfig:
    """~60M params — head_dim=48 (same as working 32M). Should work on XLA."""
    n_layers: int = 16
    n_embd: int = 384
    n_head: int = 8      # head_dim = 384/8 = 48 (matches working model)
    ffn_hidden: int = 1536
    context_len: int = 256
    vocab_size: int = 50257

    block_size: int = 64
    n_codebook: int = 8

    batch_size: int = 32
    lr: float = 6e-4
    codebook_lr: float = 6e-5
    max_steps: int = 10000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    codebook_grad_clip: float = 1.0
    revive_every: int = 200
    log_every: int = 50
    weight_decay: float = 0.1
    dataset: str = "wikitext-103"
    seed: int = 42


class MediumTPUConfig(BaseTPUConfig):
    """125M param config."""
    n_layers: int = 12
    n_embd: int = 768
    n_head: int = 12
    ffn_hidden: int = 3072
    context_len: int = 1024
    batch_size: int = 32
    lr: float = 6e-4
    codebook_lr: float = 6e-5
    max_steps: int = 10000
    warmup_steps: int = 500
    weight_decay: float = 0.1
    dataset: str = "wikitext-103"


EXPERIMENTS = [
    # 1. NativeBit 3-bit baseline (already running with bs=8, will be skipped on resume)
    {
        "name": "tpu_nativebit_3bit_bs64",
        "use_nativebit": True,
        "config_overrides": {},
        "description": "NativeBit 3-bit, 32M params, baseline (bs=8)",
    },
    # 2. Float baseline — bs=64 for speed
    {
        "name": "tpu_float_baseline",
        "use_nativebit": False,
        "config_overrides": {},
        "description": "Float baseline, 32M params, bs=64",
    },
    # 3. Block size sweep
    {
        "name": "tpu_nativebit_bs128",
        "use_nativebit": True,
        "config_overrides": {"block_size": 128},
        "description": "NativeBit 3-bit, block_size=128, bs=64",
    },
    # 4. 4-bit
    {
        "name": "tpu_nativebit_4bit",
        "use_nativebit": True,
        "config_overrides": {"n_codebook": 16},
        "description": "NativeBit 4-bit (16 entries), bs=64",
    },
    # 5. 2-bit
    {
        "name": "tpu_nativebit_2bit",
        "use_nativebit": True,
        "config_overrides": {"n_codebook": 4},
        "description": "NativeBit 2-bit (4 entries), bs=64",
    },
    # 6. DoE: block_size=128 × 4-bit (missing from matrix)
    {
        "name": "tpu_nativebit_bs128_4bit",
        "use_nativebit": True,
        "config_overrides": {"block_size": 128, "n_codebook": 16},
        "description": "DoE: block_size=128, 4-bit (16 entries)",
    },
    # 7. DoE: block_size=128 × 2-bit (missing from matrix)
    {
        "name": "tpu_nativebit_bs128_2bit",
        "use_nativebit": True,
        "config_overrides": {"block_size": 128, "n_codebook": 4},
        "description": "DoE: block_size=128, 2-bit (4 entries)",
    },
    # 8. LARGER MODEL: 60M params (head_dim=48, should work on XLA)
    {
        "name": "tpu_nativebit_60M",
        "use_nativebit": True,
        "config_class": "LargerTPUConfig",
        "config_overrides": {},
        "description": "NativeBit 3-bit, 60M params (16L, 384d, 8h)",
    },
    # 7. 60M float baseline
    {
        "name": "tpu_float_60M",
        "use_nativebit": False,
        "config_class": "LargerTPUConfig",
        "config_overrides": {},
        "description": "Float baseline, 60M params",
    },
    # 8. 125M param run (if time allows)
    {
        "name": "tpu_nativebit_125M",
        "use_nativebit": True,
        "config_class": "MediumTPUConfig",
        "config_overrides": {},
        "description": "NativeBit 3-bit, 125M params",
    },
]


def run_experiment(exp: dict, log_dir: str, data_dir: str, device: torch.device):
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"{'='*60}\n")

    config_class = exp.get("config_class", None)
    config_map = {
        "MediumTPUConfig": MediumTPUConfig,
        "LargerTPUConfig": LargerTPUConfig,
    }
    if config_class and config_class in config_map:
        config = config_map[config_class]()
    else:
        config = BaseTPUConfig()

    # Apply overrides
    for k, v in exp.get("config_overrides", {}).items():
        setattr(config, k, v)

    set_seed(config.seed)

    use_nativebit = exp["use_nativebit"]
    model = build_model_from_config(config, use_nativebit=use_nativebit)

    # Re-init codebooks
    if use_nativebit and hasattr(model, 'get_nativebit_layers'):
        import torch.nn.functional as F
        from nativebit.codebook_utils import init_codebook_kmeans_batch
        for layer in model.get_nativebit_layers():
            w_flat = layer.weight.data.view(-1)
            if layer._padded_len > layer.total_weights:
                w_flat = F.pad(w_flat, (0, layer._padded_len - layer.total_weights))
            w_blocks = w_flat.view(layer.num_blocks, layer.block_size)
            layer.codebook.data.copy_(init_codebook_kmeans_batch(w_blocks, layer.n_entries))

    if is_cuda(device):
        model = torch.compile(model)

    start_time = time.time()
    results = train(model, config, device, exp["name"], log_dir, data_dir)
    elapsed = time.time() - start_time

    results["experiment"] = exp["name"]
    results["description"] = exp["description"]
    results["use_nativebit"] = use_nativebit
    results["elapsed_min"] = round(elapsed / 60, 1)
    results["config"] = {k: getattr(config, k) for k in dir(config)
                         if not k.startswith("_") and not callable(getattr(config, k))}

    return results


def main():
    parser = argparse.ArgumentParser(description="TPU overnight experiments")
    parser.add_argument("--log-dir", type=str, default="logs/tpu")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from experiment index (for resuming)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment name")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    results_path = os.path.join(args.log_dir, "overnight_results.json")
    queue_path = os.path.join(args.log_dir, "queue.json")
    all_results = []

    # Load existing results if resuming
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results")

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]

    # Write queue.json so the dashboard knows what's planned
    def _update_queue(experiments, all_results, current_name=None):
        completed = {r["experiment"] for r in all_results if "experiment" in r}
        failed = {r["experiment"] for r in all_results if r.get("error")}
        queue = []
        for exp in experiments:
            status = "completed" if exp["name"] in completed else \
                     "failed" if exp["name"] in failed else \
                     "running" if exp["name"] == current_name else "pending"
            queue.append({
                "name": exp["name"],
                "description": exp["description"],
                "use_nativebit": exp["use_nativebit"],
                "status": status,
                "config_overrides": exp.get("config_overrides", {}),
            })
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)

    _update_queue(experiments, all_results)

    for i, exp in enumerate(experiments):
        if i < args.start_from:
            print(f"Skipping {exp['name']} (start_from={args.start_from})")
            continue

        # Skip if already completed
        if any(r["experiment"] == exp["name"] for r in all_results):
            print(f"Skipping {exp['name']} (already completed)")
            continue

        try:
            _update_queue(experiments, all_results, current_name=exp["name"])
            results = run_experiment(exp, args.log_dir, args.data_dir, device)
            all_results.append(results)

            # Save after each experiment (crash-safe)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            _update_queue(experiments, all_results)

            print(f"\n  Result: val_ppl={results['val_ppl']:.2f}  "
                  f"test_ppl={results['test_ppl']:.2f}  "
                  f"time={results['elapsed_min']}min")

        except Exception as e:
            print(f"\n  FAILED: {exp['name']}: {e}")
            all_results.append({
                "experiment": exp["name"],
                "error": str(e),
            })
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  OVERNIGHT RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['experiment']:30s}  FAILED: {r['error'][:50]}")
        else:
            nb = "NB" if r.get("use_nativebit") else "FP"
            print(f"  {r['experiment']:30s}  [{nb}]  val_ppl={r['val_ppl']:>8.2f}  "
                  f"test_ppl={r['test_ppl']:>8.2f}  {r['elapsed_min']}min")


if __name__ == "__main__":
    main()
