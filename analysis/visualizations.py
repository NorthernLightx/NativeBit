"""Training visualization — reads JSONL logs and generates plots.

Usage:
    python analysis/visualizations.py logs/exp_b_nativebit_3bit.jsonl
    python analysis/visualizations.py logs/exp_a_float_baseline.jsonl logs/exp_b_nativebit_3bit.jsonl --compare
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_single_run(records: list[dict], name: str, out_dir: str) -> list[str]:
    """Generate plots for a single training run. Returns list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    steps = [r["step"] for r in records]

    # 1. Loss and Perplexity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, [r["loss"] for r in records], "b-", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{name} — Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, [r["perplexity"] for r in records], "r-", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title(f"{name} — Perplexity")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{name}_loss_ppl.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # 2. Codebook health (if available)
    if "dead_entries" in records[0]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(steps, [r["dead_entries"] for r in records], "m-", linewidth=1.5)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        total = records[0].get("total_entries", 1)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Dead Entries")
        ax1.set_title(f"{name} — Dead Codebook Entries (total={total})")
        ax1.grid(True, alpha=0.3)

        dead_pcts = [r.get("dead_pct", 0) for r in records]
        ax2.fill_between(steps, dead_pcts, alpha=0.3, color="red")
        ax2.plot(steps, dead_pcts, "r-", linewidth=1.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Dead %")
        ax2.set_title(f"{name} — Dead Entry Percentage")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        p = os.path.join(out_dir, f"{name}_codebook_health.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # 3. Gradient ratio (if available)
    if "grad_ratio_cb_w" in records[0]:
        fig, ax = plt.subplots(figsize=(10, 5))

        ratios = [r.get("grad_ratio_cb_w", 0) for r in records]
        ax.plot(steps, ratios, "g-", linewidth=1.5)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1:1 ratio")
        ax.set_xlabel("Step")
        ax.set_ylabel("Codebook / Weight Gradient Ratio")
        ax.set_title(f"{name} — Gradient Magnitude Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        p = os.path.join(out_dir, f"{name}_grad_ratio.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # 4. Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, [r["lr"] for r in records], "k-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{name} — LR Schedule")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{name}_lr.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    return saved


def plot_comparison(runs: dict[str, list[dict]], out_dir: str) -> list[str]:
    """Compare multiple runs on the same plots. Returns list of saved file paths."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Loss comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, records) in enumerate(runs.items()):
        steps = [r["step"] for r in records]
        c = colors[i % len(colors)]
        ax1.plot(steps, [r["loss"] for r in records], color=c, linewidth=1.5, label=name)
        ax2.plot(steps, [r["perplexity"] for r in records], color=c, linewidth=1.5, label=name)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity Comparison")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "comparison_loss_ppl.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # Dead entries comparison (only for runs that have it)
    runs_with_dead = {n: r for n, r in runs.items() if "dead_entries" in r[0]}
    if runs_with_dead:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (name, records) in enumerate(runs_with_dead.items()):
            steps = [r["step"] for r in records]
            c = colors[i % len(colors)]
            ax.plot(steps, [r.get("dead_pct", 0) for r in records],
                    color=c, linewidth=1.5, label=name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Dead Entry %")
        ax.set_title("Codebook Dead Entry % Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        p = os.path.join(out_dir, "comparison_dead_entries.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("logs", nargs="+", help="Path(s) to .jsonl log file(s)")
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots")
    parser.add_argument("--out-dir", type=str, default="logs/plots", help="Output directory for plots")
    args = parser.parse_args()

    all_runs = {}
    for log_path in args.logs:
        name = Path(log_path).stem
        records = load_jsonl(log_path)
        all_runs[name] = records
        print(f"Loaded {len(records)} records from {log_path}")

    # Individual plots
    for name, records in all_runs.items():
        saved = plot_single_run(records, name, args.out_dir)
        for p in saved:
            print(f"  Saved: {p}")

    # Comparison plots
    if args.compare and len(all_runs) > 1:
        saved = plot_comparison(all_runs, args.out_dir)
        for p in saved:
            print(f"  Saved: {p}")

    print("\nDone!")


if __name__ == "__main__":
    main()
