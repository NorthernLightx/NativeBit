"""CLI entry point for the autoresearch system."""

import argparse
import signal
import sys

from autoresearch.runner import AutoresearchRunner
from autoresearch.report import (
    print_leaderboard,
    print_param_importance,
    print_session_report,
    print_strategy_stats,
)
from autoresearch.strategies import StrategyManager


def main():
    parser = argparse.ArgumentParser(
        description="NativeBit Autoresearch — autonomous hyperparameter search"
    )
    parser.add_argument("--max-trials", type=int, default=0,
                        help="Stop after N trials (0 = unlimited)")
    parser.add_argument("--max-hours", type=float, default=0,
                        help="Stop after N hours (0 = unlimited)")
    parser.add_argument("--max-vram-gb", type=float, default=6.0,
                        help="CUDA VRAM limit in GB (default: 6.0)")
    parser.add_argument("--bootstrap", type=str, default=None,
                        help="Path to DoE results JSON to bootstrap from")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous session")
    parser.add_argument("--report", action="store_true",
                        help="Show current leaderboard and exit")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Base log directory")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    args = parser.parse_args()

    # Set CUDA VRAM limit — any allocation beyond this raises OOM,
    # which the evaluator catches and retries with smaller batch size
    import torch
    if torch.cuda.is_available() and args.max_vram_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        fraction = min(args.max_vram_gb / total, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"  VRAM limit: {args.max_vram_gb:.1f} GB ({fraction:.0%} of {total:.1f} GB)")

    runner = AutoresearchRunner(log_dir=args.log_dir, data_dir=args.data_dir)

    # Always try to load existing state
    had_state = runner.load_state()

    # Bootstrap from DoE
    if args.bootstrap:
        print(f"  Bootstrapping from {args.bootstrap}")
        runner.bootstrap_from_doe(args.bootstrap)
        if args.report:
            _show_report(runner)
            return

    # Resume
    if args.resume and not had_state:
        print("  No previous session found. Starting fresh.")

    # Report only
    if args.report:
        _show_report(runner)
        return

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n  Ctrl+C received — finishing current trial and saving state...")
        runner.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # Run the search loop
    try:
        runner.run(max_trials=args.max_trials, max_hours=args.max_hours)
    except KeyboardInterrupt:
        print("\n  Force quit. Saving state...")
    finally:
        runner.save_state()
        print("  State saved. Run with --resume to continue.")


def _show_report(runner: AutoresearchRunner):
    """Print full report and exit."""
    print_leaderboard(runner.lb, n=10)
    print_param_importance(runner.lb)
    print_strategy_stats(runner.lb, runner.mgr)


if __name__ == "__main__":
    main()
