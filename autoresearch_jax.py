"""JAX/TPU autoresearch CLI — time-based autonomous hyperparameter search.

Runs autonomous NativeBit hyperparameter optimization on TPU.
Results are continuously written to docs/RESEARCH_REPORT.md in scientific format.

Usage:
    # Run on TPU (auto-searches for best NativeBit config)
    python autoresearch_jax.py --max-hours 4 --data-dir data

    # Resume from previous state
    python autoresearch_jax.py --resume --max-hours 2

    # Report current status
    python autoresearch_jax.py --report
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch evaluator BEFORE importing runner (which imports evaluator, which imports torch)
import autoresearch.evaluator_jax as _jax_eval
import types
# Create a minimal mock evaluator module to prevent torch import
_mock_eval = types.ModuleType("autoresearch.evaluator")
_mock_eval.evaluate_trial = _jax_eval.evaluate_trial_jax
_mock_eval.SCREEN_STEPS = 0
sys.modules["autoresearch.evaluator"] = _mock_eval

from autoresearch.runner import AutoresearchRunner
from autoresearch.report import print_leaderboard, print_session_report
from autoresearch.report_writer import update_research_report, add_finding


class JaxAutoresearchRunner(AutoresearchRunner):
    """Autoresearch with JAX evaluator and research report updates."""

    def __init__(self, log_dir="logs", data_dir="data",
                 report_path="docs/RESEARCH_REPORT.md"):
        super().__init__(log_dir, data_dir)
        self.report_path = report_path

    def on_trial_complete(self, trial):
        """Hook called after each trial — update research report."""
        update_research_report(trial, self.report_path)

        # Log significant findings
        if trial.status == "accepted":
            config_summary = ", ".join(
                f"{k}={v}" for k, v in sorted(trial.config.items())
                if k in ("requantize_every", "ema_decay", "block_size", "n_codebook")
            )
            add_finding(
                f"Trial {trial.trial_id} accepted: PPL {trial.confirm_ppl:.1f} "
                f"({config_summary})",
                self.report_path,
            )
        elif trial.status == "rejected_screen" and hasattr(trial, "screen_ppl"):
            # Only log interesting rejections
            if trial.screen_ppl and trial.screen_ppl < self.lb.best_ppl * 1.10:
                add_finding(
                    f"Trial {trial.trial_id} narrowly rejected at screen: "
                    f"PPL {trial.screen_ppl:.1f} (threshold: {self.lb.best_ppl * 1.05:.1f})",
                    self.report_path,
                )

        print(f"  [Report updated: {self.report_path}]")


def main():
    parser = argparse.ArgumentParser(description="NativeBit JAX Autoresearch")
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--report-path", default="docs/RESEARCH_REPORT.md")
    args = parser.parse_args()

    if args.report:
        runner = JaxAutoresearchRunner(args.log_dir, args.data_dir, args.report_path)
        runner.load_state()
        print_leaderboard(runner.lb)
        print_session_report(runner)
        return

    # Monkey-patch evaluator to use JAX version
    import autoresearch.runner as runner_mod
    from autoresearch.evaluator_jax import evaluate_trial_jax
    runner_mod.evaluate_trial = evaluate_trial_jax

    runner = JaxAutoresearchRunner(args.log_dir, args.data_dir, args.report_path)
    if args.resume:
        runner.load_state()

    print(f"\n=== NativeBit JAX Autoresearch ===")
    print(f"  Backend: JAX ({__import__('jax').default_backend()})")
    print(f"  Max hours: {args.max_hours}")
    print(f"  Report: {args.report_path}")
    print()

    runner.run(max_hours=args.max_hours)


if __name__ == "__main__":
    main()
