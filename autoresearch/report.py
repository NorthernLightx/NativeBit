"""Session reports, parameter importance, and summary statistics."""

from .leaderboard import Leaderboard
from .strategies import StrategyManager


def print_leaderboard(lb: Leaderboard, n: int = 10):
    """Print the top N trials."""
    top = lb.top_trials(n)
    if not top:
        print("\n  No trials recorded yet.")
        return

    print(f"\n{'='*80}")
    print(f"  LEADERBOARD — Top {min(n, len(top))} of {len(lb.trials)} trials")
    print(f"{'='*80}")
    print(f"  {'#':>3s} {'ID':>5s} {'Strategy':<22s} {'Screen':>8s} {'Valid':>8s} {'Confirm':>12s} {'Status':<10s}")
    print(f"  {'-'*75}")

    for rank, trial in enumerate(top, 1):
        screen = f"{trial.screen_ppl:.1f}" if trial.screen_ppl is not None else "-"
        valid = f"{trial.validate_ppl:.1f}" if trial.validate_ppl is not None else "-"
        if trial.confirm_ppl is not None:
            confirm = f"{trial.confirm_ppl:.1f}±{trial.confirm_std:.1f}"
        else:
            confirm = "-"
        print(f"  {rank:>3d} {trial.trial_id:>5d} {trial.strategy:<22s} "
              f"{screen:>8s} {valid:>8s} {confirm:>12s} {trial.status:<10s}")

    # Show best config details
    best = lb.best_trial()
    if best:
        print(f"\n  Best config (trial {best.trial_id}, PPL={best.best_ppl():.2f}):")
        for k, v in sorted(best.config.items()):
            print(f"    {k}: {v}")


def print_param_importance(lb: Leaderboard):
    """Print parameter importance rankings."""
    importance = lb.compute_param_importance()
    if not importance:
        print("\n  Not enough data for parameter importance (need >= 5 trials).")
        return

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  PARAMETER IMPORTANCE (correlation with PPL):")
    print(f"  {'Parameter':<22s} {'Importance':>12s} {'Bar':<20s}")
    print(f"  {'-'*55}")
    max_imp = max(v for _, v in sorted_imp) if sorted_imp else 1.0
    for name, imp in sorted_imp:
        bar_len = int(20 * imp / max_imp) if max_imp > 0 else 0
        bar = "#" * bar_len
        print(f"  {name:<22s} {imp:>12.3f} {bar}")


def print_strategy_stats(lb: Leaderboard, mgr: StrategyManager):
    """Print strategy performance stats."""
    stats = lb.strategy_stats()
    mgr_stats = mgr.get_stats()

    print(f"\n  STRATEGY STATS:")
    print(f"  {'Strategy':<22s} {'Trials':>7s} {'Accepted':>9s} {'Win%':>6s} {'Weight':>8s}")
    print(f"  {'-'*55}")

    for s in mgr_stats["weights"]:
        total = stats.get(s, {}).get("total", 0)
        accepted = stats.get(s, {}).get("accepted", 0)
        win_pct = f"{100*accepted/total:.0f}%" if total > 0 else "-"
        weight = mgr_stats["weights"].get(s, 0)
        print(f"  {s:<22s} {total:>7d} {accepted:>9d} {win_pct:>6s} {weight:>7.1f}%")


def print_session_report(lb: Leaderboard, mgr: StrategyManager,
                         session_trials: int, session_time: float):
    """Print a full session summary."""
    print(f"\n{'='*80}")
    print(f"  SESSION REPORT")
    print(f"  Trials this session: {session_trials}")
    print(f"  Total time: {session_time/60:.1f} min")
    print(f"  Total trials in leaderboard: {len(lb.trials)}")
    print(f"{'='*80}")

    print_leaderboard(lb)
    print_param_importance(lb)
    print_strategy_stats(lb, mgr)

    # Common diagnosis issues across all trials
    all_issues: dict[str, int] = {}
    for t in lb.trials:
        for issue in t.diagnosis_issues:
            all_issues[issue] = all_issues.get(issue, 0) + 1
    if all_issues:
        print(f"\n  COMMON ISSUES ACROSS TRIALS:")
        for issue, count in sorted(all_issues.items(), key=lambda x: -x[1]):
            print(f"    {issue}: {count} trials")

    # Compute time stats
    times = [t.total_time() for t in lb.trials if t.total_time() > 0]
    if times:
        avg_time = sum(times) / len(times)
        rejected_at_screen = sum(1 for t in lb.trials if t.reject_phase == "screen")
        total_with_result = sum(1 for t in lb.trials if t.best_ppl() < float("inf"))
        print(f"\n  TIME EFFICIENCY:")
        print(f"    Avg trial time: {avg_time/60:.1f} min")
        print(f"    Rejected at screen: {rejected_at_screen}/{total_with_result} "
              f"({100*rejected_at_screen/total_with_result:.0f}%)" if total_with_result > 0 else "")
    print()
