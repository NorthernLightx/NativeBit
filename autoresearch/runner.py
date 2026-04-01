"""Core autoresearch loop: diagnose → prescribe → train → evaluate → learn.

Two modes:
  - Autopilot: hyperparameter search with diagnostics (runs unattended)
  - Research checkpoint: when autopilot stagnates, writes NEXT_SESSION.md
    for Claude to read in the next conversation and make code-level changes
"""

import json
import os
import time
from typing import Optional

from .analyzer import Diagnosis, analyze_log, recommend_from_diagnosis
from .config_space import configs_similar, get_default_config, PARAM_SPACE
from .evaluator import evaluate_trial, SCREEN_STEPS
from .leaderboard import Leaderboard
from .report import print_leaderboard, print_session_report
from .strategies import StrategyManager
from .trial import Trial


MAX_DEDUP_RETRIES = 15
# After this many consecutive screen rejections, knob-turning is exhausted
STAGNATION_LIMIT = 12


class AutoresearchRunner:
    """Autonomous research loop that learns from its own experiments."""

    def __init__(self, log_dir: str = "logs", data_dir: str = "data"):
        self.log_dir = os.path.join(log_dir, "autoresearch")
        self.data_dir = data_dir
        self.lb = Leaderboard(os.path.join(self.log_dir, "leaderboard.json"))
        self.mgr = StrategyManager()
        self.session_trials = 0
        self.session_start = time.time()
        self._shutdown = False
        self.last_diagnosis: Optional[Diagnosis] = None

    def load_state(self) -> bool:
        """Load leaderboard and strategy state. Returns True if state existed."""
        loaded = self.lb.load()
        state_path = os.path.join(self.log_dir, "strategy_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                self.mgr = StrategyManager.from_dict(json.load(f))
        return loaded

    def save_state(self):
        """Save leaderboard and strategy state."""
        self.lb.save()
        os.makedirs(self.log_dir, exist_ok=True)
        state_path = os.path.join(self.log_dir, "strategy_state.json")
        with open(state_path, "w") as f:
            json.dump(self.mgr.to_dict(), f, indent=2)

    def bootstrap_from_doe(self, doe_results_path: str):
        """Import DoE configs and run screen evaluation on each.

        Instead of trusting DoE PPLs (which were trained for 5000 steps),
        we re-screen each config at SCREEN_STEPS (1500) so the baseline
        is comparable to all future autoresearch trials.
        """
        with open(doe_results_path, "r") as f:
            doe_results = json.load(f)

        imported = 0
        for i, r in enumerate(doe_results):
            if r.get("aborted"):
                continue
            overrides = r.get("overrides", {})
            config = _doe_to_autoresearch_config(overrides)
            if self.lb.is_duplicate(config):
                continue

            trial = Trial(
                trial_id=self.lb.next_trial_id(),
                strategy="bootstrap_doe",
                seed=42,
                config=config,
            )

            print(f"\n  Bootstrap {i+1}/{len(doe_results)}: screening DoE config")
            _print_config_summary(config)

            # Run full 3-phase evaluation against current best
            best_ppl = self.lb.best_ppl()
            best_screen_ppl = self.lb.best_screen_ppl()
            trial = evaluate_trial(trial, best_ppl, self.log_dir, self.data_dir,
                                   best_screen_ppl=best_screen_ppl)
            self.lb.add_trial(trial)
            imported += 1

            print(f"  Bootstrap trial {trial.trial_id}: {trial.status} "
                  f"(PPL={trial.best_ppl():.2f})")

        print(f"  Bootstrapped {imported} DoE configs (screened at {SCREEN_STEPS} steps)")
        self.save_state()

    def run(self, max_trials: int = 0, max_hours: float = 0):
        """Main autonomous loop.

        Runs hyperparameter search until:
        - max_trials or max_hours reached
        - Ctrl+C
        - Stagnation detected (no improvement in STAGNATION_LIMIT consecutive trials)

        On stagnation, writes NEXT_SESSION.md briefing for Claude to pick up
        and make code-level changes in the next conversation.
        """
        self.session_start = time.time()
        trial_count = 0
        consecutive_rejections = 0

        print(f"\n{'='*70}")
        print(f"  AUTORESEARCH — Autonomous Research Loop")
        print(f"  Max trials: {max_trials if max_trials > 0 else 'unlimited'}")
        print(f"  Max hours: {max_hours if max_hours > 0 else 'unlimited'}")
        print(f"  Current best PPL: {self.lb.best_ppl():.2f}")
        print(f"  Trials in leaderboard: {len(self.lb.trials)}")
        print(f"  Stagnation limit: {STAGNATION_LIMIT} consecutive rejections")
        print(f"{'='*70}\n")

        while not self._shutdown:
            # Check stopping conditions
            if max_trials > 0 and trial_count >= max_trials:
                print(f"\n  Reached max_trials={max_trials}")
                break
            if max_hours > 0:
                elapsed_h = (time.time() - self.session_start) / 3600
                if elapsed_h >= max_hours:
                    print(f"\n  Reached max_hours={max_hours}")
                    break

            # Stagnation check: knob-turning exhausted, need code changes
            if consecutive_rejections >= STAGNATION_LIMIT:
                print(f"\n  {'='*70}")
                print(f"  STAGNATION DETECTED — {consecutive_rejections} consecutive rejections")
                print(f"  Hyperparameter search cannot improve beyond PPL={self.lb.best_ppl():.2f}")
                print(f"  Writing research briefing for next session...")
                print(f"  {'='*70}")
                self._write_next_session()
                break

            # Update param importance for strategies
            self.mgr.param_importance = self.lb.compute_param_importance()

            # Pick strategy
            strategy = self.mgr.pick_strategy()

            # Sample config with full context
            best_config = self.lb.best_config()
            top_configs = self.lb.top_configs(5)
            all_tried = [t.config for t in self.lb.trials if t.best_ppl() < float("inf")]

            config = None
            for _ in range(MAX_DEDUP_RETRIES):
                candidate = self.mgr.sample_config(
                    strategy, best_config, top_configs,
                    last_diagnosis=self.last_diagnosis,
                    all_tried_configs=all_tried,
                )
                if not self.lb.is_duplicate(candidate):
                    config = candidate
                    break
            if config is None:
                from .config_space import sample_uniform
                config = sample_uniform()
                strategy = "random_exploration"

            # Create trial
            trial = Trial(
                trial_id=self.lb.next_trial_id(),
                strategy=strategy,
                seed=42,
                config=config,
            )

            best_ppl = self.lb.best_ppl()
            best_screen_ppl = self.lb.best_screen_ppl()
            trial_count += 1
            self.session_trials += 1

            print(f"\n{'='*70}")
            print(f"  TRIAL {trial.trial_id} (session #{trial_count}) — {strategy}")
            print(f"  Current best PPL: {best_ppl:.2f} (screen: {best_screen_ppl:.2f})")
            print(f"  Consecutive rejections: {consecutive_rejections}/{STAGNATION_LIMIT}")
            _print_config_summary(config)
            if self.last_diagnosis and self.last_diagnosis.issues:
                print(f"  Last trial issues: {', '.join(self.last_diagnosis.issues)}")
            print(f"{'='*70}")

            # Evaluate (3-phase protocol)
            try:
                trial = evaluate_trial(trial, best_ppl, self.log_dir, self.data_dir,
                                       best_screen_ppl=best_screen_ppl)
            except KeyboardInterrupt:
                print(f"\n  Interrupted during trial {trial.trial_id}")
                trial.status = "rejected"
                trial.reject_phase = "interrupted"
                self.lb.add_trial(trial)
                self._shutdown = True
                break

            # ── Analyze training logs ─────────────────────────────────────
            screen_log = os.path.join(
                self.log_dir, f"ar_trial{trial.trial_id:04d}_screen.jsonl"
            )
            diag = analyze_log(screen_log)
            if diag:
                trial.diagnosis_issues = diag.issues
                trial.diagnosis_recs = recommend_from_diagnosis(diag, config)
                self.last_diagnosis = diag

                if diag.issues:
                    print(f"\n  DIAGNOSIS: {', '.join(diag.issues)}")
                    print(f"    Dead entries: {diag.dead_pct_final:.1f}% "
                          f"(peak: {diag.dead_pct_peak:.1f}%)")
                    if diag.plateau:
                        print(f"    Plateau from {diag.plateau_start_frac:.0%} of training")
                    if diag.grad_ratio_high:
                        print(f"    Grad ratio: {diag.grad_ratio_mean:.1f}x (HIGH)")
                    if trial.diagnosis_recs:
                        print(f"    Recommendations: {trial.diagnosis_recs}")
                else:
                    print(f"\n  DIAGNOSIS: healthy (no issues detected)")

            # Record trial
            self.lb.add_trial(trial)
            is_win = trial.status == "accepted"
            self.mgr.record_result(strategy, is_win)
            self.save_state()

            # Track stagnation
            if is_win:
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1

            # Summary
            status_str = trial.status.upper()
            if trial.status == "rejected" and trial.reject_phase:
                status_str += f" at {trial.reject_phase}"
            print(f"\n  Trial {trial.trial_id}: {status_str} "
                  f"(PPL={trial.best_ppl():.2f}, time={trial.total_time()/60:.1f} min)")

            # Periodic detailed report
            if self.session_trials % 10 == 0:
                elapsed = time.time() - self.session_start
                print_session_report(self.lb, self.mgr, self.session_trials, elapsed)

            # Brief leaderboard after each trial
            print_leaderboard(self.lb, n=5)

        # Final report
        elapsed = time.time() - self.session_start
        print_session_report(self.lb, self.mgr, self.session_trials, elapsed)

    def request_shutdown(self):
        """Signal the loop to stop gracefully after current trial."""
        self._shutdown = True

    def _write_next_session(self):
        """Write NEXT_SESSION.md — a structured briefing for Claude to read.

        This is the handoff from autopilot (knob-turning) to research mode
        (code changes). Claude reads this at the start of the next conversation
        and implements the suggested code-level changes.
        """
        best = self.lb.best_trial()
        best_ppl = self.lb.best_ppl()
        top5 = self.lb.top_trials(5)
        importance = self.lb.compute_param_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Gather all diagnosis issues across recent trials
        recent = self.lb.trials[-STAGNATION_LIMIT:]
        issue_counts: dict[str, int] = {}
        for t in recent:
            for issue in t.diagnosis_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Gather what structural features were tried
        structural_tried = {
            "learned_distance": set(),
            "n_codebooks": set(),
            "factored_init": set(),
            "distill_alpha_nonzero": set(),
        }
        for t in self.lb.trials:
            c = t.config
            structural_tried["learned_distance"].add(c.get("learned_distance", False))
            structural_tried["n_codebooks"].add(c.get("n_codebooks", 1))
            structural_tried["factored_init"].add(c.get("factored_init", False))
            if c.get("distill_alpha", 0) > 0:
                structural_tried["distill_alpha_nonzero"].add(True)

        # Gather parameter ranges actually explored
        explored_ranges: dict[str, str] = {}
        all_tried = [t.config for t in self.lb.trials if t.best_ppl() < float("inf")]
        for pname, spec in PARAM_SPACE.items():
            vals = [c.get(pname) for c in all_tried if c.get(pname) is not None]
            if not vals:
                continue
            ptype = spec["type"]
            if ptype in ("continuous", "log-continuous", "integer"):
                nums = [float(v) for v in vals]
                explored_ranges[pname] = f"{min(nums):.4g} — {max(nums):.4g} (space: {spec.get('low', '?')} — {spec.get('high', '?')})"
            elif ptype in ("categorical", "boolean"):
                explored_ranges[pname] = str(sorted(set(str(v) for v in vals)))

        # Determine the dominant failure mode
        if issue_counts:
            top_issue = max(issue_counts, key=issue_counts.get)
            top_issue_pct = issue_counts[top_issue] / len(recent) * 100
        else:
            top_issue = "none (configs are healthy but not good enough)"
            top_issue_pct = 0

        # Build the briefing
        lines = [
            "# NEXT_SESSION.md — Autoresearch Handoff",
            "",
            "**Status**: Hyperparameter search stagnated. The autopilot has exhausted",
            f"knob-turning — {STAGNATION_LIMIT} consecutive trials failed to beat the",
            f"current best. Code-level changes are needed to improve further.",
            "",
            "## Current Best",
            f"- **PPL: {best_ppl:.2f}** (trial {best.trial_id}, strategy: {best.strategy})" if best else "- No best trial",
        ]
        if best:
            lines.append(f"- Config: {json.dumps(best.config, indent=2)}")
        lines += [
            "",
            f"## Search Summary",
            f"- Total trials run: {len(self.lb.trials)}",
            f"- Accepted: {sum(1 for t in self.lb.trials if t.status == 'accepted')}",
            f"- Rejected at screen: {sum(1 for t in self.lb.trials if t.reject_phase == 'screen')}",
            "",
            "## What Was Tried",
        ]
        for pname, range_str in explored_ranges.items():
            lines.append(f"- {pname}: {range_str}")
        lines += [
            "",
            "## Structural Features Tested",
            f"- learned_distance: {structural_tried['learned_distance']}",
            f"- n_codebooks (residual): {structural_tried['n_codebooks']}",
            f"- factored_init: {structural_tried['factored_init']}",
            f"- distillation: {'tried' if True in structural_tried['distill_alpha_nonzero'] else 'NOT tried'}",
            "",
            "## Parameter Importance (from all trials)",
        ]
        for pname, imp in sorted_imp[:7]:
            lines.append(f"- {pname}: {imp:.3f}")
        lines += [
            "",
            "## Dominant Failure Mode",
            f"- **{top_issue}** ({top_issue_pct:.0f}% of last {STAGNATION_LIMIT} trials)" if top_issue_pct > 0 else f"- {top_issue}",
            "",
            "## All Recent Issues",
        ]
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {issue}: {count}/{len(recent)} trials")
        if not issue_counts:
            lines.append("- No diagnostic issues detected — configs are healthy but PPL plateaued")
        lines += [
            "",
            "## Suggested Research Directions",
            "",
            "The autopilot cannot make code changes. The following require implementing",
            "new features in `nativebit/layers.py`, `nativebit/model.py`, or `train.py`:",
            "",
        ]

        # Generate specific suggestions based on patterns
        suggestions = _generate_research_suggestions(
            issue_counts, best.config if best else {}, importance, structural_tried
        )
        for i, (title, desc) in enumerate(suggestions, 1):
            lines.append(f"### {i}. {title}")
            lines.append(desc)
            lines.append("")

        lines += [
            "## How to Continue",
            "",
            "1. Read this file and the leaderboard (`python autoresearch_run.py --report`)",
            "2. Pick a research direction above (or propose your own)",
            "3. Implement the code change",
            "4. Add the new feature as a config parameter in `autoresearch/config_space.py`",
            "5. Run `python autoresearch_run.py --resume` to let the autopilot search",
            "   the expanded space",
            "",
            "After implementing a code change, reset stagnation by deleting",
            "`logs/autoresearch/strategy_state.json` to give the autopilot fresh weights.",
        ]

        path = os.path.join(os.path.dirname(self.log_dir), "NEXT_SESSION.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n  Research briefing written to: {path}")
        print(f"  Start next Claude conversation and ask to read it.")


# ── Helpers ───────────────────────────────────────────────────────────────

def _generate_research_suggestions(
    issue_counts: dict[str, int],
    best_config: dict,
    importance: dict[str, float],
    structural_tried: dict,
) -> list[tuple[str, str]]:
    """Generate prioritized research directions based on autopilot findings."""
    suggestions = []

    # Dead entries dominant → codebook algorithm needs work
    if issue_counts.get("high_dead_entries", 0) > 3:
        suggestions.append((
            "Adaptive revival mechanism",
            "Dead entries persist despite EMA + revival. Try: (a) per-block adaptive "
            "revival frequency (blocks with more dead entries get revived more often), "
            "(b) warm restart — reinit dead entries from the distribution of live ones "
            "instead of weight percentiles, (c) split the most-used entry into two "
            "when an entry dies (usage-based splitting)."
        ))

    # Plateau dominant → training dynamics need work
    if issue_counts.get("plateau", 0) > 3:
        suggestions.append((
            "Cyclic quantization annealing",
            "Loss plateaus suggest quantization noise is blocking gradient descent. "
            "Try periodically relaxing quantization (soft→hard→soft cycles) to let "
            "the optimizer escape local minima. Implement as a cyclical tau schedule "
            "in train.py instead of monotonic decay."
        ))

    # High grad ratio → codebook-weight interaction
    if issue_counts.get("high_grad_ratio", 0) > 2:
        suggestions.append((
            "Gradient-aware block sizing",
            "Some blocks have gradient ratios >5x, meaning codebook updates dominate "
            "weight updates. Try per-layer or per-block adaptive block_size: layers "
            "with higher grad ratios get larger blocks (less aggressive quantization)."
        ))

    # Healthy but stuck → need fundamentally new approach
    if not issue_counts or all(v <= 2 for v in issue_counts.values()):
        suggestions.append((
            "Quantization-aware knowledge distillation v2",
            "Configs are healthy but PPL won't drop further. The current distillation "
            "uses a separate float teacher trained simultaneously. Try: (a) train the "
            "float teacher FIRST for N steps, freeze it, then train the NativeBit "
            "student with KD loss, (b) progressive distillation — start with high "
            "distill_alpha and decay to 0."
        ))

    # Residual quantization not tried or underexplored
    if structural_tried.get("n_codebooks") == {1}:
        suggestions.append((
            "Residual quantization with learned routing",
            "n_codebooks=2 was never tried or always failed. The current residual "
            "implementation uses additive codebooks (w ≈ cb1[i] + cb2[j]). Try: "
            "(a) multiplicative interaction (w ≈ cb1[i] * cb2[j]), (b) learned "
            "gating (w ≈ g*cb1[i] + (1-g)*cb2[j] where g is a per-block scalar)."
        ))

    # n_codebook importance high → bit-width is the key lever
    if importance.get("n_codebook", 0) > 0.3:
        suggestions.append((
            "Mixed-precision quantization (per-layer bit width)",
            "n_codebook is the most important parameter, suggesting different layers "
            "need different bit widths. Implement per-layer n_codebook selection: "
            "attention layers get more entries (higher precision), FFN layers get "
            "fewer. Could be fixed schedule or learned via entropy-based criteria."
        ))

    # block_size importance high
    if importance.get("block_size", 0) > 0.3:
        suggestions.append((
            "Adaptive block sizing",
            "block_size has high impact. Different weight matrices may benefit from "
            "different granularity. Implement per-layer block_size: small blocks for "
            "attention (fine-grained), large blocks for FFN (parameter-efficient). "
            "Or: start with large blocks and progressively split during training."
        ))

    # Always suggest: longer training on best config
    suggestions.append((
        "Scale up the best config",
        "Run the current best config for 15k-30k steps on WikiText-2 or switch to "
        "TinyStories with MediumConfig. The screen phase (1500 steps) may not be "
        "long enough to see the full potential of some configs. If scaling confirms "
        "the ranking, the autopilot's search was effective; if not, the screen "
        "phase needs to be longer."
    ))

    return suggestions


def _doe_to_autoresearch_config(overrides: dict) -> dict:
    """Convert DoE override dict to autoresearch config dict."""
    config = get_default_config()

    mapping = {
        "A_n_codebook": "n_codebook",
        "B_block_size": "block_size",
        "C_ema_decay": "ema_decay",
        "G_lr": "learning_rate",
        "F_weight_decay": "weight_decay",
        "H_delay_quant": "delay_quant_steps",
        "J_entropy_lambda": "entropy_lambda",
        "E_factored_init": "factored_init",
        "I_distill_alpha": "distill_alpha",
        "D_learned_distance": "learned_distance",
    }

    for doe_key, ar_key in mapping.items():
        if doe_key in overrides:
            val = overrides[doe_key]
            if ar_key in ("factored_init", "learned_distance"):
                val = bool(val)
            config[ar_key] = val

    bs_attn = overrides.get("K_block_size_attn", 0)
    config["block_size_attn"] = bs_attn if bs_attn > 0 else None

    return config


def _print_config_summary(config: dict):
    """Print a compact config summary."""
    parts = []
    parts.append(f"cb={config.get('n_codebook', '?')}")
    parts.append(f"bs={config.get('block_size', '?')}")
    decay = config.get('ema_decay', '?')
    parts.append(f"decay={decay:.4f}" if isinstance(decay, float) else f"decay={decay}")
    lr = config.get('learning_rate', '?')
    parts.append(f"lr={lr:.1e}" if isinstance(lr, float) else f"lr={lr}")
    wd = config.get('weight_decay', '?')
    parts.append(f"wd={wd:.3f}" if isinstance(wd, float) else f"wd={wd}")
    if config.get("delay_quant_steps", 0) > 0:
        parts.append(f"delay={config['delay_quant_steps']}")
    if config.get("entropy_lambda", 0) > 0:
        parts.append(f"ent={config['entropy_lambda']:.4f}")
    if config.get("factored_init"):
        parts.append("fac_init")
    if config.get("learned_distance"):
        parts.append("lrn_dist")
    if config.get("n_codebooks", 1) > 1:
        parts.append(f"n_cb={config['n_codebooks']}")
    if config.get("distill_alpha", 0) > 0:
        parts.append(f"distill={config['distill_alpha']:.2f}")
    if config.get("block_size_attn") is not None:
        parts.append(f"bs_attn={config['block_size_attn']}")
    print(f"  Config: {' | '.join(parts)}")
