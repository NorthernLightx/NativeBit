"""Persistent JSON leaderboard with deduplication and parameter importance."""

import json
import os
import math
from typing import Optional

from .trial import Trial
from .config_space import configs_similar, PARAM_SPACE


class Leaderboard:
    """Manages the persistent leaderboard of trials."""

    def __init__(self, path: str):
        self.path = path
        self.trials: list[Trial] = []
        self._next_id = 1

    def load(self) -> bool:
        """Load from JSON file. Returns True if loaded successfully."""
        if not os.path.exists(self.path):
            return False
        with open(self.path, "r") as f:
            data = json.load(f)
        self.trials = [Trial.from_dict(t) for t in data.get("trials", [])]
        self._next_id = data.get("next_id", len(self.trials) + 1)
        return True

    def save(self):
        """Save to JSON file."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "next_id": self._next_id,
            "trials": [t.to_dict() for t in self.trials],
        }
        # Sanitize inf/nan to null for valid JSON
        def _sanitize(obj):
            if isinstance(obj, float):
                if obj != obj or obj == float("inf") or obj == float("-inf"):
                    return None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj
        data = _sanitize(data)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def next_trial_id(self) -> int:
        """Get and increment the next trial ID."""
        tid = self._next_id
        self._next_id += 1
        return tid

    def add_trial(self, trial: Trial):
        """Add a completed trial to the leaderboard."""
        self.trials.append(trial)
        self.save()

    def is_duplicate(self, config: dict) -> bool:
        """Check if a config is too similar to any existing trial."""
        for trial in self.trials:
            if configs_similar(config, trial.config):
                return True
        return False

    def best_trial(self) -> Optional[Trial]:
        """Return the trial with the lowest confirmed/validated/screened PPL."""
        accepted = [t for t in self.trials if t.status == "accepted"]
        if accepted:
            return min(accepted, key=lambda t: t.best_ppl())
        # Fall back to any trial with results
        with_results = [t for t in self.trials if t.best_ppl() < float("inf")]
        if with_results:
            return min(with_results, key=lambda t: t.best_ppl())
        return None

    def best_ppl(self) -> float:
        """Return the best PPL across all trials."""
        best = self.best_trial()
        return best.best_ppl() if best else float("inf")

    def best_screen_ppl(self) -> float:
        """Return the screen PPL of the best accepted trial.

        Used for screen-phase thresholding: new trials should be compared
        against the champion's screen PPL (not confirm PPL) since screen
        runs for fewer steps and naturally has higher PPL.
        """
        best = self.best_trial()
        if best and best.screen_ppl is not None:
            return best.screen_ppl
        return float("inf")

    def best_config(self) -> Optional[dict]:
        """Return the config of the best trial."""
        best = self.best_trial()
        return best.config if best else None

    def top_configs(self, n: int = 5) -> list[dict]:
        """Return the top N configs by PPL."""
        with_results = [t for t in self.trials if t.best_ppl() < float("inf")]
        sorted_trials = sorted(with_results, key=lambda t: t.best_ppl())
        return [t.config for t in sorted_trials[:n]]

    def top_trials(self, n: int = 10) -> list[Trial]:
        """Return the top N trials by PPL."""
        with_results = [t for t in self.trials if t.best_ppl() < float("inf")]
        return sorted(with_results, key=lambda t: t.best_ppl())[:n]

    def compute_param_importance(self) -> dict[str, float]:
        """Compute parameter importance as correlation between param values and PPL.

        Returns dict mapping param_name -> importance score (higher = more important).
        Uses absolute Pearson correlation for continuous params, and PPL difference
        for categorical/boolean.
        """
        completed = [t for t in self.trials if t.best_ppl() < float("inf") and not t.aborted]
        if len(completed) < 5:
            return {}

        importance = {}
        ppls = [t.best_ppl() for t in completed]
        mean_ppl = sum(ppls) / len(ppls)

        for param_name, spec in PARAM_SPACE.items():
            ptype = spec["type"]
            values = [t.config.get(param_name) for t in completed]

            if ptype in ("continuous", "log-continuous", "integer"):
                # Pearson correlation
                numeric_vals = []
                numeric_ppls = []
                for v, p in zip(values, ppls):
                    if v is not None:
                        numeric_vals.append(float(v) if ptype != "log-continuous" else math.log(max(float(v), 1e-10)))
                        numeric_ppls.append(p)
                if len(numeric_vals) < 3:
                    importance[param_name] = 0.0
                    continue
                mean_v = sum(numeric_vals) / len(numeric_vals)
                mean_p = sum(numeric_ppls) / len(numeric_ppls)
                cov = sum((v - mean_v) * (p - mean_p) for v, p in zip(numeric_vals, numeric_ppls))
                var_v = sum((v - mean_v) ** 2 for v in numeric_vals)
                var_p = sum((p - mean_p) ** 2 for p in numeric_ppls)
                denom = (var_v * var_p) ** 0.5
                corr = cov / denom if denom > 0 else 0.0
                importance[param_name] = abs(corr)

            elif ptype in ("categorical", "boolean"):
                # Group PPLs by value, measure spread of group means
                groups: dict = {}
                for v, p in zip(values, ppls):
                    key = str(v)
                    groups.setdefault(key, []).append(p)
                if len(groups) < 2:
                    importance[param_name] = 0.0
                    continue
                group_means = [sum(ps) / len(ps) for ps in groups.values()]
                spread = max(group_means) - min(group_means)
                # Normalize by overall PPL range
                ppl_range = max(ppls) - min(ppls)
                importance[param_name] = spread / ppl_range if ppl_range > 0 else 0.0

            else:
                importance[param_name] = 0.0

        return importance

    def strategy_stats(self) -> dict:
        """Count wins and totals per strategy."""
        stats: dict = {}
        for trial in self.trials:
            s = trial.strategy
            if s not in stats:
                stats[s] = {"total": 0, "accepted": 0, "rejected": 0}
            stats[s]["total"] += 1
            if trial.status == "accepted":
                stats[s]["accepted"] += 1
            elif trial.status == "rejected":
                stats[s]["rejected"] += 1
        return stats
