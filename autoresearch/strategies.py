"""Search strategies: diagnostic-driven, novelty exploration, and adaptive weights."""

import random
from typing import Optional

from .analyzer import (
    Diagnosis,
    config_distance,
    find_underexplored_params,
    recommend_from_diagnosis,
)
from .config_space import (
    PARAM_SPACE,
    _perturb_param,
    _sample_param,
    crossover,
    param_names,
    perturb_config,
    sample_uniform,
)


# ── Strategy definitions ──────────────────────────────────────────────────

STRATEGY_NAMES = [
    "diagnostic_driven",
    "local_perturbation",
    "novelty_exploration",
    "random_exploration",
    "combination_mining",
]

DEFAULT_WEIGHTS = {
    "diagnostic_driven": 40.0,
    "local_perturbation": 25.0,
    "novelty_exploration": 20.0,
    "random_exploration": 10.0,
    "combination_mining": 5.0,
}

STAGNATION_THRESHOLD = 8
MIN_WEIGHT = 2.0
MAX_WEIGHT = 60.0


class StrategyManager:
    """Manages strategy selection, adaptive weights, and stagnation detection."""

    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.wins = {s: 0 for s in STRATEGY_NAMES}
        self.trials_since_win = {s: 0 for s in STRATEGY_NAMES}
        self.total_trials = {s: 0 for s in STRATEGY_NAMES}
        self.param_importance: dict[str, float] = {}

    def pick_strategy(self) -> str:
        """Weighted random selection, boosted by stagnation."""
        weights = dict(self.weights)

        # Global stagnation: if no strategy has won recently, boost exploration
        if self.trials_since_win:
            min_since_win = min(self.trials_since_win.values())
            if min_since_win >= STAGNATION_THRESHOLD:
                weights["novelty_exploration"] *= 2.5
                weights["random_exploration"] *= 2.0
                weights["diagnostic_driven"] *= 0.7

        names = list(weights.keys())
        w = [weights[n] for n in names]
        return random.choices(names, weights=w, k=1)[0]

    def sample_config(
        self,
        strategy: str,
        best_config: Optional[dict],
        top_configs: list[dict],
        last_diagnosis: Optional[Diagnosis] = None,
        all_tried_configs: Optional[list[dict]] = None,
    ) -> dict:
        """Sample a config according to the chosen strategy."""

        if best_config is None:
            return sample_uniform()

        if strategy == "diagnostic_driven":
            return _diagnostic_driven(best_config, last_diagnosis)
        elif strategy == "local_perturbation":
            total = sum(self.total_trials.values())
            return _local_perturbation(best_config, n_explored=total)
        elif strategy == "novelty_exploration":
            return _novelty_exploration(
                all_tried_configs or [],
                best_config,
            )
        elif strategy == "random_exploration":
            return sample_uniform()
        elif strategy == "combination_mining":
            return _combination_mining(best_config, top_configs)
        else:
            return sample_uniform()

    def record_result(self, strategy: str, is_win: bool):
        """Update weights based on trial outcome."""
        self.total_trials[strategy] = self.total_trials.get(strategy, 0) + 1

        if is_win:
            self.wins[strategy] = self.wins.get(strategy, 0) + 1
            self.trials_since_win[strategy] = 0
            self.weights[strategy] = min(MAX_WEIGHT, self.weights[strategy] * 1.3)
        else:
            self.trials_since_win[strategy] = self.trials_since_win.get(strategy, 0) + 1
            if self.trials_since_win[strategy] >= STAGNATION_THRESHOLD:
                self.weights[strategy] = max(MIN_WEIGHT, self.weights[strategy] * 0.85)

        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for s in self.weights:
                self.weights[s] = self.weights[s] / total * 100.0

    def get_stats(self) -> dict:
        return {
            "weights": {s: round(w, 1) for s, w in self.weights.items()},
            "wins": dict(self.wins),
            "total_trials": dict(self.total_trials),
        }

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "wins": self.wins,
            "trials_since_win": self.trials_since_win,
            "total_trials": self.total_trials,
            "param_importance": self.param_importance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyManager":
        mgr = cls()
        mgr.weights = d.get("weights", dict(DEFAULT_WEIGHTS))
        mgr.wins = d.get("wins", {s: 0 for s in STRATEGY_NAMES})
        mgr.trials_since_win = d.get("trials_since_win", {s: 0 for s in STRATEGY_NAMES})
        mgr.total_trials = d.get("total_trials", {s: 0 for s in STRATEGY_NAMES})
        mgr.param_importance = d.get("param_importance", {})
        return mgr


# ── Strategy implementations ──────────────────────────────────────────────

def _diagnostic_driven(best_config: dict, diagnosis: Optional[Diagnosis]) -> dict:
    """Analyze what went wrong in the last trial, prescribe a targeted fix.

    If diagnosis is None (first trial or no log), falls back to
    sampling an underexplored parameter.
    """
    if diagnosis is None:
        # No diagnosis yet — try a structural feature not in the best config
        return _try_untested_structural(best_config)

    new_config = dict(best_config)

    # Get concrete recommendations from the analyzer
    recs = recommend_from_diagnosis(diagnosis, best_config)

    if recs:
        # Apply all recommendations
        for param, value in recs.items():
            if param in PARAM_SPACE:
                new_config[param] = value

        # Add slight noise to 1 unrecommended param to avoid exact repeats
        unrecommended = [p for p in PARAM_SPACE if p not in recs]
        if unrecommended:
            noise_param = random.choice(unrecommended)
            new_config[noise_param] = _perturb_param(
                new_config[noise_param], PARAM_SPACE[noise_param], strength=0.5
            )
    else:
        # Diagnosis found no issues — try a structural change
        new_config = _try_untested_structural(best_config)

    return new_config


def _try_untested_structural(config: dict) -> dict:
    """Toggle a structural feature that hasn't been tried in the best config."""
    new_config = dict(config)

    # Priority order for structural experiments
    structural_options = [
        ("learned_distance", True, "learned_distance not tried"),
        ("n_codebooks", 2, "residual quantization not tried"),
        ("factored_init", True, "factored init not tried"),
        ("distill_alpha", 0.2, "distillation not tried"),
    ]

    for param, target_val, _reason in structural_options:
        current = config.get(param, PARAM_SPACE[param]["default"] if param in PARAM_SPACE else None)
        if current != target_val:
            new_config[param] = target_val
            if param == "distill_alpha":
                new_config["distill_temp"] = 2.0
            # Also perturb 1-2 other params for variety
            new_config = perturb_config(new_config, n_params=1, strength=0.7)
            return new_config

    # All structural options tried — do a wider perturbation
    return perturb_config(config, n_params=3, strength=1.5)


def _local_perturbation(best_config: dict, n_explored: int = 0) -> dict:
    """Hill-climbing from best config. Gets wider as local area gets explored."""
    if n_explored < 10:
        n = random.choice([1, 2])
        strength = 1.0
    elif n_explored < 30:
        n = random.choice([1, 2, 3])
        strength = 1.3
    else:
        n = random.choice([2, 3, 4])
        strength = 1.8
    return perturb_config(best_config, n_params=n, strength=strength)


def _novelty_exploration(
    all_tried_configs: list[dict],
    best_config: dict,
    n_candidates: int = 40,
) -> dict:
    """Generate the most different config from everything tried so far.

    Uses maximin design: generate random candidates, pick the one with
    maximum minimum-distance to any tried config.
    """
    if not all_tried_configs:
        return sample_uniform()

    # Find underexplored parameters and bias sampling towards them
    underexplored = find_underexplored_params(all_tried_configs)

    candidates = []
    for _ in range(n_candidates):
        # 50% fully random, 50% biased towards underexplored params
        if random.random() < 0.5 or not underexplored:
            candidates.append(sample_uniform())
        else:
            # Start from best, but randomize underexplored params
            c = dict(best_config)
            for param in underexplored:
                if param in PARAM_SPACE:
                    c[param] = _sample_param(PARAM_SPACE[param])
            # Also perturb a couple explored params
            explored = [p for p in PARAM_SPACE if p not in underexplored]
            if explored:
                for p in random.sample(explored, min(2, len(explored))):
                    c[p] = _perturb_param(c[p], PARAM_SPACE[p], strength=1.5)
            candidates.append(c)

    # Pick candidate with maximum minimum-distance to tried configs
    best_candidate = candidates[0]
    best_min_dist = -1.0

    for c in candidates:
        min_dist = min(config_distance(c, t) for t in all_tried_configs)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_candidate = c

    return best_candidate


def _combination_mining(best_config: dict, top_configs: list[dict]) -> dict:
    """Crossover between top configs, with mutation."""
    if len(top_configs) >= 2:
        parents = random.sample(top_configs[:5], 2)
        child = crossover(parents[0], parents[1])
    elif top_configs:
        child = crossover(best_config, top_configs[0])
    else:
        child = dict(best_config)

    # Always mutate 1 param to avoid exact copies
    child = perturb_config(child, n_params=1, strength=1.0)
    return child
