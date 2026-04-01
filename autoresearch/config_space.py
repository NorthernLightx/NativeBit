"""Parameter space definition, sampling, and perturbation for autoresearch."""

import math
import random
from typing import Any


# ── Parameter definitions ──────────────────────────────────────────────────

PARAM_SPACE = {
    "n_codebook": {
        "type": "categorical",
        "values": [4, 6, 8, 12, 16],
        "default": 8,
    },
    "block_size": {
        "type": "categorical",
        "values": [16, 32, 64, 128, 256],
        "default": 64,
    },
    "ema_decay": {
        "type": "continuous",
        "low": 0.95,
        "high": 0.9999,
        "default": 0.99,
    },
    "learning_rate": {
        "type": "log-continuous",
        "low": 5e-5,
        "high": 2e-3,
        "default": 3e-4,
    },
    "weight_decay": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.2,
        "default": 0.1,
    },
    "delay_quant_steps": {
        "type": "integer",
        "low": 0,
        "high": 300,  # capped — higher values cause loss spikes when quant kicks in
        "default": 0,
    },
    "entropy_lambda": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.1,
        "default": 0.0,
    },
    "block_size_attn": {
        "type": "categorical",
        "values": [None, 16, 32, 64, 128],
        "default": None,  # means use block_size
    },
    "requantize_every": {
        "type": "categorical",
        "values": [1, 3, 5, 10, 20, 50],
        "default": 10,
    },
    "factored_init": {
        "type": "boolean",
        "default": False,
    },
    "distill_alpha": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.5,
        "default": 0.0,
    },
    "distill_temp": {
        "type": "continuous",
        "low": 1.0,
        "high": 4.0,
        "default": 2.0,
    },
    "learned_distance": {
        "type": "boolean",
        "default": False,
    },
    "n_codebooks": {
        "type": "categorical",
        "values": [1, 2],
        "default": 1,
    },
    "n_codebook_attn": {
        "type": "categorical",
        "values": [None, 4, 6, 8, 12],
        "default": None,  # None = use n_codebook
    },
    "n_codebook_ffn": {
        "type": "categorical",
        "values": [None, 2, 4, 6, 8],
        "default": None,  # None = use n_codebook
    },
    "quant_warmup_steps": {
        "type": "integer",
        "low": 0,
        "high": 500,
        "default": 0,
    },
    "quant_dropout": {
        "type": "continuous",
        "low": 0.0,
        "high": 0.3,
        "default": 0.0,
    },
}


def get_default_config() -> dict:
    """Return the default config as a dict."""
    return {name: spec["default"] for name, spec in PARAM_SPACE.items()}


def sample_uniform() -> dict:
    """Sample a config uniformly from the parameter space."""
    config = {}
    for name, spec in PARAM_SPACE.items():
        config[name] = _sample_param(spec)
    return config


def _sample_param(spec: dict) -> Any:
    """Sample a single parameter from its spec."""
    ptype = spec["type"]
    if ptype == "categorical":
        return random.choice(spec["values"])
    elif ptype == "boolean":
        return random.choice([True, False])
    elif ptype == "continuous":
        return random.uniform(spec["low"], spec["high"])
    elif ptype == "log-continuous":
        log_low = math.log(spec["low"])
        log_high = math.log(spec["high"])
        return math.exp(random.uniform(log_low, log_high))
    elif ptype == "integer":
        return random.randint(spec["low"], spec["high"])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def perturb_config(config: dict, n_params: int = 1, strength: float = 1.0) -> dict:
    """Take a config and randomly tweak n_params parameters.

    Args:
        config: Base config to perturb.
        n_params: Number of params to change.
        strength: Perturbation magnitude multiplier (1.0 = normal, 2.0 = wider).
    """
    new_config = ensure_all_params(config)
    all_params = list(PARAM_SPACE.keys())
    to_perturb = random.sample(all_params, min(n_params, len(all_params)))

    for name in to_perturb:
        spec = PARAM_SPACE[name]
        current = config.get(name, spec["default"])
        new_config[name] = _perturb_param(current, spec, strength)

    return new_config


def _perturb_param(current: Any, spec: dict, strength: float = 1.0) -> Any:
    """Perturb a single parameter value.

    strength=1.0 gives ~15% of range gaussian noise.
    strength=2.0 gives ~30%, making it likely to jump further.
    """
    ptype = spec["type"]
    sigma_base = 0.15 * strength

    if ptype == "categorical":
        values = spec["values"]
        if current in values:
            idx = values.index(current)
            # At strength>1.5, allow jumping to any value (not just adjacent)
            if strength > 1.5 or len(values) <= 3:
                return random.choice(values)
            delta = random.choice([-1, 1])
            new_idx = max(0, min(len(values) - 1, idx + delta))
            return values[new_idx]
        return random.choice(values)
    elif ptype == "boolean":
        return not current
    elif ptype == "continuous":
        range_size = spec["high"] - spec["low"]
        delta = random.gauss(0, sigma_base * range_size)
        return max(spec["low"], min(spec["high"], current + delta))
    elif ptype == "log-continuous":
        log_current = math.log(max(current, 1e-10))
        log_range = math.log(spec["high"]) - math.log(spec["low"])
        delta = random.gauss(0, sigma_base * log_range)
        new_val = math.exp(log_current + delta)
        return max(spec["low"], min(spec["high"], new_val))
    elif ptype == "integer":
        range_size = spec["high"] - spec["low"]
        delta = round(random.gauss(0, (sigma_base + 0.05) * range_size))
        return max(spec["low"], min(spec["high"], current + delta))
    else:
        return current


def ensure_all_params(config: dict) -> dict:
    """Ensure config has all params from PARAM_SPACE, filling missing with defaults."""
    full = dict(config)
    for name, spec in PARAM_SPACE.items():
        if name not in full:
            full[name] = spec["default"]
    return full


def push_to_extreme(config: dict, param_name: str) -> dict:
    """Push one parameter to its boundary (low or high, chosen randomly)."""
    new_config = ensure_all_params(config)
    spec = PARAM_SPACE[param_name]
    ptype = spec["type"]

    if ptype == "categorical":
        values = spec["values"]
        new_config[param_name] = random.choice([values[0], values[-1]])
    elif ptype == "boolean":
        new_config[param_name] = random.choice([True, False])
    elif ptype in ("continuous", "log-continuous"):
        new_config[param_name] = random.choice([spec["low"], spec["high"]])
    elif ptype == "integer":
        new_config[param_name] = random.choice([spec["low"], spec["high"]])

    return new_config


def crossover(config_a: dict, config_b: dict) -> dict:
    """Create a child config by randomly selecting each param from one parent."""
    a = ensure_all_params(config_a)
    b = ensure_all_params(config_b)
    child = {}
    for name in PARAM_SPACE:
        child[name] = random.choice([a[name], b[name]])
    return child


def configs_similar(a: dict, b: dict, tolerance: float = 0.05) -> bool:
    """Check if two configs are within tolerance on all continuous params.

    Two configs are "similar" only if ALL params are close. Any single
    differing categorical/boolean or continuous param outside tolerance
    makes them distinct.
    """
    for name, spec in PARAM_SPACE.items():
        va, vb = a.get(name), b.get(name)
        if va is None or vb is None:
            if va != vb:
                return False
            continue

        ptype = spec["type"]
        if ptype in ("categorical", "boolean"):
            if va != vb:
                return False
        elif ptype in ("continuous", "log-continuous"):
            range_size = spec.get("high", 1) - spec.get("low", 0)
            if range_size > 0 and abs(va - vb) / range_size > tolerance:
                return False
        elif ptype == "integer":
            range_size = spec.get("high", 1) - spec.get("low", 0)
            if range_size > 0 and abs(va - vb) / range_size > tolerance:
                return False
    return True


def param_names() -> list[str]:
    """Return all parameter names."""
    return list(PARAM_SPACE.keys())
