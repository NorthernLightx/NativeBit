"""Training log analyzer — reads JSONL logs, diagnoses problems, recommends fixes."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .config_space import PARAM_SPACE


@dataclass
class Diagnosis:
    """Structured diagnosis from analyzing a training run's logs."""

    # Final metrics
    final_ppl: float = float("inf")
    final_loss: float = float("inf")
    steps_completed: int = 0

    # Codebook health
    dead_pct_final: float = 0.0
    dead_pct_peak: float = 0.0
    dead_improving: bool = False  # dead entries decreased during training

    # Convergence
    converged: bool = False
    diverged: bool = False
    plateau: bool = False
    plateau_start_frac: float = 1.0  # 0=beginning, 1=end
    late_improvement_frac: float = 0.0  # fractional PPL improvement in last 30%

    # Gradients
    grad_ratio_mean: float = 0.0
    grad_ratio_high: bool = False

    # Issues (machine-readable tags)
    issues: list[str] = field(default_factory=list)

    def severity(self) -> int:
        """0=healthy, 1=minor issues, 2=significant, 3=broken."""
        if self.diverged:
            return 3
        if "high_dead_entries" in " ".join(self.issues) or self.dead_pct_final > 15:
            return 2
        if self.issues:
            return 1
        return 0


def analyze_log(log_path: str) -> Optional[Diagnosis]:
    """Parse a JSONL training log and produce a diagnosis."""
    if not os.path.exists(log_path):
        return None

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if len(records) < 3:
        return None

    diag = Diagnosis()

    # Final metrics
    diag.final_ppl = records[-1].get("perplexity", float("inf"))
    diag.final_loss = records[-1].get("loss", float("inf"))
    diag.steps_completed = records[-1].get("step", 0)

    # ── Dead entries ──────────────────────────────────────────────────────
    dead_pcts = [r["dead_pct"] for r in records if "dead_pct" in r]
    if dead_pcts:
        diag.dead_pct_final = dead_pcts[-1]
        diag.dead_pct_peak = max(dead_pcts)
        n = len(dead_pcts)
        if n >= 6:
            early = sum(dead_pcts[:n // 3]) / (n // 3)
            late = sum(dead_pcts[-(n // 3):]) / (n // 3)
            diag.dead_improving = late < early * 0.7

    # ── Convergence ───────────────────────────────────────────────────────
    ppls = [r.get("perplexity", float("inf")) for r in records]
    losses = [r.get("loss", float("inf")) for r in records]

    # Diverged?
    if ppls[-1] > 1e6 or losses[-1] > 15:
        diag.diverged = True
        diag.issues.append("diverged")

    # Plateau: did the last 30% of training barely improve?
    n = len(ppls)
    if n >= 10 and not diag.diverged:
        cutoff = int(n * 0.7)
        late_ppls = ppls[cutoff:]
        if len(late_ppls) >= 3 and late_ppls[0] > 0:
            diag.late_improvement_frac = (late_ppls[0] - late_ppls[-1]) / late_ppls[0]
            if diag.late_improvement_frac < 0.02:
                diag.plateau = True
                # Find where improvement stopped
                for i in range(n - 1, 0, -1):
                    if i > 0 and ppls[i - 1] > 0:
                        step_improvement = (ppls[i - 1] - ppls[i]) / ppls[i - 1]
                        if step_improvement > 0.01:
                            diag.plateau_start_frac = i / n
                            break

    diag.converged = not diag.diverged and diag.final_ppl < 1e5

    # ── Gradient ratio ────────────────────────────────────────────────────
    grad_ratios = [r["grad_ratio_cb_w"] for r in records if "grad_ratio_cb_w" in r and r["grad_ratio_cb_w"] > 0]
    if grad_ratios:
        diag.grad_ratio_mean = sum(grad_ratios) / len(grad_ratios)
        diag.grad_ratio_high = diag.grad_ratio_mean > 5.0

    # ── Build issue tags ──────────────────────────────────────────────────
    if diag.dead_pct_final > 15:
        diag.issues.append("high_dead_entries")
    elif diag.dead_pct_final > 5:
        diag.issues.append("moderate_dead_entries")

    if diag.plateau:
        diag.issues.append("plateau")

    if diag.grad_ratio_high:
        diag.issues.append("high_grad_ratio")

    # Fast initial drop then stagnation → possible codebook collapse
    if n >= 10 and not diag.diverged:
        early_ppls = ppls[:n // 3]
        if len(early_ppls) >= 2 and early_ppls[0] > 0:
            early_drop = (early_ppls[0] - early_ppls[-1]) / early_ppls[0]
            if early_drop > 0.3 and diag.plateau:
                diag.issues.append("fast_then_stuck")

    # Late loss spike: loss jumps >3x the running average in the back half
    if n >= 10 and not diag.diverged:
        mid = n // 2
        back_half = losses[mid:]
        if len(back_half) >= 4:
            running_avg = sum(back_half[:3]) / 3
            for l in back_half[3:]:
                if running_avg > 0 and l > running_avg * 3:
                    diag.issues.append("late_loss_spike")
                    break
                running_avg = 0.9 * running_avg + 0.1 * l

    return diag


def recommend_from_diagnosis(
    diag: Diagnosis,
    current_config: dict,
) -> dict[str, Any]:
    """Map diagnosis issues to concrete parameter recommendations.

    Returns dict of param_name -> suggested_value. The strategy layer
    decides how strictly to follow these.
    """
    if diag is None:
        return {}

    recs: dict[str, Any] = {}

    # ── Diverged: pull back everything ────────────────────────────────────
    if diag.diverged:
        recs["learning_rate"] = current_config.get("learning_rate", 3e-4) * 0.5
        wd = current_config.get("weight_decay", 0.1)
        recs["weight_decay"] = min(wd + 0.03, PARAM_SPACE["weight_decay"]["high"])
        return recs

    # ── Dead entries: faster codebook adaptation ──────────────────────────
    if diag.dead_pct_final > 15:
        decay = current_config.get("ema_decay", 0.99)
        recs["ema_decay"] = max(PARAM_SPACE["ema_decay"]["low"], decay - 0.02)
        cb = current_config.get("n_codebook", 8)
        cb_vals = PARAM_SPACE["n_codebook"]["values"]
        if cb in cb_vals:
            idx = cb_vals.index(cb)
            if idx > 0:
                recs["n_codebook"] = cb_vals[idx - 1]
        bs = current_config.get("block_size", 64)
        bs_vals = PARAM_SPACE["block_size"]["values"]
        if bs in bs_vals:
            idx = bs_vals.index(bs)
            if idx > 0:
                recs["block_size"] = bs_vals[idx - 1]

    elif diag.dead_pct_final > 5:
        decay = current_config.get("ema_decay", 0.99)
        recs["ema_decay"] = max(PARAM_SPACE["ema_decay"]["low"], decay - 0.01)

    # ── Plateau: needs more learning or float warmup ──────────────────────
    if diag.plateau:
        delay = current_config.get("delay_quant_steps", 0)
        if delay == 0:
            recs["delay_quant_steps"] = 300
        else:
            # Already using delay, try higher LR
            lr = current_config.get("learning_rate", 3e-4)
            recs["learning_rate"] = min(lr * 1.5, PARAM_SPACE["learning_rate"]["high"])

    # ── Fast then stuck: quantization is disrupting a good start ──────────
    if "fast_then_stuck" in diag.issues:
        delay = current_config.get("delay_quant_steps", 0)
        recs["delay_quant_steps"] = max(delay + 200, 500)

    # ── High gradient ratio: codebook updates too aggressive ──────────────
    if diag.grad_ratio_high:
        bs = current_config.get("block_size", 64)
        bs_vals = PARAM_SPACE["block_size"]["values"]
        if bs in bs_vals:
            idx = bs_vals.index(bs)
            if idx > 0:
                recs["block_size"] = bs_vals[idx - 1]

    # ── Late loss spike: delayed quantization disrupted training ─────────
    if "late_loss_spike" in diag.issues:
        delay = current_config.get("delay_quant_steps", 0)
        if delay > 0:
            recs["delay_quant_steps"] = max(0, delay // 2)
        else:
            lr = current_config.get("learning_rate", 3e-4)
            recs["learning_rate"] = lr * 0.7

    # ── No obvious issues but PPL isn't great: try structural changes ─────
    if not diag.issues and diag.converged:
        if not current_config.get("learned_distance", False):
            recs["learned_distance"] = True
        elif current_config.get("n_codebooks", 1) == 1:
            recs["n_codebooks"] = 2
        elif not current_config.get("factored_init", False):
            recs["factored_init"] = True
        else:
            # Everything structural tried — try distillation
            if current_config.get("distill_alpha", 0) == 0:
                recs["distill_alpha"] = 0.2
                recs["distill_temp"] = 2.0

    return recs


def config_distance(a: dict, b: dict) -> float:
    """Compute normalized distance between two configs (0 = identical, ~13 = max)."""
    total = 0.0
    for name, spec in PARAM_SPACE.items():
        va = a.get(name, spec.get("default"))
        vb = b.get(name, spec.get("default"))
        ptype = spec["type"]

        if ptype in ("categorical", "boolean"):
            total += 0.0 if va == vb else 1.0
        elif ptype == "continuous":
            range_size = spec["high"] - spec["low"]
            if range_size > 0 and va is not None and vb is not None:
                total += abs(float(va) - float(vb)) / range_size
        elif ptype == "log-continuous":
            if va is not None and vb is not None and va > 0 and vb > 0:
                log_range = math.log(spec["high"]) - math.log(spec["low"])
                if log_range > 0:
                    total += abs(math.log(va) - math.log(vb)) / log_range
        elif ptype == "integer":
            range_size = spec["high"] - spec["low"]
            if range_size > 0 and va is not None and vb is not None:
                total += abs(va - vb) / range_size

    return total


def find_underexplored_params(tried_configs: list[dict]) -> list[str]:
    """Find parameters where most trials used the same value.

    These are params with low variance across trials — the system hasn't
    explored them much.
    """
    if len(tried_configs) < 5:
        return list(PARAM_SPACE.keys())

    underexplored = []
    for name, spec in PARAM_SPACE.items():
        values = [c.get(name, spec.get("default")) for c in tried_configs]
        ptype = spec["type"]

        if ptype in ("categorical", "boolean"):
            unique = len(set(str(v) for v in values))
            possible = len(spec.get("values", [True, False]))
            if unique < possible * 0.6:
                underexplored.append(name)
        elif ptype in ("continuous", "log-continuous"):
            nums = [float(v) for v in values if v is not None]
            if len(nums) >= 3:
                range_size = spec["high"] - spec["low"]
                spread = max(nums) - min(nums)
                if range_size > 0 and spread / range_size < 0.3:
                    underexplored.append(name)
        elif ptype == "integer":
            nums = [v for v in values if v is not None]
            if len(nums) >= 3:
                range_size = spec["high"] - spec["low"]
                spread = max(nums) - min(nums)
                if range_size > 0 and spread / range_size < 0.3:
                    underexplored.append(name)

    return underexplored
