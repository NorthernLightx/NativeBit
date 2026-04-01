"""Trial dataclass — stores config, results, and metadata for one experiment."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trial:
    """One autoresearch experiment."""

    # Identity
    trial_id: int = 0
    strategy: str = ""
    seed: int = 42

    # Config (the hyperparameters being searched)
    config: dict = field(default_factory=dict)

    # Results (filled in as phases complete)
    screen_ppl: Optional[float] = None
    validate_ppl: Optional[float] = None
    confirm_ppl: Optional[float] = None       # mean across seeds
    confirm_std: Optional[float] = None        # std across seeds
    confirm_ppls: list = field(default_factory=list)  # per-seed PPLs

    # Status
    status: str = "pending"  # pending, screening, validating, confirming, accepted, rejected
    reject_phase: Optional[str] = None  # which phase rejected it

    # Timing
    screen_time: float = 0.0
    validate_time: float = 0.0
    confirm_time: float = 0.0

    # Diagnosis (filled after screen phase log analysis)
    diagnosis_issues: list = field(default_factory=list)
    diagnosis_recs: dict = field(default_factory=dict)

    # Extra
    aborted: bool = False
    abort_reason: str = ""

    def best_ppl(self) -> float:
        """Return the best (most validated) PPL available."""
        if self.confirm_ppl is not None:
            return self.confirm_ppl
        if self.validate_ppl is not None:
            return self.validate_ppl
        if self.screen_ppl is not None:
            return self.screen_ppl
        return float("inf")

    def total_time(self) -> float:
        return self.screen_time + self.validate_time + self.confirm_time

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "config": self.config,
            "screen_ppl": self.screen_ppl,
            "validate_ppl": self.validate_ppl,
            "confirm_ppl": self.confirm_ppl,
            "confirm_std": self.confirm_std,
            "confirm_ppls": self.confirm_ppls,
            "status": self.status,
            "reject_phase": self.reject_phase,
            "screen_time": round(self.screen_time, 1),
            "validate_time": round(self.validate_time, 1),
            "confirm_time": round(self.confirm_time, 1),
            "diagnosis_issues": self.diagnosis_issues,
            "diagnosis_recs": self.diagnosis_recs,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trial":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
