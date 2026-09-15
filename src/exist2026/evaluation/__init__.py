"""Official evaluation (PyEvALL) plus the diagnostic reports around it."""

from exist2026.evaluation.pyevall import (
    HARD_METRICS,
    SOFT_METRICS,
    evaluate,
    hard_gold,
    hierarchy_for,
    soft_gold,
)

__all__ = [
    "HARD_METRICS",
    "SOFT_METRICS",
    "evaluate",
    "hard_gold",
    "hierarchy_for",
    "soft_gold",
]
