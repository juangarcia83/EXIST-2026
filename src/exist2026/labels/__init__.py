"""Aggregation of raw annotator votes into hard and soft labels."""

from exist2026.labels.hard import (
    HardLabelThresholds,
    categorization_consensus,
    hard_categorization,
    hard_identification,
    hard_intention,
    identification_consensus,
    intention_consensus,
)
from exist2026.labels.soft import (
    soft_categorization,
    soft_identification,
    soft_intention,
    soft_labels,
)

__all__ = [
    "HardLabelThresholds",
    "categorization_consensus",
    "hard_categorization",
    "hard_identification",
    "hard_intention",
    "identification_consensus",
    "intention_consensus",
    "soft_categorization",
    "soft_identification",
    "soft_intention",
    "soft_labels",
]
