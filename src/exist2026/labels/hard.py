"""Hard labels: majority aggregation with the official vote thresholds.

The thresholds come from the EXIST 2026 Lab Guidelines V0.5 (p. 17) and differ
between modalities, so they are data rather than constants:

* **Memes** (6 annotators): ``x.1 > 3``, ``x.2 > 2``, ``x.3 > 1`` votes.
* **Videos** (3 annotators): ``> 1`` vote for all three subtasks, because of
  "the complexity of video labeling".

An instance whose votes do not reach the threshold has *no* hard label; those
functions return ``None`` and callers must drop the instance from hard
evaluation rather than inventing a label for it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from exist2026.taxonomy import (
    INTENTION_NO_RAW,
    INTENTION_STR_TO_INT,
    SEXISM_CATEGORIES,
    UNKNOWN,
    Modality,
)


@dataclass(frozen=True)
class HardLabelThresholds:
    """Minimum number of votes (strictly greater than) per subtask."""

    identification: int
    intention: int
    categorization: int

    #: Videos additionally require the winning option not to be beaten by the
    #: other one, and drop whole annotations containing ``UNKNOWN`` in x.3.
    require_plurality: bool = False
    drop_unknown_annotations: bool = False

    @classmethod
    def official(cls, modality: Modality) -> HardLabelThresholds:
        if modality is Modality.MEMES:
            return cls(identification=3, intention=2, categorization=1)
        return cls(
            identification=1,
            intention=1,
            categorization=1,
            require_plurality=True,
            drop_unknown_annotations=True,
        )


def _valid(labels: Iterable[str]) -> list[str]:
    return [label for label in labels if label != UNKNOWN]


def hard_identification(labels: Sequence[str], thresholds: HardLabelThresholds) -> int | None:
    """``1`` (YES), ``0`` (NO) or ``None`` when the votes are inconclusive."""
    votes = _valid(labels) if thresholds.require_plurality else list(labels)
    if thresholds.require_plurality and not votes:
        return None
    yes = sum(1 for v in votes if v == "YES")
    no = sum(1 for v in votes if v == "NO")
    if thresholds.require_plurality:
        if yes > thresholds.identification and yes >= no:
            return 1
        if no > thresholds.identification and no >= yes:
            return 0
        return None
    if yes > thresholds.identification:
        return 1
    if no > thresholds.identification:
        return 0
    return None


def hard_intention(labels: Sequence[str], thresholds: HardLabelThresholds) -> int | None:
    """``0`` (NO), ``1`` (DIRECT), ``2`` (JUDGEMENTAL) or ``None``."""
    clean = _valid(labels)
    if not clean:
        return None
    mapped = ["NO" if label == INTENTION_NO_RAW else label for label in clean]
    top, votes = Counter(mapped).most_common(1)[0]
    if votes <= thresholds.intention:
        return None
    return INTENTION_STR_TO_INT[top]


def hard_categorization(annotations: Sequence[Sequence[str]], thresholds: HardLabelThresholds) -> np.ndarray:
    """Multi-hot vector over :data:`SEXISM_CATEGORIES`.

    A category is active when strictly more than ``thresholds.categorization``
    annotators selected it. Unlike the other two subtasks this never returns
    ``None``: an all-zero vector means "no category reached consensus", which
    the hierarchy reads as *not sexist*.
    """
    counts: Counter[str] = Counter()
    for annotation in annotations:
        if thresholds.drop_unknown_annotations and UNKNOWN in annotation:
            continue
        for label in annotation:
            if label != UNKNOWN:
                counts[label] += 1
    vector = np.zeros(len(SEXISM_CATEGORIES), dtype=np.float32)
    for i, category in enumerate(SEXISM_CATEGORIES):
        if counts.get(category, 0) > thresholds.categorization:
            vector[i] = 1.0
    return vector


def active_categories(multi_hot: np.ndarray) -> list[str]:
    """Category names of an active multi-hot vector, in canonical order."""
    return [c for c, on in zip(SEXISM_CATEGORIES, multi_hot, strict=True) if on > 0.5]


# --------------------------------------------------------------------------- #
# Annotator agreement, used to rank few-shot candidates.
# --------------------------------------------------------------------------- #


def identification_consensus(labels: Sequence[str]) -> float:
    """Share of valid annotators that agree with the majority option."""
    valid = _valid(labels)
    if not valid:
        return 0.0
    return Counter(valid).most_common(1)[0][1] / len(valid)


def intention_consensus(labels: Sequence[str]) -> float:
    valid = _valid(labels)
    if not valid:
        return 0.0
    mapped = ["NO" if label == INTENTION_NO_RAW else label for label in valid]
    return Counter(mapped).most_common(1)[0][1] / len(mapped)


def categorization_consensus(annotations: Sequence[Sequence[str]], category: str) -> tuple[float, float]:
    """``(single_label_rate, agreement)`` of one category.

    ``single_label_rate`` is the share of annotations that picked *only* this
    category — a proxy for how prototypical the instance is — and ``agreement``
    the share that picked it at all. Both are computed over annotations without
    ``UNKNOWN``; an instance with none of those scores ``(0.0, 0.0)``.
    """
    valid = [a for a in annotations if UNKNOWN not in a]
    if not valid:
        return 0.0, 0.0
    n_single = sum(1 for a in valid if category in a and len(a) == 1)
    n_with = sum(1 for a in valid if category in a)
    return n_single / len(valid), n_with / len(valid)
