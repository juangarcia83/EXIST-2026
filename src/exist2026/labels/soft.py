"""Soft labels for the *Learning with Disagreements* (LeWiDi) paradigm.

Systems are scored on how well they reproduce the annotator-disagreement
distribution, so the training target is the *proportion of annotators* per
class rather than a single aggregated label. Class order is fixed here and
shared by the losses, the PyEvALL writers and the model heads:

* ``x.1`` -> ``[P(YES), P(NO)]``
* ``x.2`` -> ``[P(NO), P(DIRECT), P(JUDGEMENTAL)]`` (``"-"`` counts as NO)
* ``x.3`` -> ``[P(NO), P(cat) for cat in SEXISM_CATEGORIES]`` (multi-label, so
  the vector does *not* sum to one)

``UNKNOWN`` is always ignored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from exist2026.taxonomy import (
    INTENTION_NO_RAW,
    SEXISM_CATEGORIES,
    UNKNOWN,
    Modality,
    Subtask,
    annotation_field,
    subtask_key,
)


def soft_identification(labels: Sequence[str]) -> np.ndarray | None:
    """``[P(YES), P(NO)]``, or ``None`` when every annotation is ``UNKNOWN``."""
    valid = [x for x in labels if x != UNKNOWN]
    if not valid:
        return None
    p_yes = sum(1 for x in valid if x == "YES") / len(valid)
    return np.array([p_yes, 1.0 - p_yes], dtype=np.float32)


def soft_intention(labels: Sequence[str]) -> np.ndarray:
    """``[P(NO), P(DIRECT), P(JUDGEMENTAL)]``.

    Unlike the other two subtasks this always returns a vector: an instance
    with no valid x.2 annotation is *not sexist*, which is exactly ``P(NO)=1``.
    """
    valid = [x for x in labels if x != UNKNOWN]
    if not valid:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n = len(valid)
    return np.array(
        [
            sum(1 for x in valid if x == INTENTION_NO_RAW) / n,
            sum(1 for x in valid if x == "DIRECT") / n,
            sum(1 for x in valid if x == "JUDGEMENTAL") / n,
        ],
        dtype=np.float32,
    )


def soft_categorization(annotations: Sequence[Sequence[str]]) -> np.ndarray | None:
    """``[P(NO), P(cat)...]``, or ``None`` when no annotation is usable."""
    valid = [a for a in annotations if UNKNOWN not in a]
    if not valid:
        return None
    n = len(valid)
    out = np.zeros(1 + len(SEXISM_CATEGORIES), dtype=np.float32)
    out[0] = sum(1 for a in valid if list(a) == [INTENTION_NO_RAW]) / n
    for i, category in enumerate(SEXISM_CATEGORIES):
        out[i + 1] = sum(1 for a in valid if category in a) / n
    return out


def _raw_annotations(instance: Mapping, modality: Modality, subtask: Subtask) -> list:
    """Read one annotation list, tolerating the stray space some dumps carry.

    A few released records spell the key ``"labels_ task2_2"``; accepting both
    spellings here keeps that quirk out of every call site.
    """
    canonical = annotation_field(modality, subtask)
    for key in (canonical, canonical.replace("labels_task", "labels_ task")):
        if key in instance:
            return instance[key] or []
    return []


def soft_labels(instance: Mapping, modality: Modality) -> dict[str, np.ndarray] | None:
    """Soft targets of the three subtasks, keyed by subtask key.

    Returns ``None`` when x.1 or x.3 has no usable annotation — such instances
    cannot supervise the hierarchy and are dropped from training.
    """
    identification = soft_identification(_raw_annotations(instance, modality, Subtask.IDENTIFICATION))
    categorization = soft_categorization(_raw_annotations(instance, modality, Subtask.CATEGORIZATION))
    if identification is None or categorization is None:
        return None
    intention = soft_intention(_raw_annotations(instance, modality, Subtask.INTENTION))
    return {
        subtask_key(modality, Subtask.IDENTIFICATION): identification,
        subtask_key(modality, Subtask.INTENTION): intention,
        subtask_key(modality, Subtask.CATEGORIZATION): categorization,
    }
