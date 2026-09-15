"""Train / validation split, designed around each modality's leakage risk.

**Videos** are split *by creator*: the TikTok handle is recovered from the
instance URL and no creator appears on both sides. A per-instance split would
let a model learn a creator's style in training and be rewarded for recognising
it in validation, which flatters the score without improving the system.
Creator groups are then stratified by their mean YES rate so the split stays
balanced.

**Memes** have no such grouping, so they are split per instance, stratified by
``(language, majority x.1 vote)`` to keep both languages and both classes
represented.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exist2026.taxonomy import Modality

_TIKTOK_HANDLE = re.compile(r"@([\w_.]+)")


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    frame: pd.DataFrame

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.val)
        if overlap:
            raise ValueError(f"{len(overlap)} instances appear in both sides of the split")

    def summary(self) -> pd.DataFrame:
        return (
            self.frame.groupby("split")
            .agg(n=("id", "size"), yes_rate=("hard_yes", "mean"))
            .round(3)
            .reset_index()
        )


def creator_of(record: Mapping[str, Any], instance_id: str) -> str:
    """TikTok handle from the instance URL, or a unique per-instance group."""
    match = _TIKTOK_HANDLE.search(record.get("url", "") or "")
    return match.group(1) if match else f"__solo_{instance_id}"


def build_split(
    records: Mapping[str, Mapping[str, Any]],
    soft_labels: Mapping[str, Mapping[str, np.ndarray]],
    identification_key: str,
    modality: Modality,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    """Split the instances that have usable soft labels."""
    from sklearn.model_selection import train_test_split

    rows = []
    for instance_id in soft_labels:
        record = records[str(instance_id)]
        p_yes = float(soft_labels[instance_id][identification_key][0])
        rows.append(
            {
                "id": instance_id,
                "lang": record.get("lang", "en"),
                "group": creator_of(record, instance_id) if modality is Modality.VIDEOS else instance_id,
                "p_yes": p_yes,
                "hard_yes": int(p_yes >= 0.5),
            }
        )
    frame = pd.DataFrame(rows)

    if modality is Modality.VIDEOS:
        groups = frame.groupby("group").agg(p_yes=("p_yes", "mean")).reset_index()
        # Rank-then-quartile keeps stratification well defined even when many
        # creators share the same mean YES rate.
        groups["stratum"] = pd.qcut(groups.p_yes.rank(method="first"), q=4, labels=False)
        try:
            train_groups, _ = train_test_split(
                groups.group, test_size=val_fraction, random_state=seed, stratify=groups.stratum
            )
        except ValueError:
            # Too few creators for four strata (small subsets, smoke tests).
            # Grouping still matters more than stratification, so keep it.
            train_groups, _ = train_test_split(groups.group, test_size=val_fraction, random_state=seed)
        frame["split"] = np.where(frame.group.isin(set(train_groups)), "train", "val")
    else:
        strata = frame.lang.astype(str) + "_" + frame.hard_yes.astype(str)
        _, val_index = train_test_split(
            frame.index, test_size=val_fraction, random_state=seed, stratify=strata
        )
        frame["split"] = "train"
        frame.loc[val_index, "split"] = "val"

    return Split(
        train=frame.loc[frame.split == "train", "id"].tolist(),
        val=frame.loc[frame.split == "val", "id"].tolist(),
        frame=frame,
    )


def yes_subset(
    instance_ids: Sequence[str],
    soft_labels: Mapping[str, Mapping[str, np.ndarray]],
    identification_key: str,
    threshold: float = 0.5,
) -> list[str]:
    """Instances the annotators mostly called sexist.

    Subtasks x.2 and x.3 are trained on these only — they are undefined for
    non-sexist instances — but are always *evaluated* on the full split with the
    x.1 gate applied, so the score reflects the deployed cascade.
    """
    return [i for i in instance_ids if soft_labels[i][identification_key][0] >= threshold]
