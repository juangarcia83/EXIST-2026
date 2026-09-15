"""Loading the EXIST 2026 corpus into one table, for memes and videos alike.

Both modalities ship the same record shape — an id keyed dict with the media
filename, the language, the on-screen text and one list of raw annotations per
subtask — so a single loader serves both. The only modality-specific parts are
the media extension and the official hard-label thresholds, both of which
arrive through :class:`~exist2026.config.FewShotConfig`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from exist2026.io_utils import read_json
from exist2026.labels.hard import (
    HardLabelThresholds,
    hard_categorization,
    hard_identification,
    hard_intention,
)
from exist2026.taxonomy import (
    SEXISM_CATEGORIES,
    Modality,
    Subtask,
    annotation_field,
    subtask_key,
)

TRAIN = "train"
TEST = "test"


@dataclass(frozen=True)
class Corpus:
    """One loaded split pair, plus the raw records for annotation lookups.

    ``frame`` has one row per instance with the columns used across the code
    base: ``id_EXIST``, ``split``, ``lang``, ``text``, ``media_path``,
    ``media_ok`` and one ``hard_<subtask key>`` column per subtask.
    """

    frame: pd.DataFrame
    raw: Mapping[str, Mapping[str, Any]]
    modality: Modality
    thresholds: HardLabelThresholds

    @property
    def train(self) -> pd.DataFrame:
        return self.frame[self.frame["split"] == TRAIN].reset_index(drop=True)

    @property
    def test(self) -> pd.DataFrame:
        return self.frame[self.frame["split"] == TEST].reset_index(drop=True)

    def split(self, name: str) -> pd.DataFrame:
        return self.train if name == TRAIN else self.test

    def row(self, instance_id: str) -> pd.Series:
        """The single row of one instance. Raises ``KeyError`` if unknown."""
        matches = self.frame.loc[self.frame["id_EXIST"] == str(instance_id)]
        if matches.empty:
            raise KeyError(f"unknown instance id: {instance_id!r}")
        return matches.iloc[0]

    def annotations(self, instance_id: str, subtask: Subtask) -> list:
        """Raw annotator votes of one instance and subtask."""
        record = self.raw.get(str(instance_id), {})
        return record.get(annotation_field(self.modality, subtask)) or []

    def text_of(self, instance_id: str) -> str:
        return (self.row(instance_id)["text"] or "").strip()

    def media_of(self, instance_id: str) -> str:
        return self.row(instance_id)["media_path"]

    def hard_column(self, subtask: Subtask) -> str:
        return f"hard_{subtask_key(self.modality, subtask)}"

    def most_frequent_category(self, default: str = "STEREOTYPING-DOMINANCE") -> str:
        """Modal sexism category in TRAIN, used as the x.3 parser fallback."""
        counts: Counter[str] = Counter()
        column = self.hard_column(Subtask.CATEGORIZATION)
        for vector in self.train[column]:
            if vector is None:
                continue
            for category, active in zip(SEXISM_CATEGORIES, vector, strict=True):
                if active > 0.5:
                    counts[category] += 1
        return counts.most_common(1)[0][0] if counts else default


def _rows(
    records: Mapping[str, Mapping[str, Any]],
    media_dir: Path,
    split: str,
    modality: Modality,
    thresholds: HardLabelThresholds,
) -> list[dict[str, Any]]:
    keys = {s: subtask_key(modality, s) for s in Subtask}
    out: list[dict[str, Any]] = []
    for instance_id, record in records.items():
        filename = record.get(modality.media_field) or f"{instance_id}{modality.media_suffix}"
        media_path = media_dir / filename
        identification = record.get(annotation_field(modality, Subtask.IDENTIFICATION)) or []
        intention = record.get(annotation_field(modality, Subtask.INTENTION)) or []
        categorization = record.get(annotation_field(modality, Subtask.CATEGORIZATION)) or []
        out.append(
            {
                "id_EXIST": str(instance_id),
                "split": split,
                "lang": (record.get("lang") or "en"),
                "text": (record.get("text") or "").strip(),
                "media_path": str(media_path),
                "media_ok": media_path.exists(),
                f"hard_{keys[Subtask.IDENTIFICATION]}": (
                    hard_identification(identification, thresholds) if identification else None
                ),
                f"hard_{keys[Subtask.INTENTION]}": (
                    hard_intention(intention, thresholds) if intention else None
                ),
                f"hard_{keys[Subtask.CATEGORIZATION]}": hard_categorization(categorization, thresholds),
            }
        )
    return out


def load_corpus(
    modality: Modality,
    thresholds: HardLabelThresholds,
    *,
    train_json: Path,
    train_media_dir: Path,
    test_json: Path | None = None,
    test_media_dir: Path | None = None,
    require_media: bool = True,
) -> Corpus:
    """Load TRAIN (always) and TEST (when given) into a single :class:`Corpus`.

    TRAIN is always loaded even for a submission run: the few-shot pools are
    built from it, and building them anywhere else would leak test instances
    into the prompts.

    ``require_media`` asserts every referenced file is on disk. Keep it on for
    real runs — a missing file would otherwise surface as a per-instance
    exception hundreds of predictions into a run.
    """
    raw_train = read_json(train_json)
    rows = _rows(raw_train, train_media_dir, TRAIN, modality, thresholds)
    raw: dict[str, Mapping[str, Any]] = {str(k): v for k, v in raw_train.items()}

    if test_json is not None:
        if test_media_dir is None:
            raise ValueError("test_json was given without test_media_dir")
        raw_test = read_json(test_json)
        rows += _rows(raw_test, test_media_dir, TEST, modality, thresholds)
        raw.update({str(k): v for k, v in raw_test.items()})

    frame = pd.DataFrame(rows)
    corpus = Corpus(frame=frame, raw=raw, modality=modality, thresholds=thresholds)

    if require_media:
        for split in (TRAIN, TEST):
            part = corpus.split(split)
            if part.empty:
                continue
            missing = part.loc[~part["media_ok"], "id_EXIST"].tolist()
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {modality.value} in {split} have no media file "
                    f"(e.g. {missing[:3]}); check the media directory in your config."
                )
    return corpus


def describe(corpus: Corpus) -> pd.DataFrame:
    """Per-split counts of instances and of valid hard labels."""
    rows = []
    for split in (TRAIN, TEST):
        part = corpus.split(split)
        if part.empty:
            continue
        row: dict[str, Any] = {"split": split, "n": len(part), "media_ok": int(part["media_ok"].sum())}
        for subtask in (Subtask.IDENTIFICATION, Subtask.INTENTION):
            column = corpus.hard_column(subtask)
            row[f"{column}_valid"] = int(part[column].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def exclude(frame: pd.DataFrame, ids: Iterable[str]) -> pd.DataFrame:
    """Rows whose ``id_EXIST`` is not in ``ids`` (used to drop pool members)."""
    blocked = {str(i) for i in ids}
    return frame[~frame["id_EXIST"].isin(blocked)]
