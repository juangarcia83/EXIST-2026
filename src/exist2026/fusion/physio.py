"""The physiological branch: eye tracking, heart rate and EEG.

The lab releases recordings for a pool of subjects who viewed the stimuli, but
each stimulus was seen by only 2-4 of them, so the branch must tolerate missing
subjects and missing features by construction rather than by imputation alone.

Two representations are supported:

``matrix`` (videos)
    One row per subject, ``(max_subjects, n_features)``, plus a validity mask.
    The model pools subjects with attention, so the representation stays
    permutation-invariant and a missing subject is masked out instead of being
    silently averaged in.
``flat`` (memes)
    Per-feature ``(mean, std, min, max)`` across whatever subjects exist, giving
    a fixed-width vector. Cheaper, and enough when the per-subject structure is
    not modelled.

All preprocessing statistics are fitted on the *training* split only: KNN
imputation and per-modality z-scoring both see validation data otherwise, and
the leak would be invisible in the metrics.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MODALITIES = ("ET", "HR", "EEG")
AGGREGATIONS = ("mean", "std", "min", "max")
_AGGREGATORS = {"mean": np.nanmean, "std": np.nanstd, "min": np.nanmin, "max": np.nanmax}


@dataclass(frozen=True)
class FeatureSchema:
    """Canonical feature order, discovered from the corpus.

    The released files do not guarantee a feature order, so it is derived once
    by scanning the data and sorting; every vector built afterwards uses it.
    Deriving it (instead of hard-coding ~100 names) keeps the code working when
    the organizers add a feature.
    """

    eye_tracking: tuple[str, ...]
    heart_rate: tuple[str, ...]
    eeg: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return self.eye_tracking + self.heart_rate + self.eeg

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def modality_of(self) -> np.ndarray:
        return np.array(
            ["ET"] * len(self.eye_tracking) + ["HR"] * len(self.heart_rate) + ["EEG"] * len(self.eeg)
        )

    def columns_of(self, modality: str) -> np.ndarray:
        return np.where(self.modality_of == modality)[0]

    def features_of(self, modality: str) -> tuple[str, ...]:
        return {"ET": self.eye_tracking, "HR": self.heart_rate, "EEG": self.eeg}[modality]

    @classmethod
    def discover(cls, records: Iterable[Mapping[str, Any]]) -> FeatureSchema:
        found: dict[str, set[str]] = {modality: set() for modality in MODALITIES}
        for record in records:
            modalities = (record.get("sensorial", {}) or {}).get("modalities", {}) or {}
            for modality, payload in modalities.items():
                if modality not in found:
                    continue
                for subject in (payload.get("by_user", {}) or {}).values():
                    if isinstance(subject, Mapping):
                        found[modality].update(subject.keys())
        return cls(
            eye_tracking=tuple(sorted(found["ET"])),
            heart_rate=tuple(sorted(found["HR"])),
            eeg=tuple(sorted(found["EEG"])),
        )


def _as_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def subject_matrix(
    record: Mapping[str, Any], schema: FeatureSchema, max_subjects: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """``(max_subjects, n_features)`` plus a ``(max_subjects,)`` validity mask.

    Rows of absent subjects are zeros *and* masked; features missing for a
    present subject stay ``NaN`` so the imputer can fill them.
    """
    sensorial = record.get("sensorial", {}) or {}
    subjects = (sensorial.get("users", []) or [])[:max_subjects]
    modalities = sensorial.get("modalities", {}) or {}
    modality_of = schema.modality_of

    matrix = np.full((max_subjects, schema.dim), np.nan, dtype=np.float32)
    mask = np.zeros(max_subjects, dtype=np.float32)
    for row, subject in enumerate(subjects):
        for column, feature in enumerate(schema.names):
            payload = (modalities.get(modality_of[column], {}) or {}).get("by_user", {}) or {}
            matrix[row, column] = _as_float((payload.get(subject) or {}).get(feature))
        mask[row] = 1.0
    matrix = np.where(np.isnan(matrix) & (mask[:, None] == 0), 0.0, matrix)
    return matrix, mask


def flat_vector(record: Mapping[str, Any], schema: FeatureSchema) -> np.ndarray:
    """Per-feature ``(mean, std, min, max)`` across the subjects that exist."""
    modalities = (record.get("sensorial", {}) or {}).get("modalities", {}) or {}
    parts = []
    for modality in ("EEG", "HR", "ET"):
        features = schema.features_of(modality)
        by_subject = (modalities.get(modality, {}) or {}).get("by_user", {}) or {}
        width = len(features) * len(AGGREGATIONS)
        if not by_subject or not features:
            parts.append(np.zeros(width, dtype=np.float32))
            continue
        subjects = list(by_subject)
        values = np.full((len(subjects), len(features)), np.nan, dtype=np.float32)
        for row, subject in enumerate(subjects):
            payload = by_subject[subject]
            if not isinstance(payload, Mapping):
                continue
            for column, feature in enumerate(features):
                values[row, column] = _as_float(payload.get(feature))
        with warnings.catch_warnings():
            # All-NaN columns are expected: that subject lacks the sensor.
            warnings.simplefilter("ignore", RuntimeWarning)
            for aggregation in AGGREGATIONS:
                aggregated = _AGGREGATORS[aggregation](values, axis=0)
                parts.append(np.where(np.isfinite(aggregated), aggregated, 0.0).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class PhysioPreprocessor:
    """Impute and standardize physiology with statistics fitted on TRAIN only.

    ``matrix``: KNN imputation over valid subject rows, then a z-score per
    sensor family (ET / HR / EEG), since their scales have nothing in common.
    ``flat``: a global z-score — extraction already replaced absences with 0.
    """

    def __init__(self, kind: str, schema: FeatureSchema) -> None:
        if kind not in ("flat", "matrix"):
            raise ValueError(f"kind must be 'flat' or 'matrix', got {kind!r}")
        self.kind = kind
        self.schema = schema
        self._imputer = None
        self._scalers: dict[str, Any] = {}
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.fitted = False

    def fit(
        self, values: Sequence[np.ndarray], masks: Sequence[np.ndarray] | None = None
    ) -> PhysioPreprocessor:
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import StandardScaler

        stacked = np.stack(list(values))
        if self.kind == "flat":
            self._mean = stacked.mean(axis=0).astype(np.float32)
            self._std = (stacked.std(axis=0) + 1e-8).astype(np.float32)
            self.fitted = True
            return self

        if masks is None:
            raise ValueError("matrix physiology needs subject masks to fit")
        valid = np.stack(list(masks)).reshape(-1) > 0
        flat = stacked.reshape(-1, self.schema.dim)
        self._imputer = KNNImputer(n_neighbors=5, weights="distance").fit(flat[valid])
        imputed = self._imputer.transform(flat)
        for modality in MODALITIES:
            columns = self.schema.columns_of(modality)
            if columns.size:
                self._scalers[modality] = StandardScaler().fit(imputed[valid][:, columns])
        self.fitted = True
        return self

    def transform(self, value: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PhysioPreprocessor.transform called before fit")
        if self.kind == "flat":
            return ((value - self._mean) / self._std).astype(np.float32)
        flat = self._imputer.transform(value.reshape(-1, self.schema.dim))
        for modality, scaler in self._scalers.items():
            columns = self.schema.columns_of(modality)
            flat[:, columns] = scaler.transform(flat[:, columns])
        return flat.reshape(value.shape).astype(np.float32)


def extract(
    records: Mapping[str, Mapping[str, Any]],
    instance_ids: Sequence[str],
    schema: FeatureSchema,
    kind: str,
    max_subjects: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Raw physiology per instance; masks are empty for the ``flat`` kind."""
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for instance_id in instance_ids:
        record = records[str(instance_id)]
        if kind == "matrix":
            matrix, mask = subject_matrix(record, schema, max_subjects)
            values[instance_id], masks[instance_id] = matrix, mask
        else:
            values[instance_id] = flat_vector(record, schema)
    return values, masks


def input_dim(kind: str, schema: FeatureSchema, example: np.ndarray | None = None) -> int:
    """Width the physiological encoder must accept."""
    if kind == "matrix":
        return schema.dim
    if example is None:
        return schema.dim * len(AGGREGATIONS)
    return int(example.shape[0])
