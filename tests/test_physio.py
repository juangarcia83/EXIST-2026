"""The physiological branch: sparse subjects, and no leakage from validation."""

from __future__ import annotations

import numpy as np
import pytest

from exist2026.fusion.physio import (
    FeatureSchema,
    PhysioPreprocessor,
    extract,
    flat_vector,
    input_dim,
    subject_matrix,
)


def record(subjects, values, *, eeg=None):
    """A record with two ET features, one HR feature and optional EEG."""
    modalities = {
        "ET": {"by_user": {s: {"fix_count": v[0], "saccades": v[1]} for s, v in values.items()}},
        "HR": {"by_user": {s: {"bpm": v[2]} for s, v in values.items()}},
    }
    if eeg:
        modalities["EEG"] = {"by_user": eeg}
    return {"sensorial": {"users": subjects, "modalities": modalities}}


@pytest.fixture
def schema():
    sample = record(["a"], {"a": (1.0, 2.0, 60.0)}, eeg={"a": {"alpha": 0.5}})
    return FeatureSchema.discover([sample])


class TestSchema:
    def test_orders_features_by_sensor_family_then_name(self, schema):
        assert schema.eye_tracking == ("fix_count", "saccades")
        assert schema.heart_rate == ("bpm",)
        assert schema.eeg == ("alpha",)
        assert schema.names == ("fix_count", "saccades", "bpm", "alpha")

    def test_maps_columns_back_to_their_sensor(self, schema):
        assert schema.columns_of("ET").tolist() == [0, 1]
        assert schema.columns_of("HR").tolist() == [2]
        assert schema.dim == 4


class TestSubjectMatrix:
    def test_absent_subjects_are_zeroed_and_masked(self, schema):
        instance = record(["a"], {"a": (1.0, 2.0, 60.0)}, eeg={"a": {"alpha": 0.5}})
        matrix, mask = subject_matrix(instance, schema, max_subjects=4)
        assert matrix.shape == (4, 4)
        assert mask.tolist() == [1.0, 0.0, 0.0, 0.0]
        assert (matrix[1:] == 0).all()

    def test_a_missing_feature_of_a_present_subject_stays_nan(self, schema):
        instance = {"sensorial": {"users": ["a"], "modalities": {"ET": {"by_user": {"a": {}}}}}}
        matrix, mask = subject_matrix(instance, schema, max_subjects=2)
        assert mask[0] == 1.0
        assert np.isnan(matrix[0]).all()  # left for the imputer, not silently zeroed

    def test_extra_subjects_are_truncated(self, schema):
        values = dict.fromkeys("abcdef", (1.0, 2.0, 60.0))
        matrix, mask = subject_matrix(record(list("abcdef"), values), schema, max_subjects=4)
        assert matrix.shape[0] == 4
        assert mask.sum() == 4


class TestFlatVector:
    def test_width_is_features_times_aggregations(self, schema):
        instance = record(
            ["a", "b"], {"a": (1.0, 2.0, 60.0), "b": (3.0, 4.0, 80.0)}, eeg={"a": {"alpha": 0.5}}
        )
        vector = flat_vector(instance, schema)
        assert vector.shape[0] == schema.dim * 4

    def test_aggregates_across_subjects(self, schema):
        instance = record(["a", "b"], {"a": (1.0, 2.0, 60.0), "b": (3.0, 4.0, 80.0)})
        vector = flat_vector(instance, schema)
        # EEG comes first (no subjects here, so zeros), then HR, then ET.
        assert vector.sum() != 0
        assert np.isfinite(vector).all()

    def test_an_instance_without_physiology_is_all_zeros(self, schema):
        assert flat_vector({}, schema).tolist() == [0.0] * (schema.dim * 4)


class TestPreprocessor:
    def test_flat_standardization_uses_training_statistics_only(self, schema):
        train = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        preprocessor = PhysioPreprocessor("flat", schema).fit(train)
        # A validation point 10 sigma away must not shift the statistics.
        transformed = preprocessor.transform(np.array([100.0, 100.0], dtype=np.float32))
        assert transformed[0] > 10
        # Training points still standardize to +-1.
        assert preprocessor.transform(train[0]).tolist() == pytest.approx([-1.0, -1.0], abs=1e-4)

    def test_transform_before_fit_is_an_error(self, schema):
        with pytest.raises(RuntimeError, match="before fit"):
            PhysioPreprocessor("flat", schema).transform(np.zeros(4, dtype=np.float32))

    def test_matrix_kind_needs_masks(self, schema):
        values = [np.zeros((4, schema.dim), dtype=np.float32)]
        with pytest.raises(ValueError, match="subject masks"):
            PhysioPreprocessor("matrix", schema).fit(values)

    def test_matrix_imputation_fills_missing_features(self, schema):
        rng = np.random.default_rng(0)
        values, masks = [], []
        for _ in range(8):
            matrix = rng.normal(size=(2, schema.dim)).astype(np.float32)
            values.append(matrix)
            masks.append(np.ones(2, dtype=np.float32))
        preprocessor = PhysioPreprocessor("matrix", schema).fit(values, masks)
        gappy = values[0].copy()
        gappy[0, 0] = np.nan
        result = preprocessor.transform(gappy)
        assert np.isfinite(result).all()
        assert result.shape == gappy.shape

    def test_an_unknown_kind_is_rejected(self, schema):
        with pytest.raises(ValueError, match="flat"):
            PhysioPreprocessor("tensor", schema)


class TestExtract:
    def test_matrix_extraction_returns_masks(self, schema):
        records = {"1": record(["a"], {"a": (1.0, 2.0, 60.0)})}
        values, masks = extract(records, ["1"], schema, "matrix", max_subjects=3)
        assert values["1"].shape == (3, schema.dim)
        assert masks["1"].shape == (3,)

    def test_flat_extraction_returns_no_masks(self, schema):
        records = {"1": record(["a"], {"a": (1.0, 2.0, 60.0)})}
        values, masks = extract(records, ["1"], schema, "flat")
        assert masks == {}
        assert values["1"].ndim == 1


def test_input_dim_matches_the_representation(schema):
    assert input_dim("matrix", schema) == schema.dim
    assert input_dim("flat", schema) == schema.dim * 4
