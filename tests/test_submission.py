"""Submission format: the invariants the organizers' loader assumes."""

from __future__ import annotations

import json
import zipfile

import pytest

from exist2026.submission import (
    PYEVALL_TEST_CASE,
    SubmissionError,
    build_records,
    check_coverage,
    package,
    validate,
    write_predictions,
)
from exist2026.taxonomy import Modality


class TestRecords:
    def test_every_record_carries_the_mandated_test_case(self):
        records = build_records({"1": "YES", "2": "NO"})
        assert {r["test_case"] for r in records} == {PYEVALL_TEST_CASE}

    def test_ids_are_stringified(self):
        assert build_records({7: "YES"})[0]["id"] == "7"


class TestValidation:
    def test_accepts_well_formed_predictions(self):
        validate(build_records({"1": "YES"}), "t21")
        validate(build_records({"1": "JUDGEMENTAL"}), "t22")
        validate(build_records({"1": ["OBJECTIFICATION", "SEXUAL-VIOLENCE"]}), "t23")

    def test_rejects_an_out_of_space_label(self):
        with pytest.raises(SubmissionError, match=r"invalid x\.1"):
            validate(build_records({"1": "MAYBE"}), "t21")

    def test_rejects_unknown_as_a_prediction(self):
        with pytest.raises(SubmissionError, match=r"invalid x\.3"):
            validate(build_records({"1": ["UNKNOWN"]}), "t23")

    def test_rejects_an_empty_category_list(self):
        with pytest.raises(SubmissionError, match="non-empty list"):
            validate(build_records({"1": []}), "t23")

    def test_rejects_no_mixed_with_real_categories(self):
        with pytest.raises(SubmissionError, match="mixes NO"):
            validate(build_records({"1": ["NO", "OBJECTIFICATION"]}), "t23")

    def test_rejects_a_wrong_test_case(self):
        records = [{"test_case": "EXIST2026", "id": "1", "value": "YES"}]
        with pytest.raises(SubmissionError, match="test_case"):
            validate(records, "t21")

    def test_lets_soft_distributions_through(self):
        validate([{"test_case": PYEVALL_TEST_CASE, "id": "1", "value": {"YES": 0.8, "NO": 0.2}}], "t21")


class TestCoverage:
    def test_missing_predictions_are_an_error(self):
        with pytest.raises(SubmissionError, match="no prediction"):
            check_coverage({"1": "YES"}, ["1", "2"], "t21")

    def test_predictions_outside_the_split_are_an_error(self):
        with pytest.raises(SubmissionError, match="outside the split"):
            check_coverage({"1": "YES", "9": "NO"}, ["1"], "t21")

    def test_exact_coverage_passes(self):
        check_coverage({"1": "YES", "2": "NO"}, ["2", "1"], "t21")


class TestWriting:
    def test_write_predictions_validates_before_writing(self, tmp_path):
        path = tmp_path / "pred.json"
        with pytest.raises(SubmissionError):
            write_predictions(path, {"1": "MAYBE"}, "t21")
        assert not path.exists()

    def test_written_file_round_trips(self, tmp_path):
        path = write_predictions(tmp_path / "pred.json", {"1": "YES"}, "t21")
        assert json.loads(path.read_text())[0]["value"] == "YES"


class TestPackaging:
    @pytest.fixture
    def predictions(self):
        return {
            "t21": {"1": "YES", "2": "NO"},
            "t22": {"1": "DIRECT", "2": "NO"},
            "t23": {"1": ["OBJECTIFICATION"], "2": ["NO"]},
        }

    def test_uses_the_official_naming(self, tmp_path, predictions):
        base = package(
            predictions,
            modality=Modality.MEMES,
            output_dir=tmp_path,
            team_name="ELiRF_UPV",
            run_id=1,
        )
        assert base.name == "exist2026_ELiRF_UPV"
        names = sorted(p.name for p in base.iterdir())
        assert names == [
            "task2_1_hard_ELiRF_UPV_1",
            "task2_2_hard_ELiRF_UPV_1",
            "task2_3_hard_ELiRF_UPV_1",
        ]

    def test_video_predictions_use_task3_names(self, tmp_path):
        base = package(
            {"t31": {"1": "YES"}, "t32": {"1": "DIRECT"}, "t33": {"1": ["OBJECTIFICATION"]}},
            modality=Modality.VIDEOS,
            output_dir=tmp_path,
            team_name="team",
            run_id=2,
        )
        assert (base / "task3_1_hard_team_2").exists()

    def test_writes_a_zip_next_to_the_directory(self, tmp_path, predictions):
        package(
            predictions,
            modality=Modality.MEMES,
            output_dir=tmp_path,
            team_name="team",
            run_id=1,
        )
        archive = tmp_path / "exist2026_team.zip"
        assert archive.exists()
        with zipfile.ZipFile(archive) as bundle:
            assert any(name.endswith("task2_1_hard_team_1") for name in bundle.namelist())

    def test_rejects_an_out_of_range_run_id(self, tmp_path, predictions):
        with pytest.raises(SubmissionError, match="run_id"):
            package(
                predictions,
                modality=Modality.MEMES,
                output_dir=tmp_path,
                team_name="team",
                run_id=4,
            )

    def test_rejects_an_unknown_evaluation_context(self, tmp_path, predictions):
        with pytest.raises(SubmissionError, match="evaluation_context"):
            package(
                predictions,
                modality=Modality.MEMES,
                output_dir=tmp_path,
                team_name="team",
                run_id=1,
                evaluation_context="fuzzy",
            )

    def test_repackaging_replaces_the_previous_contents(self, tmp_path, predictions):
        base = package(predictions, modality=Modality.MEMES, output_dir=tmp_path, team_name="team", run_id=1)
        (base / "leftover.txt").write_text("stale")
        base = package(predictions, modality=Modality.MEMES, output_dir=tmp_path, team_name="team", run_id=1)
        assert not (base / "leftover.txt").exists()
