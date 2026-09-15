"""Writing PyEvALL files and packaging an official EXIST 2026 submission.

The organizers accept a narrow format and reject silently-malformed files, so
everything written here goes through :func:`validate` first. The invariants are
cheap to check and expensive to get wrong after a multi-hour inference run:

* every record carries ``test_case == "EXIST2025"`` (guidelines p. 12);
* x.1 is YES/NO, x.2 is NO/DIRECT/JUDGEMENTAL, x.3 is a non-empty list;
* ``UNKNOWN`` is never predicted;
* in x.3 ``NO`` is exclusive — it cannot share a list with real categories;
* the run covers exactly the instances of the split, no more and no fewer.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from exist2026.io_utils import write_json
from exist2026.taxonomy import (
    SEXISM_CATEGORIES,
    Modality,
    Subtask,
    official_submission_stem,
    subtask_of,
)

#: Mandated by the guidelines; unchanged from EXIST 2025 despite the name.
PYEVALL_TEST_CASE = "EXIST2025"

HARD = "hard"
SOFT = "soft"


class SubmissionError(ValueError):
    """A prediction file violates the official format."""


def build_records(predictions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``{id: value}`` as a PyEvALL record list.

    Works for hard values (a string or list of strings) and soft ones (a
    ``{label: probability}`` dict) alike — PyEvALL distinguishes them by shape.
    """
    return [
        {"test_case": PYEVALL_TEST_CASE, "id": str(instance_id), "value": value}
        for instance_id, value in predictions.items()
    ]


def validate(records: Sequence[Mapping[str, Any]], key: str) -> None:
    """Assert the hard-format invariants of one subtask. Raises :class:`SubmissionError`."""
    subtask = subtask_of(key)
    valid_intention = {"NO", "DIRECT", "JUDGEMENTAL"}
    valid_categories = set(SEXISM_CATEGORIES) | {"NO"}

    for record in records:
        if "id" not in record:
            raise SubmissionError(f"[{key}] a record has no 'id'")
        if record.get("test_case") != PYEVALL_TEST_CASE:
            raise SubmissionError(
                f"[{key}] id={record.get('id')} has test_case={record.get('test_case')!r}, "
                f"expected {PYEVALL_TEST_CASE!r}"
            )
        value = record["value"]
        if isinstance(value, Mapping):
            continue  # soft distribution, checked by PyEvALL itself
        if subtask is Subtask.IDENTIFICATION:
            if value not in {"YES", "NO"}:
                raise SubmissionError(f"[{key}] id={record['id']}: invalid x.1 value {value!r}")
        elif subtask is Subtask.INTENTION:
            if value not in valid_intention:
                raise SubmissionError(f"[{key}] id={record['id']}: invalid x.2 value {value!r}")
        else:
            if not isinstance(value, list) or not value:
                raise SubmissionError(f"[{key}] id={record['id']}: x.3 must be a non-empty list")
            unknown = [v for v in value if v not in valid_categories]
            if unknown:
                raise SubmissionError(f"[{key}] id={record['id']}: invalid x.3 categories {unknown}")
            if "NO" in value and value != ["NO"]:
                raise SubmissionError(f"[{key}] id={record['id']}: x.3 mixes NO with categories: {value}")


def check_coverage(predictions: Mapping[str, Any], expected_ids: Iterable[str], key: str) -> None:
    """Assert the run predicted exactly ``expected_ids``."""
    expected = {str(i) for i in expected_ids}
    got = {str(i) for i in predictions}
    missing = expected - got
    extra = got - expected
    if missing:
        raise SubmissionError(
            f"[{key}] {len(missing)} instances have no prediction (e.g. {sorted(missing)[:3]})"
        )
    if extra:
        raise SubmissionError(
            f"[{key}] {len(extra)} predictions fall outside the split (e.g. {sorted(extra)[:3]})"
        )


def write_predictions(
    path: Path, predictions: Mapping[str, Any], key: str, *, validate_format: bool = True
) -> Path:
    """Validate and write one PyEvALL prediction file."""
    records = build_records(predictions)
    if validate_format:
        validate(records, key)
    return write_json(path, records)


def package(
    predictions_by_key: Mapping[str, Mapping[str, Any]],
    *,
    modality: Modality,
    output_dir: Path,
    team_name: str,
    run_id: int,
    evaluation_context: str = HARD,
    make_zip: bool = True,
) -> Path:
    """Write the ``exist2026_<team>/`` directory (and its ZIP) for upload.

    Layout required by the guidelines (pp. 14-15)::

        exist2026_<team_name>/
            task2_1_hard_<team_name>_<run_id>
            task2_2_hard_<team_name>_<run_id>
            task2_3_hard_<team_name>_<run_id>

    The per-subtask files deliberately carry no extension: that is the naming
    the organizers' form expects.
    """
    if evaluation_context not in (HARD, SOFT):
        raise SubmissionError(f"evaluation_context must be 'hard' or 'soft', got {evaluation_context!r}")
    if not 1 <= run_id <= 3:
        raise SubmissionError(f"run_id must be 1, 2 or 3 (up to three runs per subtask); got {run_id}")

    base = output_dir / f"exist2026_{team_name}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    for key, predictions in predictions_by_key.items():
        records = build_records(predictions)
        validate(records, key)
        stem = official_submission_stem(modality, subtask_of(key))
        write_json(base / f"{stem}_{evaluation_context}_{team_name}_{run_id}", records)

    if make_zip:
        archive = output_dir / f"exist2026_{team_name}"
        zip_path = archive.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(archive), "zip", root_dir=str(output_dir), base_dir=base.name)
    return base
