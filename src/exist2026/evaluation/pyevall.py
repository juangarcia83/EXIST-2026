"""Official scoring through PyEvALL.

Two settings are used, and they are not interchangeable:

* **hard-hard** (``ICM``, ``ICMNorm``, ``FMeasure``) scores the few-shot
  cascade, which emits single labels.
* **soft-soft** (``ICMSoft``, ``ICMSoftNorm``) is the lab's primary setting and
  scores the trained system's predicted distributions.

The subtask hierarchy is declared to PyEvALL through ``PARAM_HIERARCHY`` and
applied *at evaluation time* only. Training keeps one independent head per
subtask; conflating the two is what makes a hierarchy leak into the model.

PyEvALL's report object differs across releases, so :func:`parse_report` tries
the documented attributes, then a DataFrame, then the printed text. That is
deliberate defensive parsing: a metric silently read as ``None`` would be far
worse than a noisy fallback.
"""

from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from exist2026.io_utils import write_json
from exist2026.taxonomy import SEXISM_CATEGORIES, Subtask, subtask_of

HARD_METRICS = ("ICM", "ICMNorm", "FMeasure")
SOFT_METRICS = ("ICMSoft", "ICMSoftNorm")

_METRIC_ALIASES = {
    "ICMNorm": ("ICMNorm", "ICM-Norm", "ICM Norm"),
    "FMeasure": ("FMeasure", "F-Measure", "F1"),
}

_REPORT_ATTRIBUTES = (
    "report",
    "dict_report",
    "json_report",
    "metrics_data",
    "_report",
    "data",
    "result",
    "results",
)


class PyEvALLNotInstalled(ImportError):
    """PyEvALL is an optional dependency; scoring needs it, nothing else does."""


def _pyevall():
    try:
        from pyevall.evaluation import PyEvALLEvaluation
        from pyevall.utils.utils import PyEvALLUtils
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise PyEvALLNotInstalled(
            "PyEvALL is required for official scoring. Install it with "
            "`pip install -e '.[eval]'` or `pip install PyEvALL`."
        ) from error
    return PyEvALLEvaluation, PyEvALLUtils


def hierarchy_for(key: str) -> dict[str, list[str]] | None:
    """PyEvALL hierarchy of one subtask; ``None`` for the flat binary one."""
    subtask = subtask_of(key)
    if subtask is Subtask.IDENTIFICATION:
        return None
    if subtask is Subtask.INTENTION:
        return {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}
    return {"YES": list(SEXISM_CATEGORIES), "NO": []}


# --------------------------------------------------------------------------- #
# Gold construction
# --------------------------------------------------------------------------- #


def hard_gold(corpus, instance_ids: Sequence[str], key: str) -> list[dict[str, Any]]:
    """Hard gold records, skipping instances whose votes were inconclusive.

    Instances without a valid hard label are dropped rather than defaulted:
    the official thresholds declare them undecided, and scoring a prediction
    against an invented gold would report agreement that does not exist.
    """
    from exist2026.labels.hard import (
        active_categories,
        hard_identification,
        hard_intention,
    )
    from exist2026.submission import PYEVALL_TEST_CASE
    from exist2026.taxonomy import IDENTIFICATION_INT_TO_STR, INTENTION_INT_TO_STR

    subtask = subtask_of(key)
    records = []
    for instance_id in instance_ids:
        instance_id = str(instance_id)
        annotations = corpus.annotations(instance_id, subtask)
        if subtask is Subtask.IDENTIFICATION:
            value_int = hard_identification(annotations, corpus.thresholds)
            if value_int is None:
                continue
            value: Any = IDENTIFICATION_INT_TO_STR[value_int]
        elif subtask is Subtask.INTENTION:
            value_int = hard_intention(annotations, corpus.thresholds)
            if value_int is None:
                continue
            value = INTENTION_INT_TO_STR[value_int]
        else:
            multi_hot = corpus.row(instance_id)[corpus.hard_column(subtask)]
            categories = active_categories(np.asarray(multi_hot))
            value = categories or ["NO"]
        records.append({"test_case": PYEVALL_TEST_CASE, "id": instance_id, "value": value})
    return records


def soft_gold(
    soft_labels: Mapping[str, Mapping[str, np.ndarray]], instance_ids: Sequence[str], key: str
) -> list[dict[str, Any]]:
    """Soft gold records: the annotator distribution as ``{label: probability}``."""
    from exist2026.submission import PYEVALL_TEST_CASE

    return [
        {
            "test_case": PYEVALL_TEST_CASE,
            "id": str(instance_id),
            "value": soft_value(key, soft_labels[str(instance_id)][key]),
        }
        for instance_id in instance_ids
    ]


def soft_value(key: str, vector: Sequence[float]) -> dict[str, float]:
    """Name the entries of a soft vector, in this package's canonical order."""
    subtask = subtask_of(key)
    if subtask is Subtask.IDENTIFICATION:
        return {"YES": float(vector[0]), "NO": float(vector[1])}
    if subtask is Subtask.INTENTION:
        return {"NO": float(vector[0]), "DIRECT": float(vector[1]), "JUDGEMENTAL": float(vector[2])}
    value = {"NO": float(vector[0])}
    value.update({c: float(vector[i + 1]) for i, c in enumerate(SEXISM_CATEGORIES)})
    return value


def certain_no(key: str) -> dict[str, float]:
    """The distribution used when the x.1 gate says NO: all mass on NO."""
    value = soft_value(key, [0.0] * (len(SEXISM_CATEGORIES) + 1))
    return {label: (1.0 if label == "NO" else 0.0) for label in value}


# --------------------------------------------------------------------------- #
# Report parsing
# --------------------------------------------------------------------------- #


def _walk(node: Any, metric: str, depth: int = 0, max_depth: int = 8) -> float | None:
    """Find ``metric`` anywhere in a nested report, averaging leaves."""
    if depth > max_depth:
        return None
    if isinstance(node, Mapping):
        if metric in node:
            value = node[metric]
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, Mapping):
                numbers = []
                for entry in value.values():
                    if isinstance(entry, (int, float)):
                        numbers.append(entry)
                    elif isinstance(entry, Mapping):
                        numbers += [v for v in entry.values() if isinstance(v, (int, float))]
                if numbers:
                    return sum(numbers) / len(numbers)
        for entry in node.values():
            found = _walk(entry, metric, depth + 1, max_depth)
            if found is not None:
                return found
    elif isinstance(node, list):
        numbers = [v for v in (_walk(e, metric, depth + 1, max_depth) for e in node) if v is not None]
        if numbers:
            return sum(numbers) / len(numbers)
    return None


def parse_report(report: Any, metrics: Sequence[str]) -> dict[str, float]:
    """Extract ``metrics`` from a PyEvALL report object."""
    if report is None:
        return {}
    found: dict[str, float] = {}
    for attribute in _REPORT_ATTRIBUTES:
        payload = getattr(report, attribute, None)
        if payload is None:
            continue
        if hasattr(payload, "columns"):  # a DataFrame report
            for metric in metrics:
                if metric in payload.columns and metric not in found:
                    values = payload[metric].dropna()
                    if len(values):
                        found[metric] = float(values.mean())
        else:
            for metric in metrics:
                if metric in found:
                    continue
                value = _walk(payload, metric)
                if value is not None:
                    found[metric] = value
    if all(metric in found for metric in metrics):
        return found
    return {**_parse_printed_report(report, metrics), **found}


def _parse_printed_report(report: Any, metrics: Sequence[str]) -> dict[str, float]:
    """Last resort: scrape the numbers PyEvALL prints to stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for method in ("print_report_tsv", "print_report"):
            try:
                getattr(report, method)()
                break
            except Exception:
                continue
    text = buffer.getvalue()
    found: dict[str, float] = {}
    for metric in metrics:
        for alias in _METRIC_ALIASES.get(metric, (metric,)):
            escaped = re.escape(alias)
            match = re.search(rf"\b{escaped}\b\s*[:=]\s*(-?\d+\.\d+)", text) or re.search(
                rf"\b{escaped}\b[^\d-]{{0,100}}(-?\d+\.\d+)", text
            )
            if match:
                found[metric] = float(match.group(1))
                break
    return found


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    predictions: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    key: str,
    metrics: Sequence[str] = HARD_METRICS,
    *,
    work_dir: Path,
) -> dict[str, float]:
    """Score predictions against gold with PyEvALL.

    Predictions without a gold counterpart are dropped first: PyEvALL expects
    aligned files, and an undecided instance is simply not scorable.
    """
    PyEvALLEvaluation, PyEvALLUtils = _pyevall()

    gold_ids = {record["id"] for record in gold}
    aligned = [record for record in predictions if record["id"] in gold_ids]

    work_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = write_json(work_dir / f"_pred_{key}.json", aligned, indent=None)
    gold_path = write_json(work_dir / f"_gold_{key}.json", list(gold), indent=None)

    params: dict[Any, Any] = {PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME}
    hierarchy = hierarchy_for(key)
    if hierarchy is not None:
        params[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy

    report = PyEvALLEvaluation().evaluate(str(prediction_path), str(gold_path), list(metrics), **params)
    return parse_report(report, metrics)
