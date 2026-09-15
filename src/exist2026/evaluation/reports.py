"""Diagnostic reports that sit next to the official metrics.

PyEvALL gives one number per subtask; these give the shape of the errors. The
important design decision is the *hierarchical-conditional* filtering of x.2
and x.3: predictions for those subtasks are only the model's when x.1 said YES.
Where x.1 said NO the label was propagated by the cascade, not chosen, so
scoring those rows would measure the hierarchy rather than the model — and
would inflate accuracy with easy, automatic NOs.

Subtask x.1 keeps its full evaluation: there, NO is a genuine decision.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exist2026.taxonomy import (
    IDENTIFICATION_INT_TO_STR,
    INTENTION_INT_TO_STR,
    SEXISM_CATEGORIES,
    Subtask,
    subtask_of,
)


@dataclass
class ComparisonResult:
    """Per-instance comparison plus the headline numbers of one subtask."""

    key: str
    frame: pd.DataFrame
    metrics: dict[str, float]
    note: str = ""

    @property
    def errors(self) -> pd.DataFrame:
        if self.frame.empty:
            return self.frame
        if "ok" in self.frame.columns:
            return self.frame[~self.frame["ok"]]
        return self.frame[self.frame.apply(lambda r: set(r["gold"]) != set(r["prediction"]), axis=1)]


def compare_identification(predictions: Mapping[str, str], corpus, key: str) -> ComparisonResult:
    """Full binary comparison — both classes are real decisions here."""
    column = corpus.hard_column(Subtask.IDENTIFICATION)
    rows = []
    for instance_id, prediction in predictions.items():
        gold_int = corpus.row(instance_id)[column]
        if pd.isna(gold_int):
            continue
        gold = IDENTIFICATION_INT_TO_STR[int(gold_int)]
        rows.append({"id": instance_id, "gold": gold, "prediction": prediction, "ok": prediction == gold})
    frame = pd.DataFrame(rows)
    metrics = {"n": len(frame), "accuracy": float(frame["ok"].mean()) if len(frame) else float("nan")}
    return ComparisonResult(key=key, frame=frame, metrics=metrics)


def compare_intention(predictions: Mapping[str, str], corpus, key: str) -> ComparisonResult:
    """Comparison restricted to instances whose gold is DIRECT or JUDGEMENTAL.

    A predicted NO on a sexist gold still counts as an error; it simply cannot
    match either label.
    """
    column = corpus.hard_column(Subtask.INTENTION)
    rows = []
    for instance_id, prediction in predictions.items():
        gold_int = corpus.row(instance_id)[column]
        if pd.isna(gold_int):
            continue
        gold = INTENTION_INT_TO_STR[int(gold_int)]
        if gold == "NO":
            continue
        rows.append({"id": instance_id, "gold": gold, "prediction": prediction, "ok": prediction == gold})
    frame = pd.DataFrame(rows)
    n_no = int((frame["prediction"] == "NO").sum()) if len(frame) else 0
    metrics = {
        "n": len(frame),
        "accuracy": float(frame["ok"].mean()) if len(frame) else float("nan"),
        "n_predicted_no_on_sexist_gold": n_no,
    }
    note = f"{n_no} sexist instances received prediction NO (counted as errors)" if n_no else ""
    return ComparisonResult(key=key, frame=frame, metrics=metrics, note=note)


def compare_categorization(predictions: Mapping[str, Sequence[str]], corpus, key: str) -> ComparisonResult:
    """Multi-label comparison over instances with at least one gold category."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer

    column = corpus.hard_column(Subtask.CATEGORIZATION)
    rows, gold_sets, pred_sets = [], [], []
    for instance_id, prediction in predictions.items():
        multi_hot = corpus.row(instance_id)[column]
        if multi_hot is None:
            continue
        gold = [c for c, on in zip(SEXISM_CATEGORIES, np.asarray(multi_hot), strict=True) if on > 0.5]
        if not gold:
            continue  # gold is not sexist: the prediction was propagated, not chosen
        predicted = [c for c in prediction if c != "NO"]
        gold_sets.append(gold)
        pred_sets.append(predicted)
        rows.append({"id": instance_id, "gold": gold, "prediction": predicted})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return ComparisonResult(key=key, frame=frame, metrics={"n": 0}, note="no sexist gold in the eval set")

    binarizer = MultiLabelBinarizer(classes=list(SEXISM_CATEGORIES))
    y_true = binarizer.fit_transform(gold_sets)
    y_pred = binarizer.transform(pred_sets)

    per_class = {}
    for i, category in enumerate(SEXISM_CATEGORIES):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        per_class[category] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(y_true[:, i].sum()),
        }

    exact = sum(1 for g, p in zip(gold_sets, pred_sets, strict=True) if set(g) == set(p))
    metrics = {
        "n": len(frame),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": exact / len(frame),
        "n_empty_predictions": sum(1 for p in pred_sets if not p),
        "per_class": per_class,
    }
    return ComparisonResult(key=key, frame=frame, metrics=metrics)


def compare(predictions: Mapping[str, Any], corpus, key: str) -> ComparisonResult:
    """Dispatch to the right comparison for ``key``."""
    subtask = subtask_of(key)
    if subtask is Subtask.IDENTIFICATION:
        return compare_identification(predictions, corpus, key)
    if subtask is Subtask.INTENTION:
        return compare_intention(predictions, corpus, key)
    return compare_categorization(predictions, corpus, key)


def confusion(result: ComparisonResult, labels: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix with readable row/column names."""
    from sklearn.metrics import confusion_matrix

    if result.frame.empty:
        return pd.DataFrame()
    matrix = confusion_matrix(result.frame["gold"], result.frame["prediction"], labels=list(labels))
    return pd.DataFrame(matrix, index=[f"gold_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def save_errors(result: ComparisonResult, corpus, directory: Path) -> Path | None:
    """Write the misclassified instances, with their text, for manual review."""
    errors = result.errors
    if errors.empty:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    enriched = errors.merge(
        corpus.frame[["id_EXIST", "lang", "text"]], left_on="id", right_on="id_EXIST", how="left"
    )
    path = directory / f"errors_{result.key}.csv"
    enriched[["id", "lang", "gold", "prediction", "text"]].to_csv(path, index=False, encoding="utf-8")
    return path


def prediction_distribution(predictions: Mapping[str, Any]) -> pd.DataFrame:
    """Label frequencies — the only feedback available on an unlabelled TEST split."""
    counts: Counter[str] = Counter()
    for value in predictions.values():
        if isinstance(value, list):
            counts["NO" if value == ["NO"] else "+".join(sorted(value))] += 1
        else:
            counts[str(value)] += 1
    total = sum(counts.values()) or 1
    return pd.DataFrame(
        [{"label": label, "n": n, "pct": 100 * n / total} for label, n in counts.most_common()]
    )
