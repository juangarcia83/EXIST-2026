"""Turning free-form model output into a label, and logging when it fails.

Instruct models do not always honour an output contract: they add "Answer:",
wrap the label in punctuation, or reason at length before committing. The
parser accepts those variants; it does *not* accept guessing. Whenever it falls
back to a default label the event is counted and appended to a JSONL log, so a
run reports how much of its output was actually parsed — a silent fallback rate
is indistinguishable from a working prompt in the metrics alone.

The fallback labels differ by modality by design. Memes let the parser emit NO
in x.2/x.3, matching the hierarchical NO propagation; videos never do, because
anything reaching those subtasks was already judged sexist upstream, so NO
would contradict the cascade.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from exist2026.io_utils import append_jsonl, read_jsonl
from exist2026.taxonomy import SEXISM_CATEGORIES, Modality, Subtask

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_UP_TO_THINK_CLOSE = re.compile(r"^.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^[\s\W_]*ANSWER\s*[:=]\s*")
_THE_ANSWER_IS = re.compile(r"^[\s\W_]*THE\s+ANSWER\s+IS\s+")
_SEPARATORS = re.compile(r"[\s,.;:!?]+")

_INTENTION_SYNONYMS = {
    "DIRECTLY": "DIRECT",
    "JUDGMENT": "JUDGEMENTAL",
    "JUDGMENTAL": "JUDGEMENTAL",
    "JUDGE": "JUDGEMENTAL",
}
_AFFIRMATIVE = {"YES", "Y", "TRUE", "1"}
_NEGATIVE = {"NO", "N", "FALSE", "0"}


def strip_thinking(response: str) -> str:
    """Remove ``<think>…</think>`` reasoning before parsing.

    Chat templates often open the block implicitly, so the decoded text can
    contain only the closing tag; everything before it is then reasoning too.
    """
    if not response:
        return response
    cleaned = _THINK_BLOCK.sub("", response)
    if "</think>" in cleaned.lower():
        cleaned = _UP_TO_THINK_CLOSE.sub("", cleaned)
    return cleaned.strip()


def _normalize(response: str | None) -> str:
    text = strip_thinking(response or "").strip().upper()
    text = _ANSWER_PREFIX.sub("", text)
    return _THE_ANSWER_IS.sub("", text)


@dataclass(frozen=True)
class ParserPolicy:
    """Modality-specific decisions the parser cannot make on its own."""

    #: Label emitted when x.2 cannot be parsed.
    intention_fallback: str
    #: Labels emitted when x.3 cannot be parsed.
    categorization_fallback: tuple[str, ...]
    #: Whether NO is a legal answer in x.2 / x.3.
    allow_no_downstream: bool

    @classmethod
    def for_modality(cls, modality: Modality, categorization_fallback: str | None = None) -> ParserPolicy:
        if modality is Modality.MEMES:
            return cls(
                intention_fallback="NO",
                categorization_fallback=("NO",),
                allow_no_downstream=True,
            )
        fallback = categorization_fallback or "STEREOTYPING-DOMINANCE"
        return cls(
            intention_fallback="DIRECT",
            categorization_fallback=(fallback,),
            allow_no_downstream=False,
        )


@dataclass
class ResponseParser:
    """Stateful parser: one instance per run, so counts stay attributable."""

    policy: ParserPolicy
    log_path: Path | None = None
    total: Counter = field(default_factory=Counter)
    fallbacks: Counter = field(default_factory=Counter)

    def parse(self, response: str | None, subtask: Subtask, key: str, instance_id: str = "?"):
        """Parse one response. ``key`` is the subtask key, used for counters."""
        self.total[key] += 1
        text = _normalize(response)
        if subtask is Subtask.IDENTIFICATION:
            return self._identification(text, response, key, instance_id)
        if subtask is Subtask.INTENTION:
            return self._intention(text, response, key, instance_id)
        return self._categorization(text, response, key, instance_id)

    # -- per subtask -------------------------------------------------------- #

    def _identification(self, text: str, raw: str | None, key: str, instance_id: str) -> str:
        has_yes = bool(re.search(r"\bYES\b", text))
        has_no = bool(re.search(r"\bNO\b", text))
        if has_no and not has_yes:
            return "NO"
        if has_yes and not has_no:
            return "YES"
        first = _SEPARATORS.split(text, maxsplit=1)[0] if text else ""
        if first in _AFFIRMATIVE:
            return "YES"
        if first in _NEGATIVE:
            return "NO"
        return self._fallback(key, instance_id, raw, "NO")

    def _intention(self, text: str, raw: str | None, key: str, instance_id: str) -> str:
        valid = {"DIRECT", "JUDGEMENTAL"}
        if self.policy.allow_no_downstream:
            valid = valid | {"NO"}
        for token in _SEPARATORS.split(text):
            token = _INTENTION_SYNONYMS.get(token, token)
            if token in valid:
                return token
        return self._fallback(key, instance_id, raw, self.policy.intention_fallback)

    def _categorization(self, text: str, raw: str | None, key: str, instance_id: str) -> list[str]:
        valid = set(SEXISM_CATEGORIES)
        if self.policy.allow_no_downstream:
            valid = valid | {"NO"}
        tokens = [t.strip() for t in re.sub(r"[.;!?\n]", ",", text).split(",") if t.strip()]
        found: list[str] = []
        for token in tokens:
            token = token.replace("_", "-").replace(" ", "-")
            if token in valid:
                found.append(token)
                continue
            for category in SEXISM_CATEGORIES:
                if category.startswith(token) or token in category.split("-"):
                    found.append(category)
                    break
        found = list(dict.fromkeys(found))  # de-duplicate, keep order
        categories = [c for c in found if c != "NO"]
        if categories:
            # Charitable: a NO alongside real categories is a formatting slip.
            return categories
        if "NO" in found:
            return ["NO"]
        return self._fallback(key, instance_id, raw, list(self.policy.categorization_fallback))

    # -- bookkeeping -------------------------------------------------------- #

    def _fallback(self, key: str, instance_id: str, raw: str | None, used):
        self.fallbacks[key] += 1
        self.log(key, instance_id, raw or "", used)
        return used

    def log(self, key: str, instance_id: str, raw: str, used, exception: str | None = None) -> None:
        """Record one fallback. Also used by the run loop for failed generations."""
        if self.log_path is None:
            return
        record = {"task": key, "id": str(instance_id), "resp": str(raw)[:500], "fallback": str(used)}
        if exception is not None:
            record["exc_type"] = exception
        append_jsonl(self.log_path, record)

    def record_exception(self, key: str, instance_id: str, error: BaseException, used) -> None:
        self.fallbacks[key] += 1
        self.log(key, instance_id, f"<{type(error).__name__}: {error}>", used, type(error).__name__)

    def fallback_rate(self, key: str) -> float:
        total = self.total[key]
        return self.fallbacks[key] / total if total else 0.0

    def report(self, keys: Sequence[str]) -> pd.DataFrame:
        """Per-subtask fallback summary, split into parser misses and exceptions.

        A healthy run is a few tenths of a percent. Above ~5% the prompt or the
        token budget is the problem, not the model.
        """
        logged = read_jsonl(self.log_path) if self.log_path else []
        rows = []
        for key in keys:
            entries = [r for r in logged if r.get("task") == key]
            n_exceptions = sum(1 for r in entries if r.get("exc_type"))
            total = self.total[key]
            rows.append(
                {
                    "subtask": key,
                    "n_predictions": total,
                    "n_fallback": self.fallbacks[key],
                    "pct_fallback": 100 * self.fallback_rate(key),
                    "n_exception": n_exceptions,
                    "n_parser_miss": max(0, len(entries) - n_exceptions),
                }
            )
        return pd.DataFrame(rows)

    def exception_counts(self) -> Mapping[str, int]:
        logged = read_jsonl(self.log_path) if self.log_path else []
        return Counter(r["exc_type"] for r in logged if r.get("exc_type"))
