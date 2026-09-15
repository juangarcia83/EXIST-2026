"""Small filesystem helpers shared by both system families."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def first_existing(candidates: Iterable[Path]) -> Path | None:
    """First path that exists, or ``None``.

    The dataset ships under slightly different directory layouts depending on
    how it was downloaded, so every dataset path is declared as a list of
    candidates instead of a single hard-coded location.
    """
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON, replacing undecodable bytes.

    Some released records contain stray bytes; ``errors="replace"`` keeps a
    whole run from dying on a single corrupt character.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Write UTF-8 JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    return path


def append_jsonl(path: Path, record: Any) -> None:
    """Append one JSON record to a line-delimited log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    """Read a line-delimited JSON log, skipping malformed lines."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
