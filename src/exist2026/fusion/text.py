"""The enriched textual view (``capsanocr``) fed to the trained encoder.

Each instance is represented by the concatenation of three complementary
sources rather than by its on-screen text alone:

``cap``
    A *neutral* visual description of what the content shows, produced by
    Qwen3-VL — people, gestures, setting — with no judgement attached.
``san``
    An independent gender-relevance analysis, produced by a **separate** prompt.
    Keeping the sexism reasoning in its own generation is what prevents the
    description from being written to fit a conclusion the model already drew.
``ocr``
    The literal on-screen text: meme overlay text for Task 2, transcription and
    on-screen text (plus ASR) for Task 3.

Videos use explicit section markers because their view has four parts and the
encoder benefits from knowing which is which; memes join theirs with ``[SEP]``.
Both views degrade gracefully: with no VLM file at all, the text is the OCR.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from exist2026.io_utils import read_json
from exist2026.taxonomy import Modality

SEP = " [SEP] "


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_vlm_enrichment(path: Path | None) -> dict[str, Mapping[str, Any]]:
    """Load the VLM description/analysis file, keyed by instance id.

    Accepts both shapes the cache has been written in: a list of records with
    an ``id_EXIST`` field, or a plain ``{id: record}`` mapping.
    """
    if path is None or not path.exists():
        return {}
    raw = read_json(path)
    if isinstance(raw, list):
        return {str(record.get("id_EXIST")): record for record in raw}
    return {str(key): value for key, value in raw.items()}


def build_text(instance: Mapping[str, Any], enrichment: Mapping[str, Any] | None, modality: Modality) -> str:
    """The textual document of one instance."""
    if modality is Modality.VIDEOS:
        return _video_text(instance, enrichment)
    return _meme_text(instance, enrichment)


def _video_text(instance: Mapping[str, Any], enrichment: Mapping[str, Any] | None) -> str:
    entry = enrichment or {}
    parsed = entry.get("qwen_parsed") or {}
    if not isinstance(parsed, Mapping):
        parsed = {}
    analysis = _clean(parsed.get("sexism_analysis"))
    description = _clean(parsed.get("visual_description"))
    ocr = _clean(parsed.get("ocr_text")) or _clean(instance.get("text"))
    asr = _clean(entry.get("asr_transcript"))
    return f"[ANALYSIS] {analysis} [DESC] {description} [OCR] {ocr} [ASR] {asr}"


def _meme_text(instance: Mapping[str, Any], enrichment: Mapping[str, Any] | None) -> str:
    entry = enrichment or {}
    caption = _clean(entry.get("visual_description"))
    analysis = _clean(entry.get("sexism_analysis") or entry.get("social_analysis"))
    ocr = _clean(instance.get("text"))
    parts = [part for part in (caption, analysis, ocr) if part]
    return SEP.join(parts)


class TextView:
    """Bound ``build_text`` — the dataset calls this once per item."""

    def __init__(
        self, records: Mapping[str, Mapping[str, Any]], enrichment: Mapping[str, Any], modality: Modality
    ) -> None:
        self.records = records
        self.enrichment = enrichment
        self.modality = modality

    def __call__(self, instance_id: str) -> str:
        return build_text(
            self.records[str(instance_id)], self.enrichment.get(str(instance_id)), self.modality
        )
