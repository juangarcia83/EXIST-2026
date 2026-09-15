"""Shared fixtures: a tiny synthetic corpus in the shape the organizers ship.

The real corpus is not redistributable, so the tests build their own. It is
small enough to reason about by hand and carries every awkward case the loaders
must survive: UNKNOWN votes, the ``"-"`` encoding of NO in x.2, multi-label x.3
annotations, both languages and a missing media file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exist2026.datasets import load_corpus
from exist2026.labels.hard import HardLabelThresholds
from exist2026.taxonomy import Modality


def meme_record(
    *,
    identification,
    intention,
    categorization,
    lang="en",
    text="a meme",
    filename=None,
):
    return {
        "meme": filename,
        "lang": lang,
        "text": text,
        "labels_task2_1": identification,
        "labels_task2_2": intention,
        "labels_task2_3": categorization,
    }


def video_record(*, identification, intention, categorization, lang="en", url="", text="a video"):
    return {
        "video": None,
        "lang": lang,
        "text": text,
        "url": url,
        "labels_task3_1": identification,
        "labels_task3_2": intention,
        "labels_task3_3": categorization,
    }


#: Six annotators per meme, matching the released corpus.
MEME_RECORDS = {
    "100": meme_record(  # unanimous YES, DIRECT, single category
        identification=["YES"] * 6,
        intention=["DIRECT"] * 6,
        categorization=[["STEREOTYPING-DOMINANCE"]] * 6,
        lang="en",
    ),
    "101": meme_record(  # clear YES, JUDGEMENTAL, two categories
        identification=["YES"] * 5 + ["NO"],
        intention=["JUDGEMENTAL"] * 5 + ["-"],
        categorization=[["OBJECTIFICATION", "SEXUAL-VIOLENCE"]] * 4 + [["OBJECTIFICATION"]] * 2,
        lang="es",
    ),
    "102": meme_record(  # unanimous NO
        identification=["NO"] * 6,
        intention=["-"] * 6,
        categorization=[["-"]] * 6,
        lang="en",
    ),
    "103": meme_record(  # 3-3 tie: no hard label anywhere
        identification=["YES"] * 3 + ["NO"] * 3,
        intention=["DIRECT"] * 3 + ["-"] * 3,
        categorization=[["IDEOLOGICAL-INEQUALITY"]] + [["-"]] * 5,
        lang="es",
    ),
    "104": meme_record(  # UNKNOWN votes mixed in
        identification=["YES"] * 4 + ["UNKNOWN"] * 2,
        intention=["DIRECT"] * 4 + ["UNKNOWN"] * 2,
        categorization=[["MISOGYNY-NON-SEXUAL-VIOLENCE"]] * 3 + [["UNKNOWN"]] * 3,
        lang="en",
    ),
    "105": meme_record(
        identification=["YES"] * 6,
        intention=["JUDGEMENTAL"] * 4 + ["DIRECT"] * 2,
        categorization=[["IDEOLOGICAL-INEQUALITY"]] * 5 + [["-"]],
        lang="es",
    ),
    "106": meme_record(
        identification=["YES"] * 5 + ["NO"],
        intention=["DIRECT"] * 5 + ["-"],
        categorization=[["SEXUAL-VIOLENCE"]] * 4 + [["-"]] * 2,
        lang="en",
    ),
}

#: Three annotators per video, with the > 1 vote threshold.
VIDEO_RECORDS = {
    "200": video_record(
        identification=["YES"] * 3,
        intention=["DIRECT"] * 3,
        categorization=[["STEREOTYPING-DOMINANCE"]] * 3,
        lang="en",
        url="https://www.tiktok.com/@creator_one/video/1",
    ),
    "201": video_record(
        identification=["YES", "YES", "NO"],
        intention=["JUDGEMENTAL", "JUDGEMENTAL", "-"],
        categorization=[["OBJECTIFICATION"], ["OBJECTIFICATION"], ["-"]],
        lang="es",
        url="https://www.tiktok.com/@creator_two/video/2",
    ),
    "202": video_record(
        identification=["NO"] * 3,
        intention=["-"] * 3,
        categorization=[["-"]] * 3,
        lang="es",
        url="https://www.tiktok.com/@creator_one/video/3",
    ),
    "203": video_record(  # single YES vote: below threshold, no hard label
        identification=["YES", "NO", "UNKNOWN"],
        intention=["DIRECT", "-", "UNKNOWN"],
        categorization=[["SEXUAL-VIOLENCE"], ["-"], ["UNKNOWN"]],
        lang="en",
        url="https://www.tiktok.com/@creator_three/video/4",
    ),
}


def _write_corpus(tmp_path: Path, records: dict, modality: Modality, *, missing: set[str] = frozenset()):
    directory = tmp_path / modality.value
    media_dir = directory / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    for instance_id in records:
        if instance_id in missing:
            continue
        (media_dir / f"{instance_id}{modality.media_suffix}").write_bytes(b"stub")
    json_path = directory / "train.json"
    json_path.write_text(json.dumps(records), encoding="utf-8")
    return json_path, media_dir


@pytest.fixture
def meme_corpus(tmp_path):
    json_path, media_dir = _write_corpus(tmp_path, MEME_RECORDS, Modality.MEMES)
    return load_corpus(
        Modality.MEMES,
        HardLabelThresholds.official(Modality.MEMES),
        train_json=json_path,
        train_media_dir=media_dir,
    )


@pytest.fixture
def video_corpus(tmp_path):
    json_path, media_dir = _write_corpus(tmp_path, VIDEO_RECORDS, Modality.VIDEOS)
    return load_corpus(
        Modality.VIDEOS,
        HardLabelThresholds.official(Modality.VIDEOS),
        train_json=json_path,
        train_media_dir=media_dir,
    )


@pytest.fixture
def corpus_paths(tmp_path):
    """Factory for corpora with deliberately missing media."""

    def build(records, modality, missing=frozenset()):
        return _write_corpus(tmp_path, records, modality, missing=missing)

    return build
