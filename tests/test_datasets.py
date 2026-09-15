"""Corpus loading: one table for both modalities."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import MEME_RECORDS
from exist2026.datasets import describe, exclude, load_corpus
from exist2026.labels.hard import HardLabelThresholds
from exist2026.taxonomy import Modality, Subtask


class TestLoading:
    def test_builds_one_row_per_instance(self, meme_corpus):
        assert len(meme_corpus.frame) == len(MEME_RECORDS)
        assert set(meme_corpus.frame["split"]) == {"train"}

    def test_derives_the_media_filename_when_absent(self, meme_corpus):
        assert meme_corpus.media_of("100").endswith("100.jpeg")

    def test_exposes_hard_labels_under_subtask_keys(self, meme_corpus):
        row = meme_corpus.row("100")
        assert row[meme_corpus.hard_column(Subtask.IDENTIFICATION)] == 1
        assert row[meme_corpus.hard_column(Subtask.INTENTION)] == 1  # DIRECT

    def test_undecided_instances_have_no_hard_label(self, meme_corpus):
        # pandas stores the absent label as NaN in a mixed column, so callers
        # must test it with pd.isna: `value is None` silently never matches.
        value = meme_corpus.row("103")[meme_corpus.hard_column(Subtask.IDENTIFICATION)]
        assert pd.isna(value)

    def test_unknown_id_raises(self, meme_corpus):
        with pytest.raises(KeyError):
            meme_corpus.row("does-not-exist")

    def test_video_thresholds_are_applied(self, video_corpus):
        column = video_corpus.hard_column(Subtask.IDENTIFICATION)
        assert video_corpus.row("201")[column] == 1  # 2 of 3 votes suffice
        assert pd.isna(video_corpus.row("203")[column])  # 1-1 after dropping UNKNOWN


class TestMediaValidation:
    def test_a_missing_media_file_is_reported(self, corpus_paths):
        json_path, media_dir = corpus_paths(MEME_RECORDS, Modality.MEMES, missing={"100"})
        with pytest.raises(FileNotFoundError, match="have no media file"):
            load_corpus(
                Modality.MEMES,
                HardLabelThresholds.official(Modality.MEMES),
                train_json=json_path,
                train_media_dir=media_dir,
            )

    def test_the_check_can_be_waived(self, corpus_paths):
        json_path, media_dir = corpus_paths(MEME_RECORDS, Modality.MEMES, missing={"100"})
        corpus = load_corpus(
            Modality.MEMES,
            HardLabelThresholds.official(Modality.MEMES),
            train_json=json_path,
            train_media_dir=media_dir,
            require_media=False,
        )
        assert not corpus.row("100")["media_ok"]


class TestHelpers:
    def test_annotations_are_read_by_subtask(self, meme_corpus):
        assert meme_corpus.annotations("100", Subtask.IDENTIFICATION) == ["YES"] * 6

    def test_most_frequent_category_reflects_the_training_split(self, meme_corpus):
        assert meme_corpus.most_frequent_category() in {
            "STEREOTYPING-DOMINANCE",
            "OBJECTIFICATION",
            "IDEOLOGICAL-INEQUALITY",
            "SEXUAL-VIOLENCE",
            "MISOGYNY-NON-SEXUAL-VIOLENCE",
        }

    def test_exclude_drops_the_given_ids(self, meme_corpus):
        kept = exclude(meme_corpus.train, {"100", "101"})
        assert "100" not in set(kept["id_EXIST"])
        assert len(kept) == len(meme_corpus.train) - 2

    def test_describe_summarizes_each_split(self, meme_corpus):
        summary = describe(meme_corpus)
        assert list(summary["split"]) == ["train"]
        assert summary.iloc[0]["n"] == len(MEME_RECORDS)
