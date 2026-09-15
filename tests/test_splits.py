"""Splitting: no creator on both sides, and a balanced meme split."""

from __future__ import annotations

import numpy as np
import pytest

from exist2026.fusion.splits import Split, build_split, creator_of, yes_subset
from exist2026.taxonomy import Modality


def soft(p_yes: float) -> dict[str, np.ndarray]:
    return {"t31": np.array([p_yes, 1 - p_yes], dtype=np.float32)}


class TestCreatorExtraction:
    def test_reads_the_handle_from_a_tiktok_url(self):
        record = {"url": "https://www.tiktok.com/@some_creator.1/video/123"}
        assert creator_of(record, "9") == "some_creator.1"

    def test_falls_back_to_a_unique_group_without_a_url(self):
        assert creator_of({}, "9") == "__solo_9"

    def test_two_instances_without_urls_are_not_grouped_together(self):
        assert creator_of({}, "1") != creator_of({}, "2")


class TestVideoSplit:
    @pytest.fixture
    def data(self):
        records, labels = {}, {}
        for creator in range(40):
            for video in range(2):
                instance_id = f"{creator}_{video}"
                records[instance_id] = {
                    "url": f"https://www.tiktok.com/@creator{creator}/video/{video}",
                    "lang": "en" if creator % 2 else "es",
                }
                labels[instance_id] = soft(0.9 if creator % 3 else 0.1)
        return records, labels

    def test_no_creator_appears_on_both_sides(self, data):
        records, labels = data
        split = build_split(records, labels, "t31", Modality.VIDEOS, seed=0)
        train_creators = {creator_of(records[i], i) for i in split.train}
        val_creators = {creator_of(records[i], i) for i in split.val}
        assert not (train_creators & val_creators)

    def test_every_instance_lands_somewhere(self, data):
        records, labels = data
        split = build_split(records, labels, "t31", Modality.VIDEOS, seed=0)
        assert len(split.train) + len(split.val) == len(labels)

    def test_the_split_is_reproducible(self, data):
        records, labels = data
        first = build_split(records, labels, "t31", Modality.VIDEOS, seed=7)
        second = build_split(records, labels, "t31", Modality.VIDEOS, seed=7)
        assert first.train == second.train


class TestMemeSplit:
    @pytest.fixture
    def data(self):
        records, labels = {}, {}
        for i in range(60):
            instance_id = str(i)
            records[instance_id] = {"lang": "en" if i % 2 else "es"}
            labels[instance_id] = {"t21": np.array([0.9 if i % 3 else 0.1, 0.1], dtype=np.float32)}
        return records, labels

    def test_both_languages_reach_validation(self, data):
        records, labels = data
        split = build_split(records, labels, "t21", Modality.MEMES, val_fraction=0.25, seed=1)
        languages = {records[i]["lang"] for i in split.val}
        assert languages == {"en", "es"}

    def test_the_validation_fraction_is_honoured(self, data):
        records, labels = data
        split = build_split(records, labels, "t21", Modality.MEMES, val_fraction=0.25, seed=1)
        assert len(split.val) == pytest.approx(15, abs=1)


class TestSmallGroupFallback:
    def test_too_few_creators_for_four_strata_still_splits_by_group(self):
        records, labels = {}, {}
        for creator in range(6):
            for video in range(2):
                instance_id = f"{creator}_{video}"
                records[instance_id] = {
                    "url": f"https://www.tiktok.com/@creator{creator}/video/{video}",
                    "lang": "en",
                }
                labels[instance_id] = soft(0.9 if creator % 2 else 0.1)
        split = build_split(records, labels, "t31", Modality.VIDEOS, seed=0)
        train_creators = {creator_of(records[i], i) for i in split.train}
        val_creators = {creator_of(records[i], i) for i in split.val}
        assert val_creators
        assert not (train_creators & val_creators)


class TestGuards:
    def test_an_overlapping_split_is_rejected(self):
        import pandas as pd

        with pytest.raises(ValueError, match="both sides"):
            Split(train=["a", "b"], val=["b"], frame=pd.DataFrame())


class TestYesSubset:
    def test_keeps_only_instances_above_the_threshold(self):
        labels = {"1": soft(0.9), "2": soft(0.4), "3": soft(0.5)}
        assert yes_subset(["1", "2", "3"], labels, "t31") == ["1", "3"]

    def test_the_threshold_is_configurable(self):
        labels = {"1": soft(0.9), "2": soft(0.6)}
        assert yes_subset(["1", "2"], labels, "t31", threshold=0.8) == ["1"]
