"""The official vote thresholds, which differ between memes and videos."""

from __future__ import annotations

import numpy as np
import pytest

from exist2026.labels.hard import (
    HardLabelThresholds,
    active_categories,
    categorization_consensus,
    hard_categorization,
    hard_identification,
    hard_intention,
    identification_consensus,
)
from exist2026.labels.soft import (
    soft_categorization,
    soft_identification,
    soft_intention,
    soft_labels,
)
from exist2026.taxonomy import SEXISM_CATEGORIES, Modality

MEMES = HardLabelThresholds.official(Modality.MEMES)
VIDEOS = HardLabelThresholds.official(Modality.VIDEOS)


class TestIdentificationThresholds:
    def test_memes_need_more_than_three_votes(self):
        assert hard_identification(["YES"] * 4 + ["NO"] * 2, MEMES) == 1
        assert hard_identification(["NO"] * 4 + ["YES"] * 2, MEMES) == 0

    def test_memes_reject_a_bare_majority(self):
        # 3-3 and even 3-2 fall below the ">3" threshold and stay undecided.
        assert hard_identification(["YES"] * 3 + ["NO"] * 3, MEMES) is None
        assert hard_identification(["YES"] * 3 + ["NO"] * 2, MEMES) is None

    def test_videos_need_more_than_one_vote(self):
        assert hard_identification(["YES", "YES", "NO"], VIDEOS) == 1
        assert hard_identification(["NO", "NO", "YES"], VIDEOS) == 0

    def test_videos_require_the_winner_not_to_be_beaten(self):
        assert hard_identification(["YES", "NO", "UNKNOWN"], VIDEOS) is None

    def test_videos_drop_unknown_before_counting(self):
        assert hard_identification(["UNKNOWN"] * 3, VIDEOS) is None
        assert hard_identification(["YES", "YES", "UNKNOWN"], VIDEOS) == 1


class TestIntention:
    def test_dash_counts_as_no(self):
        assert hard_intention(["-"] * 4, MEMES) == 0

    def test_memes_need_more_than_two_votes(self):
        assert hard_intention(["DIRECT"] * 3 + ["JUDGEMENTAL"], MEMES) == 1
        assert hard_intention(["DIRECT"] * 2 + ["JUDGEMENTAL"] * 2, MEMES) is None

    def test_videos_need_more_than_one_vote(self):
        assert hard_intention(["JUDGEMENTAL"] * 2 + ["-"], VIDEOS) == 2
        assert hard_intention(["JUDGEMENTAL", "DIRECT", "-"], VIDEOS) is None

    def test_all_unknown_is_undecided(self):
        assert hard_intention(["UNKNOWN", "UNKNOWN"], MEMES) is None


class TestCategorization:
    def test_a_category_needs_more_than_one_vote(self):
        annotations = [["OBJECTIFICATION"], ["OBJECTIFICATION"], ["SEXUAL-VIOLENCE"]]
        assert active_categories(hard_categorization(annotations, MEMES)) == ["OBJECTIFICATION"]

    def test_multi_label_annotations_activate_every_agreed_category(self):
        annotations = [["OBJECTIFICATION", "SEXUAL-VIOLENCE"]] * 3
        assert set(active_categories(hard_categorization(annotations, MEMES))) == {
            "OBJECTIFICATION",
            "SEXUAL-VIOLENCE",
        }

    def test_videos_discard_whole_annotations_containing_unknown(self):
        annotations = [["OBJECTIFICATION", "UNKNOWN"]] * 3
        assert active_categories(hard_categorization(annotations, VIDEOS)) == []
        # Memes only drop the UNKNOWN label itself, not its annotation.
        assert active_categories(hard_categorization(annotations, MEMES)) == ["OBJECTIFICATION"]

    def test_no_consensus_yields_an_all_zero_vector(self):
        vector = hard_categorization([["OBJECTIFICATION"]], MEMES)
        assert vector.shape == (len(SEXISM_CATEGORIES),)
        assert vector.sum() == 0


class TestConsensus:
    def test_identification_consensus_ignores_unknown(self):
        assert identification_consensus(["YES", "YES", "UNKNOWN"]) == 1.0
        assert identification_consensus(["UNKNOWN"]) == 0.0

    def test_categorization_consensus_separates_single_label_from_agreement(self):
        annotations = [["OBJECTIFICATION"], ["OBJECTIFICATION", "SEXUAL-VIOLENCE"]]
        single, agreement = categorization_consensus(annotations, "OBJECTIFICATION")
        assert single == 0.5
        assert agreement == 1.0


class TestSoftLabels:
    def test_identification_is_yes_then_no(self):
        assert soft_identification(["YES", "YES", "NO", "NO"]).tolist() == [0.5, 0.5]

    def test_identification_ignores_unknown(self):
        assert soft_identification(["YES", "UNKNOWN"]).tolist() == [1.0, 0.0]
        assert soft_identification(["UNKNOWN"]) is None

    def test_intention_defaults_to_certain_no(self):
        assert soft_intention(["UNKNOWN"]).tolist() == [1.0, 0.0, 0.0]

    def test_intention_order_is_no_direct_judgemental(self):
        assert soft_intention(["-", "DIRECT", "JUDGEMENTAL", "JUDGEMENTAL"]).tolist() == [
            0.25,
            0.25,
            0.5,
        ]

    def test_categorization_does_not_normalize(self):
        vector = soft_categorization([["OBJECTIFICATION", "SEXUAL-VIOLENCE"]] * 2)
        assert vector[0] == 0.0  # P(NO)
        assert vector.sum() == pytest.approx(2.0)

    def test_categorization_counts_a_pure_dash_as_no(self):
        assert soft_categorization([["-"], ["-"], ["OBJECTIFICATION"]])[0] == pytest.approx(2 / 3)

    def test_soft_labels_drop_instances_without_usable_x1_or_x3(self):
        record = {
            "labels_task2_1": ["UNKNOWN"] * 6,
            "labels_task2_2": ["-"] * 6,
            "labels_task2_3": [["-"]] * 6,
        }
        assert soft_labels(record, Modality.MEMES) is None

    def test_soft_labels_tolerate_the_stray_space_in_the_key(self):
        record = {
            "labels_task2_1": ["YES"] * 6,
            "labels_ task2_2": ["DIRECT"] * 6,
            "labels_task2_3": [["OBJECTIFICATION"]] * 6,
        }
        labels = soft_labels(record, Modality.MEMES)
        assert labels is not None
        assert np.argmax(labels["t22"]) == 1  # DIRECT, read through the odd key
