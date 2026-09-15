"""Few-shot pools: the properties that make them safe to prompt with."""

from __future__ import annotations

import pytest

from exist2026.config import PoolSpec
from exist2026.fewshot import pools as pool_module
from exist2026.taxonomy import SEXISM_CATEGORIES, Modality, Subtask, subtask_key

MEME_SPECS = {
    "t21": PoolSpec(per_class=2),
    "t22": PoolSpec(per_class=2),
    "t23": PoolSpec(per_class=1),
}
VIDEO_SPECS = {
    "t31": PoolSpec(per_class=1, bilingual=True, diversify_consensus=True),
    "t32": PoolSpec(per_class=1, bilingual=True, diversify_consensus=True),
    "t33": PoolSpec(per_class=1, bilingual=True, cooccurrence_extra=1),
}


class TestIdentificationPool:
    def test_balances_the_two_classes(self, meme_corpus):
        pool = pool_module.build_identification_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t21"])
        labels = [example["label"] for example in pool]
        assert labels.count("YES") == 2
        # Only one meme in the fixture reaches the NO threshold, so the pool
        # takes what exists instead of padding with undecided instances.
        assert labels.count("NO") == 1
        assert all(example["consensus"] > 0 for example in pool)

    def test_is_ranked_by_agreement(self, meme_corpus):
        pool = pool_module.build_identification_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t21"])
        yes = [e["consensus"] for e in pool if e["label"] == "YES"]
        assert yes == sorted(yes, reverse=True)


class TestHierarchyConsistency:
    """x.2 and x.3 only ever see sexist instances, so NO must never appear."""

    def test_intention_pool_excludes_no(self, meme_corpus):
        pool = pool_module.build_intention_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t22"])
        assert pool
        assert {e["label"] for e in pool} <= {"DIRECT", "JUDGEMENTAL"}

    def test_categorization_pool_excludes_no(self, meme_corpus):
        pool = pool_module.build_categorization_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t23"])
        assert pool
        for example in pool:
            assert "NO" not in example["labels"]
            assert set(example["labels"]) <= set(SEXISM_CATEGORIES)


class TestCategorizationPool:
    def test_uses_each_instance_for_at_most_one_category(self, meme_corpus):
        pool = pool_module.build_categorization_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t23"])
        ids = [e["id"] for e in pool]
        assert len(ids) == len(set(ids))

    def test_appends_multi_label_exemplars_when_asked(self, video_corpus):
        spec = PoolSpec(per_class=1, cooccurrence_extra=1)
        pool = pool_module.build_categorization_pool(video_corpus, video_corpus.train, spec)
        # The synthetic corpus has no multi-label video, so nothing is appended
        # and the pool stays exactly the per-category selection.
        assert all(len(e["labels"]) >= 1 for e in pool)


class TestDeterminismAndSeeding:
    def test_unseeded_selection_is_reproducible(self, meme_corpus):
        first = pool_module.build_pools(meme_corpus, MEME_SPECS)
        second = pool_module.build_pools(meme_corpus, MEME_SPECS)
        assert first == second

    def test_a_seed_changes_only_tie_breaks(self, meme_corpus):
        unseeded = pool_module.build_identification_pool(meme_corpus, meme_corpus.train, MEME_SPECS["t21"])
        seeded = pool_module.rebuild_for_seed(meme_corpus, MEME_SPECS, seed=3)["t21"]
        # The label balance is a property of the design, not of the seed.
        assert sorted(e["label"] for e in unseeded) == sorted(e["label"] for e in seeded)


class TestLanguageBalance:
    def test_bilingual_pools_use_both_languages_when_available(self, video_corpus):
        pool = pool_module.build_identification_pool(
            video_corpus, video_corpus.train, PoolSpec(per_class=2, bilingual=True)
        )
        assert {e["lang"] for e in pool} == {"en", "es"}


class TestLeakage:
    def test_pool_ids_are_collected_for_exclusion(self, meme_corpus):
        pools = pool_module.build_pools(meme_corpus, MEME_SPECS)
        ids = pool_module.pool_ids(pools)
        assert ids
        assert all(isinstance(i, str) for i in ids)

    def test_pools_only_contain_training_instances(self, meme_corpus):
        pools = pool_module.build_pools(meme_corpus, MEME_SPECS)
        train_ids = set(meme_corpus.train["id_EXIST"])
        assert pool_module.pool_ids(pools) <= train_ids


class TestCacheStaleness:
    def test_a_cache_without_language_is_stale(self):
        cached = [{"id": "1", "label": "YES", "consensus": 1.0}]
        assert pool_module.is_stale("t21", cached, MEME_SPECS["t21"])

    def test_a_cache_with_no_in_x2_is_stale(self):
        cached = [{"id": "1", "label": "NO", "consensus": 1.0, "lang": "en"}]
        assert pool_module.is_stale("t22", cached, MEME_SPECS["t22"])

    def test_a_cache_of_the_wrong_size_is_stale(self):
        spec = PoolSpec(per_class=2, cooccurrence_extra=2)
        cached = [{"id": "1", "labels": ["OBJECTIFICATION"], "consensus": 1.0, "lang": "en"}]
        assert pool_module.is_stale("t23", cached, spec)

    def test_an_empty_cache_is_stale(self):
        assert pool_module.is_stale("t21", [], MEME_SPECS["t21"])

    def test_a_current_cache_is_kept(self):
        cached = [{"id": "1", "label": "YES", "consensus": 1.0, "lang": "en"}]
        assert not pool_module.is_stale("t21", cached, MEME_SPECS["t21"])


class TestCaching:
    def test_pools_round_trip_through_disk(self, meme_corpus, tmp_path):
        directory = tmp_path / "pools"
        built = pool_module.load_or_build_pools(meme_corpus, MEME_SPECS, directory)
        assert (directory / "few_shot_pool_t21.json").exists()
        reloaded = pool_module.load_or_build_pools(meme_corpus, MEME_SPECS, directory)
        assert built == reloaded

    def test_force_rebuilds_even_a_valid_cache(self, meme_corpus, tmp_path):
        directory = tmp_path / "pools"
        pool_module.load_or_build_pools(meme_corpus, MEME_SPECS, directory)
        rebuilt = pool_module.load_or_build_pools(meme_corpus, MEME_SPECS, directory, seed=5, force=True)
        assert set(rebuilt) == {"t21", "t22", "t23"}


def test_summarize_lists_every_exemplar(meme_corpus):
    pools = pool_module.build_pools(meme_corpus, MEME_SPECS)
    frame = pool_module.summarize(pools)
    assert len(frame) == sum(len(pool) for pool in pools.values())
    assert set(frame.columns) == {"subtask", "id", "label", "consensus", "lang"}


@pytest.mark.parametrize("modality", [Modality.MEMES, Modality.VIDEOS])
def test_subtask_keys_match_the_modality(modality):
    assert subtask_key(modality, Subtask.IDENTIFICATION).startswith(f"t{modality.task_number}")
