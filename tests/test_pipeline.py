"""The cascade itself, driven by a scripted backend instead of a real VLM.

A stub backend makes the pipeline's own behaviour testable: the hierarchy, the
checkpoint resume and the exclusion of pool members are properties of this code,
not of whichever model happens to be loaded.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from exist2026.config import PoolSpec, load_fewshot_config
from exist2026.fewshot import pools as pool_module
from exist2026.fewshot.parsing import ParserPolicy, ResponseParser
from exist2026.fewshot.pipeline import (
    CheckpointStore,
    evaluation_frame,
    exception_fallback,
    majority_baseline,
    propagated_value,
    run_cascade,
)
from exist2026.fewshot.prompts import builders_for
from exist2026.taxonomy import Modality

SPECS = {"t21": PoolSpec(per_class=1), "t22": PoolSpec(per_class=1), "t23": PoolSpec(per_class=1)}

# Fixed pools rather than built ones: the fixture corpus is small enough that a
# generated pool would hold out nearly every instance, leaving nothing to
# evaluate. Pinning them keeps this file about the cascade.
POOLS = {
    "t21": [
        {"id": "100", "label": "YES", "consensus": 1.0, "lang": "en"},
        {"id": "102", "label": "NO", "consensus": 1.0, "lang": "en"},
    ],
    "t22": [
        {"id": "100", "label": "DIRECT", "consensus": 1.0, "lang": "en"},
        {"id": "105", "label": "JUDGEMENTAL", "consensus": 0.83, "lang": "es"},
    ],
    "t23": [{"id": "100", "labels": ["STEREOTYPING-DOMINANCE"], "consensus": 1.0, "lang": "en"}],
}


class ScriptedBackend:
    """Returns answers from a per-subtask script and records every call."""

    def __init__(self, answers, failures=()):
        self.answers = answers
        self.failures = set(failures)
        self.calls = []

    def generate(self, messages, max_new_tokens):
        # The subtask is identifiable from the question in the final user turn.
        question = messages[-1]["content"][1]["text"]
        if "is this meme sexist" in question:
            key = "t21"
        elif "source intention" in question:
            key = "t22"
        else:
            key = "t23"
        media = messages[-1]["content"][0]["image"]
        instance_id = media.rsplit("/", 1)[-1].split(".")[0]
        self.calls.append((key, instance_id))
        if (key, instance_id) in self.failures:
            raise RuntimeError("simulated generation failure")
        return self.answers[key](instance_id)


@pytest.fixture
def setup(meme_corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("EXIST2026_ROOT", str(tmp_path))
    config = replace(
        load_fewshot_config("fewshot_memes", stage="sanity"),
        max_eval_instances=None,
        checkpoint_every=2,
        pools=SPECS,
    )
    config.dirs.create()
    pools = POOLS
    blocked = pool_module.pool_ids(pools)
    parser = ResponseParser(ParserPolicy.for_modality(Modality.MEMES), config.dirs.logs / "fb.log")
    return config, meme_corpus, pools, blocked, parser


def run(setup, backend, store=None):
    config, corpus, pools, blocked, parser = setup
    store = store or CheckpointStore(config.dirs.checkpoints, config.checkpoint_suffix)
    result = run_cascade(
        backend=backend,
        builders=builders_for(Modality.MEMES),
        pools=pools,
        parser=parser,
        corpus=corpus,
        config=config,
        store=store,
        blocked_ids=blocked,
    )
    return result, store, parser


ALL_YES = {
    "t21": lambda i: "YES",
    "t22": lambda i: "DIRECT",
    "t23": lambda i: "OBJECTIFICATION",
}


class TestHierarchy:
    def test_a_no_from_x1_propagates_without_a_generation(self, setup):
        backend = ScriptedBackend(
            {"t21": lambda i: "NO", "t22": lambda i: "DIRECT", "t23": lambda i: "OBJECTIFICATION"}
        )
        result, _, _ = run(setup, backend)
        keys = setup[0].subtask_keys
        assert all(value == "NO" for value in result[keys[1]].values())
        assert all(value == ["NO"] for value in result[keys[2]].values())
        # x.2 and x.3 were never prompted at all.
        assert {key for key, _ in backend.calls} == {"t21"}

    def test_x2_and_x3_only_run_on_instances_x1_called_sexist(self, setup):
        sexist = {"101", "104"}
        backend = ScriptedBackend(
            {
                "t21": lambda i: "YES" if i in sexist else "NO",
                "t22": lambda i: "DIRECT",
                "t23": lambda i: "OBJECTIFICATION",
            }
        )
        result, _, _ = run(setup, backend)
        prompted = {instance for key, instance in backend.calls if key == "t22"}
        assert prompted <= sexist
        assert result.sexist_ids == sexist & set(result[setup[0].subtask_keys[0]])

    def test_the_gate_overrides_a_cached_downstream_value(self, setup):
        config, *_ = setup
        store = CheckpointStore(config.dirs.checkpoints, config.checkpoint_suffix)
        key2 = config.subtask_keys[1]
        store.save(key2, {"101": "JUDGEMENTAL"})  # stale value from an earlier design
        backend = ScriptedBackend(
            {"t21": lambda i: "NO", "t22": lambda i: "DIRECT", "t23": lambda i: "OBJECTIFICATION"}
        )
        result, _, _ = run(setup, backend, store)
        assert result[key2]["101"] == "NO"


class TestLeakage:
    def test_pool_members_are_never_evaluated(self, setup):
        config, _corpus, _pools, blocked, _ = setup
        backend = ScriptedBackend(ALL_YES)
        result, _, _ = run(setup, backend)
        assert blocked
        assert not (set(result[config.subtask_keys[0]]) & blocked)


class TestCheckpointing:
    def test_a_second_run_skips_everything_already_predicted(self, setup):
        config = setup[0]
        store = CheckpointStore(config.dirs.checkpoints, config.checkpoint_suffix)
        first = ScriptedBackend(ALL_YES)
        run(setup, first, store)
        assert first.calls

        second = ScriptedBackend(ALL_YES)
        run(setup, second, store)
        assert second.calls == []

    def test_checkpoints_are_namespaced_by_stage(self, setup):
        config = setup[0]
        sanity = CheckpointStore(config.dirs.checkpoints, "sanity800")
        submission = CheckpointStore(config.dirs.checkpoints, "test")
        assert sanity.path("t21") != submission.path("t21")


class TestFailures:
    def test_a_generation_failure_falls_back_and_is_logged(self, setup):
        config = setup[0]
        key1 = config.subtask_keys[0]
        backend = ScriptedBackend(ALL_YES, failures={("t21", "101")})
        result, _, parser = run(setup, backend)
        assert result[key1]["101"] == "NO"
        assert parser.exception_counts()["RuntimeError"] == 1

    def test_one_failure_does_not_stop_the_run(self, setup):
        config = setup[0]
        backend = ScriptedBackend(ALL_YES, failures={("t21", "101")})
        result, _, _ = run(setup, backend)
        assert len(result[config.subtask_keys[0]]) > 1


class TestHelpers:
    def test_propagated_values_match_the_subtask_shape(self):
        assert propagated_value("t22") == "NO"
        assert propagated_value("t23") == ["NO"]

    def test_exception_fallbacks_follow_the_parser_policy(self):
        videos = ResponseParser(ParserPolicy.for_modality(Modality.VIDEOS, "OBJECTIFICATION"))
        assert exception_fallback("t31", videos) == "NO"
        assert exception_fallback("t32", videos) == "DIRECT"
        assert exception_fallback("t33", videos) == ["OBJECTIFICATION"]

    def test_evaluation_frames_exclude_pool_members(self, setup):
        config, corpus, _, blocked, _ = setup
        frame = evaluation_frame(config, corpus, blocked, config.subtask_keys[0])
        assert not (set(frame["id_EXIST"]) & blocked)

    def test_majority_baseline_is_hierarchy_consistent(self, setup):
        config, corpus, _, blocked, _ = setup
        baseline = majority_baseline(config, corpus, blocked)
        k1, k2, _ = config.subtask_keys
        labels = set(baseline[k1].values())
        assert len(labels) == 1
        if labels == {"NO"}:
            assert set(baseline[k2].values()) == {"NO"}
        else:
            assert "NO" not in set(baseline[k2].values())
