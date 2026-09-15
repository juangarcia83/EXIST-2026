"""Prompts: files present, contract honoured, messages correctly shaped."""

from __future__ import annotations

import pytest

from exist2026.fewshot.prompts import (
    MediaSpec,
    PromptBuilder,
    builders_for,
    prompt_path,
    questions,
    system_prompt,
)
from exist2026.taxonomy import SEXISM_CATEGORIES, Modality, Subtask, subtask_key

ALL = [(modality, subtask) for modality in Modality for subtask in Subtask]


@pytest.mark.parametrize("modality,subtask", ALL)
def test_every_subtask_has_a_published_prompt(modality, subtask):
    assert prompt_path(modality, subtask).exists()
    assert len(system_prompt(modality, subtask)) > 500


@pytest.mark.parametrize("modality,subtask", ALL)
def test_prompts_state_an_output_contract(modality, subtask):
    """Each prompt ends with an explicit, parseable output section."""
    text = system_prompt(modality, subtask)
    assert "# OUTPUT" in text
    contract = text.split("# OUTPUT", 1)[1].upper()
    assert "EXACT" in contract  # "EXACTLY one token" / "the EXACT label names"


@pytest.mark.parametrize("modality", list(Modality))
def test_downstream_prompts_say_the_instance_is_already_sexist(modality):
    """x.2 and x.3 only ever receive instances the cascade already gated."""
    for subtask in (Subtask.INTENTION, Subtask.CATEGORIZATION):
        opening = system_prompt(modality, subtask)[:600].lower()
        assert "already" in opening and "sexist" in opening


@pytest.mark.parametrize("modality", list(Modality))
def test_the_intention_contract_offers_only_the_two_sexist_labels(modality):
    contract = system_prompt(modality, Subtask.INTENTION).split("# OUTPUT", 1)[1]
    assert "DIRECT" in contract and "JUDGEMENTAL" in contract


@pytest.mark.parametrize("modality", list(Modality))
def test_the_categorization_prompt_lists_the_five_categories(modality):
    text = system_prompt(modality, Subtask.CATEGORIZATION)
    for category in SEXISM_CATEGORIES:
        assert category in text


@pytest.mark.parametrize("modality", list(Modality))
def test_questions_cover_every_subtask(modality):
    block = questions(modality)
    for subtask in Subtask:
        assert subtask_key(modality, subtask) in block
    assert "{text}" in block["with_text"]


class TestMessageStructure:
    @pytest.fixture
    def builder(self):
        return builders_for(Modality.MEMES)["t21"]

    def test_alternates_user_and_assistant_after_the_system_turn(self, builder):
        pool = [{"id": "1", "label": "YES"}, {"id": "2", "label": "NO"}]
        messages = builder.build("/q.jpeg", "query", pool, lambda i: f"/{i}.jpeg", lambda i: "text")
        assert [m["role"] for m in messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]

    def test_the_query_turn_has_no_answer(self, builder):
        messages = builder.build("/q.jpeg", "query", [], lambda i: i, lambda i: "")
        assert messages[-1]["role"] == "user"
        assert "query" in messages[-1]["content"][1]["text"]

    def test_media_precedes_text(self, builder):
        turn = builder.user_turn("/q.jpeg", "hello")
        assert turn["content"][0]["type"] == "image"
        assert turn["content"][1]["type"] == "text"

    def test_an_instance_without_text_says_so(self, builder):
        turn = builder.user_turn("/q.jpeg", "   ")
        assert "no extracted text" in turn["content"][1]["text"]


class TestMediaSpec:
    def test_memes_are_images(self):
        assert MediaSpec.for_modality(Modality.MEMES).block("/a.jpeg") == {
            "type": "image",
            "image": "/a.jpeg",
        }

    def test_videos_carry_sampling_hints(self):
        block = MediaSpec.for_modality(Modality.VIDEOS, fps=0.5, max_pixels=1000).block("/a.mp4")
        assert block == {"type": "video", "video": "/a.mp4", "fps": 0.5, "max_pixels": 1000}

    def test_sampling_hints_are_optional(self):
        assert MediaSpec.for_modality(Modality.VIDEOS).block("/a.mp4") == {
            "type": "video",
            "video": "/a.mp4",
        }


class TestAnswerFormatting:
    def test_categorization_answers_are_comma_separated(self):
        builder = builders_for(Modality.MEMES)["t23"]
        assert builder.format_answer({"labels": ["OBJECTIFICATION", "SEXUAL-VIOLENCE"]}) == (
            "OBJECTIFICATION, SEXUAL-VIOLENCE"
        )

    def test_rationales_are_off_by_default(self):
        assert builders_for(Modality.VIDEOS)["t31"].format_answer({"label": "YES"}) == "YES"

    def test_rationales_name_the_rule_they_illustrate(self):
        builder = builders_for(Modality.VIDEOS, use_rationale=True)["t32"]
        answer = builder.format_answer({"label": "DIRECT"})
        assert answer.startswith("DIRECT - D4:")

    def test_asking_for_rationales_without_a_file_fails_loudly(self, tmp_path):
        (tmp_path / "memes").mkdir()
        for subtask in Subtask:
            path = prompt_path(Modality.MEMES, subtask, tmp_path)
            path.write_text(system_prompt(Modality.MEMES, subtask), encoding="utf-8")
        (tmp_path / "memes" / "questions.yaml").write_text(
            (prompt_path(Modality.MEMES, Subtask.IDENTIFICATION).parent / "questions.yaml").read_text(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"rationales\.yaml"):
            PromptBuilder(
                Modality.MEMES,
                Subtask.IDENTIFICATION,
                MediaSpec.for_modality(Modality.MEMES),
                use_rationale=True,
                prompt_root=tmp_path,
            )
