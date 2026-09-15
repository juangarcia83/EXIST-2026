"""Response parsing, and the fallback bookkeeping that keeps it honest."""

from __future__ import annotations

import pytest

from exist2026.fewshot.parsing import ParserPolicy, ResponseParser, strip_thinking
from exist2026.taxonomy import Modality, Subtask


@pytest.fixture
def meme_parser(tmp_path):
    return ResponseParser(ParserPolicy.for_modality(Modality.MEMES), tmp_path / "fallbacks.log")


@pytest.fixture
def video_parser(tmp_path):
    policy = ParserPolicy.for_modality(Modality.VIDEOS, "OBJECTIFICATION")
    return ResponseParser(policy, tmp_path / "fallbacks.log")


class TestStripThinking:
    def test_removes_a_complete_block(self):
        assert strip_thinking("<think>reasoning</think>YES") == "YES"

    def test_drops_everything_before_an_unmatched_closing_tag(self):
        # Chat templates often open <think> implicitly, so only the close appears.
        assert strip_thinking("the user wants...</think>NO") == "NO"

    def test_is_a_no_op_without_thinking(self):
        assert strip_thinking("YES") == "YES"


class TestIdentification:
    @pytest.mark.parametrize(
        "response,expected",
        [
            ("YES", "YES"),
            ("no", "NO"),
            ("Yes.", "YES"),
            ("Answer: NO", "NO"),
            ("The answer is YES", "YES"),
            ("**YES**", "YES"),
            ("<think>hmm</think>NO", "NO"),
            ("Y", "YES"),
            ("1", "YES"),
            ("0", "NO"),
        ],
    )
    def test_accepts_common_variants(self, meme_parser, response, expected):
        assert meme_parser.parse(response, Subtask.IDENTIFICATION, "t21") == expected

    def test_both_labels_present_falls_back_to_the_first_token(self, meme_parser):
        assert meme_parser.parse("yes or no?", Subtask.IDENTIFICATION, "t21") == "YES"
        assert meme_parser.fallbacks["t21"] == 0

    def test_an_answer_committing_to_neither_label_falls_back(self, meme_parser):
        assert meme_parser.parse("maybe yes, maybe no", Subtask.IDENTIFICATION, "t21") == "NO"
        assert meme_parser.fallbacks["t21"] == 1

    def test_empty_output_falls_back(self, meme_parser):
        assert meme_parser.parse("", Subtask.IDENTIFICATION, "t21") == "NO"
        assert meme_parser.fallbacks["t21"] == 1


class TestIntention:
    def test_reads_the_label_out_of_a_sentence(self, meme_parser):
        assert meme_parser.parse("DIRECT.", Subtask.INTENTION, "t22") == "DIRECT"

    @pytest.mark.parametrize("spelling", ["JUDGMENTAL", "Judgemental", "judgment"])
    def test_accepts_spelling_variants(self, meme_parser, spelling):
        assert meme_parser.parse(spelling, Subtask.INTENTION, "t22") == "JUDGEMENTAL"

    def test_memes_fall_back_to_no(self, meme_parser):
        assert meme_parser.parse("unclear", Subtask.INTENTION, "t22") == "NO"

    def test_videos_never_emit_no(self, video_parser):
        # Anything reaching x.2 was already judged sexist, so NO would
        # contradict the cascade; DIRECT is the modal class instead.
        assert video_parser.parse("NO", Subtask.INTENTION, "t32") == "DIRECT"
        assert video_parser.parse("unclear", Subtask.INTENTION, "t32") == "DIRECT"


class TestCategorization:
    def test_parses_a_comma_separated_list(self, meme_parser):
        assert meme_parser.parse("OBJECTIFICATION, SEXUAL-VIOLENCE", Subtask.CATEGORIZATION, "t23") == [
            "OBJECTIFICATION",
            "SEXUAL-VIOLENCE",
        ]

    def test_accepts_truncated_and_underscored_names(self, meme_parser):
        assert meme_parser.parse("IDEOLOGICAL", Subtask.CATEGORIZATION, "t23") == ["IDEOLOGICAL-INEQUALITY"]
        assert meme_parser.parse("STEREOTYPING_DOMINANCE", Subtask.CATEGORIZATION, "t23") == [
            "STEREOTYPING-DOMINANCE"
        ]

    def test_deduplicates_while_keeping_order(self, meme_parser):
        assert meme_parser.parse("OBJECTIFICATION. OBJECTIFICATION", Subtask.CATEGORIZATION, "t23") == [
            "OBJECTIFICATION"
        ]

    def test_a_stray_no_beside_real_categories_is_ignored(self, meme_parser):
        assert meme_parser.parse("NO, OBJECTIFICATION", Subtask.CATEGORIZATION, "t23") == ["OBJECTIFICATION"]
        assert meme_parser.fallbacks["t23"] == 0

    def test_memes_may_answer_no(self, meme_parser):
        assert meme_parser.parse("NO", Subtask.CATEGORIZATION, "t23") == ["NO"]

    def test_videos_fall_back_to_the_configured_category(self, video_parser):
        assert video_parser.parse("NO", Subtask.CATEGORIZATION, "t33") == ["OBJECTIFICATION"]
        assert video_parser.parse("???", Subtask.CATEGORIZATION, "t33") == ["OBJECTIFICATION"]


class TestBookkeeping:
    def test_report_separates_parser_misses_from_exceptions(self, video_parser):
        video_parser.parse("YES", Subtask.IDENTIFICATION, "t31")
        video_parser.parse("???", Subtask.IDENTIFICATION, "t31")
        video_parser.total["t31"] += 1
        video_parser.record_exception("t31", "9", RuntimeError("out of memory"), "NO")

        report = video_parser.report(["t31"]).set_index("subtask").loc["t31"]
        assert report["n_predictions"] == 3
        assert report["n_fallback"] == 2
        assert report["n_exception"] == 1
        assert report["n_parser_miss"] == 1
        assert video_parser.exception_counts()["RuntimeError"] == 1

    def test_fallback_rate_is_zero_without_predictions(self, meme_parser):
        assert meme_parser.fallback_rate("t21") == 0.0
