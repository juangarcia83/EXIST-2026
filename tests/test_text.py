"""The enriched `capsanocr` text view."""

from __future__ import annotations

import json

import pytest

from exist2026.fusion.text import SEP, TextView, build_text, load_vlm_enrichment
from exist2026.taxonomy import Modality

MEME = {"text": "get back in the kitchen"}
MEME_ENRICHMENT = {
    "visual_description": "a person standing in a kitchen",
    "sexism_analysis": "assigns a domestic role on the basis of gender",
}

VIDEO = {"text": "on-screen caption"}
VIDEO_ENRICHMENT = {
    "qwen_parsed": {
        "sexism_analysis": "mocks women drivers",
        "visual_description": "a car and a person",
        "ocr_text": "caption from the frame",
    },
    "asr_transcript": "spoken words",
}


class TestMemeView:
    def test_joins_caption_analysis_and_ocr(self):
        text = build_text(MEME, MEME_ENRICHMENT, Modality.MEMES)
        assert text == SEP.join(
            [
                MEME_ENRICHMENT["visual_description"],
                MEME_ENRICHMENT["sexism_analysis"],
                MEME["text"],
            ]
        )

    def test_accepts_the_legacy_analysis_field_name(self):
        text = build_text(MEME, {"social_analysis": "an analysis"}, Modality.MEMES)
        assert "an analysis" in text

    def test_falls_back_to_ocr_without_enrichment(self):
        assert build_text(MEME, None, Modality.MEMES) == MEME["text"]

    def test_empty_parts_are_dropped_rather_than_left_as_separators(self):
        text = build_text(MEME, {"visual_description": "", "sexism_analysis": ""}, Modality.MEMES)
        assert not text.startswith(SEP)
        assert text == MEME["text"]

    def test_an_instance_with_nothing_yields_an_empty_string(self):
        assert build_text({}, None, Modality.MEMES) == ""


class TestVideoView:
    def test_uses_explicit_section_markers(self):
        text = build_text(VIDEO, VIDEO_ENRICHMENT, Modality.VIDEOS)
        assert text.startswith("[ANALYSIS] mocks women drivers")
        for marker in ("[DESC]", "[OCR]", "[ASR]"):
            assert marker in text

    def test_prefers_the_vlm_ocr_over_the_released_text(self):
        text = build_text(VIDEO, VIDEO_ENRICHMENT, Modality.VIDEOS)
        assert "caption from the frame" in text
        assert "on-screen caption" not in text

    def test_falls_back_to_the_released_text_when_the_vlm_found_none(self):
        enrichment = {"qwen_parsed": {"ocr_text": ""}}
        assert "on-screen caption" in build_text(VIDEO, enrichment, Modality.VIDEOS)

    def test_survives_a_malformed_parsed_block(self):
        text = build_text(VIDEO, {"qwen_parsed": "not a mapping"}, Modality.VIDEOS)
        assert "[ANALYSIS]" in text
        assert "on-screen caption" in text

    def test_keeps_every_marker_even_when_a_part_is_missing(self):
        # Section markers are positional cues for the encoder, so they stay put
        # rather than silently shifting when a source is absent.
        text = build_text(VIDEO, None, Modality.VIDEOS)
        assert text.count("[") == 4


class TestEnrichmentLoading:
    def test_reads_a_mapping_file(self, tmp_path):
        path = tmp_path / "vlm.json"
        path.write_text(json.dumps({"1": {"visual_description": "x"}}), encoding="utf-8")
        assert load_vlm_enrichment(path)["1"]["visual_description"] == "x"

    def test_reads_a_list_file_keyed_by_id_exist(self, tmp_path):
        path = tmp_path / "vlm.json"
        path.write_text(json.dumps([{"id_EXIST": 7, "visual_description": "x"}]), encoding="utf-8")
        assert "7" in load_vlm_enrichment(path)

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert load_vlm_enrichment(tmp_path / "absent.json") == {}
        assert load_vlm_enrichment(None) == {}


def test_text_view_binds_records_and_enrichment():
    view = TextView({"1": MEME}, {"1": MEME_ENRICHMENT}, Modality.MEMES)
    assert view("1") == build_text(MEME, MEME_ENRICHMENT, Modality.MEMES)


def test_text_view_raises_on_an_unknown_instance():
    view = TextView({}, {}, Modality.MEMES)
    with pytest.raises(KeyError):
        view("nope")
