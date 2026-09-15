"""Filesystem helpers. Small, but every pipeline depends on them."""

from __future__ import annotations

import json

from exist2026.io_utils import append_jsonl, first_existing, read_json, read_jsonl, write_json


class TestFirstExisting:
    def test_returns_the_first_path_that_exists(self, tmp_path):
        missing = tmp_path / "a.json"
        present = tmp_path / "b.json"
        present.write_text("{}")
        assert first_existing([missing, present]) == present

    def test_returns_none_when_nothing_exists(self, tmp_path):
        assert first_existing([tmp_path / "a", tmp_path / "b"]) is None

    def test_an_empty_candidate_list_is_none(self):
        assert first_existing([]) is None


class TestJson:
    def test_round_trips(self, tmp_path):
        payload = {"a": [1, 2], "b": "ñ"}
        path = write_json(tmp_path / "nested" / "x.json", payload)
        assert read_json(path) == payload

    def test_creates_parent_directories(self, tmp_path):
        path = write_json(tmp_path / "deep" / "deeper" / "x.json", {})
        assert path.exists()

    def test_writes_utf8_without_escapes(self, tmp_path):
        path = write_json(tmp_path / "x.json", {"k": "misogínia"})
        assert "misogínia" in path.read_text(encoding="utf-8")

    def test_undecodable_bytes_do_not_abort_a_read(self, tmp_path):
        # A single corrupt byte in a released file must not lose a whole run.
        path = tmp_path / "x.json"
        path.write_bytes(b'{"k": "caf\xe9"}')
        assert "k" in read_json(path)


class TestJsonLines:
    def test_appends_one_record_per_line(self, tmp_path):
        path = tmp_path / "log.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"a": 2})
        assert [json.loads(line) for line in path.read_text().splitlines()] == [{"a": 1}, {"a": 2}]

    def test_reading_a_missing_log_yields_nothing(self, tmp_path):
        assert read_jsonl(tmp_path / "absent.jsonl") == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text('{"a": 1}\nnot json\n\n{"a": 2}\n', encoding="utf-8")
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
