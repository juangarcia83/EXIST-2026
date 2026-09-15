"""Scoring helpers: hierarchy, gold construction and report parsing."""

from __future__ import annotations

import numpy as np
import pytest

from exist2026.evaluation.pyevall import (
    certain_no,
    hard_gold,
    hierarchy_for,
    parse_report,
    soft_gold,
    soft_value,
)
from exist2026.evaluation.reports import (
    compare,
    confusion,
    prediction_distribution,
    save_errors,
)
from exist2026.taxonomy import SEXISM_CATEGORIES


class TestHierarchy:
    def test_identification_is_flat(self):
        assert hierarchy_for("t21") is None

    def test_intention_hangs_off_yes(self):
        assert hierarchy_for("t22") == {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}

    def test_categorization_hangs_the_five_categories_off_yes(self):
        assert hierarchy_for("t33") == {"YES": list(SEXISM_CATEGORIES), "NO": []}


class TestSoftValues:
    def test_identification_order(self):
        assert soft_value("t21", [0.7, 0.3]) == {"YES": 0.7, "NO": 0.3}

    def test_intention_order(self):
        assert soft_value("t22", [0.1, 0.6, 0.3]) == {
            "NO": 0.1,
            "DIRECT": 0.6,
            "JUDGEMENTAL": 0.3,
        }

    def test_categorization_names_every_category(self):
        value = soft_value("t23", [0.1] + [0.2] * 5)
        assert set(value) == {"NO", *SEXISM_CATEGORIES}

    def test_certain_no_puts_all_mass_on_no(self):
        value = certain_no("t23")
        assert value["NO"] == 1.0
        assert sum(v for k, v in value.items() if k != "NO") == 0.0


class TestGold:
    def test_hard_gold_skips_undecided_instances(self, meme_corpus):
        gold = hard_gold(meme_corpus, ["100", "103"], "t21")
        assert [record["id"] for record in gold] == ["100"]

    def test_hard_gold_of_a_non_sexist_instance_is_the_no_list(self, meme_corpus):
        gold = hard_gold(meme_corpus, ["102"], "t23")
        assert gold[0]["value"] == ["NO"]

    def test_soft_gold_uses_the_named_distribution(self):
        labels = {"1": {"t21": np.array([0.8, 0.2], dtype=np.float32)}}
        gold = soft_gold(labels, ["1"], "t21")
        assert gold[0]["value"] == {"YES": pytest.approx(0.8), "NO": pytest.approx(0.2)}


class TestReportParsing:
    def test_reads_a_flat_dictionary_report(self):
        class Report:
            def __init__(self):
                self.report = {"ICM": 0.5, "ICMNorm": 0.7}

        assert parse_report(Report(), ("ICM", "ICMNorm")) == {"ICM": 0.5, "ICMNorm": 0.7}

    def test_reads_a_nested_report(self):
        class Report:
            def __init__(self):
                self.report = {"results": [{"metrics": {"ICMSoftNorm": 0.42}}]}

        assert parse_report(Report(), ("ICMSoftNorm",)) == {"ICMSoftNorm": 0.42}

    def test_averages_per_test_case_values(self):
        class Report:
            def __init__(self):
                self.report = {"ICM": {"case_a": 0.2, "case_b": 0.4}}

        assert parse_report(Report(), ("ICM",))["ICM"] == pytest.approx(0.3)

    def test_falls_back_to_the_printed_report(self):
        class Report:
            def print_report(self):
                print("ICM: 0.1234\nICMNorm = 0.5678")

        parsed = parse_report(Report(), ("ICM", "ICMNorm"))
        assert parsed == {"ICM": pytest.approx(0.1234), "ICMNorm": pytest.approx(0.5678)}

    def test_a_missing_metric_is_simply_absent(self):
        class Report:
            def __init__(self):
                self.report = {"ICM": 0.5}

        assert "FMeasure" not in parse_report(Report(), ("ICM", "FMeasure"))

    def test_no_report_yields_nothing(self):
        assert parse_report(None, ("ICM",)) == {}


class TestComparisons:
    def test_identification_keeps_both_classes(self, meme_corpus):
        result = compare({"100": "YES", "102": "NO", "103": "YES"}, meme_corpus, "t21")
        assert result.metrics["n"] == 2  # 103 is undecided
        assert result.metrics["accuracy"] == 1.0

    def test_intention_drops_non_sexist_gold(self, meme_corpus):
        # 102's gold is NO: its x.2 prediction was propagated by the cascade,
        # not chosen by the model, so scoring it would measure the hierarchy.
        result = compare({"100": "DIRECT", "102": "NO"}, meme_corpus, "t22")
        assert set(result.frame["id"]) == {"100"}

    def test_a_predicted_no_on_sexist_gold_counts_as_an_error(self, meme_corpus):
        result = compare({"100": "NO"}, meme_corpus, "t22")
        assert result.metrics["accuracy"] == 0.0
        assert result.metrics["n_predicted_no_on_sexist_gold"] == 1
        assert "counted as errors" in result.note

    def test_categorization_reports_per_class_scores(self, meme_corpus):
        result = compare({"100": ["STEREOTYPING-DOMINANCE"], "102": ["NO"]}, meme_corpus, "t23")
        assert result.metrics["n"] == 1
        assert result.metrics["exact_match"] == 1.0
        assert result.metrics["per_class"]["STEREOTYPING-DOMINANCE"]["support"] == 1

    def test_confusion_matrix_is_labelled(self, meme_corpus):
        result = compare({"100": "YES", "102": "NO"}, meme_corpus, "t21")
        matrix = confusion(result, ["NO", "YES"])
        assert list(matrix.index) == ["gold_NO", "gold_YES"]
        assert matrix.loc["gold_YES", "pred_YES"] == 1

    def test_errors_are_written_with_their_text(self, meme_corpus, tmp_path):
        result = compare({"100": "NO"}, meme_corpus, "t21")
        path = save_errors(result, meme_corpus, tmp_path)
        assert path is not None
        assert "100" in path.read_text()

    def test_nothing_is_written_when_there_are_no_errors(self, meme_corpus, tmp_path):
        result = compare({"100": "YES"}, meme_corpus, "t21")
        assert save_errors(result, meme_corpus, tmp_path) is None


class TestDistribution:
    def test_counts_single_labels(self):
        frame = prediction_distribution({"1": "YES", "2": "YES", "3": "NO"})
        assert frame.set_index("label").loc["YES", "n"] == 2
        assert frame.set_index("label").loc["YES", "pct"] == pytest.approx(66.67, abs=0.01)

    def test_joins_multi_label_predictions(self):
        frame = prediction_distribution({"1": ["OBJECTIFICATION", "SEXUAL-VIOLENCE"], "2": ["NO"]})
        assert "OBJECTIFICATION+SEXUAL-VIOLENCE" in set(frame["label"])
        assert "NO" in set(frame["label"])
