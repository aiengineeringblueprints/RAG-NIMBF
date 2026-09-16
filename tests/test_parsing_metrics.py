"""Tests for parsing text metrics (OCR-02): CER/WER + normalized edit distance."""

from __future__ import annotations

import jiwer
import pytest

from benchmark.parsing.base import ParsedPage, ParseResult
from benchmark.parsing.metrics import (
    ParsingTextMetricsResult,
    cer,
    compute_parsing_text_metrics,
    normalize_text,
    normalized_edit_distance,
    wer,
)

# ── Pure text functions ──────────────────────────────────────────────


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  a\n\t b   c ") == "a b c"

    def test_unicode_folding(self):
        # NFKC: fullwidth -> ASCII, ligature -> expanded, then lower/strip accents
        assert normalize_text("ＨＥＬＬＯ") == "hello"
        assert normalize_text("ﬁ") == "fi"

    def test_strips_accents(self):
        assert normalize_text("Café") == "cafe"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestCer:
    def test_identical(self):
        assert cer("hello world", "hello world") == 0.0

    def test_hand_computed_single_substitution(self):
        # "cat" -> "cut": 1 edit / 3 chars
        assert cer("cut", "cat") == pytest.approx(1 / 3)

    def test_hand_computed_insertion(self):
        # "helo" vs "hello": 1 insertion / 5 chars
        assert cer("helo", "hello") == pytest.approx(1 / 5)

    def test_case_insensitive_after_normalize(self):
        assert cer("HELLO", "hello") == 0.0

    def test_whitespace_insensitive(self):
        assert cer("hello  world", "hello\nworld") == 0.0

    def test_both_blank(self):
        assert cer("", "") == 0.0
        assert cer("   ", "  \n ") == 0.0

    def test_prediction_blank_ground_truth_not(self):
        assert cer("", "abc") == 1.0

    def test_ground_truth_blank_prediction_not(self):
        assert cer("abc", "") == 1.0

    def test_matches_jiwer_oracle(self):
        pred = "the quick brown fox"
        gt = "the quack brown foxx"
        assert cer(pred, gt) == pytest.approx(
            jiwer.cer(normalize_text(gt), normalize_text(pred))
        )


class TestWer:
    def test_identical(self):
        assert wer("hello world", "hello world") == 0.0

    def test_hand_computed_one_substitution(self):
        # 1 wrong word / 4 words
        assert wer("the quick brown cow", "the quick brown fox") == pytest.approx(0.25)

    def test_hand_computed_extra_word(self):
        # jiwer aligns "big" -> "very" as a substitution: 1 edit over 4 aligned words
        assert wer("hello big world", "hello very big world") == pytest.approx(0.25)

    def test_case_insensitive(self):
        assert wer("HELLO WORLD", "hello world") == 0.0

    def test_both_blank(self):
        assert wer("", "") == 0.0

    def test_prediction_blank(self):
        assert wer("", "a b") == 1.0

    def test_matches_jiwer_oracle(self):
        pred = "a b c d"
        gt = "a x c"
        assert wer(pred, gt) == pytest.approx(jiwer.wer(gt, pred))


class TestNormalizedEditDistance:
    def test_identical(self):
        assert normalized_edit_distance("abc", "abc") == 0.0

    def test_hand_computed_one_substitution(self):
        # Levenshtein("abc", "abd") = 1, normalized by max(len)=3
        assert normalized_edit_distance("abc", "abd") == pytest.approx(1 / 3)

    def test_completely_different(self):
        assert normalized_edit_distance("abc", "xyz") == 1.0

    def test_both_blank(self):
        assert normalized_edit_distance("", "") == 0.0

    def test_prediction_blank(self):
        assert normalized_edit_distance("", "abc") == 1.0

    def test_unicode_folding_before_distance(self):
        assert normalized_edit_distance("ＡＢＣ", "abc") == 0.0


# ── Aggregation over ParseResult ─────────────────────────────────────


def _result(*page_markdowns: str) -> ParseResult:
    return ParseResult(
        document_id="doc1",
        pages=tuple(
            ParsedPage(page_number=i + 1, markdown=md)
            for i, md in enumerate(page_markdowns)
        ),
        parser_name="stub",
    )


class TestComputeParsingTextMetrics:
    def test_per_page_scores_page_number_aligned(self):
        result = _result("hello world", "helo world")
        gt = ["hello world", "hello world"]
        metrics = compute_parsing_text_metrics(result, gt)
        assert [p.page_number for p in metrics.per_page] == [1, 2]
        assert metrics.per_page[0].cer == 0.0
        assert metrics.per_page[1].cer == pytest.approx(
            jiwer.cer("hello world", "helo world")
        )
        assert metrics.per_page[0].wer == 0.0
        assert metrics.per_page[1].wer == pytest.approx(1 / 2)

    def test_document_aggregation_is_mean_of_pages(self):
        result = _result("abc", "abd")
        gt = ["abc", "abc"]
        metrics = compute_parsing_text_metrics(result, gt)
        # page1: 0.0, page2: 1/3
        assert metrics.metric_means["cer"] == pytest.approx((0.0 + 1 / 3) / 2)
        assert metrics.metric_means["wer"] == pytest.approx(0.5)
        assert metrics.metric_means["normalized_edit_distance"] == pytest.approx(
            (0.0 + 1 / 3) / 2
        )

    def test_ground_truth_dict_keyed_by_page_number(self):
        result = _result("abc")
        gt = {1: "abc", 2: "missing page"}
        metrics = compute_parsing_text_metrics(result, gt)
        # page 2 has no predicted page -> full error for that page
        assert [p.page_number for p in metrics.per_page] == [1, 2]
        assert metrics.per_page[1].cer == 1.0

    def test_blank_pages_score_zero(self):
        result = _result("", "content")
        gt = ["", "content"]
        metrics = compute_parsing_text_metrics(result, gt)
        assert metrics.per_page[0].cer == 0.0
        assert metrics.per_page[0].wer == 0.0

    def test_normalization_applied_before_distance(self):
        result = _result("  HELLO   World ", "hello world")
        metrics = compute_parsing_text_metrics(result, ["hello world"])
        assert metrics.per_page[0].cer == 0.0
        assert metrics.per_page[0].normalized_edit_distance == 0.0

    def test_result_type_and_valid_counts(self):
        result = _result("abc")
        metrics = compute_parsing_text_metrics(result, ["abc"])
        assert isinstance(metrics, ParsingTextMetricsResult)
        assert metrics.pages_with_valid_scores == {
            "cer": 1,
            "wer": 1,
            "normalized_edit_distance": 1,
        }

    def test_markdown_joined_document_level_view_matches_per_page_mean(self):
        pages = ["the quick brown fox", "jumps over the dog"]
        gt = ["the quick brown fox", "jumps over the cat"]
        metrics = compute_parsing_text_metrics(_result(*pages), gt)
        # page1: 0.0; page2: jiwer scores 1 substitution over 4 aligned words
        assert metrics.metric_means["wer"] == pytest.approx(0.125)
