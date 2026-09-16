"""Tests for GT↔prediction alignment (OCR-03): vendored OmniDocBench quick_match."""

from __future__ import annotations

import pytest

from benchmark.parsing.alignment import (
    MATCH_ALGORITHM_LICENSE,
    MATCH_ALGORITHM_NAME,
    MATCH_ALGORITHM_VERSION,
    alignment_run_metadata,
    quick_match,
    split_paragraphs,
)
from benchmark.parsing.base import ParsedPage, ParseResult
from benchmark.parsing.metrics import (
    compute_aligned_parsing_text_metrics,
    compute_parsing_text_metrics,
)

# ── Paragraph segmentation ───────────────────────────────────────────


class TestSplitParagraphs:
    def test_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph."
        assert split_paragraphs(text) == ["First paragraph.", "Second paragraph."]

    def test_single_newlines_are_soft_wraps_within_a_paragraph(self):
        text = "One wrapped\nparagraph.\n\nAnother."
        assert split_paragraphs(text) == ["One wrapped\nparagraph.", "Another."]

    def test_single_block_with_single_newlines_falls_back_to_lines(self):
        text = "Line one.\nLine two."
        assert split_paragraphs(text) == ["Line one.", "Line two."]

    def test_strips_and_drops_empty(self):
        assert split_paragraphs("  \n\n  a  \n\n  ") == ["a"]

    def test_empty_text(self):
        assert split_paragraphs("") == []
        assert split_paragraphs(None) == []


# ── Matching metadata ────────────────────────────────────────────────


class TestMatchingMetadata:
    def test_algorithm_name_is_quick_match(self):
        assert MATCH_ALGORITHM_NAME == "quick_match"

    def test_version_pins_upstream(self):
        assert "OmniDocBench" in MATCH_ALGORITHM_VERSION
        assert MATCH_ALGORITHM_LICENSE == "Apache-2.0"

    def test_run_metadata_dict(self):
        meta = alignment_run_metadata()
        assert meta["algorithm"] == MATCH_ALGORITHM_NAME
        assert meta["version"] == MATCH_ALGORITHM_VERSION
        assert meta["license"] == MATCH_ALGORITHM_LICENSE
        assert meta["source"].startswith("https://github.com/opendatalab/OmniDocBench")


# ── quick_match alignment ────────────────────────────────────────────


class TestQuickMatchEdgeCases:
    def test_exact_single_pair_has_zero_edit(self):
        pairs = quick_match(["hello world"], ["hello world"])
        assert len(pairs) == 1
        assert pairs[0].edit == pytest.approx(0.0)
        assert pairs[0].matched is True

    def test_single_pair_reports_normalized_edit_distance(self):
        pairs = quick_match(["cat"], ["cut"])
        assert pairs[0].edit == pytest.approx(1 / 3)

    def test_no_predictions_leaves_all_gt_unmatched(self):
        pairs = quick_match(["alpha", "beta"], [])
        assert [p.matched for p in pairs] == [False, False]
        assert all(p.pred == "" for p in pairs)
        assert all(p.edit == pytest.approx(1.0) for p in pairs)

    def test_no_ground_truth_leaves_all_predictions_unmatched(self):
        pairs = quick_match([], ["alpha", "beta"])
        assert [p.matched for p in pairs] == [False, False]
        assert all(p.gt == "" for p in pairs)

    def test_both_empty(self):
        assert quick_match([], []) == []

    def test_matching_is_case_and_whitespace_insensitive(self):
        pairs = quick_match(["Hello  World"], ["hello\nworld"])
        assert pairs[0].edit == pytest.approx(0.0)


class TestQuickMatchTruncationAndMerge:
    def test_split_paragraph_is_merged_back(self):
        # GT is one paragraph; the prediction split it into two adjacent chunks.
        gt = ["The quick brown fox jumps over the lazy dog"]
        pred = ["The quick brown fox", "jumps over the lazy dog"]
        pairs = quick_match(gt, pred)
        matched = [p for p in pairs if p.matched]
        assert len(matched) == 1
        assert matched[0].edit == pytest.approx(0.0)
        # Both prediction lines were merged into the single matched pair.
        assert sorted(matched[0].pred_indices) == [0, 1]

    def test_truncated_prediction_is_matched(self):
        # Prediction truncates the GT paragraph mid-sentence.
        gt = ["The quick brown fox jumps over the lazy dog"]
        pred = ["The quick brown fox jumps"]
        pairs = quick_match(gt, pred)
        assert len(pairs) == 1
        assert pairs[0].matched is True
        assert pairs[0].edit < 0.7

    def test_merged_paragraph_is_split_back(self):
        # GT has two paragraphs; the prediction emitted them as one merged
        # blob. Upstream semantics: the blob is assigned to the best GT and
        # the remaining GT paragraph falls back to unmatched.
        gt = ["Alpha beta gamma delta", "Epsilon zeta eta theta"]
        pred = ["Alpha beta gamma delta Epsilon zeta eta theta"]
        pairs = quick_match(gt, pred)
        matched = [p for p in pairs if p.matched]
        assert len(matched) == 1
        assert matched[0].edit < 0.7
        assert matched[0].gt_indices == (0,)
        assert pairs[-1].gt_indices == (1,)
        assert pairs[-1].matched is False

    def test_fixture_document_split_merge_and_missing(self):
        # Worked example over a fixture-like page: prediction splits para 1,
        # merges paras 2+3, and drops para 4 entirely.
        gt = [
            "Headers and footers are stripped from this page",
            "The second paragraph discusses retrieval quality",
            "and continues with an analysis of chunking",
            "A final paragraph about evaluation metrics",
        ]
        pred = [
            "Headers and footers are stripped",
            "from this page",
            (
                "The second paragraph discusses retrieval quality and "
                "continues with an analysis of chunking"
            ),
        ]
        pairs = quick_match(gt, pred)
        matched = {p.gt_indices[0]: p for p in pairs if p.gt_indices}
        # Para 1 recovered from the split pair; the merged blob is assigned
        # to para 2; para 3 (absorbed into the blob but already consumed)
        # and para 4 (missing) count as full errors, matching upstream.
        assert matched[0].matched and matched[0].edit == pytest.approx(0.0)
        assert matched[1].matched and matched[1].edit < 0.7
        assert matched[2].matched is False
        assert matched[3].matched is False
        assert matched[3].edit == pytest.approx(1.0)

    def test_dissimilar_lines_do_not_match(self):
        gt = ["completely unrelated ground truth text"]
        pred = ["something entirely different here"]
        pairs = quick_match(gt, pred)
        assert pairs[0].matched is False
        assert pairs[0].edit > 0.7

    def test_mixed_scene_alignment(self):
        gt = [
            "Introduction paragraph with common words",
            "Conclusion paragraph ends the document here",
        ]
        pred = [
            "Introduction paragraph with common words",
            "An extra hallucinated paragraph",
            "Conclusion paragraph ends the document here",
        ]
        pairs = quick_match(gt, pred)
        matched_pairs = sorted(
            (p for p in pairs if p.matched), key=lambda p: p.gt_indices[0]
        )
        assert len(matched_pairs) == 2
        assert matched_pairs[0].edit == pytest.approx(0.0)
        assert matched_pairs[1].edit == pytest.approx(0.0)
        # The hallucinated paragraph shows up as an unmatched pair.
        unmatched_preds = [p for p in pairs if not p.matched and p.gt == ""]
        assert len(unmatched_preds) == 1

    def test_missing_prediction_for_one_of_two_paragraphs(self):
        gt = ["First paragraph text", "Second paragraph text"]
        pred = ["First paragraph text"]
        pairs = quick_match(gt, pred)
        by_gt = {p.gt_indices[0]: p for p in pairs if p.gt_indices}
        assert by_gt[0].matched is True
        assert by_gt[1].matched is False
        assert by_gt[1].edit == pytest.approx(1.0)


# ── Aligned text metrics ─────────────────────────────────────────────


def _result(*pages: str) -> ParseResult:
    return ParseResult(
        document_id="doc",
        pages=tuple(
            ParsedPage(page_number=i + 1, markdown=text)
            for i, text in enumerate(pages)
        ),
        parser_name="test",
    )


class TestComputeAlignedParsingTextMetrics:
    def test_aligned_pair_scores_beat_whole_page_scores(self):
        # Whole-page string comparison penalizes paragraph reordering as
        # character edits; alignment matches each paragraph exactly.
        gt = ["Alpha beta gamma delta", "Delta epsilon zeta eta"]
        pred = ["Delta epsilon zeta eta", "Alpha beta gamma delta"]
        whole = compute_parsing_text_metrics(_result("\n\n".join(pred)), gt)
        aligned = compute_aligned_parsing_text_metrics(
            _result("\n\n".join(pred)), gt
        )
        assert whole.per_page[0].cer > 0.0
        assert aligned.per_page[0].cer == pytest.approx(0.0)

    def test_unmatched_gt_counts_as_full_error(self):
        gt = ["First paragraph text", "Second paragraph text"]
        aligned = compute_aligned_parsing_text_metrics(_result(""), gt)
        assert aligned.per_page[0].cer == pytest.approx(1.0)
        assert aligned.per_page[0].wer == pytest.approx(1.0)

    def test_missing_prediction_page(self):
        gt = ["hello world"]
        aligned = compute_aligned_parsing_text_metrics(_result(), gt)
        assert aligned.per_page[0].cer == pytest.approx(1.0)

    def test_alignment_metadata_recorded(self):
        aligned = compute_aligned_parsing_text_metrics(_result("hello"), ["hello"])
        assert aligned.alignment["algorithm"] == MATCH_ALGORITHM_NAME
        assert aligned.alignment["version"] == MATCH_ALGORITHM_VERSION
        assert "matched_pairs" in aligned.alignment
        assert "unmatched_pairs" in aligned.alignment

    def test_format_alignment_summary(self):
        from benchmark.parsing.alignment import format_alignment_summary

        summary = format_alignment_summary(alignment_run_metadata() | {"matched_pairs": 3, "unmatched_pairs": 1})
        assert "quick_match" in summary
        assert "OmniDocBench" in summary
        assert "3 matched" in summary
        assert "1 unmatched pairs" in summary
