"""Table metrics over aligned table regions (OCR-04)."""

from __future__ import annotations

import pytest

from benchmark.parsing.base import ParsedPage, ParseResult
from benchmark.parsing.table_metrics import compute_parsing_table_metrics

PIPE_TABLE = "| A | B |\n|---|---|\n| 1 | 2 |"
PIPE_TABLE_SAME = PIPE_TABLE
PIPE_TABLE_ALTERED = "| A | B |\n|---|---|\n| 1 | Z |"
GT_MD = f"# Page\n\n{PIPE_TABLE}\n\nSome text."
PRED_MD_PERFECT = f"# Page\n\n{PIPE_TABLE_SAME}\n\nOther text."
PRED_MD_ALTERED = f"# Page\n\n{PIPE_TABLE_ALTERED}\n\nOther text."


def _result(*page_markdowns: str) -> ParseResult:
    return ParseResult(
        document_id="doc",
        pages=tuple(
            ParsedPage(page_number=i + 1, markdown=md)
            for i, md in enumerate(page_markdowns)
        ),
        parser_name="test",
    )


def test_perfect_tables_score_one():
    result = _result(PRED_MD_PERFECT)
    out = compute_parsing_table_metrics(result, [GT_MD])
    assert out.metric_means["teds"] == pytest.approx(1.0)
    assert out.metric_means["teds_s"] == pytest.approx(1.0)
    assert out.pages_with_valid_scores["teds"] == 1
    assert out.per_page[0].page_number == 1
    assert out.per_page[0].teds == pytest.approx(1.0)


def test_missing_prediction_table_scores_zero():
    result = _result("no table here")
    out = compute_parsing_table_metrics(result, [GT_MD])
    assert out.metric_means["teds"] == pytest.approx(0.0)
    assert out.metric_means["teds_s"] == pytest.approx(0.0)


def test_altered_cell_below_one_but_structure_only_one():
    result = _result(PRED_MD_ALTERED)
    out = compute_parsing_table_metrics(result, [GT_MD])
    assert 0.0 < out.metric_means["teds"] < 1.0
    assert out.metric_means["teds_s"] == pytest.approx(1.0)


def test_document_without_tables_scores_without_error():
    result = _result("plain text, no tables")
    out = compute_parsing_table_metrics(result, ["also no tables"])
    assert out.metric_means == {}
    assert out.pages_with_valid_scores == {}
    assert out.per_table == []


def test_per_table_scores_aggregate_to_document_score():
    gt = [GT_MD, "| X |\n|---|\n| y |"]
    pred = [PRED_MD_ALTERED, "| X |\n|---|\n| y |"]
    out = compute_parsing_table_metrics(_result(*pred), gt)
    assert len(out.per_table) == 2
    teds_values = [t["teds"] for t in out.per_table]
    assert out.metric_means["teds"] == pytest.approx(sum(teds_values) / 2)


def test_extra_prediction_tables_are_ignored():
    pred = PRED_MD_PERFECT + "\n\n| extra |\n|---|\n| table |"
    out = compute_parsing_table_metrics(_result(pred), [GT_MD])
    assert len(out.per_table) == 1
    assert out.metric_means["teds"] == pytest.approx(1.0)


def test_html_gt_tables_score_against_markdown_pred():
    gt_html = (
        "# Page\n\n<table><thead><tr><th>A</th></tr></thead>"
        "<tbody><tr><td>1</td></tr></tbody></table>"
    )
    pred_md = "| A |\n|---|\n| 1 |"
    out = compute_parsing_table_metrics(_result(pred_md), [gt_html])
    assert out.metric_means["teds"] == pytest.approx(1.0)


def test_mapping_ground_truth_by_page_number():
    out = compute_parsing_table_metrics(
        _result(PRED_MD_PERFECT), {1: GT_MD}
    )
    assert out.metric_means["teds"] == pytest.approx(1.0)


def test_run_metadata_records_teds_provenance():
    out = compute_parsing_table_metrics(_result(PRED_MD_PERFECT), [GT_MD])
    assert out.metric_metadata["algorithm"] == "teds"
    assert out.metric_metadata["license"] == "Apache-2.0"
