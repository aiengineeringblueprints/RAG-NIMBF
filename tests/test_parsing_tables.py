"""Markdown table → HTML conversion for TEDS scoring (OCR-04)."""

from __future__ import annotations

import pytest

from benchmark.parsing.tables import (
    extract_markdown_tables,
    markdown_tables_to_html,
)


def test_pipe_table_converts_to_html():
    md = (
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |"
    )
    htmls = markdown_tables_to_html(md)
    assert len(htmls) == 1
    html = htmls[0]
    assert "<table>" in html
    assert "<th>A</th>" in html and "<th>B</th>" in html
    assert "<td>1</td>" in html and "<td>4</td>" in html


def test_alignment_row_is_stripped():
    md = (
        "| Left | Right |\n"
        "|:-----|------:|\n"
        "| a    | b     |"
    )
    html = markdown_tables_to_html(md)[0]
    assert "---" not in html
    assert "<td>a</td>" in html and "<td>b</td>" in html


def test_html_table_passes_through_normalized():
    md = "<table><tr><th>H</th></tr><tr><td>v</td></tr></table>"
    htmls = markdown_tables_to_html(md)
    assert len(htmls) == 1
    assert "<table>" in htmls[0] and "<td>v</td>" in htmls[0]


def test_document_without_tables_returns_empty():
    assert markdown_tables_to_html("# Title\n\nJust text.\n\nNo tables.") == []
    assert markdown_tables_to_html("") == []
    assert markdown_tables_to_html(None) == []


def test_multiple_tables_extracted_in_order():
    md = (
        "text before\n\n"
        "| A |\n|---|\n| 1 |\n\n"
        "text between\n\n"
        "<table><tr><td>x</td></tr></table>\n\n"
        "text after"
    )
    htmls = markdown_tables_to_html(md)
    assert len(htmls) == 2
    assert "<td>1</td>" in htmls[0]
    assert "<td>x</td>" in htmls[1]


def test_extract_markdown_tables_preserves_source():
    md = "| A |\n|---|\n| 1 |"
    tables = extract_markdown_tables(md)
    assert tables == [md]


def test_html_table_uses_thead_tbody_structure():
    md = "<table><tr><th>H</th></tr><tr><td>v</td></tr></table>"
    html = markdown_tables_to_html(md)[0]
    assert "<thead>" in html and "<tbody>" in html


def test_pipe_table_without_header_separator_not_a_table():
    md = "| A | B |\n| 1 | 2 |"
    # Without a separator row this is not a valid pipe table; it must not
    # crash or mis-detect — no table is extracted.
    assert markdown_tables_to_html(md) == []
