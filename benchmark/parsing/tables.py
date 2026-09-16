"""Markdown table extraction and markdown→HTML conversion (OCR-04).

Markdown-only parsers emit tables as pipe tables (or raw HTML); TEDS needs
comparable HTML trees.  This module extracts tables from a Markdown page and
converts each into a normalized ``<html><body><table>…`` document:

- pipe tables (``| A | B |`` + ``|---|---|`` separator, optional alignment
  colons) become a ``<thead>`` header row plus a ``<tbody>`` body
- existing HTML tables are re-serialized with the same ``<thead>`` /
  ``<tbody>`` structure so both flavors are structurally comparable

Cell text is escaped, surrounding whitespace in cells is stripped, and the
separator row is dropped — cosmetic differences that TEDS must not see.
"""

from __future__ import annotations

import re
from html import escape

from lxml import html as lxml_html

__all__ = [
    "extract_markdown_tables",
    "markdown_tables_to_html",
]

# A pipe-table row: leading pipe optional, cells separated by pipes.
_PIPE_ROW = re.compile(r"^\s*\|(.*)\|\s*$")
# The separator row between header and body (colons mark alignment).
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _split_pipe_row(line: str) -> list[str] | None:
    """Split one Markdown table line into cells, or None if not a row."""
    match = _PIPE_ROW.match(line)
    if not match:
        return None
    return [cell.strip() for cell in match.group(1).split("|")]


def _pipe_table_to_html(rows: list[list[str]], has_header: bool) -> str:
    """Render split pipe-table rows as a normalized HTML table."""
    parts = ["<table>"]
    body_start = 0
    if has_header:
        parts.append("<thead><tr>")
        for cell in rows[0]:
            parts.append(f"<th>{escape(cell)}</th>")
        parts.append("</tr></thead>")
        body_start = 1
    parts.append("<tbody>")
    for row in rows[body_start:]:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _html_table_to_normalized_html(raw: str) -> str:
    """Re-serialize an HTML table with thead/tbody normalization."""
    parser = lxml_html.HTMLParser(remove_comments=True, encoding="utf-8")
    fragment = lxml_html.fromstring(raw, parser=parser)
    table = fragment if fragment.tag == "table" else fragment.find(".//table")
    if table is None:
        return ""

    def cell_tag(cell: "lxml_html.HtmlElement") -> str:
        return "th" if cell.tag == "th" else "td"

    all_rows = table.xpath("./tr | ./thead/tr | ./tbody/tr")
    thead_rows = [r for r in all_rows if r.xpath("./th")]
    body_rows = [r for r in all_rows if r not in thead_rows]

    def render_row(row, tag_hint: str) -> str:
        cells = "".join(
            f"<{cell_tag(cell)}>{escape((cell.text_content() or '').strip())}</{cell_tag(cell)}>"
            for cell in row.xpath("./td | ./th")
        )
        return f"<tr>{cells}</tr>"

    parts = ["<table>"]
    if thead_rows:
        parts.append("<thead>")
        parts.extend(render_row(r, "th") for r in thead_rows)
        parts.append("</thead>")
    if body_rows:
        parts.append("<tbody>")
        parts.extend(render_row(r, "td") for r in body_rows)
        parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def extract_markdown_tables(markdown: str | None) -> list[str]:
    """Extract tables from a Markdown page, in document order.

    Pipe tables are returned in their raw source form; HTML tables as their
    raw source. Use :func:`markdown_tables_to_html` for TEDS-ready HTML.
    """
    if not markdown:
        return []

    tables: list[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if "<table" in line.lower():
            # consume until the closing tag
            buffer = [line]
            while index < len(lines) and "</table>" not in buffer[-1].lower():
                index += 1
                if index < len(lines):
                    buffer.append(lines[index])
            tables.append("\n".join(buffer))
            index += 1
            continue

        header_cells = _split_pipe_row(line)
        if (
            header_cells
            and index + 1 < len(lines)
            and _SEPARATOR_ROW.match(lines[index + 1])
        ):
            header_index = index
            body_start = index + 2  # header + separator
            index = body_start
            while index < len(lines):
                cells = _split_pipe_row(lines[index])
                if not cells:
                    break
                index += 1
            tables.append("\n".join(lines[header_index:index]))
            continue

        index += 1
    return tables


def markdown_tables_to_html(markdown: str | None) -> list[str]:
    """Extract every table from a Markdown page as TEDS-comparable HTML.

    Each returned string is a full ``<html><body><table>…`` document, the
    format the vendored TEDS implementation expects.
    """
    results: list[str] = []
    for raw in extract_markdown_tables(markdown):
        if "<table" in raw.lower():
            html_table = _html_table_to_normalized_html(raw)
        else:
            lines = raw.splitlines()
            rows = [
                cells
                for line in lines
                if (cells := _split_pipe_row(line)) is not None
                and not _SEPARATOR_ROW.match(line)
            ]
            has_header = bool(len(lines) > 1 and _SEPARATOR_ROW.match(lines[1]))
            html_table = _pipe_table_to_html(rows, has_header)
        if html_table:
            results.append(f"<html><body>{html_table}</body></html>")
    return results
