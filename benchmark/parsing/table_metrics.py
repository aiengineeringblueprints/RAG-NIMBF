"""Table-parsing quality metrics over aligned table regions (OCR-04).

Scores a parser's ``ParseResult`` against ground-truth pages that provide
table structure.  Both sides' tables are extracted with
:func:`benchmark.parsing.tables.markdown_tables_to_html` — Markdown-only
parsers therefore score on tables without emitting HTML themselves — and
matched region-by-region in document order:

- each ground-truth table is paired with the prediction table at the same
  position; TEDS and TEDS-S are computed per pair
- ground-truth tables without a prediction counterpart count as full
  errors (score 0.0); extra prediction tables are ignored
- pages without ground-truth tables are excluded from the aggregates, so
  documents without tables score without error

Per-table scores aggregate to per-page and document-level means, mirroring
the text-metrics result shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from benchmark.parsing.base import ParseResult
from benchmark.parsing.tables import markdown_tables_to_html
from benchmark.parsing.teds import (
    TEDS_ALGORITHM_LICENSE,
    TEDS_ALGORITHM_NAME,
    TEDS_ALGORITHM_SOURCE,
    TEDS_ALGORITHM_VERSION,
    teds,
)

__all__ = [
    "PageTableMetrics",
    "ParsingTableMetricsResult",
    "compute_parsing_table_metrics",
]


@dataclass(frozen=True)
class PageTableMetrics:
    """Per-page table scores; ``None`` when the page has no GT tables."""

    page_number: int
    teds: float | None
    teds_s: float | None


@dataclass(frozen=True)
class ParsingTableMetricsResult:
    """Aggregated table metrics, mirroring the text-metrics result shape."""

    per_page: list[PageTableMetrics]
    per_table: list[dict[str, float]] = field(default_factory=list)
    metric_means: dict[str, float] = field(default_factory=dict)
    pages_with_valid_scores: dict[str, int] = field(default_factory=dict)
    metric_metadata: dict[str, str] = field(default_factory=dict)


def compute_parsing_table_metrics(
    result: ParseResult,
    ground_truth: Sequence[str] | Mapping[int, str],
    structure_only_variant: bool = True,
) -> ParsingTableMetricsResult:
    """Score a ParseResult's tables against per-page ground-truth tables.

    ``ground_truth`` is either a sequence of page Markdown aligned by
    position or a mapping keyed by page number (1-based). Pages present on
    only one side are still scored (a missing prediction table is a full
    error); pages without ground-truth tables are excluded.
    """
    if isinstance(ground_truth, Mapping):
        gt_by_page: dict[int, str] = {int(k): v for k, v in ground_truth.items()}
    else:
        gt_by_page = {index + 1: text for index, text in enumerate(ground_truth)}

    predicted = {page.page_number: page.markdown for page in result.pages}
    page_numbers = sorted(
        {page.page_number for page in result.pages} | set(gt_by_page)
    )

    per_page: list[PageTableMetrics] = []
    per_table: list[dict[str, float]] = []
    accum: dict[str, list[float]] = {}

    for page_number in page_numbers:
        if page_number not in gt_by_page:
            continue
        gt_tables = markdown_tables_to_html(gt_by_page[page_number])
        if not gt_tables:
            continue
        pred_tables = markdown_tables_to_html(predicted.get(page_number, ""))

        page_scores: dict[str, list[float]] = {}
        for position, gt_html in enumerate(gt_tables):
            pred_html = (
                pred_tables[position] if position < len(pred_tables) else ""
            )
            scores = {
                "teds": teds(pred_html, gt_html),
            }
            if structure_only_variant:
                scores["teds_s"] = teds(pred_html, gt_html, structure_only=True)
            per_table.append(scores)
            for name, value in scores.items():
                page_scores.setdefault(name, []).append(value)
                accum.setdefault(name, []).append(value)

        per_page.append(
            PageTableMetrics(
                page_number=page_number,
                **{
                    name: sum(vals) / len(vals)
                    for name, vals in page_scores.items()
                },  # type: ignore[arg-type]
            )
        )

    means = {key: sum(vals) / len(vals) for key, vals in accum.items() if vals}
    valid_counts = {key: len(vals) for key, vals in accum.items() if vals}

    return ParsingTableMetricsResult(
        per_page=per_page,
        per_table=per_table,
        metric_means=means,
        pages_with_valid_scores=valid_counts,
        metric_metadata={
            "algorithm": TEDS_ALGORITHM_NAME,
            "version": TEDS_ALGORITHM_VERSION,
            "license": TEDS_ALGORITHM_LICENSE,
            "source": TEDS_ALGORITHM_SOURCE,
        },
    )
