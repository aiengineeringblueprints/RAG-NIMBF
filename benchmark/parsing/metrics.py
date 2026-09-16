"""Text metrics for parsing evaluation (OCR-02).

Compares a parser's ``ParseResult`` against per-page ground-truth text using
CER, WER (jiwer) and normalized edit distance (RapidFuzz), all computed over
OmniDocBench-style normalized text (case folding, whitespace collapsing,
NFKC unicode folding, accent stripping) so cosmetic differences do not count
as errors.

Blank-page convention: if prediction and ground truth are both blank the
score is 0.0 (exact match); if exactly one side is blank the score is 1.0
(maximal error), because the edit-distance denominators would otherwise
degenerate to zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jiwer
from rapidfuzz.distance import Levenshtein

from benchmark.parsing.alignment import (
    AlignedPair,
    alignment_run_metadata,
    normalize_text,
    quick_match,
    split_paragraphs,
)
from benchmark.parsing.base import ParseResult

# ── Normalization ────────────────────────────────────────────────────

__all__ = [
    "PageTextMetrics",
    "ParsingTextMetricsResult",
    "cer",
    "compute_aligned_parsing_text_metrics",
    "compute_parsing_text_metrics",
    "normalize_text",
    "normalized_edit_distance",
    "wer",
]


# normalize_text is re-exported from benchmark.parsing.alignment.

# ── Blank-aware metric primitives ────────────────────────────────────


def cer(prediction: str, ground_truth: str) -> float:
    """Character error rate over normalized text (jiwer-based)."""
    pred, gt = normalize_text(prediction), normalize_text(ground_truth)
    if not gt and not pred:
        return 0.0
    if not gt or not pred:
        return 1.0
    return jiwer.cer(gt, pred)


def wer(prediction: str, ground_truth: str) -> float:
    """Word error rate over normalized text (jiwer-based)."""
    pred, gt = normalize_text(prediction), normalize_text(ground_truth)
    if not gt and not pred:
        return 0.0
    if not gt or not pred:
        return 1.0
    return jiwer.wer(gt, pred)


def normalized_edit_distance(prediction: str, ground_truth: str) -> float:
    """Levenshtein distance normalized to [0, 1] by max string length."""
    pred, gt = normalize_text(prediction), normalize_text(ground_truth)
    if not gt and not pred:
        return 0.0
    if not gt or not pred:
        return 1.0
    return Levenshtein.normalized_distance(pred, gt)


# ── Aggregation ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageTextMetrics:
    """Per-page parsing text scores; ``None`` when no GT exists for a page."""

    page_number: int
    cer: float | None
    wer: float | None
    normalized_edit_distance: float | None


@dataclass(frozen=True)
class ParsingTextMetricsResult:
    """Aggregated parsing text metrics, mirroring the gold-retrieval shape."""

    per_page: list[PageTextMetrics]
    metric_means: dict[str, float]
    pages_with_valid_scores: dict[str, int]
    alignment: dict | None = None


def compute_parsing_text_metrics(
    result: ParseResult,
    ground_truth: Sequence[str] | Mapping[int, str],
) -> ParsingTextMetricsResult:
    """Score a ParseResult against per-page ground-truth text.

    ``ground_truth`` is either a sequence of page texts aligned by position
    or a mapping keyed by page number (1-based). Pages present on only one
    side still get scored (missing text counts as a full error); pages
    without ground truth at all are excluded from the aggregates.
    """
    if isinstance(ground_truth, Mapping):
        gt_by_page: dict[int, str] = {
            int(k): v for k, v in ground_truth.items()
        }
    else:
        gt_by_page = {
            index + 1: text for index, text in enumerate(ground_truth)
        }

    per_page: list[PageTextMetrics] = []
    accum: dict[str, list[float]] = {}

    page_numbers = sorted(
        {page.page_number for page in result.pages} | set(gt_by_page)
    )
    predicted = {page.page_number: page.markdown for page in result.pages}

    for page_number in page_numbers:
        if page_number not in gt_by_page:
            continue
        gt = gt_by_page[page_number]
        pred = predicted.get(page_number, "")
        scores = {
            "cer": cer(pred, gt),
            "wer": wer(pred, gt),
            "normalized_edit_distance": normalized_edit_distance(pred, gt),
        }
        for name, value in scores.items():
            accum.setdefault(name, []).append(value)
        per_page.append(
            PageTextMetrics(page_number=page_number, **scores)  # type: ignore[arg-type]
        )

    means = {key: sum(vals) / len(vals) for key, vals in accum.items() if vals}
    valid_counts = {key: len(vals) for key, vals in accum.items() if vals}

    return ParsingTextMetricsResult(
        per_page=per_page,
        metric_means=means,
        pages_with_valid_scores=valid_counts,
    )


def _pair_scores(pair: AlignedPair) -> dict[str, float] | None:
    """Score one aligned pair; pairs without GT cannot be scored."""
    if not pair.gt_indices:
        return None
    scores = {
        "cer": cer(pair.pred, pair.gt),
        "wer": wer(pair.pred, pair.gt),
        "normalized_edit_distance": normalized_edit_distance(pair.pred, pair.gt),
    }
    return scores


def compute_aligned_parsing_text_metrics(
    result: ParseResult,
    ground_truth: Sequence[str] | Mapping[int, str],
) -> ParsingTextMetricsResult:
    """Score a ParseResult on quick_match-aligned paragraph pairs.

    Each page's prediction and ground truth are segmented into paragraphs
    and aligned with the vendored OmniDocBench ``quick_match``; CER/WER and
    normalized edit distance are computed per aligned pair and averaged per
    page, then across pages. Pairs without GT text (hallucinated prediction
    paragraphs) cannot be scored and are excluded from the aggregates, while
    GT paragraphs without a prediction count as full errors. The matching
    algorithm identity is recorded in ``result.alignment`` for run metadata
    and reports.
    """
    if isinstance(ground_truth, Mapping):
        gt_by_page: dict[int, str] = {int(k): v for k, v in ground_truth.items()}
    else:
        gt_by_page = {index + 1: text for index, text in enumerate(ground_truth)}

    predicted = {page.page_number: page.markdown for page in result.pages}
    page_numbers = sorted(
        {page.page_number for page in result.pages} | set(gt_by_page)
    )

    per_page: list[PageTextMetrics] = []
    accum: dict[str, list[float]] = {}
    total_matched = 0
    total_unmatched = 0

    for page_number in page_numbers:
        if page_number not in gt_by_page:
            continue
        gt = gt_by_page[page_number]
        pred = predicted.get(page_number, "")
        pairs = quick_match(split_paragraphs(gt), split_paragraphs(pred))

        page_scores: dict[str, list[float]] = {}
        for pair in pairs:
            if pair.matched:
                total_matched += 1
            else:
                total_unmatched += 1
            scores = _pair_scores(pair)
            if scores is None:
                continue
            for name, value in scores.items():
                page_scores.setdefault(name, []).append(value)
                accum.setdefault(name, []).append(value)

        scores_out = {
            name: sum(vals) / len(vals) for name, vals in page_scores.items() if vals
        }
        per_page.append(
            PageTextMetrics(page_number=page_number, **scores_out)  # type: ignore[arg-type]
        )

    means = {key: sum(vals) / len(vals) for key, vals in accum.items() if vals}
    valid_counts = {key: len(vals) for key, vals in accum.items() if vals}

    alignment_meta = alignment_run_metadata()
    alignment_meta["matched_pairs"] = total_matched
    alignment_meta["unmatched_pairs"] = total_unmatched

    return ParsingTextMetricsResult(
        per_page=per_page,
        metric_means=means,
        pages_with_valid_scores=valid_counts,
        alignment=alignment_meta,
    )
