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

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jiwer
from rapidfuzz.distance import Levenshtein

from benchmark.parsing.base import ParseResult

# ── Normalization ────────────────────────────────────────────────────


def normalize_text(text: str | None) -> str:
    """Fold text the way OmniDocBench-style OCR evaluation expects.

    Applies NFKC unicode normalization (fullwidth -> ASCII, ligature
    expansion), accent stripping, lowercasing, and whitespace collapsing to
    single spaces.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    folded = unicodedata.normalize("NFKC", stripped)
    return " ".join(folded.casefold().split())


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
