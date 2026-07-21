"""Gold-document retrieval metrics for datasets with evidence IDs.

These metrics evaluate whether retrieved chunks came from the known gold
document(s) for a question. They are intentionally separate from the existing
heuristic context-overlap metrics so both modes can be compared.

Supports both single-gold datasets (SQuAD-style, one gold doc per question)
and multi-gold datasets (HotpotQA / MuSiQue / 2WikiMultiHopQA, where
``supporting_facts`` references several gold paragraphs). For multi-hop data
the gold set is auto-derived from ``supporting_facts`` / ``supporting_contexts``
in the sample metadata when ``gold_doc_id`` is absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GoldRetrievalMetricsResult:
    metric_means: dict[str, float]
    per_sample: list[dict[str, float | None]]
    samples_with_valid_scores: dict[str, int]
    skipped_samples: int = 0


# ── Identifier extraction ────────────────────────────────────────────


def _extract_doc_id(metadata: dict[str, Any] | None) -> str | None:
    """Pull a comparable doc identifier out of a retrieved-chunk metadata dict.

    Multi-hop corpora expose gold evidence as paragraph titles (HotpotQA,
    2WikiMultiHopQA) or paragraph ids (MuSiQue). Retrieved chunks may therefore
    carry the title in ``title`` / ``paragraph_id`` rather than ``doc_id``.
    """
    if not metadata:
        return None
    for key in ("doc_id", "title", "paragraph_id", "id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _gold_titles_from_supporting_facts(raw: Any) -> set[str]:
    """Extract unique titles from ``supporting_facts`` / ``supporting_contexts``.

    Both fields follow HotpotQA's shape: ``list[[title, sent_idx], ...]``.
    Some datasets use ``list[{"title": ...}]`` — handled too.
    """
    titles: set[str] = set()
    if not raw:
        return titles
    if isinstance(raw, dict):
        raw = raw.get("title") or raw.get("titles") or []
    for item in raw:
        if isinstance(item, str):
            titles.add(item)
        elif isinstance(item, (list, tuple)) and item:
            titles.add(str(item[0]))
        elif isinstance(item, dict):
            title = item.get("title") or item.get("paragraph_id")
            if title:
                titles.add(str(title))
    return titles


def _resolve_gold_set(
    gold: str | Iterable[str] | None,
    sample_metadata: dict[str, Any] | None,
) -> set[str]:
    """Normalise the per-sample gold argument into a set of comparable ids.

    Order of precedence:
      1. explicit ``gold`` argument (str or iterable of str)
      2. ``metadata["gold_doc_id"]``
      3. ``metadata["gold_doc_ids"]``
      4. titles derived from ``metadata["supporting_facts"]``
      5. titles derived from ``metadata["supporting_contexts"]``
    """
    if gold is not None:
        if isinstance(gold, str):
            return {gold} if gold else set()
        return {str(g) for g in gold if g}

    if sample_metadata:
        single = sample_metadata.get("gold_doc_id")
        if single:
            return {str(single)}
        multi = sample_metadata.get("gold_doc_ids")
        if multi:
            return {str(g) for g in multi if g}
        supporting = _gold_titles_from_supporting_facts(
            sample_metadata.get("supporting_facts")
        )
        if supporting:
            return supporting
        supporting = _gold_titles_from_supporting_facts(
            sample_metadata.get("supporting_contexts")
        )
        if supporting:
            return supporting
    return set()


# ── Metric primitives ────────────────────────────────────────────────


def _hit_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """1.0 if ANY gold id appears in the top-k retrieved, else 0.0."""
    if not gold or k <= 0:
        return 0.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def _ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """NDCG@k with binary multi-relevance.

    IDCG ranks every gold item in the first positions, capped at k.
    """
    if not gold or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(top_k)
        if doc_id in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of gold ids retrieved in top-k."""
    if not gold or k <= 0:
        return 0.0
    hits = len(gold & set(retrieved[:k]))
    return hits / len(gold)


# ── Public entry point ───────────────────────────────────────────────


def compute_gold_doc_retrieval_metrics(
    gold_doc_ids: list[str | None | Iterable[str]],
    retrieved_metadata: list[list[dict[str, Any]]],
    *,
    sample_metadata: list[dict[str, Any]] | None = None,
    k_values: list[int] | None = None,
) -> GoldRetrievalMetricsResult:
    """Compute document-level retrieval metrics from gold and retrieved IDs.

    ``gold_doc_ids`` and ``retrieved_metadata`` must be parallel lists. Each
    entry of ``gold_doc_ids`` may be a single id, ``None``, or an iterable of
    ids (for multi-hop datasets with several gold paragraphs). A sample without
    any resolvable gold id is skipped.

    When ``sample_metadata`` is provided, the resolver falls back to
    ``supporting_facts`` / ``gold_doc_ids`` entries so callers that only pass
    multi-hop metadata do not need to pre-flatten gold titles.
    """
    ks = k_values or [1, 3, 5]
    per_sample: list[dict[str, float | None]] = []
    accum: dict[str, list[float]] = {}
    skipped = 0

    for i, (gold, metadata_list) in enumerate(zip(gold_doc_ids, retrieved_metadata)):
        meta = sample_metadata[i] if sample_metadata and i < len(sample_metadata) else None
        gold_set = _resolve_gold_set(gold, meta)
        scores: dict[str, float | None] = {}

        if not gold_set:
            skipped += 1
            for k in ks:
                scores[f"hit@{k}"] = None
                scores[f"ndcg@{k}"] = None
                scores[f"recall@{k}"] = None
                scores[f"supporting_fact_coverage@{k}"] = None
            per_sample.append(scores)
            continue

        retrieved_doc_ids = [
            doc_id
            for doc_id in (_extract_doc_id(metadata) for metadata in metadata_list)
            if doc_id is not None
        ]

        for k in ks:
            hit = _hit_at_k(gold_set, retrieved_doc_ids, k)
            ndcg = _ndcg_at_k(gold_set, retrieved_doc_ids, k)
            recall = _recall_at_k(gold_set, retrieved_doc_ids, k)
            # supporting_fact_coverage@k is the title-level fraction of gold
            # supporting facts found in the top-k. For multi-hop datasets this
            # is exactly multi-relevant recall@k; for single-gold data it
            # collapses to hit@k. Exposed under its own name so downstream
            # reporting can pick the multi-hop interpretation explicitly.
            coverage = recall
            for name, value in (
                (f"hit@{k}", hit),
                (f"ndcg@{k}", ndcg),
                (f"recall@{k}", recall),
                (f"supporting_fact_coverage@{k}", coverage),
            ):
                scores[name] = value
                accum.setdefault(name, []).append(value)

        per_sample.append(scores)

    means = {key: sum(values) / len(values) for key, values in accum.items() if values}
    valid_counts = {key: len(values) for key, values in accum.items() if values}

    return GoldRetrievalMetricsResult(
        metric_means=means,
        per_sample=per_sample,
        samples_with_valid_scores=valid_counts,
        skipped_samples=skipped,
    )
