"""Gold-document retrieval metrics for datasets with evidence IDs.

These metrics evaluate whether retrieved chunks came from the known gold
document for a question. They are intentionally separate from the existing
heuristic context-overlap metrics so both modes can be compared.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoldRetrievalMetricsResult:
    metric_means: dict[str, float]
    per_sample: list[dict[str, float | None]]
    samples_with_valid_scores: dict[str, int]
    skipped_samples: int = 0


def _extract_doc_id(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    doc_id = metadata.get("doc_id")
    return str(doc_id) if doc_id is not None else None


def _hit_at_k(gold_doc_id: str, retrieved_doc_ids: list[str], k: int) -> float:
    return 1.0 if gold_doc_id in retrieved_doc_ids[:k] else 0.0


def _ndcg_at_k(gold_doc_id: str, retrieved_doc_ids: list[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id == gold_doc_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def compute_gold_doc_retrieval_metrics(
    gold_doc_ids: list[str | None],
    retrieved_metadata: list[list[dict[str, Any]]],
    *,
    k_values: list[int] | None = None,
) -> GoldRetrievalMetricsResult:
    """Compute document-level retrieval metrics from gold and retrieved IDs.

    ``gold_doc_ids`` and ``retrieved_metadata`` must be parallel lists. A sample
    without a gold document ID is skipped for these metrics.
    """
    ks = k_values or [1, 3, 5]
    per_sample: list[dict[str, float | None]] = []
    accum: dict[str, list[float]] = {}
    skipped = 0

    for gold_doc_id, metadata_list in zip(gold_doc_ids, retrieved_metadata):
        scores: dict[str, float | None] = {}
        if not gold_doc_id:
            skipped += 1
            for k in ks:
                scores[f"hit@{k}"] = None
                scores[f"ndcg@{k}"] = None
                scores[f"recall@{k}"] = None
            per_sample.append(scores)
            continue

        retrieved_doc_ids = [
            doc_id
            for doc_id in (_extract_doc_id(metadata) for metadata in metadata_list)
            if doc_id is not None
        ]

        for k in ks:
            hit = _hit_at_k(gold_doc_id, retrieved_doc_ids, k)
            ndcg = _ndcg_at_k(gold_doc_id, retrieved_doc_ids, k)
            # With one known gold document per SQuAD-style question, recall@k
            # is equivalent to hit@k.
            recall = hit
            for name, value in (
                (f"hit@{k}", hit),
                (f"ndcg@{k}", ndcg),
                (f"recall@{k}", recall),
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
