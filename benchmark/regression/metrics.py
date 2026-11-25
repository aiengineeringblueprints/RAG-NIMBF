"""Per-sample metric extraction and the per-run dimension summary."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from benchmark.answer_match import declines
from benchmark.regression.question_set import question_categories, sample_keys

# Gold-label retrieval scores in ``custom_scores`` (benchmark.gold_retrieval_metrics
# and the RAGBench relevant-sentence recall).
RETRIEVAL_PREFIXES = (
    "hit@",
    "recall@",
    "ndcg@",
    "mrr",
    "relevant_sentence_recall",
    "supporting_fact_coverage@",
)

# Per-sample metrics where a lower value is better; every other metric is
# higher-is-better.
LOWER_IS_BETTER = frozenset(
    {"false_refusal", "total_seconds", "ttft_seconds", "estimated_cost_usd", "total_tokens"}
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return float(value)


def sample_metrics(result: Mapping[str, Any], category: str = "unlabeled") -> dict[str, float | None]:
    """Per-sample metric values of one result record; ``None`` = not scored.

    Retrieval metrics count only when the sample has gold document IDs and
    is not known to be unanswerable. Abstention is split by category:
    ``abstained`` (declining an unanswerable question is correct) and
    ``false_refusal`` (declining an answerable one is an error); ``ambiguous``
    and ``unlabeled`` samples count for neither.
    """
    ragas = result.get("ragas_scores") or {}
    custom = result.get("custom_scores") or {}
    metrics: dict[str, float | None] = {
        "answer_correctness": _number(ragas.get("answer_correctness")),
        "bert_score_f1": _number(custom.get("bert_score_f1")),
        "faithfulness": _number(ragas.get("faithfulness")),
    }
    if result.get("ground_truth_doc_ids") and category != "unanswerable":
        for name, value in custom.items():
            if name.startswith(RETRIEVAL_PREFIXES):
                metrics[name] = _number(value)
    refused = float(declines(result.get("answer") or ""))
    metrics["abstained"] = refused if category == "unanswerable" else None
    metrics["false_refusal"] = refused if category == "answerable" else None
    metrics["total_seconds"] = _number(result.get("total_seconds"))
    metrics["ttft_seconds"] = _number(result.get("ttft_seconds"))
    metrics["estimated_cost_usd"] = _number(result.get("estimated_cost_usd"))
    # Adapters that do not report usage write 0 tokens; that is "unknown".
    metrics["total_tokens"] = _number(result.get("total_tokens")) or None
    return metrics


def _values(per_sample: list[dict[str, float | None]], name: str) -> list[float]:
    return [m[name] for m in per_sample if m.get(name) is not None]


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _dimension(metrics: dict[str, float], reason: str) -> dict[str, Any]:
    if metrics:
        return {"status": "ok", "metrics": metrics}
    return {"status": "unavailable", "reason": reason, "metrics": {}}


def summarize_run(
    results: Sequence[Mapping[str, Any]],
    question_set: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce one run's per-sample results to the evaluation dimensions."""
    categories = question_categories(question_set)
    per_sample = [
        sample_metrics(result, categories.get(key, "unlabeled"))
        for key, result in zip(sample_keys(results), results)
    ]

    def means(names: Sequence[str]) -> dict[str, float]:
        return {
            name: statistics.fmean(values)
            for name in names
            if (values := _values(per_sample, name))
        }

    retrieval_names = sorted({n for m in per_sample for n in m if n.startswith(RETRIEVAL_PREFIXES)})

    abstention: dict[str, float] = {}
    if n_ambiguous := sum(1 for key in sample_keys(results) if categories.get(key) == "ambiguous"):
        abstention["n_ambiguous"] = n_ambiguous
    if abstained := _values(per_sample, "abstained"):
        abstention["abstention_rate"] = statistics.fmean(abstained)
        abstention["n_unanswerable"] = len(abstained)
    if false_refusals := _values(per_sample, "false_refusal"):
        abstention["false_refusal_rate"] = statistics.fmean(false_refusals)

    latency: dict[str, float] = {}
    if seconds := _values(per_sample, "total_seconds"):
        latency = {
            "total_seconds_mean": statistics.fmean(seconds),
            "total_seconds_median": statistics.median(seconds),
            "total_seconds_p95": _p95(seconds),
        }
        latency.update({f"{k}_mean": v for k, v in means(["ttft_seconds"]).items()})

    cost: dict[str, float] = {}
    if costs := _values(per_sample, "estimated_cost_usd"):
        cost["estimated_cost_usd_total"] = sum(costs)
    if tokens := _values(per_sample, "total_tokens"):
        cost["total_tokens_mean"] = statistics.fmean(tokens)

    return {
        "n": len(results),
        "dimensions": {
            "correctness": _dimension(
                means(["answer_correctness", "bert_score_f1"]),
                "no answer_correctness (RAGAS) or bert_score_f1 scores",
            ),
            "faithfulness": _dimension(
                means(["faithfulness"]), "RAGAS faithfulness not scored (RAGAS disabled?)"
            ),
            "retrieval": _dimension(
                means(retrieval_names),
                "no gold evidence labels (ground_truth_doc_ids) on answerable samples",
            ),
            "abstention": _dimension(
                abstention, "no unanswerable questions labelled in the question set"
            ),
            "latency": _dimension(latency, "no total_seconds recorded"),
            "cost": _dimension(cost, "adapter reports neither token usage nor cost"),
        },
    }
