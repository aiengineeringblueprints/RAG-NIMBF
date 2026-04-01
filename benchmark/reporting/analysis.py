from __future__ import annotations

from dataclasses import dataclass

from benchmark.reporting.models import BenchmarkResultExtended


@dataclass(frozen=True)
class ConfigRank:
    config_name: str
    rank: int
    composite_score: float
    metric_ranks: dict[str, int]
    best_metrics: tuple[str, ...]
    worst_metrics: tuple[str, ...]


@dataclass(frozen=True)
class RankTable:
    ranks: tuple[ConfigRank, ...]
    insights: tuple[str, ...]


# Weights for composite scoring: RAGAS metrics sum to 0.8, perf metrics sum to 0.2
_METRIC_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.20,
    "answer_relevancy": 0.20,
    "context_precision": 0.20,
    "context_recall": 0.20,
    "ttft": 0.10,       # lower is better
    "tok_per_s": 0.10,  # higher is better
}

_LOWER_IS_BETTER = {"ttft"}


def _get_metric_values(results: list[BenchmarkResultExtended], key: str) -> list[float | None]:
    mapping: dict[str, str] = {
        "faithfulness": "ragas_faithfulness",
        "answer_relevancy": "ragas_answer_relevancy",
        "context_precision": "ragas_context_precision",
        "context_recall": "ragas_context_recall",
        "ttft": "avg_ttft_seconds",
        "tok_per_s": "avg_tokens_per_second",
    }
    attr = mapping[key]
    return [getattr(r, attr) for r in results]


def _normalize(values: list[float | None], lower_is_better: bool) -> list[float | None]:
    clean = [(i, v) for i, v in enumerate(values) if v is not None]
    if not clean:
        return [None] * len(values)

    raw_vals = [v for _, v in clean]
    lo, hi = min(raw_vals), max(raw_vals)
    span = hi - lo if hi != lo else 1.0

    result: list[float | None] = [None] * len(values)
    for idx, val in clean:
        norm = (val - lo) / span
        if lower_is_better:
            norm = 1.0 - norm
        result[idx] = norm
    return result


def compute_rankings(results: list[BenchmarkResultExtended]) -> RankTable:
    if not results:
        return RankTable(ranks=(), insights=())

    # Single config: no comparative ranking, just report absolute values
    if len(results) == 1:
        r = results[0]
        insights = _single_config_insights(r)
        cr = ConfigRank(
            config_name=r.config_name,
            rank=1,
            composite_score=0.0,
            metric_ranks={},
            best_metrics=(),
            worst_metrics=(),
        )
        return RankTable(ranks=(cr,), insights=tuple(insights))

    # Compute normalized scores per metric
    normalized: dict[str, list[float | None]] = {}
    for key in _METRIC_WEIGHTS:
        raw = _get_metric_values(results, key)
        normalized[key] = _normalize(raw, key in _LOWER_IS_BETTER)

    # Composite score per config
    composites: list[float] = []
    for i in range(len(results)):
        score = 0.0
        for key, weight in _METRIC_WEIGHTS.items():
            val = normalized[key][i]
            score += weight * val if val is not None else 0.0
        composites.append(score)

    # Rank by composite (descending)
    sorted_indices = sorted(range(len(results)), key=lambda i: composites[i], reverse=True)

    # Per-metric ranks
    metric_ranks_per_config: list[dict[str, int]] = [{} for _ in results]
    for key in _METRIC_WEIGHTS:
        lower = key in _LOWER_IS_BETTER
        raw = _get_metric_values(results, key)
        clean_pairs = [(i, v) for i, v in enumerate(raw) if v is not None]
        if not clean_pairs:
            continue
        sorted_by_metric = sorted(clean_pairs, key=lambda p: p[1], reverse=not lower)
        for rank, (idx, _) in enumerate(sorted_by_metric, 1):
            metric_ranks_per_config[idx][key] = rank

    # Build ConfigRank objects
    config_ranks: list[ConfigRank] = []
    for rank_pos, idx in enumerate(sorted_indices, 1):
        r = results[idx]
        m_ranks = metric_ranks_per_config[idx]
        best = tuple(k for k, v in m_ranks.items() if v == 1)
        worst = tuple(k for k, v in m_ranks.items() if v == len(results))
        config_ranks.append(ConfigRank(
            config_name=r.config_name,
            rank=rank_pos,
            composite_score=composites[idx],
            metric_ranks=dict(m_ranks),
            best_metrics=best,
            worst_metrics=worst,
        ))

    insights = _multi_config_insights(results, config_ranks)

    return RankTable(ranks=tuple(config_ranks), insights=tuple(insights))


def _single_config_insights(r: BenchmarkResultExtended) -> list[str]:
    lines: list[str] = []
    lines.append(f"Single configuration: {r.config_name}")
    lines.append(f"  Processed {r.num_questions} questions across {r.num_chunks} chunks")
    if r.ttft_stats:
        lines.append(f"  TTFT: {r.ttft_stats.mean:.3f}s (std {r.ttft_stats.std:.3f}s)")
    if r.tps_stats:
        lines.append(f"  Throughput: {r.tps_stats.mean:.1f} tok/s (std {r.tps_stats.std:.1f})")
    if r.ragas_faithfulness is not None:
        lines.append(f"  Faithfulness: {r.ragas_faithfulness:.3f}")
    if r.ragas_answer_relevancy is not None:
        lines.append(f"  Answer Relevancy: {r.ragas_answer_relevancy:.3f}")
    if r.ragas_context_precision is not None:
        lines.append(f"  Context Precision: {r.ragas_context_precision:.3f}")
    if r.ragas_context_recall is not None:
        lines.append(f"  Context Recall: {r.ragas_context_recall:.3f}")
    lines.append(f"  Total time: {r.total_time_seconds:.1f}s")
    return lines


def _multi_config_insights(
    results: list[BenchmarkResultExtended], ranks: list[ConfigRank]
) -> list[str]:
    lines: list[str] = []
    best = ranks[0]
    lines.append(f"Best overall: {best.config_name} (composite score: {best.composite_score:.3f})")
    if best.best_metrics:
        lines.append(f"  Best at: {', '.join(best.best_metrics)}")

    # Faithfulness leader
    faith_vals = [(r.config_name, r.ragas_faithfulness) for r in results if r.ragas_faithfulness is not None]
    if faith_vals:
        top = max(faith_vals, key=lambda p: p[1])
        lines.append(f"Highest faithfulness: {top[0]} ({top[1]:.3f})")

    # Answer relevancy leader
    rel_vals = [(r.config_name, r.ragas_answer_relevancy) for r in results if r.ragas_answer_relevancy is not None]
    if rel_vals:
        top = max(rel_vals, key=lambda p: p[1])
        lines.append(f"Highest answer relevancy: {top[0]} ({top[1]:.3f})")

    # TTFT range
    ttfts = [r.avg_ttft_seconds for r in results]
    if len(ttfts) > 1:
        lo_t, hi_t = min(ttfts), max(ttfts)
        ratio = hi_t / lo_t if lo_t > 0 else float("inf")
        fastest = results[ttfts.index(lo_t)].config_name
        lines.append(f"Fastest TTFT: {fastest} ({lo_t:.2f}s) — range: {lo_t:.2f}s to {hi_t:.2f}s ({ratio:.1f}x spread)")

    # Throughput leader
    tps_vals = [(r.config_name, r.avg_tokens_per_second) for r in results if r.avg_tokens_per_second > 0]
    if len(tps_vals) > 1:
        top = max(tps_vals, key=lambda p: p[1])
        lines.append(f"Best throughput: {top[0]} ({top[1]:.1f} tok/s)")

    return lines
