"""Calibration report: direct RAGAS vs. post-run MLflow scorers (issue #46).

Pairs every sample's direct RAGAS score (from the config's result JSON) with
the MLflow scorer assessment on the same sample's trace, and reports per
metric pair how far the two evaluators agree: mean difference, correlation,
failure rates and runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
from scipy import stats as _scipy_stats

from benchmark.mlflow_scorers import RUNTIME_METRIC, SCORER_NAMES_TAG
from benchmark.reporting.exports import load_json_report

# Direct RAGAS stage wall time, logged by the runner's stage timer.
DIRECT_RUNTIME_METRIC = "stage_ragas_eval_seconds"

# Per-trace assessment that carries a scorer's per-sample value, keyed by the
# ``mlflow_scorers`` config name. RAGAS-via-MLflow scorers are named after
# the RAGAS metric class; RetrievalRelevance emits one verdict per chunk plus
# a per-span precision, which is the per-sample value. Others: config name.
ASSESSMENT_NAMES: dict[str, str] = {
    "retrieval_relevance": "retrieval_relevance/precision",
    "ragas_faithfulness": "Faithfulness",
    "ragas_context_precision": "ContextPrecision",
    "ragas_context_recall": "ContextRecall",
    "ragas_factual_correctness": "FactualCorrectness",
}

_YES_NO = {"yes": 1.0, "no": 0.0}


@dataclass(frozen=True)
class MetricPair:
    """A direct RAGAS metric and an MLflow scorer that measures the same idea."""

    direct: str  # key in a sample's ``ragas_scores``
    judge: str  # ``mlflow_scorers`` config name
    relation: str  # how comparable the two are


# Same-math pairs first, then the binary / differently defined proxies
# (doc 13, section 3.2). semantic_similarity has no judge counterpart.
METRIC_PAIRS: tuple[MetricPair, ...] = (
    MetricPair("faithfulness", "ragas_faithfulness", "same RAGAS metric via MLflow"),
    MetricPair("context_precision", "ragas_context_precision", "same RAGAS metric via MLflow"),
    MetricPair("context_recall", "ragas_context_recall", "same RAGAS metric via MLflow"),
    MetricPair("answer_correctness", "answer_correctness", "custom claim-F1 judge (#45)"),
    MetricPair("answer_correctness", "ragas_factual_correctness", "claim F1, no similarity term"),
    MetricPair("faithfulness", "retrieval_groundedness", "binary proxy"),
    MetricPair("context_precision", "retrieval_relevance", "unweighted relevant-chunk share"),
    MetricPair("context_recall", "retrieval_sufficiency", "binary proxy"),
    MetricPair("answer_correctness", "correctness", "binary, recall-only"),
    MetricPair("answer_relevancy", "relevance_to_query", "binary proxy"),
)


@dataclass(frozen=True)
class PairStats:
    """Agreement of two evaluators over the same samples.

    Means and differences cover only samples both evaluators scored; a
    missing score (``None``) counts as a failure of that evaluator.
    """

    n_samples: int
    n_paired: int
    direct_mean: float | None
    judge_mean: float | None
    mean_diff: float | None  # judge - direct
    mean_abs_diff: float | None
    pearson: float | None
    spearman: float | None
    direct_failure_rate: float
    judge_failure_rate: float


def pair_stats(
    direct: Sequence[float | None], judge: Sequence[float | None]
) -> PairStats:
    """Compare per-sample scores of the direct RAGAS call and an MLflow judge."""
    if len(direct) != len(judge):
        raise ValueError(
            f"score lists cover different samples: {len(direct)} direct vs. {len(judge)} judge"
        )
    n = len(direct)
    pairs = [
        (d, j) for d, j in zip(direct, judge) if _is_score(d) and _is_score(j)
    ]
    xs = [d for d, _ in pairs]
    ys = [j for _, j in pairs]
    diffs = [j - d for d, j in pairs]
    return PairStats(
        n_samples=n,
        n_paired=len(pairs),
        direct_mean=_mean(xs),
        judge_mean=_mean(ys),
        mean_diff=_mean(diffs),
        mean_abs_diff=_mean([abs(x) for x in diffs]),
        pearson=_correlation(_scipy_stats.pearsonr, xs, ys),
        spearman=_correlation(_scipy_stats.spearmanr, xs, ys),
        direct_failure_rate=_failure_rate(direct),
        judge_failure_rate=_failure_rate(judge),
    )


def _is_score(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _failure_rate(values: Sequence[float | None]) -> float:
    if not values:
        return 0.0
    return sum(not _is_score(v) for v in values) / len(values)


def _correlation(fn, xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Correlation, or ``None`` when undefined (< 2 pairs or a constant side)."""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(fn(xs, ys).statistic)


def judge_scores(
    run_id: str, names: Sequence[str], *, n_samples: int
) -> dict[str, list[float | None]]:
    """Per-sample MLflow scorer values of a scored config run.

    Reads the traces linked to ``run_id`` and places each scorer's latest
    valid assessment at the trace's ``benchmark.sample_index``. Yes/no
    verdicts become 1/0; judge errors, unparseable values and samples
    without a trace or assessment are ``None``.
    """
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    traces = mlflow.search_traces(
        locations=[experiment_id], run_id=run_id, return_type="list"
    )
    scores: dict[str, list[float | None]] = {name: [None] * n_samples for name in names}
    for trace in traces:
        index = _sample_index(trace)
        if index is None or not 0 <= index < n_samples:
            continue
        latest = _latest_feedback(trace.info.assessments or [])
        for name in names:
            feedback = latest.get(ASSESSMENT_NAMES.get(name, name))
            if feedback is not None:
                scores[name][index] = _numeric(feedback)
    return scores


def _sample_index(trace: Any) -> int | None:
    for span in trace.data.spans:
        if span.parent_id is None:
            index = span.attributes.get("benchmark.sample_index")
            return int(index) if index is not None else None
    return None


def _latest_feedback(assessments: Sequence[Any]) -> Mapping[str, Any]:
    """Newest valid feedback per assessment name (re-scoring appends)."""
    latest: dict[str, Any] = {}
    for assessment in sorted(assessments, key=lambda a: a.create_time_ms or 0):
        if getattr(assessment, "feedback", None) is None or assessment.valid is False:
            continue
        latest[assessment.name] = assessment
    return latest


def _numeric(assessment: Any) -> float | None:
    if assessment.feedback.error is not None:
        return None
    value = assessment.feedback.value
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        return _YES_NO.get(value.strip().lower())
    return None


@dataclass(frozen=True)
class CalibrationReport:
    """Agreement of direct RAGAS and MLflow scorers on one config run."""

    config_name: str
    run_id: str
    n_samples: int
    pairs: tuple[tuple[MetricPair, PairStats], ...]
    direct_seconds: float | None
    judge_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "run_id": self.run_id,
            "n_samples": self.n_samples,
            "direct_seconds": self.direct_seconds,
            "judge_seconds": self.judge_seconds,
            "pairs": [{**asdict(pair), **asdict(stats)} for pair, stats in self.pairs],
        }


def build_report(
    results_path: Path, run_id: str, *, config_name: str | None = None
) -> CalibrationReport:
    """Compare a config's direct RAGAS scores with its run's MLflow scorers.

    ``results_path`` is the worker's config JSON (``results/runN/configs``)
    of the run that was logged as ``run_id``. A pair is reported when direct
    RAGAS scored the metric and the scorer was run (``mlflow_scorer_names``
    tag; older runs: at least one score), even if it failed on every sample.
    """
    result = _select_result(load_json_report(Path(results_path)), config_name)
    samples = result.get("per_sample") or []
    n = len(samples)
    judged = judge_scores(run_id, sorted({p.judge for p in METRIC_PAIRS}), n_samples=n)

    run = mlflow.get_run(run_id)
    configured = set(filter(None, run.data.tags.get(SCORER_NAMES_TAG, "").split(",")))
    pairs = []
    for pair in METRIC_PAIRS:
        direct = [_direct_score(sample, pair.direct) for sample in samples]
        judge = judged[pair.judge]
        judge_ran = pair.judge in configured or any(_is_score(v) for v in judge)
        if any(_is_score(v) for v in direct) and judge_ran:
            pairs.append((pair, pair_stats(direct, judge)))

    metrics = run.data.metrics
    return CalibrationReport(
        config_name=str(result.get("config_name", "")),
        run_id=run_id,
        n_samples=n,
        pairs=tuple(pairs),
        direct_seconds=metrics.get(DIRECT_RUNTIME_METRIC),
        judge_seconds=metrics.get(RUNTIME_METRIC),
    )


def _select_result(report: Mapping[str, Any], config_name: str | None) -> Mapping[str, Any]:
    results = report.get("results") or []
    if config_name is not None:
        results = [r for r in results if r.get("config_name") == config_name]
    if len(results) != 1:
        names = [r.get("config_name") for r in report.get("results") or []]
        raise ValueError(f"expected exactly one config result, pick one of {names} with --config")
    return results[0]


def _direct_score(sample: Mapping[str, Any], metric: str) -> float | None:
    value = (sample.get("ragas_scores") or {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def render_markdown(report: CalibrationReport) -> str:
    """Markdown table: one row per metric pair, plus runtimes."""
    lines = [
        f"Config `{report.config_name}`, MLflow run `{report.run_id}`, "
        f"{report.n_samples} samples. Δ = judge − direct over paired samples; "
        "failure = no score for a sample.",
        "",
        "| Direct RAGAS | MLflow scorer | Relation | Paired | Direct mean | Judge mean "
        "| Mean Δ | Mean \\|Δ\\| | Pearson | Spearman | Direct fail | Judge fail |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pair, s in report.pairs:
        lines.append(
            f"| {pair.direct} | {pair.judge} | {pair.relation} | {s.n_paired}/{s.n_samples} "
            f"| {_fmt(s.direct_mean)} | {_fmt(s.judge_mean)} | {_fmt(s.mean_diff, signed=True)} "
            f"| {_fmt(s.mean_abs_diff)} | {_fmt(s.pearson)} | {_fmt(s.spearman)} "
            f"| {s.direct_failure_rate:.0%} | {s.judge_failure_rate:.0%} |"
        )
    lines += [
        "",
        "| Runtime | Seconds | Per sample |",
        "|---|---|---|",
        _runtime_row(f"Direct RAGAS (`{DIRECT_RUNTIME_METRIC}`)", report.direct_seconds, report.n_samples),
        _runtime_row(f"MLflow scorers (`{RUNTIME_METRIC}`)", report.judge_seconds, report.n_samples),
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _runtime_row(label: str, seconds: float | None, n_samples: int) -> str:
    if seconds is None:
        return f"| {label} | – | – |"
    per_sample = seconds / n_samples if n_samples else None
    return f"| {label} | {seconds:.0f} | {_fmt(per_sample)} |"


def main(argv: Sequence[str] | None = None) -> CalibrationReport:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.reporting.judge_calibration",
        description="Compare direct RAGAS scores with post-run MLflow scorers (issue #46).",
    )
    parser.add_argument("results", type=Path, help="worker config JSON (results/runN/configs/*.json)")
    parser.add_argument("--run-id", required=True, help="MLflow config run that holds the judge assessments")
    parser.add_argument("--config", dest="config_name", help="config name if the JSON holds several")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. sqlite:///mlflow.db")
    parser.add_argument("--out", type=Path, help="write Markdown here and JSON next to it")
    args = parser.parse_args(argv)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    report = build_report(args.results, args.run_id, config_name=args.config_name)
    markdown = render_markdown(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
    print(markdown)
    return report


if __name__ == "__main__":
    main()
