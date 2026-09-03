"""Paired statistical comparison between benchmark configs.

Compares every config against the first (baseline) config using per-sample
RAGAS and custom metric scores. Pairing is positional: both configs answer
the same dataset in the same order, so question difficulty cancels out.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from benchmark.reporting.models import BenchmarkResultExtended
from benchmark.statistics import MetricComparison, compare_per_sample

_MIN_RELIABILITY_NOTE = (
    "reliable = 95% bootstrap CI of the paired delta excludes 0 AND "
    "permutation p < 0.05"
)


def _comparison_rows(per_sample) -> list[dict]:
    """Merge ragas + custom scores and per-sample agent token counts.

    Agent token counts come from the adapter diagnostics, so token deltas
    between an agentic config and the baseline get the same bootstrap CI and
    real/noise verdict as quality metrics.
    """
    rows: list[dict] = []
    for sample in per_sample:
        row = sample.ragas_scores | sample.custom_scores
        diagnostics = sample.adapter_diagnostics or {}
        if diagnostics.get("execution_mode") == "agentic":
            tokens = diagnostics.get("agent_tokens_total")
            if tokens is not None:
                row["agent_tokens_total"] = float(tokens)
            rounds = diagnostics.get("agent_rounds")
            if rounds is not None:
                row["agent_rounds"] = float(rounds)
            tool_records = diagnostics.get("tool_call_records")
            if tool_records:
                from benchmark.agentic.metrics import (
                    ToolCallRecord,
                    compute_tool_call_metrics,
                )

                tool_metrics = compute_tool_call_metrics(
                    [ToolCallRecord.from_dict(entry) for entry in tool_records]
                )
                for key in (
                    "tool_calls_total",
                    "invalid_call_count",
                    "redundant_call_count",
                    "tool_seconds_total",
                ):
                    row[key] = float(tool_metrics[key])
        rows.append(row)
    return rows


def config_comparisons(
    results: list[BenchmarkResultExtended],
    *,
    n_resamples: int = 10_000,
) -> list[dict]:
    """Compare each config against the baseline (first) config."""
    if len(results) < 2:
        return []
    baseline = results[0]
    baseline_rows = _comparison_rows(baseline.per_sample)
    baseline_questions = [s.question for s in baseline.per_sample]
    output: list[dict] = []
    for other in results[1:]:
        other_rows = _comparison_rows(other.per_sample)
        other_questions = [s.question for s in other.per_sample]
        try:
            comparisons = compare_per_sample(
                baseline_rows,
                other_rows,
                n_resamples=n_resamples,
                questions_a=baseline_questions,
                questions_b=other_questions,
            )
        except ValueError as exc:
            output.append(
                {
                    "baseline": baseline.config_name,
                    "config": other.config_name,
                    "paired_samples": 0,
                    "error": str(exc),
                    "comparisons": [],
                }
            )
            continue
        output.append(
            {
                "baseline": baseline.config_name,
                "config": other.config_name,
                "paired_samples": min(len(baseline.per_sample), len(other.per_sample)),
                "comparisons": [c.as_dict() for c in comparisons],
            }
        )
    return output


def _format_comparison(c: MetricComparison) -> str:
    sign = "+" if c.delta >= 0 else ""
    verdict = "real" if c.reliable else "noise"
    return (
        f"| {c.metric} | {c.mean_b:.3f} | {sign}{c.delta:.3f} "
        f"| [{c.ci_low:.3f}, {c.ci_high:.3f}] | {c.p_value:.3f} "
        f"| {c.effect_size:+.2f} | {verdict} |"
    )


def render_comparisons_markdown(payload: list[dict]) -> str:
    lines = [
        "# Statistical Config Comparison",
        "",
        f"Baseline: `{payload[0]['baseline']}`. Delta = config − baseline.",
        f"{_MIN_RELIABILITY_NOTE}.",
        "",
    ]
    for entry in payload:
        lines.append(f"## `{entry['config']}` vs `{entry['baseline']}`")
        lines.append("")
        if entry.get("error"):
            lines.append(f"Skipped: {entry['error']}")
            lines.append("")
            continue
        lines.append(
            "| metric | baseline mean | delta | 95% CI | p | Cohen's d | verdict |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for c in entry["comparisons"]:
            lines.append(
                _format_comparison(
                    MetricComparison(
                        metric=c["metric"],
                        mean_a=c["mean_a"],
                        mean_b=c["mean_b"],
                        delta=c["delta"],
                        ci_low=c["ci_low"],
                        ci_high=c["ci_high"],
                        p_value=c["p_value"],
                        effect_size=c["effect_size"],
                    )
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_comparisons_payload(
    payload: list[dict],
    results_dir: Path,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "comparisons.json"
    json_path.write_text(
        json.dumps(
            {"method": "paired bootstrap + permutation", "note": _MIN_RELIABILITY_NOTE, "comparisons": payload},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path = results_dir / "comparisons.md"
    md_path.write_text(render_comparisons_markdown(payload), encoding="utf-8")
    return md_path


def save_comparisons_report(
    results: list[BenchmarkResultExtended],
    results_dir: Path,
    *,
    n_resamples: int = 10_000,
) -> Path | None:
    """Write ``comparisons.json`` and ``comparisons.md`` for a sweep."""
    payload = config_comparisons(results, n_resamples=n_resamples)
    if not payload:
        return None
    md_path = write_comparisons_payload(payload, results_dir)
    Console().print(f"[green]Statistical comparison saved to {md_path}[/green]")
    return md_path
