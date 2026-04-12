from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from benchmark.reporting.models import (
    BenchmarkResultExtended,
    BenchmarkRun,
    StatSummary,
)
from benchmark.reporting.analysis import RankTable


def _stats_to_dict(s: StatSummary | None) -> dict | None:
    if s is None:
        return None
    return {
        "mean": s.mean,
        "std": s.std,
        "min": s.min,
        "max": s.max,
        "median": s.median,
        "count": s.count,
    }


def _result_to_dict(r: BenchmarkResultExtended) -> dict:
    d = {}
    # Old fields (backward compatible keys)
    for key in (
        "config_name", "llm_model", "embedding_model", "prompt_template",
        "chunking_strategy",
        "chunk_size", "chunk_overlap", "num_chunks", "num_questions",
        "avg_ttft_seconds", "avg_tokens_per_second",
        "avg_gpu_utilization_pct", "avg_gpu_memory_used_mb",
        "ragas_faithfulness", "ragas_answer_relevancy", "ragas_answer_correctness",
        "ragas_context_precision", "ragas_context_recall",
        "total_time_seconds",
    ):
        val = getattr(r, key)
        d[key] = val

    # Stats summary
    d["stats"] = {
        "ttft_seconds": _stats_to_dict(r.ttft_stats),
        "tokens_per_second": _stats_to_dict(r.tps_stats),
        "gpu_utilization_pct": _stats_to_dict(r.gpu_util_stats),
        "gpu_memory_mb": _stats_to_dict(r.gpu_mem_stats),
        "ragas_faithfulness": _stats_to_dict(r.ragas_faithfulness_stats),
        "ragas_answer_relevancy": _stats_to_dict(r.ragas_answer_relevancy_stats),
        "ragas_answer_correctness": _stats_to_dict(r.ragas_answer_correctness_stats),
        "ragas_context_precision": _stats_to_dict(r.ragas_context_precision_stats),
        "ragas_context_recall": _stats_to_dict(r.ragas_context_recall_stats),
    }

    # Per-sample data
    d["per_sample"] = [
        {
            "question": s.question,
            "ground_truth": s.ground_truth,
            "answer": s.answer,
            "contexts": list(s.contexts),
            "ttft_seconds": s.ttft_seconds,
            "total_seconds": s.total_seconds,
            "token_count": s.token_count,
            "tokens_per_second": s.tokens_per_second,
            "gpu_usage": s.gpu_usage,
            "ragas_scores": s.ragas_scores,
        }
        for s in r.per_sample
    ]

    return d


def save_json_report(run: BenchmarkRun, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run.timestamp
    filepath = results_dir / f"benchmark_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "num_configs": len(run.results),
        "dataset": {
            "name": run.dataset_name,
            "subset": run.dataset_subset,
            "sample_size": run.dataset_sample_size,
        },
        "system_info": run.system_info,
        "results": [_result_to_dict(r) for r in run.results],
    }

    filepath.write_text(json.dumps(output, indent=2, default=str))
    return filepath


def save_csv_report(
    results: list[BenchmarkResultExtended], results_dir: Path
) -> tuple[Path, Path | None]:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV: one row per config
    summary_rows: list[dict] = []
    for r in results:
        row = {
            "config_name": r.config_name,
            "llm_model": r.llm_model,
            "embedding_model": r.embedding_model,
            "prompt_template": r.prompt_template,
            "chunking_strategy": r.chunking_strategy,
            "chunk_size": r.chunk_size,
            "chunk_overlap": r.chunk_overlap,
            "num_chunks": r.num_chunks,
            "num_questions": r.num_questions,
            "avg_ttft_seconds": r.avg_ttft_seconds,
            "avg_tokens_per_second": r.avg_tokens_per_second,
            "avg_gpu_utilization_pct": r.avg_gpu_utilization_pct,
            "avg_gpu_memory_used_mb": r.avg_gpu_memory_used_mb,
            "ragas_faithfulness": r.ragas_faithfulness,
            "ragas_answer_relevancy": r.ragas_answer_relevancy,
            "ragas_answer_correctness": r.ragas_answer_correctness,
            "ragas_context_precision": r.ragas_context_precision,
            "ragas_context_recall": r.ragas_context_recall,
            "total_time_seconds": r.total_time_seconds,
        }
        # Add stat summaries
        for metric, stats in (
            ("ttft", r.ttft_stats),
            ("tps", r.tps_stats),
            ("gpu_util", r.gpu_util_stats),
            ("gpu_mem", r.gpu_mem_stats),
            ("faithfulness", r.ragas_faithfulness_stats),
            ("answer_relevancy", r.ragas_answer_relevancy_stats),
            ("answer_correctness", r.ragas_answer_correctness_stats),
            ("context_precision", r.ragas_context_precision_stats),
            ("context_recall", r.ragas_context_recall_stats),
        ):
            if stats:
                row[f"{metric}_std"] = stats.std
                row[f"{metric}_min"] = stats.min
                row[f"{metric}_max"] = stats.max
                row[f"{metric}_median"] = stats.median
        summary_rows.append(row)

    summary_path = results_dir / "results_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # Per-sample CSV
    sample_rows: list[dict] = []
    for r in results:
        for s in r.per_sample:
            sample_rows.append({
                "config_name": r.config_name,
                "question": s.question[:200],
                "ground_truth": s.ground_truth[:200],
                "answer": s.answer[:500],
                "ttft_seconds": s.ttft_seconds,
                "total_seconds": s.total_seconds,
                "token_count": s.token_count,
                "tokens_per_second": s.tokens_per_second,
                "faithfulness": s.ragas_scores.get("faithfulness"),
                "answer_relevancy": s.ragas_scores.get("answer_relevancy"),
                "answer_correctness": s.ragas_scores.get("answer_correctness"),
                "context_precision": s.ragas_scores.get("context_precision"),
                "context_recall": s.ragas_scores.get("context_recall"),
            })

    sample_path: Path | None = None
    if sample_rows:
        sample_path = results_dir / "results_per_sample.csv"
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False)

    return summary_path, sample_path


def save_markdown_report(
    results: list[BenchmarkResultExtended],
    rankings: RankTable,
    results_dir: Path,
    timestamp: str = "",
    dataset_name: str = "",
    dataset_subset: str = "",
    dataset_sample_size: int = 0,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    filepath = results_dir / "report.md"

    lines: list[str] = []
    lines.append("# RAG Benchmark Report")
    lines.append("")
    if timestamp:
        lines.append(f"**Date:** {timestamp}")
    if dataset_name or dataset_subset:
        label = dataset_name or "unknown"
        if dataset_subset:
            label += f"/{dataset_subset}"
        lines.append(f"**Dataset:** {label} ({dataset_sample_size} samples)")
        lines.append(f"**Dataset:** {dataset_subset} ({dataset_sample_size} samples)")
    lines.append(f"**Configurations:** {len(results)}")
    lines.append("")

    # Summary table
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Config | TTFT (s) | Tok/s | GPU % | Total (s) |")
    lines.append("|--------|----------|-------|-------|-----------|")
    for r in results:
        ttft = _fmt_md(r.avg_ttft_seconds)
        tps = _fmt_md(r.avg_tokens_per_second, ".1f")
        gpu = _fmt_md(r.avg_gpu_utilization_pct, ".1f")
        total = _fmt_md(r.total_time_seconds, ".1f")
        lines.append(f"| {_short(r.config_name)} | {ttft} | {tps} | {gpu} | {total} |")
    lines.append("")

    # RAGAS table
    lines.append("## RAGAS Scores")
    lines.append("")
    lines.append("| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |")
    lines.append("|--------|-------------|-------------|--------------|---------------|------------|")
    for r in results:
        f = _fmt_md(r.ragas_faithfulness)
        ar = _fmt_md(r.ragas_answer_relevancy)
        ac = _fmt_md(r.ragas_answer_correctness)
        cp = _fmt_md(r.ragas_context_precision)
        cr = _fmt_md(r.ragas_context_recall)
        lines.append(f"| {_short(r.config_name)} | {f} | {ar} | {ac} | {cp} | {cr} |")
    lines.append("")

    # Insights
    if rankings.insights:
        lines.append("## Insights")
        lines.append("")
        for insight in rankings.insights:
            lines.append(f"- {insight}")
        lines.append("")

    # Per-config stats
    lines.append("## Detailed Statistics")
    lines.append("")
    for r in results:
        lines.append(f"### {_short(r.config_name)}")
        lines.append("")
        if r.ttft_stats:
            lines.append(f"- TTFT: {r.ttft_stats.mean:.3f}s +/- {r.ttft_stats.std:.3f}s (range: {r.ttft_stats.min:.3f}-{r.ttft_stats.max:.3f})")
        if r.tps_stats:
            lines.append(f"- Throughput: {r.tps_stats.mean:.1f} tok/s +/- {r.tps_stats.std:.1f}")
        for label, stats in (
            ("Faithfulness", r.ragas_faithfulness_stats),
            ("Answer Relevancy", r.ragas_answer_relevancy_stats),
            ("Answer Correctness", r.ragas_answer_correctness_stats),
            ("Context Precision", r.ragas_context_precision_stats),
            ("Context Recall", r.ragas_context_recall_stats),
        ):
            if stats:
                lines.append(f"- {label}: {stats.mean:.3f} +/- {stats.std:.3f} (range: {stats.min:.3f}-{stats.max:.3f})")
        lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath


def _fmt_md(val: float | None, fmt: str = ".3f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _short(name: str, max_len: int = 30) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."
