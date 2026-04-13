#!/usr/bin/env python3
"""Compare benchmark runs side-by-side.

Usage:
    python compare_runs.py results/run12_RTX_No-Think results/run15_SPARK_GPToss:20b
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _load_run(run_dir: Path) -> dict | None:
    """Load the benchmark JSON from a run directory."""
    json_files = sorted(run_dir.glob("benchmark_*.json"))
    if not json_files:
        return None
    with open(json_files[0]) as f:
        return json.load(f)


def _get_run_label(run_dir: Path) -> str:
    """Extract a readable label from the run directory name."""
    return run_dir.name


def compare_runs(run_dirs: list[Path]) -> None:
    console = Console()

    runs: list[tuple[str, dict]] = []
    for rd in run_dirs:
        if not rd.is_dir():
            console.print(f"[red]Not a directory: {rd}[/red]")
            continue
        data = _load_run(rd)
        if data is None:
            console.print(f"[red]No benchmark_*.json found in {rd}[/red]")
            continue
        label = _get_run_label(rd)
        runs.append((label, data))

    if len(runs) < 2:
        console.print("[red]Need at least 2 valid run directories to compare.[/red]")
        sys.exit(1)

    # ── System info table ──
    sys_table = Table(title="System Info", show_header=True)
    sys_table.add_column("Property", style="bold")
    for label, _ in runs:
        sys_table.add_column(label)

    all_keys: list[str] = []
    for _, data in runs:
        for k in data.get("system_info", {}):
            if k not in all_keys:
                all_keys.append(k)
    for key in all_keys:
        row = [key]
        for _, data in runs:
            row.append(str(data.get("system_info", {}).get(key, "-")))
        sys_table.add_row(*row)
    console.print(sys_table)
    console.print()

    # ── RAGAS comparison table ──
    ragas_metrics = [
        ("Faithfulness", "ragas_faithfulness"),
        ("Answer Relevancy", "ragas_answer_relevancy"),
        ("Answer Correctness", "ragas_answer_correctness"),
        ("Context Precision", "ragas_context_precision"),
        ("Context Recall", "ragas_context_recall"),
    ]

    ragas_table = Table(title="RAGAS Scores", show_header=True)
    ragas_table.add_column("Metric", style="bold")
    for label, _ in runs:
        ragas_table.add_column(label, justify="center")

    # Find best score per metric
    for metric_label, metric_key in ragas_metrics:
        row = [metric_label]
        values: list[float | None] = []
        for _, data in runs:
            results = data.get("results", [])
            if results:
                val = results[0].get(metric_key)
                values.append(val)
            else:
                values.append(None)

        best_val = max((v for v in values if v is not None), default=None)

        for v in values:
            if v is None:
                row.append("[dim]N/A[/dim]")
            elif best_val is not None and v == best_val:
                row.append(f"[bold green]{v:.3f}[/bold green]")
            else:
                diff = abs(v - best_val) if best_val else 0
                row.append(f"{v:.3f}")
        ragas_table.add_row(*row)

    console.print(ragas_table)
    console.print()

    # ── Performance comparison table ──
    perf_metrics = [
        ("TTFT (s)", "avg_ttft_seconds", "lower"),
        ("Tokens/s", "avg_tokens_per_second", "higher"),
        ("GPU %", "avg_gpu_utilization_pct", "higher"),
        ("Total (s)", "total_time_seconds", "lower"),
    ]

    perf_table = Table(title="Performance", show_header=True)
    perf_table.add_column("Metric", style="bold")
    for label, _ in runs:
        perf_table.add_column(label, justify="center")

    for metric_label, metric_key, direction in perf_metrics:
        row = [metric_label]
        values: list[float | None] = []
        for _, data in runs:
            results = data.get("results", [])
            if results:
                val = results[0].get(metric_key)
                values.append(val)
            else:
                values.append(None)

        valid_vals = [v for v in values if v is not None]
        best_val = min(valid_vals) if direction == "lower" and valid_vals else (
            max(valid_vals) if direction == "higher" and valid_vals else None
        )

        for v in values:
            if v is None:
                row.append("[dim]N/A[/dim]")
            elif best_val is not None and v == best_val:
                row.append(f"[bold green]{v:.3f}[/bold green]")
            else:
                row.append(f"{v:.3f}")
        perf_table.add_row(*row)

    console.print(perf_table)
    console.print()

    # ── Config comparison table ──
    config_fields = [
        ("LLM Model", "llm_model"),
        ("Embedding", "embedding_model"),
        ("Template", "prompt_template"),
        ("Chunking", "chunking_strategy"),
        ("Chunk Size", "chunk_size"),
        ("Chunk Overlap", "chunk_overlap"),
        ("Chunks", "num_chunks"),
        ("Questions", "num_questions"),
    ]

    config_table = Table(title="Configuration", show_header=True)
    config_table.add_column("Property", style="bold")
    for label, _ in runs:
        config_table.add_column(label)
    for field_label, field_key in config_fields:
        row = [field_label]
        for _, data in runs:
            results = data.get("results", [])
            val = results[0].get(field_key, "-") if results else "-"
            row.append(str(val))
        config_table.add_row(*row)

    console.print(config_table)
    console.print()

    # ── Per-sample worst cases ──
    worst_table = Table(title="Worst Answer Correctness (bottom 5)", show_header=True)
    worst_table.add_column("Question", max_width=60)
    for label, _ in runs:
        worst_table.add_column(label, justify="center")

    # Collect all questions across runs
    all_questions: list[str] = []
    question_scores: dict[str, dict[str, float | None]] = {}
    for label, data in runs:
        results = data.get("results", [])
        if not results:
            continue
        for sample in results[0].get("per_sample", []):
            q = sample.get("question", "")
            if q not in question_scores:
                question_scores[q] = {}
                all_questions.append(q)
            score = sample.get("ragas_scores", {}).get("answer_correctness")
            question_scores[q][label] = score

    # Sort by average score across runs (worst first)
    scored = []
    for q in all_questions:
        vals = [v for v in question_scores[q].values() if v is not None]
        avg = sum(vals) / len(vals) if vals else 0
        scored.append((q, avg))
    scored.sort(key=lambda x: x[1])

    for q, _ in scored[:5]:
        row = [q[:60]]
        for label, _ in runs:
            val = question_scores[q].get(label)
            if val is None:
                row.append("[dim]N/A[/dim]")
            elif val < 0.2:
                row.append(f"[red]{val:.3f}[/red]")
            elif val < 0.5:
                row.append(f"[yellow]{val:.3f}[/yellow]")
            else:
                row.append(f"[green]{val:.3f}[/green]")
        worst_table.add_row(*row)

    console.print(worst_table)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dirs = [Path(arg) for arg in sys.argv[1:]]
    compare_runs(run_dirs)


if __name__ == "__main__":
    main()
