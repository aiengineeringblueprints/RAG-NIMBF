from __future__ import annotations

import math

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from benchmark.reporting.models import BenchmarkResultExtended, StatSummary
from benchmark.reporting.analysis import RankTable

console = Console()


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    return f"{val:{fmt}}"


def _fmt_with_std(mean: float | None, stats: StatSummary | None, fmt: str = ".3f") -> str:
    if mean is None:
        return "N/A"
    text = f"{mean:{fmt}}"
    if stats and stats.count > 1:
        text += f" +/- {stats.std:{fmt}}"
    return text


def _score_style(value: float | None, best: float | None, worst: float | None) -> str:
    if value is None:
        return "dim"
    if best is None or worst is None:
        return ""
    if best == worst:
        return ""
    if value == best:
        return "bold green"
    if value == worst:
        return "bold red"
    return ""


def _ragas_color(value: float | None) -> str:
    if value is None:
        return "dim"
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


def _make_sparkline(value: float | None, max_val: float, width: int = 10) -> str:
    if value is None or max_val <= 0:
        return "?" * width
    filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _short_name(config_name: str, max_len: int = 35) -> str:
    if len(config_name) <= max_len:
        return config_name
    return config_name[: max_len - 3] + "..."


def display_report(
    results: list[BenchmarkResultExtended],
    rankings: RankTable,
    dataset_subset: str = "",
    dataset_sample_size: int = 0,
    total_time: float = 0.0,
    system_info: dict | None = None,
) -> None:
    console.print()

    # --- Summary Panel ---
    _display_summary_panel(results, dataset_subset, dataset_sample_size, total_time, system_info)

    # --- Performance Metrics Table ---
    _display_performance_table(results, rankings)

    # --- RAGAS Scores Table ---
    _display_ragas_table(results, rankings)

    # --- Sparkline Comparison ---
    _display_sparklines(results)

    # --- Insights ---
    _display_insights(rankings)

    console.print()


def _display_summary_panel(
    results: list[BenchmarkResultExtended],
    dataset_subset: str,
    dataset_sample_size: int,
    total_time: float,
    system_info: dict | None,
) -> None:
    parts: list[str] = []
    parts.append(f"[bold]Configurations:[/bold] {len(results)}")
    if dataset_subset:
        parts.append(f"[bold]Dataset:[/bold] {dataset_subset} ({dataset_sample_size} samples)")
    if total_time > 0:
        parts.append(f"[bold]Total wall time:[/bold] {total_time:.1f}s")
    if system_info:
        gpu = system_info.get("gpu", "")
        if gpu:
            parts.append(f"[bold]GPU:[/bold] {gpu}")
        parts.append(f"[bold]Python:[/bold] {system_info.get('python', '?')}")

    console.print(Panel("\n".join(parts), title="Benchmark Summary", border_style="blue"))


def _display_performance_table(results: list[BenchmarkResultExtended], rankings: RankTable) -> None:
    table = Table(title="Performance Metrics", show_lines=True)
    table.add_column("Rank", justify="right", style="dim", width=4)
    table.add_column("Config", style="cyan", max_width=35)
    table.add_column("TTFT (s)", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("GPU %", justify="right")
    table.add_column("GPU MB", justify="right")
    table.add_column("Total (s)", justify="right")

    # Pre-compute best/worst for numeric columns
    ttft_vals = [r.avg_ttft_seconds for r in results if r.avg_ttft_seconds > 0]
    tps_vals = [r.avg_tokens_per_second for r in results if r.avg_tokens_per_second > 0]
    ttft_best, ttft_worst = (min(ttft_vals), max(ttft_vals)) if ttft_vals else (None, None)
    tps_best, tps_worst = (max(tps_vals), min(tps_vals)) if tps_vals else (None, None)

    rank_map = {cr.config_name: cr.rank for cr in rankings.ranks}

    for r in results:
        rank_str = str(rank_map.get(r.config_name, "-"))

        ttft_text = _fmt_with_std(r.avg_ttft_seconds, r.ttft_stats, ".3f")
        tps_text = _fmt_with_std(r.avg_tokens_per_second, r.tps_stats, ".1f")
        gpu_pct_text = _fmt(r.avg_gpu_utilization_pct, ".1f")
        gpu_mb_text = _fmt(r.avg_gpu_memory_used_mb, ".0f")
        total_text = _fmt(r.total_time_seconds, ".1f")

        row_style = ""
        if rankings.ranks and rankings.ranks[0].config_name == r.config_name:
            row_style = "on grey15"

        table.add_row(
            rank_str,
            _short_name(r.config_name),
            Text(ttft_text, style=_score_style(r.avg_ttft_seconds, ttft_best, ttft_worst)),
            Text(tps_text, style=_score_style(r.avg_tokens_per_second, tps_best, tps_worst)),
            gpu_pct_text,
            gpu_mb_text,
            total_text,
            style=row_style or None,
        )

    console.print(table)


def _display_ragas_table(results: list[BenchmarkResultExtended], rankings: RankTable) -> None:
    table = Table(title="RAGAS Evaluation Scores", show_lines=True)
    table.add_column("Rank", justify="right", style="dim", width=4)
    table.add_column("Config", style="cyan", max_width=35)
    table.add_column("Faithfulness", justify="right")
    table.add_column("Answer Rel.", justify="right")
    table.add_column("Ctx Precision", justify="right")
    table.add_column("Ctx Recall", justify="right")

    rank_map = {cr.config_name: cr.rank for cr in rankings.ranks}

    for r in results:
        rank_str = str(rank_map.get(r.config_name, "-"))

        def _ragas_cell(val: float | None) -> Text:
            text = _fmt(val)
            return Text(text, style=_ragas_color(val))

        row_style = ""
        if rankings.ranks and rankings.ranks[0].config_name == r.config_name:
            row_style = "on grey15"

        table.add_row(
            rank_str,
            _short_name(r.config_name),
            _ragas_cell(r.ragas_faithfulness),
            _ragas_cell(r.ragas_answer_relevancy),
            _ragas_cell(r.ragas_context_precision),
            _ragas_cell(r.ragas_context_recall),
            style=row_style or None,
        )

    console.print(table)


def _display_sparklines(results: list[BenchmarkResultExtended]) -> None:
    if not results:
        return

    metrics = [
        ("Faithfulness", "ragas_faithfulness"),
        ("Answer Rel.", "ragas_answer_relevancy"),
        ("Ctx Prec.", "ragas_context_precision"),
        ("Ctx Recall", "ragas_context_recall"),
    ]

    # Compute max per metric for scaling
    maxima: dict[str, float] = {}
    for _, attr in metrics:
        vals = [getattr(r, attr) for r in results]
        clean = [v for v in vals if v is not None]
        maxima[attr] = max(clean) if clean else 1.0

    console.print("\n[bold]Quick Comparison[/bold]")
    for r in results:
        parts: list[str] = []
        for label, attr in metrics:
            val = getattr(r, attr)
            bar = _make_sparkline(val, maxima[attr], width=8)
            parts.append(f"{bar} {_fmt(val)}")
        line = f"  {_short_name(r.config_name, 30):<30}  {'  '.join(parts)}"
        console.print(line)


def _display_insights(rankings: RankTable) -> None:
    if not rankings.insights:
        return

    console.print()
    md_text = "### Insights\n\n" + "\n\n".join(f"- {i}" for i in rankings.insights)
    console.print(Panel(Markdown(md_text), title="Analysis", border_style="green"))
