from __future__ import annotations

from pathlib import Path

from benchmark.reporting.models import BenchmarkResultExtended, BenchmarkRun, collect_system_info
from benchmark.reporting.analysis import compute_rankings
from benchmark.reporting.terminal import display_report
from benchmark.reporting.exports import save_json_report


def generate_report(
    results: list[BenchmarkResultExtended],
    results_dir: Path = Path("results"),
    *,
    timestamp: str = "",
    dataset_name: str = "",
    dataset_subset: str = "",
    dataset_sample_size: int = 0,
    total_time: float = 0.0,
    show_terminal: bool = True,
    save_json: bool = True,
) -> None:
    rankings = compute_rankings(results)
    system_info = collect_system_info()

    if show_terminal:
        display_report(
            results,
            rankings,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            dataset_sample_size=dataset_sample_size,
            total_time=total_time,
            system_info=system_info,
        )

    results_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        run = BenchmarkRun(
            timestamp=timestamp,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            dataset_sample_size=dataset_sample_size,
            system_info=system_info,
            results=tuple(results),
        )
        path = save_json_report(run, results_dir)
        from rich.console import Console
        Console().print(f"[green]JSON saved to {path}[/green]")
