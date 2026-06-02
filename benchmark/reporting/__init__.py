from __future__ import annotations

from pathlib import Path

from benchmark.reporting.models import BenchmarkResultExtended, BenchmarkRun, collect_system_info
from benchmark.reporting.analysis import compute_rankings
from benchmark.reporting.terminal import display_report
from benchmark.reporting.exports import save_json_report, save_csv_report, save_markdown_report
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
    save_csv: bool = True,
    save_markdown: bool = True,
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

    if save_csv:
        summary_path, sample_path = save_csv_report(results, results_dir)
        from rich.console import Console
        c = Console()
        c.print(f"[green]CSV summary saved to {summary_path}[/green]")
        if sample_path:
            c.print(f"[green]CSV per-sample saved to {sample_path}[/green]")

    if save_markdown:
        path = save_markdown_report(
            results, rankings, results_dir,
            timestamp=timestamp,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            dataset_sample_size=dataset_sample_size,
        )
        from rich.console import Console
        Console().print(f"[green]Markdown report saved to {path}[/green]")

    if save_plots:
        paths = generate_plots(results, rankings, results_dir)
        if paths:
            from rich.console import Console
            c = Console()
            c.print(f"[green]Generated {len(paths)} plot(s) in {results_dir / 'results_plots'}[/green]")
