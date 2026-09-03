"""Retroactive paired statistical comparison across separate benchmark runs.

Compares configs from multiple ``results/runN`` directories that answered
the same dataset, including historical runs that predate the in-run
comparison report:

    python -m benchmark.compare results/run3 results/run5 --baseline internal
    python -m benchmark.compare results/run7 --baseline mcp --out /tmp/cmp

Requires each run dir to contain a ``benchmark_*.json`` report with
per-sample scores (written by every standard sweep). Question texts are
checked for alignment before any statistics are computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console

from benchmark.reporting.comparisons import write_comparisons_payload
from benchmark.statistics import compare_per_sample

console = Console()


def _latest_report(run_dir: Path) -> dict[str, Any]:
    reports = sorted(run_dir.glob("benchmark_*.json"))
    if not reports:
        raise FileNotFoundError(f"no benchmark_*.json report in {run_dir}")
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def collect_configs(
    run_dirs: list[Path],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load runs, verify dataset parity, return (dataset_label, configs by name)."""
    configs: dict[str, dict[str, Any]] = {}
    dataset_labels: set[str] = set()
    for run_dir in run_dirs:
        report = _latest_report(run_dir)
        label = (
            f"{report.get('dataset_name', '?')}/"
            f"{report.get('dataset_subset') or '-'}"
            f"@{report.get('dataset_sample_size', '?')}"
        )
        dataset_labels.add(label)
        for result in report.get("results", []):
            name = result.get("config_name")
            if not name:
                continue
            if name in configs:
                console.print(
                    f"[yellow]Duplicate config {name!r} across runs; "
                    "keeping the first occurrence.[/yellow]"
                )
                continue
            configs[name] = result
    if len(dataset_labels) > 1:
        raise ValueError(
            "runs used different datasets: " + ", ".join(sorted(dataset_labels))
        )
    (dataset_label,) = dataset_labels
    return dataset_label, configs


def _score_rows(per_sample: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for sample in per_sample:
        row = sample.get("ragas_scores", {}) | sample.get("custom_scores", {})
        diagnostics = sample.get("adapter_diagnostics") or {}
        if diagnostics.get("execution_mode") == "agentic":
            tokens = diagnostics.get("agent_tokens_total")
            if tokens is not None:
                row["agent_tokens_total"] = float(tokens)
            rounds = diagnostics.get("agent_rounds")
            if rounds is not None:
                row["agent_rounds"] = float(rounds)
        rows.append(row)
    return rows


def cross_run_comparisons(
    run_dirs: list[Path],
    *,
    baseline: str | None = None,
    n_resamples: int = 10_000,
) -> list[dict]:
    """Compare every config against ``baseline`` (default: first by name)."""
    dataset_label, configs = collect_configs(run_dirs)
    if len(configs) < 2:
        raise ValueError(
            f"need at least 2 distinct configs across runs, got {sorted(configs)}"
        )
    names = sorted(configs)
    baseline_name = baseline or names[0]
    if baseline_name not in configs:
        raise ValueError(
            f"baseline {baseline_name!r} not among configs: {', '.join(names)}"
        )
    base = configs[baseline_name]
    base_rows = _score_rows(base["per_sample"])
    base_questions = [s.get("question", "") for s in base["per_sample"]]

    payload: list[dict] = []
    for name in names:
        if name == baseline_name:
            continue
        other = configs[name]
        other_rows = _score_rows(other["per_sample"])
        other_questions = [s.get("question", "") for s in other["per_sample"]]
        try:
            comparisons = compare_per_sample(
                base_rows,
                other_rows,
                n_resamples=n_resamples,
                questions_a=base_questions,
                questions_b=other_questions,
            )
        except ValueError as exc:
            payload.append(
                {
                    "baseline": baseline_name,
                    "config": name,
                    "paired_samples": 0,
                    "error": str(exc),
                    "comparisons": [],
                }
            )
            continue
        payload.append(
            {
                "baseline": baseline_name,
                "config": name,
                "paired_samples": min(len(base_rows), len(other_rows)),
                "comparisons": [c.as_dict() for c in comparisons],
            }
        )
    console.print(f"Dataset: {dataset_label}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap comparison of configs across separate "
            "benchmark run directories."
        ),
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--baseline",
        help="config_name to compare against (default: first config by name)",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10_000,
        help="bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: first run directory)",
    )
    args = parser.parse_args()

    payload = cross_run_comparisons(
        args.run_dirs, baseline=args.baseline, n_resamples=args.n_resamples
    )
    out_dir = args.out or args.run_dirs[0]
    md_path = write_comparisons_payload(payload, out_dir)
    console.print(f"[green]Comparison report saved to {md_path}[/green]")


if __name__ == "__main__":
    main()
