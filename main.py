"""Thin CLI entry point for the benchmark framework.

The orchestration loop lives in ``benchmark.orchestration.worker``; this
script only resolves the experiment manifest, prepares tracking, and
delegates the config matrix to the worker.

Usage:
    python main.py experiments/<name>.yaml
    BENCHMARK_CONFIG_FILE=experiments/<name>.yaml python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rich.console import Console

from benchmark.orchestration.matrix import (
    build_configs_from_spec,
    load_experiment_spec,
)
from benchmark.orchestration.runner import (  # noqa: F401  (re-export)
    run_single_benchmark,
)
from benchmark.orchestration.worker import ExperimentWorker, WorkerOptions
from benchmark.reporting.models import BenchmarkResultExtended
from benchmark.tracking import setup_mlflow
from benchmark.tracing import setup_tracing

console = Console()

MANIFEST_HINT = (
    "No experiment manifest configured. Set "
    "BENCHMARK_CONFIG_FILE=experiments/<name>.yaml or pass one explicitly: "
    "python main.py experiments/<name>.yaml"
)


def resolve_manifest_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    configured = os.getenv("BENCHMARK_CONFIG_FILE") or os.getenv(
        "EXPERIMENT_MANIFEST"
    )
    return Path(configured) if configured else None


def main(argv: list[str] | None = None) -> list[BenchmarkResultExtended]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a benchmark experiment manifest through the resumable "
            "worker loop."
        ),
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        help=(
            "Experiment manifest (JSON/YAML). Defaults to "
            "BENCHMARK_CONFIG_FILE."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = resolve_manifest_path(args.manifest)
    if manifest_path is None:
        console.print(f"[red]{MANIFEST_HINT}[/red]")
        raise SystemExit(2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    console.print("[bold]RAG Benchmarking Framework[/bold]")
    console.print("=" * 50)

    configs = build_configs_from_spec(load_experiment_spec(manifest_path))
    tracking_uri = setup_tracing()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    if not args.no_mlflow:
        setup_mlflow()

    worker = ExperimentWorker(
        configs,
        WorkerOptions(
            run_dir=args.run_dir,
            experiment_name=manifest_path.stem,
            write_reports=not args.no_reports,
            log_mlflow=not args.no_mlflow,
        ),
    )
    return worker.run()


if __name__ == "__main__":
    main()
