"""Benchmark runner node: executes the full pipeline for a proposed config."""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console

from agentic.state import AgentState

console = Console()
logger = logging.getLogger(__name__)


def benchmark_runner_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: execute the benchmark for current_config."""
    from agentic.tools import run_full_pipeline
    from agentic.state import ExplorationResult

    config = state.get("current_config")
    if config is None:
        logger.error("No current_config in state")
        return {
            "current_result": ExplorationResult(
                config={}, metrics={}, total_time_seconds=0,
                num_chunks=0, error="No config provided",
            ),
            "phase": "analyze",
        }

    console.print(f"  [dim]Running pipeline for config...[/dim]")

    full_result, exploration_result = run_full_pipeline(config, state)

    completed = list(state.get("completed_runs", []))
    completed.append(exploration_result)

    return {
        "current_result": exploration_result,
        "completed_runs": completed,
        "phase": "analyze",
    }
