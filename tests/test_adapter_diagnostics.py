"""Uniform adapter diagnostic summary contract.

The orchestrator must not read adapter-internal diagnostic keys. Instead it
consumes a generic summary through the adapter interface:

- ``adapter_stage_timings`` — per-sample stage-timing contributions as
  ``{stage_name: seconds}`` floats, resolved via an adapter's optional
  ``diagnostic_stage_timings()`` method (falling back to a generic
  ``stage_timings`` entry inside the sample diagnostics).
- ``adapter_aggregate_metrics`` — run-level adapter metrics via an optional
  ``aggregate_metrics()`` method; adapters without one contribute nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from benchmark.adapters.lifecycle import (
    adapter_aggregate_metrics,
    adapter_stage_timings,
)
from benchmark.adapters.mcp import McpRagAdapter


class _TimingAdapter:
    name = "timing"

    def diagnostic_stage_timings(self, diagnostics: dict[str, Any]):
        return {"custom_stage": diagnostics["x"]}


class _PlainAdapter:
    name = "plain"


def test_stage_timings_use_adapter_method_when_available():
    adapter = _TimingAdapter()
    timings = adapter_stage_timings(adapter, {"x": 1.25})
    assert timings == {"custom_stage": 1.25}


def test_stage_timings_fall_back_to_generic_diagnostics_entry():
    adapter = _PlainAdapter()
    diagnostics = {"stage_timings": {"a": 0.5, "b": 2}}
    assert adapter_stage_timings(adapter, diagnostics) == {"a": 0.5, "b": 2.0}


def test_stage_timings_default_to_empty_without_entries():
    assert adapter_stage_timings(_PlainAdapter(), {}) == {}


def test_aggregate_metrics_return_none_for_adapters_without_policy():
    assert adapter_aggregate_metrics(_PlainAdapter(), [{}]) is None


def _mcp_adapter() -> McpRagAdapter:
    @dataclass
    class FakeRunner:
        tools: tuple = ("search",)

    return McpRagAdapter(
        transport="stdio",
        tool_name="search",
        execution_mode="fixed",
        result_mode="answer",
        timeout_seconds=5,
        question_argument="query",
        top_k_argument="top_k",
        result_field="answer",
        tool_arguments={"collection": "docs"},
        command="python",
        command_args=["server.py"],
        session_factory=lambda **kwargs: FakeRunner(),
    )


def _mcp_diagnostics(**overrides: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "connection_seconds": 0.3,
        "tool_calls": [
            {"tool": "search", "total_seconds": 0.4, "timeout_attempts": 1},
            {"tool": "fetch", "total_seconds": 0.2, "timeout_attempts": 0},
        ],
        "generation_seconds": 1.5,
    }
    diagnostics.update(overrides)
    return diagnostics


def test_mcp_adapter_maps_diagnostics_to_stage_timings():
    adapter = _mcp_adapter()
    timings = adapter_stage_timings(adapter, _mcp_diagnostics())
    assert timings == {
        "mcp_connection": 0.3,
        "mcp_tool": pytest.approx(0.6),
        "mcp_generation": 1.5,
    }


def test_mcp_aggregate_metrics_summarize_run():
    adapter = _mcp_adapter()
    diagnostics = [
        _mcp_diagnostics(),
        _mcp_diagnostics(connection_seconds=0.0, error="boom", timeout=True),
    ]
    metrics = adapter_aggregate_metrics(adapter, diagnostics)
    assert metrics is not None
    assert metrics["sample_count"] == 2
    assert metrics["failure_count"] == 1
    assert metrics["failure_rate"] == 0.5
    assert metrics["tool_call_count"] == 4
    assert metrics["tool_call_counts"] == {"search": 2, "fetch": 2}
    assert metrics["connection_seconds"] == 0.3
    assert metrics["timeout_attempt_count"] == 2
    assert metrics["cold_tool_seconds"] == pytest.approx(0.6)
    assert metrics["warm_tool_mean_seconds"] == pytest.approx(0.6)


def test_mcp_aggregate_metrics_none_when_no_diagnostics():
    assert adapter_aggregate_metrics(_mcp_adapter(), []) is None
