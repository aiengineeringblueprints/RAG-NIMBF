"""Run-level metrics for RAG systems that act as MCP hosts."""

import pytest

from benchmark.mcp_host import (
    McpHostEvent,
    aggregate_by_category,
    aggregate_mcp_host_metrics,
    mcp_host_diagnostics,
)


def _sample(events, expected_tools=None):
    return mcp_host_diagnostics(events, expected_tools=expected_tools)


def test_call_success_rate_counts_only_tool_and_resource_attempts():
    diagnostics = [
        _sample(
            [
                McpHostEvent("demo", "discover", "ok"),
                McpHostEvent("demo", "tool", "ok", name="lookup", duration_seconds=0.2),
                McpHostEvent("demo", "tool", "tool_error", name="lookup", duration_seconds=0.4),
            ]
        ),
        _sample(
            [
                McpHostEvent("demo", "discover", "ok"),
                McpHostEvent("demo", "resource", "ok", name="policy", duration_seconds=0.1),
                McpHostEvent("demo", "tool", "invalid_arguments", name="add"),
            ]
        ),
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["sample_count"] == 2
    assert metrics["call_count"] == 4
    assert metrics["call_success_count"] == 2
    assert metrics["call_success_rate"] == pytest.approx(0.5)
    assert metrics["call_status_counts"] == {
        "ok": 2,
        "tool_error": 1,
        "invalid_arguments": 1,
    }
    assert metrics["invalid_arguments_rate"] == pytest.approx(0.25)
    assert metrics["calls_per_sample_mean"] == pytest.approx(2.0)
    assert metrics["samples_with_calls"] == 2


def test_latency_percentiles_use_only_timed_calls():
    diagnostics = [
        _sample(
            [
                McpHostEvent("a", "tool", "ok", name="t", duration_seconds=1.0),
                McpHostEvent("a", "tool", "ok", name="t", duration_seconds=2.0),
                McpHostEvent("a", "tool", "ok", name="t", duration_seconds=3.0),
                McpHostEvent("a", "tool", "ok", name="t", duration_seconds=4.0),
                McpHostEvent("a", "tool", "unavailable", name="t"),
            ]
        )
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["timed_call_count"] == 4
    assert metrics["call_latency_p50_seconds"] == pytest.approx(2.0)
    assert metrics["call_latency_p95_seconds"] == pytest.approx(4.0)
    assert metrics["call_latency_max_seconds"] == pytest.approx(4.0)
    assert metrics["call_seconds_total"] == pytest.approx(10.0)


def test_per_server_and_per_tool_breakdown():
    diagnostics = [
        _sample(
            [
                McpHostEvent("demo", "discover", "ok"),
                McpHostEvent("policies", "discover", "unavailable"),
                McpHostEvent("demo", "tool", "ok", name="add", duration_seconds=0.5),
                McpHostEvent("demo", "tool", "tool_error", name="add", duration_seconds=0.5),
                McpHostEvent("demo", "tool", "ok", name="lookup", duration_seconds=1.0),
            ]
        )
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["discovery_count"] == 2
    assert metrics["discovery_success_rate"] == pytest.approx(0.5)
    assert metrics["per_server"]["demo"]["call_count"] == 3
    assert metrics["per_server"]["demo"]["call_success_rate"] == pytest.approx(2 / 3)
    assert metrics["per_server"]["demo"]["discovery_success_rate"] == 1.0
    assert metrics["per_server"]["policies"]["discovery_success_rate"] == 0.0
    assert metrics["per_server"]["policies"]["call_count"] == 0
    assert metrics["per_tool"]["demo/add"]["call_success_rate"] == pytest.approx(0.5)
    assert metrics["per_tool"]["demo/lookup"]["latency_p50_seconds"] == pytest.approx(1.0)


def test_budget_exhaustion_denials_and_duplicates():
    diagnostics = [
        _sample(
            [
                McpHostEvent("demo", "tool", "ok", name="add", duration_seconds=0.1),
                McpHostEvent("demo", "tool", "ok", name="add", duration_seconds=0.1),
                McpHostEvent("host", "call", "budget_exhausted"),
                McpHostEvent("host", "loop", "budget_exhausted"),
            ]
        ),
        _sample(
            [
                McpHostEvent("legal", "connect", "denied"),
                McpHostEvent("unknown", "call", "denied"),
            ]
        ),
        _sample([]),
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["call_count"] == 4
    assert metrics["call_status_counts"]["budget_exhausted"] == 1
    assert metrics["call_status_counts"]["denied"] == 1
    assert metrics["budget_exhausted_sample_count"] == 1
    assert metrics["budget_exhausted_sample_rate"] == pytest.approx(1 / 3)
    assert metrics["connect_denied_count"] == 1
    assert metrics["duplicate_call_count"] == 1
    assert metrics["samples_with_calls"] == 2
    assert metrics["calls_per_sample_max"] == 3


def test_tool_selection_against_expected_tools():
    diagnostics = [
        # Needed lookup, called lookup: exact.
        _sample(
            [McpHostEvent("demo", "tool", "ok", name="lookup")],
            expected_tools=["lookup"],
        ),
        # Needed lookup, also called add: one false positive.
        _sample(
            [
                McpHostEvent("demo", "tool", "ok", name="lookup"),
                McpHostEvent("demo", "tool", "ok", name="add"),
            ],
            expected_tools=["demo/lookup"],
        ),
        # Needed lookup, called nothing: missed.
        _sample([], expected_tools=["lookup"]),
        # Document-only question, but a tool was called: unnecessary.
        _sample([McpHostEvent("demo", "tool", "ok", name="add")], expected_tools=[]),
        # Document-only question, no tool: correct abstention.
        _sample([], expected_tools=[]),
        # Unlabelled sample: excluded from selection metrics.
        _sample([McpHostEvent("demo", "tool", "ok", name="add")]),
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["tool_selection_sample_count"] == 5
    assert metrics["tool_selection_exact_match_rate"] == pytest.approx(2 / 5)
    # tp=2 (lookup twice), fp=2 (add twice), fn=1 (missed lookup)
    assert metrics["tool_selection_precision"] == pytest.approx(0.5)
    assert metrics["tool_selection_recall"] == pytest.approx(2 / 3)
    assert metrics["tool_selection_f1"] == pytest.approx(4 / 7)
    assert metrics["missed_tool_sample_rate"] == pytest.approx(1 / 3)
    assert metrics["unnecessary_call_sample_rate"] == pytest.approx(0.5)


def test_failed_attempt_still_counts_as_selected_tool():
    diagnostics = [
        _sample(
            [McpHostEvent("demo", "tool", "unavailable", name="lookup")],
            expected_tools=["lookup"],
        )
    ]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["tool_selection_recall"] == 1.0
    assert metrics["call_success_rate"] == 0.0


def test_samples_without_mcp_host_trace_are_ignored():
    assert aggregate_mcp_host_metrics([{}, {"error": "x"}]) is None


def test_failed_samples_are_counted_separately_from_call_failures():
    failed = mcp_host_diagnostics([], expected_tools=["lookup"])
    failed["error"] = "HTTP 502"
    diagnostics = [failed, _sample([McpHostEvent("d", "tool", "ok", name="t")])]

    metrics = aggregate_mcp_host_metrics(diagnostics)

    assert metrics["sample_count"] == 2
    assert metrics["failed_sample_count"] == 1
    assert metrics["failed_sample_rate"] == pytest.approx(0.5)
    # A request that failed before any trace is not a tool-selection miss.
    assert metrics["tool_selection_sample_count"] == 0


def test_category_breakdown_combines_answers_and_tool_use():
    def labelled(category, match, events, expected):
        d = mcp_host_diagnostics(events, expected_tools=expected)
        d.update({"category": category, "answer_match": match})
        return d

    diagnostics = [
        labelled("doc_single", True, [], []),
        labelled("doc_single", False, [McpHostEvent("d", "tool", "ok", name="t")], []),
        labelled("mcp_single", True, [McpHostEvent("d", "tool", "ok", name="t")], ["t"]),
        labelled("mcp_single", True, [McpHostEvent("d", "tool", "tool_error", name="t")], ["t"]),
        # Without an MCP trace (MCP off) only answer metrics are reported.
        {"category": "mcp_single", "answer_match": False},
    ]

    summary = aggregate_by_category(diagnostics)

    assert summary["answer_match_count"] == 3
    assert summary["answer_match_rate"] == pytest.approx(3 / 5)
    doc = summary["per_category"]["doc_single"]
    assert doc["sample_count"] == 2
    assert doc["answer_match_rate"] == pytest.approx(0.5)
    assert doc["unnecessary_call_sample_rate"] == pytest.approx(0.5)
    mcp = summary["per_category"]["mcp_single"]
    assert mcp["sample_count"] == 3
    assert mcp["answer_match_rate"] == pytest.approx(2 / 3)
    assert mcp["call_success_rate"] == pytest.approx(0.5)
    assert mcp["tool_selection_recall"] == 1.0


def test_category_breakdown_without_labels_is_none():
    assert aggregate_by_category([{}, {"use_mcp": False}]) is None


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status"):
        McpHostEvent("demo", "tool", "weird")
