"""Metrics for RAG systems that act as MCP hosts.

The system under test owns the MCP client and the tool loop; the benchmark
only observes the operation trace the system reports for each answer.
Adapters translate their system's trace into :class:`McpHostEvent` records
and store them with :func:`mcp_host_diagnostics`; the run-level summary comes
from :func:`aggregate_mcp_host_metrics`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

MCP_HOST_STATUSES = (
    "ok",
    "tool_error",
    "invalid_arguments",
    "denied",
    "timeout",
    "unavailable",
    "result_too_large",
    "budget_exhausted",
)
# connect/discover describe server sessions; tool/resource are executed
# capabilities; call is an attempt rejected before reaching a capability;
# loop marks a host-level event of the model/tool loop.
MCP_HOST_KINDS = ("connect", "discover", "tool", "resource", "call", "loop")
CALL_KINDS = frozenset({"tool", "resource", "call"})


@dataclass(frozen=True)
class McpHostEvent:
    """One MCP operation performed by the system under test."""

    server: str
    kind: str
    status: str
    name: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.status not in MCP_HOST_STATUSES:
            raise ValueError(
                f"Unknown MCP host status {self.status!r}; "
                f"use one of: {', '.join(MCP_HOST_STATUSES)}"
            )
        if self.kind not in MCP_HOST_KINDS:
            raise ValueError(
                f"Unknown MCP host event kind {self.kind!r}; "
                f"use one of: {', '.join(MCP_HOST_KINDS)}"
            )


def mcp_host_diagnostics(
    events: Iterable[McpHostEvent],
    *,
    expected_tools: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Per-sample diagnostics entry consumed by the aggregation.

    ``expected_tools`` labels the capabilities a correct answer needs, as
    ``name`` or ``server/name``. An empty list means no tool should be used;
    ``None`` leaves the sample out of tool-selection metrics.
    """
    return {
        "mcp_host": {
            "events": [asdict(event) for event in events],
            "expected_tools": (
                None if expected_tools is None else [str(t) for t in expected_tools]
            ),
        }
    }


def mcp_host_stage_timings(diagnostics: dict[str, Any]) -> dict[str, float]:
    """Seconds the system spent in timed MCP calls for one sample."""
    events = _events(diagnostics)
    return {
        "mcp_host_calls": sum(
            float(e["duration_seconds"])
            for e in events
            if e["kind"] in CALL_KINDS and e.get("duration_seconds") is not None
        )
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, matching benchmark.llm_performance."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _events(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    trace = diagnostics.get("mcp_host") if diagnostics else None
    return list(trace.get("events") or []) if isinstance(trace, dict) else []


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _call_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        float(c["duration_seconds"])
        for c in calls
        if c.get("duration_seconds") is not None
    ]
    successes = sum(c["status"] == "ok" for c in calls)
    return {
        "call_count": len(calls),
        "call_success_count": successes,
        "call_success_rate": _rate(successes, len(calls)),
        "status_counts": dict(Counter(c["status"] for c in calls)),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
    }


def _selection_counts(
    events: list[dict[str, Any]], expected: list[str]
) -> tuple[int, int, int]:
    """Return (true positives, false positives, false negatives)."""
    called = {
        (e["server"], e["name"])
        for e in events
        if e["kind"] in {"tool", "resource"} and e.get("name")
    }

    def matches(label: str, server: str, name: str) -> bool:
        return label == name or label == f"{server}/{name}"

    tp = sum(any(matches(label, s, n) for s, n in called) for label in expected)
    fp = sum(
        not any(matches(label, s, n) for label in expected) for s, n in called
    )
    return tp, fp, len(expected) - tp


_CATEGORY_HOST_KEYS = (
    "call_count",
    "call_success_rate",
    "calls_per_sample_mean",
    "tool_selection_exact_match_rate",
    "tool_selection_precision",
    "tool_selection_recall",
    "missed_tool_sample_rate",
    "unnecessary_call_sample_rate",
)


def aggregate_by_category(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Answer-match rate overall and per ``category``, plus tool use per category.

    Samples carry ``answer_match`` (bool) and ``category`` (str) in their
    diagnostics. Tool-use keys appear only for categories with MCP traces.
    """
    matched = [d for d in diagnostics if d and isinstance(d.get("answer_match"), bool)]
    categorized = [d for d in diagnostics if d and d.get("category")]
    if not matched and not categorized:
        return None
    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(d["category"]) for d in categorized}):
        group = [d for d in categorized if str(d["category"]) == category]
        answers = [d["answer_match"] for d in group if isinstance(d.get("answer_match"), bool)]
        entry: dict[str, Any] = {
            "sample_count": len(group),
            "answer_match_rate": _rate(sum(answers), len(answers)),
        }
        host = aggregate_mcp_host_metrics(group)
        if host is not None:
            entry.update({key: host[key] for key in _CATEGORY_HOST_KEYS})
        per_category[category] = entry
    hits = sum(d["answer_match"] for d in matched)
    return {
        "answer_match_count": hits,
        "answer_match_rate": _rate(hits, len(matched)),
        "per_category": per_category,
    }


def aggregate_mcp_host_metrics(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Fold per-sample MCP host traces into one run-level summary."""
    samples = [d for d in diagnostics if d and isinstance(d.get("mcp_host"), dict)]
    if not samples:
        return None

    all_events = [e for d in samples for e in _events(d)]
    calls = [e for e in all_events if e["kind"] in CALL_KINDS]
    discoveries = [e for e in all_events if e["kind"] == "discover"]
    calls_per_sample = [
        sum(e["kind"] in CALL_KINDS for e in _events(d)) for d in samples
    ]
    latencies = [
        float(c["duration_seconds"])
        for c in calls
        if c.get("duration_seconds") is not None
    ]
    status_counts = dict(Counter(c["status"] for c in calls))
    successes = status_counts.get("ok", 0)

    duplicates = 0
    for d in samples:
        named = Counter(
            (e["server"], e["kind"], e["name"])
            for e in _events(d)
            if e["kind"] in {"tool", "resource"} and e.get("name")
        )
        duplicates += sum(count - 1 for count in named.values())

    budget_samples = sum(
        any(e["status"] == "budget_exhausted" for e in _events(d)) for d in samples
    )
    failed_samples = sum(bool(d.get("error")) for d in samples)

    servers = sorted({e["server"] for e in all_events if e["kind"] != "loop"})
    per_server: dict[str, dict[str, Any]] = {}
    for server in servers:
        server_calls = [c for c in calls if c["server"] == server]
        server_discoveries = [d for d in discoveries if d["server"] == server]
        summary = _call_summary(server_calls)
        summary["discovery_success_rate"] = _rate(
            sum(d["status"] == "ok" for d in server_discoveries),
            len(server_discoveries),
        )
        per_server[server] = summary

    per_tool: dict[str, dict[str, Any]] = {}
    for key in sorted(
        {f"{c['server']}/{c['name']}" for c in calls if c.get("name")}
    ):
        server, _, name = key.partition("/")
        per_tool[key] = _call_summary(
            [c for c in calls if c["server"] == server and c.get("name") == name]
        )

    tp = fp = fn = exact = 0
    missed = needs_tool = unnecessary = needs_none = 0
    labelled = 0
    for d in samples:
        expected = d["mcp_host"].get("expected_tools")
        if expected is None or d.get("error"):
            continue
        labelled += 1
        s_tp, s_fp, s_fn = _selection_counts(_events(d), expected)
        tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
        exact += s_fp == 0 and s_fn == 0
        if expected:
            needs_tool += 1
            missed += s_fn > 0
        else:
            needs_none += 1
            unnecessary += s_fp > 0
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    return {
        "sample_count": len(samples),
        "failed_sample_count": failed_samples,
        "failed_sample_rate": failed_samples / len(samples),
        "call_count": len(calls),
        "call_success_count": successes,
        "call_success_rate": _rate(successes, len(calls)),
        "call_status_counts": status_counts,
        **{
            f"{status}_rate": _rate(status_counts.get(status, 0), len(calls))
            for status in MCP_HOST_STATUSES
            if status != "ok"
        },
        "samples_with_calls": sum(count > 0 for count in calls_per_sample),
        "calls_per_sample_mean": sum(calls_per_sample) / len(samples),
        "calls_per_sample_max": max(calls_per_sample),
        "duplicate_call_count": duplicates,
        "duplicate_call_rate": _rate(duplicates, len(calls)),
        "budget_exhausted_sample_count": budget_samples,
        "budget_exhausted_sample_rate": budget_samples / len(samples),
        "connect_denied_count": sum(
            e["kind"] == "connect" and e["status"] == "denied" for e in all_events
        ),
        "discovery_count": len(discoveries),
        "discovery_success_rate": _rate(
            sum(d["status"] == "ok" for d in discoveries), len(discoveries)
        ),
        "timed_call_count": len(latencies),
        "call_seconds_total": sum(latencies),
        "call_latency_p50_seconds": percentile(latencies, 0.50),
        "call_latency_p95_seconds": percentile(latencies, 0.95),
        "call_latency_max_seconds": max(latencies) if latencies else None,
        "tool_selection_sample_count": labelled,
        "tool_selection_exact_match_rate": _rate(exact, labelled),
        "tool_selection_precision": precision,
        "tool_selection_recall": recall,
        "tool_selection_f1": f1,
        "missed_tool_sample_rate": _rate(missed, needs_tool),
        "unnecessary_call_sample_rate": _rate(unnecessary, needs_none),
        "per_server": per_server,
        "per_tool": per_tool,
    }
