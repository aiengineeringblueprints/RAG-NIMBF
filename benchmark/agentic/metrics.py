"""Metrics for benchmarking agent tool calls.

Complements the efficiency metrics in ``benchmark.adapters.mcp.aggregate_agent_metrics``
with correctness-of-tool-use metrics: selection precision/recall against
expected tool labels, hallucinated-tool rate, invalid-call rate and
redundant-call rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """One observed tool invocation by an agent."""

    tool: str
    round: int
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    total_seconds: float = 0.0
    result_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "round": self.round,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "total_seconds": self.total_seconds,
            "result_chars": self.result_chars,
        }

    @classmethod
    def from_dict(cls, data: "ToolCallRecord | dict[str, Any]") -> "ToolCallRecord":
        if isinstance(data, ToolCallRecord):
            return data
        return cls(
            tool=str(data.get("tool", "unknown")),
            round=int(data.get("round", 0) or 0),
            arguments=dict(data.get("arguments") or {}),
            ok=bool(data.get("ok", True)),
            error=data.get("error"),
            total_seconds=float(data.get("total_seconds", 0.0) or 0.0),
            result_chars=int(data.get("result_chars", 0) or 0),
        )


def _canonical_arguments(arguments: dict[str, Any]) -> tuple:
    """Order-independent hashable view of a call's arguments."""
    return tuple(sorted((key, repr(value)) for key, value in arguments.items()))


def compute_tool_call_metrics(
    records: Sequence[ToolCallRecord],
    *,
    expected_tools: Sequence[str] | None = None,
    available_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-question (or per-run) tool-call metrics from records.

    ``expected_tools`` enables selection precision/recall (needs gold labels).
    ``available_tools`` enables hallucinated-tool detection.
    """
    total = len(records)
    if total == 0:
        return {
            "tool_calls_total": 0,
            "calls_by_tool": {},
            "invalid_call_count": 0,
            "invalid_call_rate": 0.0,
            "redundant_call_count": 0,
            "redundant_call_rate": 0.0,
            "hallucinated_tool_count": 0,
            "hallucinated_tool_rate": 0.0,
            "tool_selection_precision": None,
            "tool_selection_recall": None,
            "tool_seconds_total": 0.0,
        }

    calls_by_tool: dict[str, int] = {}
    invalid = 0
    seen_calls: set[tuple[str, tuple]] = set()
    redundant = 0
    for record in records:
        calls_by_tool[record.tool] = calls_by_tool.get(record.tool, 0) + 1
        if not record.ok:
            invalid += 1
        signature = (record.tool, _canonical_arguments(record.arguments))
        if signature in seen_calls:
            redundant += 1
        seen_calls.add(signature)

    metrics: dict[str, Any] = {
        "tool_calls_total": total,
        "calls_by_tool": calls_by_tool,
        "invalid_call_count": invalid,
        "invalid_call_rate": invalid / total,
        "redundant_call_count": redundant,
        "redundant_call_rate": redundant / total,
        "hallucinated_tool_count": 0,
        "hallucinated_tool_rate": 0.0,
        "tool_selection_precision": None,
        "tool_selection_recall": None,
        "tool_seconds_total": sum(r.total_seconds for r in records),
    }

    if available_tools is not None:
        available = set(available_tools)
        hallucinated = sum(1 for r in records if r.tool not in available)
        metrics["hallucinated_tool_count"] = hallucinated
        metrics["hallucinated_tool_rate"] = hallucinated / total

    if expected_tools is not None:
        expected = set(expected_tools)
        called = {r.tool for r in records if r.ok}
        correct = sum(
            1 for r in records if r.ok and r.tool in expected
        )
        metrics["tool_selection_precision"] = correct / total
        metrics["tool_selection_recall"] = (
            len(called & expected) / len(expected) if expected else None
        )

    return metrics


def aggregate_tool_call_metrics(
    diagnostics: Iterable[dict[str, Any]],
    *,
    available_tools: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate per-sample ``tool_call_records`` into run-level metrics.

    The mean is taken over all diagnostics passed in, mirroring
    ``aggregate_agent_metrics``; callers should pre-filter to agentic samples.
    """
    items = list(diagnostics)
    all_records: list[ToolCallRecord] = []
    for item in items:
        raw = item.get("tool_call_records") or []
        records = [ToolCallRecord.from_dict(entry) for entry in raw]
        all_records.extend(records)

    per_run = compute_tool_call_metrics(all_records, available_tools=available_tools)
    per_run["tool_calls_sample_count"] = len(items)
    per_run["tool_calls_mean"] = (
        per_run["tool_calls_total"] / len(items) if items else 0.0
    )
    return per_run
