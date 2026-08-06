"""Operation payloads produced by the workload generator.

Each dataclass captures the inputs required to dispatch one RAGPerf §3.2
op: Query / Insert / Update / Remove. The runner consumes these and
records per-op timing and outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Operation:
    """Base operation.

    ``op_id`` is unique within a single generated workload. ``arrive_at``
    is the wall-clock offset (seconds from start) at which the runner
    should release the op into the executor — this enforces the open-loop
    arrival pattern.
    """

    op_id: int
    arrive_at: float
    kind: str

    def as_log_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "arrive_at": self.arrive_at, "kind": self.kind}


@dataclass(frozen=True)
class QueryOp(Operation):
    """Run the full retrieve -> generate path for a question."""

    question: str = ""
    ground_truth: str = ""
    sample: dict[str, Any] = field(default_factory=dict)
    kind: str = "query"

    def as_log_dict(self) -> dict[str, Any]:
        out = super().as_log_dict()
        out["question"] = self.question[:160]
        return out


@dataclass(frozen=True)
class InsertOp(Operation):
    """Ingest one new document, embed it, and index it."""

    doc_id: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "insert"

    def as_log_dict(self) -> dict[str, Any]:
        out = super().as_log_dict()
        out["doc_id"] = self.doc_id
        return out


@dataclass(frozen=True)
class UpdateOp(Operation):
    """Modify an existing document's text and re-index.

    NOTE: the new ground truth is not synthesised here. The runner flags
    affected Query ops as ``stale-check`` so a downstream dynamic
    ground-truth synth agent can rebuild them.
    """

    doc_id: str = ""
    new_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "update"

    def as_log_dict(self) -> dict[str, Any]:
        out = super().as_log_dict()
        out["doc_id"] = self.doc_id
        return out


@dataclass(frozen=True)
class RemoveOp(Operation):
    """Delete a document (all of its chunks) from the index."""

    doc_id: str = ""
    kind: str = "remove"

    def as_log_dict(self) -> dict[str, Any]:
        out = super().as_log_dict()
        out["doc_id"] = self.doc_id
        return out


OP_KINDS: tuple[str, ...] = ("query", "insert", "update", "remove")
