"""Workload runner: execute the generated op stream against the live RAG.

Design notes
------------
* **Open-loop arrival**: the runner waits until each op's ``arrive_at``
  offset before submitting it. A separate scheduler thread sleeps and
  submits; workers from a ``ThreadPoolExecutor`` execute.
* **Bounded concurrency**: ``ThreadPoolExecutor(max_workers=concurrency)``
  caps in-flight work.
* **Thread-safe writes**: all vector-store mutations route through
  ``benchmark.retrieval`` adapter helpers that hold the shared Chroma
  write lock. Queries are also routed through that lock defensively —
  Chroma tolerates concurrent reads but coupling with concurrent writes
  was occasionally flaky in CI.
* **Per-op record**: each completed op yields a ``OpRecord`` with
  start/end, outcome, and (for queries) the answer / contexts / ground
  truth that the rest of the framework can feed into Ragas.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.documents import Document

from benchmark.workload.generator import WorkloadConfig, WorkloadGenerator
from benchmark.workload.operations import (
    InsertOp,
    Operation,
    QueryOp,
    RemoveOp,
    UpdateOp,
)


logger = logging.getLogger(__name__)


@dataclass
class OpRecord:
    """One executed op + its outcome + timing."""

    op_id: int
    kind: str
    submit_at: float
    start_at: float
    end_at: float
    outcome: str  # ok | error | stale-check
    error: str | None = None
    question: str | None = None
    answer: str | None = None
    ground_truth: str | None = None
    contexts: list[str] | None = None
    doc_id: str | None = None
    latency_seconds: float = 0.0
    queue_depth: int = 0

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "submit_at": self.submit_at,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "outcome": self.outcome,
            "error": self.error,
            "question": (self.question or "")[:160],
            "answer": (self.answer or "")[:160],
            "ground_truth": (self.ground_truth or "")[:160],
            "contexts_count": len(self.contexts) if self.contexts else 0,
            "doc_id": self.doc_id,
            "latency_seconds": self.latency_seconds,
            "queue_depth": self.queue_depth,
        }


@dataclass
class WorkloadSummary:
    """Aggregate report produced by ``WorkloadRunner.run``."""

    total_submitted: int = 0
    total_completed: int = 0
    total_errors: int = 0
    total_stale_check: int = 0
    counts_by_kind: dict[str, int] = field(default_factory=dict)
    latencies_by_kind: dict[str, list[float]] = field(default_factory=dict)
    wall_seconds: float = 0.0
    offered_qps: float = 0.0
    observed_qps: float = 0.0
    queue_depth_samples: list[int] = field(default_factory=list)
    records: list[OpRecord] = field(default_factory=list)
    stale_check_doc_ids: list[str] = field(default_factory=list)

    def add(self, record: OpRecord) -> None:
        self.records.append(record)
        self.counts_by_kind[record.kind] = self.counts_by_kind.get(record.kind, 0) + 1
        self.latencies_by_kind.setdefault(record.kind, []).append(
            record.latency_seconds
        )
        if record.outcome == "error":
            self.total_errors += 1
        elif record.outcome == "stale-check":
            self.total_stale_check += 1
        if record.kind == "update" and record.doc_id:
            self.stale_check_doc_ids.append(record.doc_id)


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class WorkloadRunner:
    """Execute a workload against a live vector store + generator.

    Parameters
    ----------
    config:
        The same ``WorkloadConfig`` used by the generator. ``concurrency``
        controls the ``ThreadPoolExecutor`` size; ``target_qps`` is honoured
        by the scheduler.
    generator:
        Pre-built generator (already seeded).
    vector_store:
        Live Chroma / LanceDB vector store.
    query_callable:
        ``query_callable(question: str) -> dict`` returning at least
        ``{"answer": str, "contexts": list[str]}``. Wired to the
        framework's retrieve + generate path by ``main.py``.
    embed_callable:
        Optional callable returning an embedding vector for a piece of
        text. Currently unused — Chroma's ``add_documents`` re-embeds
        internally via the configured embedding function — but kept on
        the seam for stores that need explicit embedding (e.g. LanceDB).
    """

    config: WorkloadConfig
    generator: WorkloadGenerator
    vector_store: Any
    query_callable: Callable[[str], dict]
    embed_callable: Callable[[str], list[float]] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> WorkloadSummary:
        """Execute the workload and return an aggregate summary."""
        summary = WorkloadSummary()
        summary.offered_qps = self.config.target_qps
        lock = threading.Lock()
        in_flight: list[Future] = []
        # Wait until the first arrival so we don't sleep a tiny epsilon.
        start_wall = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            for op in self.generator:
                # Open-loop arrival: sleep until op.arrive_at relative to start.
                now = time.perf_counter() - start_wall
                delay = op.arrive_at - now
                if delay > 0:
                    time.sleep(delay)
                # Record queue depth at submit time.
                with lock:
                    depth = sum(1 for f in in_flight if not f.done())
                    summary.queue_depth_samples.append(depth)
                future = pool.submit(self._execute_op, op)
                future.add_done_callback(
                    lambda fut, _op=op: self._record_result(fut, _op, summary, lock)
                )
                in_flight.append(future)
                # Reap finished futures to keep the list short.
                in_flight = [f for f in in_flight if not f.done()]
                summary.total_submitted += 1
            # Wait for everything to drain.
            for future in in_flight:
                future.result()
        summary.wall_seconds = time.perf_counter() - start_wall
        summary.total_completed = len(summary.records)
        completed_for_qps = summary.total_completed
        if summary.wall_seconds > 0:
            summary.observed_qps = completed_for_qps / summary.wall_seconds
        return summary

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _execute_op(self, op: Operation) -> OpRecord:
        """Dispatch one op to its handler."""
        submit_at = op.arrive_at
        start_at = time.perf_counter()
        try:
            if isinstance(op, QueryOp):
                record = self._handle_query(op, submit_at, start_at)
            elif isinstance(op, InsertOp):
                record = self._handle_insert(op, submit_at, start_at)
            elif isinstance(op, UpdateOp):
                record = self._handle_update(op, submit_at, start_at)
            elif isinstance(op, RemoveOp):
                record = self._handle_remove(op, submit_at, start_at)
            else:  # pragma: no cover — defensive
                raise ValueError(f"Unknown op kind: {op.kind}")
        except Exception as exc:  # noqa: BLE001 — log every failure
            logger.exception("op %s (%s) failed", op.op_id, op.kind)
            end_at = time.perf_counter()
            record = OpRecord(
                op_id=op.op_id,
                kind=op.kind,
                submit_at=submit_at,
                start_at=start_at,
                end_at=end_at,
                outcome="error",
                error=f"{type(exc).__name__}: {exc}",
                latency_seconds=end_at - start_at,
            )
        return record

    def _handle_query(
        self, op: QueryOp, submit_at: float, start_at: float
    ) -> OpRecord:
        result = self.query_callable(op.question)
        end_at = time.perf_counter()
        return OpRecord(
            op_id=op.op_id,
            kind="query",
            submit_at=submit_at,
            start_at=start_at,
            end_at=end_at,
            outcome="ok",
            question=op.question,
            answer=str(result.get("answer", "")),
            contexts=list(result.get("contexts", []) or []),
            ground_truth=op.ground_truth,
            latency_seconds=end_at - start_at,
        )

    def _handle_insert(
        self, op: InsertOp, submit_at: float, start_at: float
    ) -> OpRecord:
        from benchmark.retrieval import workload_insert_documents

        metadata = dict(op.metadata or {})
        metadata.setdefault("doc_id", op.doc_id)
        doc = Document(page_content=op.text, metadata=metadata)
        added = workload_insert_documents(self.vector_store, [doc])
        end_at = time.perf_counter()
        return OpRecord(
            op_id=op.op_id,
            kind="insert",
            submit_at=submit_at,
            start_at=start_at,
            end_at=end_at,
            outcome="ok" if added > 0 else "error",
            error=None if added > 0 else "no documents inserted",
            doc_id=op.doc_id,
            latency_seconds=end_at - start_at,
        )

    def _handle_update(
        self, op: UpdateOp, submit_at: float, start_at: float
    ) -> OpRecord:
        """Re-write a doc. Recorded as ``stale-check`` per design doc."""
        from benchmark.retrieval import workload_update_document

        workload_update_document(
            self.vector_store,
            op.doc_id,
            op.new_text,
            metadata=op.metadata,
        )
        end_at = time.perf_counter()
        return OpRecord(
            op_id=op.op_id,
            kind="update",
            submit_at=submit_at,
            start_at=start_at,
            end_at=end_at,
            # Update ops flag stale ground truth for any future Query on this
            # doc_id. The synth agent (separate worktree) consumes this list.
            outcome="stale-check",
            doc_id=op.doc_id,
            latency_seconds=end_at - start_at,
        )

    def _handle_remove(
        self, op: RemoveOp, submit_at: float, start_at: float
    ) -> OpRecord:
        from benchmark.retrieval import workload_remove_document

        removed = workload_remove_document(self.vector_store, op.doc_id)
        end_at = time.perf_counter()
        return OpRecord(
            op_id=op.op_id,
            kind="remove",
            submit_at=submit_at,
            start_at=start_at,
            end_at=end_at,
            outcome="ok" if removed > 0 else "error",
            error=None if removed > 0 else f"no chunks found for doc_id={op.doc_id}",
            doc_id=op.doc_id,
            latency_seconds=end_at - start_at,
        )

    def _record_result(
        self,
        future: Future,
        op: Operation,
        summary: WorkloadSummary,
        lock: threading.Lock,
    ) -> None:
        try:
            record = future.result()
        except Exception as exc:  # noqa: BLE001 — last-resort capture
            end_at = time.perf_counter()
            record = OpRecord(
                op_id=op.op_id,
                kind=op.kind,
                submit_at=op.arrive_at,
                start_at=end_at,
                end_at=end_at,
                outcome="error",
                error=f"runner-level: {type(exc).__name__}: {exc}",
                latency_seconds=0.0,
            )
        with lock:
            summary.add(record)
