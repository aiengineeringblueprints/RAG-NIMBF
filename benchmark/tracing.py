"""MLflow tracing — OpenTelemetry-based observability via MLflow.

When MLflow tracing is enabled, pipeline functions are traced with
parent-child span relationships. Optionally exports traces via OTLP
to an external OpenTelemetry collector.
"""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable

import mlflow
from mlflow.entities import SpanType

logger = logging.getLogger(__name__)

_OTEL_EXPORTER_CONFIGURED = False


def setup_tracing() -> str | None:
    """Configure MLflow tracing and optional OTLP exporter.

    Returns the tracking URI if tracing is enabled, None otherwise.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try connecting to MLflow server; fall back to SQLite
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        sqlite_uri = "sqlite:///mlflow.db"
        logger.warning(
            "MLflow server at %s unreachable, falling back to %s",
            tracking_uri, sqlite_uri,
        )
        tracking_uri = sqlite_uri

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # Optional OTLP exporter for external OTel backends
    global _OTEL_EXPORTER_CONFIGURED
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint and not _OTEL_EXPORTER_CONFIGURED:
        _configure_otlp_exporter(otlp_endpoint)
        _OTEL_EXPORTER_CONFIGURED = True

    return tracking_uri


def _configure_otlp_exporter(endpoint: str) -> None:
    """Configure OpenTelemetry OTLP exporter alongside MLflow tracing."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        from mlflow.tracing.provider import _get_tracer

        resource = Resource.create({"service.name": "rag-benchmark"})
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)

        tracer = _get_tracer()
        if hasattr(tracer, "provider") and isinstance(tracer.provider, TracerProvider):
            tracer.provider.add_span_processor(processor)
            logger.info("OTLP exporter configured: %s", endpoint)
        else:
            logger.warning("Could not attach OTLP exporter to MLflow tracer")
    except ImportError:
        logger.warning(
            "opentelemetry-exporter-otlp not installed. "
            "Install with: pip install opentelemetry-exporter-otlp"
        )
    except Exception as e:
        logger.warning("Failed to configure OTLP exporter: %s", e)


# ---------------------------------------------------------------------------
# Per-sample tracing (issue #42)
# ---------------------------------------------------------------------------


def contexts_to_documents(
    contexts: Sequence[Any],
    metadata: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert generator contexts into MLflow ``Document`` dicts.

    Every context becomes exactly one document, in order, so retrieval
    judges see what the generator saw. Bare strings without metadata are
    first-class input: they get a positional id instead of being dropped.
    """
    from mlflow.entities import Document

    metadata = list(metadata or [])
    documents: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        meta = dict(metadata[index]) if index < len(metadata) and metadata[index] else {}
        doc_id = next(
            (meta[key] for key in ("doc_id", "chunk_id", "id") if meta.get(key) is not None),
            f"ctx_{index}",
        )
        meta.setdefault("rank", index + 1)
        documents.append(
            Document(
                page_content=str(context),
                metadata=_jsonable(meta),
                id=str(doc_id),
            ).to_dict()
        )
    return documents


def _jsonable(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep span attributes serializable: stringify exotic metadata values."""
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


def _langchain_docs_to_documents(docs: Sequence[Any]) -> list[dict[str, Any]]:
    return contexts_to_documents(
        [doc.page_content for doc in docs], [doc.metadata or {} for doc in docs]
    )


def traced_rerank(reranker: Any, query: str, docs: list[Any], top_k: int) -> list[Any]:
    """Run ``reranker.rerank`` inside a RERANKER span (only while tracing)."""
    if mlflow.get_current_active_span() is None:
        return reranker.rerank(query, docs, top_k)
    with mlflow.start_span(name="rerank", span_type=SpanType.RERANKER) as span:
        span.set_inputs({
            "query": query,
            "documents": _langchain_docs_to_documents(docs),
            "top_k": top_k,
        })
        reranked = reranker.rerank(query, docs, top_k)
        span.set_outputs(_langchain_docs_to_documents(reranked))
    return reranked


def traced_model_call(model: Any, messages: list[Any], *, round_number: int) -> Any:
    """Invoke a chat model inside an LLM span (only while tracing).

    Used by agentic adapters that call ``model.invoke`` directly instead of
    going through ``generate_answer``.
    """
    if mlflow.get_current_active_span() is None:
        return model.invoke(messages)
    with mlflow.start_span(name="agent_model_call", span_type=SpanType.LLM) as span:
        span.set_inputs({
            "round": round_number,
            "messages": [
                {"type": getattr(m, "type", type(m).__name__), "content": str(getattr(m, "content", m))}
                for m in messages
            ],
        })
        response = model.invoke(messages)
        span.set_outputs({
            "content": str(getattr(response, "content", "")),
            "tool_calls": [
                {"name": call.get("name"), "args": call.get("args")}
                for call in (getattr(response, "tool_calls", None) or [])
            ],
        })
    return response


class SampleTrace:
    """Handle on one sample's trace; a no-op when tracing is disabled."""

    def __init__(
        self, root: Any | None = None, *, question: str = "", adapter_name: str = ""
    ) -> None:
        self._root = root
        self._question = question
        self._adapter_name = adapter_name
        self.trace_id: str | None = None

    @contextmanager
    def adapter_call(self, *, retrieve_only: bool) -> Iterator[None]:
        """Client-side span around the adapter call (internal or external)."""
        if self._root is None:
            yield
            return
        stage = "retrieve" if retrieve_only else "generate"
        with mlflow.start_span(
            name=f"adapter.{stage}", span_type=SpanType.CHAIN
        ) as span:
            span.set_attributes({"rag.adapter": self._adapter_name, "rag.stage": stage})
            yield

    def record(
        self,
        answer: str,
        contexts: Sequence[Any],
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Emit the single RETRIEVER span from the generator's contexts."""
        if self._root is None:
            return
        with mlflow.start_span(name="retrieved_contexts", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": self._question})
            span.set_outputs(contexts_to_documents(contexts, metadata))
        self._root.set_outputs({"response": answer})


@contextmanager
def trace_sample(
    *,
    enabled: bool,
    question: str,
    config_name: str,
    adapter_name: str,
    index: int,
    cached: bool = False,
) -> Iterator[SampleTrace]:
    """Wrap one benchmark sample in one MLflow trace.

    Disabled yields a no-op handle. Callers control the global tracing state
    with :func:`tracing_state` so decorated pipeline functions outside a
    sample (ingestion, evaluation) do not leak stray root traces.
    """
    if not enabled:
        yield SampleTrace()
        return
    with mlflow.start_span(name="rag_sample", span_type=SpanType.CHAIN) as root:
        root.set_inputs({"question": question})
        root.set_attributes({
            "benchmark.config_name": config_name,
            "benchmark.sample_index": index,
            "benchmark.cached": cached,
            "rag.adapter": adapter_name,
        })
        handle = SampleTrace(root, question=question, adapter_name=adapter_name)
        yield handle
    handle.trace_id = root.trace_id


@contextmanager
def tracing_state(enabled: bool) -> Iterator[None]:
    """Force MLflow tracing on or off for a block, restoring the prior state.

    ``mlflow.tracing.enable/disable`` swap the global tracer provider, so
    callers toggle once per phase (not per sample).
    """
    from mlflow.tracing.provider import is_tracing_enabled

    was_enabled = is_tracing_enabled()
    if was_enabled != enabled:
        (mlflow.tracing.enable if enabled else mlflow.tracing.disable)()
    try:
        yield
    finally:
        if was_enabled != enabled:
            (mlflow.tracing.enable if was_enabled else mlflow.tracing.disable)()


def flush_traces() -> None:
    """Block until asynchronously exported traces are persisted."""
    try:
        mlflow.flush_trace_async_logging()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Flushing MLflow traces failed: %s", exc)


def sample_expectations(sample: dict[str, Any]) -> dict[str, Any]:
    """Ground truth of one sample as MLflow expectations (issue #43).

    Uses MLflow's reserved names so built-in judges pick them up:
    ``expected_response`` is the reference answer, ``expected_retrieved_context``
    lists the gold documents (``doc_uri`` = gold doc id, matching the RETRIEVER
    span's document ids). Gold text is attached only for a single
    ``gold_doc_id``, where the sample context *is* that document; multi-gold
    contexts (multi-hop) mix in distractors, so they carry ids only. A
    sample with both keys uses ``gold_doc_id``; loaders set only one of them.
    ``content`` is the dataset's reference context: identical to the indexed
    gold document for per-sample datasets and deduplicated corpora, but for
    ``jsonl-shared`` it is whatever the context field holds (possibly a
    passage of the corpus document).
    """
    expectations: dict[str, Any] = {}
    reference = str(sample.get("ground_truth") or "").strip()
    if reference:
        expectations["expected_response"] = reference

    metadata = sample.get("metadata") or {}
    gold_contexts: list[dict[str, str]] = []
    if metadata.get("gold_doc_id"):
        gold = {"doc_uri": str(metadata["gold_doc_id"])}
        context = sample.get("context")
        if isinstance(context, list):
            context = "\n".join(str(part) for part in context)
        if context:
            gold["content"] = str(context)
        gold_contexts.append(gold)
    else:
        gold_contexts = [
            {"doc_uri": str(doc_id)}
            for doc_id in metadata.get("gold_doc_ids") or []
            if doc_id
        ]
    if gold_contexts:
        expectations["expected_retrieved_context"] = gold_contexts
    return expectations


def log_sample_expectations(
    samples_by_trace: Sequence[tuple[str, dict[str, Any]]], *, source_id: str
) -> int:
    """Attach each sample's ground truth to its trace; returns the count logged.

    Flushes async trace logging first: an expectation can only be attached
    to a trace that is already persisted. Failures are logged, never raised —
    a missing expectation must not fail the benchmark.
    """
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    if not samples_by_trace:
        return 0
    flush_traces()
    source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=source_id)
    logged = 0
    for trace_id, sample in samples_by_trace:
        for name, value in sample_expectations(sample).items():
            try:
                mlflow.log_expectation(
                    trace_id=trace_id, name=name, value=value, source=source
                )
                logged += 1
            except Exception as exc:
                logger.warning(
                    "Logging expectation %r on trace %s failed: %s", name, trace_id, exc
                )
    return logged


@contextmanager
def expectations_logged_on_exit(
    *, source_id: str
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Collect ``(trace_id, sample)`` pairs; attach their ground truth on exit.

    Runs on any exit, so traces persisted before a mid-loop crash still get
    their expectations instead of silently looking complete without them.
    """
    traced_samples: list[tuple[str, dict[str, Any]]] = []
    try:
        yield traced_samples
    finally:
        log_sample_expectations(traced_samples, source_id=source_id)


def _noop_observe(name: str | None = None) -> Callable:
    """Identity decorator used when tracing is disabled."""
    def decorator(fn: Callable) -> Callable:
        return fn
    return decorator


# Module-level decorator: real @observe when tracing is enabled, no-op otherwise.
# For now, always use no-op since LangFuse has been removed.
observe = _noop_observe


def setup_langfuse() -> str | None:
    """Deprecated: LangFuse has been removed.

    This function exists for backward compatibility and always returns None.
    """
    logger.info("LangFuse tracing has been removed, using MLflow tracing instead")
    return None
