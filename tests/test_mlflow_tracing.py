"""MLflow Tracing through the run seam (issue #42).

Every sample run by ``run_single_benchmark`` must produce exactly one trace
with exactly one RETRIEVER span whose documents are the contexts the
generator actually saw — independent of the adapter (internal pipeline,
HTTP, MCP-style ``answer()`` adapters) and of the retrieval strategy
(multi-hop, hybrid, rerank). Traces are read back from a throwaway SQLite
tracking store.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from types import SimpleNamespace

import mlflow
import pytest
from conftest import StubEmbedder
from test_run_path_uniformity import UniformityConfig

import benchmark.adapters.components as components_module
import benchmark.adapters.http as http_module
import benchmark.generation
from benchmark.adapters import (
    RAG_ADAPTER_REGISTRY,
    RagSystemOutput,
    register_rag_adapter,
)
from benchmark.adapters.components import ComponentBundle
from benchmark.orchestration.runner import run_single_benchmark

DOCS = {
    "doc-a": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "doc-b": "Gustave Eiffel's company designed and built the tower in 1889.",
    "doc-c": "The Colosseum is an ancient amphitheatre in the centre of Rome.",
}
CORPUS = [
    {"id": doc_id, "context": text, "question": "q", "ground_truth": "a"}
    for doc_id, text in DOCS.items()
]
SAMPLES = [
    {"question": "Where is the Eiffel Tower?", "ground_truth": "Paris"},
    {"question": "Who built the Eiffel Tower?", "ground_truth": "Gustave Eiffel"},
]


class _FollowupLLM:
    """Chat-model stand-in: always proposes a follow-up multi-hop query."""

    def invoke(self, messages, config=None):
        return SimpleNamespace(content="tower built 1889 company")


class _ReversingReranker:
    def rerank(self, query, documents, top_k):
        return list(reversed(documents))[:top_k]


@pytest.fixture
def tracking_store(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Same experiment the worker and log_benchmark_run use in production.
    experiment_id = mlflow.set_experiment("RAG-Benchmark").experiment_id
    yield experiment_id
    mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def generator_calls(monkeypatch):
    """Stub the generator and record the contexts it was handed."""
    calls: list[list[str]] = []

    def _generate(llm, question, contexts, **kwargs):
        calls.append(list(contexts))
        return benchmark.generation.GenerationResult(
            answer=f"answer to: {question}",
            ttft_seconds=0.01,
            total_seconds=0.05,
            token_count=3,
            tokens_per_second=60.0,
            gpu_usage=None,
            raw_content=f"answer to: {question}",
        )

    monkeypatch.setattr(benchmark.generation, "generate_answer", _generate)
    return calls


def _inject(monkeypatch, **slots):
    monkeypatch.setattr(
        components_module,
        "build_components",
        lambda config, accepts=None: ComponentBundle(
            embedder=StubEmbedder(), llm=_FollowupLLM(), **slots
        ),
    )


def _traces(experiment_id):
    traces = mlflow.search_traces(
        locations=[experiment_id], return_type="list"
    )
    return sorted(traces, key=_sample_index)


def _sample_index(trace):
    (root,) = [span for span in trace.data.spans if span.parent_id is None]
    return root.attributes["benchmark.sample_index"]


def _spans_of_type(trace, span_type):
    return [span for span in trace.data.spans if span.span_type == span_type]


def _retriever_contents(trace):
    (span,) = _spans_of_type(trace, "RETRIEVER")
    return [doc["page_content"] for doc in span.outputs]


def _traced(**overrides):
    config = UniformityConfig(**{"retrieval_top_k": 2, **overrides})
    config.extra["mlflow_tracing_enabled"] = True
    return config


def _assert_one_trace_per_sample_matching(experiment_id, result, contexts_seen):
    traces = _traces(experiment_id)
    assert len(traces) == len(SAMPLES)
    assert list(result.trace_ids) == [t.info.trace_id for t in traces]
    for trace, contexts in zip(traces, contexts_seen):
        assert len(_spans_of_type(trace, "RETRIEVER")) == 1
        assert _retriever_contents(trace) == contexts
        assert contexts, "generator must have seen at least one context"


def test_internal_similarity_one_trace_per_sample(
    tracking_store, generator_calls, monkeypatch
):
    _inject(monkeypatch)

    result = run_single_benchmark(_traced(), SAMPLES, corpus=CORPUS)

    _assert_one_trace_per_sample_matching(tracking_store, result, generator_calls)


def test_internal_multihop_emits_single_retriever_span(
    tracking_store, generator_calls, monkeypatch
):
    _inject(monkeypatch)
    config = _traced(retrieval_multihop=True, retrieval_top_k=3)

    result = run_single_benchmark(config, SAMPLES, corpus=CORPUS)

    _assert_one_trace_per_sample_matching(tracking_store, result, generator_calls)


def test_internal_hybrid_emits_single_retriever_span(
    tracking_store, generator_calls, monkeypatch
):
    _inject(monkeypatch)

    result = run_single_benchmark(
        _traced(retrieval_strategy="hybrid"), SAMPLES, corpus=CORPUS
    )

    _assert_one_trace_per_sample_matching(tracking_store, result, generator_calls)


def test_internal_rerank_records_reranked_contexts(
    tracking_store, generator_calls, monkeypatch
):
    _inject(monkeypatch, reranker=_ReversingReranker())
    config = _traced(retrieval_top_k=3)
    config.extra["reranker_top_k"] = 2

    result = run_single_benchmark(config, SAMPLES, corpus=CORPUS)

    _assert_one_trace_per_sample_matching(tracking_store, result, generator_calls)
    for trace in _traces(tracking_store):
        assert len(_spans_of_type(trace, "RERANKER")) == 1


class _PlainStringAdapter:
    """MCP-style legacy adapter: ``answer()`` with bare string contexts."""

    name = "plain-strings"

    def __init__(self, config):
        pass

    def prepare(self, config, data, corpus=None):
        return None

    def answer(self, sample, config):
        return RagSystemOutput(
            answer="Paris",
            contexts=[DOCS["doc-a"], DOCS["doc-b"]],
        )


def test_plain_string_contexts_become_documents(tracking_store):
    register_rag_adapter("plain-strings", _PlainStringAdapter)
    try:
        result = run_single_benchmark(
            _traced(rag_system_adapter="plain-strings"), SAMPLES, corpus=CORPUS
        )
    finally:
        RAG_ADAPTER_REGISTRY.pop("plain-strings", None)

    expected = [[DOCS["doc-a"], DOCS["doc-b"]]] * len(SAMPLES)
    _assert_one_trace_per_sample_matching(tracking_store, result, expected)


def test_http_adapter_traced_client_side(tracking_store, monkeypatch):
    response = {
        "answer": "Paris",
        "contexts": [DOCS["doc-c"], DOCS["doc-a"]],
        "metadata": [{"doc_id": "doc-c"}, {"doc_id": "doc-a"}],
    }
    monkeypatch.setattr(
        http_module.urllib.request,
        "urlopen",
        lambda req, timeout=None: io.BytesIO(json.dumps(response).encode()),
    )
    config = _traced(rag_system_adapter="http")
    config.extra.update(
        rag_http_endpoint_url="http://rag.invalid/answer",
        rag_http_headers="",
        rag_http_auth_header="",
        rag_http_auth_value="",
        rag_http_timeout_seconds=5.0,
        rag_http_answer_field="answer",
        rag_http_contexts_field="contexts",
        rag_http_metadata_field="metadata",
        rag_http_timings_field="timings",
    )

    result = run_single_benchmark(config, SAMPLES, corpus=CORPUS)

    expected = [[DOCS["doc-c"], DOCS["doc-a"]]] * len(SAMPLES)
    _assert_one_trace_per_sample_matching(tracking_store, result, expected)
    (span,) = _spans_of_type(_traces(tracking_store)[0], "RETRIEVER")
    assert [doc["id"] for doc in span.outputs] == ["doc-c", "doc-a"]


def test_tracing_off_records_no_traces(tracking_store, generator_calls, monkeypatch):
    _inject(monkeypatch)
    config = replace(_traced(retrieval_multihop=True), extra={})

    result = run_single_benchmark(config, SAMPLES, corpus=CORPUS)

    assert _traces(tracking_store) == []
    assert result.trace_ids == ()


def test_traces_are_linked_to_the_config_run(
    tracking_store, generator_calls, monkeypatch
):
    from benchmark.tracking import log_benchmark_run

    _inject(monkeypatch)
    with mlflow.start_run(run_name="parent"):
        result = run_single_benchmark(_traced(), SAMPLES, corpus=CORPUS)
        run_id = log_benchmark_run(result, nested=True)

    linked = mlflow.search_traces(
        locations=[tracking_store], run_id=run_id, return_type="list"
    )
    assert sorted(t.info.trace_id for t in linked) == sorted(result.trace_ids)
    assert len(linked) == len(SAMPLES)
