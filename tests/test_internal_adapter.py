"""Managed-lifecycle tests for the internal pipeline adapter.

Exercises ``InternalRagAdapter`` exclusively through the managed lifecycle
interface (capabilities, prepare, retrieve, generate, cleanup) with a tiny
in-memory corpus, a deterministic stub embedder, and a stubbed generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings

from benchmark.adapters import (
    RAG_ADAPTER_REGISTRY,
    cleanup_adapter,
    generate_adapter,
    get_adapter_capabilities,
    get_rag_adapter,
    prepare_adapter,
    require_adapter_capabilities,
    retrieve_adapter,
)
from benchmark.adapters.components import ComponentBundle
from benchmark.adapters.internal import InternalRagAdapter
from benchmark.generation import GenerationResult

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 "


class _StubEmbedder(Embeddings):
    """Deterministic bag-of-characters embeddings; no network, no disk."""

    def _vec(self, text: str) -> list[float]:
        counts = [float(text.lower().count(c)) for c in ALPHABET]
        norm = sum(v * v for v in counts) ** 0.5 or 1.0
        return [v / norm for v in counts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


DOC_A = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 and is 330 metres tall."
)
DOC_B = (
    "The Amazon River flows through South America. It carries more water "
    "than any other river in the world by a wide margin."
)

CORPUS = [
    {"id": "doc-a", "context": DOC_A, "question": "how tall?", "ground_truth": "330m"},
    {"id": "doc-b", "context": DOC_B, "question": "which river?", "ground_truth": "Amazon"},
]

SAMPLE = {
    "question": "How tall is the Eiffel Tower?",
    "ground_truth": "330m",
    "metadata": {"gold_doc_id": "doc-a"},
}


def stub_generator(llm, question, contexts, **kwargs) -> GenerationResult:
    return GenerationResult(
        answer=f"stub answer to: {question}",
        ttft_seconds=0.01,
        total_seconds=0.05,
        token_count=5,
        tokens_per_second=100.0,
        gpu_usage=None,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        raw_content=f"stub answer to: {question}",
    )


@dataclass
class InternalConfig:
    name: str = "internal-test"
    rag_system_adapter: str = "internal"
    retrieval_mode: str = "retrieval"
    retrieval_top_k: int = 1
    retrieval_strategy: str = "similarity"
    retrieval_fetch_k: int | None = None
    retrieval_mmr_lambda: float = 0.5
    retrieval_use_hyde: bool = False
    retrieval_multihop: bool = False
    retrieval_multihop_rounds: int = 2
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 0
    semantic_breakpoint_type: str = "percentile"
    semantic_breakpoint_amount: int = 95
    embedding_model: str = "stub-embedder"
    embedding_provider: str = "stub"
    llm_provider: str = "stub"
    llm_model: str = "stub-llm"
    max_new_tokens: int = 64
    prompt_template: str = "concise"
    reranker_model: str | None = None
    reranker_top_k: int = 3
    benchmark_stage: str = "all"
    dataset_name: str = "unit-test"
    dataset_subset: str = ""
    dataset_sample_size: int | None = None
    vector_db_backend: str = "chroma"
    lancedb_path: str = ".lancedb"
    llm_answer_strip_mode: str = "tags_only"
    llm_answer_value_fallback: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None
    rag_adapter_accepts: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def embedding_base_url(self) -> str:
        return "http://localhost:11434"

    def embedding_api_key(self) -> str | None:
        return None

    def llm_base_url(self) -> str:
        return "http://localhost:11434"

    def llm_api_key(self) -> str | None:
        return None

    def __getattr__(self, item: str) -> Any:
        try:
            return self.extra[item]
        except KeyError as exc:  # pragma: no cover - guard against typos
            raise AttributeError(item) from exc


def make_adapter(**config_kwargs) -> InternalRagAdapter:
    bundle = ComponentBundle(embedder=_StubEmbedder())
    adapter = InternalRagAdapter(generator=stub_generator)
    adapter.set_components(bundle)
    return adapter


def test_registry_returns_managed_internal_adapter():
    cfg = InternalConfig()
    adapter = get_rag_adapter(cfg)
    assert isinstance(adapter, InternalRagAdapter)
    assert "internal" in RAG_ADAPTER_REGISTRY


def test_capabilities_declare_full_internal_pipeline():
    adapter = InternalRagAdapter()
    caps = get_adapter_capabilities(adapter)
    assert caps.ingestion
    assert caps.retrieval
    assert caps.generation
    assert caps.chunk_configuration
    assert caps.embedding_configuration
    assert caps.reranking
    assert caps.token_usage
    assert caps.cleanup
    require_adapter_capabilities(adapter, "ingestion", "retrieval", "generation")


def test_supports_component_injection_slots():
    adapter = InternalRagAdapter()
    slots = adapter.supports_components()
    assert set(slots) == {"chunker", "embedder", "retriever", "reranker", "llm", "prompt"}
    assert all(slots.values())


def test_full_lifecycle_prepare_retrieve_generate_cleanup():
    adapter = make_adapter()
    config = InternalConfig()
    target = prepare_adapter(adapter, config, [], corpus=CORPUS)

    assert target.target_id
    assert target.metadata["chunk_count"] >= 1
    assert target.metadata["cache_key"]

    retrieval = retrieve_adapter(adapter, target, SAMPLE, config)
    assert len(retrieval.chunks) == 1
    assert retrieval.chunks[0].rank == 1
    assert "Eiffel" in retrieval.chunks[0].text

    output = generate_adapter(adapter, target, SAMPLE, config, retrieval=retrieval)
    assert output.answer == "stub answer to: How tall is the Eiffel Tower?"
    assert output.contexts == [retrieval.chunks[0].text]
    assert output.total_tokens == 15

    cleanup_adapter(adapter, target, config)
    assert adapter._store is None


def test_generate_without_prior_retrieval_runs_retrieval_itself():
    adapter = make_adapter()
    config = InternalConfig()
    target = prepare_adapter(adapter, config, [], corpus=CORPUS)

    output = generate_adapter(adapter, target, SAMPLE, config)
    assert output.answer.startswith("stub answer to:")
    assert "Eiffel" in output.contexts[0]
    cleanup_adapter(adapter, target, config)


def test_index_only_stage_is_expressible_through_lifecycle():
    adapter = make_adapter()
    config = InternalConfig(benchmark_stage="index")
    target = prepare_adapter(adapter, config, [], corpus=CORPUS)
    assert adapter._store is not None
    cleanup_adapter(adapter, target, config)
    assert adapter._store is None


def test_direct_mode_uses_sample_context_without_retrieval():
    adapter = make_adapter()
    config = InternalConfig(retrieval_mode="direct")
    target = prepare_adapter(adapter, config, [], corpus=CORPUS)

    sample = dict(SAMPLE, context=DOC_B)
    output = generate_adapter(adapter, target, sample, config)
    assert output.contexts == [DOC_B]
    cleanup_adapter(adapter, target, config)


def test_retrieve_before_prepare_raises():
    from benchmark.adapters import PreparedTarget

    adapter = make_adapter()
    with pytest.raises(RuntimeError, match="prepare"):
        adapter.retrieve(PreparedTarget(), SAMPLE, InternalConfig())


def test_prepare_uses_shared_index_cache_key():
    from benchmark.retrieval import index_cache_key

    adapter = make_adapter()
    config = InternalConfig()
    target = prepare_adapter(adapter, config, [], corpus=CORPUS)
    expected = index_cache_key(
        config.embedding_model,
        config.chunk_size,
        config.chunk_overlap,
        config.chunking_strategy,
        dataset_name=config.dataset_name,
        embedding_provider=config.embedding_provider,
        dataset_subset=config.dataset_subset,
        dataset_sample_size=config.dataset_sample_size,
        corpus_fingerprint=target.metadata["corpus_fingerprint"],
        vector_db_backend=config.vector_db_backend,
    )
    assert target.metadata["cache_key"] == expected
