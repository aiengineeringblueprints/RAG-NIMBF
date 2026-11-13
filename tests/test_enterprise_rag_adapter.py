"""Smoke test for the Enterprise_RAG_Blueprint reference adapter.

Mocks the Blueprint imports (chain.retriever, chain.load_chain,
chain.prompts.promt_manager) so the test does not require the full
Blueprint codebase, Chroma, or any LLM/embedding backend.

Verifies the adapter:
1. Declares expected capability flags (embedder + llm only).
2. Routes bundle.embedder -> create_retriever(existing_embeddings=).
3. Routes bundle.llm -> load_chain(model=).
4. Reads config.retrieval_top_k -> create_retriever(returned_docs=).
5. Normalizes rag_chain tuple output -> RagSystemOutput with contexts.
"""
import sys
import types
from enum import Enum
from unittest.mock import MagicMock

import pytest


class _FakePromptKey(str, Enum):
    SOURCE = "rag_source"
    SUMMARY = "rag_summary"


@pytest.fixture
def fake_blueprint(monkeypatch):
    """Inject fake `chain.*` modules that mimic Blueprint API."""
    chain_pkg = types.ModuleType("chain")
    retriever_mod = types.ModuleType("chain.retriever")
    load_chain_mod = types.ModuleType("chain.load_chain")
    prompts_pkg = types.ModuleType("chain.prompts")
    promt_manager_mod = types.ModuleType("chain.prompts.promt_manager")

    fake_retriever = MagicMock(name="fake_retriever")
    retriever_mod.create_retriever = MagicMock(return_value=fake_retriever)

    fake_chain = MagicMock(name="fake_chain")
    load_chain_mod.load_chain = MagicMock(return_value=fake_chain)
    load_chain_mod.rag_chain = MagicMock(
        return_value=(
            "Paris is the capital of France.",
            [
                "wiki_france->Paris is the capital of France.",
                "wiki_eu->France is in Europe.",
            ],
        )
    )

    promt_manager_mod.PromptKey = _FakePromptKey
    prompts_pkg.promt_manager = promt_manager_mod

    chain_pkg.retriever = retriever_mod
    chain_pkg.load_chain = load_chain_mod
    chain_pkg.prompts = prompts_pkg

    monkeypatch.setitem(sys.modules, "chain", chain_pkg)
    monkeypatch.setitem(sys.modules, "chain.retriever", retriever_mod)
    monkeypatch.setitem(sys.modules, "chain.load_chain", load_chain_mod)
    monkeypatch.setitem(sys.modules, "chain.prompts", prompts_pkg)
    monkeypatch.setitem(sys.modules, "chain.prompts.promt_manager", promt_manager_mod)

    return {
        "retriever": fake_retriever,
        "create_retriever": retriever_mod.create_retriever,
        "chain": fake_chain,
        "load_chain": load_chain_mod.load_chain,
        "rag_chain": load_chain_mod.rag_chain,
    }


def test_supports_components(fake_blueprint):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter
    adapter = EnterpriseRagAdapter(MagicMock())
    caps = adapter.supports_components()
    assert caps["embedder"] is True
    assert caps["llm"] is True
    assert caps["chunker"] is True
    assert caps["retriever"] is False
    assert caps["reranker"] is False
    assert caps["prompt"] is False


def test_prepare_routes_bundle_and_config(fake_blueprint):
    from benchmark.adapters.components import ComponentBundle
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    cfg = MagicMock()
    cfg.retrieval_top_k = 7
    cfg.retriever_similarity_threshold = 0.4

    adapter = EnterpriseRagAdapter(cfg)
    adapter._replace_blueprint_index = MagicMock()
    adapter.set_components(ComponentBundle(
        chunker="FAKE_CHUNKER",
        embedder="FAKE_EMBEDDER",
        llm="FAKE_LLM",
    ))
    adapter.prepare(cfg, data=[], corpus=[{"context": "doc1"}])

    adapter._replace_blueprint_index.assert_called_once_with(
        cfg, data=[], corpus=[{"context": "doc1"}]
    )

    # create_retriever received embedder + top_k from config
    fake_blueprint["create_retriever"].assert_called_once()
    _, kwargs = fake_blueprint["create_retriever"].call_args
    assert kwargs["existing_embeddings"] == "FAKE_EMBEDDER"
    assert kwargs["returned_docs"] == 7
    assert kwargs["similarity_threshold"] == 0.4

    # load_chain received bundle.llm + retriever
    fake_blueprint["load_chain"].assert_called_once()
    _, kwargs = fake_blueprint["load_chain"].call_args
    assert kwargs["model"] == "FAKE_LLM"
    assert kwargs["context"] is fake_blueprint["retriever"]


def test_prepare_defaults_top_k_when_missing(fake_blueprint):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter
    cfg = MagicMock()
    cfg.retrieval_top_k = None
    cfg.retriever_similarity_threshold = None
    adapter = EnterpriseRagAdapter(cfg)
    adapter._replace_blueprint_index = MagicMock()
    adapter.prepare(cfg, data=[])
    _, kwargs = fake_blueprint["create_retriever"].call_args
    assert kwargs["returned_docs"] == 3
    assert kwargs["similarity_threshold"] == 0.5


def test_answer_normalizes_tuple_output(fake_blueprint):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    adapter = EnterpriseRagAdapter(MagicMock())
    adapter._replace_blueprint_index = MagicMock()
    adapter.prepare(MagicMock(), data=[])
    out = adapter.answer({"question": "What is the capital of France?"}, MagicMock())

    assert out.answer == "Paris is the capital of France."
    assert out.contexts == [
        "Paris is the capital of France.",
        "France is in Europe.",
    ]
    assert {"source": "wiki_france", "doc_id": "wiki_france"} in out.metadata
    assert {"source": "wiki_eu", "doc_id": "wiki_eu"} in out.metadata
    assert out.total_seconds >= 0


def test_answer_returns_invalid_output_on_chain_failure(fake_blueprint):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    fake_blueprint["rag_chain"].side_effect = RuntimeError("upstream timeout")
    adapter = EnterpriseRagAdapter(MagicMock())
    adapter._replace_blueprint_index = MagicMock()
    adapter.prepare(MagicMock(), data=[])

    out = adapter.answer({"question": "x"}, MagicMock())

    assert out.answer == ""
    assert out.answer_valid is False
    assert out.metadata[0]["error_type"] == "RuntimeError"
    assert "upstream timeout" in out.metadata[0]["error"]


def test_answer_raises_if_not_prepared(fake_blueprint):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter
    adapter = EnterpriseRagAdapter(MagicMock())
    with pytest.raises(RuntimeError, match="prepare\\(\\) was not called"):
        adapter.answer({"question": "x"}, MagicMock())


def test_documents_from_dataset_uses_stamped_gold_doc_ids_as_source():
    """Framework gold IDs (issue 16) must match the source IDs the Blueprint
    adapter writes when indexing per-sample-context datasets."""
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    data = [
        {
            "context": "doc zero",
            "metadata": {"doc_id": "dataset-doc-0", "gold_doc_id": "dataset-doc-0"},
        },
        {
            "context": "doc one",
            "metadata": {"doc_id": "dataset-doc-1", "gold_doc_id": "dataset-doc-1"},
        },
    ]

    docs = EnterpriseRagAdapter._documents_from_dataset(data)

    assert [d.metadata["source"] for d in docs] == [
        "dataset-doc-0",
        "dataset-doc-1",
    ]
    assert [d.metadata["doc_id"] for d in docs] == [
        "dataset-doc-0",
        "dataset-doc-1",
    ]


# ── End-to-end contract: adapter output → gold retrieval metrics ────


class TestGoldDocIdEndToEnd:
    """Adapter output metadata must resolve to the same document IDs the
    dataset stamped as gold, across every adapter path (issue #16 review)."""

    @staticmethod
    def _compute(gold, retrieved_metadata):
        from benchmark.gold_retrieval_metrics import (
            compute_gold_doc_retrieval_metrics,
        )

        return compute_gold_doc_retrieval_metrics(
            [gold],
            [retrieved_metadata],
            k_values=[1, 3, 5, 10],
        )

    def test_blueprint_string_sources_score_against_gold(self, fake_blueprint):
        """String "src->content" sources carry a doc_id the gold metric
        recognizes; a correct hit is no longer scored as a miss."""
        from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

        fake_blueprint["rag_chain"].return_value = (
            "Paris is the capital of France.",
            [
                "dataset-doc-0->Paris is the capital of France.",
                "dataset-doc-1->France is in Europe.",
            ],
        )
        adapter = EnterpriseRagAdapter(MagicMock())
        adapter._replace_blueprint_index = MagicMock()
        adapter.prepare(MagicMock(), data=[])
        out = adapter.answer(
            {"question": "What is the capital of France?"}, MagicMock()
        )

        result = self._compute("dataset-doc-0", out.metadata)

        assert result.skipped_samples == 0
        assert result.per_sample[0]["hit@1"] == 1.0

    def test_blueprint_indexing_and_gold_ids_stay_identical(self):
        """Indexed source IDs equal the IDs assign_gold_doc_ids stamped,
        including path-like and dotted values."""
        from benchmark.dataset import assign_gold_doc_ids
        from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

        data = [
            {"context": "a", "metadata": {"source": "/docs/manual.pdf"}},
            {"context": "b", "metadata": {}},
            {"context": "c", "metadata": {"source": ""}},
            {"context": "d", "metadata": {"source": None, "doc_id": "dotted.id"}},
        ]
        data = assign_gold_doc_ids(data)
        docs = EnterpriseRagAdapter._documents_from_dataset(data)

        assert [d.metadata["source"] for d in docs] == [
            d["metadata"]["gold_doc_id"] for d in data
        ]
        # Path-like sources reduce to their stem; empty/None fall back to
        # dataset-doc-{index} consistently on both sides.
        assert docs[0].metadata["doc_id"] == "manual"
        assert docs[1].metadata["doc_id"] == "dataset-doc-1"
        assert docs[2].metadata["doc_id"] == "dataset-doc-2"
        assert docs[3].metadata["doc_id"] == "dotted"

    def test_mcp_path_like_source_matches_gold(self):
        """MCP retrieval metadata canonicalizes path-like sources the same way
        the dataset side does."""
        from benchmark.adapters.mcp import _as_metadata

        metadata = _as_metadata(
            [
                {"source": "/docs/manual.pdf", "text": "irrelevant"},
            ],
            1,
            "query_documents",
            "stdio",
        )

        assert metadata[0]["doc_id"] == "manual"
        result = self._compute("manual", metadata)
        assert result.per_sample[0]["hit@1"] == 1.0

    def test_internal_pipeline_chunk_metadata_scores_against_gold(self):
        """The internal pipeline copies sample metadata (incl. doc_id) onto
        chunks; gold metrics must score those chunks correctly."""
        from benchmark.dataset import assign_gold_doc_ids

        samples = assign_gold_doc_ids(
            [
                {"question": "q", "ground_truth": "a", "context": "doc", "metadata": {}},
            ]
        )
        gold = samples[0]["metadata"]["gold_doc_id"]
        # Chunk metadata as produced by the internal pipeline (sample
        # metadata copied onto the retrieved chunk).
        chunk_metadata = [{"doc_id": gold, "chunk_index": 0}]

        result = self._compute(gold, chunk_metadata)

        assert result.per_sample[0]["hit@1"] == 1.0
        assert result.metric_means["hit@10"] == 1.0


def test_replace_blueprint_index_batches_above_chroma_max_batch_size(tmp_path, monkeypatch):
    """A corpus larger than Chroma's max batch size must still index fully."""
    import chromadb
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from examples.enterprise_rag_plugin.adapter import replace_blueprint_index

    monkeypatch.setenv("VECTORDB_DIR", str(tmp_path))
    monkeypatch.setenv("INDEX_NAME", "batch-test")
    max_batch = chromadb.PersistentClient(path=str(tmp_path)).get_max_batch_size()
    corpus = [{"context": f"document {i}"} for i in range(max_batch + 1)]
    bundle = types.SimpleNamespace(
        embedder=DeterministicFakeEmbedding(size=4),
        chunker=types.SimpleNamespace(split_documents=lambda docs: docs),
    )

    replace_blueprint_index(bundle, MagicMock(), data=[], corpus=corpus)

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("batch-test")
    assert collection.count() == max_batch + 1
