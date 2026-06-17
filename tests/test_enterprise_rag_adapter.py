"""Smoke test for the Enterprise_RAG_Blueprint reference adapter.

Mocks the Enterprise_RAG chain imports so the test does not require the
full blueprint codebase. Verifies the adapter:
1. Declares expected capability flags.
2. Honors set_components by routing bundle slots into the chain.
3. Returns a valid RagSystemOutput with contexts + metadata.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_enterprise_modules(monkeypatch):
    """Inject fake ``chain.retriever`` and ``chain.load_chain`` modules."""
    fake_chain_pkg = types.ModuleType("chain")
    fake_retriever_mod = types.ModuleType("chain.retriever")
    fake_load_mod = types.ModuleType("chain.load_chain")

    fake_retriever = MagicMock()
    fake_retriever_mod.EnterpriseRetriever = MagicMock(return_value=fake_retriever)

    fake_chain_obj = MagicMock()
    fake_chain_obj.invoke.return_value = {
        "answer": "42",
        "source_documents": [
            MagicMock(page_content="ctx-A", metadata={"doc_id": "doc-a"}),
            MagicMock(page_content="ctx-B", metadata={"doc_id": "doc-b"}),
        ],
    }
    fake_load_mod.build_chain = MagicMock(return_value=fake_chain_obj)

    fake_chain_pkg.retriever = fake_retriever_mod
    fake_chain_pkg.load_chain = fake_load_mod

    monkeypatch.setitem(sys.modules, "chain", fake_chain_pkg)
    monkeypatch.setitem(sys.modules, "chain.retriever", fake_retriever_mod)
    monkeypatch.setitem(sys.modules, "chain.load_chain", fake_load_mod)
    return {
        "retriever": fake_retriever,
        "chain": fake_chain_obj,
        "build_chain": fake_load_mod.build_chain,
        "EnterpriseRetriever": fake_retriever_mod.EnterpriseRetriever,
    }


def test_supports_components(fake_enterprise_modules):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter
    adapter = EnterpriseRagAdapter(MagicMock())
    caps = adapter.supports_components()
    assert caps["chunker"] is True
    assert caps["embedder"] is True
    assert caps["retriever"] is False
    assert caps["reranker"] is True
    assert caps["llm"] is True
    assert caps["prompt"] is False


def test_prepare_passes_injected_components(fake_enterprise_modules):
    from benchmark.adapters.components import ComponentBundle
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    adapter = EnterpriseRagAdapter(MagicMock())
    adapter.set_components(ComponentBundle(
        chunker="FAKE_CHUNKER",
        embedder="FAKE_EMBEDDER",
        llm="FAKE_LLM",
    ))
    adapter.prepare(MagicMock(), data=[], corpus=[{"context": "doc1"}])

    assert adapter.retriever.chunker == "FAKE_CHUNKER"
    assert adapter.retriever.embedder == "FAKE_EMBEDDER"
    fake_enterprise_modules["build_chain"].assert_called_once()
    _, kwargs = fake_enterprise_modules["build_chain"].call_args
    assert kwargs["llm"] == "FAKE_LLM"


def test_answer_returns_valid_output(fake_enterprise_modules):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    adapter = EnterpriseRagAdapter(MagicMock())
    adapter.prepare(MagicMock(), data=[], corpus=[{"context": "doc1"}])
    out = adapter.answer({"question": "what?"}, MagicMock())
    assert out.answer == "42"
    assert out.contexts == ["ctx-A", "ctx-B"]
    assert {"doc_id": "doc-a"} in out.metadata
    assert out.total_seconds >= 0
