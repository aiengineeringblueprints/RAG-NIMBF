"""Tests for the reference Python RAG plugin (examples/python_rag_plugin).

Run with ``python -m pytest tests/test_demo_python_rag_plugin.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


@pytest.fixture(scope="module")
def adapter_factory():
    from benchmark.adapters import RAG_ADAPTER_REGISTRY

    if "demo_python" not in RAG_ADAPTER_REGISTRY:
        import python_rag_plugin.demo_adapter  # noqa: F401  side-effect: register
    return RAG_ADAPTER_REGISTRY["demo_python"]


class _StubConfig:
    retrieval_top_k = 2
    name = "stub"


@pytest.fixture()
def corpus():
    return [
        {
            "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
            "metadata": {"doc_id": "eiffel"},
        },
        {
            "context": "Mount Everest is the highest mountain on Earth, sitting on the Nepal-China border.",
            "metadata": {"doc_id": "everest"},
        },
        {
            "context": "The Great Barrier Reef stretches along the north-east coast of Australia.",
            "metadata": {"doc_id": "reef"},
        },
    ]


def test_plugin_registered(adapter_factory):
    assert callable(adapter_factory)


def test_prepare_indexes_unique_doc_ids(adapter_factory, corpus):
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=[], corpus=corpus)
    assert {d.doc_id for d in adapter.docs} == {"eiffel", "everest", "reef"}


def test_prepare_dedups_doc_ids(adapter_factory, corpus):
    dup = corpus + [corpus[0]]
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=[], corpus=dup)
    assert len(adapter.docs) == 3


def test_retrieval_ranks_relevant_doc_first(adapter_factory, corpus):
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=[], corpus=corpus)
    sample = {"question": "Where is the Eiffel Tower located?"}
    out = adapter.answer(sample, _StubConfig())
    assert out.metadata, "expected at least one hit"
    assert out.metadata[0]["doc_id"] == "eiffel"
    assert out.contexts
    assert out.contexts[0] == corpus[0]["context"]


def test_answer_returns_full_contract(adapter_factory, corpus):
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=[], corpus=corpus)
    out = adapter.answer(
        {"question": "Where is the Eiffel Tower located?"},
        _StubConfig(),
    )

    assert isinstance(out.answer, str)
    assert out.answer.strip(), "extractive answer should be non-empty"
    assert out.answer_valid is True
    assert len(out.contexts) == len(out.metadata)
    assert all("doc_id" in m and "score" in m for m in out.metadata)
    assert out.total_seconds >= 0.0
    assert out.token_count == out.input_tokens + out.output_tokens
    assert out.total_tokens == out.input_tokens + out.output_tokens
    assert out.token_count > 0
    assert out.tokens_per_second >= 0.0


def test_empty_question_yields_invalid_answer(adapter_factory, corpus):
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=[], corpus=corpus)
    out = adapter.answer({"question": ""}, _StubConfig())
    assert out.answer_valid is False
    assert out.contexts == []


def test_missing_corpus_uses_data(adapter_factory):
    data = [
        {
            "question": "q1",
            "ground_truth": "g1",
            "context": "Python is a programming language with dynamic typing.",
            "metadata": {"doc_id": "py"},
        }
    ]
    adapter = adapter_factory(_StubConfig())
    adapter.prepare(_StubConfig(), data=data, corpus=None)
    assert len(adapter.docs) == 1
    out = adapter.answer({"question": "What is Python?"}, _StubConfig())
    assert out.metadata[0]["doc_id"] == "py"
