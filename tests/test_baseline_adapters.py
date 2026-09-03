from __future__ import annotations

from dataclasses import dataclass

import pytest

from benchmark.adapters import get_rag_adapter
from benchmark.adapters.baselines import (
    NoRetrievalAdapter,
    OracleRetrievalAdapter,
    RandomRetrievalAdapter,
)
from benchmark.generation import GenerationResult
from benchmark.prompt_templates.types import PromptTemplate


def _template() -> PromptTemplate:
    return PromptTemplate("test", "system", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")


@dataclass
class Config:
    llm_provider: str = "ollama"
    llm_model: str = "test-model"
    retrieval_top_k: int = 3
    max_new_tokens: int = 64
    prompt_template: str = "concise"
    llm_answer_strip_mode: str = "tags_only"
    llm_answer_value_fallback: bool = True

    def llm_base_url(self):
        return None

    def llm_api_key(self):
        return None


CORPUS = [
    {"context": f"doc {i}", "metadata": {"doc_id": f"doc-{i}"}}
    for i in range(10)
]


def _generation(monkeypatch, answer="42"):
    captured: dict = {}

    def fake_generate(llm, question, contexts, **kwargs):
        captured.update(question=question, contexts=contexts, kwargs=kwargs)
        return GenerationResult(
            answer=answer,
            ttft_seconds=0.01,
            total_seconds=0.02,
            token_count=1,
            tokens_per_second=50.0,
            gpu_usage=None,
            input_tokens=5,
            output_tokens=1,
            total_tokens=6,
            raw_content=answer,
        )

    monkeypatch.setattr("benchmark.adapters.baselines.generate_answer", fake_generate)
    return captured


def _generation_result_class():
    return GenerationResult


def test_registry_resolves_baseline_adapters():
    assert isinstance(get_rag_adapter(Config(rag_system_adapter="no_retrieval") if False else _cfg("no_retrieval")), NoRetrievalAdapter)
    assert isinstance(get_rag_adapter(_cfg("random_retrieval")), RandomRetrievalAdapter)
    assert isinstance(get_rag_adapter(_cfg("oracle_retrieval")), OracleRetrievalAdapter)


def _cfg(name: str) -> Config:
    config = Config()
    config.rag_system_adapter = name  # type: ignore[attr-defined]
    return config


def test_no_retrieval_answers_without_context(monkeypatch):
    captured = _generation(monkeypatch)
    adapter = NoRetrievalAdapter(llm=object(), prompt_template=object())
    adapter.prepare(Config(), [])
    output = adapter.answer(
        {"question": "What?", "ground_truth": "42"}, Config()
    )

    assert output.answer == "42"
    assert output.contexts == []
    assert captured["contexts"] == []
    assert output.diagnostics["baseline"] == "no_retrieval"


def test_random_retrieval_is_deterministic_per_question_and_uses_corpus(monkeypatch):
    _generation(monkeypatch)
    adapter = RandomRetrievalAdapter(llm=object(), prompt_template=_template())
    config = Config()
    adapter.prepare(config, [], corpus=CORPUS)

    first = adapter.answer({"question": "Q1"}, config)
    again = adapter.answer({"question": "Q1"}, config)
    other = adapter.answer({"question": "Q2"}, config)

    assert first.contexts == again.contexts
    assert len(first.contexts) == config.retrieval_top_k
    assert first.contexts != other.contexts or "Q1" == "Q2"
    assert all(ctx.startswith("doc ") for ctx in first.contexts)
    adapter.cleanup(None, config)
    with pytest.raises(ValueError):
        adapter.prepare(config, [], corpus=[])


def test_oracle_retrieval_uses_gold_context(monkeypatch):
    captured = _generation(monkeypatch)
    adapter = OracleRetrievalAdapter(llm=object(), prompt_template=_template())
    adapter.prepare(Config(), [])
    sample = {
        "question": "What?",
        "context": "gold passage",
        "ground_truth": "42",
        "metadata": {"gold_doc_id": "doc-7"},
    }
    output = adapter.answer(sample, Config())

    assert output.contexts == ["gold passage"]
    assert captured["contexts"] == ["gold passage"]
    assert output.metadata[0]["doc_id"] == "doc-7"

    with pytest.raises(ValueError):
        adapter.answer({"question": "What?", "context": ""}, Config())
