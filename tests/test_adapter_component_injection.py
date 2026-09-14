"""Component-injection tests driving the adapter seam directly.

The injection policy lives beside the component factory in the adapter
infrastructure (``benchmark.adapters.components.inject_components``) and is
exercised through the adapter interface — no entry-point imports, no
patching of ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from conftest import StubEmbedder

import benchmark.adapters.components as components_module
from benchmark.adapters import (
    generate_adapter,
    inject_components,
    prepare_adapter,
    retrieve_adapter,
)
from benchmark.adapters.components import ComponentBundle
from benchmark.adapters.internal import InternalRagAdapter
from benchmark.generation import GenerationResult


@dataclass
class InjectionConfig:
    """Dataclass config so capability override can use dataclasses.replace."""

    rag_adapter_accepts: str = ""
    dataset_name: str = "unit-test"
    dataset_subset: str = ""
    dataset_sample_size: int | None = None
    vector_db_backend: str = "chroma"
    embedding_provider: str = "stub"
    benchmark_stage: str = "all"
    lancedb_path: str = ".lancedb"
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 0
    embedding_model: str = "stub-embedder"
    llm_model: str = "stub-llm"
    reranker_model: str | None = None
    prompt_template: str = "concise"
    extra: dict[str, Any] = field(default_factory=dict)

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "retrieval_mode": "retrieval",
        "retrieval_top_k": 1,
        "retrieval_strategy": "similarity",
        "retrieval_fetch_k": None,
        "retrieval_mmr_lambda": 0.5,
        "retrieval_use_hyde": False,
        "retrieval_multihop": False,
        "retrieval_multihop_rounds": 2,
        "reranker_top_k": 3,
        "llm_answer_strip_mode": "tags_only",
        "llm_answer_value_fallback": True,
        "semantic_breakpoint_type": "percentile",
        "semantic_breakpoint_amount": 95,
    }

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
        except KeyError:
            pass
        try:
            return self._DEFAULTS[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FullAdapter:
    name = "full"

    def __init__(self) -> None:
        self.bundle: ComponentBundle | None = None

    def supports_components(self) -> dict[str, bool]:
        return {"llm": True, "chunker": True}

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle


class _LegacyAdapter:
    name = "legacy"


def test_inject_components_hands_built_slots_to_accepting_adapter(monkeypatch):
    adapter = _FullAdapter()
    fake_bundle = ComponentBundle(llm="FAKE_LLM", chunker="FAKE_CHUNKER")
    seen_configs: list[Any] = []

    def fake_build(config):
        seen_configs.append(config)
        return fake_bundle

    monkeypatch.setattr(components_module, "build_components", fake_build)

    cfg = InjectionConfig(rag_adapter_accepts="llm,chunker")
    inject_components(adapter, cfg)

    assert adapter.bundle is fake_bundle
    assert len(seen_configs) == 1
    # Capability override from supports_components wins over RAG_ADAPTER_ACCEPTS.
    assert seen_configs[0].rag_adapter_accepts == "chunker,llm"
    # The user config is not mutated.
    assert cfg.rag_adapter_accepts == "llm,chunker"


def test_inject_components_skips_adapters_without_set_components(monkeypatch):
    adapter = _LegacyAdapter()
    called = []
    monkeypatch.setattr(
        components_module, "build_components", lambda config: called.append(config)
    )

    inject_components(adapter, InjectionConfig())

    assert called == []
    assert not hasattr(adapter, "bundle")


def test_end_to_end_injection_through_adapter_lifecycle(monkeypatch):
    """Injected components drive a full prepare -> retrieve -> generate run."""

    def fake_build(config):
        return ComponentBundle(embedder=StubEmbedder())

    monkeypatch.setattr(components_module, "build_components", fake_build)

    adapter = InternalRagAdapter(generator=_stub_generator)
    cfg = InjectionConfig()
    inject_components(adapter, cfg)
    assert adapter._bundle.embedder is not None

    corpus = [
        {
            "id": "doc-a",
            "context": "The Eiffel Tower is in Paris, France.",
            "question": "how tall?",
            "ground_truth": "330m",
        }
    ]
    sample = {
        "question": "Where is the Eiffel Tower?",
        "ground_truth": "Paris",
        "metadata": {"gold_doc_id": "doc-a"},
    }

    target = prepare_adapter(adapter, cfg, [], corpus=corpus)
    retrieval = retrieve_adapter(adapter, target, sample, cfg)
    output = generate_adapter(adapter, target, sample, cfg)

    assert "Paris" in retrieval.contexts[0]
    assert output.answer == "stub answer to: Where is the Eiffel Tower?"


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


def _stub_generator(llm, question, contexts, **kwargs) -> GenerationResult:
    return stub_generator(llm, question, contexts, **kwargs)


@pytest.mark.parametrize("accepts", ["", "llm,chunker"])
def test_capability_override_covers_all_declared_slots(accepts: str):
    captured: dict = {}

    class RecordingAdapter(_FullAdapter):
        def set_components(self, bundle: ComponentBundle) -> None:
            captured["called"] = True

    adapter = RecordingAdapter()
    monkeypatch_build = ComponentBundle(llm="FAKE_LLM")
    from unittest import mock

    with mock.patch.object(
        components_module, "build_components", return_value=monkeypatch_build
    ) as bc_mock:
        inject_components(adapter, InjectionConfig(rag_adapter_accepts=accepts))

    bc_mock.assert_called_once()
    assert bc_mock.call_args.args[0].rag_adapter_accepts == "chunker,llm"
    assert captured["called"] is True
