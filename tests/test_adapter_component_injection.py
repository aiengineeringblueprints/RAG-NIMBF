"""Component-injection policy tests driving the adapter seam directly.

The decide-whether-and-what-to-inject policy lives in the adapter
infrastructure (``benchmark.adapters.components``). The orchestrator hands
over the adapter and a component bundle and nothing else; no code path
mutates the config object. Tests exercise the policy through the adapter
interface — no entry-point imports, no patching of ``main``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from conftest import StubEmbedder

import benchmark.adapters.components as components_module
from benchmark.adapters import (
    build_components,
    generate_adapter,
    inject_components,
    prepare_adapter,
    resolve_injection_slots,
    retrieve_adapter,
)
from benchmark.adapters.components import ComponentBundle, build_injected_components
from benchmark.adapters.internal import InternalRagAdapter
from benchmark.generation import GenerationResult


@dataclass
class InjectionConfig:
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
    llm_provider: str = "stub"
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
    """Declares support for every slot."""

    name = "full"

    def __init__(self) -> None:
        self.bundle: ComponentBundle | None = None

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": True,
            "embedder": True,
            "retriever": True,
            "reranker": True,
            "llm": True,
            "prompt": True,
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle


class _PartialAdapter:
    """Declares support for some slots."""

    name = "partial"

    def __init__(self) -> None:
        self.bundle: ComponentBundle | None = None

    def supports_components(self) -> dict[str, bool]:
        return {"llm": True, "chunker": True, "embedder": False}

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle


class _NoSlotAdapter:
    """Declares the protocol but accepts nothing."""

    name = "noslot"

    def __init__(self) -> None:
        self.bundle: ComponentBundle | None = None

    def supports_components(self) -> dict[str, bool]:
        return {"llm": False}

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle


class _BlackBoxAdapter:
    """Pure black-box: no component protocol at all."""

    name = "blackbox"


# resolve_injection_slots: the policy at the adapter seam


def test_policy_all_slots():
    assert resolve_injection_slots(_FullAdapter(), InjectionConfig()) == {
        "chunker",
        "embedder",
        "retriever",
        "reranker",
        "llm",
        "prompt",
    }


def test_policy_some_slots_only_true_capabilities():
    slots = resolve_injection_slots(_PartialAdapter(), InjectionConfig())
    assert slots == {"llm", "chunker"}


def test_policy_no_slots_resolves_empty():
    assert resolve_injection_slots(_NoSlotAdapter(), InjectionConfig()) == set()


def test_policy_black_box_adapter_resolves_empty():
    assert resolve_injection_slots(_BlackBoxAdapter(), InjectionConfig()) == set()


def test_policy_user_intent_override_narrows_adapter_capability():
    cfg = InjectionConfig(rag_adapter_accepts="llm")
    assert resolve_injection_slots(_PartialAdapter(), cfg) == {"llm"}


def test_policy_user_intent_override_cannot_add_unsupported_slots():
    cfg = InjectionConfig(rag_adapter_accepts="llm,reranker")
    assert resolve_injection_slots(_PartialAdapter(), cfg) == {"llm"}


def test_policy_never_mutates_the_config_object():
    cfg = InjectionConfig(rag_adapter_accepts="llm")
    resolve_injection_slots(_PartialAdapter(), cfg)
    assert cfg.rag_adapter_accepts == "llm"


# build_injected_components + inject_components: the orchestrator handover


def test_black_box_adapter_gets_no_bundle():
    adapter = _BlackBoxAdapter()
    bundle = build_injected_components(adapter, InjectionConfig())
    assert bundle == ComponentBundle()
    inject_components(adapter, bundle)  # must be a no-op, not an error


def test_no_slot_adapter_gets_empty_bundle():
    adapter = _NoSlotAdapter()
    bundle = build_injected_components(adapter, InjectionConfig())
    assert bundle == ComponentBundle()
    inject_components(adapter, bundle)
    assert adapter.bundle is None  # empty bundle is never delivered


@pytest.fixture
def fake_factories(monkeypatch):
    monkeypatch.setattr(
        components_module, "get_llm", lambda **kwargs: "FAKE_LLM", raising=False
    )
    monkeypatch.setattr(
        components_module,
        "get_embedding_model",
        lambda *args, **kwargs: "FAKE_EMBEDDER",
        raising=False,
    )
    monkeypatch.setattr(
        components_module,
        "get_chunker",
        lambda *args, **kwargs: "FAKE_CHUNKER",
        raising=False,
    )


def test_full_adapter_receives_populated_bundle(fake_factories):
    adapter = _FullAdapter()
    cfg = InjectionConfig()
    bundle = build_injected_components(adapter, cfg)
    inject_components(adapter, bundle)
    assert adapter.bundle is bundle
    assert adapter.bundle.llm == "FAKE_LLM"
    assert adapter.bundle.chunker == "FAKE_CHUNKER"
    assert adapter.bundle.embedder == "FAKE_EMBEDDER"


def test_partial_adapter_gets_only_declared_slots_even_when_user_lists_more(fake_factories):
    adapter = _PartialAdapter()
    cfg = InjectionConfig(rag_adapter_accepts="llm,chunker,embedder,reranker")
    bundle = build_injected_components(adapter, cfg)
    inject_components(adapter, bundle)
    assert adapter.bundle.llm == "FAKE_LLM"
    assert adapter.bundle.chunker == "FAKE_CHUNKER"
    assert adapter.bundle.embedder is None  # declared False
    assert adapter.bundle.reranker is None  # not declared


def test_build_components_builds_exactly_the_requested_slots(fake_factories):
    bundle = build_components(InjectionConfig(), {"llm", "embedder"})
    assert bundle.llm == "FAKE_LLM"
    assert bundle.embedder == "FAKE_EMBEDDER"
    assert bundle.chunker is None
    assert bundle.prompt_template is None


def test_user_intent_override_restores_llm_only_injection(fake_factories):
    adapter = _PartialAdapter()
    cfg = InjectionConfig(rag_adapter_accepts="llm")
    bundle = build_injected_components(adapter, cfg)
    inject_components(adapter, bundle)
    assert adapter.bundle.llm is not None
    assert adapter.bundle.chunker is None


def test_end_to_end_injection_through_adapter_lifecycle(monkeypatch):
    monkeypatch.setattr(
        components_module,
        "build_components",
        lambda config, accepts=None: ComponentBundle(embedder=StubEmbedder()),
    )
    adapter = InternalRagAdapter(generator=_stub_generator)
    cfg = InjectionConfig()
    bundle = build_injected_components(adapter, cfg)
    inject_components(adapter, bundle)
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
