"""Structural identity of internal and external runs through the adapter seam.

Both runs drive ``main.run_single_benchmark`` — the internal pipeline via its
registered adapter and a stubbed external managed adapter — and must produce
structurally identical results: same per-sample answers/contexts/token
counts, the same stage-timing keys, and the same chunk accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest
from conftest import StubEmbedder

import benchmark.adapters.components as components_module
import benchmark.generation
from benchmark.adapters import (
    AdapterGenerationResult,
    PreparedTarget,
    RetrievalResult,
    RetrievedChunk,
    register_rag_adapter,
)
from benchmark.adapters.components import ComponentBundle

main = pytest.importorskip("main")

DOC_A = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."


def _stub_generate_answer(llm, question, contexts, **kwargs):
    return benchmark.generation.GenerationResult(
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


class _StubExternalAdapter:
    """Managed adapter mimicking the stubbed internal pipeline outputs."""

    name = "stub-external"

    def __init__(self, config: Any) -> None:
        self.cleaned = False

    def capabilities(self):
        from benchmark.adapters import AdapterCapabilities

        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            token_usage=True,
            cleanup=True,
        )

    def prepare(self, config, data, corpus=None):
        return PreparedTarget(
            target_id="stub-target",
            metadata={"chunk_count": 1},
        )

    def retrieve(self, target, sample, config):
        return RetrievalResult(
            chunks=(RetrievedChunk(text=DOC_A, rank=1),),
            total_seconds=0.02,
        )

    def generate(self, target, sample, config, retrieval=None):
        retrieval = retrieval or self.retrieve(target, sample, config)
        return AdapterGenerationResult(
            answer=f"stub answer to: {sample['question']}",
            retrieval=retrieval,
            ttft_seconds=0.01,
            total_seconds=0.05,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            diagnostics={
                "stage_timings": {"stub_stage": 0.25},
                "some_internal_key": "orchestrator must never read this",
            },
        )

    def aggregate_metrics(self, diagnostics: list[dict[str, Any]]):
        if not any(diagnostics):
            return None
        return {"stub_stage_seconds": sum(
            float(d.get("stage_timings", {}).get("stub_stage", 0.0))
            for d in diagnostics
        )}

    def cleanup(self, target, config):
        self.cleaned = True


@dataclass
class UniformityConfig:
    name: str = "uniformity"
    rag_system_adapter: str = "internal"
    retrieval_mode: str = "retrieval"
    benchmark_stage: str = "all"
    ragas_enabled: bool = False
    custom_metrics_enabled: bool = False
    custom_retrieval_metrics_mode: str = "off"
    llm_model: str = "stub-llm"
    llm_provider: str = "stub"
    embedding_model: str = "stub-embedder"
    embedding_provider: str = "stub"
    prompt_template: str = "concise"
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 0
    reranker_model: str | None = None
    retrieval_strategy: str = "similarity"
    retrieval_top_k: int = 1
    retrieval_use_hyde: bool = False
    retrieval_multihop: bool = False
    dataset_name: str = "unit-test"
    dataset_subset: str = ""
    dataset_sample_size: int | None = None
    vector_db_backend: str = "chroma"
    lancedb_path: str = ".lancedb"
    rag_adapter_accepts: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "max_new_tokens": 64,
        "llm_answer_strip_mode": "tags_only",
        "llm_answer_value_fallback": True,
        "reranker_top_k": 3,
        "retrieval_fetch_k": None,
        "retrieval_mmr_lambda": 0.5,
        "retrieval_multihop_rounds": 2,
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


CORPUS = [
    {
        "id": "doc-a",
        "context": DOC_A,
        "question": "where is the tower?",
        "ground_truth": "Paris",
    }
]

SAMPLES = [
    {"question": "Where is the Eiffel Tower?", "ground_truth": "Paris"},
]


@pytest.fixture
def stubbed_seam(monkeypatch):
    # The component-injection seam hands the internal pipeline its stub
    # embedder/LLM; the generator itself is stubbed at the generation seam.
    monkeypatch.setattr(
        components_module,
        "build_components",
        lambda config, accepts=None: ComponentBundle(
            embedder=StubEmbedder(), llm=object()
        ),
    )
    monkeypatch.setattr(
        benchmark.generation, "generate_answer", _stub_generate_answer
    )
    register_rag_adapter("stub-external", _StubExternalAdapter)
    yield
    from benchmark.adapters import RAG_ADAPTER_REGISTRY

    RAG_ADAPTER_REGISTRY.pop("stub-external", None)


def _per_sample_projection(result):
    return [
        (s.question, s.answer, s.contexts, s.input_tokens, s.output_tokens)
        for s in result.per_sample
    ]


def test_internal_and_stubbed_external_runs_are_structurally_identical(
    stubbed_seam,
):
    internal = main.run_single_benchmark(UniformityConfig(), SAMPLES, corpus=CORPUS)
    external = main.run_single_benchmark(
        UniformityConfig(rag_system_adapter="stub-external"),
        SAMPLES,
        corpus=CORPUS,
    )

    assert _per_sample_projection(internal) == _per_sample_projection(external)
    assert internal.num_chunks == external.num_chunks == 1
    assert internal.num_questions == external.num_questions == len(SAMPLES)
    # Orchestrator-owned stage keys are identical; adapter-contributed keys
    # (e.g. "stub_stage") are adapter-specific by design.
    orchestrator_keys = {"adapter_prepare", "external_rag", "total"}
    assert orchestrator_keys <= set(internal.stage_timings)
    assert orchestrator_keys <= set(external.stage_timings)
    assert set(internal.stage_timings) - orchestrator_keys == set()
    assert internal.adapter_metrics is None


def test_adapter_diagnostics_surface_in_run_result(stubbed_seam):
    """Stage-timing contributions and run metrics flow through the seam."""
    result = main.run_single_benchmark(
        UniformityConfig(rag_system_adapter="stub-external"),
        SAMPLES,
        corpus=CORPUS,
    )

    assert result.stage_timings["stub_stage"] == pytest.approx(0.25)
    assert result.adapter_metrics == {"stub_stage_seconds": 0.25}
    # Per-sample diagnostics are preserved as an opaque blob for reports.
    assert result.per_sample[0].adapter_diagnostics["some_internal_key"] == (
        "orchestrator must never read this"
    )


def test_injected_stub_embedder_drives_internal_index(stubbed_seam, monkeypatch):
    built: dict[str, Any] = {}

    def fake_build_components(config, accepts=None):
        built["called"] = True
        return ComponentBundle(embedder=StubEmbedder(), llm=object())

    monkeypatch.setattr(
        components_module, "build_components", fake_build_components
    )

    result = main.run_single_benchmark(
        UniformityConfig(), SAMPLES, corpus=CORPUS
    )

    assert built["called"]
    assert result.per_sample[0].contexts
    assert result.per_sample[0].contexts[0] == DOC_A
