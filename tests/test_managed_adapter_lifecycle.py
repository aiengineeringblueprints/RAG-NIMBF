from __future__ import annotations

import pytest

import main
import benchmark.orchestration.runner
from benchmark.adapters import (
    AdapterCapabilities,
    AdapterGenerationResult,
    PreparedTarget,
    RagSystemOutput,
    RetrievedChunk,
    RetrievalResult,
    UnsupportedAdapterCapability,
    cleanup_adapter,
    generate_adapter,
    get_adapter_capabilities,
    prepare_adapter,
    require_adapter_capabilities,
    retrieve_adapter,
)


class _LegacyAdapter:
    name = "legacy"

    def prepare(self, config, data, corpus=None):
        return None

    def answer(self, sample, config):
        return RagSystemOutput(answer="legacy answer", contexts=["legacy context"])


class _ManagedAdapter:
    name = "managed"

    def __init__(self):
        self.cleaned = None

    def capabilities(self):
        return AdapterCapabilities(
            ingestion=True,
            retrieval=True,
            generation=True,
            references=True,
            token_usage=True,
            cleanup=True,
        )

    def prepare(self, config, data, corpus=None):
        return PreparedTarget(target_id="target-1", dataset_ids=("dataset-1",))

    def retrieve(self, target, sample, config):
        return RetrievalResult(
            chunks=(
                RetrievedChunk(
                    text="managed context",
                    rank=1,
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_id="source-1",
                    score=0.9,
                ),
            ),
            total_seconds=0.1,
        )

    def generate(self, target, sample, config, retrieval=None):
        return AdapterGenerationResult(
            answer="managed answer",
            retrieval=retrieval or self.retrieve(target, sample, config),
            total_seconds=0.5,
            input_tokens=8,
            output_tokens=2,
            total_tokens=10,
        )

    def cleanup(self, target, config):
        self.cleaned = target


def test_legacy_adapter_keeps_prepare_and_answer_contract():
    adapter = _LegacyAdapter()
    target = prepare_adapter(adapter, object(), [], corpus=[])

    assert target.metadata == {"legacy_adapter": True}
    assert get_adapter_capabilities(adapter) == AdapterCapabilities(generation=True)
    assert (
        generate_adapter(adapter, target, {"question": "q"}, object()).answer
        == "legacy answer"
    )


def test_managed_lifecycle_normalizes_retrieval_and_generation():
    adapter = _ManagedAdapter()
    config = object()
    target = prepare_adapter(adapter, config, [], corpus=[])
    retrieval = retrieve_adapter(adapter, target, {"question": "q"}, config)
    output = generate_adapter(
        adapter, target, {"question": "q"}, config, retrieval=retrieval
    )
    cleanup_adapter(adapter, target, config)

    assert output.answer == "managed answer"
    assert output.contexts == ["managed context"]
    assert output.metadata == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "doc_id": "source-1",
            "source_id": "source-1",
            "score": 0.9,
            "rank": 1,
        }
    ]
    assert output.input_tokens == 8
    assert output.output_tokens == 2
    assert output.total_tokens == 10
    assert output.tokens_per_second == 4.0
    assert adapter.cleaned is target


def test_capability_check_fails_instead_of_silently_ignoring_setting():
    adapter = _LegacyAdapter()

    try:
        require_adapter_capabilities(adapter, "retrieval")
    except UnsupportedAdapterCapability as exc:
        assert "retrieval" in str(exc)
    else:
        raise AssertionError("unsupported retrieval capability was accepted")


def test_invalid_managed_result_is_rejected():
    class InvalidAdapter(_ManagedAdapter):
        def retrieve(self, target, sample, config):
            return {"chunks": []}

    adapter = InvalidAdapter()
    target = prepare_adapter(adapter, object(), [])

    try:
        retrieve_adapter(adapter, target, {"question": "q"}, object())
    except TypeError as exc:
        assert "RetrievalResult" in str(exc)
    else:
        raise AssertionError("invalid retrieval result was accepted")


def test_benchmark_wrapper_cleans_managed_target_after_failure(monkeypatch):
    adapter = _ManagedAdapter()
    target = PreparedTarget(target_id="target-on-error")

    def fail_after_prepare(*args, cleanup_registry, **kwargs):
        cleanup_registry.append((adapter, target, args[0]))
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        benchmark.orchestration.runner, "_run_single_benchmark_impl", fail_after_prepare
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        main.run_single_benchmark(object(), [])

    assert adapter.cleaned is target
