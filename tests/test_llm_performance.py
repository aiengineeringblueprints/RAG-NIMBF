from pathlib import Path

import pytest

from benchmark.llm_performance import (
    PerformanceCall,
    _sanitize_error,
    performance_from_generation,
    performance_cache_key,
    run_llm_performance_benchmark,
    save_llm_performance_result,
)
from tests.test_config import _make_config
from benchmark.reporting.models import PerSampleResult


def test_integrated_performance_profiles_sequential_and_parallel(monkeypatch):
    config = _make_config(
        llm_performance_enabled=True,
        llm_performance_call_counts=(1, 3),
        llm_performance_warmup=False,
    )

    def fake_call(config, prompt, *, max_tokens=None):
        return PerformanceCall(
            latency_s=1.0,
            success=True,
            tokens_in=5,
            tokens_out=10,
            ttft_s=0.1,
            tpot_s=0.01,
            itl_p50_s=0.01,
            itl_p95_s=0.02,
        )

    monkeypatch.setattr("benchmark.llm_performance._measure_call", fake_call)

    result = run_llm_performance_benchmark(config, ["q1", "q2", "q3"])

    assert result.call_counts == (1, 3)
    assert result.metrics["n3_sequential_success_rate"] == 1.0
    assert result.metrics["n3_parallel_success_rate"] == 1.0
    assert result.metrics["n3_parallel_output_tokens_per_s"] == pytest.approx(10.0)
    assert result.metrics["n3_sequential_latency_p95_s"] == pytest.approx(1.0)
    assert len(result.calls) == 8


def test_performance_uses_available_prompts_when_profile_is_larger(monkeypatch):
    config = _make_config(
        llm_performance_call_counts=(10,),
        llm_performance_warmup=False,
    )
    monkeypatch.setattr(
        "benchmark.llm_performance._measure_call",
        lambda *args, **kwargs: PerformanceCall(latency_s=0.1, success=True),
    )

    result = run_llm_performance_benchmark(config, ["q1", "q2"])

    assert result.metrics["n10_actual_calls"] == 2
    assert result.metrics["n10_sequential_calls"] == 2
    assert result.metrics["n10_parallel_calls"] == 2


def test_cache_key_changes_with_endpoint_or_token_limit():
    config = _make_config(llm_performance_call_counts=(1, 3))
    other_tokens = _make_config(
        llm_performance_call_counts=(1, 3),
        max_new_tokens=512,
    )
    other_endpoint = _make_config(
        llm_performance_call_counts=(1, 3),
        llm_ollama_base_url="http://other:11434",
    )

    assert performance_cache_key(config) != performance_cache_key(other_tokens)
    assert performance_cache_key(config) != performance_cache_key(other_endpoint)


def test_detailed_result_is_saved_without_api_key(monkeypatch, tmp_path: Path):
    config = _make_config(
        ollama_api_key="secret-value",
        llm_performance_call_counts=(1,),
        llm_performance_warmup=False,
    )
    monkeypatch.setattr(
        "benchmark.llm_performance._measure_call",
        lambda *args, **kwargs: PerformanceCall(latency_s=0.1, success=True),
    )
    result = run_llm_performance_benchmark(config, ["q1"])

    path = save_llm_performance_result(result, tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "secret-value" not in content
    assert result.model in content


def test_persisted_endpoint_redacts_url_credentials(monkeypatch, tmp_path: Path):
    config = _make_config(
        ollama_base_url="https://user:password@example.test:11434/ollama",
        llm_performance_call_counts=(1,),
        llm_performance_warmup=False,
    )
    monkeypatch.setattr(
        "benchmark.llm_performance._measure_call",
        lambda *args, **kwargs: PerformanceCall(latency_s=0.1, success=True),
    )

    result = run_llm_performance_benchmark(config, ["q1"])
    path = save_llm_performance_result(result, tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "user" not in result.base_url
    assert "password" not in content
    assert result.base_url == "https://example.test:11434/ollama"


def test_error_sanitization_removes_api_key_and_url_credentials():
    config = _make_config(
        ollama_api_key="secret-value",
        ollama_base_url="https://user:password@example.test/ollama",
    )
    error = RuntimeError(
        "request to https://user:password@example.test/ollama "
        "with secret-value failed"
    )

    sanitized = _sanitize_error(config, error)

    assert "secret-value" not in sanitized
    assert "password" not in sanitized
    assert "<redacted>" in sanitized


def test_generation_metrics_reuse_real_rag_samples_without_calls(monkeypatch):
    config = _make_config(llm_performance_source="generation")
    monkeypatch.setattr(
        "benchmark.llm_performance._measure_call",
        lambda *args, **kwargs: pytest.fail("generation metrics replayed the LLM"),
    )
    samples = (
        PerSampleResult(
            question="q1",
            ground_truth="a1",
            answer="generated",
            contexts=("retrieved context",),
            ttft_seconds=0.2,
            total_seconds=1.2,
            token_count=20,
            tokens_per_second=20 / 1.2,
            gpu_usage=None,
            ragas_scores={},
            input_tokens=100,
            output_tokens=20,
        ),
        PerSampleResult(
            question="q2",
            ground_truth="a2",
            answer="generated",
            contexts=("other context",),
            ttft_seconds=0.4,
            total_seconds=2.0,
            token_count=30,
            tokens_per_second=15.0,
            gpu_usage=None,
            ragas_scores={},
            input_tokens=120,
            output_tokens=30,
        ),
    )

    result = performance_from_generation(
        config,
        samples,
        generation_wall_s=3.5,
    )

    assert result.metrics["generation_actual_calls"] == 2
    assert result.metrics["generation_sequential_latency_mean_s"] == pytest.approx(1.6)
    assert result.metrics["generation_sequential_latency_p95_s"] == pytest.approx(2.0)
    assert result.metrics["generation_sequential_ttft_mean_s"] == pytest.approx(0.3)
    assert result.metrics["generation_sequential_wall_s"] == pytest.approx(3.5)
    assert result.metrics["generation_tpot_is_estimated"] == 1.0
    assert len(result.calls) == 2
