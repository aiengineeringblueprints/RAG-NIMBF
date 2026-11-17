"""Post-run MLflow scorer step judged by the critic LLM (issue #44)."""

from __future__ import annotations

import os
from dataclasses import replace

import mlflow

from benchmark.mlflow_scorers import (
    JudgeEndpoint,
    build_scorers,
    resolve_judge_endpoint,
    score_run_traces,
)
from config import MLFLOW_SCORER_NAMES
from test_config import _make_config
from test_mlflow_tracing import (  # noqa: F401 - pytest fixtures
    CORPUS,
    SAMPLES,
    _inject,
    _traced,
    generator_calls,
    tracking_store,
)


class TestJudgeEndpoint:
    def test_ollama_critic_uses_openai_compatible_v1_route(self):
        config = _make_config(
            eval_critic_llm="ollama:gemma3:12b",
            eval_critic_ollama_base_url="http://spark:11434/",
        )

        endpoint = resolve_judge_endpoint(config)

        assert endpoint.model_uri == "openai:/gemma3:12b"
        assert endpoint.base_url == "http://spark:11434/v1"

    def test_unprefixed_critic_falls_back_to_shared_ollama_url(self):
        config = _make_config(
            eval_critic_llm="gemma3:12b",
            eval_critic_ollama_base_url=None,
            ollama_base_url="http://localhost:11434",
        )

        endpoint = resolve_judge_endpoint(config)

        assert endpoint.model_uri == "openai:/gemma3:12b"
        assert endpoint.base_url == "http://localhost:11434/v1"

    def test_openai_compatible_critic_keeps_its_base_url_and_key(self):
        config = _make_config(
            eval_critic_llm="openai:Qwen/Qwen3-32B-AWQ",
            eval_critic_openai_compat_base_url="https://vllm.example/v1/",
            eval_critic_openai_compat_api_key="secret",
        )

        endpoint = resolve_judge_endpoint(config)

        assert endpoint.model_uri == "openai:/Qwen/Qwen3-32B-AWQ"
        assert endpoint.base_url == "https://vllm.example/v1"
        assert endpoint.api_key == "secret"

    def test_keyless_local_endpoint_gets_placeholder_key(self):
        # MLflow's OpenAI provider refuses to start without OPENAI_API_KEY.
        config = _make_config(
            eval_critic_llm="ollama:gemma3:12b",
            eval_critic_ollama_api_key=None,
            ollama_api_key=None,
        )

        assert resolve_judge_endpoint(config).api_key

    def test_critic_inference_settings_are_reused(self):
        config = _make_config(
            eval_critic_llm="ollama:gemma3:12b",
            eval_critic_temperature=0.0,
            eval_critic_max_tokens=2048,
        )

        endpoint = resolve_judge_endpoint(config)

        assert endpoint.inference_params == {"temperature": 0.0, "max_tokens": 2048}

    def test_null_temperature_is_omitted(self):
        config = replace(
            _make_config(eval_critic_llm="ollama:gemma3:12b"),
            eval_critic_temperature=None,
        )

        assert "temperature" not in resolve_judge_endpoint(config).inference_params


# ---------------------------------------------------------------------------
# Scoring a finished run's traces
# ---------------------------------------------------------------------------

ENDPOINT = JudgeEndpoint(
    model_uri="openai:/gemma3:12b",
    base_url="http://judge.local:11434/v1",
    api_key="judge-key",
)


def _finished_config_run(monkeypatch):
    from benchmark.orchestration.runner import run_single_benchmark
    from benchmark.tracking import log_benchmark_run

    _inject(monkeypatch)
    with mlflow.start_run(run_name="parent"):
        result = run_single_benchmark(_traced(), SAMPLES, corpus=CORPUS)
        run_id = log_benchmark_run(result, nested=True)
    return run_id


def _reference_judge(seen_env):
    """Deterministic stand-in for an LLM judge: reads trace + expectations."""
    from mlflow.genai.scorers import scorer

    @scorer(name="reference_judge")
    def judge(trace, expectations) -> float:
        seen_env.append((os.environ.get("OPENAI_API_BASE"), os.environ.get("OPENAI_API_KEY")))
        retriever = [s for s in trace.data.spans if s.span_type == "RETRIEVER"]
        has_context = bool(retriever and retriever[0].outputs)
        known = expectations.get("expected_response") in {s["ground_truth"] for s in SAMPLES}
        return 1.0 if has_context and known else 0.0

    return judge


def test_scores_land_on_the_config_run_without_rerunning(
    tracking_store, generator_calls, monkeypatch
):
    run_id = _finished_config_run(monkeypatch)
    pipeline_calls = len(generator_calls)
    seen_env: list = []

    metrics = score_run_traces(
        run_id,
        ("stub",),
        ENDPOINT,
        factories={"stub": lambda endpoint: _reference_judge(seen_env)},
    )

    assert metrics == {"mlflow_judge_stub_mean": 1.0}
    logged = mlflow.get_run(run_id).data.metrics
    assert logged["mlflow_judge_stub_mean"] == 1.0
    assert len(generator_calls) == pipeline_calls
    assert len(seen_env) == len(SAMPLES)


def test_judge_reaches_the_critic_endpoint_and_env_is_restored(
    tracking_store, generator_calls, monkeypatch
):
    run_id = _finished_config_run(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    seen_env: list = []

    score_run_traces(
        run_id,
        ("stub",),
        ENDPOINT,
        factories={"stub": lambda endpoint: _reference_judge(seen_env)},
    )

    assert set(seen_env) == {("http://judge.local:11434/v1", "judge-key")}
    assert os.environ["OPENAI_API_KEY"] == "real-openai-key"
    assert "OPENAI_API_BASE" not in os.environ


def test_judge_failure_is_non_fatal(tracking_store, generator_calls, monkeypatch):
    run_id = _finished_config_run(monkeypatch)

    def broken(endpoint):
        raise RuntimeError("judge unreachable")

    assert score_run_traces(run_id, ("stub",), ENDPOINT, factories={"stub": broken}) == {}
    assert mlflow.get_run(run_id).data.tags["mlflow_scorers"] == "failed"


def test_every_configured_scorer_name_builds_an_mlflow_scorer():
    from mlflow.genai.scorers import Scorer

    endpoint = resolve_judge_endpoint(_make_config(eval_critic_llm="ollama:gemma3:12b"))

    scorers = build_scorers(MLFLOW_SCORER_NAMES, endpoint)

    assert set(scorers) == set(MLFLOW_SCORER_NAMES)
    assert all(isinstance(s, Scorer) for s in scorers.values())
