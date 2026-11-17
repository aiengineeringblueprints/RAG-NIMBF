"""Post-run MLflow scorer step judged by the critic LLM (issue #44)."""

from __future__ import annotations

import os
from dataclasses import replace

import mlflow

import benchmark.mlflow_scorers as mlflow_scorers_module
from benchmark.mlflow_scorers import (
    JudgeEndpoint,
    build_scorers,
    judge_environment,
    resolve_judge_endpoint,
    score_config_run,
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

    def test_hosted_openai_critic_keeps_the_real_openai_environment(self, monkeypatch):
        # openai: critic without a compat URL = the actual OpenAI API; the
        # user's own OPENAI_API_KEY must survive scoring untouched.
        monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        config = _make_config(eval_critic_llm="openai:gpt-4o-mini")

        endpoint = resolve_judge_endpoint(config)
        with judge_environment(endpoint):
            seen = (os.environ.get("OPENAI_API_BASE"), os.environ.get("OPENAI_API_KEY"))

        assert endpoint.model_uri == "openai:/gpt-4o-mini"
        assert seen == (None, "real-openai-key")

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


def test_endpoint_resolution_failure_is_non_fatal(monkeypatch):
    def unusable(config):
        raise ValueError("no usable critic endpoint")

    monkeypatch.setattr(mlflow_scorers_module, "resolve_judge_endpoint", unusable)
    config = _make_config(mlflow_tracing_enabled=True, mlflow_scorers=("correctness",))

    assert score_config_run("run-1", config) == {}


def test_scorer_without_a_mean_metric_is_reported(
    tracking_store, generator_calls, monkeypatch, caplog
):
    from mlflow.genai.scorers import scorer

    run_id = _finished_config_run(monkeypatch)

    @scorer(name="label_judge")
    def label_judge(outputs) -> str:
        return "free-text label without a mean"

    with caplog.at_level("WARNING", logger="benchmark.mlflow_scorers"):
        metrics = score_run_traces(
            run_id, ("label",), ENDPOINT, factories={"label": lambda e: label_judge}
        )

    assert metrics == {}
    assert mlflow.get_run(run_id).data.tags["mlflow_scorers"] == "no_scores"
    assert "label" in caplog.text


# ---------------------------------------------------------------------------
# answer_correctness: RAGAS-style claim F1 (issue #45)
# ---------------------------------------------------------------------------

QUESTION = {"question": "Where and when was Marie Curie born?"}
REFERENCE = {"expected_response": "Marie Curie was born in Warsaw in 1867."}


def _claims_judge(tp: int, fp: int, fn: int, rationale: str = "claims listed"):
    """Stub for the claim-classifying LLM judge with a fixed verdict."""
    from mlflow.entities import Feedback

    calls: list = []

    def judge(*, inputs=None, outputs=None, expectations=None):
        calls.append((inputs, outputs, expectations))
        return Feedback(
            name="answer_correctness_claims",
            value={"true_positives": tp, "false_positives": fp, "false_negatives": fn},
            rationale=rationale,
        )

    judge.calls = calls
    return judge


def _score(judge, answer: str):
    from benchmark.mlflow_scorers import answer_correctness_scorer

    scorer = answer_correctness_scorer(judge)
    return scorer(inputs=QUESTION, outputs=answer, expectations=REFERENCE)


class TestAnswerCorrectness:
    def test_complete_answer_scores_one(self):
        feedback = _score(_claims_judge(tp=2, fp=0, fn=0), "Warsaw, in 1867.")

        assert feedback.value == 1.0

    def test_missing_fact_is_penalised(self):
        # Answer: "Warsaw." -> birthplace matches, the year is missing.
        feedback = _score(_claims_judge(tp=1, fp=0, fn=1), "Warsaw.")

        assert feedback.value == 2 / 3

    def test_wrong_added_content_is_penalised(self):
        # All reference facts present, plus a false claim. MLflow's built-in
        # Correctness only checks presence and would pass this answer.
        feedback = _score(
            _claims_judge(tp=2, fp=1, fn=0),
            "Warsaw, in 1867. She was born a French citizen.",
        )

        assert feedback.value == 0.8

    def test_wrong_answer_scores_zero(self):
        feedback = _score(_claims_judge(tp=0, fp=2, fn=2), "Paris, in 1901.")

        assert feedback.value == 0.0

    def test_judge_sees_question_answer_and_reference(self):
        judge = _claims_judge(tp=2, fp=0, fn=0)

        _score(judge, "Warsaw, in 1867.")

        assert judge.calls == [(QUESTION, "Warsaw, in 1867.", REFERENCE)]

    def test_rationale_carries_the_judge_reasoning_and_counts(self):
        feedback = _score(
            _claims_judge(tp=1, fp=0, fn=1, rationale="FN: born in 1867"), "Warsaw."
        )

        assert "FN: born in 1867" in feedback.rationale
        assert "TP=1" in feedback.rationale and "FN=1" in feedback.rationale

    def test_unusable_judge_output_becomes_an_error_not_a_score(self):
        from mlflow.entities import Feedback

        def garbled(**kwargs):
            return Feedback(name="answer_correctness_claims", value="looks fine to me")

        feedback = _score(garbled, "Warsaw, in 1867.")

        assert feedback.value is None
        assert feedback.error is not None

    def test_no_claims_on_either_side_is_an_error_not_a_perfect_score(self):
        feedback = _score(_claims_judge(tp=0, fp=0, fn=0), "")

        assert feedback.value is None
        assert feedback.error is not None

    def test_missing_reference_is_an_error_without_calling_the_judge(self):
        from benchmark.mlflow_scorers import answer_correctness_scorer

        judge = _claims_judge(tp=0, fp=2, fn=0)
        scorer = answer_correctness_scorer(judge)

        feedback = scorer(inputs=QUESTION, outputs="Warsaw, in 1867.", expectations={})

        assert feedback.value is None
        assert feedback.error is not None
        assert judge.calls == []


def test_answer_correctness_judge_reaches_the_critic(monkeypatch):
    # The factory wires make_judge to the critic endpoint and settings.
    captured: dict = {}

    def fake_make_judge(**kwargs):
        captured.update(kwargs)
        return _claims_judge(tp=1, fp=0, fn=0)

    monkeypatch.setattr("mlflow.genai.judges.make_judge", fake_make_judge)
    endpoint = JudgeEndpoint(
        model_uri="openai:/gemma3:12b",
        base_url="http://judge.local:11434/v1",
        api_key="judge-key",
        inference_params={"temperature": 0.0, "max_tokens": 2048},
    )

    scorer = build_scorers(("answer_correctness",), endpoint)["answer_correctness"]
    feedback = scorer(inputs=QUESTION, outputs="Warsaw, in 1867.", expectations=REFERENCE)

    assert scorer.name == "answer_correctness"
    assert feedback.value == 1.0
    assert captured["model"] == "openai:/gemma3:12b"
    assert captured["inference_params"] == {"temperature": 0.0, "max_tokens": 2048}
    for variable in ("{{ inputs }}", "{{ outputs }}", "{{ expectations }}"):
        assert variable in captured["instructions"]


def test_answer_correctness_is_logged_on_the_config_run(
    tracking_store, generator_calls, monkeypatch
):
    from benchmark.mlflow_scorers import answer_correctness_scorer

    run_id = _finished_config_run(monkeypatch)
    judge = _claims_judge(tp=1, fp=1, fn=0)

    metrics = score_run_traces(
        run_id,
        ("answer_correctness",),
        ENDPOINT,
        factories={"answer_correctness": lambda endpoint: answer_correctness_scorer(judge)},
    )

    assert metrics == {"mlflow_judge_answer_correctness_mean": 2 / 3}
    assert mlflow.get_run(run_id).data.metrics["mlflow_judge_answer_correctness_mean"] == 2 / 3
    # Ground truth from the traces' expectations reaches the judge.
    references = {call[2]["expected_response"] for call in judge.calls}
    assert references == {s["ground_truth"] for s in SAMPLES}
