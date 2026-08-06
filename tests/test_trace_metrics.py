"""Tests for benchmark.trace_metrics — TRACe Utilization / Completeness."""

from __future__ import annotations

import json

import pytest

from benchmark.trace_metrics import (
    TraceMetricsResult,
    compute_trace_metrics,
    _parse_judge_response,
    _heuristic_utilization,
    _heuristic_completeness,
)


# ── Heuristic primitives ─────────────────────────────────────────────

class TestHeuristic:
    def test_full_utilization_when_answer_is_in_context(self):
        ctx = ["the capital of france is paris"]
        ans = "the capital of france is paris"
        assert _heuristic_utilization(ans, ctx) == pytest.approx(1.0)

    def test_zero_utilization_when_answer_disjoint(self):
        ctx = ["the cat sat on the mat"]
        ans = "berlin is germany capital"
        assert _heuristic_utilization(ans, ctx) == 0.0

    def test_zero_completeness_when_no_overlap(self):
        ctx = ["alpha beta gamma delta epsilon zeta eta theta"]
        ans = "completely different tokens here"
        assert _heuristic_completeness(ans, ctx) == 0.0

    def test_completeness_one_when_answer_copies_context(self):
        ctx = ["alpha beta gamma delta epsilon zeta eta theta"]
        ans = "alpha beta gamma delta epsilon zeta eta theta"
        assert _heuristic_completeness(ans, ctx) == pytest.approx(1.0)


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_refusal_answer_scores_zero(self):
        result = compute_trace_metrics(
            questions=["q"],
            contexts=[["the answer is 42"]],
            answers=["I cannot answer this from the provided context."],
            llm_judge=False,
        )
        assert result.metric_means["trace_utilization"] == 0.0
        assert result.metric_means["trace_completeness"] == 0.0
        assert result.per_sample_scores[0]["trace_mode"] == 0.0

    def test_empty_context_scores_zero(self):
        result = compute_trace_metrics(
            questions=["q"],
            contexts=[[]],
            answers=["some answer"],
            llm_judge=False,
        )
        assert result.metric_means["trace_utilization"] == 0.0
        assert result.metric_means["trace_completeness"] == 0.0

    def test_empty_questions_returns_error(self):
        result = compute_trace_metrics(
            questions=[], contexts=[], answers=[], llm_judge=False
        )
        assert isinstance(result, TraceMetricsResult)
        assert result.error == "No questions provided."
        assert result.metric_means == {}


# ── Full utilization via heuristic ───────────────────────────────────

class TestFullUtilizationHeuristic:
    def test_full_utilization_partial_completeness(self):
        # Answer copies a strict subset of the context.
        ctx = ["the quick brown fox jumps over the lazy dog at noon"]
        ans = "the quick brown fox"
        result = compute_trace_metrics(
            questions=["q"], contexts=[ctx], answers=[ans], llm_judge=False
        )
        util = result.metric_means["trace_utilization"]
        comp = result.metric_means["trace_completeness"]
        assert util == pytest.approx(1.0)  # every answer token grounded
        assert 0.0 < comp < 1.0  # but doesn't cover whole context


# ── LLM-judge path (mocked) ──────────────────────────────────────────

class TestLLMJudge:
    def test_judge_responses_are_parsed(self):
        captured_prompts: list[str] = []

        def fake_caller(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({"utilization": 0.8, "completeness": 0.6})

        result = compute_trace_metrics(
            questions=["q"],
            contexts=[["some context"]],
            answers=["some answer"],
            llm_judge=True,
            judge_caller=fake_caller,
        )
        assert len(captured_prompts) == 1
        assert "some context" in captured_prompts[0]
        assert "some answer" in captured_prompts[0]
        assert result.metric_means["trace_utilization"] == pytest.approx(0.8)
        assert result.metric_means["trace_completeness"] == pytest.approx(0.6)
        assert result.per_sample_scores[0]["trace_mode"] == 1.0  # llm

    def test_judge_failure_falls_back_to_heuristic(self):
        def broken_caller(prompt: str) -> str:
            return "the model rambled with no json at all"

        result = compute_trace_metrics(
            questions=["q"],
            contexts=[["the quick brown fox jumps"]],
            answers=["the quick brown fox"],
            llm_judge=True,
            judge_caller=broken_caller,
        )
        # Heuristic fallback gives full util for grounded answer.
        assert result.metric_means["trace_utilization"] == pytest.approx(1.0)
        assert result.per_sample_scores[0]["trace_mode"] == 0.0

    def test_all_judge_failures_surface_soft_error(self):
        def always_broken(prompt: str) -> str:
            return "garbage"

        result = compute_trace_metrics(
            questions=["q1", "q2"],
            contexts=[["ctx one"], ["ctx two"]],
            answers=["ans one", "ans two"],
            llm_judge=True,
            judge_caller=always_broken,
        )
        assert result.error is not None
        assert "heuristic" in result.error.lower()

    def test_judge_response_with_surrounding_prose(self):
        def chatty_caller(prompt: str) -> str:
            return (
                "Here is my evaluation: "
                '{"utilization": 0.9, "completeness": 0.3} '
                "as required."
            )

        result = compute_trace_metrics(
            questions=["q"],
            contexts=[["ctx"]],
            answers=["ans"],
            llm_judge=True,
            judge_caller=chatty_caller,
        )
        assert result.metric_means["trace_utilization"] == pytest.approx(0.9)
        assert result.metric_means["trace_completeness"] == pytest.approx(0.3)

    def test_out_of_range_judge_value_rejected(self):
        # 1.5 should be rejected → heuristic fallback for that sample.
        def liar_caller(prompt: str) -> str:
            return '{"utilization": 1.5, "completeness": 0.5}'

        result = compute_trace_metrics(
            questions=["q"],
            contexts=[["the quick brown fox"]],
            answers=["the quick brown fox"],
            llm_judge=True,
            judge_caller=liar_caller,
        )
        assert result.per_sample_scores[0]["trace_mode"] == 0.0


# ── Judge response parser unit tests ─────────────────────────────────

class TestParseJudgeResponse:
    def test_clean_json(self):
        u, c = _parse_judge_response('{"utilization": 0.7, "completeness": 0.4}')
        assert u == 0.7
        assert c == 0.4

    def test_embedded_json(self):
        u, c = _parse_judge_response(
            'Sure! {"utilization": 0.2, "completeness": 0.9} hope this helps'
        )
        assert u == 0.2
        assert c == 0.9

    def test_empty_returns_none(self):
        assert _parse_judge_response("") == (None, None)

    def test_missing_keys_return_none(self):
        u, c = _parse_judge_response('{"utilization": 0.5}')
        assert u == 0.5
        assert c is None

    def test_string_numbers_parsed(self):
        u, c = _parse_judge_response('{"utilization": "0.8", "completeness": "0.1"}')
        assert u == 0.8
        assert c == 0.1
