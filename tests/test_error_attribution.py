"""Tests for benchmark.error_attribution — RAGChecker-style metrics."""

from __future__ import annotations

import json

import pytest

from benchmark.error_attribution import (
    ErrorAttributionResult,
    compute_error_attribution,
    _aggregate_judge,
    _parse_judge_response,
    _heuristic_faithfulness,
    _heuristic_context_recall,
)


# ── Heuristic primitives ─────────────────────────────────────────────

class TestHeuristic:
    def test_full_faithfulness_when_answer_in_context(self):
        ctx = ["the capital of france is paris"]
        ans = "the capital of france is paris"
        assert _heuristic_faithfulness(ans, ctx) == pytest.approx(1.0)

    def test_zero_faithfulness_when_disjoint(self):
        ctx = ["the cat sat on the mat"]
        ans = "berlin capital germany river"
        assert _heuristic_faithfulness(ans, ctx) == 0.0

    def test_context_recall_full_when_gt_in_context(self):
        ctx = ["the capital of france is paris"]
        assert _heuristic_context_recall("the capital of france is paris", ctx) == pytest.approx(1.0)

    def test_context_recall_zero_when_disjoint(self):
        ctx = ["cats are small"]
        assert _heuristic_context_recall("the capital is paris", ctx) == 0.0


# ── Aggregation from judge output ────────────────────────────────────

class TestAggregate:
    def test_all_categories(self):
        judge = {
            "relevant_context_indices": [0],
            "answer_claims": [
                {"text": "a", "category": "correct"},
                {"text": "b", "category": "correct"},
                {"text": "c", "category": "hallucinated"},
                {"text": "d", "category": "relevant_noise"},
                {"text": "e", "category": "irrelevant_noise"},
                {"text": "f", "category": "self_knowledge"},
            ],
            "ground_truth_claims": [
                {"text": "a", "covered_in_response": True, "covered_in_context": True},
                {"text": "z", "covered_in_response": False, "covered_in_context": True},
            ],
        }
        m = _aggregate_judge(judge, ["ctx0", "ctx1"])
        assert m["rc_claim_precision"] == pytest.approx(2 / 6)
        assert m["rc_claim_recall"] == pytest.approx(0.5)
        assert m["rc_faithfulness"] == pytest.approx(4 / 6)  # correct + 2 noise
        assert m["rc_hallucination"] == pytest.approx(1 / 6)
        assert m["rc_noise_sensitivity_relevant"] == pytest.approx(1 / 6)
        assert m["rc_noise_sensitivity_irrelevant"] == pytest.approx(1 / 6)
        assert m["rc_self_knowledge"] == pytest.approx(1 / 6)
        assert m["rc_context_precision"] == pytest.approx(0.5)  # 1 relevant / 2
        assert m["rc_context_recall"] == pytest.approx(1.0)

    def test_claim_f1_harmonic(self):
        judge = {
            "relevant_context_indices": [],
            "answer_claims": [{"text": "a", "category": "correct"}],
            "ground_truth_claims": [
                {"text": "a", "covered_in_response": True, "covered_in_context": True},
                {"text": "b", "covered_in_response": False, "covered_in_context": True},
            ],
        }
        m = _aggregate_judge(judge, ["ctx"])
        # precision 1.0, recall 0.5 -> f1 = 2*(1*0.5)/(1.5) = 0.666...
        assert m["rc_claim_precision"] == pytest.approx(1.0)
        assert m["rc_claim_recall"] == pytest.approx(0.5)
        assert m["rc_claim_f1"] == pytest.approx(2 / 3)

    def test_empty_answer_claims(self):
        judge = {
            "relevant_context_indices": [],
            "answer_claims": [],
            "ground_truth_claims": [],
        }
        m = _aggregate_judge(judge, ["ctx"])
        assert m["rc_claim_precision"] is None
        assert m["rc_faithfulness"] is None


# ── Parse judge response ─────────────────────────────────────────────

class TestParseJudge:
    def test_clean_json(self):
        raw = json.dumps({"relevant_context_indices": [0]})
        assert _parse_judge_response(raw) == {"relevant_context_indices": [0]}

    def test_embedded_json(self):
        obj = _parse_judge_response('Sure! {"relevant_context_indices": [1]} ok')
        assert obj == {"relevant_context_indices": [1]}

    def test_garbage_returns_none(self):
        assert _parse_judge_response("no json here") is None
        assert _parse_judge_response("") is None


# ── LLM-judge path (mocked) ──────────────────────────────────────────

class TestComputeLLM:
    def _good_judge(self):
        return {
            "relevant_context_indices": [0],
            "answer_claims": [
                {"text": "a", "category": "correct"},
                {"text": "b", "category": "hallucinated"},
            ],
            "ground_truth_claims": [
                {"text": "a", "covered_in_response": True, "covered_in_context": True},
            ],
        }

    def test_llm_judge_produces_metrics(self):
        captured: list[str] = []

        def fake_caller(prompt: str) -> str:
            captured.append(prompt)
            return json.dumps(self._good_judge())

        result = compute_error_attribution(
            questions=["q"],
            ground_truths=["gt"],
            contexts=[["some context"]],
            answers=["some answer"],
            llm_judge=True,
            judge_caller=fake_caller,
        )
        assert len(captured) == 1
        assert "q" in captured[0]
        assert result.per_sample_scores[0]["rc_mode"] == 1.0
        assert result.metric_means["rc_claim_precision"] == pytest.approx(0.5)
        assert result.metric_means["rc_hallucination"] == pytest.approx(0.5)
        assert result.metric_means["rc_context_precision"] == pytest.approx(1.0)

    def test_judge_failure_falls_back_to_heuristic(self):
        def broken_caller(prompt: str) -> str:
            return "no json at all"

        result = compute_error_attribution(
            questions=["q"],
            ground_truths=["the capital of france is paris"],
            contexts=[["the capital of france is paris"]],
            answers=["the capital of france is paris"],
            llm_judge=True,
            judge_caller=broken_caller,
        )
        assert result.per_sample_scores[0]["rc_mode"] == 0.0
        assert result.metric_means["rc_faithfulness"] == pytest.approx(1.0)
        # Heuristic has no claim precision signal.
        assert "rc_claim_precision" not in result.metric_means

    def test_all_judge_failures_surface_soft_error(self):
        def always_broken(prompt: str) -> str:
            return "garbage"

        result = compute_error_attribution(
            questions=["q1", "q2"],
            ground_truths=["g1", "g2"],
            contexts=[["c1"], ["c2"]],
            answers=["a1", "a2"],
            llm_judge=True,
            judge_caller=always_broken,
        )
        assert result.error is not None
        assert "heuristic" in result.error.lower()


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_questions_returns_error(self):
        result = compute_error_attribution(
            questions=[], ground_truths=[], contexts=[], answers=[]
        )
        assert isinstance(result, ErrorAttributionResult)
        assert result.error == "No questions provided."
        assert result.metric_means == {}

    def test_refusal_scores_zero(self):
        result = compute_error_attribution(
            questions=["q"],
            ground_truths=["gt"],
            contexts=[["ctx"]],
            answers=["I cannot answer this from the provided context."],
            llm_judge=False,
        )
        assert result.metric_means["rc_hallucination"] == 0.0
        assert result.metric_means["rc_self_knowledge"] == 0.0

    def test_empty_context(self):
        result = compute_error_attribution(
            questions=["q"],
            ground_truths=["gt"],
            contexts=[[]],
            answers=["some answer"],
            llm_judge=False,
        )
        assert result.per_sample_scores[0]["rc_mode"] == 0.0
        assert result.metric_means == {}
