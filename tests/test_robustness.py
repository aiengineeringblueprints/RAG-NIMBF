"""Tests for benchmark.robustness — negative rejection + noise robustness."""

from __future__ import annotations

import pytest

from benchmark.robustness import (
    measure_negative_rejection,
    add_distractor_contexts,
    measure_noise_robustness,
    measure_noise_robustness_fn,
)


class TestNegativeRejection:
    def test_rejects_when_unanswerable(self):
        result = measure_negative_rejection(
            questions=["q1", "q2", "q3"],
            answers=[
                "I cannot answer this from the context.",
                "The answer is 42.",
                "Also a non-refusal.",
            ],
            unanswerable=[True, True, False],
        )
        assert result.unanswerable_count == 2
        assert result.rejection_rate == pytest.approx(0.5)
        assert result.hallucination_rate == pytest.approx(0.5)

    def test_full_rejection(self):
        result = measure_negative_rejection(
            questions=["q1", "q2"],
            answers=["I don't have enough information.", "Cannot be answered."],
            unanswerable=[True, True],
        )
        assert result.rejection_rate == pytest.approx(1.0)
        assert result.hallucination_rate == pytest.approx(0.0)

    def test_length_mismatch(self):
        result = measure_negative_rejection(
            questions=["q1"], answers=["a"], unanswerable=[True, False]
        )
        assert result.error is not None

    def test_no_unanswerable(self):
        result = measure_negative_rejection(
            questions=["q1"], answers=["some answer"], unanswerable=[False]
        )
        assert result.unanswerable_count == 0
        assert result.rejection_rate is None
        assert result.hallucination_rate is None


class TestDistractors:
    def test_appends_distractors(self):
        contexts = [["a", "b"], ["c"]]
        out = add_distractor_contexts(contexts, ["X", "Y", "Z"], n_distractors=2, seed=1)
        assert out[0][:2] == ["a", "b"]
        assert len(out[0]) == 4
        assert len(out[1]) == 3  # 1 original + 2 distractors

    def test_preserves_original_order(self):
        contexts = [["a", "b", "c"]]
        out = add_distractor_contexts(contexts, ["X"], n_distractors=1)
        assert out[0][:3] == ["a", "b", "c"]
        assert out[0][3] == "X"


class TestNoiseRobustness:
    def test_degradation_delta(self):
        out = measure_noise_robustness([1.0, 1.0, 1.0], [0.5, 0.6, 0.7])
        assert out["robustness_delta"] == pytest.approx(0.4)
        assert out["robustness_ratio"] == pytest.approx(0.6)

    def test_length_mismatch(self):
        out = measure_noise_robustness([1.0, 1.0], [1.0])
        assert out["robustness_delta"] is None

    def test_empty(self):
        out = measure_noise_robustness([], [])
        assert out["robustness_delta"] is None


class TestNoiseRobustnessFn:
    def test_wires_score_function(self):
        def score_fn(q, ctx, a):
            # score = 1.0 - fraction of contexts that are distractors
            return [1.0 / len(c) for c in ctx]

        out = measure_noise_robustness_fn(
            questions=["q"],
            contexts=[["real"]],
            answers=["a"],
            distractors=["noise"],
            score_fn=score_fn,
            n_distractors=1,
        )
        # clean: 1/1 = 1.0 ; noisy: 1/2 = 0.5 ; delta = 0.5
        assert out["robustness_delta"] == pytest.approx(0.5)
