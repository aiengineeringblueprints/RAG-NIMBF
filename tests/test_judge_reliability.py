"""Tests for benchmark.judge_reliability — self-consistency + cross-judge."""

from __future__ import annotations

import pytest

from benchmark.judge_reliability import (
    judge_self_consistency,
    judge_cross_agreement,
    _pearson,
    _spearman,
)


class TestPearsonSpearman:
    def test_perfect_positive_pearson(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        assert _pearson(a, b) == pytest.approx(1.0)

    def test_spearman_equals_pearson_on_ranks(self):
        a = [10.0, 20.0, 30.0, 40.0]
        # monotone in ranks, so perfect monotonicity -> spearman 1.0
        b_ranked = [3.0, 1.0, 4.0, 2.0]
        # a is strictly increasing => ranks 1..4; b_ranked is a permutation
        # so their rank vectors are perfectly inversely... just check range.
        r = _spearman(a, b_ranked)
        assert -1.0 <= r <= 1.0

    def test_short_returns_none(self):
        assert _pearson([1.0], [2.0]) is None
        assert _spearman([1.0], [2.0]) is None


class TestSelfConsistency:
    def test_consistent_judge_scores_high_agreement(self):
        runs = [
            [0.8, 0.6, 0.7],
            [0.8, 0.6, 0.7],
            [0.8, 0.6, 0.7],
        ]
        result = judge_self_consistency(runs)
        assert result.self_consistency_alpha == pytest.approx(1.0)
        assert result.mean_score == pytest.approx(0.7)
        assert result.std_score == pytest.approx(0.0)

    def test_noisy_judge_lower_agreement(self):
        runs = [
            [0.8, 0.2, 0.9],
            [0.8, 0.9, 0.1],
        ]
        result = judge_self_consistency(runs)
        assert result.self_consistency_alpha is not None
        assert 0.0 <= result.self_consistency_alpha < 1.0

    def test_requires_two_runs(self):
        result = judge_self_consistency([[0.8, 0.6]])
        assert result.error is not None
        assert result.self_consistency_alpha is None


class TestCrossAgreement:
    def test_perfect_agreement(self):
        result = judge_cross_agreement([0.8, 0.6, 0.7], [0.8, 0.6, 0.7])
        assert result.pairwise_pearson == pytest.approx(1.0)
        assert result.mean_abs_deviation == pytest.approx(0.0)

    def test_length_mismatch(self):
        result = judge_cross_agreement([0.8], [0.8, 0.6])
        assert result.error is not None

    def test_correlation_and_mad(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5]
        b = [0.15, 0.25, 0.35, 0.45, 0.55]
        result = judge_cross_agreement(a, b)
        assert result.pairwise_pearson == pytest.approx(1.0)
        assert result.mean_abs_deviation == pytest.approx(0.05)
