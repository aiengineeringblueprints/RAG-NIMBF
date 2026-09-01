"""Tests for benchmark.statistics — CIs + Wilcoxon signed-rank test."""

from __future__ import annotations

import pytest

from benchmark.statistics import (
    bootstrap_ci,
    paired_delta_ci,
    wilcoxon_signed_rank,
    compare_paired,
)


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        lo, hi = bootstrap_ci(samples, n_boot=500, seed=0)
        assert lo <= sum(samples) / len(samples) <= hi

    def test_delta_ci(self):
        a = [1.0] * 50
        b = [2.0] * 50
        lo, hi = paired_delta_ci(a, b, n_boot=500, seed=0)
        assert lo <= 1.0 <= hi


class TestWilcoxon:
    def test_significant_difference(self):
        a = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.5, 4.5, 2.5, 3.0]
        b = [x + 3.0 for x in a]
        p = wilcoxon_signed_rank(a, b)
        assert p is not None
        assert p < 0.05

    def test_no_difference(self):
        a = [3.0, 4.0, 2.0, 5.0]
        b = [3.0, 4.0, 2.0, 5.0]
        assert wilcoxon_signed_rank(a, b) is None  # all diffs zero

    def test_exact_matches_approx_small_n(self):
        # n=10 is exact; just ensure it returns a valid p in [0,1].
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        p = wilcoxon_signed_rank(a, b)
        assert p is not None
        assert 0.0 <= p <= 1.0


class TestComparePaired:
    def test_full_comparison(self):
        a = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.5, 4.5, 2.5, 3.0] * 5
        b = [x + 2.0 for x in a]
        result = compare_paired(a, b, n_boot=200, seed=0)
        assert result.n == len(a)
        assert result.mean_delta == pytest.approx(2.0)
        assert result.wilcoxon_p is not None
        assert result.wilcoxon_p < 0.05
        assert result.statistically_significant is True

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compare_paired([1.0], [1.0, 2.0])
