from __future__ import annotations

import pytest

from benchmark.statistics import (
    bootstrap_ci,
    cohens_d,
    compare_per_sample,
    paired_bootstrap_delta_ci,
    paired_comparison,
    paired_permutation_pvalue,
)


def test_bootstrap_ci_covers_true_mean_and_shrinks_with_n():
    scores = [0.5 + (i % 7) * 0.01 for i in range(100)]  # mean 0.53
    low, high = bootstrap_ci(scores, n_resamples=2000, seed=1)
    assert low < 0.53 < high
    assert 0.0 <= low < high <= 1.0
    big = [0.5 + (i % 7) * 0.01 for i in range(2000)]
    low_big, high_big = bootstrap_ci(big, n_resamples=2000, seed=1)
    assert (high_big - low_big) < (high - low)


def test_bootstrap_ci_is_deterministic_with_seed():
    scores = [0.1, 0.5, 0.9, 0.4, 0.7]
    assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)


def test_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([float("nan")])
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], confidence=1.5)


def test_paired_bootstrap_detects_shift_and_null():
    rng = __import__("random").Random(3)
    base = [rng.random() for _ in range(100)]
    a = [x + 0.1 for x in base]  # clear shift
    delta, low, high = paired_bootstrap_delta_ci(a, base, n_resamples=2000, seed=5)
    assert abs(delta - 0.1) < 0.01
    assert low > 0.0  # shift is reliable
    _delta2, low2, high2 = paired_bootstrap_delta_ci(base, base, n_resamples=2000, seed=5)
    assert low2 <= 0.0 <= high2  # no difference -> CI contains 0


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_delta_ci([0.1, 0.2], [0.1])


def test_permutation_pvalue_small_for_shift_and_large_for_null():
    rng = __import__("random").Random(11)
    base = [rng.random() for _ in range(60)]
    a = [x + 0.2 for x in base]
    assert paired_permutation_pvalue(a, base, n_resamples=2000, seed=2) <= 0.01
    assert paired_permutation_pvalue(base, base, n_resamples=2000, seed=2) >= 0.9


def test_cohens_d():
    a = [0.5, 1.5] * 10
    b = [1.5, 2.5] * 10
    assert cohens_d(a, b) == pytest.approx(-1.9493588689617931)
    assert cohens_d([1.0, 2.0], [1.0, 2.0]) == 0.0  # zero pooled variance guard


def test_paired_comparison_flags_reliable_and_noise():
    rng = __import__("random").Random(9)
    base = [rng.random() for _ in range(80)]
    win = [x + 0.15 for x in base]
    noise = [x + rng.choice([-0.01, 0.01]) for x in base]

    reliable = paired_comparison(win, base, "faithfulness", seed=4)
    assert reliable.reliable is True
    assert reliable.delta > 0
    assert reliable.ci_low > 0

    not_reliable = paired_comparison(noise, base, "faithfulness", seed=4)
    assert not_reliable.reliable is False

    payload = reliable.as_dict()
    assert payload["metric"] == "faithfulness"
    assert payload["reliable"] is True


def test_compare_per_sample_skips_sparse_metrics_and_pairs_rows():
    per_sample_a = [
        {"faithfulness": 0.8, "ndcg": 0.5},
        {"faithfulness": 0.65, "ndcg": 0.7},
        {"faithfulness": 0.9, "ndcg": None},
        {"faithfulness": 0.75, "ndcg": 0.6},
        {"faithfulness": 0.7, "ndcg": 0.55},
        {"faithfulness": 0.85, "ndcg": 0.65},
    ]
    per_sample_b = [
        {"faithfulness": 0.4, "ndcg": 0.5},
        {"faithfulness": 0.3, "ndcg": 0.7},
        {"faithfulness": 0.5, "ndcg": 0.9},
        {"faithfulness": 0.35, "ndcg": 0.6},
        {"faithfulness": 0.45, "ndcg": 0.5},
        {"faithfulness": 0.55, "ndcg": 0.6},
    ]
    comparisons = compare_per_sample(per_sample_a, per_sample_b, n_resamples=500)
    by_metric = {c.metric: c for c in comparisons}
    # ndcg has only 3 valid pairs (row 3 has None in a) -> still >= 3, kept
    assert set(by_metric) == {"faithfulness", "ndcg"}
    assert by_metric["faithfulness"].reliable is True

    # fewer than 3 valid pairs -> metric dropped
    tiny = compare_per_sample(
        [{"m": 0.1}, {"m": 0.2}], [{"m": 0.3}, {"m": 0.4}], n_resamples=100
    )
    assert tiny == []


def test_assert_paired_alignment_detects_mismatch():
    from benchmark.statistics import assert_paired_alignment

    assert_paired_alignment(["q1", "q2"], ["q1", "q2"])  # ok
    with pytest.raises(ValueError, match="misaligned"):
        assert_paired_alignment(["q1", "q2"], ["q1", "other"])
    with pytest.raises(ValueError, match="misaligned"):
        assert_paired_alignment(["q1", "q2"], ["q1"])


def test_compare_per_sample_rejects_misaligned_questions():
    rows = [{"faithfulness": 0.8}, {"faithfulness": 0.6}, {"faithfulness": 0.9}]
    with pytest.raises(ValueError, match="misaligned at sample 1"):
        compare_per_sample(
            rows,
            rows,
            n_resamples=100,
            questions_a=["q1", "q2", "q3"],
            questions_b=["q1", "X", "q3"],
        )
