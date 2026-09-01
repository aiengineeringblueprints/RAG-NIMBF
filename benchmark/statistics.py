"""Statistical significance and uncertainty for benchmark comparisons.

Benchmark metric means are point estimates; two configs can look different
by chance.  This module provides small, dependency-light inferential
statistics over per-sample score arrays so comparisons between two runs
(e.g. different chunk sizes, embeddings, or LLMs) can be reported with
confidence intervals and significance tests.

Functions:
    * ``bootstrap_ci``        – percentile bootstrap confidence interval for
                                a mean.
    * ``paired_delta_ci``     – bootstrap CI for the mean *difference*
                                between two paired configs.
    * ``wilcoxon_signed_rank``– non-parametric test on paired differences
                                (no normality assumption).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SignificanceResult:
    """Result of a paired comparison between two configs."""

    n: int
    mean_a: float
    mean_b: float
    mean_delta: float  # B - A
    ci_low: float
    ci_high: float
    wilcoxon_p: float | None
    statistically_significant: bool | None  # at alpha=0.05


def _mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def bootstrap_ci(
    samples: list[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of *samples*."""
    if not samples:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(samples)
    boot_means: list[float] = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        boot_means.append(_mean(resample))
    boot_means.sort()
    lo = int((alpha / 2.0) * n_boot)
    hi = int((1.0 - alpha / 2.0) * n_boot) - 1
    hi = max(lo, hi)
    return (boot_means[lo], boot_means[hi])


def paired_delta_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap CI for the mean paired difference (B - A)."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired arrays must be equal length.")
    if not scores_a:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(scores_a)
    deltas = [b - a for a, b in zip(scores_a, scores_b)]
    boot_means: list[float] = []
    for _ in range(n_boot):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(_mean(resample))
    boot_means.sort()
    lo = int((alpha / 2.0) * n_boot)
    hi = int((1.0 - alpha / 2.0) * n_boot) - 1
    hi = max(lo, hi)
    return (boot_means[lo], boot_means[hi])


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank test p-value on paired differences.

    Exact for n <= 30, normal approximation otherwise.  Returns ``None`` if
    there is insufficient data or all differences are zero.
    """
    if len(a) != len(b):
        return None
    diffs = [y - x for x, y in zip(a, b)]
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return None

    abs_diffs = sorted((abs(d), i) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[abs_diffs[k][1]] = avg
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)

    if n <= 30:
        # Exact two-sided p-value via enumeration.
        return _exact_wilcoxon_p(w_plus, n)

    # Normal approximation with continuity correction.
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return None
    z = (w_plus - mu + 0.5) / sigma
    p = 2.0 * _normal_sf(abs(z))
    return min(1.0, p)


def _exact_wilcoxon_p(w_plus: float, n: int) -> float | None:
    """Exact two-sided p-value by enumerating subset sums of ranks 1..n."""
    target = int(round(w_plus))
    total = n * (n + 1) // 2
    if target > total // 2:
        target = total - target
    if target < 0:
        return None
    # Count subsets of {1..n} summing to each value <= target.
    counts = [0] * (target + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for s in range(target, rank - 1, -1):
            counts[s] += counts[s - rank]
    # p = 2 * P(W+ <= observed_min_side)
    p_lower = sum(counts) / (2 ** n)
    return min(1.0, 2.0 * p_lower)


def _normal_sf(z: float) -> float:
    """Standard normal survival function (one-sided)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def compare_paired(
    scores_a: list[float],
    scores_b: list[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = None,
) -> SignificanceResult:
    """Full paired comparison between two configs' per-sample scores."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired arrays must be equal length.")
    if not scores_a:
        return SignificanceResult(
            n=0,
            mean_a=0.0,
            mean_b=0.0,
            mean_delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            wilcoxon_p=None,
            statistically_significant=None,
        )

    mean_a = _mean(scores_a)
    mean_b = _mean(scores_b)
    mean_delta = mean_b - mean_a
    ci_low, ci_high = paired_delta_ci(
        scores_a, scores_b, alpha=alpha, n_boot=n_boot, seed=seed
    )
    p = wilcoxon_signed_rank(scores_a, scores_b)

    return SignificanceResult(
        n=len(scores_a),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        wilcoxon_p=p,
        statistically_significant=(p is not None and p < alpha),
    )
