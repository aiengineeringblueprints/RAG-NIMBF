"""Bootstrap confidence intervals and paired significance tests.

Post-processing only: every function here operates on lists of per-sample
scores that already exist in ``results_per_sample.csv`` and the per-config
JSON reports. No model calls, no new benchmark samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Callable, Sequence

StatFunction = Callable[[Sequence[float]], float]


def _clean(values: Sequence[float]) -> list[float]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not cleaned:
        raise ValueError("at least one finite value is required")
    return cleaned


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: StatFunction = lambda xs: sum(xs) / len(xs),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic(values)``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    data = _clean(values)
    rng = Random(seed)
    n = len(data)
    stats: list[float] = []
    for _ in range(n_resamples):
        stats.append(statistic([data[rng.randrange(n)] for _ in range(n)]))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = min(n_resamples - 1, max(0, int(math.floor(alpha * n_resamples))))
    high_index = min(
        n_resamples - 1, max(0, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    )
    return stats[low_index], stats[high_index]


def paired_bootstrap_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """CI for ``mean(a) - mean(b)`` resampling sample indices jointly.

    ``a[i]`` and ``b[i]`` must answer the same benchmark question. Joint
    resampling keeps the pairing, which is what makes the comparison
    sensitive: question difficulty cancels out within each resample.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length score lists")
    xa, xb = _clean(a), _clean(b)
    pairs = list(zip(xa, xb))
    rng = Random(seed)
    n = len(pairs)
    deltas: list[float] = []
    for _ in range(n_resamples):
        sum_a = sum_b = 0.0
        for _ in range(n):
            fa, fb = pairs[rng.randrange(n)]
            sum_a += fa
            sum_b += fb
        deltas.append((sum_a - sum_b) / n)
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = min(n_resamples - 1, max(0, int(math.floor(alpha * n_resamples))))
    high_index = min(
        n_resamples - 1, max(0, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    )
    point = sum(xa) / n - sum(xb) / n
    return point, deltas[low_index], deltas[high_index]


def paired_permutation_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> float:
    """Two-sided paired permutation test p-value for ``mean(a) - mean(b)``."""
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length score lists")
    xa, xb = _clean(a), _clean(b)
    diffs = [x - y for x, y in zip(xa, xb)]
    observed = sum(diffs) / len(diffs)
    if all(d == 0.0 for d in diffs):
        return 1.0
    rng = Random(seed)
    count_ge = 0
    for _ in range(n_resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(flipped) >= abs(observed) - 1e-12:
            count_ge += 1
    return count_ge / n_resamples


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Pooled-standard-deviation effect size for two independent samples."""
    xa, xb = _clean(a), _clean(b)
    n_a, n_b = len(xa), len(xb)
    mean_a = sum(xa) / n_a
    mean_b = sum(xb) / n_b
    var_a = sum((x - mean_a) ** 2 for x in xa) / max(1, n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in xb) / max(1, n_b - 1)
    pooled = math.sqrt((var_a * (n_a - 1) + var_b * (n_b - 1)) / max(1, n_a + n_b - 2))
    if pooled == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    effect_size: float

    @property
    def reliable(self) -> bool:
        """True when the 95% CI excludes zero and p < 0.05."""
        return (self.ci_low > 0.0 or self.ci_high < 0.0) and self.p_value < 0.05

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "metric": self.metric,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "reliable": self.reliable,
        }


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    metric: str,
    *,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> MetricComparison:
    """Full comparison: delta with bootstrap CI, permutation p, Cohen's d."""
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length score lists")
    xa, xb = _clean(a), _clean(b)
    delta, low, high = paired_bootstrap_delta_ci(
        xa, xb, n_resamples=n_resamples, seed=seed
    )
    p = paired_permutation_pvalue(xa, xb, n_resamples=n_resamples, seed=seed)
    return MetricComparison(
        metric=metric,
        mean_a=sum(xa) / len(xa),
        mean_b=sum(xb) / len(xb),
        delta=delta,
        ci_low=low,
        ci_high=high,
        p_value=p,
        effect_size=cohens_d(xa, xb),
    )


def _paired_metric_arrays(
    per_sample_a: Sequence[dict],
    per_sample_b: Sequence[dict],
) -> dict[str, tuple[list[float], list[float]]]:
    """Extract paired per-metric arrays from two per-sample score lists.

    Each per-sample row is a dict of metric name -> score (or None). Rows
    pair positionally because both configs answer the same dataset in the
    same order.
    """
    arrays: dict[str, tuple[list[float], list[float]]] = {}
    for row_a, row_b in zip(per_sample_a, per_sample_b):
        for metric in set(row_a) | set(row_b):
            va, vb = row_a.get(metric), row_b.get(metric)
            if va is None or vb is None:
                continue
            arr = arrays.setdefault(metric, ([], []))
            arr[0].append(float(va))
            arr[1].append(float(vb))
    return arrays


def assert_paired_alignment(
    questions_a: Sequence[str] | None,
    questions_b: Sequence[str] | None,
) -> None:
    """Raise if two per-sample sequences do not answer the same questions.

    Paired statistics are only valid when row *i* of both configs refers to
    the same benchmark sample. Comparing misaligned rows would produce
    confident nonsense, so callers should always pass the question texts.
    """
    if questions_a is None or questions_b is None:
        return
    if len(questions_a) != len(questions_b):
        raise ValueError(
            "paired comparison misaligned: "
            f"{len(questions_a)} questions vs {len(questions_b)}"
        )
    mismatched = [
        index
        for index, (qa, qb) in enumerate(zip(questions_a, questions_b))
        if str(qa).strip() != str(qb).strip()
    ]
    if mismatched:
        first = mismatched[0]
        raise ValueError(
            "paired comparison misaligned at sample "
            f"{first}: {questions_a[first]!r} != {questions_b[first]!r}"
        )


def compare_per_sample(
    per_sample_a: Sequence[dict],
    per_sample_b: Sequence[dict],
    *,
    n_resamples: int = 10_000,
    seed: int | None = 42,
    questions_a: Sequence[str] | None = None,
    questions_b: Sequence[str] | None = None,
) -> list[MetricComparison]:
    """Compare two configs on every metric with enough paired valid samples.

    ``questions_a`` / ``questions_b`` (optional but strongly recommended)
    are the per-sample question texts; they are checked for alignment before
    any statistics are computed.
    """
    assert_paired_alignment(questions_a, questions_b)
    comparisons: list[MetricComparison] = []
    arrays = _paired_metric_arrays(per_sample_a, per_sample_b)
    for metric in sorted(arrays):
        arr_a, arr_b = arrays[metric]
        if len(arr_a) < 3:
            continue
        comparisons.append(
            paired_comparison(arr_a, arr_b, metric, n_resamples=n_resamples, seed=seed)
        )
    return comparisons
