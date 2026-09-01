"""Judge reliability / meta-evaluation for LLM-judged metrics.

LLM-as-judge scores are noisy.  Before trusting them, a benchmark should
quantify how reliable the judge is.  This module provides two cheap,
self-contained checks that mirror the reliability concern RAGChecker raises
about untrusted automated metrics:

    * Self-consistency  — run the same judge multiple times on the same
      samples (with sampling temperature) and measure agreement.  Low
      agreement ⇒ the judge (or the underlying metric) is unstable.
    * Cross-judge agreement — compare two different judge models on the
      same samples and report rank/linear correlation and mean absolute
      deviation.

Both are computed over pre-collected per-sample score arrays, so they need
no model calls here; callers produce the repeated / alternate-judge scores
with their own wiring (see ``judge_caller`` seams in the other modules).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeReliabilityResult:
    """Reliability statistics for a judge or pair of judges."""

    mean_score: float | None
    std_score: float | None
    self_consistency_alpha: float | None  # agreement among repeated runs
    pairwise_pearson: float | None  # correlation judge A vs judge B
    pairwise_spearman: float | None
    mean_abs_deviation: float | None  # |A - B| averaged
    error: str | None = None


# ── Agreement statistics ─────────────────────────────────────────────

def _mean(x: list[float]) -> float | None:
    return sum(x) / len(x) if x else None


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    ma = _mean(a)
    mb = _mean(b)
    if ma is None or mb is None:
        return None
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    denom = (da * db) ** 0.5
    if denom == 0:
        return None
    return num / denom


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation via rank assignment."""
    if len(a) != len(b) or len(a) < 2:
        return None

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    return _pearson(_rank(a), _rank(b))


def _pairwise_agreement(runs: list[list[float]]) -> float | None:
    """Mean pairwise agreement (1 - normalised L1) across repeated runs.

    ``runs`` is a list of score-vectors, one per judge run, each aligned
    across samples.  Returns the average agreement in [0, 1] where 1 means
    the judge is perfectly self-consistent.
    """
    if len(runs) < 2:
        return None
    n_samples = len(runs[0])
    if any(len(r) != n_samples for r in runs):
        return None
    if n_samples == 0:
        return None

    agreement_sum = 0.0
    pairs = 0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            diffs = [
                abs(x - y) for x, y in zip(runs[i], runs[j]) if x is not None and y is not None
            ]
            if not diffs:
                continue
            pairs += 1
            agreement_sum += 1.0 - (sum(diffs) / len(diffs))
    if pairs == 0:
        return None
    return agreement_sum / pairs


# ── Public entry points ──────────────────────────────────────────────

def judge_self_consistency(scores_per_run: list[list[float]]) -> JudgeReliabilityResult:
    """Assess judge self-consistency from repeated judge runs.

    ``scores_per_run`` is a list of per-sample score vectors, one per judge
    run over the same samples (e.g. with temperature > 0).
    """
    if not scores_per_run or len(scores_per_run) < 2:
        return JudgeReliabilityResult(
            mean_score=None,
            std_score=None,
            self_consistency_alpha=None,
            pairwise_pearson=None,
            pairwise_spearman=None,
            mean_abs_deviation=None,
            error="Need at least two judge runs for self-consistency.",
        )

    # Mean and std across all runs, per-sample then averaged.
    n_samples = len(scores_per_run[0])
    if any(len(r) != n_samples for r in scores_per_run):
        return JudgeReliabilityResult(
            mean_score=None,
            std_score=None,
            self_consistency_alpha=None,
            pairwise_pearson=None,
            pairwise_spearman=None,
            mean_abs_deviation=None,
            error="All judge runs must have the same number of samples.",
        )

    per_sample_means: list[float] = []
    per_sample_stds: list[float] = []
    for s in range(n_samples):
        vals = [r[s] for r in scores_per_run if r[s] is not None]
        if not vals:
            continue
        m = _mean(vals)
        per_sample_means.append(m)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        per_sample_stds.append(var ** 0.5)

    return JudgeReliabilityResult(
        mean_score=_mean(per_sample_means),
        std_score=_mean(per_sample_stds),
        self_consistency_alpha=_pairwise_agreement(scores_per_run),
        pairwise_pearson=None,
        pairwise_spearman=None,
        mean_abs_deviation=None,
    )


def judge_cross_agreement(
    scores_judge_a: list[float],
    scores_judge_b: list[float],
) -> JudgeReliabilityResult:
    """Compare two judges on the same samples.

    Reports Pearson + Spearman rank correlation and mean absolute deviation
    between the two judges' per-sample scores.
    """
    if len(scores_judge_a) != len(scores_judge_b):
        return JudgeReliabilityResult(
            mean_score=None,
            std_score=None,
            self_consistency_alpha=None,
            pairwise_pearson=None,
            pairwise_spearman=None,
            mean_abs_deviation=None,
            error="Judges must score the same samples.",
        )

    pairs = [
        (a, b)
        for a, b in zip(scores_judge_a, scores_judge_b)
        if a is not None and b is not None
    ]
    if len(pairs) < 2:
        return JudgeReliabilityResult(
            mean_score=None,
            std_score=None,
            self_consistency_alpha=None,
            pairwise_pearson=None,
            pairwise_spearman=None,
            mean_abs_deviation=None,
            error="Not enough valid score pairs.",
        )

    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    mad = sum(abs(x - y) for x, y in pairs) / len(pairs)

    return JudgeReliabilityResult(
        mean_score=None,
        std_score=None,
        self_consistency_alpha=None,
        pairwise_pearson=_pearson(a, b),
        pairwise_spearman=_spearman(a, b),
        mean_abs_deviation=mad,
    )
