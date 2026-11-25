"""Paired comparison of two runs and the accept / reject / inconclusive verdict."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.regression.criteria import RegressionCriteria, as_criteria
from benchmark.regression.metrics import LOWER_IS_BETTER, sample_metrics
from benchmark.regression.question_set import SampleKey, question_categories, sample_keys


@dataclass(frozen=True)
class Paired:
    """One metric's values on the samples both runs scored.

    ``clusters`` names the question of each sample. TechQA repeats each
    question with a second reference answer; both copies share documents and
    retrieval, so they are not independent evidence. Statistics resample and
    count whole clusters, never single rows.
    """

    baseline: np.ndarray
    candidate: np.ndarray
    clusters: tuple[str, ...]

    @classmethod
    def from_triples(cls, triples: Sequence[tuple[float, float, str]]) -> "Paired":
        return cls(
            baseline=np.array([b for b, _, _ in triples], dtype=float),
            candidate=np.array([c for _, c, _ in triples], dtype=float),
            clusters=tuple(q for _, _, q in triples),
        )

    @property
    def diffs(self) -> np.ndarray:
        return self.candidate - self.baseline

    @cached_property
    def groups(self) -> list[np.ndarray]:
        index: dict[str, list[int]] = {}
        for i, cluster in enumerate(self.clusters):
            index.setdefault(cluster, []).append(i)
        return [np.array(ix) for ix in index.values()]

    @property
    def is_binary(self) -> bool:
        return set(np.unique(np.concatenate([self.baseline, self.candidate]))) <= {0.0, 1.0}

    def bootstrap_ci(self, n_boot: int, seed: int) -> tuple[float, float]:
        """Paired cluster bootstrap 95% CI of the mean delta."""
        diffs = self.diffs
        sums = np.array([diffs[ix].sum() for ix in self.groups])
        counts = np.array([len(ix) for ix in self.groups])
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.groups), size=(n_boot, len(self.groups)))
        boot_means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        low, high = np.percentile(boot_means, [2.5, 97.5])
        return float(low), float(high)

    def sign_test(self) -> dict[str, Any]:
        """Exact two-sided sign test over questions (McNemar for 0/1 rows).

        Each question counts once: it favours the candidate when its mean
        outcome is higher there, the baseline when lower; ties drop out. With
        one row per question this is the exact McNemar test.
        """
        only_candidate = only_baseline = 0
        for ix in self.groups:
            delta = self.candidate[ix].mean() - self.baseline[ix].mean()
            only_candidate += int(delta > 0)
            only_baseline += int(delta < 0)
        n = only_candidate + only_baseline
        k = min(only_candidate, only_baseline)
        p_value = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
        return {"only_candidate": only_candidate, "only_baseline": only_baseline, "p_value": p_value}


def _metric_comparison(paired: Paired, criteria: RegressionCriteria) -> dict[str, Any]:
    delta = float(paired.diffs.mean())
    base_mean = float(paired.baseline.mean())
    entry: dict[str, Any] = {
        "n": len(paired.clusters),
        "n_questions": len(paired.groups),
        "baseline": base_mean,
        "candidate": float(paired.candidate.mean()),
        "delta": delta,
        "delta_pct": (delta / base_mean * 100) if base_mean else None,
        "ci95": paired.bootstrap_ci(criteria.n_boot, criteria.seed),
    }
    if paired.is_binary:
        entry["mcnemar"] = paired.sign_test()
    return entry


def _worsening(name: str, value: float) -> float:
    """Signed change in the *worse* direction (positive = worse)."""
    return value if name in LOWER_IS_BETTER else -value


def _verdict(metrics: Mapping[str, Mapping[str, Any]], criteria: RegressionCriteria) -> dict[str, Any]:
    primary = next((m for m in criteria.primary if m in metrics), None)
    reasons: list[str] = []
    skipped: list[str] = []

    breached = False
    for name, limits in criteria.guardrails.items():
        entry = metrics.get(name)
        if entry is None:
            skipped.append(name)
            continue
        worse = _worsening(name, entry["delta"])
        if limits.max_worsening is not None and worse > limits.max_worsening:
            breached = True
            reasons.append(f"{name} worsened by {worse:.4f} (> {limits.max_worsening})")
        if limits.max_worsening_pct is not None and entry["delta_pct"] is not None:
            worse_pct = _worsening(name, entry["delta_pct"])
            if worse_pct > limits.max_worsening_pct:
                breached = True
                reasons.append(f"{name} worsened by {worse_pct:.1f}% (> {limits.max_worsening_pct}%)")

    if primary is None:
        return {
            "decision": "reject" if breached else "inconclusive",
            "primary": None,
            "primary_fallback": False,
            "reasons": reasons + ["no primary metric available in both runs"],
            "skipped_guardrails": skipped,
        }
    primary_fallback = primary != criteria.primary[0]
    if primary_fallback:
        reasons.append(
            f"primary metric fell back to {primary} ({criteria.primary[0]} not scored in both runs)"
        )

    entry = metrics[primary]
    # Orient delta and CI so that positive = better; a CI entirely above
    # zero is a significant gain, entirely below a significant loss.
    sign = -1.0 if primary in LOWER_IS_BETTER else 1.0
    gain = sign * entry["delta"]
    gain_low, gain_high = sorted(sign * bound for bound in entry["ci95"])

    if breached:
        decision = "reject"
    elif gain_high < 0:
        decision = "reject"
        reasons.append(f"{primary} significantly worse (gain {gain:.4f}, 95% CI {gain_low:.4f}..{gain_high:.4f})")
    elif gain_low > 0 and gain >= criteria.min_improvement:
        decision = "accept"
        reasons.append(f"{primary} improved by {gain:.4f} (95% CI {gain_low:.4f}..{gain_high:.4f})")
    else:
        decision = "inconclusive"
        reasons.append(
            f"{primary} gain {gain:.4f} (95% CI {gain_low:.4f}..{gain_high:.4f}) "
            f"not a significant improvement of at least {criteria.min_improvement}"
        )
    return {
        "decision": decision,
        "primary": primary,
        "primary_fallback": primary_fallback,
        "reasons": reasons,
        "skipped_guardrails": skipped,
    }


def compare_runs(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    criteria: RegressionCriteria | Mapping[str, Any],
    question_set: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Paired comparison of ``candidate`` against ``baseline`` plus a verdict.

    Both runs must cover the same questions; a comparison on different
    question sets is not fair and raises. Each metric is compared on the
    samples both runs scored: mean delta (candidate - baseline), paired
    cluster-bootstrap 95% CI, and an exact per-question sign test (McNemar)
    for 0/1 metrics.
    """
    criteria = as_criteria(criteria)
    base_keys, cand_keys = sample_keys(baseline), sample_keys(candidate)
    if set(base_keys) != set(cand_keys):
        missing = len(set(base_keys) - set(cand_keys))
        extra = len(set(cand_keys) - set(base_keys))
        raise ValueError(
            f"runs cover different question sets: candidate lacks {missing} "
            f"and adds {extra} baseline questions; rerun on the frozen set"
        )

    if question_set is not None:
        frozen_keys = sample_keys(question_set)
        if set(frozen_keys) != set(base_keys):
            raise ValueError(
                "runs do not cover the frozen question set "
                f"({len(base_keys)} run questions vs {len(frozen_keys)} frozen); "
                "rerun on the frozen dev or holdout split"
            )

    categories = question_categories(question_set)

    def by_key(results: Sequence[Mapping[str, Any]], keys: list[SampleKey]) -> dict:
        return {
            key: sample_metrics(result, categories.get(key, "unlabeled"))
            for key, result in zip(keys, results)
        }

    base, cand = by_key(baseline, base_keys), by_key(candidate, cand_keys)
    names = sorted({name for m in base.values() for name in m})
    metrics: dict[str, dict[str, Any]] = {}
    for name in names:
        # Per-sample metrics differ by category (no retrieval scores on
        # unanswerable questions), so compare on the samples both scored.
        triples = [
            (base[key][name], cand[key][name], key[0])
            for key in base_keys
            if base[key].get(name) is not None and cand[key].get(name) is not None
        ]
        if triples:
            metrics[name] = _metric_comparison(Paired.from_triples(triples), criteria)

    return {"n": len(base_keys), "metrics": metrics, "verdict": _verdict(metrics, criteria)}
