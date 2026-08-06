"""ID sampling distributions for the workload generator.

Both samplers take a sequence of identifiers (question IDs, document IDs,
etc.) and return a single sampled ID. ``uniform_sample`` gives every ID
equal probability; ``zipfian_sample`` favours a small hot set, modelling
real-world request skew.

The implementations deliberately accept IDs as input rather than an
integer ``N`` so the caller controls which items are exposed — this lets
the runner mutate the live ID pool (Insert adds IDs, Remove takes them
away) without re-wiring the sampler.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def uniform_sample(ids: Sequence[str], rng: random.Random | None = None) -> str:
    """Return one of ``ids`` with equal probability.

    Raises
    ------
    ValueError
        If ``ids`` is empty.
    """
    if not ids:
        raise ValueError("uniform_sample requires at least one id")
    rng = rng or random
    return rng.choice(list(ids))


def _zipf_pmf(rank: int, theta: float, harmonic: float) -> float:
    """Unnormalised Zipf probability for a 1-indexed rank."""
    return 1.0 / ((rank + 1) ** theta * harmonic)


def _zipf_cumulative(n: int, theta: float) -> list[float]:
    """Return the CDF table over ``n`` ranks for a Zipf(theta)."""
    if n <= 0:
        return []
    harmonic = sum(1.0 / ((k + 1) ** theta) for k in range(n))
    cdf: list[float] = []
    running = 0.0
    for k in range(n):
        running += _zipf_pmf(k, theta, harmonic)
        cdf.append(running)
    # Numerical safety: ensure the final bucket hits 1.0 exactly.
    if cdf:
        cdf[-1] = 1.0
    return cdf


def zipfian_sample(
    ids: Sequence[str],
    theta: float = 0.8,
    rng: random.Random | None = None,
) -> str:
    """Sample one of ``ids`` with Zipf(theta) skew.

    The first id in ``ids`` is the hottest. Higher ``theta`` concentrates
    more probability mass on the top of the ordering. ``theta <= 0``
    degenerates to uniform.

    The CDF is recomputed on each call: fine for the corpus sizes a single
    RAG benchmark cares about (low thousands). For very large pools the
    caller should cache the table externally.
    """
    if not ids:
        raise ValueError("zipfian_sample requires at least one id")
    if theta <= 0:
        return uniform_sample(ids, rng)
    rng = rng or random
    n = len(ids)
    cdf = _zipf_cumulative(n, theta)
    if n == 1:
        return ids[0]
    target = rng.random()
    # Linear scan; n is small in practice. For huge n, swap for bisect.
    for index, cumulative in enumerate(cdf):
        if target <= cumulative:
            return ids[index]
    return ids[-1]


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats. Useful for tests asserting skew exists."""
    return -sum(p * math.log(p) for p in probs if p > 0.0)
