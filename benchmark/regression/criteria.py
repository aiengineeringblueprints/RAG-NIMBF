"""Typed decision policy of a regression comparison (``criteria.yaml``).

The criteria are an offline analysis policy, not a run knob: they decide how
two finished runs are judged and never change how a run executes, so they live
in their own versioned file instead of ``BenchmarkConfig``
(doc/adr/006-regression-criteria-are-an-analysis-policy.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

_CRITERIA_KEYS = {"primary", "min_improvement", "guardrails", "n_boot", "seed"}
_GUARDRAIL_KEYS = {"max_worsening", "max_worsening_pct"}


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Guardrail:
    """Largest tolerated worsening of one metric (point estimate)."""

    max_worsening: float | None = None
    max_worsening_pct: float | None = None

    @classmethod
    def from_mapping(cls, metric: str, limits: Mapping[str, Any]) -> "Guardrail":
        if not isinstance(limits, Mapping):
            raise ValueError(f"guardrail {metric}: expected a mapping of limits, got {limits!r}")
        unknown = set(limits) - _GUARDRAIL_KEYS
        if unknown:
            raise ValueError(
                f"guardrail {metric}: unknown limit(s) {sorted(unknown)}; use {sorted(_GUARDRAIL_KEYS)}"
            )
        if not limits:
            raise ValueError(f"guardrail {metric}: no limit given")
        return cls(**{key: _number(value, f"guardrail {metric}.{key}") for key, value in limits.items()})


@dataclass(frozen=True)
class RegressionCriteria:
    """When a candidate counts as better than a reference run.

    - ``primary``: ordered metric preference; the first metric both runs
      scored decides.
    - ``min_improvement``: smallest gain (metric units) worth accepting.
    - ``guardrails``: trade-offs a candidate may cost at most.
    - ``n_boot`` / ``seed``: paired cluster bootstrap settings.
    """

    primary: tuple[str, ...]
    min_improvement: float = 0.0
    guardrails: Mapping[str, Guardrail] = field(default_factory=dict)
    n_boot: int = 2000
    seed: int = 0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RegressionCriteria":
        unknown = set(raw) - _CRITERIA_KEYS
        if unknown:
            raise ValueError(f"criteria: unknown key(s) {sorted(unknown)}; use {sorted(_CRITERIA_KEYS)}")
        primary = raw.get("primary")
        if not primary or isinstance(primary, str) or not all(isinstance(m, str) for m in primary):
            raise ValueError("criteria.primary must be a non-empty list of metric names")
        n_boot = raw.get("n_boot", 2000)
        seed = raw.get("seed", 0)
        if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 100:
            raise ValueError(f"criteria.n_boot must be an integer >= 100, got {n_boot!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"criteria.seed must be an integer, got {seed!r}")
        return cls(
            primary=tuple(primary),
            min_improvement=_number(raw.get("min_improvement", 0.0), "criteria.min_improvement"),
            guardrails={
                metric: Guardrail.from_mapping(metric, limits)
                for metric, limits in (raw.get("guardrails") or {}).items()
            },
            n_boot=n_boot,
            seed=seed,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegressionCriteria":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primary"] = list(self.primary)
        data["guardrails"] = {
            metric: {k: v for k, v in asdict(limit).items() if v is not None}
            for metric, limit in self.guardrails.items()
        }
        return data


def as_criteria(criteria: "RegressionCriteria | Mapping[str, Any]") -> RegressionCriteria:
    return criteria if isinstance(criteria, RegressionCriteria) else RegressionCriteria.from_mapping(criteria)
