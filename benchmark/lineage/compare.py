"""Paired comparison of a baseline and a candidate run (issue #65).

Samples are paired by stable ``sample_id``. A metric missing on one side is
counted as missing, never scored as 0. Confidence intervals come from the
paired bootstrap already used by ``benchmark.regression``, resampling whole
questions (``question_id``) so a question asked with several reference
answers is not counted as independent evidence.

Status:
- ``paired``: verified same scorer and corpus, identical sample sets.
- ``partially_paired``: verified same scorer and corpus, overlapping but
  different sample sets; statistics use the overlap only.
- ``unverified``: shared samples, but the scorer or corpus identity of at
  least one side is unknown (e.g. a run recorded before lineage existed);
  deltas are shown but must not be read as a paired comparison.
- ``not_comparable``: different scorer identity, different corpus
  fingerprint, or no shared samples.

Rendering lives in ``benchmark.lineage.report``, MLflow publishing in
``benchmark.lineage.publish``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.lineage.provenance import SystemProvenance
from benchmark.lineage.runs import RunRecord, scorer_fingerprint
from benchmark.regression.compare import Paired
from benchmark.regression.metrics import LOWER_IS_BETTER

COMPARISON_SCHEMA_VERSION = 1
NEUTRAL = frozenset({"abstained"})
N_BOOT = 2000
SEED = 0


def _direction(metric: str) -> str:
    if metric in NEUTRAL:
        return "neutral"
    return "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"


def _side(record: RunRecord) -> dict[str, Any]:
    lineage = record.lineage or {}
    scorer = lineage.get("scorer") or None
    return {
        "ref": record.ref,
        "config_name": record.config_name,
        "experiment": dict(lineage.get("experiment") or {}),
        "system": SystemProvenance.from_dict(lineage.get("system")).as_dict(),
        "dataset": dict(lineage.get("dataset") or {}),
        "scorer": scorer,
        "scorer_fingerprint": scorer_fingerprint(scorer),
        "framework": dict(lineage.get("framework") or {}),
        "mlflow_run_id": record.mlflow_run_id,
        "num_samples": len(record.samples),
    }


def _by_id(record: RunRecord) -> dict[str, Mapping[str, Any]]:
    return {s["sample_id"]: s for s in record.samples}


def _cluster(sample: Mapping[str, Any]) -> str:
    """Bootstrap cluster: the question, so repeats are resampled together."""
    return sample.get("question_id") or sample["sample_id"]


def _stats(pairs: Sequence[tuple[float, float, str]]) -> dict[str, Any]:
    paired = Paired.from_triples(pairs)
    base, cand = float(paired.baseline.mean()), float(paired.candidate.mean())
    clusters = len({cluster for _, _, cluster in pairs})
    ci = list(paired.bootstrap_ci(N_BOOT, SEED)) if clusters >= 2 else None
    return {
        "baseline": base,
        "candidate": cand,
        "delta": cand - base,
        "n": len(pairs),
        "clusters": clusters,
        "ci95": ci,
    }


def _unverified_reasons(base: dict, cand: dict) -> list[str]:
    """Conditions that cannot be checked because one side never recorded them."""
    reasons = []
    for label, side in (("baseline", base), ("candidate", cand)):
        if side["scorer_fingerprint"] is None:
            reasons.append(f"{label} scorer identity unknown (run predates lineage)")
    b_fp = base["dataset"].get("corpus_fingerprint")
    c_fp = cand["dataset"].get("corpus_fingerprint")
    if bool(b_fp) != bool(c_fp):
        unknown = "baseline" if not b_fp else "candidate"
        reasons.append(f"{unknown} corpus fingerprint unknown; same corpus cannot be verified")
    return reasons


def _error_counts(record: RunRecord) -> dict[str, Any]:
    return {
        "invalid_answers": sum(s.get("answer_valid") is False for s in record.samples),
        "evaluation_error": record.run_errors.get("evaluation_error"),
    }


def _blocking_reasons(base: dict, cand: dict, shared: list[str]) -> list[str]:
    reasons = []
    if (
        base["scorer_fingerprint"]
        and cand["scorer_fingerprint"]
        and base["scorer_fingerprint"] != cand["scorer_fingerprint"]
    ):
        changed = sorted(
            k for k in set(base["scorer"]) | set(cand["scorer"])
            if base["scorer"].get(k) != cand["scorer"].get(k)
        )
        reasons.append(f"scorer identity differs ({', '.join(changed)})")
    b_fp = base["dataset"].get("corpus_fingerprint")
    c_fp = cand["dataset"].get("corpus_fingerprint")
    if b_fp and c_fp and b_fp != c_fp:
        reasons.append("corpus fingerprints differ: retrieval searched different documents")
    if not shared:
        reasons.append("no shared samples between baseline and candidate")
    return reasons


def _warnings(base: dict, cand: dict) -> list[str]:
    warnings = []
    for label, side in (("baseline", base), ("candidate", cand)):
        if side["system"]["status"] == "not_available":
            warnings.append(f"{label} system provenance not available")
        elif side["system"]["status"] == "dirty":
            warnings.append(f"{label} system checkout was dirty (diff hash recorded)")
    b_ds, c_ds = base["dataset"], cand["dataset"]
    if b_ds.get("name") and c_ds.get("name") and b_ds["name"] != c_ds["name"]:
        warnings.append(f"dataset names differ: {b_ds['name']} vs {c_ds['name']}")
    intended = cand["experiment"].get("baseline_run_id")
    if not intended:
        warnings.append("candidate does not declare baseline_run_id in its experiment metadata")
    elif intended not in (base["ref"], base.get("mlflow_run_id")):
        warnings.append(
            f"candidate declares baseline_run_id {intended}, "
            f"but was compared against {base['ref']}"
        )
    return warnings


def compare_runs(baseline: RunRecord, candidate: RunRecord) -> dict[str, Any]:
    base_side, cand_side = _side(baseline), _side(candidate)
    base_samples, cand_samples = _by_id(baseline), _by_id(candidate)
    shared = [sid for sid in base_samples if sid in cand_samples]
    reasons = _blocking_reasons(base_side, cand_side, shared)
    unverified = [] if reasons else _unverified_reasons(base_side, cand_side)
    pairing = {
        "paired": len(shared),
        "only_baseline": len(base_samples) - len(shared),
        "only_candidate": len(cand_samples) - len(shared),
    }
    if reasons:
        status = "not_comparable"
    elif unverified:
        status = "unverified"
        reasons = unverified
    elif pairing["only_baseline"] or pairing["only_candidate"]:
        status = "partially_paired"
    else:
        status = "paired"

    result: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": status,
        "reasons": reasons,
        "warnings": _warnings(base_side, cand_side),
        "pairing": pairing,
        "errors": {"baseline": _error_counts(baseline), "candidate": _error_counts(candidate)},
        "baseline": base_side,
        "candidate": cand_side,
        "metrics": {},
        "categories": {},
        "questions": [],
    }
    if status == "not_comparable":
        return result

    metric_names = sorted(
        {m for sid in shared for m in base_samples[sid]["metrics"]}
        | {m for sid in shared for m in cand_samples[sid]["metrics"]}
    )
    per_metric: dict[str, list[tuple[float, float, str]]] = {m: [] for m in metric_names}
    per_category: dict[str, dict[str, list[tuple[float, float, str]]]] = {}
    for sid in shared:
        b, c = base_samples[sid], cand_samples[sid]
        category = c.get("category") or b.get("category") or "uncategorized"
        deltas: dict[str, float | None] = {}
        for metric in metric_names:
            bv, cv = b["metrics"].get(metric), c["metrics"].get(metric)
            if bv is None or cv is None:
                deltas[metric] = None
                continue
            pair = (bv, cv, _cluster(c))
            per_metric[metric].append(pair)
            per_category.setdefault(category, {}).setdefault(metric, []).append(pair)
            deltas[metric] = cv - bv
        result["questions"].append({"sample_id": sid, "category": category, "deltas": deltas})

    for metric in metric_names:
        missing_b = sum(base_samples[s]["metrics"].get(metric) is None for s in shared)
        missing_c = sum(cand_samples[s]["metrics"].get(metric) is None for s in shared)
        entry: dict[str, Any] = {"direction": _direction(metric)}
        if per_metric[metric]:
            entry.update(_stats(per_metric[metric]))
        else:
            entry.update(baseline=None, candidate=None, delta=None, n=0, clusters=0, ci95=None)
        entry.update(missing_baseline=missing_b, missing_candidate=missing_c)
        result["metrics"][metric] = entry

    for category, metrics in sorted(per_category.items()):
        result["categories"][category] = {
            metric: {
                "delta": float(np.mean([c - b for b, c, _ in pairs])),
                "n": len(pairs),
            }
            for metric, pairs in sorted(metrics.items())
        }
    return result
