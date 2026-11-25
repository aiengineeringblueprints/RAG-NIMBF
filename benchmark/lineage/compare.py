"""Paired comparison of a baseline and a candidate run (issue #65).

Samples are paired by stable ``sample_id``. A metric missing on one side is
counted as missing, never scored as 0. Confidence intervals come from the
paired bootstrap already used by ``benchmark.regression``.

Status:
- ``paired``: same scorer, identical sample sets.
- ``partially_paired``: same scorer, overlapping but different sample sets;
  statistics use the overlap only.
- ``not_comparable``: different scorer identity, different corpus
  fingerprint, or no shared samples.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark.lineage.provenance import FRAMEWORK_ROOT, SystemProvenance
from benchmark.lineage.runs import RunRecord, lineage_tags, scorer_fingerprint
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


def _stats(pairs: Sequence[tuple[float, float, str]]) -> dict[str, Any]:
    paired = Paired.from_triples(pairs)
    base, cand = float(paired.baseline.mean()), float(paired.candidate.mean())
    ci = list(paired.bootstrap_ci(N_BOOT, SEED)) if len(pairs) >= 2 else None
    return {"baseline": base, "candidate": cand, "delta": cand - base, "n": len(pairs), "ci95": ci}


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
        if side["scorer_fingerprint"] is None:
            warnings.append(f"{label} scorer identity unknown (run predates lineage)")
        if side["system"]["status"] == "not_available":
            warnings.append(f"{label} system provenance not available")
        elif side["system"]["status"] == "dirty":
            warnings.append(f"{label} system checkout was dirty (diff hash recorded)")
    b_ds, c_ds = base["dataset"], cand["dataset"]
    if b_ds.get("name") and c_ds.get("name") and b_ds["name"] != c_ds["name"]:
        warnings.append(f"dataset names differ: {b_ds['name']} vs {c_ds['name']}")
    intended = cand["experiment"].get("baseline_run_id")
    if intended and intended not in (base["ref"], base.get("mlflow_run_id")):
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
    pairing = {
        "paired": len(shared),
        "only_baseline": len(base_samples) - len(shared),
        "only_candidate": len(cand_samples) - len(shared),
    }
    if reasons:
        status = "not_comparable"
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
            per_metric[metric].append((bv, cv, sid))
            per_category.setdefault(category, {}).setdefault(metric, []).append((bv, cv, sid))
            deltas[metric] = cv - bv
        result["questions"].append({"sample_id": sid, "category": category, "deltas": deltas})

    for metric in metric_names:
        missing_b = sum(base_samples[s]["metrics"].get(metric) is None for s in shared)
        missing_c = sum(cand_samples[s]["metrics"].get(metric) is None for s in shared)
        entry: dict[str, Any] = {"direction": _direction(metric)}
        if per_metric[metric]:
            entry.update(_stats(per_metric[metric]))
        else:
            entry.update(baseline=None, candidate=None, delta=None, n=0, ci95=None)
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


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "–"
    if isinstance(value, (int, float)):
        return f"{value:+.4f}" if signed else f"{value:.4f}"
    return str(value)


def _provenance_line(side: Mapping[str, Any]) -> str:
    system = side["system"]
    version = system.get("version") or "unknown"
    branch = f", branch {system['branch']}" if system.get("branch") else ""
    return (
        f"`{side['ref']}` ({side.get('config_name') or 'n/a'}): "
        f"system **{system.get('system_id')}** {version}{branch} "
        f"[{system.get('status')}, source {system.get('version_source')}]"
    )


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Run comparison",
        "",
        f"**Status:** `{result['status']}`",
        "",
        f"- Baseline: {_provenance_line(result['baseline'])}",
        f"- Candidate: {_provenance_line(result['candidate'])}",
        f"- Pairing: {result['pairing']['paired']} paired, "
        f"{result['pairing']['only_baseline']} only in baseline, "
        f"{result['pairing']['only_candidate']} only in candidate",
    ]
    for side in ("baseline", "candidate"):
        ds = result[side]["dataset"]
        lines.append(
            f"- {side.title()} dataset: {ds.get('name') or 'unknown'}, "
            f"question set `{ds.get('question_set_hash') or 'unknown'}`, "
            f"corpus `{ds.get('corpus_fingerprint') or 'unknown'}`, "
            f"scorer `{result[side]['scorer_fingerprint'] or 'unknown'}`"
        )
    experiment = result["candidate"]["experiment"]
    if experiment:
        lines += ["", "## Experiment", ""]
        lines += [f"- {key}: {value}" for key, value in experiment.items()]
    if result["reasons"]:
        lines += ["", "## Not comparable", ""] + [f"- {r}" for r in result["reasons"]]
    if result["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in result["warnings"]]
    if result["metrics"]:
        lines += [
            "",
            "## Metrics (paired samples)",
            "",
            "| Metric | Direction | Baseline | Candidate | Delta | 95% CI | n | Missing (b/c) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, m in result["metrics"].items():
            ci = f"[{_fmt(m['ci95'][0], signed=True)}, {_fmt(m['ci95'][1], signed=True)}]" if m["ci95"] else "–"
            lines.append(
                f"| {name} | {m['direction']} | {_fmt(m['baseline'])} | {_fmt(m['candidate'])} "
                f"| {_fmt(m['delta'], signed=True)} | {ci} | {m['n']} "
                f"| {m['missing_baseline']}/{m['missing_candidate']} |"
            )
    if result["categories"]:
        lines += ["", "## Categories (mean delta, n)", ""]
        for category, metrics in result["categories"].items():
            cells = ", ".join(f"{k} {_fmt(v['delta'], signed=True)} (n={v['n']})" for k, v in metrics.items())
            lines.append(f"- **{category}**: {cells}")
    lines += ["", "Per-question deltas: see `comparison.json` (`questions`).", ""]
    return "\n".join(lines)


def write_comparison(result: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "comparison.json", out / "comparison.md"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def default_output_dir(baseline: RunRecord, candidate: RunRecord) -> Path:
    def slug(record: RunRecord) -> str:
        name = record.config_name or Path(record.ref.partition("#")[0]).stem or record.ref
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:60]

    base_path = Path(baseline.ref.partition("#")[0])
    root = (
        base_path.parent / "comparisons"
        if base_path.is_file()
        else FRAMEWORK_ROOT / "results" / "comparisons"
    )
    return root / f"{slug(baseline)}__vs__{slug(candidate)}"


def log_comparison_to_mlflow(result: Mapping[str, Any], json_path: Path, md_path: Path) -> str:
    """Log the comparison as its own MLflow run, linked to both inputs by tag."""
    import mlflow

    tags = {
        "run_type": "comparison",
        "comparison.status": result["status"],
        "comparison.baseline": str(result["baseline"]["ref"]),
        "comparison.candidate": str(result["candidate"]["ref"]),
    }
    tags.update(lineage_tags({"experiment": result["candidate"]["experiment"]}))
    for side in ("baseline", "candidate"):
        if result[side].get("mlflow_run_id"):
            tags[f"comparison.{side}_run_id"] = result[side]["mlflow_run_id"]
    with mlflow.start_run(run_name=f"compare:{json_path.parent.name}"[:250], tags=tags) as run:
        for name, m in result["metrics"].items():
            if m["delta"] is not None:
                mlflow.log_metric(f"delta.{name}", m["delta"])
        mlflow.log_artifact(str(json_path), artifact_path="comparison")
        mlflow.log_artifact(str(md_path), artifact_path="comparison")
        return run.info.run_id
