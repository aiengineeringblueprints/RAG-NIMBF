"""Render and write a run comparison (issue #65): Markdown + JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from benchmark.lineage.provenance import FRAMEWORK_ROOT
from benchmark.lineage.runs import RunRecord


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
        heading = "Unverified: not a paired comparison" if result["status"] == "unverified" else "Not comparable"
        lines += ["", f"## {heading}", ""] + [f"- {r}" for r in result["reasons"]]
    if result["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in result["warnings"]]
    if errors := result.get("errors"):
        lines += ["", "## Errors", ""]
        for side in ("baseline", "candidate"):
            err = errors[side]
            lines.append(
                f"- {side.title()}: {err['invalid_answers']} invalid answers, "
                f"evaluation error: {err['evaluation_error'] or 'none'}"
            )
    if result["metrics"]:
        lines += [
            "",
            "## Metrics (paired samples)",
            "",
            "| Metric | Direction | Baseline | Candidate | Delta | 95% CI | n (questions) | Missing (b/c) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, m in result["metrics"].items():
            ci = f"[{_fmt(m['ci95'][0], signed=True)}, {_fmt(m['ci95'][1], signed=True)}]" if m["ci95"] else "–"
            lines.append(
                f"| {name} | {m['direction']} | {_fmt(m['baseline'])} | {_fmt(m['candidate'])} "
                f"| {_fmt(m['delta'], signed=True)} | {ci} | {m['n']} ({m.get('clusters', m['n'])}) "
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
