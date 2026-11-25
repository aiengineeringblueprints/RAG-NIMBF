"""Markdown rendering of a regression comparison."""

from __future__ import annotations

from typing import Any, Mapping


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def render_report(report: Mapping[str, Any]) -> str:
    lines = [f"# Blueprint regression report ({report['split']} split)", ""]
    for name, path in report["runs"].items():
        lines.append(f"- **{name}**: `{path}`")
    lines.append("")
    check = report.get("config_check")
    if check:
        lines.append(f"Config check: **{check['status']}**")
        if check["missing"]:
            lines.append(f"- no recorded config for: {', '.join(check['missing'])}")
        for field, values in check["mismatches"].items():
            lines.append(f"- evaluation setting `{field}` differs: {values}")
        for ref, changed in check["changed"].items():
            components = ", ".join(f"`{field}`" for field in changed) or "none"
            lines.append(f"- candidate changes vs {ref}: {components}")
        lines.append("")
    for ref, comparison in report["comparisons"].items():
        verdict = comparison["verdict"]
        lines += [
            f"## Candidate vs {ref}: **{verdict['decision']}**",
            "",
            f"Primary metric: `{verdict['primary']}`" + (" (fallback)" if verdict.get("primary_fallback") else ""),
            "",
            *[f"- {reason}" for reason in verdict["reasons"]],
            *[f"- guardrail `{g}` skipped: not scored in both runs" for g in verdict["skipped_guardrails"]],
            "",
            f"| metric | {ref} | candidate | delta | 95% CI | sign test p | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, entry in comparison["metrics"].items():
            p_value = entry.get("mcnemar", {}).get("p_value")
            low, high = entry["ci95"]
            lines.append(
                f"| {metric} | {_fmt(entry['baseline'])} | {_fmt(entry['candidate'])} | "
                f"{_fmt(entry['delta'])} | {_fmt(low)}..{_fmt(high)} | {_fmt(p_value, 3)} | {entry['n']} |"
            )
        lines.append("")
    lines += ["## Dimensions per run", ""]
    for name, summary in report["summaries"].items():
        lines.append(f"### {name} (n={summary['n']})")
        for dim, info in summary["dimensions"].items():
            if info["status"] == "ok":
                values = ", ".join(f"{k}={_fmt(v)}" for k, v in info["metrics"].items())
                lines.append(f"- {dim}: {values}")
            else:
                lines.append(f"- {dim}: unavailable — {info['reason']}")
        lines.append("")
    return "\n".join(lines)
