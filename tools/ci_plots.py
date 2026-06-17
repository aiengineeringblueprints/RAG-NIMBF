"""Bootstrap CI + paired Wilcoxon plots for RAG-NIMBF poster.

Reads per-sample scores from results/run*/benchmark_*.json, computes
bootstrap 95% CI on the mean per group, runs paired Wilcoxon signed-rank
between groups on shared questions, and writes standalone PNGs to
results/cross_run_plots/. Does not modify existing plots or the poster.

Usage:
    python tools/ci_plots.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "cross_run_plots"

METRICS = {
    "ragas_faithfulness": ("faithfulness", "ragas"),
    "custom_ndcg@5": ("ndcg@5", "custom"),
    "ragas_context_recall": ("context_recall", "ragas"),
}

BAR_COLORS = {
    "concise": "#0E7C7B",
    "detailed": "#F2A541",
    "similarity": "#233142",
    "mmr": "#0E7C7B",
    "recursive": "#233142",
    "semantic": "#0E7C7B",
    "text": "#F2A541",
    "token": "#C0392B",
    "markdown": "#7F8C8D",
}
DEFAULT_COLOR = "#7F8C8D"


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size < 2:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def _infer_retrieval(entry: dict) -> str:
    rs = entry.get("retrieval_strategy")
    if rs:
        return str(rs)
    # Fall back to parsing config_name — old runs encode `mmr-l0.5` or omit it
    name = (entry.get("config_name") or "").lower()
    if "mmr" in name:
        return "mmr"
    if "hyde" in name:
        return "hyde"
    return "similarity"


def collect() -> list[dict]:
    rows = []
    for jf in sorted(RESULTS.glob("run*/benchmark_*.json")):
        run_name = jf.parent.name
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for entry in data.get("results", []):
            per_sample = entry.get("per_sample") or []
            if not per_sample:
                continue
            rows.append({
                "run_name": run_name,
                "prompt_template": entry.get("prompt_template"),
                "retrieval_strategy": _infer_retrieval(entry),
                "chunking_strategy": entry.get("chunking_strategy"),
                "llm_model": entry.get("llm_model"),
                "per_sample": per_sample,
            })
    return rows


def group_scores(rows: list[dict], dim: str, metric_label: str, source: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        key = (r.get(dim) or "unknown").lower()
        for s in r["per_sample"]:
            scores = s.get(f"{source}_scores") or {}
            v = scores.get(metric_label)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out[key].append(float(v))
    return out


def paired_by_question(rows: list[dict], dim: str, metric_label: str, source: str) -> dict[str, dict[str, float]]:
    """For each level of dim, map question -> score, aggregated across runs."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get(dim) or "unknown").lower()
        for s in r["per_sample"]:
            q = s.get("question")
            if not q:
                continue
            scores = s.get(f"{source}_scores") or {}
            v = scores.get(metric_label)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out[key][q].append(float(v))
    # collapse multi-occurrences to mean per question
    collapsed = {}
    for k, qmap in out.items():
        collapsed[k] = {q: float(np.mean(vs)) for q, vs in qmap.items()}
    return collapsed


def wilcoxon_paired(a_scores: dict[str, float], b_scores: dict[str, float]) -> tuple[float, float] | None:
    shared = sorted(set(a_scores) & set(b_scores))
    if len(shared) < 5:
        return None
    a = [a_scores[q] for q in shared]
    b = [b_scores[q] for q in shared]
    diff = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diff):
        return (0.0, float(len(shared)))
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return None
    return (float(p), float(len(shared)))


def annotate_significance(ax, x1: float, x2: float, y: float, p: float) -> None:
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    ax.plot([x1, x1, x2, x2], [y, y * 1.02, y * 1.02, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y * 1.025, f"{sig}\np={p:.3g}", ha="center", va="bottom", fontsize=8)


def plot_grouped_bars_ci(
    groups: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
    paired: dict[str, dict[str, float]] | None = None,
) -> Path | None:
    if not groups:
        return None
    levels = sorted(groups.keys())
    stats = [bootstrap_ci(groups[lv]) for lv in levels]
    means = [s[0] for s in stats]
    errs = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    ns = [len(groups[lv]) for lv in levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(levels))
    colors = [BAR_COLORS.get(lv, DEFAULT_COLOR) for lv in levels]
    bars = ax.bar(x, means, yerr=errs, capsize=6, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m / 2, f"n={n}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(levels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(mean with bootstrap 95% CI, n = per-question scores)")
    ax.set_ylim(0, max(m + e + 0.05 for m, e in zip(means, errs[1])) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Pairwise Wilcoxon brackets if paired question-level data supplied
    if paired and len(levels) == 2:
        p = wilcoxon_paired(paired[levels[0]], paired[levels[1]])
        if p is not None:
            pval, _n = p
            y_top = max(m + e for m, e in zip(means, errs[1])) * 1.02
            annotate_significance(ax, x[0], x[1], y_top, pval)
    elif paired and len(levels) > 2:
        y_top = max(m + e for m, e in zip(means, errs[1])) * 1.02
        step = y_top * 0.05
        for i, (a, b) in enumerate(combinations(range(len(levels)), 2)):
            p = wilcoxon_paired(paired[levels[a]], paired[levels[b]])
            if p is None:
                continue
            pval, _n = p
            y = y_top + i * step * 2
            annotate_significance(ax, x[a], x[b], y, pval)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")
    return out_path


def main() -> None:
    rows = collect()
    print(f"collected {len(rows)} config rows across runs")

    # Plot 1: faithfulness by prompt template
    g = group_scores(rows, "prompt_template", "faithfulness", "ragas")
    p = paired_by_question(rows, "prompt_template", "faithfulness", "ragas")
    plot_grouped_bars_ci(
        g,
        title="RAGAS faithfulness by prompt template",
        ylabel="faithfulness (mean)",
        out_path=OUT / "ci_faithfulness_by_prompt.png",
        paired=p,
    )

    # Plot 2: nDCG@5 by retrieval strategy
    g = group_scores(rows, "retrieval_strategy", "ndcg@5", "custom")
    p = paired_by_question(rows, "retrieval_strategy", "ndcg@5", "custom")
    plot_grouped_bars_ci(
        g,
        title="nDCG@5 by retrieval strategy",
        ylabel="nDCG@5 (mean)",
        out_path=OUT / "ci_ndcg5_by_retrieval.png",
        paired=p,
    )

    # Plot 3: faithfulness by chunking strategy
    g = group_scores(rows, "chunking_strategy", "faithfulness", "ragas")
    p = paired_by_question(rows, "chunking_strategy", "faithfulness", "ragas")
    plot_grouped_bars_ci(
        g,
        title="RAGAS faithfulness by chunking strategy",
        ylabel="faithfulness (mean)",
        out_path=OUT / "ci_faithfulness_by_chunking.png",
        paired=p,
    )

    # Also print a small text summary so numbers can be cross-checked
    print("\n=== Summary (mean, 95% CI, n) ===")
    for dim, (metric_label, source) in [
        ("prompt_template", ("faithfulness", "ragas")),
        ("retrieval_strategy", ("ndcg@5", "custom")),
        ("chunking_strategy", ("faithfulness", "ragas")),
    ]:
        print(f"\n[{dim} | {metric_label}]")
        g = group_scores(rows, dim, metric_label, source)
        for lv, vals in g.items():
            m, lo, hi = bootstrap_ci(vals)
            print(f"  {lv:12s} mean={m:.4f}  CI=[{lo:.4f}, {hi:.4f}]  n={len(vals)}")


if __name__ == "__main__":
    main()
