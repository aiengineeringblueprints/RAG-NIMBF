"""Pareto quality vs speed using external LLM_Performance_Tests throughput.

Replaces paper_quality_vs_speed.png with X-axis = median gen_tok_s from
benchmark_checkpoints/*/*.json in LLM_Performance_Tests-main, Y-axis = mean
ragas_faithfulness from main RAG results.

Usage:
    python tools/pareto_with_perf.py
"""
from __future__ import annotations

import glob
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PERF_DIR = REPO / "LLM_Performance_Tests-main" / "benchmark_checkpoints"
OUT_PATH = REPO / "results" / "cross_run_plots" / "paper_quality_vs_speed.png"

# Map perf model_label -> RAG llm_short (substring match if None)
EXPLICIT = {
    "qwen3-32b-awq": "Qwen3-32B-AWQ",
    "qwen3.6-27b-nvfp4": "Qwen3.6-27B-Text-NVFP4-MTP",
    "qwen3.5-think": "qwen3.5-think",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-oss-20b": "gpt-oss_20b",
}


def _strip_label(label: str) -> str:
    """Drop hardware prefix + parenthetical provider suffix."""
    s = re.sub(r"\([^)]*\)", "", label).strip()
    parts = s.split(maxsplit=1)
    return parts[-1].strip().lower() if len(parts) > 1 else s.lower()


def collect_perf() -> dict[str, dict]:
    """Return {rag_llm_short: {median_tps, mean_tps, std_tps, n, source_label}}."""
    out: dict[str, dict] = {}
    for f in sorted(PERF_DIR.glob("*/*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        label = d.get("model_label", "")
        stripped = _strip_label(label)
        rag_key = None
        for sub, mapped in EXPLICIT.items():
            if sub in stripped:
                rag_key = mapped
                break
        if not rag_key:
            continue
        toks = [r.get("gen_tok_s") for r in d.get("results", []) if r.get("gen_tok_s") is not None and r.get("success", True)]
        if not toks:
            continue
        agg = out.setdefault(rag_key, {"tokens": [], "label": label})
        agg["tokens"].extend(toks)
    final: dict[str, dict] = {}
    for k, v in out.items():
        t = v["tokens"]
        final[k] = {
            "median_tps": float(statistics.median(t)),
            "mean_tps": float(statistics.mean(t)),
            "std_tps": float(statistics.stdev(t)) if len(t) > 1 else 0.0,
            "n": len(t),
            "label": v["label"],
        }
    return final


def collect_rag_faithfulness() -> dict[str, dict]:
    sys.path.insert(0, str(REPO))
    from benchmark.reporting.run_tracker import scan_all_results
    df = scan_all_results(REPO / "results")
    df = df.dropna(subset=["ragas_faithfulness"])
    df = df[df["ragas_faithfulness"] > 0]
    df = df[~df["llm_short"].astype(str).str.contains("397B", case=False, na=False)]
    g = df.groupby("llm_short")["ragas_faithfulness"].agg(["mean", "std", "count"])
    return {
        idx: {"mean": r["mean"], "std": r["std"] or 0.0, "n": int(r["count"])}
        for idx, r in g.iterrows()
    }


def main() -> None:
    perf = collect_perf()
    rag = collect_rag_faithfulness()
    print(f"perf models: {len(perf)} | rag models: {len(rag)}")

    rows = []
    for k, p in perf.items():
        if k not in rag:
            print(f"  [skip] {k} — no RAG data")
            continue
        r = rag[k]
        rows.append({
            "label": k,
            "perf_label": p["label"],
            "x": p["median_tps"],
            "x_std": p["std_tps"],
            "y": r["mean"],
            "y_std": r["std"],
            "n_perf": p["n"],
            "n_rag": r["n"],
        })
    print(f"matched models: {len(rows)}")
    for r in rows:
        print(f"  {r['label']:35s} x={r['x']:.1f} tok/s  y={r['y']:.3f}")

    if not rows:
        print("no rows — aborting")
        return

    PALETTE = ["#0E7C7B", "#F2A541", "#233142", "#C0392B", "#7F8C8D", "#2C7873", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, r in enumerate(rows):
        ax.scatter(
            r["x"], r["y"],
            s=120, alpha=0.8,
            color=PALETTE[i % len(PALETTE)],
            edgecolors="white", linewidths=0.8,
            label=r["label"],
        )

    ax.set_xlabel("Throughput — median gen tok/s (LLM_Performance_Tests)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Faithfulness", fontsize=11, fontweight="bold")
    ax.set_title("Quality vs. Speed Trade-off", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
