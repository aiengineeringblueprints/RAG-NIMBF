"""Interaction plot: mean faithfulness per LLM x prompt template.

Reproduces the paper Figure 3 (interaction.pdf): grouped bars
(Concise vs Detailed) per generation LLM, with relative deltas.
Excludes the critic-only Qwen3.5-397B-A17B variant.

Usage:
    python tools/interaction_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "results" / "cross_run_plots" / "interaction.png"

# Display order (matches paper narrative: large generators gain more from Detailed)
ORDER = [
    "gpt-4o-mini",
    "Qwen3.6-27B-Text-NVFP4-MTP",
    "qwen3.5-think",
    "Qwen3-32B-AWQ",
    "gpt-oss_20b",
]
DISPLAY = {
    "gpt-4o-mini": "GPT-4o-mini",
    "Qwen3.6-27B-Text-NVFP4-MTP": "Qwen3.6-27B",
    "qwen3.5-think": "Qwen3.5-think",
    "Qwen3-32B-AWQ": "Qwen3-32B",
    "gpt-oss_20b": "GPT-OSS-20B",
}


def collect() -> dict[str, dict[str, float]]:
    sys.path.insert(0, str(REPO))
    from benchmark.reporting.run_tracker import scan_all_results
    df = scan_all_results(REPO / "results")
    df = df.dropna(subset=["ragas_faithfulness", "prompt_template"])
    df = df[df["ragas_faithfulness"] > 0]
    df = df[~df["llm_short"].astype(str).str.contains("397B", case=False, na=False)]
    df = df[df["llm_short"].isin(DISPLAY.keys())]
    df["prompt_template"] = df["prompt_template"].str.lower()

    out: dict[str, dict[str, float]] = {}
    for llm, grp in df.groupby("llm_short"):
        per_prompt = {}
        for tmpl, sub in grp.groupby("prompt_template"):
            per_prompt[tmpl] = float(sub["ragas_faithfulness"].mean())
        if "concise" in per_prompt and "detailed" in per_prompt:
            out[llm] = per_prompt
    return out


def main() -> None:
    data = collect()
    models = [m for m in ORDER if m in data]
    if not models:
        print("no models matched — aborting")
        return

    concise = [data[m]["concise"] for m in models]
    detailed = [data[m]["detailed"] for m in models]
    labels = [DISPLAY[m] for m in models]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_c = ax.bar(x - width / 2, concise, width, label="Concise",
                    color="#9BB7D4", edgecolor="white")
    bars_d = ax.bar(x + width / 2, detailed, width, label="Detailed",
                    color="#1F3B73", edgecolor="white")

    for i, (c, d) in enumerate(zip(concise, detailed)):
        delta_pp = (d - c) * 100 if c > 0 else 0.0
        ymax = max(c, d)
        ax.text(i, ymax + 0.01, f"+{delta_pp:.0f}pp", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color="#1F3B73")
        ax.text(i - width / 2, c - 0.015, f"{c:.2f}", ha="center",
                va="top", fontsize=8, color="#444")
        ax.text(i + width / 2, d - 0.015, f"{d:.2f}", ha="center",
                va="top", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Faithfulness", fontsize=11, fontweight="bold")
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUT_PATH}")
    for m, c, d in zip(labels, concise, detailed):
        print(f"  {m:18s} concise={c:.3f} detailed={d:.3f}")


if __name__ == "__main__":
    main()
