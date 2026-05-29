#!/usr/bin/env python3
"""Comparison plots for LLM Performance Test checkpoint results."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

CHECKPOINT_DIR = Path(__file__).parent / "benchmark_checkpoints" / "2026-05-29_07-51-22"
OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────

def parse_filename(name: str):
    """N{n}_T{tokens}__{host}_{model}__{provider}.json"""
    m = re.match(r"N(\d+)_T(\d+)__(.+)\.json", name)
    if not m:
        return None
    n_calls, max_tokens, label = m.groups()
    # label like "RTX5090_Qwen3-32B-AWQ__vLLM_" — model is between first _ and __provider_
    parts = label.split("__")
    model_label = parts[0].replace("_", " ", 1).strip()  # e.g. "RTX5090 Qwen3-32B-AWQ"
    # cleaner: extract just model name
    host_model = parts[0]  # "RTX5090_Qwen3-32B-AWQ"
    provider = parts[1].rstrip("_") if len(parts) > 1 else "unknown"
    return {
        "n_calls": int(n_calls),
        "max_tokens": int(max_tokens),
        "model_label": host_model.replace("_", " "),
        "provider": provider,
    }


def load_all():
    records = []
    for fp in sorted(CHECKPOINT_DIR.glob("*.json")):
        meta = parse_filename(fp.name)
        if not meta:
            continue
        with open(fp) as f:
            data = json.load(f)
        for r in data["results"]:
            records.append({**meta, **r})
    return records


# ── color palette ─────────────────────────────────────────────────────────

PALETTE = {
    "RTX5090 Qwen3-32B-AWQ": "#1f77b4",
    "Sigurt GPT-4o-mini": "#ff7f0e",
    "Sigurt Qwen3.5-397B-A17B": "#2ca02c",
    "Sigurt Qwen3.5-Think": "#d62728",
    "Spark gpt-oss-20B": "#9467bd",
}

MODEL_SHORT = {
    "RTX5090 Qwen3-32B-AWQ": "Qwen3-32B\n(vLLM/RTX5090)",
    "Sigurt GPT-4o-mini": "GPT-4o-mini\n(Cloud)",
    "Sigurt Qwen3.5-397B-A17B": "Qwen3.5-397B\n(Cloud)",
    "Sigurt Qwen3.5-Think": "Qwen3.5-Think\n(Cloud)",
    "Spark gpt-oss-20B": "gpt-oss-20B\n(Ollama)",
}


def color_for(model: str) -> str:
    return PALETTE.get(model, "#333333")


def short_name(model: str) -> str:
    return MODEL_SHORT.get(model, model)


# ── plot helpers ──────────────────────────────────────────────────────────

def agg_by(records, group_keys, metric, agg="mean"):
    """Aggregate metric by group keys. Returns {(group): value}."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        if r.get(metric) is None:
            continue
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(r[metric])
    out = {}
    for key, vals in buckets.items():
        arr = np.array(vals)
        if agg == "mean":
            out[key] = arr.mean()
        elif agg == "median":
            out[key] = np.median(arr)
        elif agg == "std":
            out[key] = arr.std()
    return out


# ── PLOT 1: Throughput (tokens/s) by concurrency, per model ─────────────

def plot_throughput_vs_concurrency(records):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(set(r["model_label"] for r in records))
    n_vals = sorted(set(r["n_calls"] for r in records))

    for model in models:
        means = agg_by(records, ["model_label", "n_calls"], "gen_tok_s", "mean")
        stds = agg_by(records, ["model_label", "n_calls"], "gen_tok_s", "std")
        xs, ys, es = [], [], []
        for n in n_vals:
            key = (model, n)
            if key in means:
                xs.append(n)
                ys.append(means[key])
                es.append(stds.get(key, 0))
        ax.errorbar(xs, ys, yerr=es, marker="o", label=short_name(model),
                    color=color_for(model), linewidth=2, capsize=4)

    ax.set_xlabel("Concurrent Requests (N)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency (all max_tokens averaged)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '01_throughput_vs_concurrency.png'}")


# ── PLOT 2: Latency vs max_tokens, per model (N=1) ──────────────────────

def plot_latency_vs_tokens(records):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(set(r["model_label"] for r in records))
    t_vals = sorted(set(r["max_tokens"] for r in records))

    for model in models:
        subset = [r for r in records if r["model_label"] == model and r["n_calls"] == 1 and r["mode"] == "sequential"]
        means = agg_by(subset, ["model_label", "max_tokens"], "latency_s", "mean")
        xs, ys = [], []
        for t in t_vals:
            key = (model, t)
            if key in means:
                xs.append(t)
                ys.append(means[key])
        ax.plot(xs, ys, marker="s", label=short_name(model),
                color=color_for(model), linewidth=2)

    ax.set_xlabel("Max Output Tokens")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs Output Length (N=1, sequential)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_latency_vs_tokens.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '02_latency_vs_tokens.png'}")


# ── PLOT 3: TTFT comparison bar chart (N=1, T=1024) ─────────────────────

def plot_ttft_bar(records):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = [r for r in records if r["n_calls"] == 1 and r["max_tokens"] == 1024 and r["mode"] == "sequential"]
    models = sorted(set(r["model_label"] for r in subset))
    means = agg_by(subset, ["model_label"], "ttft_s", "mean")

    vals = [means.get((m,), 0) for m in models]
    colors = [color_for(m) for m in models]
    labels = [short_name(m) for m in models]

    bars = ax.barh(range(len(models)), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Time to First Token (seconds)")
    ax.set_title("TTFT Comparison (N=1, max_tokens=1024)")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_ttft_comparison.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '03_ttft_comparison.png'}")


# ── PLOT 4: Throughput bar chart (N=1, T=1024) ───────────────────────────

def plot_throughput_bar(records):
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = [r for r in records if r["n_calls"] == 1 and r["max_tokens"] == 1024 and r["mode"] == "sequential"]
    models = sorted(set(r["model_label"] for r in subset))
    means = agg_by(subset, ["model_label"], "gen_tok_s", "mean")

    vals = [means.get((m,), 0) for m in models]
    colors = [color_for(m) for m in models]
    labels = [short_name(m) for m in models]

    bars = ax.barh(range(len(models)), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_title("Throughput Comparison (N=1, max_tokens=1024)")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} tok/s", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_throughput_bar.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '04_throughput_bar.png'}")


# ── PLOT 5: Heatmap — throughput (model × concurrency) ───────────────────

def plot_throughput_heatmap(records):
    models = sorted(set(r["model_label"] for r in records))
    n_vals = sorted(set(r["n_calls"] for r in records))

    means = agg_by(records, ["model_label", "n_calls"], "gen_tok_s", "mean")
    matrix = np.array([[means.get((m, n), 0) for n in n_vals] for m in models])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels([f"N={n}" for n in n_vals])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m).replace("\n", " ") for m in models], fontsize=9)

    for i in range(len(models)):
        for j in range(len(n_vals)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    ax.set_title("Throughput Heatmap (tokens/s, averaged over max_tokens)")
    fig.colorbar(im, ax=ax, label="tokens/s")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_throughput_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '05_throughput_heatmap.png'}")


# ── PLOT 6: ITL p95 vs concurrency ───────────────────────────────────────

def plot_itl_p95(records):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(set(r["model_label"] for r in records))
    n_vals = sorted(set(r["n_calls"] for r in records))

    for model in models:
        means = agg_by(records, ["model_label", "n_calls"], "itl_p95_s", "mean")
        xs, ys = [], []
        for n in n_vals:
            key = (model, n)
            if key in means:
                xs.append(n)
                ys.append(means[key] * 1000)  # ms
        ax.plot(xs, ys, marker="D", label=short_name(model),
                color=color_for(model), linewidth=2)

    ax.set_xlabel("Concurrent Requests (N)")
    ax.set_ylabel("ITL p95 (ms)")
    ax.set_title("Inter-Token Latency p95 vs Concurrency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_itl_p95_vs_concurrency.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '06_itl_p95_vs_concurrency.png'}")


# ── PLOT 7: Radar chart — multi-metric comparison (N=1, T=1024) ──────────

def plot_radar(records):
    from matplotlib.patches import FancyBboxPatch

    subset = [r for r in records if r["n_calls"] == 1 and r["max_tokens"] == 1024 and r["mode"] == "sequential"]
    models = sorted(set(r["model_label"] for r in subset))

    metrics = {
        "Throughput\n(tok/s)": ("gen_tok_s", True),     # higher is better
        "TTFT\n(s)": ("ttft_s", False),                  # lower is better
        "Latency\n(s)": ("latency_s", False),             # lower is better
        "TPOT\n(s)": ("tpot_s", False),                   # lower is better
        "ITL p95\n(ms)": ("itl_p95_s", False),            # lower is better
    }

    # Normalize each metric to 0–1
    raw = {}
    for mname, (key, higher_better) in metrics.items():
        vals = {}
        for r in subset:
            if r.get(key) is not None:
                vals.setdefault(r["model_label"], []).append(r[key])
        avg = {m: np.mean(v) for m, v in vals.items()}
        all_vals = list(avg.values())
        vmin, vmax = min(all_vals), max(all_vals)
        for model, val in avg.items():
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            if not higher_better:
                norm = 1 - norm
            raw.setdefault(model, {})[mname] = norm

    categories = list(metrics.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for model in models:
        if model not in raw:
            continue
        values = [raw[model].get(c, 0) for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=short_name(model).replace("\n", " "),
                color=color_for(model))
        ax.fill(angles, values, alpha=0.1, color=color_for(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title("Multi-Metric Radar (N=1, max_tokens=1024)\n1.0 = best", y=1.08, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_radar_comparison.png", dpi=150)
    plt.close(fig)
    print(f"✓ {OUT_DIR / '07_radar_comparison.png'}")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading checkpoint data...")
    records = load_all()
    print(f"  {len(records)} result records from {len(set(r['model_label'] for r in records))} models")

    plot_throughput_vs_concurrency(records)
    plot_latency_vs_tokens(records)
    plot_ttft_bar(records)
    plot_throughput_bar(records)
    plot_throughput_heatmap(records)
    plot_itl_p95(records)
    plot_radar(records)

    print(f"\nDone. All plots saved to {OUT_DIR}/")
