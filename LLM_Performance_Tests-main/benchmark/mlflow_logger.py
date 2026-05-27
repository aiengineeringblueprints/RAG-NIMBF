"""
MLflow-Integration — bildet die Terminal-Ausgabe 1:1 ab.

Struktur:
  Experiment (= Gesamtübersicht, Runs-Tabelle mit Metriken als Spalten)
   └─ Run pro Parameterkombi "3 Prompts | 512 Tokens"
       ├─ Metriken: pro Modell wall_s, latency_mean etc. (Spalten in Runs-Tabelle)
       ├─ Tabelle "vergleich": Übersichtstabelle seq vs par (wie Terminal)
       ├─ Tabelle "details":  Einzel-Latenzen pro Modell+Modus+Prompt (wie Terminal)
       └─ Tabelle "antworten": Frage-Antwort-Paare pro Call
"""

import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from .types import CallResult, ModelDef, ModelStats


def _tags_for(models: list[ModelDef] | None, label: str) -> dict:
    """Holt die Tags eines Modells anhand des Labels."""
    if not models:
        return {}
    for m in models:
        if m.label == label:
            return dict(m.tags or {})
    return {}


def _hw_for(models: list[ModelDef] | None, label: str) -> str:
    return _tags_for(models, label).get("hardware", "")

_DEFAULT_DB = f"sqlite:///{Path(__file__).parent.parent / 'mlflow.db'}"


# ═══════════════════════════════════════════════════════════════════════════════
# Öffentliche API
# ═══════════════════════════════════════════════════════════════════════════════

def setup_tracing(tracking_uri: str, experiment: str) -> str:
    """Initialisiert MLflow und legt für jeden Programmstart ein eigenes
    Experiment mit Zeitstempel an. Gibt den tatsächlich verwendeten
    Experiment-Namen zurück, damit `log_to_mlflow_trends` denselben Namen
    wiederverwenden kann.
    """
    effective_uri = tracking_uri or _DEFAULT_DB
    os.environ["MLFLOW_TRACKING_URI"] = effective_uri
    mlflow.set_tracking_uri(effective_uri)

    # Pro Programmstart ein eigenes Experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_name = f"{experiment} · {timestamp}"
    mlflow.set_experiment(full_name)

    # System Metrics deaktiviert — der Benchmark läuft auf dem Client,
    # die LLM-Inferenz aber remote auf dem Spark/RTX5090-Server.
    # Lokale CPU/GPU-Metriken des Clients sind nicht aussagekräftig.

    try:
        mlflow.openai.autolog()
    except Exception:
        pass
    return full_name


def log_experiment_incremental(
    cd: dict[str, Any],
    n_calls: int,
    max_tokens: int,
    experiment: str,
    tracking_uri: str = "",
    models: list[ModelDef] | None = None,
) -> None:
    """Loggt ein einzelnes Experiment (eine (n_calls, max_tokens)-Kombination)
    sofort in MLflow. Dient als Absicherung gegen Daten-Verlust bei Crashes:
    Wenn dein Benchmark bei Modell 5 von 7 hängen bleibt, hast du die vorigen
    Experimente bereits sicher in MLflow.
    """
    effective_uri = tracking_uri or _DEFAULT_DB
    os.environ["MLFLOW_TRACKING_URI"] = effective_uri
    mlflow.set_tracking_uri(effective_uri)
    mlflow.set_experiment(experiment)
    _log_run(cd, n_calls, max_tokens, models=models)


def log_to_mlflow_trends(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
    experiment: str,
    tracking_uri: str = "",
    prompts: list[str] | None = None,
    models: list[ModelDef] | None = None,
    skip_per_run_logging: bool = False,
) -> None:
    """Loggt alle Experimente, Trends und Vergleichstabellen.

    Wenn `skip_per_run_logging=True`, werden die Einzel-Experiment-Runs nicht
    neu angelegt (weil sie schon inkrementell via `log_experiment_incremental`
    gespeichert wurden). Trend-Runs und Übersichtstabellen werden aber immer
    neu generiert, damit sie den Gesamtstand abbilden.
    """
    effective_uri = tracking_uri or _DEFAULT_DB
    os.environ["MLFLOW_TRACKING_URI"] = effective_uri
    mlflow.set_tracking_uri(effective_uri)
    # Wir nutzen das von setup_tracing() bereits angelegte (und mit Timestamp
    # versehene) Experiment und hängen keinen weiteren Timestamp an.
    mlflow.set_experiment(experiment)

    # Ein Run pro Parameterkombination (nur wenn nicht schon inkrementell geloggt)
    if not skip_per_run_logging:
        for max_tokens in max_tokens_list:
            for n_calls in call_counts:
                key = (n_calls, max_tokens)
                if key not in data_by_key:
                    continue
                _log_run(data_by_key[key], n_calls, max_tokens, models=models)

    # Trend-Runs (bei mehreren call_counts → automatische MLflow-Linien)
    if len(call_counts) >= 2:
        _log_trend_runs(data_by_key, call_counts, max_tokens_list, models=models)

    # Vergleichstabellen
    if len(data_by_key) > 1:
        _log_speedup_table(data_by_key, call_counts, max_tokens_list)
        _log_model_comparison_table(data_by_key, call_counts, max_tokens_list,
                                    models=models)
        # Master-Modellvergleich (RAG-Sicht über alle Konfigurationen)
        _log_master_rag_table(data_by_key, call_counts, max_tokens_list, models)

    print(f"\n✓ MLflow-Daten in Experiment '{experiment}' gespeichert.")
    print(f"  UI starten mit:  mlflow ui --backend-store-uri \"{effective_uri}\"")


# ═══════════════════════════════════════════════════════════════════════════════
# Ein Run pro Parameterkombination
# ═══════════════════════════════════════════════════════════════════════════════

def _log_run(cd: dict, n_calls: int, max_tokens: int,
             models: list[ModelDef] | None = None) -> None:
    stats: list[ModelStats] = cd["stats"]
    results: list[CallResult] = cd["results"]
    prompts_used: list[str] = cd.get("prompts_used", [])

    # ── 1. Übersichts-Run (alle Modelle in einem Run, wie bisher) ─────────
    run_name = f"{n_calls} Prompts | {max_tokens} Tokens"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"n_calls": n_calls, "max_tokens": max_tokens})

        # Tags aus allen genutzten Modellen aggregieren (z.B. mehrere Hardwares)
        if models:
            hws = sorted({(m.tags or {}).get("hardware", "")
                          for m in models if m.tags})
            hws = [h for h in hws if h]
            if hws:
                mlflow.set_tag("hardwares", ", ".join(hws))

        # ── Metriken (erscheinen als Spalten in der Experiment-Runs-Tabelle) ──
        _log_metrics(stats)

        # ── Tabelle 0: Modellvergleich (RAG-Sicht) ───────────────────────────
        _log_rag_comparison_table(stats, models)

        # ── Tabelle 1: Übersicht seq vs par (= Terminal "ÜBERSICHT") ──────────
        _log_comparison_table(stats, models)

        # ── Tabelle 2: Einzel-Latenzen (= Terminal "DETAILANSICHT") ───────────
        _log_detail_table(stats, results, prompts_used)

        # ── Tabelle 3: Frage-Antwort-Paare (Drill-Down) ──────────────────────
        _log_responses_table(results, prompts_used)

        # ── Chart: Vergleichsplot ─────────────────────────────────────────────
        _log_comparison_chart(stats, n_calls, max_tokens)

    # ── 2. Pro-Modell-Runs (einfache Metriken, ideal für Compare-View) ────
    # Jedes Modell bekommt einen eigenen Run mit kurzen Metrik-Namen.
    # So kann man in der MLflow-UI einfach 2+ Runs anklicken → Compare
    # und bekommt direkt "tok_s", "ttft_ms", "tpot_ms" nebeneinander.
    by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
    for s in stats:
        by_model[s.model][s.mode] = s

    for model_label, modes in sorted(by_model.items()):
        model_tags = _tags_for(models, model_label)
        seq = modes.get("sequential")
        par = modes.get("parallel")

        with mlflow.start_run(
            run_name=f"{model_label} · N={n_calls} T={max_tokens}"
        ):
            mlflow.log_params({
                "n_calls": n_calls,
                "max_tokens": max_tokens,
                "model": model_label,
            })
            mlflow.set_tags({
                "type": "model_run",
                **model_tags,
            })

            # Kurze, vergleichbare Metrik-Namen
            m: dict[str, float] = {}
            if seq:
                m["seq_tok_s"]     = round(seq.tps, 1) if seq.tps else 0
                m["seq_wall_s"]    = round(seq.wall_s, 2)
                m["seq_mean_s"]    = round(seq.mean_s, 3)
                m["seq_p95_s"]     = round(seq.p95_s, 3)
                m["seq_success"]   = seq.success_rate
                if seq.ttft_mean_s is not None:
                    m["seq_ttft_ms"]  = round(seq.ttft_mean_s * 1000, 1)
                if seq.tpot_mean_s is not None:
                    m["seq_tpot_ms"]  = round(seq.tpot_mean_s * 1000, 2)
                if seq.gen_tps_mean is not None:
                    m["seq_gen_tok_s"] = round(seq.gen_tps_mean, 1)
            if par:
                m["par_tok_s"]     = round(par.tps, 1) if par.tps else 0
                m["par_wall_s"]    = round(par.wall_s, 2)
                m["par_mean_s"]    = round(par.mean_s, 3)
                m["par_p95_s"]     = round(par.p95_s, 3)
                m["par_success"]   = par.success_rate
                if par.ttft_mean_s is not None:
                    m["par_ttft_ms"]  = round(par.ttft_mean_s * 1000, 1)
                if par.tpot_mean_s is not None:
                    m["par_tpot_ms"]  = round(par.tpot_mean_s * 1000, 2)
                if par.gen_tps_mean is not None:
                    m["par_gen_tok_s"] = round(par.gen_tps_mean, 1)
                if seq and seq.wall_s and par.wall_s:
                    m["speedup"] = round(seq.wall_s / par.wall_s, 2)

            mlflow.log_metrics(m)


def _log_metrics(stats: list[ModelStats]) -> None:
    """Pro Modell×Modus Metriken loggen → sichtbar als Spalten in der Runs-Tabelle."""
    metrics: dict[str, float] = {}
    for s in stats:
        pfx = _safe(s.model, s.mode)
        metrics[f"{pfx}_wall_s"]      = round(s.wall_s, 2)
        metrics[f"{pfx}_mean_s"]      = round(s.mean_s, 3)
        metrics[f"{pfx}_p95_s"]       = round(s.p95_s, 3)
        metrics[f"{pfx}_success"]     = s.success_rate
        if s.tps is not None:
            metrics[f"{pfx}_tok_s"]   = round(s.tps, 1)

        # Streaming-Metriken
        if s.ttft_mean_s is not None:
            metrics[f"{pfx}_ttft_ms"] = round(s.ttft_mean_s * 1000, 1)
        if s.ttft_p95_s is not None:
            metrics[f"{pfx}_ttft_p95_ms"] = round(s.ttft_p95_s * 1000, 1)
        if s.tpot_mean_s is not None:
            metrics[f"{pfx}_tpot_ms"] = round(s.tpot_mean_s * 1000, 2)
        if s.gen_tps_mean is not None:
            metrics[f"{pfx}_gen_tok_s"] = round(s.gen_tps_mean, 1)

        # Speedup
        if s.mode == "parallel":
            seq = next((x for x in stats
                        if x.model == s.model and x.mode == "sequential"), None)
            if seq and seq.wall_s and s.wall_s:
                metrics[f"{pfx}_speedup"] = round(seq.wall_s / s.wall_s, 2)

    mlflow.log_metrics(metrics)


def _log_rag_comparison_table(stats: list[ModelStats],
                              models: list[ModelDef] | None) -> None:
    """
    Tabelle 'modellvergleich_rag' — Modell-zentrierte RAG-Sicht mit Hardware.
    """
    by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
    for s in stats:
        by_model[s.model][s.mode] = s

    fastest_par = min(
        (s.wall_s for s in stats if s.mode == "parallel" and s.wall_s),
        default=0,
    )

    rows: list[dict] = []
    for model in sorted(by_model):
        par = by_model[model].get("parallel")
        seq = by_model[model].get("sequential")
        ref = par or seq
        if not ref:
            continue
        tags = _tags_for(models, model)
        if par and par.wall_s and fastest_par:
            f = par.wall_s / fastest_par
            factor = "Ref" if f <= 1.05 else f"{f:.2f}x"
        else:
            factor = ""

        rows.append({
            "Modell":          model,
            "Hardware":        tags.get("hardware", ""),
            "Aktive Params":   tags.get("params_active", ""),
            "Total Params":    tags.get("params_total", ""),
            "Quantisierung":   tags.get("quantization", ""),
            "TTFT (ms) par":   round(par.ttft_mean_s * 1000, 1)
                                if par and par.ttft_mean_s else 0,
            "TPOT (ms) par":   round(par.tpot_mean_s * 1000, 2)
                                if par and par.tpot_mean_s else 0,
            "Gen tok/s par":   round(par.gen_tps_mean, 1)
                                if par and par.gen_tps_mean else 0,
            "TTFT (ms) seq":   round(seq.ttft_mean_s * 1000, 1)
                                if seq and seq.ttft_mean_s else 0,
            "Gen tok/s seq":   round(seq.gen_tps_mean, 1)
                                if seq and seq.gen_tps_mean else 0,
            "Wall par (s)":    round(par.wall_s, 1) if par else 0,
            "Erfolg par":      round(par.success_rate * 100, 0) if par else 0,
            "Faktor":          factor,
        })

    if rows:
        mlflow.log_table(_cols(rows), artifact_file="modellvergleich_rag.json")


def _log_comparison_table(stats: list[ModelStats],
                          models: list[ModelDef] | None = None) -> None:
    """
    Tabelle 'vergleich' — 1:1 die Terminal-Übersichtstabelle.
    Klick auf Run → Artifacts → vergleich → nativ gerendert.
    """
    by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
    for s in stats:
        by_model[s.model][s.mode] = s

    rows: list[dict] = []
    for model in sorted(by_model):
        seq = by_model[model].get("sequential")
        par = by_model[model].get("parallel")
        speedup = ""
        if seq and par and seq.wall_s and par.wall_s:
            speedup = f"{seq.wall_s / par.wall_s:.1f}x"

        def _ms(v):
            return round(v * 1000, 1) if v is not None else 0

        rows.append({
            "Modell":         model,
            "Hardware":       _hw_for(models, model),
            "SEQ N":          seq.n if seq else 0,
            "SEQ Ø (s)":      round(seq.mean_s, 3) if seq else 0,
            "SEQ Med (s)":    round(seq.median_s, 3) if seq else 0,
            "SEQ P95 (s)":    round(seq.p95_s, 3) if seq else 0,
            "SEQ Wall (s)":   round(seq.wall_s, 1) if seq else 0,
            "SEQ TTFT (ms)":  _ms(seq.ttft_mean_s) if seq else 0,
            "SEQ TPOT (ms)":  _ms(seq.tpot_mean_s) if seq else 0,
            "SEQ Gen tok/s":  round(seq.gen_tps_mean, 1) if seq and seq.gen_tps_mean else 0,
            "PAR N":          par.n if par else 0,
            "PAR Ø (s)":      round(par.mean_s, 3) if par else 0,
            "PAR Med (s)":    round(par.median_s, 3) if par else 0,
            "PAR P95 (s)":    round(par.p95_s, 3) if par else 0,
            "PAR Wall (s)":   round(par.wall_s, 1) if par else 0,
            "PAR TTFT (ms)":  _ms(par.ttft_mean_s) if par else 0,
            "PAR TPOT (ms)":  _ms(par.tpot_mean_s) if par else 0,
            "PAR Gen tok/s":  round(par.gen_tps_mean, 1) if par and par.gen_tps_mean else 0,
            "Speedup":        speedup,
        })

    if rows:
        mlflow.log_table(_cols(rows), artifact_file="vergleich.json")


def _log_detail_table(
    stats: list[ModelStats],
    results: list[CallResult],
    prompts_used: list[str],
) -> None:
    """
    Tabelle 'details' — 1:1 die Terminal-Detailansicht.
    Zeigt pro Modell+Modus die Einzel-Latenzen und Min/Max/Ø.
    """
    rows: list[dict] = []
    n_prompts = max((r.prompt_idx for r in results), default=-1) + 1

    by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
    for s in stats:
        by_model[s.model][s.mode] = s

    for model in sorted(by_model):
        for mode in ("sequential", "parallel"):
            mode_results = [r for r in results
                            if r.model == model and r.mode == mode]
            if not mode_results:
                continue

            lats_ok = [r.latency_s for r in mode_results if r.success]
            short = "seq" if mode == "sequential" else "par"

            for p_idx in range(n_prompts):
                calls = [r for r in mode_results if r.prompt_idx == p_idx]
                for r in sorted(calls, key=lambda x: x.call_idx):
                    prompt_short = (prompts_used[r.prompt_idx][:80]
                                    if r.prompt_idx < len(prompts_used) else "?")
                    rows.append({
                        "Modell":     model,
                        "Modus":      short,
                        "Prompt #":   r.prompt_idx + 1,
                        "Prompt":     prompt_short,
                        "Latenz (s)": round(r.latency_s, 3),
                        "TTFT (ms)":  round(r.ttft_s * 1000, 1) if r.ttft_s else 0,
                        "TPOT (ms)":  round(r.tpot_s * 1000, 2) if r.tpot_s else 0,
                        "Gen tok/s":  round(r.gen_tok_s, 1) if r.gen_tok_s else 0,
                        "Erfolg":     "✓" if r.success else "✗",
                        "Tok in":     r.tokens_in or 0,
                        "Tok out":    r.tokens_out or 0,
                        "Fehler":     r.error or "",
                    })

            # Zusammenfassungszeile
            if lats_ok:
                ttfts = [r.ttft_s for r in mode_results if r.success and r.ttft_s]
                tpots = [r.tpot_s for r in mode_results if r.success and r.tpot_s]
                gens  = [r.gen_tok_s for r in mode_results if r.success and r.gen_tok_s]
                rows.append({
                    "Modell":     model,
                    "Modus":      f"── {short} Σ",
                    "Prompt #":   0,
                    "Prompt":     f"Min={min(lats_ok):.3f}s  Max={max(lats_ok):.3f}s  "
                                  f"({len(lats_ok)}/{len(mode_results)} ok)",
                    "Latenz (s)": round(sum(lats_ok) / len(lats_ok), 3),
                    "TTFT (ms)":  round(sum(ttfts) / len(ttfts) * 1000, 1) if ttfts else 0,
                    "TPOT (ms)":  round(sum(tpots) / len(tpots) * 1000, 2) if tpots else 0,
                    "Gen tok/s":  round(sum(gens) / len(gens), 1) if gens else 0,
                    "Erfolg":     f"{len(lats_ok)}/{len(mode_results)}",
                    "Tok in":     0,
                    "Tok out":    0,
                    "Fehler":     "",
                })

    if rows:
        mlflow.log_table(_cols(rows), artifact_file="details.json")


def _log_responses_table(
    results: list[CallResult],
    prompts_used: list[str],
) -> None:
    """
    Tabelle 'antworten' — Frage-Antwort-Paare mit Zeiten.
    Der tiefste Drill-Down-Level.
    """
    rows: list[dict] = []
    for r in sorted(results, key=lambda x: (x.model, x.mode, x.prompt_idx)):
        prompt = (prompts_used[r.prompt_idx]
                  if r.prompt_idx < len(prompts_used) else "?")
        short = "seq" if r.mode == "sequential" else "par"
        rows.append({
            "Modell":     r.model,
            "Modus":      short,
            "Prompt #":   r.prompt_idx + 1,
            "Frage":      prompt,
            "Antwort":    r.output_text or r.error or "–",
            "Latenz (s)": round(r.latency_s, 3),
            "Tok out":    r.tokens_out or 0,
            "Erfolg":     "✓" if r.success else "✗",
        })

    if rows:
        mlflow.log_table(_cols(rows), artifact_file="antworten.json")


def _log_comparison_chart(
    stats: list[ModelStats], n_calls: int, max_tokens: int
) -> None:
    """Einfacher Balkenplot: Wall-Zeit pro Modell, seq vs par."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
        for s in stats:
            by_model[s.model][s.mode] = s
        models = sorted(by_model)
        if not models:
            return

        seq_walls = [by_model[m].get("sequential")
                     for m in models]
        par_walls = [by_model[m].get("parallel")
                     for m in models]

        seq_vals = [s.wall_s if s else 0 for s in seq_walls]
        par_vals = [s.wall_s if s else 0 for s in par_walls]

        y = np.arange(len(models))
        h = 0.35

        fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.8 + 1)))
        bars1 = ax.barh(y - h / 2, seq_vals, h, label="sequential", color="#4C72B0")
        bars2 = ax.barh(y + h / 2, par_vals, h, label="parallel", color="#DD8452")

        for bars in [bars1, bars2]:
            for bar in bars:
                w = bar.get_width()
                if w > 0:
                    ax.text(w + max(seq_vals + par_vals) * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{w:.1f}s", va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel("Wall-Zeit (s)")
        ax.set_title(f"Wall-Zeit: {n_calls} Prompts | {max_tokens} Tokens")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        mlflow.log_figure(fig, "wall_zeit_vergleich.png")
        plt.close(fig)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Trend-Runs (bei mehreren CALL_COUNTS)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_trend_runs(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
    models: list[ModelDef] | None = None,
) -> None:
    """Ein Run pro Modell mit step=n_calls → MLflow zeichnet automatisch Linien.

    Loggt zusätzlich eine Tabellen-Artifact 'trend.json' mit allen Werten pro
    n_calls für direkte Inspektion in der MLflow-UI.
    """
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    for max_tokens in max_tokens_list:
        for model in sorted(all_models):
            run_name = (f"Trend · {model}"
                        if len(max_tokens_list) == 1
                        else f"Trend · {model} · T={max_tokens}")

            with mlflow.start_run(run_name=run_name):
                tags = {"type": "trend", "model": model}
                tags.update(_tags_for(models, model))
                mlflow.set_tags(tags)
                table_rows: list[dict] = []

                for n_calls in call_counts:
                    key = (n_calls, max_tokens)
                    if key not in data_by_key:
                        continue
                    sts = data_by_key[key]["stats"]
                    s_seq = next((s for s in sts
                                  if s.model == model and s.mode == "sequential"), None)
                    s_par = next((s for s in sts
                                  if s.model == model and s.mode == "parallel"), None)
                    m: dict[str, float] = {}
                    if s_seq:
                        m["seq_wall_s"]         = s_seq.wall_s
                        m["seq_latency_mean_s"] = s_seq.mean_s
                        if s_seq.ttft_mean_s is not None:
                            m["seq_ttft_ms"] = s_seq.ttft_mean_s * 1000
                        if s_seq.tpot_mean_s is not None:
                            m["seq_tpot_ms"] = s_seq.tpot_mean_s * 1000
                        if s_seq.gen_tps_mean is not None:
                            m["seq_gen_tok_s"] = s_seq.gen_tps_mean
                    if s_par:
                        m["par_wall_s"]         = s_par.wall_s
                        m["par_latency_mean_s"] = s_par.mean_s
                        if s_par.ttft_mean_s is not None:
                            m["par_ttft_ms"] = s_par.ttft_mean_s * 1000
                        if s_par.tpot_mean_s is not None:
                            m["par_tpot_ms"] = s_par.tpot_mean_s * 1000
                        if s_par.gen_tps_mean is not None:
                            m["par_gen_tok_s"] = s_par.gen_tps_mean
                        if s_seq and s_seq.wall_s and s_par.wall_s:
                            m["speedup"] = s_seq.wall_s / s_par.wall_s
                    if m:
                        mlflow.log_metrics(m, step=n_calls)

                    # Tabellen-Zeile für trend.json
                    def _r(v, nd=3):
                        return round(v, nd) if v is not None else 0

                    def _ms(v):
                        return round(v * 1000, 1) if v is not None else 0

                    table_rows.append({
                        "n_calls":           n_calls,
                        "max_tokens":        max_tokens,
                        "seq_wall_s":        _r(s_seq.wall_s, 1) if s_seq else 0,
                        "seq_mean_s":        _r(s_seq.mean_s) if s_seq else 0,
                        "seq_p95_s":         _r(s_seq.p95_s) if s_seq else 0,
                        "seq_ttft_ms":       _ms(s_seq.ttft_mean_s) if s_seq else 0,
                        "seq_tpot_ms":       _ms(s_seq.tpot_mean_s) if s_seq else 0,
                        "seq_gen_tok_s":     _r(s_seq.gen_tps_mean, 1) if s_seq else 0,
                        "par_wall_s":        _r(s_par.wall_s, 1) if s_par else 0,
                        "par_mean_s":        _r(s_par.mean_s) if s_par else 0,
                        "par_p95_s":         _r(s_par.p95_s) if s_par else 0,
                        "par_ttft_ms":       _ms(s_par.ttft_mean_s) if s_par else 0,
                        "par_tpot_ms":       _ms(s_par.tpot_mean_s) if s_par else 0,
                        "par_gen_tok_s":     _r(s_par.gen_tps_mean, 1) if s_par else 0,
                        "speedup":           _r(s_seq.wall_s / s_par.wall_s, 2)
                                              if (s_seq and s_par and s_seq.wall_s and s_par.wall_s)
                                              else 0,
                    })

                if table_rows:
                    mlflow.log_table(_cols(table_rows), artifact_file="trend.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(model: str, mode: str) -> str:
    """Sanitized Metrik-Prefix: 'Qwen35_spark_seq'."""
    import re
    s = re.sub(r"[^\w]", "_", model).strip("_")
    return f"{s}_{'seq' if mode == 'sequential' else 'par'}"


def _log_speedup_table(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
) -> None:
    """MLflow-Tabelle: Speedup parallel vs. sequential pro Modell."""
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    rows: list[dict] = []
    for model in sorted(all_models):
        for n_calls, max_tokens in experiments:
            key = (n_calls, max_tokens)
            if key not in data_by_key:
                continue
            stats = data_by_key[key]["stats"]
            par = next((s for s in stats if s.model == model
                        and s.mode == "parallel"), None)
            seq = next((s for s in stats if s.model == model
                        and s.mode == "sequential"), None)
            speedup = ""
            if par and seq and par.wall_s and seq.wall_s:
                speedup = f"{seq.wall_s / par.wall_s:.1f}×"
            rows.append({
                "Modell": model,
                "Konfig": f"N={n_calls}/T={max_tokens}",
                "Wall parallel (s)": round(par.wall_s, 1) if par else 0,
                "Wall sequential (s)": round(seq.wall_s, 1) if seq else 0,
                "Speedup": speedup,
            })

    if rows:
        with mlflow.start_run(run_name="Speedup: Parallel vs. Sequential"):
            mlflow.set_tag("type", "speedup")
            mlflow.log_table(_cols(rows), artifact_file="speedup.json")


def _log_model_comparison_table(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
    models: list[ModelDef] | None = None,
) -> None:
    """MLflow-Tabelle: Modellvergleich relativ zum schnellsten Modell."""
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    fastest: dict[tuple[int, int], float] = {}
    for n_calls, max_tokens in experiments:
        key = (n_calls, max_tokens)
        if key not in data_by_key:
            continue
        par_walls = [s.wall_s for s in data_by_key[key]["stats"]
                     if s.mode == "parallel" and s.wall_s]
        if par_walls:
            fastest[key] = min(par_walls)

    rows: list[dict] = []
    for model in sorted(all_models):
        for n_calls, max_tokens in experiments:
            key = (n_calls, max_tokens)
            if key not in data_by_key:
                continue
            par = next((s for s in data_by_key[key]["stats"]
                        if s.model == model and s.mode == "parallel"), None)
            if par and par.wall_s:
                ref = fastest.get(key)
                if ref and ref > 0:
                    factor = par.wall_s / ref
                    factor_str = "Ref" if factor <= 1.05 else f"{factor:.1f}×"
                else:
                    factor_str = "–"
                rows.append({
                    "Modell": model,
                    "Hardware": _hw_for(models, model),
                    "Konfig": f"N={n_calls}/T={max_tokens}",
                    "Wall parallel (s)": round(par.wall_s, 1),
                    "Faktor": factor_str,
                })

    if rows:
        with mlflow.start_run(run_name="Modellvergleich: Relative Performance"):
            mlflow.set_tag("type", "modellvergleich")
            mlflow.log_table(_cols(rows), artifact_file="modellvergleich.json")


def _log_master_rag_table(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
    models: list[ModelDef] | None,
) -> None:
    """
    Master-Tabelle 'modellvergleich_rag_master.json' — Modell × Konfig × Hardware
    mit allen RAG-relevanten Metriken in einem einzigen Top-Level-Run.
    """
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    rows: list[dict] = []
    for model in sorted(all_models):
        tags = _tags_for(models, model)
        for n_calls, max_tokens in experiments:
            key = (n_calls, max_tokens)
            if key not in data_by_key:
                continue
            sts = data_by_key[key]["stats"]
            par = next((s for s in sts if s.model == model
                        and s.mode == "parallel"), None)
            seq = next((s for s in sts if s.model == model
                        and s.mode == "sequential"), None)
            ref = par or seq
            if not ref:
                continue
            rows.append({
                "Modell":         model,
                "Hardware":       tags.get("hardware", ""),
                "Aktive P":       tags.get("params_active", ""),
                "Total P":        tags.get("params_total", ""),
                "Quantisierung":  tags.get("quantization", ""),
                "Konfig":         f"N={n_calls}/T={max_tokens}",
                "TTFT par (ms)":  round(par.ttft_mean_s * 1000, 1)
                                   if par and par.ttft_mean_s else 0,
                "TPOT par (ms)":  round(par.tpot_mean_s * 1000, 2)
                                   if par and par.tpot_mean_s else 0,
                "Gen tok/s par":  round(par.gen_tps_mean, 1)
                                   if par and par.gen_tps_mean else 0,
                "TTFT seq (ms)":  round(seq.ttft_mean_s * 1000, 1)
                                   if seq and seq.ttft_mean_s else 0,
                "Gen tok/s seq": round(seq.gen_tps_mean, 1)
                                   if seq and seq.gen_tps_mean else 0,
                "Wall par (s)":   round(par.wall_s, 1) if par else 0,
                "Wall seq (s)":   round(seq.wall_s, 1) if seq else 0,
                "Erfolg par (%)": round(par.success_rate * 100, 0) if par else 0,
            })

    if rows:
        with mlflow.start_run(run_name="Modellvergleich · Master (RAG)"):
            mlflow.set_tag("type", "rag_master")
            mlflow.log_table(_cols(rows), artifact_file="modellvergleich_rag_master.json")


def _cols(rows: list[dict]) -> dict[str, list]:
    """list-of-dicts → dict-of-lists (für mlflow.log_table)."""
    if not rows:
        return {}
    return {k: [r[k] for r in rows] for k in rows[0]}
