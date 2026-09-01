"""Real-time dashboard for the RAG benchmarking framework.

Usage:
    streamlit run dashboard/app.py
    # optionally: DASHBOARD_RESULTS_DIR=/path/to/results streamlit run dashboard/app.py

Auto-refresh keeps polling the selected run directory while a worker is
writing configs, so a running sweep can be watched live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional dependency
    st_autorefresh = None

# Make the project root importable regardless of the working directory, so the
# app also works when started as `streamlit run dashboard/app.py` from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import loader  # noqa: E402
from dashboard.visuals import (  # noqa: E402
    llm_performance_bar,
    metric_comparison_bar,
    metric_correlation_heatmap,
    per_sample_box,
    resource_trace_plot,
    scatter_quality_vs_perf,
    stage_timing_bar,
)

st.set_page_config(
    page_title="RAG Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)

# Resolve the default results directory against the project root so it works
# regardless of the working directory the app is launched from.
DEFAULT_RESULTS_DIR = os.environ.get(
    "DASHBOARD_RESULTS_DIR", str(PROJECT_ROOT / "results")
)


def _int_or_na(value) -> str | None:
    """Return the value as a plain int or None (Arrow-safe for mixed columns)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nullable_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame whose numeric columns are nullable (Arrow-compatible)."""
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ("Run", "Status", "Zeitstempel", "Dataset", "Subset"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------
# Sidebar: global controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 RAG Benchmark")
    results_dir = st.text_input(
        "Ergebnis-Verzeichnis",
        value=DEFAULT_RESULTS_DIR,
        help="Verzeichnis mit den runN/ Ordnern (Standard: results/)",
    )
    auto_refresh = st.checkbox("Auto-Refresh (live)", value=True)
    refresh_seconds = st.slider(
        "Refresh-Intervall (s)", min_value=2, max_value=120, value=10
    )
    refresh_clicked = st.button("Jetzt aktualisieren", width="stretch")

    if auto_refresh and st_autorefresh is None:
        st.warning(
            "Auto-Refresh braucht `streamlit-autorefresh`. "
            "`pip install streamlit-autorefresh` oder manuell aktualisieren."
        )
    st.caption("Quellen: results/runN/benchmark_*.json, progress.json, configs/*_qa.json")

    if auto_refresh and st_autorefresh is not None:
        st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")


def _need_refresh() -> bool:
    if refresh_clicked:
        return True
    if auto_refresh and st_autorefresh is not None:
        return False
    return False


@st.cache_data(ttl=10, show_spinner=False)
def _scan(results_dir: str):
    return loader.scan_runs(Path(results_dir))


def _load_run_fresh(run_dir: Path):
    return loader.load_run(run_dir)


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
results_path = Path(results_dir)
runs = _scan(results_dir)
run_by_name = {r.run_name: r for r in runs}

st.sidebar.divider()
st.sidebar.markdown(
    f"**Runs gefunden:** {len(runs)} "
    f"({sum(1 for r in runs if r.status == 'completed')} abgeschlossen, "
    f"{sum(1 for r in runs if r.status == 'running')} laufen)"
)

if not runs:
    st.info(f"Keine Run-Verzeichnisse unter `{results_dir}` gefunden.")
    st.stop()

tab_overview, tab_live, tab_compare, tab_detail = st.tabs(
    ["Übersicht", "Live-Monitor", "Vergleichen", "Run-Detail"]
)

# --------------------------------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------------------------------
with tab_overview:
    st.header("Übersicht aller Runs")

    rows = []
    for r in runs:
        rows.append(
            {
                "Run": r.run_name,
                "Status": r.status,
                "Zeitstempel": r.timestamp,
                "Dataset": r.dataset_name or "-",
                "Subset": r.dataset_subset or "-",
                "Sample-Size": _int_or_na(r.dataset_sample_size),
                "Configs": _int_or_na(r.num_configs),
                "Fertig": _int_or_na(r.num_done),
                "Fehler": _int_or_na(r.num_failed),
            }
        )
    st.dataframe(_nullable_frame(rows), width="stretch", hide_index=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Runs gesamt", len(runs))
    col2.metric(
        "Abgeschlossen", sum(1 for r in runs if r.status == "completed")
    )
    col3.metric(
        "Aktiv (laufend)",
        sum(1 for r in runs if r.status == "running"),
    )
    col4.metric(
        "Configs abgeschlossen",
        sum(r.num_done for r in runs),
    )

    with st.expander("System-Informationen"):
        for r in runs:
            if r.system_info:
                st.markdown(f"**{r.run_name}**")
                st.json(r.system_info)

# --------------------------------------------------------------------------
# Tab 2: Live monitor
# --------------------------------------------------------------------------
with tab_live:
    st.header("Live-Monitor")

    active = [r for r in runs if r.status == "running"]
    options = {r.run_name: r for r in (active + runs[:1])}
    if not active and runs:
        st.info("Aktuell läuft kein Benchmark. Wähle einen bestehenden Run zur Ansicht.")
    default = options[active[0].run_name] if active else runs[0]

    selected_live = st.selectbox(
        "Run auswählen",
        list(options),
        format_func=lambda n: options[n].run_name,
        index=0,
    )
    live_run = options[selected_live]

    if _need_refresh():
        live_data = _load_run_fresh(live_run.run_dir)
    else:
        live_data = loader.load_run(live_run.run_dir)

    info = live_data.info

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", info.status)
    c2.metric("Configs gesamt", info.num_configs)
    c3.metric("Fertig", info.num_done)
    c4.metric("Fehler", info.num_failed)

    if info.num_configs:
        progress_val = info.num_done / info.num_configs
        st.progress(progress_val, text=f"Fortschritt: {info.num_done}/{info.num_configs}")
    else:
        st.progress(0, text="Kein Fortschritt bekannt")

    # Live per-config table
    live_rows = []
    for lc in live_data.live_configs:
        live_rows.append(
            {
                "Config": lc.name,
                "Status": lc.status,
                "Gestartet": lc.started_at,
                "Fertig": lc.completed_at,
                "Fehler": (lc.error or "")[:80],
            }
        )
    if live_rows:
        st.subheader("Configs (progress.json)")
        st.dataframe(
            pd.DataFrame(live_rows), width="stretch", hide_index=True
        )

    # Live partial configs (synthesized from QA logs while still running)
    partial = [c for c in live_data.configs if c.get("_live")]
    if partial:
        st.subheader("Bereits fertige Configs (Live-Teilresultate)")
        st.dataframe(
            pd.DataFrame([loader.summarize_config(c) for c in partial]).drop(
                columns=["custom_metric_means"], errors="ignore"
            ),
            width="stretch",
            hide_index=True,
        )

    # Per-sample inspection of the most recently completed config
    if live_data.qa_logs:
        st.subheader("Per-Sample-Ansicht (QA-Logs)")
        qa_options = list(live_data.qa_logs)
        qa_choice = st.selectbox("Config", qa_options, key="live_qa_choice")
        samples = live_data.qa_logs[qa_choice]
        if samples:
            if len(samples) > 1:
                show_idx = st.slider(
                    "Sample", 0, len(samples) - 1, 0, key="live_sample_idx"
                )
            else:
                show_idx = 0
            s = samples[show_idx]
            st.markdown(f"**Frage:** {s.get('question')}")
            st.markdown(f"**Antwort:** {s.get('answer')}")
            st.markdown(f"**Ground Truth:** {s.get('ground_truth')}")
            meta = {
                k: v
                for k, v in s.items()
                if k
                not in (
                    "question",
                    "answer",
                    "ground_truth",
                    "contexts",
                    "retrieval_metadata",
                    "raw_content",
                    "raw_reasoning",
                )
            }
            st.caption(f"Latenz: {s.get('total_seconds')}s · TTFT: {s.get('ttft_seconds')}s · Tokens: {s.get('token_count')}")
            with st.expander("Details (JSON)"):
                st.json(meta)

# --------------------------------------------------------------------------
# Tab 3: Compare
# --------------------------------------------------------------------------
with tab_compare:
    st.header("Configs vergleichen")

    selected = st.multiselect(
        "Runs auswählen",
        [r.run_name for r in runs],
        default=[runs[0].run_name] if runs else [],
    )
    if not selected:
        st.info("Bitte mindestens einen Run auswählen.")
        st.stop()

    compare_configs: list[dict] = []
    origin: dict[str, str] = {}
    for run_name in selected:
        run = run_by_name[run_name]
        data = loader.load_run(run.run_dir)
        for cfg in data.configs:
            if cfg.get("_live"):
                continue
            compare_configs.append(cfg)
            origin[cfg["config_name"]] = run_name

    if not compare_configs:
        st.warning("Keine vollständigen Config-Ergebnisse in den gewählten Runs.")
        st.stop()

    metric_map = {cfg["config_name"]: loader.config_metrics(cfg) for cfg in compare_configs}

    group_options = {
        "Qualität (RAGAS)": [m for m in loader.QUALITY_METRICS if m.startswith("ragas_")],
        "Qualität (Custom/TRACe)": [
            m for m in loader.QUALITY_METRICS if m.startswith("custom_")
        ],
        "Performance": loader.PERFORMANCE_METRICS,
        "LLM-Performance": ["llm_perf_generation_sequential_latency_p95_s"],
    }

    metric_tabs = st.tabs(list(group_options))
    for tab, (group_label, group_metrics) in zip(metric_tabs, group_options.items()):
        with tab:
            valid = [m for m in group_metrics if any(metric_map[c].get(m) is not None for c in metric_map)]
            if not valid:
                st.caption("Keine Daten für diese Gruppe.")
                continue
            chosen = st.multiselect(
                "Metriken", valid, default=valid[:3], key=f"compare_{group_label}"
            )
            if not chosen:
                continue
            rows = []
            for cfg in compare_configs:
                row = {"config_name": cfg["config_name"], "run": origin[cfg["config_name"]]}
                for m in chosen:
                    row[m] = metric_map[cfg["config_name"]].get(m)
                rows.append(row)
            df = pd.DataFrame(rows)
            display_cols = ["config_name", "run"] + chosen
            st.dataframe(
                df[display_cols].round(4) if not df[display_cols].empty else df[display_cols],
                width="stretch",
                hide_index=True,
            )

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                for m in chosen[:3]:
                    st.plotly_chart(metric_comparison_bar(df, m), width="stretch")
            with plot_col2:
                for cfg in compare_configs:
                    if chosen and any(
                        loader.per_sample_metric_values(cfg, m) for m in chosen[:1]
                    ):
                        st.plotly_chart(
                            per_sample_box(df, chosen[0], compare_configs),
                            width="stretch",
                        )
                        break

    st.subheader("Korrelation")
    all_metrics = sorted({m for cfg in metric_map.values() for m in cfg})
    corr_metrics = st.multiselect(
        "Metriken für Korrelation",
        all_metrics,
        default=[
            m for m in all_metrics
            if m in ("ragas_faithfulness", "ragas_context_recall", "custom_context_relevance")
        ],
        key="corr_metrics",
    )
    if len(corr_metrics) >= 2:
        corr_df = pd.DataFrame(
            [
                {"config_name": cfg["config_name"], **metric_map[cfg["config_name"]]}
                for cfg in compare_configs
            ]
        )
        st.plotly_chart(metric_correlation_heatmap(corr_df, corr_metrics), width="stretch")

    st.subheader("Qualität vs. Performance")
    qm = st.selectbox("Qualitätsmetrik", loader.QUALITY_METRICS, key="scatter_q")
    pm = st.selectbox("Performance-Metrik", loader.PERFORMANCE_METRICS, key="scatter_p")
    scatter_df = pd.DataFrame(
        [
            {"config_name": cfg["config_name"], **metric_map[cfg["config_name"]]}
            for cfg in compare_configs
        ]
    )
    st.plotly_chart(
        scatter_quality_vs_perf(scatter_df, qm, pm), width="stretch"
    )

# --------------------------------------------------------------------------
# Tab 4: Run detail
# --------------------------------------------------------------------------
with tab_detail:
    st.header("Run-Detail")

    detail_run_name = st.selectbox("Run", [r.run_name for r in runs], key="detail_run")
    detail_run = run_by_name[detail_run_name]
    detail_data = loader.load_run(detail_run.run_dir)

    info = detail_data.info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", info.dataset_name or "-")
    c2.metric("Subset", info.dataset_subset or "-")
    c3.metric("Sample-Size", info.dataset_sample_size or "-")
    c4.metric("Configs", info.num_configs)

    if info.system_info:
        with st.expander("System-Informationen"):
            st.json(info.system_info)

    if not detail_data.configs:
        st.warning("Keine Config-Ergebnisse in diesem Run.")
        st.stop()

    config_names = [c.get("config_name") for c in detail_data.configs]
    cfg_choice = st.selectbox("Config", config_names, key="detail_config")
    cfg = next(c for c in detail_data.configs if c.get("config_name") == cfg_choice)

    metrics = loader.config_metrics(cfg)
    per_sample = cfg.get("per_sample") or []

    st.subheader("Metriken")
    quality_cols = [m for m in loader.QUALITY_METRICS if metrics.get(m) is not None]
    perf_cols = [m for m in loader.PERFORMANCE_METRICS if metrics.get(m) is not None]
    if quality_cols or perf_cols:
        col_q, col_p = st.columns(2)
        with col_q:
            st.markdown("**Qualität**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metrik": [loader.metric_label(m) for m in quality_cols],
                        "Wert": [round(metrics[m], 4) for m in quality_cols],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with col_p:
            st.markdown("**Performance**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metrik": [loader.metric_label(m) for m in perf_cols],
                        "Wert": [round(metrics[m], 4) for m in perf_cols],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    llm_perf = cfg.get("llm_performance_metrics")
    if llm_perf:
        st.plotly_chart(llm_performance_bar(llm_perf), width="stretch")

    stage_latency = cfg.get("stage_latency")
    if stage_latency:
        st.plotly_chart(stage_timing_bar(stage_latency), width="stretch")

    if per_sample:
        st.subheader("Per-Sample-Ergebnisse")
        sample_rows = []
        for i, s in enumerate(per_sample):
            ragas = s.get("ragas_scores") or {}
            custom = s.get("custom_scores") or {}
            sample_rows.append(
                {
                    "#": i,
                    "Frage": (s.get("question") or "")[:60],
                    "TTFT (s)": s.get("ttft_seconds"),
                    "Gesamt (s)": s.get("total_seconds"),
                    "Tokens": s.get("token_count") or s.get("output_tokens"),
                    "Valid": s.get("answer_valid"),
                    "Faithfulness": ragas.get("ragas_faithfulness"),
                    "Context Recall": ragas.get("ragas_context_recall"),
                    "Context Relevance": custom.get("context_relevance")
                    or custom.get("custom_context_relevance"),
                    "TRACe Util.": custom.get("trace_utilization")
                    or custom.get("custom_trace_utilization"),
                }
            )
        st.dataframe(pd.DataFrame(sample_rows), width="stretch", hide_index=True)

        sample_idx = st.slider("Sample-Detail", 0, len(per_sample) - 1, 0, key="detail_sample")
        s = per_sample[sample_idx]
        st.markdown(f"**Frage:** {s.get('question')}")
        st.markdown(f"**Antwort:** {s.get('answer')}")
        st.markdown(f"**Ground Truth:** {s.get('ground_truth')}")
        contexts = s.get("contexts") or []
        if contexts:
            with st.expander(f"Kontexte ({len(contexts)})"):
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**[{i}]** {ctx[:400]}")
        with st.expander("Rohe Scores (JSON)"):
            st.json({"ragas_scores": s.get("ragas_scores"), "custom_scores": s.get("custom_scores")})

    # Resource traces
    trace_dir = detail_run.run_dir / "resource_traces"
    if trace_dir.is_dir():
        traces = sorted(trace_dir.glob("*.csv"))
        if traces:
            st.subheader("Ressourcen-Traces")
            trace_choice = st.selectbox("Trace", [t.name for t in traces], key="trace_choice")
            st.plotly_chart(
                resource_trace_plot(trace_dir / trace_choice), width="stretch"
            )

    with st.expander("Roher Config-JSON"):
        st.json({k: v for k, v in cfg.items() if k != "per_sample"})