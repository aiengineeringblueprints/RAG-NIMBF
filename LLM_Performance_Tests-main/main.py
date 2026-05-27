"""
LLM Performance Benchmark – Einstiegspunkt
==========================================
Konfiguration über .env (siehe .env.example).

Starten:
  python main.py

Robustheit:
- Pro Modell wird ein Checkpoint in benchmark_checkpoints/ geschrieben (JSON)
- Nach jedem Experiment wird sofort in MLflow geloggt
- Bei Crashes bleiben die bisherigen Daten erhalten
"""

import asyncio
import dataclasses
import json
import os
import random as _random
import sys
import traceback
from datetime import datetime
from pathlib import Path


# ── Tee-Logger: schreibt alles in Terminal + Logfile mit Timestamps ──────────

class _TeeWriter:
    """Wraps stdout/stderr to write to both the terminal and a logfile."""

    def __init__(self, original, logfile, prefix: str = ""):
        self._original = original
        self._logfile = logfile
        self._prefix = prefix
        self._at_line_start = True

    def write(self, text: str) -> int:
        # Zum Terminal wie bisher
        self._original.write(text)

        # Ins Logfile mit Timestamp am Zeilenanfang
        if text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if self._at_line_start and line and not line.startswith("\r"):
                    ts = datetime.now().strftime("%H:%M:%S")
                    self._logfile.write(f"[{ts}] ")
                # \r-Zeilen (Live-Monitor) nicht ins Logfile schreiben
                if not line.startswith("\r"):
                    self._logfile.write(line)
                    if i < len(lines) - 1:
                        self._logfile.write("\n")
                self._at_line_start = (i < len(lines) - 1) or text.endswith("\n")
            self._logfile.flush()

        return len(text)

    def flush(self) -> None:
        self._original.flush()
        self._logfile.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    # Proxy für alle anderen Attribute (encoding, etc.)
    def __getattr__(self, name: str):
        return getattr(self._original, name)


_LOG_DIR = Path(__file__).parent / "benchmark_logs"


def _setup_logging() -> Path | None:
    """Richtet das Logfile ein und leitet stdout/stderr um."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logpath = _LOG_DIR / f"benchmark_{ts}.log"
        logfile = open(logpath, "w", encoding="utf-8")  # noqa: SIM115

        # Header
        logfile.write(f"LLM Benchmark Log — {datetime.now().isoformat()}\n")
        logfile.write(f"{'═' * 65}\n\n")
        logfile.flush()

        sys.stdout = _TeeWriter(sys.__stdout__, logfile)
        sys.stderr = _TeeWriter(sys.__stderr__, logfile, prefix="ERR ")

        print(f"  Logfile:          {logpath}")
        return logpath
    except Exception as exc:
        print(f"  [WARNUNG] Logfile konnte nicht erstellt werden: {exc}")
        return None

from benchmark.config import load_config
from benchmark.mlflow_logger import (
    log_experiment_incremental,
    log_to_mlflow_trends,
    setup_tracing,
)
from benchmark.reporter import (
    print_model_comparison_table,
    print_speedup_table,
    print_summary,
    print_trend_summary,
)
from benchmark.remote_monitor import RemoteMonitor
from benchmark.runner import run_parallel, run_sequential, warmup_model
from benchmark.stats import compute_stats
from benchmark.types import CallResult


_CHECKPOINT_DIR = Path(__file__).parent / "benchmark_checkpoints"


def _safe_slug(text: str) -> str:
    """Macht einen Text für Dateinamen sicher."""
    import re
    return re.sub(r"[^\w\-.]", "_", text)[:80]


def _save_model_checkpoint(
    run_id: str,
    n_calls: int,
    max_tokens: int,
    model_label: str,
    results: list[CallResult],
    seq_wall: float,
    par_wall: float,
) -> None:
    """Schreibt nach jedem Modell einen Disk-Checkpoint.

    Damit hast du auch bei einem harten Prozess-Crash (Segfault, Kernel-OOM,
    Ctrl-C) die bis dahin gesammelten Daten auf der Platte.
    """
    try:
        ckpt_dir = _CHECKPOINT_DIR / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fname = f"N{n_calls}_T{max_tokens}__{_safe_slug(model_label)}.json"
        path = ckpt_dir / fname
        data = {
            "timestamp":   datetime.now().isoformat(),
            "n_calls":     n_calls,
            "max_tokens":  max_tokens,
            "model_label": model_label,
            "seq_wall_s":  seq_wall,
            "par_wall_s":  par_wall,
            "results":     [dataclasses.asdict(r) for r in results],
        }
        path.write_text(json.dumps(data, indent=2, default=str),
                        encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  [Checkpoint] Schreiben fehlgeschlagen: {exc}")


async def main() -> None:
    logpath = _setup_logging()

    cfg = load_config()

    # Pro Programmstart ein neues MLflow-Experiment (mit Timestamp im Namen)
    experiment_name = setup_tracing(
        cfg["mlflow_tracking_uri"], cfg["mlflow_experiment"]
    )

    # Eindeutige Run-ID für Disk-Checkpoints (nicht MLflow-Run-ID)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    models           = cfg["models"]
    prompt_pool      = cfg["prompts"]
    call_counts      = cfg["call_counts"]
    max_tokens_list  = cfg["max_tokens_list"]
    prompt_selection = cfg["prompt_selection"]
    timeout_s        = cfg["timeout_s"]

    print("═" * 65)
    print("  LLM PERFORMANCE BENCHMARK")
    print("═" * 65)
    print(f"  Modelle:          {len(models)}")
    for m in models:
        url = f"  →  {m.base_url}" if m.base_url else ""
        print(f"    • {m.label:<30} ({m.provider}){url}")
    print(f"  Prompt-Pool:      {len(prompt_pool)} Prompts")
    print(f"  Prompt-Auswahl:   {prompt_selection}")
    print(f"  Call-Counts:      {call_counts}  (= Anzahl Prompts pro Experiment)")
    print(f"  Max-Tokens:       {max_tokens_list}")
    print(f"  Timeout:          {timeout_s}s")
    total = len(call_counts) * len(max_tokens_list)
    print(f"  Experimente:      {total} Kombinationen ({len(call_counts)} × {len(max_tokens_list)})")
    print(f"  Checkpoints:      {_CHECKPOINT_DIR / run_id}")
    print(f"  MLflow Experiment: {experiment_name}")
    if logpath:
        print(f"  Logfile:          {logpath}")
    print("═" * 65)

    # Remote Monitoring via SSH — mehrere Server parallel
    # Steuerbar über .env:
    #   REMOTE_MONITOR_ENABLED=true/false   (Default: true)
    #   REMOTE_MONITOR_INTERVAL=5           (Default: 5 Sekunden)
    #
    # Server werden automatisch aus den Modell-URLs extrahiert (jeder
    # Host nur einmal). SSH-User pro Host konfigurierbar über:
    #   REMOTE_MONITOR_HOSTS=host1:user1,host2:user2
    # Oder als Fallback: REMOTE_MONITOR_USER=nico (für alle Hosts)
    monitors: list[RemoteMonitor] = []
    monitor_enabled = os.getenv("REMOTE_MONITOR_ENABLED", "true").strip().lower()
    if monitor_enabled not in ("false", "0", "no", "off"):
        import re as _re
        interval = float(os.getenv("REMOTE_MONITOR_INTERVAL", "5"))
        default_user = os.getenv("REMOTE_MONITOR_USER", "nico")

        # Explizite Host-Liste aus .env (host:user,host:user,...)
        hosts_env = os.getenv("REMOTE_MONITOR_HOSTS", "").strip()
        host_user_map: dict[str, str] = {}
        if hosts_env:
            for entry in hosts_env.split(","):
                entry = entry.strip()
                if ":" in entry:
                    h, u = entry.split(":", 1)
                    host_user_map[h.strip()] = u.strip()
                else:
                    host_user_map[entry] = default_user

        # Auto-detect: alle einzigartigen Remote-Hosts aus den Modell-URLs
        for m in models:
            if not m.base_url:
                continue
            host_match = _re.search(r"https?://([^/:]+)", m.base_url)
            if host_match:
                host = host_match.group(1)
                if host not in ("localhost", "127.0.0.1"):
                    host_user_map.setdefault(host, default_user)

        # Einen Monitor pro Host starten (kein MLflow-Run hier —
        # wird am Ende als separater Top-Level-Run geloggt)
        for host, user in sorted(host_user_map.items()):
            mon = RemoteMonitor(
                host=host, user=user, interval_s=interval,
            )
            if mon.start():
                monitors.append(mon)

    # data_by_key[(n_calls, max_tokens)] = {results, wall_times, stats, prompts_used}
    data_by_key: dict[tuple[int, int], dict] = {}

    try:
        for max_tokens in max_tokens_list:
            for n_calls in call_counts:
                key = (n_calls, max_tokens)
                n = min(n_calls, len(prompt_pool))

                # Prompts auswählen — keine Wiederholungen, kein Cache-Hit
                if prompt_selection == "random":
                    selected = _random.sample(prompt_pool, n)
                else:
                    selected = prompt_pool[:n]

                label = f"{n} Prompts  |  {max_tokens} Max-Tokens"
                print(f"\n{'═' * 65}")
                print(f"  EXPERIMENT: {label}")
                print(f"{'═' * 65}")

                results: list[CallResult] = []
                wall_times: dict[tuple[str, str], float] = {}

                for model_def in models:
                    try:
                        # Peak-Tracking zurücksetzen vor jedem Modell
                        for mon in monitors:
                            mon.reset_peaks()
                            mon.mark_event(model_def.label, "start")

                        # Warmup: einmal pro Modell laden, damit die erste
                        # Messung nicht durch Model-Loading-Overhead verfälscht
                        # wird. Deaktivierbar per WARMUP_ENABLED=false in .env
                        warmup_model(model_def, selected[0], cfg)

                        # n_calls=1 → jeder Prompt wird genau einmal aufgerufen
                        for mon in monitors:
                            mon.mark_event(f"{model_def.label} sequential", "start")
                        seq_res, seq_wall = run_sequential(
                            model_def, selected, 1, max_tokens, cfg)
                        for mon in monitors:
                            mon.mark_event(f"{model_def.label} sequential", "end")
                        wall_times[(model_def.label, "sequential")] = seq_wall
                        results.extend(seq_res)

                        for mon in monitors:
                            mon.mark_event(f"{model_def.label} parallel", "start")
                        par_res, par_wall = await run_parallel(
                            model_def, selected, 1, max_tokens, cfg)
                        for mon in monitors:
                            mon.mark_event(f"{model_def.label} parallel", "end")
                        wall_times[(model_def.label, "parallel")] = par_wall
                        results.extend(par_res)

                        for mon in monitors:
                            mon.mark_event(model_def.label, "end")

                        # Peak-Werte für dieses Modell ausgeben
                        for mon in monitors:
                            peaks = mon.get_peaks()
                            if peaks:
                                print(f"\n  [Peak {mon.host}] {model_def.label}:"
                                      f"  GPU={peaks.get('gpu_util_pct', 0):.0f}%"
                                      f"  Mem={peaks.get('mem_used_gb', 0):.1f}GB"
                                      f"  Temp={peaks.get('gpu_temp_c', 0):.0f}°C"
                                      f"  Power={peaks.get('gpu_power_w', 0):.0f}W")

                        # Disk-Checkpoint NACH jedem Modell: selbst bei
                        # hartem Crash ist der aktuelle Stand gesichert.
                        _save_model_checkpoint(
                            run_id=run_id,
                            n_calls=n_calls,
                            max_tokens=max_tokens,
                            model_label=model_def.label,
                            results=seq_res + par_res,
                            seq_wall=seq_wall,
                            par_wall=par_wall,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"\n  [ERROR] Modell '{model_def.label}' "
                              f"übersprungen wegen Exception: {exc}")
                        traceback.print_exc()
                        continue

                stats = compute_stats(results, wall_times)
                print_summary(stats, results, models=models)

                data_by_key[key] = {
                    "results":      results,
                    "wall_times":   wall_times,
                    "stats":        stats,
                    "prompts_used": selected,
                }

                # ── Inkrementelles MLflow-Logging: nach jedem Experiment ──
                # Wenn der Prozess danach crasht, sind alle bis hier
                # gesammelten Experimente bereits sicher in MLflow.
                try:
                    log_experiment_incremental(
                        data_by_key[key],
                        n_calls=n_calls,
                        max_tokens=max_tokens,
                        experiment=experiment_name,
                        tracking_uri=cfg["mlflow_tracking_uri"],
                        models=models,
                    )
                    print(f"\n  ✓ Experiment '{label}' in MLflow gespeichert "
                          f"(inkrementell).")
                except Exception as exc:  # noqa: BLE001
                    print(f"\n  [WARNUNG] Inkrementelles MLflow-Logging "
                          f"fehlgeschlagen: {exc}")

    except KeyboardInterrupt:
        print("\n\n[ABGEBROCHEN] Durch Ctrl-C. Bisher gesammelte Daten "
              "werden noch in MLflow-Trend-Runs konsolidiert.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n\n[UNERWARTETER FEHLER] {exc}")
        traceback.print_exc()
        print("Bisher gesammelte Daten werden noch in MLflow-Trend-Runs "
              "konsolidiert.")
    finally:
        for mon in monitors:
            mon.stop()
            saved = mon.save_csv(Path(__file__).parent / "resource_traces", run_id)
            if saved:
                print(f"  [Monitor] CSV trace: {saved[0]}")

    # Remote-Monitor-Daten als eigene Top-Level-Runs loggen (nach allem)
    for mon in monitors:
        mon.log_to_mlflow()

    # Trend-Übersicht über alle Experimente (auch bei partiellen Daten)
    if len(data_by_key) > 1:
        print_trend_summary(data_by_key, call_counts, max_tokens_list)
        print_speedup_table(data_by_key, call_counts, max_tokens_list)
        print_model_comparison_table(data_by_key, call_counts, max_tokens_list)

    # Trends und Vergleichstabellen in MLflow loggen
    # (skip_per_run_logging=True, weil die Einzel-Experimente schon
    # inkrementell geloggt wurden)
    if data_by_key:
        try:
            log_to_mlflow_trends(
                data_by_key,
                call_counts,
                max_tokens_list,
                experiment_name,
                cfg["mlflow_tracking_uri"],
                prompts=prompt_pool,
                models=models,
                skip_per_run_logging=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n[WARNUNG] Trend-Runs-Logging fehlgeschlagen: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
