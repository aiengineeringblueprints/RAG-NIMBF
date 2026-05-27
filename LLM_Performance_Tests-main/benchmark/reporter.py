"""Konsolenausgabe der Benchmark-Ergebnisse."""

from collections import defaultdict
from typing import Any

from .types import CallResult, ModelDef, ModelStats


def _model_tag(models: list[ModelDef] | None, label: str, key: str,
               default: str = "–") -> str:
    """Liest einen Tag-Wert aus der ModelDef-Liste über das Label."""
    if not models:
        return default
    for m in models:
        if m.label == label and m.tags:
            return str(m.tags.get(key, default))
    return default


def _fmt(val: float | None, unit: str = "s") -> str:
    if val is None:
        return "–"
    return f"{val:.3f}{unit}"


def _fmt_ms(val: float | None) -> str:
    if val is None:
        return "–"
    return f"{val * 1000:.0f}ms"


def _fmt_tps(val: float | None) -> str:
    if val is None:
        return "–"
    return f"{val:.1f}"


def print_summary(stats: list[ModelStats],
                  all_results: list[CallResult] | None = None,
                  models: list[ModelDef] | None = None) -> None:
    """Gibt eine formatierte Auswertung aus:
    0. Modellvergleich (RAG-Sicht): TTFT/TPOT/Gen Tok/s pro Modell+Hardware
    1. Übersichtstabelle: pro Modell sequential vs. parallel nebeneinander
    2. Streaming-Metriken: TTFT, TPOT, Generierungs-Durchsatz
    3. Detailansicht: Einzel-Call-Latenzen pro Modell und Prompt
    """
    # ── Daten organisieren ────────────────────────────────────────────────────
    by_model: dict[str, dict[str, ModelStats]] = defaultdict(dict)
    for s in stats:
        by_model[s.model][s.mode] = s

    results_by_key: dict[tuple[str, str, int], list[CallResult]] = defaultdict(list)
    if all_results:
        for r in all_results:
            results_by_key[(r.model, r.mode, r.prompt_idx)].append(r)

    # ── 0. Modellvergleich (RAG-Sicht) ────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  MODELLVERGLEICH (RAG-Sicht): pro Modell + Hardware")
    print("═" * 90)

    # Schnellstes Modell pro Modus für Faktor-Berechnung
    fastest_par = min(
        (s.wall_s for s in stats if s.mode == "parallel" and s.wall_s),
        default=0,
    )

    try:
        from tabulate import tabulate

        rows = []
        for model in sorted(by_model):
            par = by_model[model].get("parallel")
            seq = by_model[model].get("sequential")
            if not par and not seq:
                continue
            ref = par or seq
            hw = _model_tag(models, model, "hardware", "–")
            active = _model_tag(models, model, "params_active", "–")

            # Faktor: Wall-Zeit parallel relativ zum schnellsten Modell
            if par and par.wall_s and fastest_par:
                f = par.wall_s / fastest_par
                factor = "Ref" if f <= 1.05 else f"{f:.1f}×"
            else:
                factor = "–"

            rows.append([
                model,
                hw,
                active,
                _fmt_ms(ref.ttft_mean_s),
                _fmt_ms(ref.tpot_mean_s),
                _fmt_tps(ref.gen_tps_mean),
                f"{par.wall_s:.1f}s" if par else "–",
                f"{par.success_rate * 100:.0f}%" if par else "–",
                factor,
            ])

        headers = [
            "Modell", "Hardware", "Aktive P",
            "TTFT Ø", "TPOT Ø", "Gen Tok/s",
            "Wall (par)", "Erfolg", "Faktor",
        ]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print("    Werte aus parallelen Calls (typische Multi-User-Last)")
        print("    Faktor: Wall-Zeit relativ zum schnellsten Modell — kleiner = besser")

    except ImportError:
        for model in sorted(by_model):
            par = by_model[model].get("parallel")
            ref = par or by_model[model].get("sequential")
            if not ref:
                continue
            hw = _model_tag(models, model, "hardware", "–")
            print(f"  {model} [{hw}]  TTFT={_fmt_ms(ref.ttft_mean_s)}  "
                  f"TPOT={_fmt_ms(ref.tpot_mean_s)}  "
                  f"Gen={_fmt_tps(ref.gen_tps_mean)}tok/s")

    # ── 1. Übersichtstabelle ──────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  ÜBERSICHT: Sequential vs. Parallel — pro Modell")
    print("═" * 90)

    try:
        from tabulate import tabulate

        rows = []
        for model, modes in sorted(by_model.items()):
            seq = modes.get("sequential")
            par = modes.get("parallel")

            speedup = "–"
            if seq and par and seq.wall_s and par.wall_s:
                speedup = f"{seq.wall_s / par.wall_s:.1f}×"

            rows.append([
                model,
                seq.n if seq else "–",
                _fmt(seq.mean_s) if seq else "–",
                _fmt(seq.median_s) if seq else "–",
                _fmt(seq.p95_s) if seq else "–",
                _fmt(seq.wall_s) if seq else "–",
                par.n if par else "–",
                _fmt(par.mean_s) if par else "–",
                _fmt(par.median_s) if par else "–",
                _fmt(par.p95_s) if par else "–",
                _fmt(par.wall_s) if par else "–",
                speedup,
            ])

        headers = [
            "Modell",
            "SEQ N", "SEQ Ø (s)", "SEQ Med (s)", "SEQ P95 (s)", "SEQ Wall (s)",
            "PAR N", "PAR Ø (s)", "PAR Med (s)", "PAR P95 (s)", "PAR Wall (s)",
            "Speedup",
        ]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    except ImportError:
        for model, modes in sorted(by_model.items()):
            seq = modes.get("sequential")
            par = modes.get("parallel")
            print(f"\n  {model}")
            if seq:
                print(f"    [sequential]  N={seq.n}  Ø={seq.mean_s:.3f}s  "
                      f"Med={seq.median_s:.3f}s  Wall={seq.wall_s:.1f}s")
            if par:
                print(f"    [parallel  ]  N={par.n}  Ø={par.mean_s:.3f}s  "
                      f"Med={par.median_s:.3f}s  Wall={par.wall_s:.1f}s")
            if seq and par and seq.wall_s and par.wall_s:
                print(f"    → Speedup: {seq.wall_s / par.wall_s:.1f}×")

    # ── 1b. Streaming-Metriken (TTFT/TPOT/Throughput) ────────────────────────
    print("\n" + "═" * 90)
    print("  STREAMING-METRIKEN: TTFT, TPOT, Generierungs-Durchsatz")
    print("═" * 90)

    try:
        from tabulate import tabulate

        rows = []
        for model, modes in sorted(by_model.items()):
            for mode_label, mode_key in (("seq", "sequential"), ("par", "parallel")):
                s = modes.get(mode_key)
                if not s:
                    continue
                rows.append([
                    model,
                    mode_label,
                    _fmt_ms(s.ttft_mean_s),
                    _fmt_ms(s.ttft_p95_s),
                    _fmt_ms(s.tpot_mean_s),
                    _fmt_tps(s.gen_tps_mean),
                    _fmt_tps(s.tps),
                ])

        headers = [
            "Modell", "Modus",
            "TTFT Ø", "TTFT P95",
            "TPOT Ø",
            "Gen Tok/s", "Gesamt Tok/s",
        ]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print("    TTFT  = Time To First Token (kleiner = responsiver)")
        print("    TPOT  = Time Per Output Token (kleiner = schneller streaming)")
        print("    Gen Tok/s = reine Generierung ohne TTFT  ·  Gesamt Tok/s = inkl. TTFT")

    except ImportError:
        for model, modes in sorted(by_model.items()):
            for mode_label, mode_key in (("seq", "sequential"), ("par", "parallel")):
                s = modes.get(mode_key)
                if not s:
                    continue
                print(f"  {model} [{mode_label}]  "
                      f"TTFT={_fmt_ms(s.ttft_mean_s)}  "
                      f"TPOT={_fmt_ms(s.tpot_mean_s)}  "
                      f"Gen={_fmt_tps(s.gen_tps_mean)}tok/s")

    # ── 2. Detailansicht: Einzel-Latenzen ────────────────────────────────────
    if all_results:
        print("\n" + "═" * 90)
        print("  DETAILANSICHT: Einzel-Call-Latenzen")
        print("═" * 90)

        # Alle Prompts sammeln (Prompt-Index → Text nicht gespeichert, nur Index)
        all_prompt_idxs: set[int] = {r.prompt_idx for r in all_results}
        n_prompts = max(all_prompt_idxs) + 1 if all_prompt_idxs else 0

        for model in sorted(by_model.keys()):
            for mode in ("sequential", "parallel"):
                key_prefix = (model, mode)
                mode_results = [r for r in all_results
                                if r.model == model and r.mode == mode]
                if not mode_results:
                    continue

                lats_ok = [r.latency_s for r in mode_results if r.success]
                label = f"  {model}  [{mode}]"
                print(f"\n{label}")
                print("  " + "─" * (len(label) - 2))

                for p_idx in range(n_prompts):
                    calls = sorted(
                        [r for r in mode_results if r.prompt_idx == p_idx],
                        key=lambda r: r.call_idx,
                    )
                    if not calls:
                        continue
                    lats_str = "  ".join(
                        f"{r.latency_s:.3f}s" if r.success else "✗"
                        for r in calls
                    )
                    print(f"    Prompt {p_idx + 1}:  {lats_str}")

                if lats_ok:
                    avg = sum(lats_ok) / len(lats_ok)
                    print(f"    → Min: {min(lats_ok):.3f}s  "
                          f"Max: {max(lats_ok):.3f}s  "
                          f"Ø: {avg:.3f}s  "
                          f"({len(lats_ok)}/{len(mode_results)} erfolgreich)")

                # Speedup-Hinweis bei parallel
                if mode == "parallel":
                    seq_stats = by_model[model].get("sequential")
                    par_stats = by_model[model].get("parallel")
                    if seq_stats and par_stats and seq_stats.wall_s and par_stats.wall_s:
                        speedup = seq_stats.wall_s / par_stats.wall_s
                        saved = seq_stats.wall_s - par_stats.wall_s
                        print(f"    → Wall-Zeit: {par_stats.wall_s:.2f}s  "
                              f"(vs. {seq_stats.wall_s:.2f}s sequentiell  "
                              f"→ {speedup:.1f}× schneller, {saved:.1f}s gespart)")

    # ── Fehler-Zusammenfassung ────────────────────────────────────────────────
    failed = [s for s in stats if s.success_rate < 1.0]
    if failed:
        print("\n  Fehlgeschlagene Calls:")
        for s in failed:
            n_failed = s.n - int(s.n * s.success_rate)
            print(f"    {s.model} [{s.mode}]: {n_failed}/{s.n} Calls fehlgeschlagen")

    print()


def print_trend_summary(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
) -> None:
    """Trend-Tabelle: Wall-Zeit + Ø-Latenz für alle Hyperparameter-Kombinationen."""

    # Alle (model, mode) Kombinationen
    all_mm: set[tuple[str, str]] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_mm.add((s.model, s.mode))

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    print("\n" + "═" * 90)
    print("  TREND: Vergleich über alle Experimente")
    print("═" * 90)

    try:
        from tabulate import tabulate

        rows = []
        for (model, mode) in sorted(all_mm):
            row: list[str] = [model, mode]
            for (n_calls, max_tokens) in experiments:
                key = (n_calls, max_tokens)
                stat = next(
                    (s for s in data_by_key[key]["stats"]
                     if s.model == model and s.mode == mode),
                    None,
                ) if key in data_by_key else None
                if stat:
                    row.extend([f"{stat.wall_s:.1f}", f"{stat.mean_s:.3f}"])
                else:
                    row.extend(["–", "–"])
            rows.append(row)

        exp_headers: list[str] = []
        for (n_calls, max_tokens) in experiments:
            lbl = f"N={n_calls}/T={max_tokens}"
            exp_headers.extend([f"{lbl} Wall(s)", f"{lbl} Ø(s)"])
        headers = ["Modell", "Modus"] + exp_headers

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    except ImportError:
        for (model, mode) in sorted(all_mm):
            parts = [f"  {model} [{mode}]"]
            for (n_calls, max_tokens) in experiments:
                key = (n_calls, max_tokens)
                stat = next(
                    (s for s in data_by_key[key]["stats"]
                     if s.model == model and s.mode == mode),
                    None,
                ) if key in data_by_key else None
                if stat:
                    parts.append(f"N={n_calls}/T={max_tokens}: "
                                 f"wall={stat.wall_s:.1f}s ø={stat.mean_s:.3f}s")
            print("  ".join(parts))

    print()


def print_speedup_table(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
) -> None:
    """Speedup-Tabelle: pro Modell Wall parallel vs. sequential + Speedup."""
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    print("\n" + "═" * 90)
    print("  SPEEDUP: Parallel vs. Sequential")
    print("═" * 90)

    try:
        from tabulate import tabulate

        rows = []
        for model in sorted(all_models):
            row: list[str] = [model]
            for n_calls, max_tokens in experiments:
                key = (n_calls, max_tokens)
                if key not in data_by_key:
                    row.extend(["–", "–", "–"])
                    continue
                stats = data_by_key[key]["stats"]
                par = next((s for s in stats if s.model == model
                            and s.mode == "parallel"), None)
                seq = next((s for s in stats if s.model == model
                            and s.mode == "sequential"), None)
                row.append(f"{par.wall_s:.1f}" if par else "–")
                row.append(f"{seq.wall_s:.1f}" if seq else "–")
                if par and seq and par.wall_s and seq.wall_s:
                    row.append(f"{seq.wall_s / par.wall_s:.1f}×")
                else:
                    row.append("–")
            rows.append(row)

        headers = ["Modell"]
        for n_calls, max_tokens in experiments:
            lbl = f"N={n_calls}/T={max_tokens}"
            headers.extend([f"{lbl} par", f"{lbl} seq", f"{lbl} Speedup"])

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    except ImportError:
        for model in sorted(all_models):
            print(f"\n  {model}")
            for n_calls, max_tokens in experiments:
                key = (n_calls, max_tokens)
                if key not in data_by_key:
                    continue
                stats = data_by_key[key]["stats"]
                par = next((s for s in stats if s.model == model
                            and s.mode == "parallel"), None)
                seq = next((s for s in stats if s.model == model
                            and s.mode == "sequential"), None)
                if par and seq and par.wall_s and seq.wall_s:
                    print(f"    N={n_calls}/T={max_tokens}: "
                          f"par={par.wall_s:.1f}s  seq={seq.wall_s:.1f}s  "
                          f"→ {seq.wall_s / par.wall_s:.1f}×")

    print()


def print_model_comparison_table(
    data_by_key: dict[tuple[int, int], dict[str, Any]],
    call_counts: list[int],
    max_tokens_list: list[int],
) -> None:
    """Modellvergleich: Wall-Zeit (parallel) relativ zum schnellsten Modell."""
    all_models: set[str] = set()
    for cd in data_by_key.values():
        for s in cd["stats"]:
            all_models.add(s.model)

    experiments = [(n, mt) for mt in max_tokens_list for n in call_counts]

    # Schnellstes Modell (parallel) pro Experiment
    fastest: dict[tuple[int, int], float] = {}
    for n_calls, max_tokens in experiments:
        key = (n_calls, max_tokens)
        if key not in data_by_key:
            continue
        par_walls = [s.wall_s for s in data_by_key[key]["stats"]
                     if s.mode == "parallel" and s.wall_s]
        if par_walls:
            fastest[key] = min(par_walls)

    print("\n" + "═" * 90)
    print("  MODELLVERGLEICH: Relative Performance (parallel)")
    print("═" * 90)

    try:
        from tabulate import tabulate

        rows = []
        for model in sorted(all_models):
            row: list[str] = [model]
            for n_calls, max_tokens in experiments:
                key = (n_calls, max_tokens)
                if key not in data_by_key:
                    row.extend(["–", "–"])
                    continue
                par = next((s for s in data_by_key[key]["stats"]
                            if s.model == model and s.mode == "parallel"),
                           None)
                if par and par.wall_s:
                    row.append(f"{par.wall_s:.1f}")
                    ref = fastest.get(key)
                    if ref and ref > 0:
                        factor = par.wall_s / ref
                        row.append("Ref" if factor <= 1.05 else f"{factor:.1f}×")
                    else:
                        row.append("–")
                else:
                    row.extend(["–", "–"])
            rows.append(row)

        headers = ["Modell"]
        for n_calls, max_tokens in experiments:
            lbl = f"N={n_calls}/T={max_tokens}"
            headers.extend([f"{lbl} Wall(s)", f"{lbl} Faktor"])

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    except ImportError:
        for model in sorted(all_models):
            print(f"\n  {model}")
            for n_calls, max_tokens in experiments:
                key = (n_calls, max_tokens)
                if key not in data_by_key:
                    continue
                par = next((s for s in data_by_key[key]["stats"]
                            if s.model == model and s.mode == "parallel"),
                           None)
                if par and par.wall_s:
                    ref = fastest.get(key)
                    if ref and ref > 0:
                        factor = par.wall_s / ref
                        fstr = "Ref" if factor <= 1.05 else f"{factor:.1f}×"
                    else:
                        fstr = "–"
                    print(f"    N={n_calls}/T={max_tokens}: "
                          f"wall={par.wall_s:.1f}s  {fstr}")

    print()
