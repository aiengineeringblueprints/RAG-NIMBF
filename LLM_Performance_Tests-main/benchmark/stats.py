"""Statistik-Berechnung über alle CallResults."""

from collections import defaultdict
from statistics import mean, median, stdev

from .types import CallResult, ModelStats


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    sv = sorted(values)
    return sv[min(int(len(sv) * 0.95), len(sv) - 1)]


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def compute_stats(
    results: list[CallResult],
    wall_times: dict[tuple[str, str], float] | None = None,
) -> list[ModelStats]:
    """Gruppiert Ergebnisse nach (Modell, Modus) und berechnet Kennzahlen."""
    groups: dict[tuple[str, str], list[CallResult]] = defaultdict(list)
    for r in results:
        groups[(r.model, r.mode)].append(r)

    stats: list[ModelStats] = []
    for (model, mode), rs in sorted(groups.items()):
        lats   = [r.latency_s for r in rs if r.success]
        n_ok   = len(lats)
        n_total = len(rs)

        wall = (wall_times or {}).get((model, mode), 0.0)

        if not lats:
            stats.append(ModelStats(
                model=model, mode=mode, n=n_total,
                mean_s=0.0, median_s=0.0, p95_s=0.0, stdev_s=0.0,
                success_rate=0.0, tps=None, wall_s=wall,
            ))
            continue

        lats_sorted = sorted(lats)
        p95_idx = min(int(len(lats_sorted) * 0.95), len(lats_sorted) - 1)

        out_toks = [r.tokens_out for r in rs if r.success and r.tokens_out]
        avg_out  = mean(out_toks) if out_toks else None
        avg_lat  = mean(lats)
        tps      = (avg_out / avg_lat) if (avg_out and avg_lat) else None

        # Streaming-Metriken: nur erfolgreiche Calls mit verfügbaren Werten
        ttft_vals = [r.ttft_s for r in rs if r.success and r.ttft_s is not None]
        tpot_vals = [r.tpot_s for r in rs if r.success and r.tpot_s is not None]
        gen_tps_vals = [r.gen_tok_s for r in rs
                        if r.success and r.gen_tok_s is not None]

        stats.append(ModelStats(
            model=model, mode=mode, n=n_total,
            mean_s=mean(lats),
            median_s=median(lats),
            p95_s=lats_sorted[p95_idx],
            stdev_s=stdev(lats) if len(lats) > 1 else 0.0,
            success_rate=n_ok / n_total,
            tps=tps,
            wall_s=wall,
            ttft_mean_s=_mean_or_none(ttft_vals),
            ttft_p95_s=_p95(ttft_vals),
            tpot_mean_s=_mean_or_none(tpot_vals),
            gen_tps_mean=_mean_or_none(gen_tps_vals),
        ))

    return stats
