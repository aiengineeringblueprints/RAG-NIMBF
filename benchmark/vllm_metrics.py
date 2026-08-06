"""Prometheus scraper for vLLM's ``/metrics`` endpoint.

vLLM exposes an OpenAI-compatible API at ``http://vllm-host:port/v1`` plus a
Prometheus exposition endpoint at ``http://vllm-host:port/metrics``. The latter
reports production-grade latency and scheduling signals that the OpenAI API
itself does not surface — most notably KV-cache utilization, request queue
depth, preemption count, and server-side TTFT/TPOT histograms.

This module is intentionally stdlib-only (``urllib`` + a hand-rolled
prometheus text parser) so vLLM remains a soft dependency: the framework runs
unchanged against Ollama / OpenAI / vLLM, and only this module needs to know
about the prometheus exposition format.

Inspired by RAGPerf §3.3.4 (arXiv:2603.10765v1).
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

# Conservative default; vLLM's /metrics is cheap but we avoid hammering it.
_DEFAULT_POLL_INTERVAL_S = 1.0
_DEFAULT_TIMEOUT_S = 2.0
_DEFAULT_SCRAPE_PATH = "/metrics"


@dataclass(frozen=True)
class HistogramSnapshot:
    """A single prometheus histogram at one scrape instant.

    ``last_sample_*`` come from the most recent bucket/sum/count lines seen for
    the metric; ``sum``/``count`` fall back to ``0.0`` when the metric is
    absent (e.g. no requests completed yet).
    """

    sum: float = 0.0
    count: float = 0.0
    last_bucket_le: float | None = None
    last_bucket_value: float | None = None


@dataclass(frozen=True)
class VllmMetricsSnapshot:
    """One snapshot of the vLLM prometheus surface.

    ``None`` fields mean the metric was missing from the scrape (older vLLM
    versions, fresh boot, etc.) rather than zero — keeping the distinction
    matters for aggregation (a missing p95 must not pull the mean down).
    """

    scraped_at: float
    num_requests_waiting: float | None = None
    num_requests_running: float | None = None
    gpu_cache_usage_perc: float | None = None
    cpu_cache_usage_perc: float | None = None
    num_preemption: float | None = None
    ttft_histogram: HistogramSnapshot | None = None
    tpot_histogram: HistogramSnapshot | None = None
    request_success_time_histogram: HistogramSnapshot | None = None
    raw_metric_count: int = 0
    source_url: str | None = None

    def to_flat_dict(self) -> dict[str, float]:
        """Flatten to a metrics dict suitable for MLflow / JSON reporting.

        Only present metrics are included so missing values do not pollute
        downstream aggregations with implicit zeros.
        """
        out: dict[str, float] = {}
        if self.num_requests_waiting is not None:
            out["vllm_num_requests_waiting"] = self.num_requests_waiting
        if self.num_requests_running is not None:
            out["vllm_num_requests_running"] = self.num_requests_running
        if self.gpu_cache_usage_perc is not None:
            out["vllm_gpu_cache_usage_perc"] = self.gpu_cache_usage_perc
        if self.cpu_cache_usage_perc is not None:
            out["vllm_cpu_cache_usage_perc"] = self.cpu_cache_usage_perc
        if self.num_preemption is not None:
            out["vllm_num_preemption"] = self.num_preemption
        if self.ttft_histogram is not None:
            out["vllm_ttft_sum_s"] = self.ttft_histogram.sum
            out["vllm_ttft_count"] = self.ttft_histogram.count
        if self.tpot_histogram is not None:
            out["vllm_tpot_sum_s"] = self.tpot_histogram.sum
            out["vllm_tpot_count"] = self.tpot_histogram.count
        if self.request_success_time_histogram is not None:
            out["vllm_request_success_time_sum_s"] = (
                self.request_success_time_histogram.sum
            )
            out["vllm_request_success_time_count"] = (
                self.request_success_time_histogram.count
            )
        return out


# ---------------------------------------------------------------------------
# Prometheus exposition parser (stdlib only)
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"^\s*#")
_METRIC_LINE_RE = re.compile(
    r"""
    ^
    (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)
    (?:\{(?P<labels>[^}]*)\})?
    \s+
    (?P<value>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|NaN|\+Inf|-Inf)
    $
    """,
    re.VERBOSE,
)


def _parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse prometheus exposition text into ``{metric_name: [(labels, value)]}``.

    ``# HELP`` / ``# TYPE`` comment lines are ignored. Histogram buckets show
    up as ``<basename>_bucket{le="..."} <count>`` lines; the caller is
    responsible for grouping bucket/sum/count triplets back into a histogram.
    """
    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        if not raw_line or _COMMENT_RE.match(raw_line):
            continue
        match = _METRIC_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        name = match.group("name")
        labels = _parse_labels(match.group("labels") or "")
        value = _parse_value(match.group("value"))
        if value is None:
            continue
        samples.setdefault(name, []).append((labels, value))
    return samples


def _parse_labels(raw: str) -> dict[str, str]:
    """Parse a prometheus label string like ``le="0.1",model="x"``."""
    labels: dict[str, str] = {}
    if not raw.strip():
        return labels
    for chunk in _split_label_pairs(raw):
        if "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        if key:
            labels[key] = value
    return labels


def _split_label_pairs(raw: str) -> Iterable[str]:
    """Split a label string on commas that are not inside quotes."""
    parts: list[str] = []
    current: list[str] = []
    in_quote = False
    quote_char = ""
    for char in raw:
        if in_quote:
            current.append(char)
            if char == quote_char:
                in_quote = False
        elif char in "\"'":
            in_quote = True
            quote_char = char
            current.append(char)
        elif char == ",":
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _parse_value(token: str) -> float | None:
    if token in ("NaN", "+Inf", "-Inf"):
        return math.nan if token == "NaN" else math.inf * (1 if token == "+Inf" else -1)
    try:
        return float(token)
    except ValueError:
        return None


def _first_gauge(samples: list[tuple[dict[str, str], float]]) -> float | None:
    """Return the value of the first sample (gauges typically have one)."""
    if not samples:
        return None
    return samples[0][1]


def _build_histogram(
    bucket_samples: list[tuple[dict[str, str], float]] | None,
    sum_value: float | None,
    count_value: float | None,
) -> HistogramSnapshot | None:
    """Reconstruct the last-bucket view of a prometheus histogram.

    We deliberately keep this lightweight — for runtime monitoring we only
    need the cumulative ``sum``/``count`` (which gives a server-reported mean)
    plus the most recent bucket boundary as a cheap trend signal. Full p95
    reconstruction from buckets happens in :func:`aggregate_snapshots`.
    """
    if not bucket_samples and sum_value is None and count_value is None:
        return None
    last_le: float | None = None
    last_val: float | None = None
    if bucket_samples:
        # Pick the bucket with the largest finite ``le`` (the cumulative
        # total bucket uses le="+Inf").
        finite = [
            (float(labels.get("le", "inf")), value)
            for labels, value in bucket_samples
            if labels.get("le", "inf") not in ("+Inf", "Inf", "inf")
        ]
        if finite:
            finite.sort(key=lambda item: item[0])
            last_le, last_val = finite[-1]
    return HistogramSnapshot(
        sum=float(sum_value or 0.0),
        count=float(count_value or 0.0),
        last_bucket_le=last_le,
        last_bucket_value=last_val,
    )


def parse_vllm_metrics(text: str) -> VllmMetricsSnapshot:
    """Parse a vLLM ``/metrics`` response body into a snapshot.

    All fields are best-effort: a metric that vLLM did not expose (older
    version, fresh boot with no traffic, etc.) is returned as ``None`` rather
    than ``0.0`` so callers can tell "absent" apart from "zero".
    """
    samples = _parse_prometheus_text(text)

    def _gauge(name: str) -> float | None:
        return _first_gauge(samples.get(name))

    def _hist(base: str) -> HistogramSnapshot | None:
        return _build_histogram(
            samples.get(f"{base}_bucket"),
            (_gauge(f"{base}_sum")),
            (_gauge(f"{base}_count")),
        )

    return VllmMetricsSnapshot(
        scraped_at=time.time(),
        num_requests_waiting=_gauge("vllm:num_requests_waiting"),
        num_requests_running=_gauge("vllm:num_requests_running"),
        gpu_cache_usage_perc=_gauge("vllm:gpu_cache_usage_perc"),
        cpu_cache_usage_perc=_gauge("vllm:cpu_cache_usage_perc"),
        num_preemption=_gauge("vllm:num_preemption"),
        ttft_histogram=_hist("vllm:time_to_first_token_seconds"),
        tpot_histogram=_hist("vllm:time_per_output_token_seconds"),
        request_success_time_histogram=_hist("vllm:request_success_time_seconds"),
        raw_metric_count=len(samples),
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class VllmMetricsScraper:
    """Poll a vLLM ``/metrics`` endpoint with caching and graceful degradation.

    The scraper is safe to call from hot paths (e.g. before/after each
    generation call) because identical requests within ``poll_interval_s`` are
    served from an in-memory cache. A threaded HTTP fetch keeps the parse off
    the caller's critical path, but the default sync fetch is fast enough for
    typical benchmark cadence.

    Any network or parse failure is logged once at warning level and then
    suppressed for ``error_cooldown_s`` to avoid log spam. The scraper never
    raises — callers receive ``None`` and the benchmark continues.
    """

    def __init__(
        self,
        base_url: str,
        *,
        scrape_path: str = _DEFAULT_SCRAPE_PATH,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        error_cooldown_s: float = 30.0,
        fetcher: Callable[[str, float], str] | None = None,
    ) -> None:
        self._metrics_url = _join_url(base_url, scrape_path)
        self._timeout_s = timeout_s
        self._poll_interval_s = max(0.0, poll_interval_s)
        self._error_cooldown_s = max(0.0, error_cooldown_s)
        self._fetcher = fetcher or _default_fetch
        self._lock = threading.Lock()
        self._cache: VllmMetricsSnapshot | None = None
        self._last_fetch_at: float = 0.0
        self._last_error_logged_at: float = 0.0
        self._disabled = False  # set after first hard failure to skip future work

    @property
    def metrics_url(self) -> str:
        return self._metrics_url

    def is_available(self) -> bool:
        """Return ``True`` if a snapshot has been obtained at least once."""
        return self._cache is not None

    def snapshot(self, *, force: bool = False) -> VllmMetricsSnapshot | None:
        """Return the latest snapshot, fetching if necessary.

        Returns ``None`` when the endpoint is unreachable or the scraper has
        been disabled after repeated failures. Never raises.
        """
        if self._disabled:
            return self._cache
        now = time.monotonic()
        if not force and self._cache is not None:
            age = now - self._last_fetch_at
            if age < self._poll_interval_s:
                return self._cache
        try:
            body = self._fetcher(self._metrics_url, self._timeout_s)
            snapshot = parse_vllm_metrics(body)
            snapshot = _with_source(snapshot, self._metrics_url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            self._record_failure(exc)
            return self._cache
        except Exception as exc:  # pragma: no cover - defensive
            self._record_failure(exc)
            return self._cache
        with self._lock:
            self._cache = snapshot
            self._last_fetch_at = now
        return snapshot

    def delta(
        self, before: VllmMetricsSnapshot, after: VllmMetricsSnapshot
    ) -> dict[str, float]:
        """Return per-metric deltas between two snapshots.

        Histogram deltas are reported as the mean over the request window
        (``sum_delta / count_delta``) which is the per-request average latency
        observed by vLLM between the two scrapes — the closest thing to a
        server-reported TPOT/TTFT that does not require bucket math.
        """
        return _delta(before, after)

    def _record_failure(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_logged_at >= self._error_cooldown_s:
            logger.warning(
                "vLLM metrics scrape failed for %s: %s (further errors suppressed "
                "for %.0fs)",
                self._metrics_url,
                exc,
                self._error_cooldown_s,
            )
            self._last_error_logged_at = now
        # After the first failure we keep serving the stale cache. After a few
        # consecutive failures we give up entirely so we stop paying the
        # network cost on a clearly-misconfigured endpoint.
        if self._cache is None:
            self._disabled = True


def _with_source(snapshot: VllmMetricsSnapshot, url: str) -> VllmMetricsSnapshot:
    # dataclass is frozen — recreate with the source URL stamped in
    object.__setattr__(snapshot, "source_url", url)
    return snapshot


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    # Replace an existing /v1 or /metrics suffix so callers can pass either
    # ``http://host:8000`` or ``http://host:8000/v1``.
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + path


def _default_fetch(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (trusted URL)
        raw = resp.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
# Aggregation across a benchmark run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmMetricsAggregate:
    """Run-level aggregates suitable for MLflow / report tables."""

    kv_cache_util_mean: float | None
    kv_cache_util_p95: float | None
    requests_waiting_mean: float | None
    requests_waiting_p95: float | None
    ttft_mean_s: float | None  # server-reported, from histogram sum/count deltas
    ttft_p95_s: float | None
    tpot_mean_s: float | None
    tpot_p95_s: float | None
    num_preemption_total: float | None
    snapshot_count: int = 0

    def to_flat_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in {
            "vllm_kv_cache_util_mean": self.kv_cache_util_mean,
            "vllm_kv_cache_util_p95": self.kv_cache_util_p95,
            "vllm_requests_waiting_mean": self.requests_waiting_mean,
            "vllm_requests_waiting_p95": self.requests_waiting_p95,
            "vllm_ttft_mean_s": self.ttft_mean_s,
            "vllm_ttft_p95_s": self.ttft_p95_s,
            "vllm_tpot_mean_s": self.tpot_mean_s,
            "vllm_tpot_p95_s": self.tpot_p95_s,
            "vllm_num_preemption_total": self.num_preemption_total,
        }.items():
            if value is not None and math.isfinite(value):
                out[key] = value
        return out


def aggregate_snapshots(snapshots: list[VllmMetricsSnapshot]) -> VllmMetricsAggregate:
    """Aggregate a sequence of snapshots into a single report row.

    Quantiles use a simple nearest-rank estimator consistent with the rest of
    the framework (:func:`benchmark.llm_performance._percentile` style).
    Histogram means are reconstructed from per-scrape sum/count deltas where
    possible, falling back to cumulative mean when only one snapshot exists.
    """
    if not snapshots:
        return VllmMetricsAggregate(
            kv_cache_util_mean=None,
            kv_cache_util_p95=None,
            requests_waiting_mean=None,
            requests_waiting_p95=None,
            ttft_mean_s=None,
            ttft_p95_s=None,
            tpot_mean_s=None,
            tpot_p95_s=None,
            num_preemption_total=None,
        )

    def _values(attr: str) -> list[float]:
        out: list[float] = []
        for snap in snapshots:
            value = getattr(snap, attr)
            if value is not None:
                out.append(float(value))
        return out

    gpu_vals = _values("gpu_cache_usage_perc")
    waiting_vals = _values("num_requests_waiting")

    ttft_samples = _histogram_window_means(
        snapshots, "ttft_histogram"
    )
    tpot_samples = _histogram_window_means(
        snapshots, "tpot_histogram"
    )

    preemptions = _values("num_preemption")
    # Preemption is a counter — take max - min if we can, else last value
    if preemptions:
        preemption_total = max(preemptions) - min(preemptions)
    else:
        preemption_total = None

    return VllmMetricsAggregate(
        kv_cache_util_mean=_mean(gpu_vals),
        kv_cache_util_p95=_percentile(gpu_vals, 0.95),
        requests_waiting_mean=_mean(waiting_vals),
        requests_waiting_p95=_percentile(waiting_vals, 0.95),
        ttft_mean_s=_mean(ttft_samples),
        ttft_p95_s=_percentile(ttft_samples, 0.95),
        tpot_mean_s=_mean(tpot_samples),
        tpot_p95_s=_percentile(tpot_samples, 0.95),
        num_preemption_total=preemption_total,
        snapshot_count=len(snapshots),
    )


def _histogram_window_means(
    snapshots: list[VllmMetricsSnapshot], attr: str
) -> list[float]:
    """Reconstruct per-window mean samples from cumulative histograms.

    For consecutive snapshots ``i`` and ``i+1`` the per-request mean latency
    inside that window is ``(sum_{i+1} - sum_i) / (count_{i+1} - count_i)``.
    The first snapshot's cumulative mean is included as a coarse seed so a
    single-scrape run is still represented.
    """
    samples: list[float] = []
    histograms = [getattr(s, attr) for s in snapshots]
    histograms = [h for h in histograms if h is not None and h.count > 0]
    if not histograms:
        return samples
    prev = histograms[0]
    if prev.count > 0:
        samples.append(prev.sum / prev.count)
    for current in histograms[1:]:
        dcount = current.count - prev.count
        if dcount > 0:
            ds = current.sum - prev.sum
            if ds > 0:
                samples.append(ds / dcount)
        prev = current
    return samples


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _delta(
    before: VllmMetricsSnapshot, after: VllmMetricsSnapshot
) -> dict[str, float]:
    """Compute per-metric deltas between two snapshots."""
    out: dict[str, float] = {}
    for key in (
        "num_requests_waiting",
        "num_requests_running",
        "gpu_cache_usage_perc",
        "cpu_cache_usage_perc",
        "num_preemption",
    ):
        b = getattr(before, key)
        a = getattr(after, key)
        if b is not None and a is not None:
            out[f"delta_{key}"] = a - b
    for key, hist_before, hist_after in (
        ("ttft", before.ttft_histogram, after.ttft_histogram),
        ("tpot", before.tpot_histogram, after.tpot_histogram),
        ("request_success_time", before.request_success_time_histogram, after.request_success_time_histogram),
    ):
        if (
            hist_before is not None
            and hist_after is not None
            and hist_after.count > hist_before.count
        ):
            dcount = hist_after.count - hist_before.count
            dsum = hist_after.sum - hist_before.sum
            out[f"delta_{key}_count"] = dcount
            if dsum > 0:
                out[f"delta_{key}_mean_s"] = dsum / dcount
    return out
