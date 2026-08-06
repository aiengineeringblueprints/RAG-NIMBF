"""Tests for benchmark.vllm_metrics — prometheus parser, scraper, aggregation."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from benchmark.vllm_metrics import (
    HistogramSnapshot,
    VllmMetricsAggregate,
    VllmMetricsScraper,
    VllmMetricsSnapshot,
    aggregate_snapshots,
    parse_vllm_metrics,
    _join_url,
    _parse_labels,
    _parse_prometheus_text,
)


SAMPLE_PROMETHEUS = """# HELP vllm:num_requests_waiting Number of requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 3
# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5
# HELP vllm:gpu_cache_usage_perc GPU cache usage %
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.421
# HELP vllm:cpu_cache_usage_perc CPU cache usage %
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc 0.10
# HELP vllm:num_preemption Number of preemptions
# TYPE vllm:num_preemption counter
vllm:num_preemption 2
# HELP vllm:time_to_first_token_seconds Time to first token
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.05"} 10
vllm:time_to_first_token_seconds_bucket{le="0.1"} 20
vllm:time_to_first_token_seconds_bucket{le="0.5"} 28
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 30
vllm:time_to_first_token_seconds_sum 1.5
vllm:time_to_first_token_seconds_count 30
# HELP vllm:time_per_output_token_seconds Time per output token
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01"} 100
vllm:time_per_output_token_seconds_bucket{le="0.02"} 200
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 300
vllm:time_per_output_token_seconds_sum 2.5
vllm:time_per_output_token_seconds_count 300
# HELP vllm:request_success_time_seconds Request success time
# TYPE vllm:request_success_time_seconds histogram
vllm:request_success_time_seconds_bucket{le="1.0"} 25
vllm:request_success_time_seconds_bucket{le="+Inf"} 30
vllm:request_success_time_seconds_sum 12.0
vllm:request_success_time_seconds_count 30
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestPrometheusParser:
    def test_parses_all_gauges(self):
        snap = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        assert snap.num_requests_waiting == 3.0
        assert snap.num_requests_running == 5.0
        assert snap.gpu_cache_usage_perc == pytest.approx(0.421)
        assert snap.cpu_cache_usage_perc == pytest.approx(0.10)
        assert snap.num_preemption == 2.0

    def test_parses_histograms(self):
        snap = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        assert snap.ttft_histogram is not None
        assert snap.ttft_histogram.sum == 1.5
        assert snap.ttft_histogram.count == 30
        # Largest finite bucket is le="0.5"
        assert snap.ttft_histogram.last_bucket_le == pytest.approx(0.5)
        assert snap.ttft_histogram.last_bucket_value == 28.0

        assert snap.tpot_histogram is not None
        assert snap.tpot_histogram.sum == 2.5
        assert snap.tpot_histogram.count == 300

        assert snap.request_success_time_histogram is not None
        assert snap.request_success_time_histogram.sum == 12.0

    def test_flat_dict_skips_missing_metrics(self):
        snap = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        flat = snap.to_flat_dict()
        assert "vllm_num_requests_waiting" in flat
        assert "vllm_gpu_cache_usage_perc" in flat
        assert "vllm_ttft_sum_s" in flat
        assert flat["vllm_num_requests_waiting"] == 3.0

    def test_empty_response_returns_zeros(self):
        snap = parse_vllm_metrics("")
        assert snap.num_requests_waiting is None
        assert snap.gpu_cache_usage_perc is None
        assert snap.ttft_histogram is None
        assert snap.raw_metric_count == 0

    def test_partial_metrics_only_some_present(self):
        text = "vllm:num_requests_waiting 7\n"
        snap = parse_vllm_metrics(text)
        assert snap.num_requests_waiting == 7.0
        assert snap.gpu_cache_usage_perc is None
        assert snap.ttft_histogram is None

    def test_comments_and_help_lines_ignored(self):
        text = (
            "# HELP some_other_metric unrelated\n"
            "# TYPE some_other_metric counter\n"
            "some_other_metric 42\n"
            "vllm:gpu_cache_usage_perc 0.5\n"
        )
        snap = parse_vllm_metrics(text)
        assert snap.gpu_cache_usage_perc == 0.5
        assert snap.raw_metric_count == 2

    def test_handles_scientific_notation(self):
        text = "vllm:gpu_cache_usage_perc 4.21e-1\n"
        snap = parse_vllm_metrics(text)
        assert snap.gpu_cache_usage_perc == pytest.approx(0.421)

    def test_label_parsing_with_quoted_commas(self):
        labels = _parse_labels('le="0.5",model="qwen,2.5"')
        assert labels == {"le": "0.5", "model": "qwen,2.5"}

    def test_label_parsing_empty(self):
        assert _parse_labels("") == {}
        assert _parse_labels("   ") == {}

    def test_prometheus_text_parser_groups_by_name(self):
        samples = _parse_prometheus_text(
            "vllm:num_requests_waiting 3\n"
            'vllm:time_to_first_token_seconds_bucket{le="0.1"} 5\n'
            'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10\n'
        )
        assert "vllm:num_requests_waiting" in samples
        assert len(samples["vllm:time_to_first_token_seconds_bucket"]) == 2


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class TestVllmMetricsScraper:
    def test_snapshot_uses_injected_fetcher(self):
        calls = []

        def fake_fetcher(url: str, timeout: float) -> str:
            calls.append((url, timeout))
            return SAMPLE_PROMETHEUS

        scraper = VllmMetricsScraper(
            "http://localhost:8000", fetcher=fake_fetcher
        )
        snap = scraper.snapshot(force=True)
        assert snap is not None
        assert snap.num_requests_waiting == 3.0
        assert snap.source_url == "http://localhost:8000/metrics"
        assert len(calls) == 1
        assert calls[0][0] == "http://localhost:8000/metrics"

    def test_metrics_url_joins_correctly(self):
        s1 = VllmMetricsScraper("http://host:8000")
        assert s1.metrics_url == "http://host:8000/metrics"

        s2 = VllmMetricsScraper("http://host:8000/")
        assert s2.metrics_url == "http://host:8000/metrics"

        s3 = VllmMetricsScraper("http://host:8000/v1")
        assert s3.metrics_url == "http://host:8000/metrics"

    def test_caches_within_poll_interval(self):
        fetch_count = 0

        def fake_fetcher(url: str, timeout: float) -> str:
            nonlocal fetch_count
            fetch_count += 1
            return SAMPLE_PROMETHEUS

        scraper = VllmMetricsScraper(
            "http://localhost:8000",
            fetcher=fake_fetcher,
            poll_interval_s=10.0,
        )
        scraper.snapshot(force=True)
        scraper.snapshot()  # cached
        scraper.snapshot()  # cached
        assert fetch_count == 1

    def test_force_bypasses_cache(self):
        fetch_count = 0

        def fake_fetcher(url: str, timeout: float) -> str:
            nonlocal fetch_count
            fetch_count += 1
            return SAMPLE_PROMETHEUS

        scraper = VllmMetricsScraper(
            "http://localhost:8000",
            fetcher=fake_fetcher,
            poll_interval_s=10.0,
        )
        scraper.snapshot(force=True)
        scraper.snapshot(force=True)
        assert fetch_count == 2

    def test_network_failure_returns_none_and_disables(self):
        def failing_fetcher(url: str, timeout: float) -> str:
            raise OSError("connection refused")

        scraper = VllmMetricsScraper(
            "http://localhost:8000",
            fetcher=failing_fetcher,
            error_cooldown_s=1000,
        )
        snap1 = scraper.snapshot(force=True)
        assert snap1 is None
        # Subsequent calls on a disabled scraper also return None
        snap2 = scraper.snapshot(force=True)
        assert snap2 is None

    def test_network_failure_keeps_stale_cache(self):
        call_count = 0

        def flaky_fetcher(url: str, timeout: float) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SAMPLE_PROMETHEUS
            raise OSError("transient")

        scraper = VllmMetricsScraper(
            "http://localhost:8000",
            fetcher=flaky_fetcher,
            poll_interval_s=0.0,
            error_cooldown_s=1000,
        )
        first = scraper.snapshot(force=True)
        assert first is not None
        second = scraper.snapshot(force=True)
        # Stale cache served; not None
        assert second is not None
        assert second.num_requests_waiting == first.num_requests_waiting

    def test_is_available_reflects_successful_scrape(self):
        scraper = VllmMetricsScraper(
            "http://localhost:8000",
            fetcher=lambda url, t: SAMPLE_PROMETHEUS,
        )
        assert not scraper.is_available()
        scraper.snapshot(force=True)
        assert scraper.is_available()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregateSnapshots:
    def test_empty_list_returns_all_none(self):
        agg = aggregate_snapshots([])
        assert agg.kv_cache_util_mean is None
        assert agg.requests_waiting_p95 is None
        assert agg.tpot_mean_s is None
        assert agg.snapshot_count == 0
        assert agg.to_flat_dict() == {}

    def test_single_snapshot_aggregates(self):
        snap = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        agg = aggregate_snapshots([snap])
        assert agg.kv_cache_util_mean == pytest.approx(0.421)
        assert agg.kv_cache_util_p95 == pytest.approx(0.421)
        assert agg.requests_waiting_mean == 3.0
        # TTFT mean = sum/count = 1.5/30 = 0.05
        assert agg.ttft_mean_s == pytest.approx(0.05)
        assert agg.snapshot_count == 1

    def test_window_mean_between_two_snapshots(self):
        snap1 = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        # Bump TPOT count by 100 and sum by 1.0 -> window mean = 0.01s
        snap2_text = SAMPLE_PROMETHEUS.replace(
            "vllm:time_per_output_token_seconds_sum 2.5",
            "vllm:time_per_output_token_seconds_sum 3.5",
        ).replace(
            "vllm:time_per_output_token_seconds_count 300",
            "vllm:time_per_output_token_seconds_count 400",
        )
        snap2 = parse_vllm_metrics(snap2_text)
        agg = aggregate_snapshots([snap1, snap2])
        # Window mean is 0.01; seed mean 2.5/300 ~= 0.00833 also included.
        # Average of those two: (0.00833 + 0.01) / 2
        assert agg.tpot_mean_s is not None
        assert 0.008 < agg.tpot_mean_s < 0.011
        assert agg.tpot_p95_s is not None

    def test_preemption_delta_when_counter_advances(self):
        snap1 = parse_vllm_metrics(SAMPLE_PROMETHEUS)
        snap2 = parse_vllm_metrics(
            SAMPLE_PROMETHEUS.replace(
                "vllm:num_preemption 2", "vllm:num_preemption 7"
            )
        )
        agg = aggregate_snapshots([snap1, snap2])
        assert agg.num_preemption_total == 5.0

    def test_flat_dict_omits_missing(self):
        snap = parse_vllm_metrics("vllm:num_requests_waiting 1\n")
        agg = aggregate_snapshots([snap])
        flat = agg.to_flat_dict()
        assert "vllm_requests_waiting_mean" in flat
        assert "vllm_kv_cache_util_mean" not in flat
        assert "vllm_tpot_mean_s" not in flat


# ---------------------------------------------------------------------------
# Backwards compatibility: no vLLM → no metrics, no error
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_non_vllm_endpoint_does_not_crash(self):
        """A non-vLLM server returning 404 must not raise from the scraper."""

        def not_found_fetcher(url: str, timeout: float) -> str:
            from urllib.error import HTTPError
            raise HTTPError(url, 404, "Not Found", {}, None)

        scraper = VllmMetricsScraper(
            "http://some-ollama-host:11434",
            fetcher=not_found_fetcher,
            error_cooldown_s=1000,
        )
        # First call returns None (404 caught), no exception
        snap = scraper.snapshot(force=True)
        assert snap is None

    def test_aggregate_over_no_snapshots_is_safe(self):
        agg = aggregate_snapshots([])
        assert agg.to_flat_dict() == {}

    def test_scraper_default_url_helpers(self):
        # _join_url handles all common vLLM URL forms
        assert _join_url("http://h:8000", "/metrics") == "http://h:8000/metrics"
        assert _join_url("http://h:8000/v1", "/metrics") == "http://h:8000/metrics"
        assert _join_url("http://h:8000/", "metrics") == "http://h:8000/metrics"


# ---------------------------------------------------------------------------
# Provider integration (parse_model_id + extract helpers)
# ---------------------------------------------------------------------------


class TestVllmProviderParsing:
    def test_parse_model_id_vllm_with_url(self):
        from benchmark.providers import parse_model_id

        assert parse_model_id(
            "vllm:Qwen2.5-7B-Instruct@http://localhost:8000"
        ) == ("vllm", "Qwen2.5-7B-Instruct@http://localhost:8000")

    def test_parse_model_id_vllm_without_url(self):
        from benchmark.providers import parse_model_id

        assert parse_model_id("vllm:Qwen2.5-7B-Instruct") == (
            "vllm",
            "Qwen2.5-7B-Instruct",
        )

    def test_extract_endpoint_from_model_id(self):
        from benchmark.providers import extract_vllm_endpoint

        assert extract_vllm_endpoint(
            "vllm:Qwen2.5-7B-Instruct@http://localhost:8000"
        ) == "http://localhost:8000/v1"

    def test_extract_endpoint_normalises_existing_v1_suffix(self):
        from benchmark.providers import extract_vllm_endpoint

        assert extract_vllm_endpoint(
            "vllm:Qwen2.5-7B-Instruct@http://localhost:8000/v1"
        ) == "http://localhost:8000/v1"

    def test_extract_endpoint_falls_back_to_env_url(self):
        from benchmark.providers import extract_vllm_endpoint

        assert extract_vllm_endpoint(
            "vllm:Qwen2.5-7B-Instruct",
            fallback_url="http://other:9000",
        ) == "http://other:9000/v1"

    def test_extract_endpoint_returns_none_when_no_url(self):
        from benchmark.providers import extract_vllm_endpoint

        assert extract_vllm_endpoint("vllm:Qwen2.5-7B-Instruct") is None

    def test_extract_model_name_strips_url(self):
        from benchmark.providers import extract_vllm_model_name

        assert (
            extract_vllm_model_name(
                "vllm:Qwen2.5-7B-Instruct@http://localhost:8000"
            )
            == "Qwen2.5-7B-Instruct"
        )

    def test_existing_providers_still_parse_correctly(self):
        """Adding vllm must not break ollama/openai/huggingface parsing."""
        from benchmark.providers import parse_model_id

        assert parse_model_id("ollama:gemma3:4b") == ("ollama", "gemma3:4b")
        assert parse_model_id("openai:Qwen/Qwen3-32B-AWQ") == (
            "openai",
            "Qwen/Qwen3-32B-AWQ",
        )
        assert parse_model_id("huggingface:BAAI/bge-small-en-v1.5") == (
            "huggingface",
            "BAAI/bge-small-en-v1.5",
        )
        assert parse_model_id("nomic-embed-text:latest") == (
            "ollama",
            "nomic-embed-text:latest",
        )
