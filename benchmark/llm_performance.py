"""Load-oriented LLM performance measurements for integrated RAG runs.

This module mirrors the useful parts of ``LLM_Performance_Tests-main`` without
depending on that standalone application's configuration or package layout.
The selected generator model, endpoint, credentials, token limit, and prompts
come directly from the active :class:`config.BenchmarkConfig`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from config import BenchmarkConfig


@dataclass(frozen=True)
class PerformanceCall:
    latency_s: float
    success: bool
    tokens_in: int | None = None
    tokens_out: int | None = None
    ttft_s: float | None = None
    tpot_s: float | None = None
    itl_p50_s: float | None = None
    itl_p95_s: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class LLMPerformanceResult:
    """Flat metrics are suitable for JSON, CSV, and MLflow logging."""

    model: str
    provider: str
    base_url: str
    max_tokens: int
    call_counts: tuple[int, ...]
    metrics: dict[str, float]
    calls: tuple[dict[str, Any], ...]
    error: str | None = None


def performance_cache_key(config: BenchmarkConfig) -> tuple[Any, ...]:
    """Return the fields that make an LLM load profile distinct."""
    return (
        config.llm_provider,
        config.llm_model,
        config.llm_base_url(),
        config.max_new_tokens,
        config.llm_performance_call_counts,
        config.llm_performance_timeout_seconds,
        config.llm_performance_warmup,
    )


def run_llm_performance_benchmark(
    config: BenchmarkConfig,
    prompts: list[str],
) -> LLMPerformanceResult:
    """Measure sequential and concurrent performance for the selected LLM."""
    call_counts = tuple(sorted(set(config.llm_performance_call_counts)))
    if not prompts:
        return _empty_result(config, call_counts, "No prompts available")
    if not call_counts:
        return _empty_result(config, call_counts, "No call counts configured")

    max_required = max(call_counts)
    selected_pool = prompts[:max_required]
    metrics: dict[str, float] = {}
    call_rows: list[dict[str, Any]] = []

    if config.llm_performance_warmup:
        warmup = _measure_call(config, selected_pool[0], max_tokens=min(16, config.max_new_tokens))
        metrics["warmup_success"] = float(warmup.success)
        metrics["warmup_latency_s"] = warmup.latency_s

    for requested_count in call_counts:
        selected = selected_pool[:requested_count]
        actual_count = len(selected)
        if actual_count == 0:
            continue

        seq_started = time.perf_counter()
        sequential = [_measure_call(config, prompt) for prompt in selected]
        seq_wall = time.perf_counter() - seq_started

        par_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=actual_count) as executor:
            parallel = list(executor.map(lambda prompt: _measure_call(config, prompt), selected))
        par_wall = time.perf_counter() - par_started

        prefix = f"n{requested_count}"
        metrics[f"{prefix}_actual_calls"] = float(actual_count)
        metrics.update(_aggregate_mode(prefix, "sequential", sequential, seq_wall))
        metrics.update(_aggregate_mode(prefix, "parallel", parallel, par_wall))
        metrics[f"{prefix}_speedup"] = seq_wall / par_wall if par_wall > 0 else 0.0

        for mode, calls in (("sequential", sequential), ("parallel", parallel)):
            for index, call in enumerate(calls):
                call_rows.append(
                    {
                        "requested_concurrency": requested_count,
                        "actual_calls": actual_count,
                        "mode": mode,
                        "call_index": index,
                        **asdict(call),
                    }
                )

    return LLMPerformanceResult(
        model=config.llm_model,
        provider=config.llm_provider,
        base_url=_safe_base_url(config.llm_base_url()),
        max_tokens=config.max_new_tokens,
        call_counts=call_counts,
        metrics=metrics,
        calls=tuple(call_rows),
    )


def performance_from_generation(
    config: BenchmarkConfig,
    samples: tuple[Any, ...],
    *,
    generation_wall_s: float | None = None,
) -> LLMPerformanceResult:
    """Aggregate the real RAG answer-generation calls without replaying them."""
    calls: list[PerformanceCall] = []
    call_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        output_tokens = int(
            getattr(sample, "output_tokens", 0)
            or getattr(sample, "token_count", 0)
            or 0
        )
        latency = float(getattr(sample, "total_seconds", 0.0) or 0.0)
        ttft = float(getattr(sample, "ttft_seconds", 0.0) or 0.0)
        generation_time = max(latency - ttft, 0.0)
        tpot = (
            generation_time / max(output_tokens - 1, 1)
            if output_tokens > 1 and generation_time > 0
            else None
        )
        call = PerformanceCall(
            latency_s=latency,
            success=bool(getattr(sample, "answer_valid", True)),
            tokens_in=int(getattr(sample, "input_tokens", 0) or 0),
            tokens_out=output_tokens,
            ttft_s=ttft,
            tpot_s=tpot,
        )
        calls.append(call)
        call_rows.append(
            {
                "source": "rag_generation",
                "mode": "sequential",
                "call_index": index,
                **asdict(call),
            }
        )

    wall_s = (
        float(generation_wall_s)
        if generation_wall_s is not None
        else sum(call.latency_s for call in calls)
    )
    metrics = _aggregate_mode("generation", "sequential", calls, wall_s)
    metrics["generation_actual_calls"] = float(len(calls))
    metrics["generation_tpot_is_estimated"] = 1.0

    # When the backend is vLLM, supplement the estimated TPOT with the
    # server-reported per-output-token latency scraped from /metrics. We pair
    # the per-sample before/after snapshots and average the window means.
    if config.llm_provider == "vllm":
        vllm_tpot_samples: list[float] = []
        vllm_ttft_samples: list[float] = []
        for sample in samples:
            before = getattr(sample, "vllm_metrics_before", None)
            after = getattr(sample, "vllm_metrics_after", None)
            if before is None or after is None:
                continue
            before_tpot = getattr(before, "tpot_histogram", None)
            after_tpot = getattr(after, "tpot_histogram", None)
            if (
                before_tpot is not None
                and after_tpot is not None
                and after_tpot.count > before_tpot.count
            ):
                dcount = after_tpot.count - before_tpot.count
                dsum = after_tpot.sum - before_tpot.sum
                if dsum > 0:
                    vllm_tpot_samples.append(dsum / dcount)
            before_ttft = getattr(before, "ttft_histogram", None)
            after_ttft = getattr(after, "ttft_histogram", None)
            if (
                before_ttft is not None
                and after_ttft is not None
                and after_ttft.count > before_ttft.count
            ):
                dcount = after_ttft.count - before_ttft.count
                dsum = after_ttft.sum - before_ttft.sum
                if dsum > 0:
                    vllm_ttft_samples.append(dsum / dcount)
        if vllm_tpot_samples:
            metrics["generation_tpot_vllm_mean_s"] = statistics.mean(vllm_tpot_samples)
            metrics["generation_tpot_vllm_p95_s"] = _percentile(vllm_tpot_samples, 0.95)
        if vllm_ttft_samples:
            metrics["generation_ttft_vllm_mean_s"] = statistics.mean(vllm_ttft_samples)
            metrics["generation_ttft_vllm_p95_s"] = _percentile(vllm_ttft_samples, 0.95)

    return LLMPerformanceResult(
        model=config.llm_model,
        provider=config.llm_provider,
        base_url=_safe_base_url(config.llm_base_url()),
        max_tokens=config.max_new_tokens,
        call_counts=(len(calls),),
        metrics=metrics,
        calls=tuple(call_rows),
    )


def save_llm_performance_result(
    result: LLMPerformanceResult,
    output_dir: Path,
    *,
    label: str | None = None,
) -> Path:
    """Persist detailed call data; credentials are intentionally absent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_filename_part(result.model)
    safe_label = (
        "_" + _safe_filename_part(label, limit=120)
        if label
        else ""
    )
    path = output_dir / (
        f"{result.provider}_{safe_model}_t{result.max_tokens}{safe_label}.json"
    )
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _empty_result(
    config: BenchmarkConfig,
    call_counts: tuple[int, ...],
    error: str,
) -> LLMPerformanceResult:
    return LLMPerformanceResult(
        model=config.llm_model,
        provider=config.llm_provider,
        base_url=_safe_base_url(config.llm_base_url()),
        max_tokens=config.max_new_tokens,
        call_counts=call_counts,
        metrics={},
        calls=(),
        error=error,
    )


def _aggregate_mode(
    prefix: str,
    mode: str,
    calls: list[PerformanceCall],
    wall_s: float,
) -> dict[str, float]:
    successful = [call for call in calls if call.success]
    latencies = [call.latency_s for call in successful]
    ttfts = [call.ttft_s for call in successful if call.ttft_s is not None]
    tpots = [call.tpot_s for call in successful if call.tpot_s is not None]
    itl_p95s = [call.itl_p95_s for call in successful if call.itl_p95_s is not None]
    output_tokens = sum(call.tokens_out or 0 for call in successful)
    total_latency = sum(latencies)
    key = f"{prefix}_{mode}"

    values: dict[str, float] = {
        f"{key}_calls": float(len(calls)),
        f"{key}_success_rate": len(successful) / len(calls) if calls else 0.0,
        f"{key}_wall_s": wall_s,
        f"{key}_request_throughput_rps": len(successful) / wall_s if wall_s > 0 else 0.0,
        f"{key}_output_tokens_per_s": output_tokens / total_latency
        if total_latency > 0
        else 0.0,
    }
    if latencies:
        values.update(
            {
                f"{key}_latency_mean_s": statistics.mean(latencies),
                f"{key}_latency_median_s": statistics.median(latencies),
                f"{key}_latency_p95_s": _percentile(latencies, 0.95),
                f"{key}_latency_std_s": statistics.stdev(latencies)
                if len(latencies) > 1
                else 0.0,
            }
        )
    if ttfts:
        values[f"{key}_ttft_mean_s"] = statistics.mean(ttfts)
        values[f"{key}_ttft_p95_s"] = _percentile(ttfts, 0.95)
    if tpots:
        values[f"{key}_tpot_mean_s"] = statistics.mean(tpots)
    if itl_p95s:
        values[f"{key}_itl_p95_mean_s"] = statistics.mean(itl_p95s)
    return values


def _safe_base_url(base_url: str) -> str:
    """Remove URL user-info before a configured endpoint is persisted."""
    try:
        parsed = urlsplit(base_url)
        if parsed.username is None and parsed.password is None:
            return base_url
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = f"{hostname}:{parsed.port}" if parsed.port else hostname
        return urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
        )
    except ValueError:
        return "<redacted-invalid-url>"


def _safe_filename_part(value: str, *, limit: int = 80) -> str:
    return re.sub(r"[^\w.-]", "_", value)[:limit]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _measure_call(
    config: BenchmarkConfig,
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> PerformanceCall:
    if config.llm_provider in ("openai", "vllm"):
        return _measure_openai(config, prompt, max_tokens or config.max_new_tokens)
    if config.llm_provider == "ollama":
        return _measure_ollama(config, prompt, max_tokens or config.max_new_tokens)
    return PerformanceCall(
        latency_s=0.0,
        success=False,
        error=f"Unsupported performance-test provider: {config.llm_provider}",
    )


def _stream_metrics(
    *,
    started: float,
    first_activity: float | None,
    finished: float,
    inter_token_times: list[float],
    tokens_out: int | None,
) -> dict[str, float | None]:
    latency = finished - started
    ttft = first_activity - started if first_activity is not None else None
    return {
        "latency_s": latency,
        "ttft_s": ttft,
        "tpot_s": statistics.mean(inter_token_times) if inter_token_times else None,
        "itl_p50_s": statistics.median(inter_token_times) if inter_token_times else None,
        "itl_p95_s": _percentile(inter_token_times, 0.95) if inter_token_times else None,
    }


def _measure_openai(
    config: BenchmarkConfig,
    prompt: str,
    max_tokens: int,
) -> PerformanceCall:
    started = time.perf_counter()
    try:
        from openai import OpenAI

        # vLLM ids can embed the deployment URL ("model@http://..."); the
        # chat completion request needs just the model portion.
        request_model = (
            config.llm_model.split("@", 1)[0]
            if config.llm_provider == "vllm" and "@" in config.llm_model
            else config.llm_model
        )
        client_kwargs: dict[str, Any] = {
            "api_key": config.llm_api_key() or "not-needed",
            "timeout": config.llm_performance_timeout_seconds,
        }
        if config.llm_base_url():
            client_kwargs["base_url"] = config.llm_base_url()
        client = OpenAI(**client_kwargs)
        stream = client.chat.completions.create(
            model=request_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        first_activity: float | None = None
        last_activity: float | None = None
        inter_token_times: list[float] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices else None
            reasoning = (
                getattr(chunk.choices[0].delta, "reasoning_content", None)
                if chunk.choices
                else None
            )
            if content or reasoning:
                now = time.perf_counter()
                if first_activity is None:
                    first_activity = now
                elif last_activity is not None:
                    inter_token_times.append(now - last_activity)
                last_activity = now
            if chunk.usage:
                tokens_in = chunk.usage.prompt_tokens
                tokens_out = chunk.usage.completion_tokens
        finished = time.perf_counter()
        measured = _stream_metrics(
            started=started,
            first_activity=first_activity,
            finished=finished,
            inter_token_times=inter_token_times,
            tokens_out=tokens_out,
        )
        return PerformanceCall(
            success=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            **measured,
        )
    except Exception as exc:
        return PerformanceCall(
            latency_s=time.perf_counter() - started,
            success=False,
            error=_sanitize_error(config, exc),
        )


def _measure_ollama(
    config: BenchmarkConfig,
    prompt: str,
    max_tokens: int,
) -> PerformanceCall:
    started = time.perf_counter()
    try:
        import requests

        headers = {"Content-Type": "application/json"}
        if config.llm_api_key():
            headers["Authorization"] = f"Bearer {config.llm_api_key()}"
        response = requests.post(
            f"{config.llm_base_url().rstrip('/')}/api/chat",
            json={
                "model": config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "think": False,
                "options": {"num_predict": max_tokens, "temperature": 0},
            },
            headers=headers,
            timeout=config.llm_performance_timeout_seconds,
            stream=True,
        )
        response.raise_for_status()

        first_activity: float | None = None
        last_activity: float | None = None
        inter_token_times: list[float] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            message = payload.get("message") or {}
            if message.get("content") or message.get("thinking"):
                now = time.perf_counter()
                if first_activity is None:
                    first_activity = now
                elif last_activity is not None:
                    inter_token_times.append(now - last_activity)
                last_activity = now
            if payload.get("done"):
                tokens_in = payload.get("prompt_eval_count")
                tokens_out = payload.get("eval_count")
        finished = time.perf_counter()
        measured = _stream_metrics(
            started=started,
            first_activity=first_activity,
            finished=finished,
            inter_token_times=inter_token_times,
            tokens_out=tokens_out,
        )
        return PerformanceCall(
            success=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            **measured,
        )
    except Exception as exc:
        return PerformanceCall(
            latency_s=time.perf_counter() - started,
            success=False,
            error=_sanitize_error(config, exc),
        )


def _sanitize_error(config: BenchmarkConfig, exc: Exception) -> str:
    message = str(exc)
    secret = config.llm_api_key()
    if secret:
        message = message.replace(secret, "<redacted>")
    base_url = config.llm_base_url()
    if base_url:
        message = message.replace(base_url, _safe_base_url(base_url))
    return message
