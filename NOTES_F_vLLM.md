# vLLM Backend Notes

This file documents how to deploy [vLLM](https://github.com/vllm-project/vllm),
point the benchmarking framework at it, and interpret the production-grade
latency metrics that vLLM exposes via its Prometheus `/metrics` endpoint.

Inspired by RAGPerf §3.3.4 (arXiv:2603.10765v1).

---

## 1. Deploying vLLM

vLLM serves any HuggingFace model behind an OpenAI-compatible HTTP API at
`http://vllm-host:port/v1`, and a Prometheus exposition endpoint at
`http://vllm-host:port/metrics`.

Quick start (single-GPU, AWQ-quantized Qwen2.5-7B):

```bash
pip install vllm

# Start the OpenAI-compatible server with prometheus metrics enabled.
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --disable-log-requests
```

vLLM's prometheus endpoint is enabled by default — no extra flag required.
Verify with:

```bash
curl http://localhost:8000/metrics | grep vllm:
curl http://localhost:8000/v1/models
```

For multi-GPU or tensor-parallel deployments:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 16384
```

---

## 2. Pointing the framework at vLLM

The model id embeds the deployment URL using the
`vllm:<model>@<server-root>` form. The framework normalises both
`http://host:8000` and `http://host:8000/v1` and reuses the OpenAI-compat
client under the hood.

### Via experiment YAML

See [`experiments/vllm_demo.yaml`](experiments/vllm_demo.yaml):

```yaml
matrix:
  llm_models:
    - "vllm:Qwen2.5-7B-Instruct@http://localhost:8000"
```

### Via `.env` (legacy matrix)

```dotenv
LLM_MODELS=vllm:Qwen2.5-7B-Instruct@http://localhost:8000
# Optional fallback if URL is omitted from the id:
LLM_OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
# Optional API key (only if vLLM is fronted by a gateway):
LLM_OPENAI_COMPAT_API_KEY=sk-not-needed
```

The framework auto-detects vLLM from the `vllm:` prefix and enables the
prometheus scraper. To force-disable scraping (e.g. when vLLM is proxied
behind an endpoint that does not expose `/metrics`):

```dotenv
VLLM_METRICS_ENABLED=false
```

---

## 3. What the framework measures

When vLLM is the active backend, the framework scrapes `/metrics` once
before and once after each generation call and aggregates the snapshots
across the run. The following vLLM metrics are surfaced as first-class
MLflow metrics and in the JSON report (prefix `vllm_`):

| MLflow metric | vLLM source | Interpretation |
|---|---|---|
| `vllm_kv_cache_util_mean` / `..._p95` | `vllm:gpu_cache_usage_perc` | KV-cache memory pressure. Approaching 1.0 means vLLM is about to evict or preempt. |
| `vllm_requests_waiting_mean` / `..._p95` | `vllm:num_requests_waiting` | Scheduler queue depth. Non-zero values indicate the model is request-bound (scheduling delay upstream of TTFT). |
| `vllm_ttft_mean_s` / `..._p95_s` | `vllm:time_to_first_token_seconds` (histogram, windowed) | Server-reported time-to-first-token, reconstructed from cumulative sum/count deltas between consecutive scrapes. |
| `vllm_tpot_mean_s` / `..._p95_s` | `vllm:time_per_output_token_seconds` (histogram, windowed) | Server-reported per-output-token latency. Compare against the framework's estimated TPOT (`llm_perf_generation_tpot_mean_s`) — large divergence suggests client-side measurement noise or streaming-buffer artefacts. |
| `vllm_num_preemption_total` | `vllm:num_preemption` (counter) | Total preemptions over the run. Non-zero means vLLM ran out of KV-cache blocks and had to swap/recompute — a strong signal that `--gpu-memory-utilization` or `--max-num-seqs` needs tuning. |

In addition, when `llm_performance_source: generation` is set, the LLM
load-test module supplements its estimated TPOT with the vLLM-reported
value:

- `llm_perf_generation_tpot_mean_s` — client-estimated (existing)
- `llm_perf_generation_tpot_vllm_mean_s` — server-reported (new)
- `llm_perf_generation_tpot_vllm_p95_s` — server-reported p95

---

## 4. Metric interpretation guide

### Healthy baseline

| Metric | Healthy range | Notes |
|---|---|---|
| `vllm_kv_cache_util_mean` | 0.3–0.7 | Headroom for burst traffic. Spikes >0.9 during load tests signal capacity ceiling. |
| `vllm_requests_waiting_p95` | 0 | Any non-zero p95 means requests queued waiting for a KV-cache slot. |
| `vllm_ttft_p95_s` | <0.5s for 7B, <1.5s for 70B | Prefill-bound; increases with prompt length and batch size. |
| `vllm_tpot_mean_s` | 0.01–0.05 | Decode-bound; fairly invariant to batch size on healthy GPUs. |
| `vllm_num_preemption_total` | 0 | Non-zero → tune `--max-num-seq` or increase `--gpu-memory-utilization`. |

### Diagnostic flows

- **High TTFT + low KV-cache util** → prefill is the bottleneck. Reduce
  `--max-model-len`, lower `--max-num-seqs`, or enable `--enable-chunked-prefill`.
- **High TPOT + low KV-cache util** → GPU compute-bound decode. Upgrade
  GPU, lower batch size, or use a smaller quantization (AWQ/GPTQ).
- **High KV-cache util + non-zero preemption** → KV-cache capacity-bound.
  Increase `--gpu-memory-utilization`, reduce `--max-model-len`, or add GPU
  memory. The framework will surface this as a non-zero
  `vllm_num_preemption_total`.
- **Non-zero `requests_waiting` with low GPU utilisation** → CPU/PCIe
  bound (data loading, tokenization). Check `--enforce-eager` and
  tokenizer parallelism.

---

## 5. Comparison vs Ollama baseline

> **Placeholder** — populate with real numbers after running
> `experiments/vllm_demo.yaml` and a matching `ollama:` baseline on the
> same hardware.

| Metric | Ollama (`gemma3:4b`) | vLLM (`Qwen2.5-7B-Instruct-AWQ`) | Δ |
|---|---|---|---|
| `avg_ttft_seconds` | TBD | TBD | TBD |
| `avg_tokens_per_second` | TBD | TBD | TBD |
| `llm_perf_generation_tpot_mean_s` | TBD | TBD | TBD |
| `llm_perf_generation_tpot_vllm_mean_s` | n/a (Ollama has no scraper) | TBD | TBD |
| `vllm_kv_cache_util_mean` | n/a | TBD | TBD |
| `vllm_requests_waiting_p95` | n/a | TBD | TBD |
| `total_time_seconds` | TBD | TBD | TBD |

Expected pattern: vLLM with continuous batching should deliver 2–5x higher
throughput under concurrent load (`llm_perf_n10_parallel_*`) and lower TTFT
variance than Ollama, at the cost of higher GPU memory utilisation. Single-
request latency (`n1_sequential_latency_mean_s`) may be similar.

---

## 6. Architecture notes

### Why a stdlib prometheus parser?

`benchmark/vllm_metrics.py` parses the prometheus exposition format with
hand-rolled regex and `urllib.request`, with no dependency on
`prometheus_client` or `requests`. This keeps vLLM a soft dependency:
the framework runs unchanged against Ollama, OpenAI, and vLLM, and only
the scraper module needs to know about the prometheus text format.

### Graceful degradation

The scraper (`VllmMetricsScraper`) never raises:

- Network errors (`URLError`, `HTTPError`, `TimeoutError`, `OSError`) are
  logged once at warning level, then suppressed for `error_cooldown_s`
  (default 30s) to avoid log spam.
- After the first failure with an empty cache, the scraper disables itself
  so subsequent calls are cheap no-ops.
- A stale cache is served when available, so transient network blips do
  not produce gaps in the metric series.
- Missing vLLM metrics (older version, fresh boot with no traffic) are
  returned as `None` rather than `0.0` so downstream aggregation does not
  conflate "absent" with "zero".

### Histogram reconstruction

vLLM exposes TTFT and TPOT as cumulative histograms
(`*_bucket{le=...}`, `*_sum`, `*_count`). For runtime monitoring we
compute per-window means by taking the delta of `sum` and `count` between
two consecutive scrapes:

```
window_mean = (sum_after - sum_before) / (count_after - count_before)
```

This avoids full bucket-math (which would be needed for exact p95) while
still giving a faithful per-request average latency observed by the server
inside the window. For run-level p95 we apply nearest-rank quantile
estimation across the per-window means — coarse, but good enough for
scheduling/perf-regression signals. Exact p95 from bucket boundaries is a
future enhancement.

---

## 7. Files changed for vLLM support

| File | Change |
|---|---|
| `benchmark/providers.py` | `vllm` provider parsing, `extract_vllm_endpoint`, `extract_vllm_model_name`, `detect_vllm_deployment`, vLLM branch in `get_chat_model`. |
| `benchmark/vllm_metrics.py` | **New.** `VllmMetricsScraper`, `VllmMetricsSnapshot`, `VllmMetricsAggregate`, stdlib prometheus parser, `aggregate_snapshots`. |
| `benchmark/generation.py` | `GenerationResult.vllm_metrics_before` / `vllm_metrics_after` fields. |
| `benchmark/llm_performance.py` | vLLM branch in `_measure_call`; `generation_tpot_vllm_*` metrics in `performance_from_generation`. |
| `benchmark/reporting/models.py` | `BenchmarkResultExtended.vllm_metrics` field. |
| `benchmark/tracking.py` | `vllm_*` metrics logged to MLflow. |
| `config.py` | `llm_base_url` / `llm_api_key` vLLM branches; `vllm_metrics_enabled` property. |
| `main.py` | Scraper lifecycle (init + per-sample before/after sampling + aggregation). |
| `tests/test_vllm_metrics.py` | **New.** 33 tests covering parser, scraper, aggregation, backwards compat, provider parsing. |
| `experiments/vllm_demo.yaml` | **New.** Example experiment manifest. |
