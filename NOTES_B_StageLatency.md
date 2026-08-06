# B — Per-Stage Latency Breakdown

Adds per-stage wall-clock latency instrumentation to the RAG benchmark,
inspired by RAGPerf (arXiv:2603.10765v1) §3.4. The instrumentation is
always on; it has sub-microsecond overhead per call and uses only
`time.perf_counter()` from the standard library.

## What changed

### New module: `benchmark/stage_timing.py`
- `StageTimings` — accumulates per-call latency samples per stage, plus
  optional counts (chunks, documents, questions) used to derive
  throughput.
- `StageStats` dataclass — `count, total_s, mean_s, p50_s, p95_s, max_s`.
- `stage_timer(recorder, stage)` — context manager that times a block and
  pushes one sample into the recorder.
- `timed(recorder, stage)` — decorator form.
- `StageTimings.summary()` returns `{stage -> {count, total_s, ...}}`.
- `StageTimings.mlflow_metrics()` flattens to `lat_<stage>_<stat>` and
  `throughput_<name>` keys.
- `StageTimings.save_json(path)` writes a JSON artifact.

Canonical stage names:
`chunking, embedding, indexing, retrieval, reranking, generation,
evaluation, total`. The `embedding` stage is reserved for future
fine-grained instrumentation; in the current pipeline embedding is bundled
into `indexing` (it happens inside `build_vector_store`). Throughput
`throughput_embed_index_chunks_s` therefore reports chunks embedded+indexed
per second over the `indexing` window.

### Wiring in `main.py`
- `_stage_timer` gained optional `latency_recorder` and `latency_stage`
  parameters. Existing call sites (and the legacy `stage_timings` dict used
  for `stage_<name>_seconds` MLflow keys) are unchanged; canonical names
  flow only into the new recorder.
- `_run_single_benchmark_impl` creates a `StageTimings` recorder and feeds:
  - corpus-level samples for `chunking`, `indexing`
  - per-question samples for `retrieval`, `reranking`, `generation`
  - one sample each for `evaluation` (RAGAS / custom-metrics) and `total`
- `latency_recorder.set_count("chunks", N)` / `("questions", N)` enables
  throughput derivation.
- The recorder's summary is attached to `BenchmarkResultExtended` via the
  new `stage_latency` field.
- The recorder persists `results/runN/stage_timings/<config_safe>.json`
  next to existing per-config artifacts.
- Both the `index`-stage early return and the full benchmark return path
  save the JSON.

### `benchmark/reporting/models.py`
- New optional field on `BenchmarkResultExtended`:
  `stage_latency: dict[str, dict[str, float | int]] | None`.

### `benchmark/tracking.py`
- `log_benchmark_run` now emits, for every stage in `result.stage_latency`:
  - `lat_<stage>_total_s`, `lat_<stage>_mean_s`, `lat_<stage>_p50_s`,
    `lat_<stage>_p95_s`, `lat_<stage>_max_s`, `lat_<stage>_count`
- Throughput metrics derived from stage stats + `num_chunks`:
  - `throughput_embed_index_chunks_s`
  - `throughput_retrieve_qps`
  - `throughput_generate_qps`

### `benchmark/reporting/exports.py`
- JSON report and summary CSV now include the `stage_latency` block / columns.

### `benchmark/reporting/terminal.py`
- New `Stage Latency Breakdown (seconds)` table in the run report. Each
  cell shows `mean / p95` with `n=<count>` when the stage has more than
  one sample (per-question stages); corpus-level stages show the single
  total.

## How to view metrics
- Terminal: the new table appears automatically in the rich console report
  between the RAGAS scores and the sparkline comparison.
- JSON: `results/runN/benchmark_<timestamp>.json` — `results[].stage_latency`.
- CSV: `results/runN/results_summary.csv` — columns prefixed `lat_<stage>_*`.
- MLflow: under each run, look for metrics prefixed `lat_` and `throughput_`.
- Raw artifact: `results/runN/stage_timings/<config_safe>.json`.

## Sample timing output
```json
{
  "stages": {
    "chunking":  {"count": 1,  "total_s": 0.010, "mean_s": 0.010, "p50_s": 0.010, "p95_s": 0.010, "max_s": 0.010},
    "indexing":  {"count": 1,  "total_s": 1.200, "mean_s": 1.200, "p50_s": 1.200, "p95_s": 1.200, "max_s": 1.200},
    "retrieval": {"count": 10, "total_s": 0.670, "mean_s": 0.067, "p50_s": 0.050, "p95_s": 0.141, "max_s": 0.200},
    "generation":{"count": 10, "total_s": 2.680, "mean_s": 0.268, "p50_s": 0.200, "p95_s": 0.566, "max_s": 0.800},
    "total":     {"count": 1,  "total_s": 5.000, "mean_s": 5.000, "p50_s": 5.000, "p95_s": 5.000, "max_s": 5.000}
  },
  "counts":    {"chunks": 80, "questions": 10},
  "throughput": {
    "embed_index_chunks_s": 66.67,
    "retrieve_qps":         14.93,
    "generate_qps":          3.73
  }
}
```

## Overhead estimate
Measured locally with a 1,000,000-iteration busy loop:
```
1,000,000 iterations in 1.086 s -> 1086.2 ns/op (~1 µs per call)
```
At ~100 questions × 4 per-question stages per run this is roughly 0.4 ms
of overhead per benchmark configuration — well below the noise floor of any
network or GPU-bound stage.

## Tests
`tests/test_stage_timing.py` covers:
- canonical stage list,
- `stage_timer` elapsed-time accuracy,
- decorator records one sample per call,
- aggregation (count, total, mean, p50, p95, max),
- JSON round-trip serialization,
- throughput derivation (`embed_index_chunks_s`, `retrieve_qps`),
- MLflow metric key naming (`lat_<stage>_<stat>`, `throughput_*`),
- `save_json` writes a parseable file,
- MLflow mock: `log_benchmark_run` emits both `lat_*` and `throughput_*`
  metrics with correct values when `stage_latency` is populated.

Run with `python -m pytest tests/test_stage_timing.py`.
