# Operations Runbook

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the deterministic benchmark from the YAML configured in `.env`:

```bash
BENCHMARK_CONFIG_FILE=experiments/full-grid-example.yaml python main.py
```

If `BENCHMARK_CONFIG_FILE` is omitted, `python main.py` falls back to the legacy `.env` matrix variables.

Plan and run a resumable experiment worker matrix:

```bash
python -m benchmark.worker plan experiments/full-grid-example.yaml
python -m benchmark.worker run experiments/full-grid-example.yaml --keep-going
```

The worker writes `results/runN/worker_manifest.json`, `progress.json`, per-config JSON under `configs/`, the normal reproducibility bundle, and aggregate reports for configs executed in that invocation. It skips completed configs when `--run-dir results/runN` is reused unless `--no-resume` is passed. Omit the manifest to use the existing `.env` matrix.

Enable MLflow system metrics for a run:

```bash
MLFLOW_ENABLE_SYSTEM_METRICS=true python main.py
```

Enable MLflow RAG judges for a run:

```bash
MLFLOW_GENAI_JUDGES_ENABLED=true MLFLOW_GENAI_JUDGE_MODEL=openai:/gpt-4o-mini python main.py
```

The judge path requires the provider credentials expected by MLflow, such as `OPENAI_API_KEY` for OpenAI models. Classic MLflow retriever metrics run by default when per-sample gold and retrieved document IDs are available; set `MLFLOW_CLASSIC_RETRIEVER_METRICS_ENABLED=false` to skip them.

Track estimated generator API cost for paid models by configuring current provider prices per one million tokens:

```bash
LLM_MODEL_PRICING_USD_PER_1M={"deepseek-v4-pro":{"input":0.0,"output":0.0}} python main.py
```

Replace the zeroes with the current provider prices. Reports and MLflow record input/output/total tokens even when pricing is omitted; cost fields stay empty until pricing is configured. This covers answer-generation calls, not additional RAGAS critic or HyDE calls.

Build/reuse vector indexes in separate stages:

```bash
BENCHMARK_STAGE=index python main.py
BENCHMARK_STAGE=query python main.py
```

Use LanceDB instead of Chroma:

```bash
VECTOR_DB_BACKEND=lancedb LANCEDB_PATH=.lancedb python main.py
```

Use an external HTTP RAG system as a black-box benchmark target:

```bash
RAG_SYSTEM_ADAPTER=http RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query RAG_HTTP_ANSWER_FIELD=answer RAG_HTTP_CONTEXTS_FIELD=contexts python main.py
```

Smoke-test the adapter without model-based metrics:

```bash
python examples/http_rag_server.py
RAG_SYSTEM_ADAPTER=http RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query DATASET_NAME=jsonl DATASET_PATH=examples/sample_dataset.jsonl RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py
```

The HTTP adapter sends a JSON object with `question`, `metadata`, `ground_truth`, and `config`. The endpoint should return at least an answer field and should return contexts when RAGAS/context metrics are needed.

Run the autonomous agent:

```bash
python -m agentic.cli --agent-model qwen3:8b --max-iterations 2 --sample-size 5
```

Run tests:

```bash
python -m unittest
```

Expected local services:

- Ollama at `OLLAMA_BASE_URL`, default `http://localhost:11434`, for local LLMs and embeddings.
- Optional OpenAI-compatible endpoint through `OPENAI_COMPAT_BASE_URL`.
- Optional MLflow UI can inspect `mlflow.db` and `mlruns/`.

Important runtime folders:

- `results/`: benchmark reports, per-config output, and `runN/reproducibility/` manifests.
- `.chroma/`: persisted Chroma cache.
- `.lancedb/`: persisted LanceDB tables when `VECTOR_DB_BACKEND=lancedb`.
- `mlruns/`, `mlflow.db`: MLflow tracking artifacts, including per-run reproducibility bundles.

Before long runs:

- Confirm `BENCHMARK_CONFIG_FILE` points to the intended YAML manifest.
- Confirm `.env` contains only the required service URLs and secrets from [[Configuration Reference]].
- Confirm required Ollama models are pulled.
- Use small `DATASET_SAMPLE_SIZE` for smoke tests.
- Be aware that RAGAS critic calls and BERTScore can dominate runtime.

Cleanup caution:

- Do not delete `.chroma/`, `results/`, `mlruns/`, or `mlflow.db` unless the user explicitly asks. They contain cache/history.

Related notes:

- [[Benchmark Pipeline]]
- [[Agentic Runner]]
- [[Reporting and Tracking]]

Generate resource-utilization traces and NGEN-AI paper figures:

```bash
BENCHMARK_RESOURCE_MONITOR=true BENCHMARK_RESOURCE_MONITOR_INTERVAL_SECONDS=1 python main.py
python -m benchmark.reporting.resource_charts --trace-dir results/runN/resource_traces --output Paper/NGEN-AI/figures/fig_resource_breakdown
```

The monitor writes one CSV and one marker file per benchmark configuration under `results/runN/resource_traces/`. Replace `runN` with the run folder created by the benchmark. PCIe counters depend on `nvidia-smi` support for `pcie.rx_throughput` and `pcie.tx_throughput`; unsupported counters are left blank in the trace and omitted visually.

## ClearML Remote Execution

Create a ClearML baseline task from `.env` or the first config in a manifest:

```bash
python -m benchmark.clearml_task experiments/full-grid-example.yaml --project-name "RAG Benchmarking" --task-name rag_eval_baseline
```

Start a worker on the execution machine:

```bash
clearml-agent daemon --queue rag-benchmark-gpu
```

Clone the task in the ClearML Web UI, edit Hyperparameters, and enqueue the clone to the queue. The ClearML entrypoint publishes non-secret `BenchmarkConfig` fields only; API keys, auth values, and raw HTTP headers must remain in environment variables on the agent machine. Pass `--remote-queue rag-benchmark-gpu` only when the local process should enqueue itself and exit.


Run the Enterprise RAG Blueprint adapter against the dataset selected in YAML:

```bash
set -a
source Enterprise_RAG_Blueprint/.env
set +a

PYTHONPATH=.:Enterprise_RAG_Blueprint:Enterprise_RAG_Blueprint/chain \
RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.adapter \
BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_demo.yaml \
python main.py
```

The adapter rebuilds the Blueprint Chroma collection from the manifest dataset at run time, so do not run `scripts/index_blueprint_corpus.py` unless you explicitly want to test the Blueprint sample documents outside this benchmark path.
