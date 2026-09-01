# RAG Benchmarking Framework

This project benchmarks RAG systems across datasets, retrieval outputs, answer
quality metrics, runtime measurements, reports, and MLflow tracking.

The framework can run its built-in RAG pipeline, or evaluate an external RAG
system as a black-box HTTP service.

## Install

```bash
pip install -r requirements.txt
```

For a fully reproducible Python 3.12 environment, install the compiled lock:

```bash
pip install -r requirements-lock.txt
```

Regenerate it after dependency changes with the command recorded at the top of
`requirements-lock.txt`.

## Run the Built-In Pipeline

```bash
BENCHMARK_CONFIG_FILE=experiments/full-grid-example.yaml python main.py
```

The built-in mode chunks/indexes the selected dataset, retrieves contexts,
generates answers, evaluates the results, writes reports under `results/`, and
logs to MLflow. Keep models, chunking, retrieval, datasets, and evaluator
settings in YAML; keep API keys and machine-local service URLs in `.env`. If
`BENCHMARK_CONFIG_FILE` is not set, `python main.py` still falls back to the
legacy `.env` matrix variables.

### Include an LLM Load Test in `main.py`

Enable the integrated version of `LLM_Performance_Tests-main` to test every
selected generator LLM with the current endpoint, dataset questions, and
`max_new_tokens` value:

```yaml
settings:
  llm_performance_enabled: true
  llm_performance_source: generation
  llm_performance_call_counts: [1, 3, 6, 10]
  llm_performance_warmup: true
  llm_performance_timeout_seconds: 60
```

The same settings can be supplied through `.env`:

```bash
LLM_PERFORMANCE_ENABLED=true
LLM_PERFORMANCE_SOURCE=generation
LLM_PERFORMANCE_CALL_COUNTS=1,3,6,10
python main.py
```

With `llm_performance_source: generation`, the framework reuses the real RAG
answer-generation calls, including their complete context prompts. It adds no
LLM requests and records mean/median/P95 latency, TTFT, estimated TPOT, output
tokens/s, requests/s, success rate, and generation wall time.

Set `llm_performance_source: load_test` when you explicitly need the additional
sequential-versus-parallel load profiles. For each configured call count, that
mode runs distinct dataset questions once sequentially and once concurrently
and also reports parallel speedup and inter-token latency.

Results appear in the terminal report, MLflow metrics (prefix
`llm_perf_`), the summary CSV, the aggregate JSON, and detailed files under
`results/runN/llm_performance/`.

In `load_test` mode, an identical
`(provider, model, endpoint, max_new_tokens, load profile)` is measured only
once per `main.py` run and reused across configurations. In `generation` mode,
every RAG configuration keeps its own observed timings because context and
prompt length can differ. External black-box RAG adapters are skipped because
their internal generator endpoint cannot be inferred safely.


## Run a Resumable Experiment Matrix

Use the worker when you want to move the repo to another machine and let it run
a full configuration matrix unattended:

```bash
python -m benchmark.worker plan experiments/full-grid-example.yaml
python -m benchmark.worker run experiments/full-grid-example.yaml --keep-going
```

The worker expands the manifest into `BenchmarkConfig` objects, runs them
sequentially through the same benchmark core as `main.py`, writes every config
result immediately, and records resume state in `results/runN/progress.json`.
Reuse a run directory to continue after interruption:

```bash
python -m benchmark.worker run experiments/full-grid-example.yaml --run-dir results/run3
```

Omit the manifest to use `BENCHMARK_CONFIG_FILE` when set, or the legacy `.env` matrix otherwise.
For the normal non-worker workflow, `BENCHMARK_CONFIG_FILE=<manifest> python main.py`
uses the same manifest format without requiring ClearML.

## Run With ClearML Agent

Use ClearML when you want to clone a benchmark task in the Web UI, edit
hyperparameters, enqueue it, and let a `clearml-agent` execute it on a worker
machine.

Create the initial task from `.env` or from the first expanded config in a
manifest:

```bash
python -m benchmark.clearml_task experiments/full-grid-example.yaml \
  --project-name "RAG Benchmarking" \
  --task-name rag_eval_baseline
```

The task exposes non-secret `BenchmarkConfig` fields such as `llm_model`,
`embedding_model`, `chunk_size`, `chunk_overlap`, `retrieval_top_k`,
`prompt_template`, `reranker_model`, and dataset settings in ClearML
Hyperparameters. API keys, auth headers, and raw HTTP headers are intentionally
not published to ClearML.

Start an agent on the execution machine:

```bash
clearml-agent daemon --queue rag-benchmark-gpu
```

Then clone the baseline task in the ClearML UI, edit Hyperparameters, and
enqueue the clone to `rag-benchmark-gpu`. For a local smoke submission you can
also let the script enqueue itself and exit locally:

```bash
python -m benchmark.clearml_task --remote-queue rag-benchmark-gpu
```

The ClearML task still runs the existing worker core, writes `results/runN/`,
logs scalar benchmark metrics to ClearML, and keeps MLflow logging enabled unless
`--no-mlflow` is passed.

## Benchmark RAG Systems

- Enterprise RAG Blueprint: see [doc/Enterprise_RAG_Blueprint_Benchmark.md](doc/Enterprise_RAG_Blueprint_Benchmark.md).
- optimiseRAG quickstart and operations: see [doc/OptimiseRAG_Benchmark.md](doc/OptimiseRAG_Benchmark.md).
- Generic managed RAG adapters: see [doc/Managed_RAG_System_Usage.md](doc/Managed_RAG_System_Usage.md).
- Your own RAG system: see [doc/Benchmark_Your_RAG.md](doc/Benchmark_Your_RAG.md).

## Use an External RAG System

Use this mode when your RAG system already exists and you want this framework to
act as a drop-in evaluation layer.

Set `RAG_SYSTEM_ADAPTER=http` and point the framework at your RAG endpoint:

```bash
RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
RAG_HTTP_ANSWER_FIELD=answer \
RAG_HTTP_CONTEXTS_FIELD=contexts \
python main.py
```

For a fast local smoke test, run the bundled demo endpoint and sample JSONL dataset:

```bash
python examples/http_rag_server.py

RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
DATASET_NAME=jsonl \
DATASET_PATH=examples/sample_dataset.jsonl \
RAGAS_ENABLED=false \
CUSTOM_METRICS_ENABLED=false \
python main.py
```

In HTTP mode, the framework skips its internal chunking, retrieval, and
generation. Your service owns the RAG pipeline; this framework sends benchmark
questions, normalizes the response, then runs the same evaluation, reporting,
and MLflow tracking path.

### Request Format

For each benchmark sample, the framework sends a JSON `POST` request:

```json
{
  "question": "What is the answer?",
  "metadata": {
    "id": "sample-1"
  },
  "ground_truth": "Expected answer",
  "config": {
    "name": "recursive_cs1000_co200_model_llm_concise_http",
    "retrieval_top_k": 5,
    "prompt_template": "concise",
    "dataset_name": "t2-ragbench"
  }
}
```

Your service should answer with a JSON object.

### Minimal Response

```json
{
  "answer": "The generated answer"
}
```

This is enough for answer-only metrics, but context-based RAG metrics will be
limited.

### Recommended Response

```json
{
  "answer": "The generated answer",
  "contexts": [
    "First retrieved context passage",
    "Second retrieved context passage"
  ],
  "metadata": [
    {"doc_id": "doc-1", "score": 0.91},
    {"doc_id": "doc-2", "score": 0.84}
  ],
  "timings": {
    "ttft_seconds": 0.12,
    "total_seconds": 1.47,
    "token_count": 128
  }
}
```

Return `contexts` whenever possible. RAGAS context metrics and custom retrieval
metrics need retrieved evidence to evaluate faithfulness and retrieval quality.

### Nested Response Fields

If your API returns nested data, map the response fields with dotted paths:

```bash
RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
RAG_HTTP_ANSWER_FIELD=result.answer \
RAG_HTTP_CONTEXTS_FIELD=result.sources \
RAG_HTTP_METADATA_FIELD=result.source_metadata \
RAG_HTTP_TIMINGS_FIELD=metrics \
python main.py
```

For context entries, the adapter accepts either strings or objects containing
one of these text fields: `text`, `content`, `page_content`, or `context`.

### Local JSONL/CSV Datasets

Use `DATASET_NAME=jsonl` or `DATASET_NAME=csv` when you want to evaluate an
external RAG system with your own question set instead of a built-in Hugging
Face dataset. Each row must provide a question and ground-truth answer. Context
and metadata are optional but recommended.

```bash
DATASET_NAME=jsonl \
DATASET_PATH=path/to/samples.jsonl \
DATASET_QUESTION_FIELD=question \
DATASET_GROUND_TRUTH_FIELD=ground_truth \
DATASET_CONTEXT_FIELD=context \
DATASET_METADATA_FIELD=metadata \
python main.py
```

For CSV datasets, `metadata` may be either blank, a plain string, or a JSON
object encoded as a string.

### Native Python Adapter Plugins

For a first-class Python integration, register an adapter in a module and ask
the config loader to import it before validation:

```python
from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput

class MyRagAdapter:
    name = "myrag"

    def prepare(self, config, data, corpus=None):
        return None

    def answer(self, sample, config):
        result = my_rag.query(sample["question"])
        return RagSystemOutput(
            answer=result.answer,
            contexts=result.contexts,
            metadata=result.metadata,
            total_seconds=result.total_seconds,
            token_count=result.token_count,
            answer_valid=bool(result.answer.strip()),
        )

register_rag_adapter("myrag", lambda config: MyRagAdapter())
```

```bash
RAG_ADAPTER_MODULES=my_package.my_adapter RAG_SYSTEM_ADAPTER=myrag python main.py
```

### Metric Controls

Use these switches for fast endpoint smoke tests or answer-only evaluations:

```bash
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py
```

`RAGAS_ENABLED=false` skips RAGAS critic calls. `CUSTOM_METRICS_ENABLED=false`
skips custom embedding/BERTScore metrics. Reporting still records answers,
contexts, timing, token counts, and validity.

### Headers and Auth

Static headers can be supplied as JSON:

```bash
RAG_HTTP_HEADERS='{"X-Project": "benchmark"}'
```

For a single auth header, prefer environment variables so secrets do not enter
source files:

```bash
RAG_HTTP_AUTH_HEADER=Authorization \
RAG_HTTP_AUTH_VALUE="Bearer $RAG_API_TOKEN"
```

### Compare Internal RAG With an MCP System

The built-in `mcp` adapter calls one tool on an MCP server. MCP is the
transport contract, not a retrieval algorithm, so the comparison measures the
concrete MCP tool/server you configure.

Two result modes are available:

- `context`: tool results are evidence for the same configured generator used
  by the framework. This isolates internal RAG retrieval versus MCP retrieval.
- `answer`: the tool result is the final answer. This benchmarks a completely
  external MCP-only QA system without a framework generator call.

Two execution modes are available:

- `fixed`: invoke `MCP_TOOL_NAME` once per question.
- `agentic`: expose the allowlisted MCP tools to the configured chat model and
  let it make several tool calls before producing a final answer.

Run the included lexical MCP baseline against internal vector RAG:

```bash
pip install -r requirements.txt
BENCHMARK_CONFIG_FILE=experiments/rag-vs-mcp.yaml python main.py
```

Additional ready-to-run comparisons are provided:

```bash
# LLM-controlled search + lookup MCP tools
BENCHMARK_CONFIG_FILE=experiments/rag-vs-mcp-agentic.yaml python main.py

# MCP tool supplies the final answer
BENCHMARK_CONFIG_FILE=experiments/rag-vs-mcp-answer.yaml python main.py
```

For a remote Streamable HTTP server, configure machine-local connectivity in
`.env` and keep the workflow fields in YAML:

```bash
MCP_TRANSPORT=streamable_http
MCP_SERVER_URL=https://mcp.example.com/mcp
MCP_TOOL_NAME=search
MCP_RESULT_MODE=context
MCP_RESULT_FIELD=contexts
```

For stdio servers, `MCP_COMMAND` and `MCP_ARGS_JSON` are passed as an executable
and argument array; the adapter never invokes a shell. `MCP_ENV_VARS` is a
comma-separated allowlist of environment-variable names that the child server
needs. HTTP headers and static tool arguments are JSON objects in
`MCP_HTTP_HEADERS_JSON` and `MCP_TOOL_ARGUMENTS_JSON`.

The client session is initialized once during adapter preparation and reused
for the full configuration. Reports preserve per-call tool names, redacted
arguments, attempts, latency, source metadata, and errors. Aggregate adapter
metrics include connection time, cold and warm tool latency, calls, retries,
timeouts, empty responses, partial completions, and failure rate.

When an experiment contains both `internal` and `mcp`, fairness validation is
enabled by default. `MCP_CORPUS_PATH` must resolve to the same corpus as
`dataset.corpus_path`, and context-mode comparisons must share the dataset,
top-k, generator, prompt, and token limit. Set `MCP_ENFORCE_FAIRNESS=false` only
for intentionally asymmetric experiments.

## Important Environment Variables

YAML-first runs should set `BENCHMARK_CONFIG_FILE` and keep secrets/service URLs
in `.env`. Workflow fields such as datasets, models, retrieval, chunking, prompt
templates, vector backend, and evaluator settings belong in `experiments/*.yaml`.

| Variable | Description |
| --- | --- |
| `BENCHMARK_CONFIG_FILE` | Optional JSON/YAML manifest for `python main.py`; falls back to legacy `.env` matrix when unset. |
| `RAG_SYSTEM_ADAPTER` | `internal`, `http`, or `mcp`; defaults to `internal`. |
| `RAG_HTTP_ENDPOINT_URL` | Required when `RAG_SYSTEM_ADAPTER=http`. |
| `RAG_HTTP_TIMEOUT_SECONDS` | HTTP request timeout; defaults to `60`. |
| `RAG_HTTP_ANSWER_FIELD` | Dotted response path for the answer; defaults to `answer`. |
| `RAG_HTTP_CONTEXTS_FIELD` | Dotted response path for contexts; defaults to `contexts`. |
| `RAG_HTTP_METADATA_FIELD` | Dotted response path for retrieval metadata; defaults to `metadata`. |
| `RAG_HTTP_TIMINGS_FIELD` | Dotted response path for timing data; defaults to `timings`. |
| `RAG_ADAPTER_MODULES` | Optional comma-separated Python modules to import before RAG adapter validation. |
| `MCP_TRANSPORT` | `stdio` or `streamable_http`; defaults to `stdio`. |
| `MCP_COMMAND` / `MCP_ARGS_JSON` | Executable and JSON argument array for a stdio MCP server. |
| `MCP_SERVER_URL` | MCP endpoint required by `streamable_http`. |
| `MCP_TOOL_NAME` | Tool invoked for each benchmark question. |
| `MCP_RESULT_MODE` | `context` (MCP evidence + framework LLM) or `answer` (tool is full QA system). |
| `MCP_RESULT_FIELD` | Optional dotted path inside structured tool output, such as `contexts`. |
| `MCP_EXECUTION_MODE` | `fixed` or `agentic`; defaults to `fixed`. |
| `MCP_ALLOWED_TOOLS_JSON` | Optional JSON tool-name allowlist for agentic mode. |
| `MCP_MAX_AGENT_ROUNDS` | Maximum model/tool rounds; defaults to `4`. |
| `MCP_MAX_RETRIES` | Transport exception retries per tool call; defaults to `1`. |
| `MCP_CONTINUE_ON_ERROR` | Record a failed sample and continue instead of aborting; defaults to `true`. |
| `MCP_CORPUS_PATH` | Declares the MCP corpus for RAG-vs-MCP fairness validation. |
| `DATASET_NAME` | Dataset adapter to benchmark; built-ins include `jsonl` and `csv` for local files. |
| `DATASET_PATH` | Required for `DATASET_NAME=jsonl` or `csv`. |
| `DATASET_QUESTION_FIELD` | Local dataset question field; defaults to `question`. |
| `DATASET_GROUND_TRUTH_FIELD` | Local dataset answer field; defaults to `ground_truth`. |
| `DATASET_CONTEXT_FIELD` | Local dataset context field; defaults to `context`. |
| `DATASET_METADATA_FIELD` | Local dataset metadata field; defaults to `metadata`. |
| `DATASET_SUBSET` | Optional dataset subset/config. |
| `DATASET_SAMPLE_SIZE` | Number of benchmark samples. |
| `RAGAS_ENABLED` | Set to `false` to skip RAGAS critic metrics. |
| `CUSTOM_METRICS_ENABLED` | Set to `false` to skip custom embedding/BERTScore metrics. |
| `EVAL_CRITIC_LLM` | Critic model used for RAGAS evaluation. |
| `EVAL_CRITIC_EMBEDDING` | Embedding model used by evaluator metrics. |
| `ELECTRICITY_PRICE_EUR_PER_KWH` | Electricity price (€/kWh) for local energy-cost estimates. Falls back to `ELECTRICITY_PRICE_USD_PER_KWH`. |
| `ELECTRICITY_PRICE_USD_PER_KWH` | Electricity price (USD/kWh) for local energy-cost estimates. |
| `BENCHMARK_RESOURCE_MONITOR` | `true` to sample GPU power/CPU/mem to CSV traces (enables per-second `gpu_power_w`). |

`BENCHMARK_STAGE=index` is only supported by the built-in adapter. External HTTP
systems own their own indexing lifecycle.

## Real-Time Dashboard

Watch a benchmark sweep live, or explore past runs, in a Streamlit dashboard:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
# optional: point at a custom results directory
DASHBOARD_RESULTS_DIR=/path/to/results streamlit run dashboard/app.py
```

Tabs:

- **Übersicht** — all `results/runN/` directories with status, dataset, and
  config progress.
- **Live-Monitor** — auto-refreshing view of a running worker. Reads
  `progress.json`, per-config QA logs under `configs/`, and stage timing
  files as they are written, so completed configs appear before the sweep
  finishes. Toggle the refresh interval and Auto-Refresh in the sidebar.
- **Vergleichen** — pick runs and compare RAGAS, custom/TRACe, performance,
  and LLM-load-test metrics as tables, bar charts, per-sample box plots,
  correlation heatmaps, and quality-vs-latency scatter plots.
- **Run-Detail** — metrics, stage latencies, LLM performance, resource
  traces (`resource_traces/*.csv`), per-sample answers, and raw JSON for a
  single config.

Data is read only from `results/`; the dashboard never writes or alters run
artifacts. It parses the same aggregate JSON, QA logs, progress ledger, and
stage timings that the worker and `main.py` produce.

## MLflow Run Comparison

Start the MLflow UI against the local SQLite store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

In the `RAG-Benchmark` experiment you can compare per-config runs as a table by selecting visible params, metrics, and tags. Each benchmark sweep also logs an aggregate summary run named like `summary_runN_<timestamp>` with:

- `tables/`: `results_summary.csv` and `results_per_sample.csv`
- `reports/`: JSON and Markdown reports
- `plots/`: generated PNG and interactive HTML plots
- `reproducibility/`: manifest and package freeze

Use the aggregate run when you want all tables and plots for one sweep in one place.

## Output

Each run writes artifacts to `results/runN/`, including per-config JSON files,
QA logs, CSV/Markdown summaries, plots, reproducibility manifests, and MLflow run data.
