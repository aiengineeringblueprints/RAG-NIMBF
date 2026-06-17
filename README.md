# RAG Benchmarking Framework

This project benchmarks RAG systems across datasets, retrieval outputs, answer
quality metrics, runtime measurements, reports, and MLflow tracking.

The framework can run its built-in RAG pipeline, or evaluate an external RAG
system as a black-box HTTP service.

## Install

```bash
pip install -r requirements.txt
```

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

## Important Environment Variables

YAML-first runs should set `BENCHMARK_CONFIG_FILE` and keep secrets/service URLs
in `.env`. Workflow fields such as datasets, models, retrieval, chunking, prompt
templates, vector backend, and evaluator settings belong in `experiments/*.yaml`.

| Variable | Description |
| --- | --- |
| `BENCHMARK_CONFIG_FILE` | Optional JSON/YAML manifest for `python main.py`; falls back to legacy `.env` matrix when unset. |
| `RAG_SYSTEM_ADAPTER` | `internal` or `http`; defaults to `internal`. |
| `RAG_HTTP_ENDPOINT_URL` | Required when `RAG_SYSTEM_ADAPTER=http`. |
| `RAG_HTTP_TIMEOUT_SECONDS` | HTTP request timeout; defaults to `60`. |
| `RAG_HTTP_ANSWER_FIELD` | Dotted response path for the answer; defaults to `answer`. |
| `RAG_HTTP_CONTEXTS_FIELD` | Dotted response path for contexts; defaults to `contexts`. |
| `RAG_HTTP_METADATA_FIELD` | Dotted response path for retrieval metadata; defaults to `metadata`. |
| `RAG_HTTP_TIMINGS_FIELD` | Dotted response path for timing data; defaults to `timings`. |
| `RAG_ADAPTER_MODULES` | Optional comma-separated Python modules to import before RAG adapter validation. |
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

`BENCHMARK_STAGE=index` is only supported by the built-in adapter. External HTTP
systems own their own indexing lifecycle.

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

