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
python main.py
```

The built-in mode chunks/indexes the selected dataset, retrieves contexts,
generates answers, evaluates the results, writes reports under `results/`, and
logs to MLflow.

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

| Variable | Description |
| --- | --- |
| `RAG_SYSTEM_ADAPTER` | `internal` or `http`; defaults to `internal`. |
| `RAG_HTTP_ENDPOINT_URL` | Required when `RAG_SYSTEM_ADAPTER=http`. |
| `RAG_HTTP_TIMEOUT_SECONDS` | HTTP request timeout; defaults to `60`. |
| `RAG_HTTP_ANSWER_FIELD` | Dotted response path for the answer; defaults to `answer`. |
| `RAG_HTTP_CONTEXTS_FIELD` | Dotted response path for contexts; defaults to `contexts`. |
| `RAG_HTTP_METADATA_FIELD` | Dotted response path for retrieval metadata; defaults to `metadata`. |
| `RAG_HTTP_TIMINGS_FIELD` | Dotted response path for timing data; defaults to `timings`. |
| `DATASET_NAME` | Dataset adapter to benchmark. |
| `DATASET_SUBSET` | Optional dataset subset/config. |
| `DATASET_SAMPLE_SIZE` | Number of benchmark samples. |
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

