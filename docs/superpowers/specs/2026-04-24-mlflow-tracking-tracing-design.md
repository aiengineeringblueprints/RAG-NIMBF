# MLflow Tracking, Tracing & GenAI Eval-Monitor Integration

**Date**: 2026-04-24
**Status**: Approved

## Goal

Replace LangFuse with MLflow as the single observability backend. Backfill existing EVAL_MATRIX runs into MLflow. Enable MLflow tracing (OpenTelemetry-based) for pipeline spans. Add GenAI eval-monitor dashboards. Set up MLflow MCP for Claude to query experiments interactively.

## Section 1: MLflow Tracking Upgrade

### Changes to `benchmark/tracking.py`

- Default tracking URI: `http://localhost:5000` (local MLflow server with file store backend). Falls back to `sqlite:///mlflow.db` if server unreachable.
- Structured run names: `{LLM}_{Chunking}_cs{size}_co{overlap}_{Retrieval}_{Template}`
- Additional tags: `retrieval_strategy`, `retrieval_top_k`, `dataset_name`, `reranker_model`
- Additional params: `retrieval_top_k`, `retrieval_strategy`, `dataset_name`, `dataset_sample_size`
- Dataset metadata logged via `mlflow.log_input()`

### MLflow Server Startup

A simple command (documented, not scripted):
```
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

### Dependencies

No new packages needed for tracking — `mlflow>=2.18` is already in requirements.txt.

---

## Section 2: MLflow Tracing (Replacing LangFuse)

### Remove LangFuse

- Delete LangFuse imports and callback handlers from `main.py`
- Remove `langfuse>=2.0` from `requirements.txt`
- Remove `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_HOST` env var references

### Rewrite `benchmark/tracing.py`

Replace with MLflow tracing module:

- `setup_tracing()` — initializes MLflow tracing, optionally configures OTLP exporter if `OTEL_EXPORTER_OTLP_ENDPOINT` is set
- `@mlflow.trace` decorator on pipeline functions:
  - `chunk_documents` — span with chunk_size, chunk_overlap, strategy attributes
  - `build_vector_store` — span with embedding_model, num_chunks attributes
  - `retrieve` — span with top_k, strategy, num_results attributes
  - `generate_answer` — span with ttft, token_count, model attributes
  - `evaluate_results` — span with critic_model attribute
  - `compute_custom_metrics` — span listing computed metric names
- Each `run_single_benchmark` call is a parent trace
- Per-sample Q&A pairs get individual traces with child retrieval + generation spans

### OpenTelemetry Integration

- MLflow tracing wraps OpenTelemetry internally
- Optional OTLP exporter via `OTEL_EXPORTER_OTLP_ENDPOINT` env var for exporting to Jaeger, Grafana Tempo, etc.
- Add `opentelemetry-api` and `opentelemetry-sdk` explicitly to requirements.txt

### Changes to `main.py`

- Replace `setup_langfuse()` call with `setup_tracing()`
- Remove LangFuse `CallbackHandler` creation and `_callbacks` passing
- Remove `Langfuse().flush()` at end
- Remove `langfuse` import

---

## Section 3: MLflow GenAI Eval-Monitor

### Approach

Additive layer on top of existing RAGAS + custom metrics. Does not replace computation.

### Implementation

After each benchmark run (in `log_benchmark_run` or a new `log_genai_eval()` function):

1. Build an eval dataset as a Pandas DataFrame with columns: `inputs` (questions), `predictions` (answers), `contexts` (retrieved), `ground_truth`
2. Call `mlflow.evaluate()` with:
   - `data` = the eval DataFrame
   - `evaluators` = `"default"` + custom evaluator configs
   - `evaluator_config` = model name for LLM-as-judge
3. Evaluators:
   - **Faithfulness**: maps to MLflow's built-in genai faithfulness judge
   - **Relevance**: custom evaluator using embedding cosine similarity
   - **Correctness**: LLM-as-judge comparing answer vs ground_truth on 1-5 scale
4. Results appear in MLflow UI Evaluate tab with comparison tables and trend charts

### Dependencies

- `mlflow[genai]` extra (adds eval dependencies)

---

## Section 4: Backfill Script + Enhanced Visualization

### Backfill Script (`backfill_mlflow.py`)

Location: project root, standalone script.

Algorithm:
1. Parse EVAL_MATRIX.md to extract all "Getestet" / "Test (N=50)" rows with their parameters
2. Scan `results/` directories for JSON files
3. Match each EVAL_MATRIX row to a result JSON by matching (LLM, Chunking, Size, Overlap, Retrieval, Template)
4. For each match, log as a completed MLflow run:
   - `mlflow.start_run(run_name=..., tags={..., "source": "backfill"})`
   - Set `start_time` to original run timestamp from JSON
   - Log all params, metrics, tags, per-sample CSV artifact
5. Print summary: "Imported X runs, Y skipped"

### Enhanced Visualization (`benchmark/reporting/visualization.py`)

Add interactive plotly charts (saved as HTML alongside existing PNGs):

1. **Metrics over time**: Line chart — x-axis = timestamp, y-axis = metric value, colored by LLM model. Separate lines for faithfulness, hit@k, ROUGE-L, BERTScore F1.
2. **Parameter impact heatmap**: Faithfulness as color value, axes = (chunk_size, overlap, retrieval_strategy). One heatmap per LLM.
3. **LLM comparison radar**: Radar chart overlaying all LLMs on axes = faithfulness, hit@1, ROUGE-L, METEOR, BERTScore F1, context_relevance.
4. **Scatter matrix**: Pairwise scatter of all custom metrics, colored by LLM model.

Existing matplotlib plots remain unchanged.

### Dependencies

- `plotly>=5.0` added to requirements.txt

---

## Section 5: MLflow MCP Server

### Configuration

Use MLflow's built-in MCP server (available since MLflow 3.4). Add to `.claude.json` project MCP servers:

```json
"mlflow-mcp": {
  "type": "stdio",
  "command": "mlflow",
  "args": ["mcp"],
  "env": {
    "MLFLOW_TRACKING_URI": "http://localhost:5000"
  }
}
```

Docs: https://mlflow.org/docs/latest/genai/mcp/

### Capabilities

The built-in MCP server exposes tools for Claude to:
- Search and browse experiments and runs
- Query traces and spans from MLflow tracing
- Compare metrics across runs
- Analyze genai evaluation results

### Dependencies

No extra package needed — built into `mlflow>=3.4`. Upgrade `mlflow` version in requirements.txt if needed.

---

## Files Changed (Summary)

| File | Action |
|------|--------|
| `benchmark/tracking.py` | Modify — upgrade tracking URI, add tags/params, add genai eval |
| `benchmark/tracing.py` | Rewrite — replace LangFuse with MLflow tracing + optional OTel OTLP |
| `main.py` | Modify — remove LangFuse, use new tracing setup |
| `benchmark/reporting/visualization.py` | Modify — add 4 interactive plotly chart types |
| `backfill_mlflow.py` | New — backfill EVAL_MATRIX runs into MLflow |
| `requirements.txt` | Modify — remove langfuse, add plotly, opentelemetry-api, opentelemetry-sdk, upgrade mlflow for built-in MCP |
| `.claude.json` (project) | Modify — add mlflow mcp server config |

## Out of Scope

- Changing RAGAS evaluation logic
- Modifying chunking/embedding/retrieval pipeline
- Databricks or remote MLflow deployment
- Full OTel collector setup (only optional OTLP exporter)
