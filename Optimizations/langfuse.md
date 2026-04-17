# LangFuse Tracing Integration

## What It Does

LangFuse adds **trace-level observability** to the benchmarking pipeline. While MLflow tracks aggregated metrics per config (avg TTFT, RAGAS scores, etc.), LangFuse shows the **full call hierarchy** of each benchmark run — which functions were called, in what order, what inputs/outputs they had, and how long each step took.

### Trace Hierarchy

A single benchmark config produces this trace tree in the LangFuse UI:

```
run_single_benchmark (trace)
├── chunk_documents              # @observe
├── ChatOllama.stream            # auto via LangChain callback
├── process_question (x N)
│   ├── ChatOllama.invoke (HyDE) # auto via callback
│   ├── Chroma.similarity_search # auto via callback
│   ├── rerank                   # @observe
│   └── ChatOllama.stream        # auto via callback
├── ragas_evaluation             # @observe
│   └── [inner RAGAS LLM calls]  # auto via callback
└── custom_metrics               # @observe
```

Each span captures: inputs, outputs, timing, and metadata. LLM spans also show token counts.

## How It Works

### Architecture

The integration has two layers:

1. **`@observe` decorators** — applied to non-LangChain functions (`chunk_documents`, `evaluate_results`, `compute_custom_metrics`, `rerank`). These create trace spans automatically when the function is called.

2. **LangChain `CallbackHandler`** — a callback passed to LangChain operations (`llm.stream()`, `llm.invoke()`, `vector_store.similarity_search()`). LangChain fires callback events that LangFuse captures automatically, creating nested spans for LLM calls, embedding operations, and retriever lookups.

### Key Files

| File | Role |
|------|------|
| `benchmark/tracing.py` | Init — reads env vars, logs whether tracing is enabled |
| `benchmark/chunking.py` | `@observe` on `chunk_documents()` |
| `benchmark/evaluation.py` | `@observe` on `evaluate_results()` |
| `benchmark/custom_metrics.py` | `@observe` on `compute_custom_metrics()` |
| `benchmark/reranker.py` | `@observe` on `CrossEncoderReranker.rerank()` |
| `benchmark/generation.py` | `callbacks` param threaded to `llm.stream()` / `llm.invoke()` |
| `benchmark/retrieval.py` | `callbacks` param on `retrieve()` and `expand_query_with_hyde()` |
| `main.py` | Creates `CallbackHandler` per run, passes to LangChain calls, flushes on exit |

### Graceful Degradation

When `LANGFUSE_PUBLIC_KEY` is not set in `.env`:
- `setup_langfuse()` returns `None` — no handler is created
- `@observe` decorators become pass-through no-ops
- All `callbacks` params are `None` — no overhead
- The benchmark runs exactly as before

MLflow is completely unaffected — the two systems are independent.

## Prerequisites

### 1. Docker (for LangFuse server)

LangFuse runs as a local Docker container with PostgreSQL.

```bash
# Verify Docker is running
docker info
```

### 2. LangFuse Python package

Already added to `requirements.txt`:
```
langfuse>=2.0
```

Install with:
```bash
pip install -r requirements.txt
```

### 3. LangFuse Server

Clone and start the official LangFuse server:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Wait ~2-3 minutes for the `langfuse-web-1` container to log "Ready".

Access the UI at **http://localhost:3000**.

### 4. Create a Project and Get API Keys

1. Open http://localhost:3000 in your browser
2. Create a new project (e.g. "RAG-Benchmark")
3. Go to **Settings > API Keys**
4. Copy the **Public Key** and **Secret Key**

### 5. Configure Environment Variables

Add these to your `.env` file:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## Usage

### Run a Benchmark

Just run the framework as usual:

```bash
python main.py
```

If the LangFuse env vars are set, traces are sent automatically. You'll see:

```
RAG Benchmarking Framework
==================================================
MLflow tracking: mlruns
LangFuse tracing: http://localhost:3000
```

### View Traces

Open **http://localhost:3000**, navigate to your project, and you'll see one trace per benchmark config. Click a trace to see the nested spans with timing and I/O.

### Disable Tracing

Comment out or remove the env vars in `.env`:

```bash
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
# LANGFUSE_HOST=http://localhost:3000
```

The framework runs without any tracing overhead.

### Stop the LangFuse Server

```bash
cd langfuse
docker compose down
```

Data persists in Docker volumes. To also remove the data, use `docker compose down -v`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No traces appear in UI | Check that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` match the keys from the LangFuse project |
| Connection refused | Verify the LangFuse container is running: `docker ps` |
| Traces are incomplete | The framework calls `Langfuse().flush()` at exit. If the process is killed, traces may be lost |
| Import errors | Run `pip install langfuse>=2.0` — langfuse v4 uses `from langfuse import observe` and `from langfuse.langchain import CallbackHandler` |
