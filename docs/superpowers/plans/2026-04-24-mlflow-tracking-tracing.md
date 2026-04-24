# MLflow Tracking, Tracing & GenAI Eval-Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LangFuse with MLflow as the single observability backend. Add MLflow tracing, GenAI eval-monitor, backfill existing runs, enhanced visualizations, and MCP integration.

**Architecture:** MLflow server runs locally on port 5000. `benchmark/tracing.py` is rewritten to use `@mlflow.trace` decorators (OpenTelemetry-based) instead of LangFuse. `benchmark/tracking.py` is upgraded with structured run names and additional tags. A backfill script imports existing EVAL_MATRIX runs. Plotly charts add interactive visualizations.

**Tech Stack:** MLflow >=3.4 (tracing + MCP), OpenTelemetry API/SDK, Plotly, Pandas

---

## File Structure

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Updated dependencies |
| `benchmark/tracing.py` | MLflow tracing setup + `@mlflow.trace` wrapper (replaces LangFuse) |
| `benchmark/tracking.py` | MLflow experiment tracking (upgraded URI, tags, params) |
| `main.py` | Remove LangFuse, use new tracing |
| `benchmark/retrieval.py` | Add `@mlflow.trace` to `retrieve` and `build_vector_store` |
| `benchmark/generation.py` | Add `@mlflow.trace` to `generate_answer` |
| `benchmark/chunking.py` | Replace `@observe` with `@mlflow.trace` |
| `benchmark/evaluation.py` | Replace `@observe` with `@mlflow.trace` |
| `benchmark/custom_metrics.py` | Replace `@observe` with `@mlflow.trace` |
| `benchmark/reporting/visualization.py` | Add 4 plotly interactive chart types |
| `backfill_mlflow.py` | Backfill EVAL_MATRIX runs into MLflow |

---

### Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace `langfuse>=2.0` and `mlflow>=2.18` lines. Add new dependencies.

Current content (lines 17-18):
```
mlflow>=2.18
langfuse>=2.0
```

Replace with:
```
mlflow>=3.4
opentelemetry-api>=1.20
opentelemetry-sdk>=1.20
opentelemetry-exporter-otlp>=1.20
plotly>=5.0
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully (mlflow 3.x, opentelemetry, plotly).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: upgrade mlflow to 3.4+, add opentelemetry and plotly, remove langfuse"
```

---

### Task 2: Rewrite `benchmark/tracing.py`

**Files:**
- Rewrite: `benchmark/tracing.py` (full replacement)

- [ ] **Step 1: Write the new tracing module**

Replace the entire contents of `benchmark/tracing.py` with:

```python
"""MLflow tracing — OpenTelemetry-based observability via MLflow.

When MLflow tracing is enabled, pipeline functions are traced with
parent-child span relationships. Optionally exports traces via OTLP
to an external OpenTelemetry collector.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import mlflow

logger = logging.getLogger(__name__)

_OTEL_EXPORTER_CONFIGURED = False


def setup_tracing() -> str | None:
    """Configure MLflow tracing and optional OTLP exporter.

    Returns the tracking URI if tracing is enabled, None otherwise.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try connecting to MLflow server; fall back to SQLite
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        sqlite_uri = "sqlite:///mlflow.db"
        logger.warning(
            "MLflow server at %s unreachable, falling back to %s",
            tracking_uri, sqlite_uri,
        )
        tracking_uri = sqlite_uri

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # Optional OTLP exporter for external OTel backends
    global _OTEL_EXPORTER_CONFIGURED
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint and not _OTEL_EXPORTER_CONFIGURED:
        _configure_otlp_exporter(otlp_endpoint)
        _OTEL_EXPORTER_CONFIGURED = True

    return tracking_uri


def _configure_otlp_exporter(endpoint: str) -> None:
    """Configure OpenTelemetry OTLP exporter alongside MLflow tracing."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        from mlflow.tracing.provider import _get_tracer

        resource = Resource.create({"service.name": "rag-benchmark"})
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)

        tracer = _get_tracer()
        if hasattr(tracer, "provider") and isinstance(tracer.provider, TracerProvider):
            tracer.provider.add_span_processor(processor)
            logger.info("OTLP exporter configured: %s", endpoint)
        else:
            logger.warning("Could not attach OTLP exporter to MLflow tracer")
    except ImportError:
        logger.warning(
            "opentelemetry-exporter-otlp not installed. "
            "Install with: pip install opentelemetry-exporter-otlp"
        )
    except Exception as e:
        logger.warning("Failed to configure OTLP exporter: %s", e)
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from benchmark.tracing import setup_tracing; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmark/tracing.py
git commit -m "refactor: replace LangFuse tracing with MLflow tracing + optional OTel OTLP"
```

---

### Task 3: Remove LangFuse from `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace imports (lines 27-28)**

Current:
```python
from benchmark.tracking import setup_mlflow, log_benchmark_run
from benchmark.tracing import setup_langfuse
```

Replace with:
```python
from benchmark.tracking import setup_mlflow, log_benchmark_run
from benchmark.tracing import setup_tracing
```

- [ ] **Step 2: Remove LangFuse callback handler setup in `run_single_benchmark` (lines 43-51)**

Remove these lines:
```python
    # Set up LangFuse callback handler for this benchmark run (one trace per config)
    _callbacks = None
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler
        _lf_handler = CallbackHandler(
            trace_name=config.name,
            metadata={"llm_model": config.llm_model, "chunking_strategy": config.chunking_strategy},
        )
        _callbacks = [_lf_handler]
```

- [ ] **Step 3: Remove `_callbacks` arguments from function calls**

In `run_single_benchmark`, remove `callbacks=_callbacks` from these calls:
- `expand_query_with_hyde(llm, sample["question"], callbacks=_callbacks)` → `expand_query_with_hyde(llm, sample["question"])`
- `retrieve(vector_store, query, ..., callbacks=_callbacks)` → remove `callbacks=_callbacks`
- `generate_answer(llm, ..., callbacks=_callbacks)` → remove `callbacks=_callbacks`

- [ ] **Step 4: Remove LangFuse flush in `run_all_benchmarks` (lines 430-433)**

Remove:
```python
    # Flush LangFuse traces before exit
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import Langfuse
        Langfuse().flush()
```

- [ ] **Step 5: Update `main()` function (lines 438-450)**

Current:
```python
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    console.print("[bold]RAG Benchmarking Framework[/bold]")
    console.print("=" * 50)
    tracking_uri = setup_mlflow()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    langfuse_host = setup_langfuse()
    if langfuse_host:
        console.print(f"[dim]LangFuse tracing: {langfuse_host}[/dim]")
    run_all_benchmarks()
```

Replace with:
```python
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    console.print("[bold]RAG Benchmarking Framework[/bold]")
    console.print("=" * 50)
    tracking_uri = setup_tracing()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    setup_mlflow()
    run_all_benchmarks()
```

- [ ] **Step 6: Verify `main.py` imports cleanly**

Run: `python -c "from main import main; print('OK')"`
Expected: `OK` (or import errors to fix)

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "refactor: remove LangFuse from main.py, use MLflow tracing"
```

---

### Task 4: Add `@mlflow.trace` decorators to pipeline functions

**Files:**
- Modify: `benchmark/chunking.py` (line 51 — replace `@observe`)
- Modify: `benchmark/retrieval.py` (lines 49, 110 — add decorators)
- Modify: `benchmark/generation.py` (line 434 — add decorator)
- Modify: `benchmark/evaluation.py` (line 32 — replace `@observe`)
- Modify: `benchmark/custom_metrics.py` (line 407 — replace `@observe`)

- [ ] **Step 1: Update `benchmark/chunking.py`**

Replace the import on the relevant line:
```python
from benchmark.tracing import observe
```
Remove this import entirely.

Replace decorator on `chunk_documents` (line 51):
```python
@observe(name="chunk_documents")
def chunk_documents(chunker, documents: list[dict], min_chunk_length: int = 50) -> list[Document]:
```
With:
```python
@mlflow.trace(name="chunk_documents", span_type="func")
def chunk_documents(chunker, documents: list[dict], min_chunk_length: int = 50) -> list[Document]:
```

Add import at top of file:
```python
import mlflow
```

- [ ] **Step 2: Update `benchmark/retrieval.py`**

Add import at top:
```python
import mlflow
```

Add decorator before `build_vector_store` (line 49):
```python
@mlflow.trace(name="build_vector_store", span_type="func", attributes={"component": "embedding"})
def build_vector_store(chunks: list[Document], embedding_model_name: str, collection_name: str, ollama_base_url: str = "http://localhost:11434", ollama_api_key: str | None = None, cache_key: str | None = None, *, embedding_provider: str = "ollama") -> Chroma:
```

Add decorator before `retrieve` (line 110):
```python
@mlflow.trace(name="retrieve", span_type="func")
def retrieve(
    vector_store: Chroma,
    query: str,
    top_k: int = 3,
    *,
    retrieval_strategy: str = "similarity",
    fetch_k: int | None = None,
    mmr_lambda: float = 0.5,
    callbacks: list | None = None,
) -> list[Document]:
```

Inside `retrieve`, after the docstring, add span attribute logging:
```python
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "retrieval.top_k": top_k,
            "retrieval.strategy": retrieval_strategy,
        })
```

- [ ] **Step 3: Update `benchmark/generation.py`**

Add import at top:
```python
import mlflow
```

Add decorator before `generate_answer` (line 434):
```python
@mlflow.trace(name="generate_answer", span_type="func")
def generate_answer(
    llm: BaseChatModel,
    question: str,
    contexts: list[str],
    *,
    system_prompt: str = (
        "Answer the question using ONLY the provided context. "
        "Return ONLY the raw value — a number, percentage, ratio, or yes/no. "
        "Do NOT include units, explanations, reasoning, or full sentences. "
        "Examples: 494.0 | 0.12 | -0.46 | 1 | 5.8"
    ),
    human_template: str = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    strip_mode: AnswerStripMode = "tags_only",
    value_fallback: bool = True,
    ground_truth: str | None = None,
    prompt_template_name: str | None = None,
    callbacks: list | None = None,
) -> GenerationResult:
```

Inside `generate_answer`, after the first line (`from langchain_core.messages import HumanMessage, SystemMessage`), add:
```python
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "generation.prompt_template": prompt_template_name or "default",
        })
```

- [ ] **Step 4: Update `benchmark/evaluation.py`**

Replace import:
```python
from benchmark.tracing import observe
```
Remove this import.

Replace decorator on `evaluate_results` (line 32):
```python
@observe(name="ragas_evaluation")
```
With:
```python
@mlflow.trace(name="ragas_evaluation", span_type="func")
```

Add import at top:
```python
import mlflow
```

Inside `evaluate_results`, after the early-return check, add:
```python
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "evaluation.critic_model": critic_llm_model or llm_model,
        })
```

- [ ] **Step 5: Update `benchmark/custom_metrics.py`**

Replace import:
```python
from benchmark.tracing import observe
```
Remove this import.

Replace decorator on `compute_custom_metrics` (line 407):
```python
@observe(name="custom_metrics")
```
With:
```python
@mlflow.trace(name="custom_metrics", span_type="func")
```

Add import at top:
```python
import mlflow
```

Inside `compute_custom_metrics`, after the docstring, add:
```python
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "metrics.num_questions": len(questions),
        })
```

- [ ] **Step 6: Verify all modules import cleanly**

Run:
```bash
python -c "
from benchmark.chunking import chunk_documents
from benchmark.retrieval import build_vector_store, retrieve
from benchmark.generation import generate_answer
from benchmark.evaluation import evaluate_results
from benchmark.custom_metrics import compute_custom_metrics
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 7: Commit**

```bash
git add benchmark/chunking.py benchmark/retrieval.py benchmark/generation.py benchmark/evaluation.py benchmark/custom_metrics.py
git commit -m "feat: add @mlflow.trace decorators to all pipeline functions"
```

---

### Task 5: Upgrade `benchmark/tracking.py`

**Files:**
- Modify: `benchmark/tracking.py`

- [ ] **Step 1: Update `setup_mlflow()` function (lines 16-26)**

Replace:
```python
def setup_mlflow() -> str:
    """Configure MLflow tracking URI and return it.

    By default, uses an SQLite database (``mlflow.db``) in the project root
    so all runs are persisted in a single queryable file.  Override via the
    ``MLFLOW_TRACKING_URI`` environment variable if needed.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    return tracking_uri
```

With:
```python
def setup_mlflow() -> str:
    """Configure MLflow experiment settings after tracing is initialized.

    Sets the experiment name to RAG-Benchmark. Called after setup_tracing()
    which handles the tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try server connection, fall back to SQLite
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("RAG-Benchmark")
    logger.info("MLflow experiment: RAG-Benchmark (URI: %s)", tracking_uri)
    return tracking_uri
```

- [ ] **Step 2: Add structured run name helper (after `_flatten_ragas_stats`)**

Add this function after the `_flatten_ragas_stats` function (after line 53):

```python
def _make_run_name(result: BenchmarkResultExtended) -> str:
    """Generate a structured run name from config parameters."""
    llm = result.llm_model.split("/")[-1].replace(":", "_")
    return f"{llm}_{result.chunking_strategy}_cs{result.chunk_size}_co{result.chunk_overlap}"


def _make_tags(result: BenchmarkResultExtended) -> dict[str, str]:
    """Build MLflow tags from benchmark result."""
    tags: dict[str, str] = {
        "llm_model": result.llm_model,
        "embedding_model": result.embedding_model,
        "chunking_strategy": result.chunking_strategy,
        "prompt_template": result.prompt_template,
    }
    if result.reranker_model:
        tags["reranker_model"] = result.reranker_model
    return tags
```

- [ ] **Step 3: Update `log_benchmark_run` to use new helpers and add params/tags**

In `log_benchmark_run` (line 56), replace the tags/params construction and the `with mlflow.start_run(...)` block.

Replace the tags block (lines 67-74):
```python
    tags: dict[str, str] = {
        "llm_model": result.llm_model,
        "embedding_model": result.embedding_model,
        "chunking_strategy": result.chunking_strategy,
        "prompt_template": result.prompt_template,
    }
    if result.reranker_model:
        tags["reranker_model"] = result.reranker_model
```

With:
```python
    tags = _make_tags(result)
```

Replace the params block (lines 76-83):
```python
    params: dict[str, Any] = {
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "num_chunks": result.num_chunks,
        "num_questions": result.num_questions,
    }
    if result.reranker_top_k is not None:
        params["reranker_top_k"] = result.reranker_top_k
```

With:
```python
    params: dict[str, Any] = {
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "num_chunks": result.num_chunks,
        "num_questions": result.num_questions,
    }
    if result.reranker_top_k is not None:
        params["reranker_top_k"] = result.reranker_top_k
```

(This stays the same — just adding tags via `_make_tags`.)

Replace the `with mlflow.start_run(...)` line (line 147):
```python
    with mlflow.start_run(run_name=result.config_name, tags=tags) as run:
```

With:
```python
    run_name = _make_run_name(result)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
```

- [ ] **Step 4: Verify tracking module imports**

Run: `python -c "from benchmark.tracking import setup_mlflow, log_benchmark_run; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add benchmark/tracking.py
git commit -m "feat: upgrade MLflow tracking with structured run names and server fallback"
```

---

### Task 6: Add GenAI Eval-Monitor to tracking

**Files:**
- Modify: `benchmark/tracking.py` (add `log_genai_eval` function)

- [ ] **Step 1: Add `log_genai_eval` function at end of `tracking.py`**

Add this function at the end of the file (after `_log_per_sample_csv`):

```python
def log_genai_eval(result: BenchmarkResultExtended) -> None:
    """Log a GenAI evaluation dataset for the MLflow eval-monitor dashboard.

    Creates an eval DataFrame from per-sample results and logs it via
    mlflow.evaluate() with built-in scorers. This is additive — it does
    not replace the RAGAS or custom metrics computation.
    """
    if not result.per_sample:
        logger.warning("No per-sample data — skipping GenAI eval logging")
        return

    try:
        import pandas as pd

        eval_data = pd.DataFrame({
            "inputs": [s.question for s in result.per_sample],
            "predictions": [s.answer for s in result.per_sample],
            "contexts": [list(s.contexts) for s in result.per_sample],
            "ground_truth": [s.ground_truth for s in result.per_sample],
        })

        with mlflow.start_run(run_name=_make_run_name(result) + "_eval", tags=_make_tags(result)):
            eval_result = mlflow.evaluate(
                data=eval_data,
                targets="ground_truth",
                predictions="predictions",
                evaluators="default",
            )
            logger.info(
                "GenAI eval logged: %s metrics", len(eval_result.metrics)
            )
    except Exception as e:
        logger.warning("GenAI eval logging failed (non-fatal): %s", e)
```

- [ ] **Step 2: Wire `log_genai_eval` into `main.py`**

In `main.py`, add import:
```python
from benchmark.tracking import setup_mlflow, log_benchmark_run, log_genai_eval
```

In `run_all_benchmarks`, after `log_benchmark_run(result)` (line 408), add:
```python
        log_genai_eval(result)
```

- [ ] **Step 3: Verify imports**

Run: `python -c "from benchmark.tracking import log_genai_eval; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add benchmark/tracking.py main.py
git commit -m "feat: add GenAI eval-monitor logging via mlflow.evaluate()"
```

---

### Task 7: Backfill Script

**Files:**
- Create: `backfill_mlflow.py`

- [ ] **Step 1: Write the backfill script**

Create `backfill_mlflow.py` in the project root:

```python
#!/usr/bin/env python3
"""Backfill EVAL_MATRIX results into MLflow.

Scans results/ directories, matches to EVAL_MATRIX rows by config parameters,
and logs each as a completed MLflow run with original timestamps.
"""
import json
import re
import sys
from pathlib import Path

import mlflow

# Map EVAL_MATRIX LLM names to result JSON model names
LLM_ALIASES = {
    "Qwen3-32B": ["Qwen/Qwen3-32B-AWQ", "Qwen3-32B"],
    "Qwen3.5-397B": ["Qwen3.5-397B", "Sigurd_Qwen3.5-397B", "Sigurd_Qwne3.5-397B"],
    "GPT-OSS-20B": ["GPT-OSS-20B", "gptoss20b", "SPARK_GPToss:20b", "SPARK-gptoss20"],
    "Qwen3.5-35B": ["Qwen3.5-35B"],
}

# Map EVAL_MATRIX retrieval names
RETRIEVAL_ALIASES = {
    "Similarity": "similarity",
    "MMR": "mmr",
}

# Map EVAL_MATRIX chunking names
CHUNKING_ALIASES = {
    "Recursive": "recursive",
    "Semantic": "semantic",
}

# Map EVAL_MATRIX template names
TEMPLATE_ALIASES = {
    "Concise": "concise",
    "Detailed": "detailed",
}


def parse_eval_matrix(path: Path) -> list[dict]:
    """Parse EVAL_MATRIX.md table rows into dicts."""
    content = path.read_text()
    lines = content.split("\n")

    # Find the main matrix table header
    header_idx = None
    for i, line in enumerate(lines):
        if "| #" in line and "LLM" in line and "Chunking" in line:
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: Could not find EVAL_MATRIX table header")
        return []

    # Parse header
    headers = [h.strip() for h in lines[header_idx].split("|")[1:-1]]
    rows = []
    for line in lines[header_idx + 2:]:
        if not line.strip() or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < len(headers):
            continue
        row = dict(zip(headers, cells))
        status = row.get("Status", "").strip()
        if status in ("Getestet", "Test (N=50)"):
            rows.append(row)

    return rows


def load_result_jsons(results_dir: Path) -> list[tuple[Path, dict]]:
    """Load all benchmark JSON files from results/ subdirectories."""
    results = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in subdir.glob("benchmark_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                results.append((json_file, data))
            except Exception as e:
                print(f"  Warning: Could not load {json_file}: {e}")
    return results


def match_result_to_row(row: dict, json_data: dict) -> bool:
    """Check if a result JSON matches an EVAL_MATRIX row."""
    for result in json_data.get("results", []):
        # Try to match by numeric parameters
        try:
            if int(result.get("chunk_size", 0)) != int(row.get("Size", 0)):
                continue
            if int(result.get("chunk_overlap", 0)) != int(row.get("Overlap", 0)):
                continue

            # Match chunking strategy
            chunking = row.get("Chunking", "")
            result_chunking = result.get("chunking_strategy", "")
            if CHUNKING_ALIASES.get(chunking, chunking.lower()) != result_chunking:
                continue

            # Match template
            template = row.get("Template", "")
            result_template = result.get("prompt_template", "")
            if TEMPLATE_ALIASES.get(template, template.lower()) != result_template:
                continue

            # Match retrieval
            retrieval = row.get("Retrieval", "")
            # Retrieval strategy is encoded in config_name
            config_name = result.get("config_name", "").lower()
            retrieval_lower = RETRIEVAL_ALIASES.get(retrieval, retrieval.lower())
            if retrieval_lower == "mmr" and "mmr" not in config_name:
                continue
            if retrieval_lower == "similarity" and "mmr" in config_name:
                continue

            return True
        except (ValueError, TypeError):
            continue
    return False


def log_backfill_run(row: dict, json_path: Path, json_data: dict) -> bool:
    """Log a single backfill run to MLflow."""
    # Extract the first matching result config
    result = None
    for r in json_data.get("results", []):
        if match_result_to_row(row, {"results": [r]}):
            result = r
            break

    if result is None:
        return False

    # Build run name
    llm_name = row.get("LLM", "unknown").split("/")[-1]
    run_name = f"{llm_name}_{row.get('Chunking', '?')}_cs{row.get('Size', '?')}_co{row.get('Overlap', '?')}"

    # Extract timestamp from filename
    ts_match = re.search(r"benchmark_(\d{8})_(\d{6})", json_path.name)
    start_time_ms = None
    if ts_match:
        from datetime import datetime
        dt = datetime.strptime(f"{ts_match.group(1)}_{ts_match.group(2)}", "%Y%m%d_%H%M%S")
        start_time_ms = int(dt.timestamp() * 1000)

    tags = {
        "llm_model": result.get("llm_model", ""),
        "embedding_model": result.get("embedding_model", ""),
        "chunking_strategy": result.get("chunking_strategy", ""),
        "prompt_template": result.get("prompt_template", ""),
        "source": "backfill",
    }

    params = {
        "chunk_size": result.get("chunk_size", 0),
        "chunk_overlap": result.get("chunk_overlap", 0),
        "num_chunks": result.get("num_chunks", 0),
        "num_questions": result.get("num_questions", 0),
    }

    metrics = {}
    # RAGAS metrics
    for key in ["ragas_faithfulness"]:
        val = result.get("faithfulness")
        if val is not None:
            metrics["ragas_faithfulness"] = val

    # Stats
    stats = result.get("stats", {})
    for metric_name, stat_data in stats.items():
        if stat_data and isinstance(stat_data, dict):
            for stat_key in ["mean", "std", "median", "min", "max"]:
                val = stat_data.get(stat_key)
                if val is not None:
                    safe_name = metric_name.replace("@", "_at_")
                    metrics[f"ragas_{safe_name}_{stat_key}"] = val

    # Custom metrics (top-level in result JSON)
    custom_keys = [
        "hit@1", "hit@3", "hit@5", "ndcg@1", "ndcg@3", "ndcg@5",
        "recall@1", "recall@3", "recall@5", "context_relevance",
        "rouge_l", "bleu", "meteor",
        "bert_score_precision", "bert_score_recall", "bert_score_f1",
    ]
    for key in custom_keys:
        val = result.get(key)
        if val is not None:
            safe = key.replace("@", "_at_")
            metrics[f"custom_{safe}"] = val

    # Custom stats
    custom_stats = result.get("custom_stats", {})
    if custom_stats and isinstance(custom_stats, dict):
        for metric_name, stat_data in custom_stats.items():
            if stat_data and isinstance(stat_data, dict):
                safe = metric_name.replace("@", "_at_")
                for stat_key in ["mean", "std", "median", "min", "max"]:
                    val = stat_data.get(stat_key)
                    if val is not None:
                        metrics[f"custom_{safe}_{stat_key}"] = val

    # Check for existing run with same name and source=backfill
    existing = mlflow.search_runs(
        filter_string=f"tags.source = 'backfill' AND tags.run_name = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ALL,
    )
    if not existing.empty:
        print(f"  SKIP (already imported): {run_name}")
        return False

    with mlflow.start_run(
        run_name=run_name,
        tags=tags,
        start_time=start_time_ms,
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log per-sample as artifact
        per_sample = result.get("per_sample", [])
        if per_sample:
            import csv
            import tempfile

            tmpdir = Path(tempfile.mkdtemp())
            csv_path = tmpdir / "per_sample_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["question", "ground_truth", "answer"])
                for s in per_sample:
                    writer.writerow([
                        s.get("question", ""),
                        s.get("ground_truth", ""),
                        s.get("answer", ""),
                    ])
            mlflow.log_artifact(str(csv_path), artifact_path="data")

    print(f"  IMPORTED: {run_name} ({len(metrics)} metrics)")
    return True


def main():
    project_root = Path(__file__).parent
    eval_matrix_path = project_root / "EVAL_MATRIX.md"
    results_dir = project_root / "results"

    if not eval_matrix_path.exists():
        print(f"ERROR: {eval_matrix_path} not found")
        sys.exit(1)

    # Setup MLflow
    tracking_uri = "http://localhost:5000"
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        tracking_uri = "sqlite:///mlflow.db"
        print(f"MLflow server unreachable, using SQLite: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("RAG-Benchmark")

    print("Parsing EVAL_MATRIX...")
    rows = parse_eval_matrix(eval_matrix_path)
    print(f"Found {len(rows)} tested rows")

    print("Loading result JSONs...")
    result_jsons = load_result_jsons(results_dir)
    print(f"Found {len(result_jsons)} result files")

    imported = 0
    skipped = 0

    for row in rows:
        row_num = row.get("#", "?")
        llm = row.get("LLM", "")
        chunking = row.get("Chunking", "")
        size = row.get("Size", "")
        overlap = row.get("Overlap", "")
        retrieval = row.get("Retrieval", "")
        template = row.get("Template", "")
        faith = row.get("Faith", "-")

        if faith == "-" or faith == "":
            print(f"  SKIP row {row_num}: no faithfulness value")
            skipped += 1
            continue

        # Try to find matching result JSON
        matched = False
        for json_path, json_data in result_jsons:
            if match_result_to_row(row, json_data):
                if log_backfill_run(row, json_path, json_data):
                    imported += 1
                else:
                    skipped += 1
                matched = True
                break

        if not matched:
            print(f"  SKIP row {row_num} ({llm} {chunking} cs{size}): no matching result file")
            skipped += 1

    print(f"\nDone. Imported: {imported}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the backfill script in dry-run mode**

Run: `python backfill_mlflow.py`
Expected: Prints parsing results and import summary. Any MLflow connection errors are handled gracefully.

- [ ] **Step 3: Commit**

```bash
git add backfill_mlflow.py
git commit -m "feat: add backfill script to import EVAL_MATRIX runs into MLflow"
```

---

### Task 8: Add Interactive Plotly Visualizations

**Files:**
- Modify: `benchmark/reporting/visualization.py`

- [ ] **Step 1: Add plotly imports at top of `visualization.py`**

After the existing imports, add:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
```

- [ ] **Step 2: Add `_plot_metrics_over_time_html` function**

Add this function before the existing `_plot_vector_distance_scatter` function:

```python
def _plot_metrics_over_time_html(results: list, output_dir: Path) -> str | None:
    """Interactive metrics-over-time line chart saved as HTML."""
    if len(results) < 2:
        return None

    metric_keys = ["ragas_faithfulness", "custom_hit_at_1", "custom_rouge_l", "custom_bert_score_f1"]
    metric_labels = ["Faithfulness", "Hit@1", "ROUGE-L", "BERTScore F1"]

    # Collect data per result
    records = []
    for r in results:
        rec = {
            "config": r.config_name,
            "llm": r.llm_model.split("/")[-1],
            "chunking": r.chunking_strategy,
        }
        if r.ragas_faithfulness is not None:
            rec["ragas_faithfulness"] = r.ragas_faithfulness
        if r.custom_metric_means:
            for k in ["hit@1", "rouge_l", "bert_score_f1"]:
                if k in r.custom_metric_means:
                    safe = k.replace("@", "_at_")
                    rec[f"custom_{safe}"] = r.custom_metric_means[k]
        records.append(rec)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_labels,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), key in zip(positions, metric_keys):
        for llm in sorted(set(r["llm"] for r in records)):
            y_vals = [r.get(key) for r in records if r["llm"] == llm and key in r]
            x_vals = list(range(len(y_vals)))
            if y_vals:
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=llm, showlegend=(row == 1 and col == 1)),
                    row=row, col=col,
                )

    fig.update_layout(title_text="Metrics Across Configurations", height=700)
    path = output_dir / "metrics_over_time.html"
    fig.write_html(str(path))
    return str(path)


def _plot_llm_comparison_radar_html(results: list, output_dir: Path) -> str | None:
    """Interactive radar chart comparing LLMs across metrics."""
    llm_data: dict[str, dict[str, float]] = {}

    for r in results:
        llm = r.llm_model.split("/")[-1]
        if llm not in llm_data:
            llm_data[llm] = []
        metrics = {}
        if r.ragas_faithfulness is not None:
            metrics["Faithfulness"] = r.ragas_faithfulness
        if r.custom_metric_means:
            for k, label in [("hit@1", "Hit@1"), ("rouge_l", "ROUGE-L"), ("meteor", "METEOR"), ("bert_score_f1", "BERTScore F1"), ("context_relevance", "Ctx Rel")]:
                if k in r.custom_metric_means:
                    metrics[label] = r.custom_metric_means[k]
        llm_data[llm].append(metrics)

    if len(llm_data) < 2:
        return None

    # Average metrics per LLM
    avg_data = {}
    for llm, metrics_list in llm_data.items():
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        avg_data[llm] = {k: sum(m.get(k, 0) for m in metrics_list) / len(metrics_list) for k in all_keys}

    categories = sorted(set(k for v in avg_data.values() for k in v))
    if not categories:
        return None

    fig = go.Figure()
    for llm, metrics in avg_data.items():
        values = [metrics.get(c, 0) for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=llm,
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="LLM Comparison")
    path = output_dir / "llm_comparison_radar.html"
    fig.write_html(str(path))
    return str(path)


def _plot_parameter_heatmap_html(results: list, output_dir: Path) -> str | None:
    """Interactive heatmap of faithfulness by chunk_size x overlap per LLM."""
    if not results:
        return None

    llm_groups: dict[str, list] = {}
    for r in results:
        llm = r.llm_model.split("/")[-1]
        if r.ragas_faithfulness is not None:
            llm_groups.setdefault(llm, []).append(r)

    if not llm_groups:
        return None

    paths = []
    for llm, runs in llm_groups.items():
        sizes = sorted(set(r.chunk_size for r in runs))
        overlaps = sorted(set(r.chunk_overlap for r in runs))

        z = [[None] * len(overlaps) for _ in sizes]
        annotations = [[None] * len(overlaps) for _ in sizes]

        for r in runs:
            si = sizes.index(r.chunk_size)
            oi = overlaps.index(r.chunk_overlap)
            z[si][oi] = r.ragas_faithfulness
            annotations[si][oi] = f"{r.ragas_faithfulness:.3f}"

        fig = go.Figure(data=go.Heatmap(
            z=z, x=[str(o) for o in overlaps], y=[str(s) for s in sizes],
            text=annotations, texttemplate="%{text}",
            colorscale="RdYlGn", zmin=0.7, zmax=1.0,
        ))
        fig.update_layout(title=f"Faithfulness: {llm}", xaxis_title="Overlap", yaxis_title="Chunk Size")

        safe_llm = llm.replace("/", "_").replace(":", "_")
        path = output_dir / f"heatmap_{safe_llm}.html"
        fig.write_html(str(path))
        paths.append(str(path))

    return paths[0] if paths else None


def _plot_scatter_matrix_html(results: list, output_dir: Path) -> str | None:
    """Interactive scatter matrix of custom metrics colored by LLM."""
    if len(results) < 2:
        return None

    records = []
    for r in results:
        rec = {"llm": r.llm_model.split("/")[-1]}
        if r.custom_metric_means:
            for k in ["hit@1", "ndcg@1", "rouge_l", "meteor", "bert_score_f1", "context_relevance"]:
                if k in r.custom_metric_means:
                    safe = k.replace("@", "_at_")
                    rec[safe] = r.custom_metric_means[k]
        if len(rec) > 2:
            records.append(rec)

    if len(records) < 2:
        return None

    import pandas as pd
    df = pd.DataFrame(records)
    numeric_cols = [c for c in df.columns if c != "llm"]
    if len(numeric_cols) < 2:
        return None

    fig = px.scatter_matrix(df, dimensions=numeric_cols, color="llm", title="Custom Metrics Scatter Matrix")
    fig.update_layout(height=800)
    path = output_dir / "scatter_matrix.html"
    fig.write_html(str(path))
    return str(path)
```

- [ ] **Step 3: Wire the new plotly functions into `generate_plots`**

In the `generate_plots` function, after the existing `_plot_vector_distance_scatter` call, add calls to the new functions:

```python
    # Interactive Plotly charts
    for plot_fn in [
        _plot_metrics_over_time_html,
        _plot_llm_comparison_radar_html,
        _plot_parameter_heatmap_html,
        _plot_scatter_matrix_html,
    ]:
        try:
            result = plot_fn(results, plots_dir)
            if result:
                generated.append(result)
        except Exception as e:
            logger.warning("Plotly chart %s failed: %s", plot_fn.__name__, e)
```

- [ ] **Step 4: Verify visualization module imports**

Run: `python -c "from benchmark.reporting.visualization import generate_plots; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add benchmark/reporting/visualization.py
git commit -m "feat: add interactive plotly charts (metrics over time, radar, heatmap, scatter matrix)"
```

---

### Task 9: Configure MLflow MCP Server

**Files:**
- Modify: `.claude.json` (project-level, under the project's `mcpServers`)

- [ ] **Step 1: Add MLflow MCP server to `.claude.json`**

Find the project entry for `/home/niclas/Schreibtisch/Projects/Benchmarking-Framework` or `/home/niclas` in `.claude.json` under `projects`. Add the MCP server to the project's `mcpServers` dict:

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

If the project entry doesn't have an `mcpServers` key, add it. If it does, add the `mlflow-mcp` entry alongside any existing servers.

- [ ] **Step 2: Verify MLflow MCP is available**

Run: `mlflow mcp --help`
Expected: Shows help text for the MCP server command. (If this fails, MLflow version may need upgrading.)

- [ ] **Step 3: Commit**

```bash
git add .claude.json
git commit -m "feat: add MLflow MCP server configuration for Claude Code"
```

---

### Task 10: End-to-End Smoke Test

**Files:**
- No new files

- [ ] **Step 1: Start MLflow server**

Run in background:
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts &
```

- [ ] **Step 2: Run backfill script**

Run: `python backfill_mlflow.py`
Expected: Imports EVAL_MATRIX runs and prints summary.

- [ ] **Step 3: Verify runs in MLflow**

Open browser: `http://localhost:5000`
Expected: RAG-Benchmark experiment with backfilled runs visible.

- [ ] **Step 4: Run a quick benchmark smoke test**

Run: `python main.py` (with a small dataset / N=1 config)
Expected: Benchmark completes, run appears in MLflow with traces visible in the Traces tab.

- [ ] **Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address issues found during smoke testing"
```
