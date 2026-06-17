# Testing and Coverage

Test folder: [tests/](../tests)

Current inspection found 14 test files under `tests/*.py`, including adapter, dataset, config, retrieval, generation, evaluation, metrics, model, provider, prompt, reranker, and chunking coverage.

Coverage map:

- `test_config.py`: env parsing, validation, combinations, adapter/backend/stage variables, YAML manifest entrypoint, and early registry checks.
- `test_dataset.py`: dataset loading, sample contract normalization, dataset adapter registry behavior, local JSONL/CSV loading, context builders, shared-corpus metadata, and RAGPerf-style loading.
- `test_adapters.py`: RAG-system adapter dispatch and HTTP response normalization.
- `test_chunking.py`: splitter creation and document chunking.
- `test_retrieval.py`: vector-store/retrieval behavior.
- `test_generation.py`: answer cleanup, validation, generation helpers.
- `test_embedding.py`: embedding model factory.
- `test_providers.py`: provider parsing/chat model wrappers.
- `test_reranker.py`: reranker behavior.
- `test_evaluation.py`: RAGAS evaluation wrapper behavior.
- `test_metrics.py`: custom metrics and GPU metrics.
- `test_gold_retrieval_metrics.py`: gold-document hit@k, nDCG@k, recall@k behavior.
- `test_models.py`: reporting/result models.
- `test_prompt_templates.py`: built-in prompt template registry.

Standalone:

- `test_ragas_wikiqa.py`: separate script for RAGAS WikiQA experimentation.

Strengths:

- Config/env parsing and validation are heavily covered, including HTTP adapter variables, local dataset variables, adapter-module autoload, and metric toggles.
- Generation cleanup and fallback behavior have detailed tests.
- Dataset adapter transforms and sample contract normalization are tested with mocked loading.
- Retrieval tests cover cache keys, provider/dataset fingerprinting, backend key separation, routing, HyDE fallback behavior, Chroma cleanup, and query-mode fail-closed behavior.
- Adapter tests cover the HTTP adapter's nested field lookup, context and metadata normalization, timing/token normalization, and internal-adapter dispatch.

Thin or absent coverage:

- LanceDB vector-store creation/query behavior with an actual temporary LanceDB database.
- Full RAG-system adapter failure paths, including invalid JSON, non-object responses, bad header JSON, request failures, and empty-answer invalidation.
- `benchmark/tracking.py` MLflow behavior, including parent/child run nesting and optional `mlflow.genai.evaluate()` judge logging.
- Custom metric helpers such as hit@k, nDCG, recall@k, ROUGE-L, BLEU, METEOR, and BERTScore.
- Reporting analysis, exports, terminal output, and plot generation.
- End-to-end `run_single_benchmark()` orchestration.
- Real data/vector store/model/artifact round trips; most tests are mock-heavy.

Recommended verification commands:

```bash
python -m unittest
python -m py_compile main.py config.py benchmark/*.py agentic/*.py
```

Risk-based testing guidance:

- Config/env/YAML changes: run `tests/test_config.py` and `tests/test_orchestration.py`, especially when touching registry validation, manifest expansion, stage restrictions, adapter variables, or vector backend variables.
- Dataset adapter changes: run `tests/test_dataset.py`; add cases for required registry fields, malformed/missing source columns, metadata copying, shared-corpus `doc_id` / `gold_doc_id`, and datasets with intentionally empty per-question context.
- RAG-system adapter changes: run `tests/test_adapters.py`; add failure-path tests for request errors, invalid JSON, non-object JSON, malformed headers, dotted-field misses, context coercion, metadata coercion, and empty answers setting `answer_valid=False`.
- Retrieval/chunking changes: run chunking/retrieval tests and inspect `.chroma/` or `.lancedb/` cache behavior. Add backend-seam tests when changing `build_vector_store()`, cache keys, query-only behavior, or LanceDB compatibility.
- Generation cleanup changes: run generation tests with thinking/refusal/numeric cases.
- Reporting/tracking changes: run reporting/model tests, `python -m py_compile main.py benchmark/tracking.py benchmark/orchestration/worker.py`, and a tiny benchmark to ensure outputs still write. Use `MLFLOW_GENAI_JUDGES_ENABLED=false` for local smoke tests unless judge credentials are configured.
- Agentic changes: run unit checks plus `python -m agentic.cli --max-iterations 1 --sample-size 1` when models are available.

Recommended next tests:

- Add real temporary LanceDB tests for table creation, query-only missing-table failure, and retrieval output metadata preservation.
- Expand HTTP adapter tests to cover invalid response and header cases.
- Add a tiny orchestration test around `run_single_benchmark()` with mocked adapters/vector stores to verify stage timing, `answer_valid`, per-sample output, resource-monitor markers, and backend/adapter routing without loading models.

Related notes:

- [[Known Risks and Gaps]]
- [[Operations Runbook]]

Resource plotting verification:

```bash
python -m py_compile main.py benchmark/resource_monitor.py benchmark/reporting/resource_charts.py
python -m benchmark.reporting.resource_charts --scenario label:path/to/trace.csv:path/to/trace_markers.csv --output /tmp/resource-chart-smoke/out
```

Current coverage does not include automated assertions for resource monitor CSV sampling or resource chart rendering.


Enterprise RAG adapter coverage:

- `tests/test_enterprise_rag_adapter.py` verifies that the adapter advertises `chunker`, `embedder`, and `llm` component injection, calls the dataset-index replacement step before creating the Blueprint retriever, routes injected components into Blueprint APIs, and normalizes `rag_chain` answer/source output.
