# MLflow Tracing: one trace per sample with a single RETRIEVER span

Date: 2026-11-17

## Summary
Opt-in MLflow Tracing (issue #42, parent #40) now records exactly one trace
per benchmark sample, with exactly one RETRIEVER span that holds the contexts
the generator actually saw. It works the same way for the internal pipeline
and for external HTTP/MCP/plugin adapters. Before this change, multi-hop and
hybrid retrieval emitted several partial RETRIEVER root traces, reranked
results were never recorded, and every decorated helper produced a separate
root trace.

## Changes

### Config
- `BenchmarkConfig.mlflow_tracing_enabled` (default `False`), YAML setting
  `mlflow_tracing_enabled`, env fallback `MLFLOW_TRACING_ENABLED`. Coerced as
  a bool in `benchmark/orchestration/matrix.py`. Files: `config.py`,
  `benchmark/orchestration/matrix.py`.

### Per-sample traces (`benchmark/tracing.py`, `benchmark/orchestration/runner.py`)
- `trace_sample(...)` wraps each sample in a `rag_sample` CHAIN root span.
  Its attributes are the config name, sample index, adapter name and a
  `cached` flag.
- `SampleTrace.adapter_call()` adds a client-side `adapter.generate` (or
  `adapter.retrieve`) span around the adapter call, so internal and external
  adapters produce the same shape.
- `SampleTrace.record()` emits the single `retrieved_contexts` RETRIEVER span
  from the adapter's final contexts. `contexts_to_documents()` turns every
  context into an MLflow `Document`. The document id is taken from
  `doc_id`/`chunk_id`/`id` metadata, with `ctx_<n>` as the fallback, so bare
  string lists are no longer lost.
- `tracing_state(enabled)` switches tracing globally, once per phase.
  `run_single_benchmark` runs with tracing off and switches it on only for the
  sample loop when the setting is enabled. Ingestion and evaluation helpers
  (`chunk_documents`, `build_vector_store`, RAGAS, custom metrics) therefore no
  longer create stray root traces.
- `BenchmarkResultExtended.trace_ids` carries the per-sample trace IDs. Async
  trace export is flushed after the loop.

### Internal pipeline spans
- `benchmark/retrieval.py`: `retrieve` now creates a `vector_search` func
  span. Before, it created a RETRIEVER span, once per multi-hop round or
  hybrid leg.
- `benchmark/adapters/internal.py`: reranking goes through
  `benchmark.tracing.traced_rerank`, which records a `rerank` RERANKER span
  with the input and output documents. It does this only while a sample span
  is active, so tracing-off runs skip the conversion. This also covers
  injected rerankers.
- `benchmark/generation.py`: `generate_answer` span type changed from `func`
  to `LLM`.

### MCP agentic generation (`benchmark/adapters/mcp.py`)
- The agentic loop invokes the model through `benchmark.tracing.traced_model_call`.
  This adds one `agent_model_call` LLM span per model round, with the messages
  as inputs and the content and tool calls as outputs. The span is recorded
  only while a sample trace is active. It was added after the PR review,
  because these model calls previously bypassed `generate_answer` and left no
  LLM span.

### Run linking (`benchmark/tracking.py`)
- `log_benchmark_run` links `result.trace_ids` to the config's child run in
  batches of 100 and sets the run tag `mlflow_tracing=linked_<n>` (or
  `link_failed`, which is non-fatal).

### Docs and manifests
- `doc/results-and-mlflow.md`: new section "MLflow Tracing" with the span tree.
- `doc/env-vars.md`: `MLFLOW_TRACING_ENABLED`. `doc/00-README.md`: index entry.
- `experiments/mlflow-tracing-smoke.yaml`: smoke manifest.

## Behaviour changes and migration
- **With the setting off (the default), no traces are recorded at all.**
  Before, every `@mlflow.trace`-decorated helper call created its own trace.
  If you relied on those traces, set `mlflow_tracing_enabled: true`.
- The internal `retrieve` span is now named `vector_search` and has type
  `func`. Anything that looked up RETRIEVER spans by name `retrieve` must use
  `retrieved_contexts` instead.
- `_log_genai_rag_judges` (`MLFLOW_GENAI_JUDGES_ENABLED`) is unchanged: it
  still replays its own evaluation traces.

## Verification
- New `tests/test_mlflow_tracing.py` (8 tests) runs the real
  `run_single_benchmark` against a temporary SQLite tracking store:
  - one trace per sample, with the single RETRIEVER span's documents equal to
    the generator's contexts, for the internal pipeline with similarity,
    multi-hop (which also asserts at least 2 `vector_search` rounds), hybrid
    and rerank (which also needs exactly one RERANKER span);
  - a legacy `answer()` adapter with bare string contexts;
  - the real `HttpRagAdapter` with a patched `urlopen` (document ids come from
    the metadata);
  - the real `McpRagAdapter` in agentic mode with a fake MCP session and a fake
    chat model: two tool rounds per sample, merged contexts in the single
    RETRIEVER span, and 3 `agent_model_call` LLM spans per trace;
  - no traces when the setting is off;
  - traces linked to the `log_benchmark_run` config run.
- `tests/test_config.py::TestMlflowTracingSetting`: default, manifest
  coercion and env fallback.
- Full suite on the DGX Spark: 964 passed (after the review fixes).
- Spark smoke run: `experiments/mlflow-tracing-smoke.yaml` →
  `results/run-tracing-smoke` (2 internal configs × 3 SQuAD questions, qwen3.8-27b,
  cross-encoder reranker):
  - 6/6 traces are in state `OK`, and each one has exactly 1 RETRIEVER,
    1 RERANKER, 1 LLM and 2 CHAIN spans;
  - in 6/6 traces the RETRIEVER documents equal the checkpointed generator
    contexts;
  - both config runs are tagged `mlflow_tracing=linked_3` under the worker
    parent run.

## Follow-ups
- No live MCP server smoke run was made in this session. The real
  `McpRagAdapter` is covered by a unit test that uses a fake MCP session.
- The `generate_answer` LLM span is checked only by the Spark smoke run; unit
  tests stub the generator.
- `tracing_state` switches MLflow tracing process-wide. That is fine for the
  sequential worker, but not for running configs concurrently in threads.
- `_log_genai_rag_judges` could score the new per-sample traces directly
  instead of replaying them. That belongs to the judge follow-up of #40.
- The RETRIEVER span is emitted after the adapter returns, so its duration is
  close to zero. Retrieval latency lives in the `adapter.*`/`vector_search`
  spans.
