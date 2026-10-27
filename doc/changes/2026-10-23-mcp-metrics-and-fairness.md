# MCP Tool-Use Metrics, Evidence Budget, and RAG-vs-MCP Fairness Fix

Date: 2026-10-23

## Summary

Added MCP-specific reliability, latency, tool-use, and overhead metrics plus
an adapter-agnostic evidence-budget metric. A smoke run with the new metrics
exposed that every shipped RAG-vs-MCP manifest compared different corpora
while passing fairness validation; validation, manifests, and the example MCP
server were fixed so both systems now search the same corpus with passages of
the same size. User guide: [../mcp-benchmarking.md](../mcp-benchmarking.md).

## Changes

### 1. MCP adapter metrics (`benchmark/adapters/mcp.py`)

New run-level aggregates in `_aggregate_mcp_metrics`, logged as `adapter_*`:

- Failure split: `transport_failure_count`, `tool_error_count`,
  `agent_failure_count`, `other_failure_count`. Each failed question now records
  `failure_kind` in its diagnostics. Previously all failures were one count,
  which hid whether the server, the network, or the model was at fault.
- Latency percentiles over individual calls: `tool_latency_p50_seconds`,
  `tool_latency_p95_seconds`, `tool_latency_max_seconds` (nearest-rank, same
  method as `benchmark/llm_performance.py`). Only a warm mean existed before.
- Tool use: `tool_calls_per_sample_mean`/`_max`, `agent_rounds_mean`/`_max`,
  `duplicate_tool_call_count`/`_rate` (same tool + arguments repeated within a
  question).
- Input-schema validation: every call's arguments are validated against the
  tool's declared JSON schema (`jsonschema`, Draft 2020-12). Results are recorded
  per call (`schema_valid`, `schema_errors`) and aggregated as
  `schema_invalid_call_count`/`_rate`. Calls are still sent; the server decides.
  Error messages contain path and rule only, never argument values, so secrets
  cannot leak into reports.
- Agentic overhead: `model_calls` and `tool_schema_chars` per question;
  `tool_schema_chars_total` and `tool_schema_chars_per_sample_mean` per run.
  Measured in characters (decision: deterministic and tokenizer-independent).

`jsonschema>=4.0` was added to `requirements.txt` (already installed as a
dependency of `mcp`, now imported directly).

### 2. Evidence-budget metrics for every adapter

`avg_contexts_per_question`, `max_contexts_per_question`,
`avg_context_chars_per_question` on `BenchmarkResultExtended`
(`benchmark/reporting/models.py`), computed in
`benchmark/orchestration/runner.py`, exported to JSON/CSV
(`benchmark/reporting/exports.py`), MLflow (`benchmark/tracking.py`), and
ClearML (`benchmark/clearml_task.py`). Reason: agentic MCP runs can collect far
more evidence than the internal top-k, which distorts retrieval metrics.

### 3. Fairness validation requires `jsonl-shared` (`benchmark/orchestration/matrix.py`)

Root cause found in the smoke run: only `jsonl-shared` builds the internal index
from `dataset.corpus_path`. With `jsonl`, the internal pipeline indexed only the
sampled questions' own contexts (15 chunks, all from NF-001, answer always
present) while the MCP server searched all 60 documents. The validator compared
only path strings, so it passed. `_validate_mcp_comparison_fairness` now
rejects RAG-vs-MCP matrices whose dataset is not `jsonl-shared`.

### 4. Shipped manifests (`experiments/rag-vs-mcp*.yaml`)

- `dataset.name: jsonl-shared`, `subset: null`.
- Pinned `retrieval_multihop: false`, `retrieval_use_hyde: false`,
  `retrieval_strategy: similarity`, `dataset_question_field`,
  `dataset_ground_truth_field`. A HotpotQA-oriented `.env` had silently enabled
  multi-hop retrieval, which collapsed duplicate chunks to a single context, and
  had broken dataset loading via `DATASET_GROUND_TRUTH_FIELD=answer`.
- `rag-vs-mcp.yaml` and `rag-vs-mcp-agentic.yaml` pass
  `--chunk-size 1000 --chunk-overlap 200` to the example server.

### 5. Example server chunking (`examples/corpus_mcp_server.py`)

New `--chunk-size`/`--chunk-overlap` options split the corpus like the internal
`recursive` strategy (`RecursiveCharacterTextSplitter`, fragments under 50
characters dropped). Default `0` keeps whole documents. `search` results gain
`chunk_index`; `lookup` and `answer` are unchanged. Reason: whole documents gave
MCP about four times the evidence of the internal pipeline.

### 6. Documentation

- New guide `doc/mcp-benchmarking.md`; README MCP section updated (metrics,
  `jsonl-shared` requirement, server chunking options).
- `doc/implementation-overview.md` and `doc/00-README.md` updated.
- New change-log convention in `doc/changes/` and `AGENTS.md`.

## Behaviour changes and migration

- **RAG-vs-MCP manifests using `dataset.name: jsonl` now fail validation.**
  Switch to `jsonl-shared` (requires `corpus_path`), or set
  `mcp_enforce_fairness: false` for an intentionally asymmetric run.
- Results from earlier RAG-vs-MCP runs with the shipped manifests compared
  different corpora; do not use them as a fair comparison.
- New keys in reports and MLflow; nothing was renamed or removed.
  `error_type` for tool errors is still `RuntimeError`.

## Verification

- Tests (written test-first): 8 new in `tests/test_mcp_adapter.py`, 1 in
  `tests/test_run_path_uniformity.py`, 1 regression test plus 2 adjusted in
  `tests/test_config.py`, new `tests/test_corpus_mcp_server.py` (drives the real
  server over stdio). Suite: 612 passed; `tests/test_enterprise_rag_adapter.py`
  excluded because it fails independently of these changes (missing `prompts`
  module in the Enterprise sandbox).
- Smoke runs (5 questions, `ollama:qwen2.5:0.5b`, RAGAS off), fixed mode:

  | | Internal | MCP whole docs | MCP chunked |
  |---|---|---|---|
  | Index | 280 chunks / 60 docs | 60 docs | 60 docs |
  | `avg_contexts_per_question` | 5 | 5 | 5 |
  | `avg_context_chars_per_question` | 4,604 | 18,957 | 4,745 |
  | `hit@1` | 0.40 | 1.00 | 1.00 |

  The remaining hit@1 gap is lexical vs dense retrieval on templated documents
  whose questions name the facility ID, not a fairness issue.
- Agentic smoke: `qwen2.5:0.5b` only called `lookup` with invalid IDs; the
  failures were classified as `tool_error` as intended.

## Follow-ups

- Tool-call correctness metrics (needs an `expected_tool_calls` dataset field).
- Gold-doc span metrics average over questions with contexts only, which reads
  too high in agentic runs with few successful calls.
- Concise-answer fallback turns non-answers into fragments (`-001`) that count
  as valid answers.
- Fairness validation cannot check server-side settings such as chunk size;
  rely on the evidence-budget metrics.
- Stronger internal baseline (hybrid/keyword-aware retrieval) for ID-heavy
  datasets.
