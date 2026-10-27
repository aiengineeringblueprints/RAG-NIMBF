# MCP host benchmarking for the Enterprise RAG Blueprint

Date: 2026-10-24

## Summary
The framework can now benchmark a RAG system that calls MCP servers itself,
instead of only MCP servers called by the framework. A system-independent
metrics module scores the system's own MCP operation trace (call success,
latency, budget exhaustion, tool selection), and a new plugin adapter
benchmarks the Enterprise RAG Blueprint's `feat/mcp-integration` MCP host
end to end, MCP off vs. on.

## Changes

**MCP host metrics** (`benchmark/mcp_host.py`, new): `McpHostEvent`,
`mcp_host_diagnostics()`, `mcp_host_stage_timings()`,
`aggregate_mcp_host_metrics()` and the shared nearest-rank `percentile()`.
Tool selection is scored against `metadata.expected_tools`. The existing
`benchmark/adapters/mcp.py` now imports `percentile` instead of keeping its
own copy. Metric reference: [MCP Host Benchmarking](../mcp-host-benchmarking.md).

**Blueprint adapter** (`examples/enterprise_rag_plugin/mcp_adapter.py`,
new, registers `enterprise_rag_mcp`): posts to the Blueprint `/call_llm` with
`use_mcp` and a per-request HMAC identity assertion, in process (FastAPI test
client over `chain_api.app`) or over HTTP; translates `mcp.events` into
`McpHostEvent`s; maps HTTP failures to `failure_kind`; optionally rebuilds
the Blueprint index from the dataset corpus.

**Existing Blueprint plugin** (`examples/enterprise_rag_plugin/adapter.py`):
fixed the import of the non-existent `Enterprise_RAG_Blueprint` package
(four tests were failing). New `ensure_blueprint_importable()` uses
`ENTERPRISE_RAG_BLUEPRINT_PATH`; indexing moved into the reusable
module function `replace_blueprint_index()`.

**Config** (`config.py`, `benchmark/orchestration/matrix.py`): new fields
`enterprise_rag_blueprint_path`, `enterprise_rag_mcp_transport`,
`enterprise_rag_mcp_endpoint_url`, `enterprise_rag_mcp_enabled`,
`enterprise_rag_mcp_roles`, `enterprise_rag_mcp_subject`,
`enterprise_rag_mcp_prompt_key`, `enterprise_rag_mcp_used_model`,
`enterprise_rag_mcp_timeout_seconds`, `enterprise_rag_mcp_index_corpus`
with env fallbacks, manifest coercion and validation. Config names for
`enterprise_rag_mcp` hash these fields, so MCP off/on runs do not collide.

**Dataset and manifest** (superseded the same day, see [2026-10-24-blueprint-mcp-host-dataset-v2](2026-10-24-blueprint-mcp-host-dataset-v2.md)): `datasets/blueprint_mcp_host/` (3-document corpus,
8 labelled questions) and `experiments/enterprise-rag-mcp-host.yaml`
(matrix `enterprise_rag_mcp_enabled: [false, true]`).

**Docs**: new `doc/mcp-host-benchmarking.md`; updated
`doc/Enterprise_RAG_Blueprint_Benchmark.md` (sibling checkout path,
`ENTERPRISE_RAG_BLUEPRINT_PATH`), `doc/00-README.md`, README env table.

## Behaviour changes and migration
The `enterprise_rag` plugin no longer expects the Blueprint inside the repo
as `Enterprise_RAG_Blueprint/`. Set `ENTERPRISE_RAG_BLUEPRINT_PATH` (or put
the Blueprint root and `chain/` on `PYTHONPATH`) instead of the old
`PYTHONPATH=.:Enterprise_RAG_Blueprint:...`. Nothing else changes for
existing manifests.

## Verification
- New tests: `tests/test_mcp_host.py` (9), `tests/test_enterprise_rag_mcp_adapter.py`
  (22), `TestEnterpriseRagMcpConfig` in `tests/test_config.py` (9).
- Full suite: 658 passed (before: 614 passed, 4 failed in
  `tests/test_enterprise_rag_adapter.py`).
- Live run against the real Blueprint (uncommitted `feat/mcp-integration`
  working tree), in-process, demo MCP server over stdio, Blueprint LLM from
  its `.env`, RAGAS off, `--no-mlflow`:
  - MCP off: retention-policy question unanswerable; arithmetic answered
    without tools.
  - MCP on: 8/8 answers correct, `call_count` 4, `call_success_rate` 1.0,
    `call_latency_p95_seconds` 0.006, `discovery_success_rate` 1.0,
    `tool_selection_f1` 1.0, `unnecessary_call_sample_rate` 0.0;
    `external_rag` 56.4 s vs. 45.0 s without MCP.
- Live negative check: wrong `MCP_IDENTITY_SECRET` yields a failed sample
  with `failure_kind: identity` (HTTP 401) and the run continues.

## Follow-ups
- Blueprint trace gaps: no duration for failed calls, timeouts reported as
  `unavailable`, no discovery timing, round count or token usage, events
  carry aliases only. Small additions to the Blueprint's
  `client_manager.py`/`execution.py` would close them.
- Fault-injection manifest (slow/broken/oversized/prompt-injecting servers),
  e.g. via the demo server's `slow_read`.
- Optional per-server/per-tool report table or plot; today these are only in
  JSON/CSV.
