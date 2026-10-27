# MCP Host Benchmarking

Benchmark a RAG system that **calls MCP servers itself**: the system owns the
MCP client, the tool catalog and the model/tool loop. The framework sends
questions, observes the system's own MCP operation trace, and reports how
reliably, quickly and appropriately the system used its tools, next to the
usual answer-quality metrics.

This is the opposite of [MCP Benchmarking](mcp-benchmarking.md), where the
framework is the MCP client (`rag_system_adapter: mcp`) and the RAG system
sits behind an MCP server.

| | `mcp` adapter | MCP host adapters (`enterprise_rag_mcp`) |
| --- | --- | --- |
| Who calls MCP tools | the framework (fixed tool or its own agent loop) | the system under test |
| What is measured | an MCP server as a retrieval/QA backend | the system's MCP client, tool loop and tool choice |
| Trace source | framework-side provenance | the system's reported operation events |

## Architecture

- `benchmark/mcp_host.py`: system-independent contract and metrics.
  - `McpHostEvent(server, kind, status, name, duration_seconds)`: one MCP
    operation. `kind` is `connect`, `discover`, `tool`, `resource`, `call`
    (attempt rejected before reaching a capability) or `loop` (host-level
    loop event). `status` is one of `ok`, `tool_error`, `invalid_arguments`,
    `denied`, `timeout`, `unavailable`, `result_too_large`,
    `budget_exhausted`.
  - `mcp_host_diagnostics(events, expected_tools=...)`: per-sample
    diagnostics entry.
  - `aggregate_mcp_host_metrics(diagnostics)`: run-level summary, returned
    by the adapter's `aggregate_metrics()` hook.
- `examples/enterprise_rag_plugin/mcp_adapter.py`: the Enterprise RAG
  Blueprint adapter (`enterprise_rag_mcp`). It translates the Blueprint's
  `mcp.events` into `McpHostEvent`s.

Another MCP-host RAG system needs only a new adapter that maps its own trace
onto `McpHostEvent` and returns `aggregate_mcp_host_metrics()` from
`aggregate_metrics()`; reporting and tracking pick the metrics up as
`adapter_metrics` without further changes.

## Metrics

All metrics land in `adapter_metrics` (JSON/CSV as `adapter_<name>`; numeric
values also in MLflow). Calls are events of kind `tool`, `resource` and
`call`.

| Metric | Meaning |
| --- | --- |
| `call_count`, `call_success_count`, `call_success_rate` | attempted MCP calls and the share with status `ok` |
| `call_status_counts`, `<status>_rate` | calls per status (`tool_error_rate`, `invalid_arguments_rate`, `denied_rate`, `timeout_rate`, `unavailable_rate`, `result_too_large_rate`, `budget_exhausted_rate`) |
| `call_latency_p50/p95/max_seconds`, `call_seconds_total`, `timed_call_count` | latency over calls that report a duration |
| `discovery_count`, `discovery_success_rate` | server catalog discoveries per request and their success share |
| `connect_denied_count` | servers skipped because the identity lacked a role |
| `calls_per_sample_mean/max`, `samples_with_calls` | how much the system used tools |
| `duplicate_call_count/rate` | repeated calls to the same capability within one answer |
| `budget_exhausted_sample_count/rate` | answers that hit the system's call or round budget |
| `failed_sample_count/rate` | requests that failed as a whole (HTTP error, transport error, missing trace) |
| `tool_selection_precision/recall/f1` | micro-averaged over labelled samples, against `metadata.expected_tools` |
| `tool_selection_exact_match_rate` | share of labelled samples whose called tools equal the expected set |
| `missed_tool_sample_rate` | share of tool-needing samples where an expected tool was not called |
| `unnecessary_call_sample_rate` | share of samples labelled `expected_tools: []` that still called a tool |
| `per_server`, `per_tool` | the call summary per server and per `server/tool` (JSON/CSV only) |
| `answer_match_count`, `answer_match_rate` | answers satisfying the sample's `answer_patterns` (fallback: the ground truth); failed requests count as misses; also reported with MCP off |
| `per_category` | per `metadata.category`: `sample_count`, `answer_match_rate` and, with MCP on, call and tool-selection rates (JSON/CSV only) |

The per-sample stage timing `mcp_host_calls` is the sum of timed call
durations, so the report shows how much of `external_rag` was MCP time.

### Answer patterns

`benchmark/answer_match.py` scores answers deterministically, so MCP off vs.
on is comparable without an LLM judge. `metadata.answer_patterns` is a list
of groups of alternatives; every group must occur in the answer:

```json
"answer_patterns": [["shipped"], ["re:(?<![\\w.])96(?:\\s+[\\w/-]+){0,3}\\s+(?:units?|sensors?)"]]
```

Matching is case-insensitive, Unicode-normalized (`℃`, non-breaking spaces,
`−`), ignores Markdown emphasis and thousands separators, and never matches
inside a longer word or number. Alternatives starting with `re:` are regular
expressions over the normalized answer. Without patterns the ground truth is
the only alternative. RAGAS metrics still apply when enabled.

### Tool-selection labels

Put the tools a correct answer needs in the sample metadata:

```json
{"question": "What is the phone extension of the lead of project PRJ-220?",
 "ground_truth": "3940", "context": "",
 "metadata": {"category": "mixed", "expected_tools": ["search_directory"],
              "gold_doc_ids": ["KI-420"], "answer_patterns": [["3940"]]}}
```

- An entry matches a call by tool name (`add`) or `server/name`
  (`demo/add`). A comma-separated string is accepted too.
- `[]` means the documents suffice and no tool should be called.
- Missing `expected_tools` excludes the sample from selection metrics.
- A failed attempt still counts as selecting the tool; whether it worked
  shows in the call metrics.
- Failed requests are excluded from selection metrics.

## Enterprise RAG Blueprint (`enterprise_rag_mcp`)

The adapter sends each question to the Blueprint's `/call_llm` endpoint with
`use_mcp: true` and a freshly signed identity assertion (HMAC-SHA256 over the
claims, `aud=rag-mcp`, 60 s lifetime; the same format the Blueprint frontend
uses). With `enterprise_rag_mcp_enabled: false` the same adapter sends
`use_mcp: false` without identity, so one manifest compares MCP off vs. on.

Answers come from `result`. `relevant_documents` become contexts:
document evidence gets `evidence_type: document` plus `source`/`doc_id`; MCP
evidence gets `evidence_type: mcp`, `mcp_server`, `mcp_capability` and no
`doc_id`. MCP evidence stays in the contexts because the answer may rely on
it (faithfulness), so gold-document retrieval precision counts MCP evidence
as non-relevant.

HTTP status codes become failed samples with `failure_kind`: `guardrail`
(400), `identity` (401), `input` (422), `model_or_tool_protocol` (502),
`configuration` (503), `timeout` (504), `http_error` (other), `transport`
(connection error) or `missing_mcp_trace` (200 without `mcp`). The run
continues.

### Transports

- `inprocess` (default): imports `chain_api` from
  `ENTERPRISE_RAG_BLUEPRINT_PATH` and calls it through FastAPI's test client.
  The full endpoint runs (identity check, guardrails, retrieval, MCP loop)
  without Docker. The Blueprint's `.env` is loaded without overriding
  variables already set, so values in the framework `.env` or the shell win
  over the Blueprint's Docker-only values. The Blueprint chain dependencies
  must be installed in the framework venv.
- `http`: posts to a running chain service at
  `enterprise_rag_mcp_endpoint_url` (e.g. `http://localhost:8000/call_llm`).
  `MCP_IDENTITY_SECRET` must equal the service's secret. The framework cannot
  change the service's models; set `enterprise_rag_mcp_apply_settings: false`
  and configure the service's own environment.

### Which settings reach the Blueprint

With `enterprise_rag_mcp_apply_settings: true` (default, in-process only) the
adapter runs the Blueprint with the benchmark config. For every request it
sets these Blueprint environment variables and restores them afterwards; the
Blueprint reads them per request (Blueprint `feat/mcp-integration` from
2026-10-27 on; older checkouts are rejected with an error).

| Benchmark config | Blueprint variable | Notes |
| --- | --- | --- |
| `llm_models` entry `ollama:<model>` | `MODEL_NAME=<model>`, `OPENAI_BASE_URL=<LLM Ollama URL>/v1`, `OPENAI_API_KEY` | Ollama's OpenAI-compatible API; URL/key from `LLM_OLLAMA_*` or `OLLAMA_*` (key sent as Bearer, `ollama` without key) |
| `llm_models` entry `openai:<model>` | `MODEL_NAME=<model>`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` | URL from `llm_openai_compat_base_url` / `LLM_OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_BASE_URL`; key from the variable named by `enterprise_rag_mcp_llm_api_key_env`, else `LLM_OPENAI_COMPAT_API_KEY` / `OPENAI_COMPAT_API_KEY` |
| `generation_temperature` | `LLM_TEMPERATURE` | the framework default is 0.1, the Blueprint default 0 |
| `embedding_models` entry (Ollama only) | `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` | URL/key from `EMBEDDING_OLLAMA_*` or `OLLAMA_*`; the key is sent as Bearer; the same model builds the index, so index and queries match |
| `retrieval_top_k` | `RETRIEVER_TOP_K` | documents the Blueprint retrieves per question |
| `enterprise_rag_mcp_query_refinement` | `DISABLE_QUERY_REFINEMENT` | `false` skips the LLM query rewrite |
| `enterprise_rag_mcp_guardrails` | `DISABLE_GUARDRAILS` | `false` skips the LLM input check |

The Blueprint uses one model for query refinement, guardrails and the
answer/tool loop. MCP on needs a model with tool calling. Each
`llm_models` × `embedding_models` × `retrieval_top_k` combination is its own
run with its own name, so the matrix compares them directly, and results and
MLflow show the model that actually answered.

Not taken from the benchmark config: `prompt_templates` (the Blueprint uses
its own prompts, selected by `enterprise_rag_mcp_prompt_key`),
`reranker_models` (the Blueprint has no reranker), retrieval strategy, HyDE
and multi-hop. Its MCP servers, roles, tool allowlist and budgets come from
its own `MCP_CONFIG_FILE`.

With `enterprise_rag_mcp_apply_settings: false` the Blueprint runs with its
own `.env` (`MODEL_NAME` etc.), and `llm_models` only labels the run.

### Index

With `enterprise_rag_mcp_index_corpus: true` (default) `prepare()` replaces
the Blueprint Chroma collection (`INDEX_NAME` in `VECTORDB_DIR`) with the
dataset corpus, chunked by the framework and tagged `General`
(`category_bitmask: 1`). Include `chunker,embedder` in `rag_adapter_accepts`.
With applied settings any Ollama embedding model works; without, the
embedding model must equal the Blueprint's `EMBEDDING_MODEL` (default
`nomic-embed-text`). For the `http` transport the chain must read the same
`VECTORDB_DIR`; otherwise upload the corpus through the Blueprint and set
`enterprise_rag_mcp_index_corpus: false`.

The default identity role `General` reads `General`-tagged documents. Other
roles need matching document tags, because the Blueprint's MCP document
policy drops documents whose tag bits the identity does not hold.

### Settings

| Field / env var | Default | Meaning |
| --- | --- | --- |
| `enterprise_rag_blueprint_path` / `ENTERPRISE_RAG_BLUEPRINT_PATH` | unset | Blueprint checkout; root and `chain/` go on `sys.path` (also used by `enterprise_rag`) |
| `enterprise_rag_mcp_transport` / `ENTERPRISE_RAG_MCP_TRANSPORT` | `inprocess` | `inprocess` or `http` |
| `enterprise_rag_mcp_endpoint_url` / `ENTERPRISE_RAG_MCP_ENDPOINT_URL` | unset | `/call_llm` URL, required for `http` |
| `enterprise_rag_mcp_enabled` / `ENTERPRISE_RAG_MCP_ENABLED` | `true` | send `use_mcp`; `false` runs the legacy non-MCP path |
| `enterprise_rag_mcp_apply_settings` / `ENTERPRISE_RAG_MCP_APPLY_SETTINGS` | `true` | run the Blueprint with the config's LLM, embedding, top-k and temperature (see above); requires `inprocess` |
| `enterprise_rag_mcp_llm_api_key_env` / `ENTERPRISE_RAG_MCP_LLM_API_KEY_ENV` | unset | name of the `.env` variable holding the Blueprint LLM's key (e.g. `ANTHROPIC_API_KEY`) |
| `enterprise_rag_mcp_query_refinement` / `ENTERPRISE_RAG_MCP_QUERY_REFINEMENT` | `true` | Blueprint query rewrite (applied settings only) |
| `enterprise_rag_mcp_guardrails` / `ENTERPRISE_RAG_MCP_GUARDRAILS` | `true` | Blueprint input guardrails (applied settings only) |
| `enterprise_rag_mcp_roles` / `ENTERPRISE_RAG_MCP_ROLES` | `General` | comma-separated roles in the signed identity |
| `enterprise_rag_mcp_subject` / `ENTERPRISE_RAG_MCP_SUBJECT` | `benchmark` | identity subject |
| `enterprise_rag_mcp_prompt_key` / `ENTERPRISE_RAG_MCP_PROMPT_KEY` | `rag_source` | `rag_source`, `rag_summary` or `rag_weekly_report` |
| `enterprise_rag_mcp_used_model` / `ENTERPRISE_RAG_MCP_USED_MODEL` | unset | Blueprint LLM *source* key sent as `used_model` (the Blueprint supports `openai`); unset sends `openai` with applied settings, else the Blueprint's `LLM_SOURCE_ANSWER` |
| `enterprise_rag_mcp_timeout_seconds` / `ENTERPRISE_RAG_MCP_TIMEOUT_SECONDS` | `120` | request timeout |
| `enterprise_rag_mcp_index_corpus` / `ENTERPRISE_RAG_MCP_INDEX_CORPUS` | `true` | rebuild the Blueprint index from the dataset corpus |
| `MCP_IDENTITY_SECRET` (env only) | — | identity signing secret, at least 32 characters; never put it in a manifest |

### Setup

Step-by-step `.env` and manifest setup, model endpoints with Bearer keys,
the `http` variant and troubleshooting:
[Enterprise RAG Blueprint Setup Guide](enterprise-rag-blueprint-setup.md).

`experiments/enterprise-rag-mcp-host.yaml` uses the
[`datasets/blueprint_mcp_host`](../datasets/blueprint_mcp_host/README.md)
fixture: 240 documents of ten kinds (datasheets, release notes, policies,
sites, customers, projects, incidents, suppliers, courses, distractors) with
realistic sections, mixed formats (tables, lists, prose, FAQ, meeting notes),
lengths from about 100 to 2,050 words and older competing values inside
documents (revision histories, estimates), 1000
questions in five categories (`doc_single` 300, `doc_multi` 100,
`mcp_single` 250, `mcp_chain` 100, `mixed` 250, 51 sub-types) and an MCP
server with eight data tools plus two decoys. The manifest samples the first
200 questions; set `sample_size: 1000` for the full set (about 4–5 Blueprint
LLM calls per question). The dataset README lists dense-retrieval
difficulty per category.

## Reference result (2026-10-24, previous 59-question version)

Measured on the previous dataset version (25 documents, 59 questions, 6 data
tools), which the 1000-question version replaces; no full run of the new
version exists yet (see `doc/experiment-todos.md` §7).
Blueprint `feat/mcp-integration` (uncommitted working tree), in-process,
`claude-opus-5-5` via the Blueprint `.env`, `nomic-embed-text`, chunks
512/64, RAGAS off. Answer match re-scored with the final patterns.

| Category | n | MCP off | MCP on |
| --- | ---: | ---: | ---: |
| `doc_single` | 20 | 14 | 14 |
| `doc_multi` | 4 | 0 | 0 |
| `mcp_single` | 18 | 0 | 18 |
| `mcp_chain` | 4 | 0 | 4 |
| `mixed` | 13 | 0 | 11 |
| **total** | 59 | 14 (24 %) | 47 (80 %) |

MCP on: 39 calls, `call_success_rate` 0.97 (one `tool_error` after a wrong
product code), p95 call latency 7 ms, `tool_selection_f1` 0.97, one
unnecessary call (a stock lookup on a `doc_multi` question), decoy never
called, no budget exhaustion. Wall time 484 s vs. 476 s with MCP off.

All document misses are retrieval misses: the Blueprint's dense-only
retriever (`k=5`) does not separate near-identical datasheets by product code
(K3/K5/K7/K7X). A direct retriever check puts the gold document in the top 5
for 15 of 20 single-document questions and never retrieves both documents of
a `doc_multi` question. The model then declines instead of guessing. The
same misses limit `mixed` questions whose tool input comes from a datasheet.

## Limitations

- The Blueprint emits `duration_ms` only for calls that returned (`ok`,
  `error`); `unavailable` covers both timeouts and transport failures, so
  `timeout_rate` stays 0 and timeouts count as `unavailable`.
- No discovery/connection timing, model-round counts or token usage are
  reported by the Blueprint, so cold-start cost and tool-loop token cost are
  not measured.
- Events carry capability aliases; names come from successful sources or
  the alias label (sanitized, at most 20 characters).
- The in-process transport skips only the HTTP layer and the separate
  frontend; the HMAC issuer is the framework.
