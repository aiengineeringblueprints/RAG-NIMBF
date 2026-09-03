# MCP Benchmark Example

This folder is a self-contained starting point for benchmarking MCP-based
retrieval and agentic MCP-RAG against the built-in vector RAG pipeline.
Run everything from the repository root.

## What gets compared

| Variant | Config | Question it answers |
| --- | --- | --- |
| `internal` | `rag_system_adapter: internal` | How good is our vector RAG? (baseline) |
| MCP retriever | `rag_system_adapter: mcp`, `mcp_result_mode: context`, `mcp_execution_mode: fixed` | Does retrieval *through an MCP tool* match internal vector retrieval, given the same generator? |
| Agentic MCP-RAG | `rag_system_adapter: mcp`, `mcp_result_mode: context`, `mcp_execution_mode: agentic` | Does letting the LLM decide when/what to retrieve improve quality enough to justify extra rounds, tokens, and latency? |
| MCP-only QA | `rag_system_adapter: mcp`, `mcp_result_mode: answer` | How good is a complete external MCP QA system with no framework generator? |

All four produce the same `answer + contexts` contract internally, so quality
metrics (faithfulness, context relevance, ndcg@5, BERTScore, RAGAS) are
directly comparable. `mcp_enforce_fairness: true` refuses context-mode runs
that do not share dataset, corpus, top-k, generator, prompt, and token limit
with the `internal` baseline.

## How the evaluation works

1. **Per sample** the adapter records `diagnostics`: every tool call with
   redacted arguments, attempts, per-attempt latency, timeouts, errors, plus
   (agentic mode) rounds, per-round tool usage, and token accounting.
2. **Per run** these are aggregated into `adapter_metrics` in the summary
   JSON/CSV and logged to MLflow with the `adapter_` prefix, e.g.
   `adapter_tool_call_count`, `adapter_failure_rate`, `adapter_warm_tool_mean_seconds`.
3. **Agentic run metrics** (prefix `adapter_agent_`):

   | Metric | Meaning |
   | --- | --- |
   | `agent_sample_count` | samples run in agentic mode |
   | `agent_rounds_mean` / `agent_rounds_max` | model/tool rounds per answer |
   | `agent_tool_calls_mean` / `_max` | tool invocations per answer |
   | `agent_rounds_exhausted_count` | answers cut off by `mcp_max_agent_rounds` |
   | `agent_no_retrieval_count` | answers produced without any tool evidence |
   | `agent_error_count` | samples that failed |
   | `agent_tokens_mean` / `_total` | LLM tokens spent per answer / per run |
   | `agent_model_seconds` / `agent_tool_seconds` | latency split: thinking vs. tools |

4. **Evaluation** then runs unchanged: RAGAS + custom metrics on
   `(question, answer, contexts, ground_truth)`, reports under
   `results/runN/`, one MLflow run per matrix cell plus an aggregate summary.

Interpretation guide:

- `agent_no_retrieval_count` high → the model answers from parametric memory;
  quality numbers are not "MCP retrieval" quality.
- `agent_rounds_exhausted_count` high → raise `mcp_max_agent_rounds` or the
  token limit; those samples are truncated, not failed retrieval.
- Quality gain vs. `fixed` smaller than `agent_tokens_mean` growth → the
  autonomy is not paying for itself.

## File structure expected

```
Benchmarking-Framework/
├── .env                        # your copy of mcp-example/.env.example (secrets, URLs)
├── mcp-example/
│   ├── README.md               # this file
│   ├── MCP_SERVERS.md          # which MCP servers make sense to benchmark
│   ├── AGENTIC_CAPABILITIES.md # what the agent can/cannot do yet
│   ├── .env.example
│   └── experiments/
│       ├── baseline-internal.yaml       # internal RAG reference run
│       ├── mcp-modes.yaml               # the 3 MCP variants
│       └── mcp-agentic-rounds.yaml      # agent budget sweep
├── datasets/northstar_facilities/       # sample corpus + questions.jsonl
└── examples/corpus_mcp_server.py        # bundled lexical MCP baseline server
```

Datasets are JSONL with `question` and `ground_truth` fields (context
optional); corpora are folders of `.md`/`.txt` files. Swap in your own via
`dataset.path` / `dataset.corpus_path`.

## Quickstart

```bash
pip install -r requirements.txt
cp mcp-example/.env.example .env         # edit as needed

# smoke test (no critic LLM calls):
BENCHMARK_CONFIG_FILE=mcp-example/experiments/baseline-internal.yaml \
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py
BENCHMARK_CONFIG_FILE=mcp-example/experiments/mcp-modes.yaml \
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py

# full benchmark (run both, they share the corpus):
BENCHMARK_CONFIG_FILE=mcp-example/experiments/baseline-internal.yaml python main.py
BENCHMARK_CONFIG_FILE=mcp-example/experiments/mcp-modes.yaml python main.py

# agentic budget sweep:
BENCHMARK_CONFIG_FILE=mcp-example/experiments/mcp-agentic-rounds.yaml python main.py
```

Results land in `results/runN/` (per-config JSON, CSV/Markdown summaries,
per-sample diagnostics) and MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

In the `RAG-Benchmark` experiment, compare runs and show the `adapter_*`
params/metrics next to the quality metrics.

## Configuring your own MCP server

Everything can be set in the experiment YAML (`settings.mcp_*`) or via `.env`
(`MCP_*`). YAML wins for workflow fields; keep secrets in `.env`.

**Local stdio server** (subprocess, no shell involved):

```yaml
mcp_transport: stdio
mcp_command: python                      # or: npx, uvx, docker, ...
mcp_args_json: ["-y", "@org/server"]     # JSON argument array
mcp_env_vars: MY_API_KEY                 # comma-separated env names passed through
mcp_tool_name: search
mcp_question_argument: query             # tool argument receiving the question
mcp_top_k_argument: top_k                # optional; gets retrieval_top_k
mcp_result_field: contexts               # dotted path to contexts/answer in output
```

**Remote streamable-HTTP server:**

```yaml
mcp_transport: streamable_http
mcp_server_url: https://mcp.example.com/mcp
mcp_http_headers_json: {"X-Project": "benchmark"}
```

Auth headers go in `mcp_http_headers_json` (or `MCP_HTTP_HEADERS_JSON` in
`.env`); keep the actual token out of the YAML — build the JSON in `.env`
only.

**Tool contract**: in `context`/`answer` mode the adapter accepts structured
JSON output or plain text; context entries may be strings or objects with a
`text`/`content`/`page_content`/`context` field; `source`/`source_id`/`doc_id`
become retrieval metadata for gold-doc metrics. SSE-only servers are not
supported.

See `MCP_SERVERS.md` for recommended servers to try next and
`AGENTIC_CAPABILITIES.md` for what the agent layer can and cannot do yet.
