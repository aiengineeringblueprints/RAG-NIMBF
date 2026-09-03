# Which MCP Servers Make Sense to Benchmark

The adapter benchmarks *whatever MCP server you point it at* — the transport
(stdio or streamable HTTP) is the only hard requirement. That makes server
selection an experiment-design decision. Recommendations below are grouped by
what they let you measure.

## Tier 1 — ship with this repo (no external dependencies)

| Server | Tools | What it isolates |
| --- | --- | --- |
| `examples/corpus_mcp_server.py` (bundled) | `search`, `lookup`, `answer` | Lexical (BM25-like) retrieval vs. internal vector retrieval on the **same corpus** — the cleanest retrieval-quality comparison. `answer` also gives a fully extractive MCP-only QA baseline. |

Good first extension (small, self-written): a **vector MCP server** that wraps
the same corpus with embeddings. Lexical-MCP vs. vector-MCP vs. internal-vector
on one corpus separates "MCP transport overhead" from "retrieval algorithm".

## Tier 2 — realistic retrieval topologies (local, moderate effort)

| Server | Why benchmark it |
| --- | --- |
| A **hybrid retrieval server** (BM25 + vectors + fusion, e.g. via one FastMCP process) | Tests whether tool-exposed hybrid retrieval beats the framework's internal single-method retrieval. |
| A **multi-store server** (e.g. one tool per data source: docs DB, wiki, SQL) | Only interesting in *agentic* mode: measures tool-selection quality — does the agent pick the right store? `agent_tool_call_counts` per tool makes this visible. |
| **Filesystem server** (`@modelcontextprotocol/server-filesystem`) | Tests agentic navigation over raw files (list/read vs. search) — a different retrieval style entirely; good stress test for `mcp_max_agent_rounds`. |
| **Git server** (`@modelcontextprotocol/server-git`) | Same as filesystem but with history queries; natural multi-step agentic workload. |

## Tier 3 — external/web (network latency, rate limits, cost)

| Server | Why benchmark it |
| --- | --- |
| **Web search MCP** (e.g. Brave/Tavily/Exa MCP servers) | Open-domain QA where no local corpus exists; context quality and `answer` mode become interesting. Watch per-call latency and API cost. |
| **Fetch server** (`@modelcontextprotocol/server-fetch`) | Pairs with search: agent searches then fetches pages — the canonical two-tool agentic pipeline. |
| **Enterprise systems** (SQL, Slack, GitHub MCP servers) | Measures whether agentic MCP-RAG can replace hand-built integrations; expect lower faithfulness (noisy contexts) — exactly what RAGAS should show. |

## Selection criteria

1. **Same-corpus comparability first** — for quality claims, the server must
   retrieve from the corpus your `internal` baseline uses
   (`mcp_corpus_path` + fairness validation enforce this in context mode).
2. **Deterministic tools** for fixed mode; stochastic/external tools are fine
   for agentic mode but increase variance — raise `sample_size`.
3. **Structured output** (`{"contexts": [{"text", "source"}]}`) so gold-doc
   retrieval metrics (ndcg@5) work; plain-text-only tools limit you to
   answer-quality metrics.
4. **Fast cold start** — stdio servers that import heavy ML models on boot
   inflate `connection_seconds` and pollute the latency comparison.

## Practical notes

- Any server needing API keys: add the key to `.env` and list the name in
  `mcp_env_vars` so it is passed to the stdio child process explicitly.
- Remote servers: `streamable_http` only (no SSE-only endpoints), latency
  includes network — report `adapter_tool_seconds` separately from
  `agent_model_seconds` when arguing about agent efficiency.
- Start every new server with a smoke run:
  `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false DATASET_SAMPLE_SIZE=5` and
  check `results/runN/*mcp*.json` diagnostics before committing to a full sweep.
