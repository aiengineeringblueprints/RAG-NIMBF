# Agentic RAG Tool Benchmarking

Documentation for the agentic tools feature (`feature/agentic-tools` branch):
a LangChain tool suite, an agent adapter that runs it, and metrics that
evaluate *how well the agent calls tools* — alongside the existing answer
quality metrics.

## Overview

The feature adds three components:

| Component | Location | Purpose |
|---|---|---|
| Tool suite | `benchmark/agentic/tools.py` | 10 injectable RAG tools an agent can call |
| Agent adapter | `benchmark/adapters/agentic.py` | `AgenticRagAdapter` — ReAct-style loop over the tools, records every call |
| Tool-call metrics | `benchmark/agentic/metrics.py` | Correctness + efficiency metrics over recorded calls |

Design principle: **every tool is built by a factory function that closes over
an injectable backend** (retriever, LLM, reranker, search function). Tests
supply fakes; the adapter wires the real pipeline components. The loop
follows the existing pattern from `benchmark/adapters/mcp.py` (hand-rolled
`bind_tools`/`invoke` round loop, no LangGraph dependency).

## 1. The tool suite

`benchmark/agentic/tools.py`

| Tool | Signature | Purpose | Required backend |
|---|---|---|---|
| `retrieve` | `(query, top_k)` | Main vector-store search | retriever |
| `retrieve_with_filter` | `(query, filters: JSON str, top_k)` | Self-query style metadata-filtered search | retriever (filter-capable) |
| `retrieve_alternate` | `(query, top_k)` | Search a second collection (routing) | alternate retriever |
| `hyde_retrieve` | `(query, top_k)` | HyDE: search with a generated hypothetical answer | hypothesize fn + retriever |
| `rewrite_query` | `(query)` | LLM query rewriting for better retrieval | llm |
| `rerank_chunks` | `(query, documents: list[str], top_k)` | Re-rank previously retrieved chunks | reranker |
| `grade_relevance` | `(query, document)` | Grade one chunk's relevance → `relevant`/`not relevant` | grader fn |
| `grade_answer` | `(answer, contexts)` | Faithfulness self-check → `supported`/`unsupported` | answer checker fn |
| `web_search` | `(query)` | Public web fallback (DuckDuckGo, lazy import) | optional custom search fn |
| `fetch_url` | `(url)` | Fetch a page's text, optional domain allowlist | optional custom fetch fn |

Build tools individually:

```python
from benchmark.agentic.tools import build_retrieve_tool

retrieve = build_retrieve_tool(my_retriever_fn, default_top_k=4)
```

Or the whole suite — tools are included only when their backend is provided
(`retrieve` and `retrieve_with_filter` are always included when a retriever is):

```python
from benchmark.agentic.tools import build_tool_suite

tools = build_tool_suite(
    retriever=retrieve_fn,            # required; signature (query, top_k, filters=None)
    alternate_retriever=alt_fn,       # optional
    reranker=rerank_fn,               # optional; (query, [Document], top_k) -> [Document]
    llm=llm_fn,                       # optional; str -> str
    hypothesize=hyp_fn,               # optional; enables hyde_retrieve
    grader=grade_fn,                  # optional; (query, doc) -> bool
    answer_checker=check_fn,          # optional; (answer, contexts) -> bool
    search_fn=search,                 # optional; default: DuckDuckGo
    fetch_fn=fetch,                   # optional; default: requests.get
    allowed_domains=["https://..."],  # optional fetch_url allowlist
    default_top_k=4,
)
```

## 2. The agent adapter

`benchmark/adapters/agentic.py` — `AgenticRagAdapter`

Implements the framework's black-box `RagSystemAdapter` protocol
(`prepare` / `answer` / `cleanup`) and component injection
(`supports_components` → retriever, reranker, llm).

**Loop** (per question):

1. Bind the tool schemas to the LLM, build `[SystemMessage, HumanMessage]`.
2. Each round: invoke the model. No tool calls → its text is the final answer.
3. Tool calls are dispatched by name. **Errors never crash the run**: unknown
   tools or failing backends are recorded (`ok: false`, `error: "...")` and an
   error `ToolMessage` goes back to the model, which can recover.
4. Results from retrieval tools (`retrieve`, `retrieve_with_filter`,
   `retrieve_alternate`, `hyde_retrieve`) become the answer's `contexts`.
5. Rounds are bounded by `max_rounds`; exhaustion raises `RuntimeError`
   (diagnostics are still stored on the adapter as `_last_diagnostics`).

**Diagnostics** (in `RagSystemOutput.diagnostics`):

```python
{
    "execution_mode": "agentic",
    "agent_rounds": 3,
    "agent_tool_calls": 4,
    "agent_retrieved": true,
    "agent_model_seconds": 2.1,
    "agent_tokens_input": 812, "agent_tokens_output": 130, "agent_tokens_total": 942,
    "agent_tools_used": ["retrieve", "rewrite_query"],
    "tool_call_records": [
        {"tool": "retrieve", "round": 1, "arguments": {"query": "...", "top_k": 4},
         "ok": true, "error": null, "total_seconds": 0.31, "result_chars": 1893},
        ...
    ],
}
```

### Registration

Registered in the adapter registry as `"agentic"`:

```python
from benchmark.adapters import get_rag_adapter
adapter = get_rag_adapter(config)   # config.rag_system_adapter == "agentic"
```

## 3. Tool-call metrics

`benchmark/agentic/metrics.py`

`ToolCallRecord` — one observed invocation. Use `.as_dict()` /
`.from_dict()` for serialization (diagnostics store dicts).

`compute_tool_call_metrics(records, *, expected_tools=None, available_tools=None)`:

| Metric | Meaning | Needs |
|---|---|---|
| `tool_calls_total`, `calls_by_tool` | Call volume + usage distribution | — |
| `invalid_call_count` / `invalid_call_rate` | Calls that errored or were invalid | — |
| `redundant_call_count` / `redundant_call_rate` | Repeated identical (tool + args) calls | — |
| `hallucinated_tool_count` / `hallucinated_tool_rate` | Calls to non-existent tools | `available_tools` |
| `tool_selection_precision` | Fraction of successful calls that targeted an expected tool | `expected_tools` (gold labels) |
| `tool_selection_recall` | Fraction of expected tools actually used | `expected_tools` |
| `tool_seconds_total` | Total time spent inside tools | — |

`aggregate_tool_call_metrics(diagnostics)` — run-level aggregation over
per-sample diagnostics dicts; mean is over all samples passed (pre-filter to
agentic samples, mirroring `aggregate_agent_metrics`).

Existing efficiency metrics in `benchmark/adapters/mcp.py:aggregate_agent_metrics`
(`agent_rounds_mean/max`, `agent_tool_calls_mean/max`,
`agent_rounds_exhausted_count`, token/second sums) also apply — the adapter
emits the same diagnostic keys.

## 4. Reporting integration

`benchmark/reporting/comparisons.py:_comparison_rows` extracts, per sample,
from `tool_call_records`:

- `tool_calls_total`
- `invalid_call_count`
- `redundant_call_count`
- `tool_seconds_total`

These become normal comparison columns, so config-vs-baseline comparisons get
the same bootstrap confidence intervals as quality metrics.

## 5. Configuration

New `BenchmarkConfig` fields (config.py):

| Field | Env var | Default | Meaning |
|---|---|---|---|
| `agentic_max_rounds` | `AGENTIC_MAX_ROUNDS` | `4` | Max agent rounds per question (validated > 0; included in run config hash) |
| `agentic_system_prompt` | `AGENTIC_SYSTEM_PROMPT` | `None` | Custom system prompt; falls back to a built-in RAG-oriented default |

`rag_system_adapter` is already an experiment-matrix axis, so you can compare
`internal` vs `mcp@agentic` vs `agentic` in one matrix.

## 6. How to run a benchmark

```bash
export RAG_SYSTEM_ADAPTER=agentic
export AGENTIC_MAX_ROUNDS=4
# optional: AGENTIC_SYSTEM_PROMPT="..."
python main.py
```

The adapter builds its retriever/reranker/LLM via component injection
(`RAG_ADAPTER_ACCEPTS=retriever,reranker,llm`) or from config, then runs the
agent on every question. Results contain answer-quality metrics (RAGAS +
custom) plus the tool-call columns in comparisons.

### Programmatic use

```python
from benchmark.adapters.agentic import AgenticRagAdapter
from benchmark.adapters import get_rag_adapter

# from config (builds LLM via benchmark.generation.get_llm)
adapter = AgenticRagAdapter.from_config(config)
adapter.prepare(config, data, corpus)

# or fully injected (e.g. tests)
adapter = AgenticRagAdapter(llm=my_model, tools=my_tools, max_rounds=4)
out = adapter.answer({"question": "..."}, config)
metrics = out.diagnostics["tool_call_records"]
```

## 7. Tests

`tests/test_agentic_tools.py` (29 tests) — all follow the repo's convention
of hand-rolled fakes + `monkeypatch` (pattern from `tests/test_mcp_adapter.py`):

- `FakeRetriever` — records `(query, top_k)` calls and filters
- `FakeAgentModel` — records bound schemas, pops scripted responses
- `FakeResponse` — AIMessage-like with `tool_calls` + `usage_metadata`

Seams covered: each tool with fake backends, metric computation edge cases,
the agent loop (success / hallucinated tool / failing backend / round
exhaustion), comparison-row extraction, and registry lookup.

Run:

```bash
python -m pytest tests/test_agentic_tools.py -q
```

## 8. Known limitations / next steps

- **Expected-tool labels**: `tool_selection_precision/recall` require gold
  `expected_tools` per dataset row — not yet part of the dataset schema.
- Tool suite currently builds the main tools from component injection;
  `alternate_retriever`, `hyde`, `grader`/`answer_checker` backends need
  explicit wiring if you want them enabled in real runs.
- `web_search` uses DuckDuckGo via `langchain_community` (no API key);
  swap the injectable `search_fn` for Tavily/Exa if preferred.
