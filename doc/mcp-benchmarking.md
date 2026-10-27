# Benchmarking MCP Systems

Added 2026-10-23. How to evaluate a system that is reached through the Model
Context Protocol (MCP), how to configure it, what every metric means, and how
to read the results without being misled.

Related: [12-mcp-in-rag-state-of-the-art.md](12-mcp-in-rag-state-of-the-art.md)
(research background), [changes/2026-10-23-mcp-metrics-and-fairness.md](changes/2026-10-23-mcp-metrics-and-fairness.md)
(what changed and why).

---

## 1. What is measured

MCP is a transport contract, not a retrieval algorithm. A benchmark therefore
measures the **concrete MCP server and tool** you configure, through the
`mcp` adapter (`benchmark/adapters/mcp.py`). The adapter is a normal RAG
system adapter, so an MCP run is scored with the same quality metrics as the
internal pipeline, plus MCP-specific reliability, latency, and tool-use
metrics.

Two independent switches define the experiment:

| Switch | Value | Behaviour |
|---|---|---|
| `mcp_result_mode` | `context` | The tool returns evidence; the framework's configured LLM writes the answer. Isolates **retrieval**: internal RAG vs MCP retrieval with the same generator. |
| | `answer` | The tool returns the final answer. Benchmarks an MCP-only QA system end to end; no retrieved contexts. |
| `mcp_execution_mode` | `fixed` | `mcp_tool_name` is called exactly once per question. Deterministic. |
| | `agentic` | The configured chat model sees the allow-listed tools and decides which to call, for up to `mcp_max_agent_rounds` rounds. Measures **tool use** as well as retrieval. |

Agentic mode needs a model that supports tool calling and is large enough to
use it: `qwen2.5:0.5b` calls tools with invalid arguments and fails most
questions; start at `gemma3:4b`-class models.

---

## 2. Quick start

Three ready-made manifests compare the internal pipeline with the example MCP
server (`examples/corpus_mcp_server.py`) on the `northstar_facilities` dataset:

```bash
source .venv/bin/activate
python main.py experiments/rag-vs-mcp.yaml          # context + fixed
python main.py experiments/rag-vs-mcp-agentic.yaml  # context + agentic (search, lookup)
python main.py experiments/rag-vs-mcp-answer.yaml   # answer mode (MCP is the QA system)
```

Smoke test (no RAGAS cost, no MLflow, nothing written to `results/`): copy a
manifest, set `sample_size: 5`, add `ragas_enabled: false` under `settings`,
and run

```bash
python -m benchmark.worker run my-copy.yaml --run-dir /tmp/opencode/mcp-smoke --no-mlflow
```

---

## 3. Configuration

**Rule:** workflow settings live in the YAML manifest; secrets and
machine-local URLs live in `.env`. YAML values override `.env`, but **every
field the YAML omits falls back to `.env`** (see §6.2).

### 3.1 Manifest (`experiments/*.yaml`)

```yaml
experiment_name: my-mcp

dataset:
  name: jsonl-shared            # required for RAG-vs-MCP, see §5
  subset: null
  path: datasets/northstar_facilities/questions.jsonl
  corpus_path: datasets/northstar_facilities/corpus
  sample_size: 20

settings:
  # Pin anything a legacy .env matrix could inject.
  dataset_question_field: question
  dataset_ground_truth_field: ground_truth
  retrieval_multihop: false
  retrieval_use_hyde: false
  retrieval_strategy: similarity
  retrieval_top_k: 5
  max_new_tokens: 512
  custom_retrieval_metrics_mode: gold_doc   # hit@k / recall@k against gold doc ids

  # MCP server: local process
  mcp_transport: stdio
  mcp_command: python
  mcp_args_json:
    - examples/corpus_mcp_server.py
    - --corpus
    - datasets/northstar_facilities/corpus
    - --chunk-size       # keep equal to chunk_sizes below
    - "1000"
    - --chunk-overlap    # keep equal to chunk_overlaps below
    - "200"
  # MCP server: remote -> mcp_transport: streamable_http, URL in .env

  mcp_tool_name: search
  mcp_question_argument: query
  mcp_top_k_argument: top_k
  mcp_result_mode: context
  mcp_result_field: contexts
  mcp_execution_mode: fixed           # or agentic
  mcp_allowed_tools_json: [search, lookup]   # agentic only
  mcp_max_agent_rounds: 4
  mcp_timeout_seconds: 60
  mcp_max_retries: 1
  mcp_continue_on_error: true
  mcp_corpus_path: datasets/northstar_facilities/corpus
  mcp_enforce_fairness: true

matrix:
  rag_system_adapter: [internal, mcp]
  llm_models: [ollama:gemma3:4b]
  embedding_models: [nomic-embed-text:latest]
  chunking_strategies: [recursive]
  chunk_sizes: [1000]
  chunk_overlaps: [200]
  prompt_templates: [concise]
  reranker_models: [null]
```

### 3.2 MCP settings reference

| Setting | Env var | Meaning |
|---|---|---|
| `mcp_transport` | `MCP_TRANSPORT` | `stdio` (spawn a process) or `streamable_http` (remote). |
| `mcp_command`, `mcp_args_json` | `MCP_COMMAND`, `MCP_ARGS_JSON` | Executable and argument list for stdio. No shell is involved. |
| `mcp_server_url` | `MCP_SERVER_URL` | Endpoint for `streamable_http`. Put it in `.env`. |
| `mcp_http_headers_json` | `MCP_HTTP_HEADERS_JSON` | JSON object of HTTP headers (auth). Put it in `.env`. |
| `mcp_env_vars` | `MCP_ENV_VARS` | Comma-separated variable **names** forwarded to a stdio server; nothing else from your environment reaches it. |
| `mcp_tool_name` | `MCP_TOOL_NAME` | Tool called in fixed mode; in agentic mode the tool that receives `top_k`. |
| `mcp_question_argument` | `MCP_QUESTION_ARGUMENT` | Tool argument that receives the question. |
| `mcp_top_k_argument` | `MCP_TOP_K_ARGUMENT` | Optional tool argument that receives `retrieval_top_k`. |
| `mcp_tool_arguments_json` | `MCP_TOOL_ARGUMENTS_JSON` | Static extra arguments sent on every call. |
| `mcp_result_mode` | `MCP_RESULT_MODE` | `context` or `answer` (§1). |
| `mcp_result_field` | `MCP_RESULT_FIELD` | Dotted path inside the tool's structured output, e.g. `contexts`. |
| `mcp_execution_mode` | `MCP_EXECUTION_MODE` | `fixed` or `agentic` (§1). |
| `mcp_allowed_tools_json` | `MCP_ALLOWED_TOOLS_JSON` | Tools the agent may call; must exist on the server. |
| `mcp_max_agent_rounds` | `MCP_MAX_AGENT_ROUNDS` | Model/tool rounds before the question fails. Default 4. |
| `mcp_timeout_seconds` | `MCP_TIMEOUT_SECONDS` | Per-operation timeout. |
| `mcp_max_retries`, `mcp_retry_backoff_seconds` | `MCP_MAX_RETRIES`, `MCP_RETRY_BACKOFF_SECONDS` | Retries for transport exceptions (not for tool errors). |
| `mcp_continue_on_error` | `MCP_CONTINUE_ON_ERROR` | Record a failed question and continue (default) instead of aborting the run. |
| `mcp_corpus_path` | `MCP_CORPUS_PATH` | Declares the corpus the server searches, for the fairness check. |
| `mcp_enforce_fairness` | `MCP_ENFORCE_FAIRNESS` | Fairness validation (§5). Default `true`. |

### 3.3 `.env` (remote servers and secrets only)

```bash
MCP_SERVER_URL=https://mcp.example.com/mcp
MCP_HTTP_HEADERS_JSON={"Authorization": "Bearer <token>"}

# stdio server that needs a secret: forward it by name
MCP_ENV_VARS=MY_SERVER_API_KEY
MY_SERVER_API_KEY=<secret>
```

### 3.4 The example server

`examples/corpus_mcp_server.py` is a lexical (TF-IDF-style keyword) search
server over a folder of `.md`/`.txt` files. Tools:

- `search(query, top_k=5)` returns `{"contexts": [{text, source, score, chunk_index?}]}`.
- `lookup(source_id)` returns one whole document by file name or stem.
- `answer(query)` returns the best-matching sentence (used by answer mode).

Options: `--corpus DIR` (required), `--chunk-size N` and `--chunk-overlap M`.
With `--chunk-size 0` (default) `search` returns whole documents. With a chunk
size it splits documents exactly like the internal `recursive` strategy
(`RecursiveCharacterTextSplitter`, fragments under 50 characters dropped).
Always pass the same values as the manifest's `chunk_sizes`/`chunk_overlaps`;
the framework cannot read server arguments, so it cannot check this for you.

Result items carry `source`; the adapter turns it into `doc_id`/`source_id`
(file stem), which is what gold-document retrieval metrics match against.
Your own server should return a `source`, `source_id`, or `doc_id` per item
for the same reason.

---

## 4. Metrics

### 4.1 Quality (every adapter)

- **RAGAS**: faithfulness, answer relevancy, answer correctness, context
  precision, context recall, semantic similarity. Faithfulness and the two
  context metrics need retrieved contexts, so in `answer` mode only the
  answer metrics are meaningful.
- **Custom metrics** (`custom_metrics_enabled`): ROUGE-L, BLEU, METEOR,
  BERTScore, embedding-based IR heuristics.
- **Gold-document retrieval** (`custom_retrieval_metrics_mode: gold_doc`):
  `hit@k`, `recall@k`, `ndcg@k`, span metrics, matched on `doc_id`.

### 4.2 Evidence budget (every adapter)

Computed by the runner for every system in the same way, so internal and MCP
are directly comparable. Reported in JSON/CSV and MLflow without prefix.

| Metric | Meaning |
|---|---|
| `avg_contexts_per_question` | Mean number of retrieved passages handed to the generator. |
| `max_contexts_per_question` | Largest number for a single question. |
| `avg_context_chars_per_question` | Mean total characters of those passages. |

**Check these before comparing retrieval or answer quality.** A system that
receives four times the text will look better on recall for reasons that have
nothing to do with retrieval quality. Characters, not tokens: deterministic,
tokenizer-independent, and identical across systems.

### 4.3 MCP adapter metrics

Run-level aggregates from `McpRagAdapter.aggregate_metrics`. MLflow and
ClearML log numeric values as `adapter_<name>`; the JSON report keeps all of
them under `adapter_metrics`, including the per-tool dictionary.

**Reliability**

| Metric | Meaning |
|---|---|
| `sample_count`, `successful_samples` | Questions run / answered without error or empty response. |
| `failure_count`, `failure_rate` | Questions with an error or an empty response. |
| `transport_failure_count` | Connection broke or timed out on every retry. Server availability problem. |
| `tool_error_count` | The server answered with `isError` (it rejected the call). |
| `agent_failure_count` | The agent called a non-allowed tool, sent non-object arguments, or ran out of rounds. Model problem. |
| `other_failure_count` | Anything else, e.g. the generator call failed. |
| `timed_out_sample_count`, `timeout_attempt_count` | Questions that ended in a timeout / individual timed-out attempts. |
| `retry_count` | Transport retries across all calls. |
| `empty_response_count` | Tool or agent returned nothing. |
| `partial_completion_count` | Failed after at least one successful tool call. |

**Latency**

| Metric | Meaning |
|---|---|
| `connection_seconds` | Session start plus MCP handshake (paid once per config). |
| `cold_tool_seconds` | Tool time of the first question. |
| `warm_tool_mean_seconds` | Mean tool time of the remaining questions. |
| `tool_latency_p50_seconds`, `tool_latency_p95_seconds`, `tool_latency_max_seconds` | Nearest-rank percentiles over **individual calls**. |

Stage timings `stage_mcp_connection_seconds`, `stage_mcp_tool_seconds`, and
`stage_mcp_generation_seconds` split total time into connection, tool, and
LLM work.

**Tool use**

| Metric | Meaning |
|---|---|
| `tool_call_count`, `tool_call_counts` | Total calls and calls per tool name. |
| `tool_calls_per_sample_mean`, `tool_calls_per_sample_max` | Calls per question. 1.0 in fixed mode by construction. |
| `agent_rounds_mean`, `agent_rounds_max` | Model turns until the final answer (agentic only; `None` in fixed mode). |
| `duplicate_tool_call_count`, `duplicate_tool_call_rate` | Calls repeating an earlier tool + arguments pair **within the same question**. A loop symptom. |
| `schema_invalid_call_count`, `schema_invalid_call_rate` | Calls whose arguments violate the tool's declared JSON input schema. These calls are still sent; the server decides. |

**Overhead**

| Metric | Meaning |
|---|---|
| `tool_schema_chars_total`, `tool_schema_chars_per_sample_mean` | Characters of tool definitions sent to the model, counted once per model call (agentic only). Grows with the number of tools and rounds and is paid in input tokens. |

### 4.4 Per-question diagnostics

Each question's `adapter_diagnostics` in the JSON report contains the raw
material for the aggregates and for debugging:

- `tool_calls[]`: `tool`, `arguments` (keys containing *authorization,
  password, secret, token, key* are redacted), `attempts`, `attempt_seconds`,
  `total_seconds`, `is_error`, `attempt_errors`, `timeout_attempts`,
  `schema_valid`, `schema_errors` (path and rule only, never argument values).
- `failure_kind` (`transport` | `tool_error` | `agent` | `other`), `error`,
  `error_type`, `timeout`, `partial_completion`, `empty_response`.
- `cold_start`, `connection_seconds`, `total_seconds`, `generation_seconds`,
  `retry_count`.
- Agentic only: `agent_rounds`, `model_calls`, `tool_schema_chars`.

### 4.5 Not measured (yet)

Tool-use **correctness against expected calls** (RAGAS tool-call accuracy/F1,
TruLens tool selection). It needs an `expected_tool_calls` field in the
dataset that no dataset has. Final-answer correctness is the current proxy.

---

## 5. Fairness validation

When a matrix contains both `internal` and `mcp` (and `mcp_enforce_fairness`
is on), `build_configs_from_spec` rejects the experiment unless:

1. the dataset is **`jsonl-shared`** (see below),
2. `dataset.corpus_path` and `mcp_corpus_path` resolve to the same directory,
3. both systems share dataset, subset, path, sample size, and `retrieval_top_k`,
4. in `context` mode they also share LLM provider/model, prompt template, and
   `max_new_tokens`.

**Why `jsonl-shared`.** Only `jsonl-shared` builds the internal index from
`corpus_path`. With plain `jsonl` the path is silently ignored and the internal
pipeline indexes only the sampled questions' own `context` fields: a tiny,
answer-guaranteed index, while the MCP server searches the whole corpus. Equal
paths, different corpora. Until 2026-10-23 the shipped manifests had exactly
this flaw.

**What validation cannot see.** Server-side settings such as the example
server's `--chunk-size`. Keep them in sync by hand and confirm with the
evidence-budget metrics (§4.2).

Set `mcp_enforce_fairness: false` only for intentionally asymmetric
experiments, and say so in the experiment name.

---

## 6. Reading results and troubleshooting

### 6.1 Checklist before comparing systems

1. `avg_contexts_per_question` and `avg_context_chars_per_question` are close
   between systems.
2. `failure_rate` is low; if not, `failure_kind` tells you whose fault it is
   (`transport` = server/network, `tool_error` = server rejected the call,
   `agent` = model).
3. Agentic: `schema_invalid_call_rate` and `duplicate_tool_call_rate` near 0;
   `agent_rounds_max` below `mcp_max_agent_rounds`.
4. The planned dataset is `jsonl-shared/` (`python -m benchmark.worker plan <manifest>`).

### 6.2 `.env` values leaking into a manifest

Fields absent from the YAML come from `.env`. A `.env` prepared for HotpotQA
(`RETRIEVAL_MULTIHOP=true`, `DATASET_SUBSET=distractor`,
`DATASET_GROUND_TRUTH_FIELD=answer`) silently turns on multi-hop retrieval or
breaks dataset loading ("missing required field 'answer'"). Symptoms: an
`_mh2` suffix in config names, `jsonl/distractor` in the plan. Pin the fields
in the manifest (the shipped ones do) or remove them from `.env`.

### 6.3 Known caveats

- **Weak agent models**: very small models answer with clarification
  questions; the concise-answer fallback can reduce those to fragments such as
  `-001`, which count as valid answers and inflate `successful_samples`.
- **Span metrics in agentic runs** are averaged over questions that have
  contexts only, so with few such questions they read too high (pre-existing
  behaviour of the gold-doc metrics).
- **Lexical vs dense retrieval**: on the templated `northstar_facilities`
  corpus the keyword server finds the gold document far more often than
  dense retrieval (hit@1 1.0 vs 0.4 in a 5-question check at equal evidence
  budget), because questions name the facility ID explicitly. This is a real
  result, not a fairness problem.

---

## 7. Where the code lives

| Concern | Location |
|---|---|
| Adapter, session, aggregation | `benchmark/adapters/mcp.py` |
| Evidence-budget metrics | `benchmark/orchestration/runner.py`, fields on `BenchmarkResultExtended` in `benchmark/reporting/models.py` |
| Fairness validation | `_validate_mcp_comparison_fairness` in `benchmark/orchestration/matrix.py` |
| Export / logging | `benchmark/reporting/exports.py`, `benchmark/tracking.py`, `benchmark/clearml_task.py` |
| Example server | `examples/corpus_mcp_server.py` |
| Tests | `tests/test_mcp_adapter.py`, `tests/test_corpus_mcp_server.py`, `tests/test_run_path_uniformity.py`, `tests/test_config.py` (fairness) |
