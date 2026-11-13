# Environment Variables Reference

Full reference for every environment variable the framework reads. YAML-first
runs should set `BENCHMARK_CONFIG_FILE` and keep secrets/service URLs in
`.env`. Workflow fields such as datasets, models, retrieval, chunking, prompt
templates, vector backend, and evaluator settings belong in `experiments/*.yaml`
— every manifest field not set falls back to its env var.

## Core

| Variable | Description |
| --- | --- |
| `BENCHMARK_CONFIG_FILE` | JSON/YAML manifest for `python main.py` and `python -m benchmark.worker`; required for `python main.py` unless a manifest path is passed explicitly. |
| `RAG_SYSTEM_ADAPTER` | `internal`, `http`, `mcp`, `ragflow` (managed, alias `optimaiserag`), or any adapter registered via a plugin module (`RAG_ADAPTER_MODULES`); defaults to `internal`. |
| `RAG_ADAPTER_MODULES` | Optional comma-separated Python modules to import before RAG adapter validation. There is no manifest-level field for plugin modules. |

## HTTP adapter

| Variable | Description |
| --- | --- |
| `RAG_HTTP_ENDPOINT_URL` | Required when `RAG_SYSTEM_ADAPTER=http`. |
| `RAG_HTTP_TIMEOUT_SECONDS` | HTTP request timeout; defaults to `60`. |
| `RAG_HTTP_ANSWER_FIELD` | Dotted response path for the answer; defaults to `answer`. |
| `RAG_HTTP_CONTEXTS_FIELD` | Dotted response path for contexts; defaults to `contexts`. |
| `RAG_HTTP_METADATA_FIELD` | Dotted response path for retrieval metadata; defaults to `metadata`. |
| `RAG_HTTP_TIMINGS_FIELD` | Dotted response path for timing data; defaults to `timings`. |

Details, request/response contract, auth: [External_RAG_System_Usage.md](External_RAG_System_Usage.md).

## Document parsers

| Variable | Description |
| --- | --- |
| `PARSER_ADAPTER` | Document parser adapter name (`http` or a registered plugin name); unset means no parser. |
| `PARSER_HTTP_ENDPOINT_URL` | Required when `PARSER_ADAPTER=http`; OpenAI-compatible `chat/completions` endpoint. |
| `PARSER_HTTP_MODEL` | Model ID sent in chat-completions payloads. |
| `PARSER_HTTP_PROMPT` | Per-page parsing instruction; defaults to a Markdown-conversion prompt. |
| `PARSER_HTTP_TIMEOUT_SECONDS` | Per-page HTTP timeout; defaults to `60`. |
| `PARSER_HTTP_HEADERS` | Extra JSON headers for parser requests. |
| `PARSER_PLUGIN_MODULE` / `PARSER_PLUGIN_ATTRIBUTE` | Import path and class name of a local parser plugin (takes precedence over `PARSER_ADAPTER`). |
| `PARSER_ADAPTER_MODULES` | Optional comma-separated Python modules to import before parser adapter validation. |
| `PARSER_VERSION` | Optional parser version pinned into run metadata. |

Details: [Benchmark_Your_Parser.md](Benchmark_Your_Parser.md).

## Internal retrieval pipeline

| Variable | Description |
| --- | --- |
| `RETRIEVAL_STRATEGY` | `similarity` (default), `mmr`, or `hybrid` (dense + BM25 fusion). |
| `RETRIEVAL_HYBRID_ALPHA` | Hybrid fusion weight: `1.0` pure dense, `0.0` pure BM25; defaults to `0.5`. |
| `RETRIEVAL_TOP_K` | Number of retrieved passages per question. |
| `RETRIEVAL_FETCH_K` | Candidates fetched from the vector store before filtering; defaults to a multiple of `RETRIEVAL_TOP_K`. |
| `RETRIEVAL_CANDIDATE_K` | Hybrid retrieval dense candidate pool size; defaults to `1024`. |
| `RETRIEVAL_VECTOR_SIMILARITY_WEIGHT` | Weight of the dense similarity score in hybrid RRF fusion; between `0` and `1`, defaults to `0.3`. |
| `RETRIEVAL_MMR_LAMBDA` | MMR diversity/divergence trade-off for `RETRIEVAL_STRATEGY=mmr`; between `0` and `1`, defaults to `0.5`. |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | Minimum similarity score for retrieved contexts; between `0` and `1`, defaults to `0.2`. |
| `RETRIEVAL_USE_HYDE` | Set to `true` to rewrite the question into a hypothetical document before retrieval (HyDE). |
| `RETRIEVAL_MULTIHOP` | Set to `true` for iterative multi-hop retrieval: an LLM generates follow-up questions from the first-round contexts and retrieves again. |
| `RETRIEVAL_MULTIHOP_ROUNDS` | Total retrieval rounds when `RETRIEVAL_MULTIHOP=true`; defaults to `1` (one follow-up round). |
| `RERANKER_MODELS` | Comma-separated reranker models applied after retrieval (empty disables reranking). |
| `RERANKER_TOP_K` | Number of passages kept after reranking; defaults to `3`. |

Multi-hop details: [multihop-retrieval.md](multihop-retrieval.md). Hybrid
retrieval background: `doc/changes/2026-11-06-hybrid-retrieval.md`.

## Datasets and corpus

| Variable | Description |
| --- | --- |
| `DATASET_NAME` | Dataset adapter to benchmark; built-ins include `jsonl` and `csv` for local files. |
| `DATASET_PATH` | Required for `DATASET_NAME=jsonl` or `csv`. |
| `DATASET_QUESTION_FIELD` | Local dataset question field; defaults to `question`. |
| `DATASET_GROUND_TRUTH_FIELD` | Local dataset answer field; defaults to `ground_truth`. |
| `DATASET_CONTEXT_FIELD` | Local dataset context field; defaults to `context`. |
| `DATASET_METADATA_FIELD` | Local dataset metadata field; defaults to `metadata`. |
| `DATASET_SUBSET` | Optional dataset subset/config. |
| `DATASET_SAMPLE_SIZE` | Number of benchmark samples. |
| `DATASET_SPLIT` | HuggingFace split to load (manifest: `dataset: split:`); unset keeps the adapter's default split. A split missing from the dataset is an error. |
| `DATASET_LICENSE` | Dataset/corpus license pinned into run metadata; defaults to the dataset adapter's license when it declares one. |
| `CORPUS_PARSER` | Registered parser name for RAG corpus ingestion; when set, documents under `DATASET_CORPUS_PATH` are parsed to Markdown before chunking (requires `DATASET_NAME=jsonl-shared`). |

## MCP adapter

| Variable | Description |
| --- | --- |
| `MCP_TRANSPORT` | `stdio` or `streamable_http`; defaults to `stdio`. |
| `MCP_COMMAND` / `MCP_ARGS_JSON` | Executable and JSON argument array for a stdio MCP server. |
| `MCP_SERVER_URL` | MCP endpoint required by `streamable_http`. |
| `MCP_TOOL_NAME` | Tool invoked for each benchmark question. |
| `MCP_RESULT_MODE` | `context` (MCP evidence + framework LLM) or `answer` (tool is full QA system). |
| `MCP_RESULT_FIELD` | Optional dotted path inside structured tool output, such as `contexts`. |
| `MCP_EXECUTION_MODE` | `fixed` or `agentic`; defaults to `fixed`. |
| `MCP_ALLOWED_TOOLS_JSON` | Optional JSON tool-name allowlist for agentic mode. |
| `MCP_MAX_AGENT_ROUNDS` | Maximum model/tool rounds; defaults to `4`. |
| `MCP_MAX_RETRIES` | Transport exception retries per tool call; defaults to `1`. |
| `MCP_CONTINUE_ON_ERROR` | Record a failed sample and continue instead of aborting; defaults to `true`. |
| `MCP_CORPUS_PATH` | Declares the MCP corpus for RAG-vs-MCP fairness validation. |

Details and full metric reference: [mcp-benchmarking.md](mcp-benchmarking.md).

## Enterprise RAG Blueprint plugins

| Variable | Description |
| --- | --- |
| `ENTERPRISE_RAG_BLUEPRINT_PATH` | Blueprint checkout for the `enterprise_rag` / `enterprise_rag_mcp` plugins. |
| `ENTERPRISE_RAG_MCP_TRANSPORT` | `inprocess` (default) or `http` for `enterprise_rag_mcp`. |
| `ENTERPRISE_RAG_MCP_ENDPOINT_URL` | Blueprint `/call_llm` URL; required for the `http` transport. |
| `ENTERPRISE_RAG_MCP_ENABLED` | Send `use_mcp: true`; `false` benchmarks the Blueprint without MCP. Defaults to `true`. |
| `ENTERPRISE_RAG_MCP_RETRIEVAL` | Blueprint retrieval with MCP on: `context` (default, documents preloaded), `tool` (the model calls a document-search tool) or `none` (MCP alone). |
| `ENTERPRISE_RAG_MCP_DECLINE_JUDGE` | `true` lets the critic LLM (`EVAL_CRITIC_LLM`) judge declines next to the phrase list. Defaults to `false`. |
| `ENTERPRISE_RAG_MCP_ROLES` | Comma-separated roles in the signed identity; defaults to `General`. |
| `ENTERPRISE_RAG_MCP_SUBJECT` | Identity subject; defaults to `benchmark`. |
| `ENTERPRISE_RAG_MCP_PROMPT_KEY` | `rag_source` (default), `rag_summary` or `rag_weekly_report`. |
| `ENTERPRISE_RAG_MCP_USED_MODEL` | Optional Blueprint LLM source key sent as `used_model`. |
| `ENTERPRISE_RAG_MCP_TIMEOUT_SECONDS` | Blueprint request timeout; defaults to `120`. |
| `ENTERPRISE_RAG_MCP_INDEX_CORPUS` | Rebuild the Blueprint index from the dataset corpus; defaults to `true`. |
| `ENTERPRISE_RAG_MCP_APPLY_SETTINGS` | Run the Blueprint with the config's LLM, embedding model, top-k and temperature (in-process only); defaults to `true`. |
| `ENTERPRISE_RAG_MCP_LLM_API_KEY_ENV` | Name of the env var holding the Blueprint LLM's API key (e.g. `ANTHROPIC_API_KEY`); unset uses the generator LLM key. |
| `ENTERPRISE_RAG_MCP_QUERY_REFINEMENT` | Blueprint query refinement with applied settings; defaults to `true`. |
| `ENTERPRISE_RAG_MCP_GUARDRAILS` | Blueprint input guardrails with applied settings; defaults to `true`. |
| `MCP_IDENTITY_SECRET` | Blueprint identity signing secret (≥ 32 chars) for `enterprise_rag_mcp`; must match the Blueprint. |

Details: [mcp-host-benchmarking.md](mcp-host-benchmarking.md),
[enterprise-rag-blueprint-setup.md](enterprise-rag-blueprint-setup.md).

## Generation and vector store

| Variable | Description |
| --- | --- |
| `VECTOR_DB_BACKEND` | Vector store backend for the internal pipeline; defaults to `chroma`. |
| `LANCEDB_PATH` | Local path for the `lance` vector store backend. |
| `GENERATION_TEMPERATURE` | Internal generator sampling temperature; between `0` and `2`, defaults to `0.1`. |
| `GENERATION_TOP_P` | Internal generator nucleus-sampling cut-off; between `0` and `1`, defaults to `0.3`. |
| `MAX_NEW_TOKENS` | Maximum tokens the internal generator may produce; defaults to `256`. |
| `LLM_ANSWER_STRIP_MODE` | How generated answers are stripped before scoring: `tags_only` (default), `full`, or `off`. |
| `LLM_ANSWER_VALUE_FALLBACK` | Fall back to the raw answer when tag stripping finds nothing; defaults to `true`. |

## Evaluation

| Variable | Description |
| --- | --- |
| `RAGAS_ENABLED` | Set to `false` to skip RAGAS critic metrics. |
| `CUSTOM_METRICS_ENABLED` | Set to `false` to skip custom embedding/BERTScore metrics. |
| `EVAL_CRITIC_LLM` | Critic model used for RAGAS evaluation. |
| `EVAL_CRITIC_EMBEDDING` | Embedding model used by evaluator metrics. |
| `EVAL_CRITIC_THINKING_CONTROL` | How reasoning is disabled on an OpenAI-compatible critic: `chat_template_kwargs` (default, vLLM/SGLang), `reasoning_effort_low` (e.g. GLM behind LiteLLM, which leaks reasoning into the answer otherwise), or `none`. |
| `EVAL_CRITIC_TEMPERATURE` | Critic sampling temperature; defaults to `0.0`. Set to `none` to omit it from requests for models that reject it (e.g. Claude Opus 5.5). |
| `EVAL_CRITIC_MAX_WORKERS` | Parallel RAGAS critic jobs; defaults to `1` (local models). Raise for hosted endpoints. |
| `EVAL_CRITIC_MAX_RETRIES` | RAGAS retries per failed critic call (e.g. proxy 500s); defaults to `2`. |

## Cost and resource monitoring

| Variable | Description |
| --- | --- |
| `ELECTRICITY_PRICE_EUR_PER_KWH` | Electricity price (€/kWh) for local energy-cost estimates. Falls back to `ELECTRICITY_PRICE_USD_PER_KWH`. |
| `ELECTRICITY_PRICE_USD_PER_KWH` | Electricity price (USD/kWh) for local energy-cost estimates. |
| `BENCHMARK_RESOURCE_MONITOR` | `true` to sample GPU power/CPU/mem to CSV traces (enables per-second `gpu_power_w`). |

## Benchmark stages

`BENCHMARK_STAGE=index` is only supported by the built-in adapter. External
HTTP systems own their own indexing lifecycle.

`BENCHMARK_STAGE=parsing` runs the parser evaluation runner (OCR-06): the
manifest declares `parser_adapters` × document GT datasets, the worker parses
and scores each document resumably (per-document checkpoints), and a
leaderboard report (`parsing_leaderboard_<timestamp>.json` with per-parser,
per-category, and per-page text/TEDS scores) lands in the run dir plus MLflow
as nested child runs.
