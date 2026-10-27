# Blueprint LLM, embedding and top-k from the manifest

Date: 2026-10-27

## Summary
The `enterprise_rag_mcp` adapter ignored `llm_models`, `embedding_models`,
`retrieval_top_k` and `generation_temperature`: the Blueprint answered with
the model from its own `.env`, embedded with a hard-coded
`nomic-embed-text` and retrieved a fixed 5 documents. Results were labelled
with a placeholder model. The Blueprint now reads these settings per
request, and the adapter sets them from the benchmark config, so the manifest
matrix compares Blueprint models, embeddings and top-k directly. The setup
guide now documents `.env` and manifest step by step.

## Changes

**Blueprint** (separate repo `Enterprise_RAG_Blueprint`, branch
`feat/mcp-integration`, uncommitted):
- `chain/handle_llms.py`: `load_llm_model` reads `MODEL_NAME` per call; new
  `retriever_top_k()` reads `RETRIEVER_TOP_K` (default 5), used by `call_llm`.
- `chain/mcp_integration/execution.py`: MCP path uses `retriever_top_k()`.
- `chain/retriever.py`, `loader/vector_store.py`: embedding model from
  `EMBEDDING_MODEL` (default `nomic-embed-text`); the retriever reads
  `OLLAMA_BASE_URL` per call; new `OLLAMA_API_KEY` is sent as Bearer token
  (Ollama behind an authenticating proxy, e.g. our Ollama proxy or the DGX
  Spark).
- `example.env`, `docs/mcp.md`: new variables documented.

**Framework**:
- `config.py`: `enterprise_rag_mcp_apply_settings` (default `true`),
  `enterprise_rag_mcp_query_refinement`, `enterprise_rag_mcp_guardrails`
  (default `true`), `enterprise_rag_mcp_llm_api_key_env`; env fallbacks
  `ENTERPRISE_RAG_MCP_*`. Applied settings are validated: in-process
  transport only, Ollama embeddings only, `ollama:`/`openai:` LLMs only,
  `openai:` needs a base URL. The three toggles join the run-name hash.
- `benchmark/orchestration/matrix.py`: the new booleans are coerced from
  manifest strings.
- `examples/enterprise_rag_plugin/mcp_adapter.py`: `blueprint_settings_env()`
  maps the config onto `MODEL_NAME`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `LLM_TEMPERATURE`, `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`,
  `RETRIEVER_TOP_K`, `DISABLE_QUERY_REFINEMENT`, `DISABLE_GUARDRAILS`;
  `ollama:` models go through Ollama's `/v1` API. The variables are set for
  each request and restored afterwards; `used_model` is sent as `openai`.
  The in-process transport rejects Blueprint checkouts that read settings
  only at import. The `nomic-embed-text`-only index check now applies only
  without applied settings and compares with the Blueprint's
  `EMBEDDING_MODEL`.
- `datasets/blueprint_mcp_host/blueprint_mcp_servers.template.json`: MCP
  server config template for `envsubst` (the Blueprint needs absolute paths).
- `experiments/enterprise-rag-mcp-host.yaml`: real model selection
  (`openai:nvidia/Qwen3.6-35B-A3B-NVFP4`; endpoints and Bearer keys from
  `.env` like every experiment), `generation_temperature: 0`, applied
  settings.
- Docs: new `doc/enterprise-rag-blueprint-setup.md` (step-by-step `.env` +
  manifest setup, endpoints with Bearer keys, `http` variant,
  troubleshooting); `doc/mcp-host-benchmarking.md` (settings mapping, links
  to the guide), README env table, `doc/00-README.md`.

## Behaviour changes and migration
- Applied settings are on by default. Manifests for `enterprise_rag_mcp`
  now run the Blueprint with their `llm_models` entry instead of the
  Blueprint `.env` model. Keep the old behaviour with
  `enterprise_rag_mcp_apply_settings: false` (required for the `http`
  transport).
- `generation_temperature` now reaches the Blueprint; the framework default
  is 0.1, the Blueprint default 0. Pin `generation_temperature: 0` to match
  earlier runs.
- The run-name hash of `enterprise_rag_mcp` configs changed; existing
  `results/runN` checkpoints of this adapter are not resumed.
- Needs the updated Blueprint checkout (older ones fail with a clear error
  when applied settings are on).

## Verification
- Blueprint: new tests for per-call `MODEL_NAME`, `EMBEDDING_MODEL` /
  `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` as Bearer (chain and loader), loader
  `EMBEDDING_MODEL`, `RETRIEVER_TOP_K` on the
  legacy and MCP paths, each written failing first. Chain suite + loader
  vector-store/tag/function tests: 312 passed (`pytest-asyncio` needed for
  the async MCP tests; minio-dependent loader tests not run, `minio` not
  installed).
- Framework: tests for the settings mapping (Ollama and OpenAI-compatible,
  named key variable), per-request env scoping and restore, untouched env
  without applied settings, index embedding checks, old-Blueprint rejection,
  config defaults/env/manifest/validation/run names, Ollama behind a Bearer
  proxy (LLM and embedding keys). Full suite: 708 passed.
- Live smoke run, in-process, 3 questions, MCP on, `ollama:qwen2.5:0.5b`,
  `nomic-embed-text`, `retrieval_top_k: 3`, RAGAS off: the Blueprint answered
  with qwen2.5:0.5b (loaded in Ollama, no call to the `.env` model),
  exactly 3 documents per answer, 2 tool calls, call success rate 1.0,
  0 failed requests. Second run with `openai:nvidia/Qwen3.6-35B-A3B-NVFP4`
  on optimaise (Bearer key from `.env`): 3 answers, 2 tool calls, call
  success rate 1.0, 0 failed requests.
- Bearer checks: optimaise and coassist `/v1/models` and the Ollama
  proxy's `/v1/models` and `/api/tags` answer with the `.env` keys; the
  Blueprint retriever embeds through the Ollama proxy with
  `qwen3-embedding:0.6b-fp16` with the key and gets 401 without it.

## Follow-ups
- From the review (pre-existing, not introduced here): the in-process
  transport's `enterprise_rag_mcp_timeout_seconds` does not bound the
  request (Starlette's test client ignores HTTPX timeouts), and
  `load_dotenv` of the Blueprint `.env` stays in the framework process for
  the rest of the run.
- Commit the Blueprint change on `feat/mcp-integration`.
- `openai:` models from two different endpoints need two manifests (one
  base URL per manifest).
- Full run of the 1000-question set with several models
  (`doc/experiment-todos.md` §7).
