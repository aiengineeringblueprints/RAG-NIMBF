# Evaluating the Enterprise RAG Blueprint (MCP host) — Setup Guide

Step-by-step setup for benchmarking the Enterprise RAG Blueprint with its MCP
integration: which files to change, which values go into `.env` and which
into the manifest, and how to select the Blueprint's LLMs and embedding
models from the manifest. Metrics, adapter internals and the reference
result are in [MCP Host Benchmarking](mcp-host-benchmarking.md).

## How it works in one paragraph

The framework imports the Blueprint chain in-process, rebuilds the
Blueprint's Chroma index from the dataset corpus, and sends every question to
the Blueprint's `/call_llm` endpoint, once with MCP off and once with MCP on.
The Blueprint retrieves, calls its MCP servers and answers by itself. With
`enterprise_rag_mcp_apply_settings: true` the framework first sets the
Blueprint's model variables from the benchmark config for that request, so
`llm_models`, `embedding_models`, `retrieval_top_k` and
`generation_temperature` in the manifest decide what the Blueprint runs with.

## Prerequisites

- Blueprint checkout on branch `feat/mcp-integration`, including the
  per-request settings change of 2026-10-27 (`chain/handle_llms.py` has
  `retriever_top_k`). Older checkouts are rejected with
  "reads its model settings only at import".
- Blueprint chain dependencies in the framework venv (once):

  ```bash
  source .venv/bin/activate
  pip install -r ../Enterprise_RAG_Blueprint/chain/requirements.txt
  ```

- Every embedding model in the manifest must exist on the embedding Ollama
  host (`ollama pull nomic-embed-text` locally, or check `/api/tags` of the
  remote host).
- Every LLM must support tool calling for MCP on.

## Step 1: Blueprint MCP server config

The Blueprint reads its MCP servers from a JSON file and requires absolute
paths for stdio servers. Generate the file for the dataset's fixture server
from the template (exposes the fixture's ten tools, no resources):

```bash
mkdir -p ~/.config/blueprint-bench
FRAMEWORK_DIR=$PWD envsubst '${FRAMEWORK_DIR}' \
  < datasets/blueprint_mcp_host/blueprint_mcp_servers.template.json \
  > ~/.config/blueprint-bench/mcp_servers.json
```

Run it from the framework root. Re-run it when the framework moves.

## Step 2: Framework `.env`

Add this block to the framework `.env`. These values win over the
Blueprint's own `.env`, whose paths point into Docker containers. The
framework itself does not read the Blueprint-side variables.

```bash
# --- Enterprise RAG Blueprint (enterprise_rag_mcp) ---
RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.mcp_adapter
ENTERPRISE_RAG_BLUEPRINT_PATH=/home/niclas/Schreibtisch/Projects/Enterprise_RAG_Blueprint
# Read by the Blueprint:
MCP_ENABLED=true
MCP_CONFIG_FILE=/home/niclas/.config/blueprint-bench/mcp_servers.json
VECTORDB_DIR=/home/niclas/.cache/blueprint-bench/vectordb
INDEX_NAME=bench-mcp-host
# Only if the Blueprint .env has none (>= 32 characters):
#MCP_IDENTITY_SECRET=...
```

- `RAG_ADAPTER_MODULES` takes several comma-separated modules if other
  experiments need other plugins.
- `MCP_IDENTITY_SECRET` is already in the Blueprint `.env`; the in-process
  transport loads that file.
- With applied settings, `MODEL_NAME`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `EMBEDDING_MODEL`, `OLLAMA_BASE_URL` and `OLLAMA_API_KEY` in the Blueprint
  `.env` are overridden per request and need no change.

## Step 3: Model endpoints (URLs + Bearer keys)

Endpoints and keys use the same `.env` variables as every other framework
experiment. The adapter forwards them to the Blueprint, which sends the key
as `Authorization: Bearer <key>`.

| What | Manifest entry | `.env` (role-specific, else shared) | Blueprint receives |
| --- | --- | --- | --- |
| OpenAI-compatible LLM (optimaise, coassist, DeepSeek, vLLM on the DGX Spark) | `llm_models: [openai:<model id>]` | `LLM_OPENAI_COMPAT_BASE_URL` / `LLM_OPENAI_COMPAT_API_KEY`, else `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY` | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME` |
| Ollama LLM (local, DGX Spark, Ollama behind a Bearer proxy) | `llm_models: [ollama:<model>]` | `LLM_OLLAMA_BASE_URL` / `LLM_OLLAMA_API_KEY`, else `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` | `OPENAI_BASE_URL=<url>/v1`, `OPENAI_API_KEY`, `MODEL_NAME` |
| Embedding model (Ollama only) | `embedding_models: [<model>]` | `EMBEDDING_OLLAMA_BASE_URL` / `EMBEDDING_OLLAMA_API_KEY`, else `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` | `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` |

Notes:

- The Blueprint talks to Ollama LLMs through Ollama's OpenAI-compatible
  `/v1` API. Proxies must forward `/v1/...` as well as `/api/...` (the Ollama
  proxy in our `.env` does).
- The embedding URL and key also build the index, so index and queries use
  the same model and host. Without a key no auth header is sent.
- One manifest has one OpenAI-compatible endpoint. For models on a second
  endpoint, use a second manifest that sets the URL in `settings` and names
  the `.env` variable holding its key:

  ```yaml
  settings:
    llm_openai_compat_base_url: https://coassist.de/hostedmodels/v1/
    enterprise_rag_mcp_llm_api_key_env: COASSIST_API_KEY   # COASSIST_API_KEY=... in .env
  ```

  Ollama and OpenAI-compatible models can share one manifest.
- Keys never go into a manifest.

Checked on 2026-10-27 with Bearer keys from `.env`: optimaise and coassist
`/v1/models` and the Ollama proxy's `/v1/models` and `/api/tags` answer;
a Blueprint run with `openai:nvidia/Qwen3.6-35B-A3B-NVFP4` (optimaise)
completed; Blueprint embeddings through the Ollama proxy work with the key
and return 401 without it. `dgx-spark` only resolves inside its network
(VPN or `/etc/hosts`).

## Step 4: Manifest

`experiments/enterprise-rag-mcp-host.yaml` is the ready-to-use manifest.
The fields that matter:

```yaml
dataset:
  name: jsonl-shared
  path: datasets/blueprint_mcp_host/questions.jsonl
  corpus_path: datasets/blueprint_mcp_host/corpus
  sample_size: 200                 # 1000 for the full set

settings:
  rag_system_adapter: enterprise_rag_mcp
  rag_adapter_accepts: "chunker,embedder"   # framework chunks + embeds the index
  # Pin dataset fields: the local .env may carry another dataset's.
  dataset_question_field: question
  dataset_ground_truth_field: ground_truth
  generation_temperature: 0        # Blueprint default; framework default is 0.1
  enterprise_rag_mcp_transport: inprocess
  enterprise_rag_mcp_apply_settings: true   # manifest controls the Blueprint
  enterprise_rag_mcp_query_refinement: true # false skips the LLM query rewrite
  enterprise_rag_mcp_guardrails: true       # false skips the LLM input check
  enterprise_rag_mcp_index_corpus: true

matrix:
  enterprise_rag_mcp_enabled: [false, true]           # MCP off vs. on
  llm_models: [openai:nvidia/Qwen3.6-35B-A3B-NVFP4, ollama:qwen3.8:27b]
  embedding_models: [nomic-embed-text:latest]
  retrieval_top_k: [5, 10]
  chunking_strategies: [recursive]
  chunk_sizes: [512]
  chunk_overlaps: [64]
  prompt_templates: [concise]      # unused by the Blueprint
  reranker_models: [null]
```

Every combination is a separate run with its own name; results and MLflow
show the model that actually answered. Every `embedding_models` entry
rebuilds the index.

The Blueprint uses one model for query refinement, guardrails and the
answer/tool loop, so each question costs several LLM calls (about 4–5 with
MCP on).

## Step 5: Run

```bash
source .venv/bin/activate
python -m benchmark.worker plan experiments/enterprise-rag-mcp-host.yaml
```

Smoke run first: copy the manifest, set `sample_size: 3` (the manifest wins
over `DATASET_SAMPLE_SIZE`), then

```bash
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false \
  python -m benchmark.worker run experiments/<copy>.yaml --keep-going --no-mlflow
```

Full run:

```bash
python -m benchmark.worker run experiments/enterprise-rag-mcp-host.yaml --keep-going
```

Results land in `results/runN/` and MLflow. The MCP metrics are under
`adapter_metrics` (`answer_match_rate`, `per_category`, `call_success_rate`,
`tool_selection_f1`, …); RAGAS uses the critic from `EVAL_CRITIC_LLM`,
independent of the Blueprint model. Metric definitions:
[MCP Host Benchmarking](mcp-host-benchmarking.md#metrics).

## What the manifest does not control

- Prompts: the Blueprint uses its own (`enterprise_rag_mcp_prompt_key`:
  `rag_source`, `rag_summary`, `rag_weekly_report`); `prompt_templates` has
  no effect.
- Reranker, retrieval strategy, HyDE, multi-hop: the Blueprint has none of
  these options.
- MCP servers, roles, tool allowlist, budgets: the Blueprint's
  `MCP_CONFIG_FILE`.

## Variant: running Blueprint service (`http`)

To benchmark a running chain service (e.g. Docker) instead:

```yaml
settings:
  enterprise_rag_mcp_transport: http
  enterprise_rag_mcp_endpoint_url: http://localhost:8000/call_llm
  enterprise_rag_mcp_apply_settings: false
  enterprise_rag_mcp_index_corpus: false   # unless the service reads the same VECTORDB_DIR
```

The framework cannot change a remote service's models. Set `MODEL_NAME`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `OLLAMA_API_KEY`,
`RETRIEVER_TOP_K` in the service's `.env`, and keep `llm_models` in sync
by hand so the run is labelled correctly. `MCP_IDENTITY_SECRET` must equal
the service's secret.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Local dataset row 0 is missing required field 'answer'` | `.env` carries another dataset's field names; pin `dataset_question_field` / `dataset_ground_truth_field` in the manifest |
| `reads its model settings only at import` | Blueprint checkout too old; update `feat/mcp-integration` |
| `MCP_IDENTITY_SECRET must be set` | no secret in the Blueprint `.env` or framework `.env` (≥ 32 characters) |
| `needs LLM_OPENAI_COMPAT_BASE_URL or OPENAI_COMPAT_BASE_URL` | `openai:` model without endpoint in `.env` or manifest |
| `enterprise_rag_mcp_llm_api_key_env names X, which is not set` | add `X=...` to `.env` |
| `The Blueprint needs an Ollama embedding model` | use an Ollama embedding model or set `enterprise_rag_mcp_apply_settings: false` |
| failed samples with `failure_kind: model_or_tool_protocol` | model or endpoint without tool calling; pick another model |
| `ResponseError Unauthorized (401)` while embedding | embedding host needs a key: `EMBEDDING_OLLAMA_API_KEY` or `OLLAMA_API_KEY` |
| `Temporary failure in name resolution` for `dgx-spark` | host only reachable in its network; connect VPN or add it to `/etc/hosts` |
| answers ignore documents of non-`General` roles | the index is tagged `General`; keep `enterprise_rag_mcp_roles: General` |
