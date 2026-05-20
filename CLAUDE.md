# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A RAG (Retrieval-Augmented Generation) benchmarking framework. Runs configurable pipelines that chunk documents, embed them, retrieve context, generate answers via LLMs, and evaluate with RAGAS + custom metrics. Results tracked in MLflow, reports saved to `results/runN/`.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run full benchmark (reads .env for all config)
python main.py

# Index-only / query-only stages (reuse vector stores)
BENCHMARK_STAGE=index python main.py
BENCHMARK_STAGE=query python main.py

# Run tests
python -m unittest

# Run single test module
python -m unittest tests.test_config

# Run autonomous agent (LangGraph loop that proposes configs, runs benchmarks, analyzes results)
python -m agentic.cli --agent-model qwen3:8b --max-iterations 2 --sample-size 5

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Architecture

**Entrypoint:** `main.py` → `run_all_benchmarks()` → `run_single_benchmark()` per config.

**Pipeline stages** (in `benchmark/`):
1. **Config** (`config.py`): `BenchmarkConfig` dataclass built from `.env` via `get_all_combinations()`. Produces cartesian product of LLM_MODELS x EMBEDDING_MODELS x CHUNK_SIZES x CHUNK_OVERLAPS x CHUNKING_STRATEGIES x RERANKER_MODELS x PROMPT_TEMPLATES.
2. **Dataset** (`dataset.py`, `dataset_adapters.py`): Registry pattern. Each adapter maps a HuggingFace dataset to `{question, ground_truth, context}`. Built-in: `t2-ragbench`, `ragbench`, `squad`, `ragas-wikiqa`, `ragperf-wikipedia-nq`.
3. **Chunking** (`chunking.py`): Recursive, semantic, or fixed strategies via LangChain text splitters.
4. **Retrieval** (`retrieval.py`): Chroma or LanceDB vector stores. Supports similarity, MMR, HyDE query expansion. Cache keyed by content fingerprint.
5. **Generation** (`generation.py`): LangChain chat models via `providers.py`. Provider routing: `ollama:` or `openai:` prefix on model IDs. Answer post-processing strips thinking tags.
6. **Evaluation** (`evaluation.py`): RAGAS metrics (faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall, semantic_similarity) using a separate critic LLM.
7. **Custom metrics** (`custom_metrics.py`): IR relevance + NLG metrics (BERTScore optional).
8. **Reporting** (`reporting/`): JSON/CSV/HTML exports, terminal summary, matplotlib/plotly visualizations.
9. **Tracking** (`tracking.py`): MLflow logging of params, metrics, artifacts, genai-eval payloads.

**Agentic runner** (`agentic/`): LangGraph `StateGraph` that loops `propose → run → analyze` to autonomously explore config space. Entrypoint `agentic/cli.py`.

**Provider routing** (`providers.py`): Model IDs use `provider:name` syntax (e.g. `ollama:gemma3:4b`, `openai:Qwen/Qwen3-32B-AWQ`). No prefix defaults to `ollama`. `_ContentAsStringChatModel` wrapper fixes RAGAS compatibility with servers that return JSON content.

## Configuration

All config via `.env` file (copy from `.env.example`). Key variables:

- `LLM_MODELS`, `EMBEDDING_MODELS` — comma-separated model lists with provider prefixes
- `CHUNK_SIZES`, `CHUNK_OVERLAPS`, `CHUNKING_STRATEGIES` — cartesian product sweep
- `RETRIEVAL_TOP_K`, `RETRIEVAL_STRATEGY`, `RETRIEVAL_USE_HYDE` — retrieval tuning
- `DATASET_NAME`, `DATASET_SUBSET`, `DATASET_SAMPLE_SIZE` — dataset selection
- `EVAL_CRITIC_LLM`, `EVAL_CRITIC_EMBEDDING` — separate critic for RAGAS
- `BENCHMARK_STAGE` — `all`, `index`, or `query`
- `VECTOR_DB_BACKEND` — `chroma` or `lancedb`
- Per-role URL overrides: `LLM_OLLAMA_BASE_URL`, `EVAL_CRITIC_OLLAMA_BASE_URL`, etc.

## Obsidian Vault

`obsidian-vault/` contains living documentation. Start at `obsidian-vault/00-Index/Home.md`. When code behavior changes, update the corresponding vault notes. Source is authoritative when vault and code disagree.

## Testing

Tests in `tests/` using `unittest` + `pytest` fixtures. `conftest.py` patches `PROMPT_TEMPLATES=concise` to stabilize config counts.

## Runtime Dependencies

- **Ollama** at `OLLAMA_BASE_URL` (default `localhost:11434`) for local LLMs/embeddings
- Optional OpenAI-compatible endpoint via `OPENAI_COMPAT_BASE_URL` (vLLM, TGI, LM Studio)
- MLflow tracking via `mlflow.db` / `mlruns/`
- `.chroma/` or `.lancedb/` for persisted vector stores — do not delete unless asked

## Paper

`Paper/` contains LaTeX source for the associated research paper.
