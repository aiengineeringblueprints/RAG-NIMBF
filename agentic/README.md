# Autonomous RAG Benchmarking Agent

An AI agent that autonomously explores RAG pipeline configurations, runs benchmarks, analyzes results, and iteratively discovers better setups — all without human intervention.

## What It Does

The agent replaces manual trial-and-error with an autonomous loop:

1. **Proposes** a RAG configuration (chunking strategy, chunk size, retrieval method, etc.)
2. **Runs** the full benchmark pipeline (chunking → retrieval → generation → evaluation)
3. **Analyzes** results and extracts insights about what worked and what didn't
4. **Iterates** — proposes a new config based on accumulated knowledge
5. **Stops** when it reaches max iterations or results converge

It uses a local Ollama model as the "brain" driving the exploration, while the actual benchmark execution reuses the existing `benchmark.*` modules.

## Architecture

```
START --> propose --> run --> analyze --> [propose | END]
                 ^                       |
                 |_______________________|
```

| Node | What happens | LLM-driven? |
|------|-------------|-------------|
| **propose** | Agent LLM decides next config to test | Yes (Ollama via ChatOllama + bind_tools) |
| **run** | Executes full benchmark pipeline programmatically | No |
| **analyze** | Agent LLM compares results, extracts insights | Yes (Ollama, low temperature) |

If the agent LLM fails to produce a valid tool call, a **heuristic fallback** picks the next untested config variation automatically.

## Parameters the Agent Explores

| Parameter | Values |
|-----------|--------|
| chunking_strategy | recursive, character, token, semantic |
| chunk_size | 200, 300, 500, 800, 1000, 1500 |
| chunk_overlap | 50, 100, 150, 200 |
| retrieval_top_k | 3, 5, 8, 10, 15 |
| retrieval_strategy | similarity, mmr |
| retrieval_use_hyde | true, false |
| prompt_template | concise, detailed, finqa |
| reranker_model | none, cross-encoder/ms-marco-MiniLM-L-6-v2 |

The LLM and embedding models are fixed per run (set via CLI flags or `.env`).

## Quick Start

```bash
# Full autonomous run (8 iterations, 20 samples)
python -m agentic.cli --agent-model qwen3:8b

# Quick test (2 iterations, 5 samples)
python -m agentic.cli --agent-model qwen3:8b --max-iterations 2 --sample-size 5

# Specify benchmark models and dataset
python -m agentic.cli \
    --agent-model qwen3:8b \
    --llm-model gemma3:4b \
    --embedding-model nomic-embed-text:latest \
    --dataset t2-ragbench \
    --subset FinQA \
    --sample-size 20 \
    --max-iterations 8
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--agent-model` | qwen3:8b | Ollama model used as the agent "brain" |
| `--llm-model` | from .env | LLM used for benchmark answer generation |
| `--embedding-model` | from .env | Embedding model for vector store |
| `--dataset` | from .env | Benchmark dataset name |
| `--subset` | from .env | Dataset subset |
| `--sample-size` | 20 | Number of questions per benchmark |
| `--max-iterations` | 8 | Maximum exploration iterations |
| `--ollama-url` | from .env | Override OLLAMA_BASE_URL |

All defaults fall back to your `.env` file, then to sensible built-in defaults.

## Output

Results are saved to `results/agent_runN/`:

```
results/agent_run1/
    agent_exploration_log.json    # All configs, metrics, insights
    configs/
        recursive_cs500_co100_...json    # Per-config results
```

The `agent_exploration_log.json` contains:
- Every configuration tested
- All metrics for each run
- Accumulated insights from the analyzer
- Best configuration found

MLflow tracking also logs each run automatically (same as the main pipeline).

## Requirements

- **Ollama** running locally with the agent model pulled (e.g., `ollama pull qwen3:8b`)
- **Benchmark LLM** available on Ollama (e.g., `ollama pull gemma3:4b`)
- **Embedding model** available (e.g., `ollama pull nomic-embed-text`)
- All existing project dependencies (no new packages needed — LangGraph and ChatOllama are already installed)

## Agent Model Choice

The agent model needs to support tool calling. Tested with:
- `qwen3:8b` — recommended, good tool calling support
- `llama3.1:8b` — works well
- `mistral:7b` — works

Smaller models may fall back to the heuristic config proposer more often, which is fine — the pipeline still works, just with less creative exploration.

## How It Differs from the Main Pipeline

| | `main.py` | `agentic/` |
|---|---|---|
| Configs | Grid search from .env variables | AI-driven adaptive search |
| Execution | Runs all configs sequentially | Runs one config per iteration |
| Analysis | Post-hoc reporting | Real-time analysis between runs |
| Adaptation | None | Learns from results to pick better configs |
| Output | Same report formats | Same + exploration log |
