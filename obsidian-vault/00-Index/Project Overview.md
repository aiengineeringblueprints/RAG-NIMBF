# Project Overview

The project benchmarks retrieval-augmented generation pipelines across datasets, chunking strategies, retrieval settings, LLMs, prompt templates, reranking, and evaluation methods.

Core goals:

- Run repeatable benchmark grids from environment-driven [[Configuration Reference]].
- Compare local Ollama and OpenAI-compatible models through [[Providers and Models]].
- Evaluate answers with [[Evaluation and Metrics]], including RAGAS and custom IR/NLG metrics.
- Persist results to local files, reports, plots, MLflow, and optional tracing.
- Support adaptive configuration search through [[Agentic Runner]].

Primary workflows:

- `python main.py`: grid benchmark runner defined in [[Benchmark Pipeline]].
- `python -m agentic.cli`: autonomous exploration workflow defined in [[Agentic Runner]].
- `python -m unittest` or project test commands: validate behavior described in [[Testing and Coverage]].

Major local outputs:

- `results/runN/`: normal benchmark reports and per-config JSON.
- `results/agent_runN/`: autonomous agent exploration logs and per-config JSON.
- `.chroma/`: persisted vector-store cache.
- `mlruns/` and `mlflow.db`: MLflow tracking state.

Related notes:

- [[System Architecture]]
- [[Repository Map]]
- [[Operations Runbook]]
- [[Research Notes]]

