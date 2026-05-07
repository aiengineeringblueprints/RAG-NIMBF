# Repository Map

Top-level files:

- `main.py`: deterministic grid benchmark entrypoint.
- `config.py`: environment parsing and `BenchmarkConfig`.
- `requirements.txt`: Python dependency list.
- `EVAL_MATRIX.md`: manual benchmark matrix and historical results.
- `prompts.md`, `info.md`, `fixes.md`: project notes and prompt/fix history.
- `compare_runs.py`, `view_mlflow_runs.py`, `backfill_mlflow.py`, `clear_mlflow.py`: helper scripts for results and MLflow.

Core package:

- `benchmark/`: shared RAG benchmark implementation.
- `benchmark/prompt_templates/`: built-in prompt templates.
- `benchmark/reporting/`: result models, terminal output, export writers, plot generation, ranking analysis.

Adaptive workflow:

- `agentic/`: autonomous benchmark exploration loop.

Research/documentation:

- `Optimizations/`: prior audits, metric ideas, RAG improvements, local judge research, Langfuse/MLflow notes.
- `Paper/`: LaTeX paper, references, and sections.
- `docs/superpowers/`: design/spec documents.

Tests:

- `tests/`: unit tests for config, dataset, chunking, retrieval, generation, providers, metrics, evaluation, reranker, prompt templates, and result models.
- `test_ragas_wikiqa.py`: standalone dataset/evaluation script.

Generated/runtime state:

- `results/`: benchmark outputs.
- `.chroma/`: vector-store cache.
- `mlruns/`, `mlflow.db`: MLflow artifacts.
- `graphify-out/`: graph/documentation output from previous analysis.

Helper script caveats:

- `compare_runs.py` compares `results/run*/benchmark_*.json` and assumes current report JSON shape.
- `view_mlflow_runs.py` reads MLflow SQLite tables directly, so it depends on MLflow schema internals.
- `backfill_mlflow.py` parses `EVAL_MATRIX.md`, matches rows to historical results, and creates synthetic finished MLflow runs.
- `clear_mlflow.py` clears MLflow data tables while preserving schema; use only when explicitly intended.

Related notes:

- [[Operations Runbook]]
- [[Research Notes]]
- [[Known Risks and Gaps]]
