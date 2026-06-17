# Project Status — RAG Benchmarking Framework

Snapshot date: 2026-06-15. Branch: `main`. Last commit: `c468850 neue ergebnisse`.

---

## 1. What this project is

A configurable RAG pipeline runner plus an evaluation harness. It can run its
own internal chunk → embed → retrieve → generate → evaluate pipeline, or act as
a black-box evaluator against any external RAG HTTP endpoint. Results land in
`results/runN/`, MLflow (`mlflow.db`), and feed the paper/poster artefacts.

---

## 2. What is ready and works

### Core pipeline (`benchmark/`)
- **Config** (`config.py`, 23 KB): `BenchmarkConfig` dataclass; cartesian
  product of LLM × embedding × chunk size/overlap/strategy × reranker × prompt
  template. YAML manifests in `experiments/` are first-class.
- **Dataset** (`dataset.py`, 377 lines): registry + adapters. Built-ins:
  `t2-ragbench`, `ragbench`, `squad`, `ragas-wikiqa`, `ragperf-wikipedia-nq`,
  plus local `jsonl`/`csv`.
- **Chunking** (`chunking.py`): recursive, semantic, fixed via LangChain splitters.
- **Retrieval** (`retrieval.py`, 431 lines): Chroma + LanceDB backends,
  similarity/MMR, HyDE expansion, content-fingerprint cache.
- **Generation** (`generation.py`, 556 lines): LangChain chat models via
  `providers.py`. `provider:name` routing (`ollama:`, `openai:`).
- **Evaluation** (`evaluation.py`, 207 lines): RAGAS faithfulness, answer
  relevancy, answer correctness, context precision/recall, semantic similarity.
  Separate critic LLM/embedding.
- **Custom metrics** (`custom_metrics.py`, 542 lines): IR relevance (nDCG@k,
  recall@k) + NLG metrics incl. BERTScore (`roberta-large`).
- **Reporting** (`benchmark/reporting/`): JSON, CSV, Markdown, HTML, plus
  paper-specific charts (`paper_charts.py`, `ngen_ai_charts.py`),
  `pareto_with_perf.py` in `tools/`.
- **Tracking** (`tracking.py`, 570 lines): full MLflow params/metrics/artefacts
  incl. aggregate `summary_runN_<ts>` runs with tables/plots/reproducibility.
- **Reproducibility** (`reproducibility.py`): manifest + package freeze written
  per run.
- **Resource monitor** (`resource_monitor.py`, 336 lines): optional
  GPU/CPU/RAM sampling per stage.
- **Costing** (`costing.py`): per-role token accounting.

### External RAG evaluation mode (`benchmark/adapters/`)
- `RAG_SYSTEM_ADAPTER=http` skips internal chunk/retrieve/generate, posts to
  `RAG_HTTP_ENDPOINT_URL`, runs evaluation on returned answers/contexts.
- Plugin API (`register_rag_adapter`) for native Python adapters.
- Dotted-path field mapping (`RAG_HTTP_ANSWER_FIELD=result.answer`, etc.).
- Demo server + sample JSONL shipped under `examples/`.

### Resumable matrix runner (`benchmark/orchestration/`, `benchmark/worker.py`)
- `python -m benchmark.worker plan <manifest>` then `run <manifest> --keep-going`.
- Per-config results written immediately, resume state in
  `results/runN/progress.json`, can re-enter via `--run-dir`.

### ClearML remote execution (`benchmark/clearml_task.py`)
- Builds a ClearML task from `.env` or first manifest config; exposes
  non-secret `BenchmarkConfig` fields as Hyperparameters; runs the worker core
  remotely via `clearml-agent`.

### Autonomous agent runner (`agentic/`)
- LangGraph loop: `propose → run → analyze → propose`.
- Local Ollama as the "brain"; heuristic fallback if the LLM tool call fails.
- CLI: `python -m agentic.cli --agent-model qwen3:8b --max-iterations N`.

### Tests (`tests/`, 258 collected)
- **258 tests, all passing** (`pytest`, ~17 s).
- Covers adapters, chunking, config, dataset, embedding, evaluation,
  generation, gold retrieval metrics, metrics, models, orchestration, prompt
  templates, providers, reranker, retrieval.
- `conftest.py` pins `PROMPT_TEMPLATES=concise` to stabilize config counts.

### Data collected
- **86 run JSONs** under `results/runN/benchmark_*.json`.
- **90 run directories** total (`results/run1` … `results/run87` plus
  `run8/test_debug`).
- Latest runs (run80–run87) target `squad` FinQA subset, 100 samples,
  evaluated with `Qwen3.5-397B-A17B_No_Thinking` critic. Per-config flat
  fields include RAGAS metrics, nDCG/recall @1/3/5, BERTScore, token counts.

### Evaluation matrix (`EVAL_MATRIX.md`, 564 lines, 502 table rows)
- German-language sweep plan + accumulated best configs.
- Fixed knobs: embedding `nomic-embed-text:latest`, squad dataset, no reranker
  or HyDE in the baseline. Sweep vars: chunk size/overlap/strategy, retrieval
  method/top-k, LLM, prompt template.
- Sections: Feste Parameter, Matrix, Mathematische Metriken, Beste
  Konfigurationen bisher, Phase-2 Variantenmatrix, Phase-2 Priorität,
  API-Kosten (Frontier-Modelle).

### Side tooling
- `LLM_Performance_Tests-main/` — independent raw LLM throughput benchmark
  with its own `main.py`, `mlflow.db`, plots, README.
- `hallucination_eval/detectors/` — hallucination detector scaffolding.
- `compare_runs.py`, `view_mlflow_runs.py`, `backfill_mlflow.py`,
  `clear_mlflow.py` — MLflow inspection utilities.
- `tools/ci_plots.py`, `tools/pareto_with_perf.py` — paper figure generation
  (CI-by-factor plots, Pareto quality-vs-perf).

### Paper artefacts
- **NGEN-AI/** (LNCS template, primary paper, rebuilt 2026-06-14):
  `main.tex` + 9 section files, `main.pdf`, bibliography, figures
  (`pareto.pdf`, `heatmap_trimmed.pdf`, `ci_*.pdf`, model comparison, speed
  bars, template effect, perf–quality). Latest edits to `results.tex`,
  `experimental_setup.tex`, `discussion.tex`, `abstract.tex`, `introduction.tex`,
  `methodology.tex`, `related_work.tex`, `conclusion.tex`, new
  `declarations.tex`.
- **Poster/** (beamer, rebuilt 2026-06-14): `main.tex` (399 lines), `main.pdf`,
  `poster_notes.md`, `ci_plots_explanation.md`, figures, README.
- **Paper/** — older draft, last built 2026-05-28; superseded by `NGEN-AI/`.

### Documentation
- `README.md` covers install, built-in pipeline, worker matrix, ClearML agent,
  external HTTP adapter (with request/response schemas, dotted paths, native
  plugin API), env-var reference, MLflow comparison, output layout.
- `doc/` (13 files): architecture gaps, feature improvements, new
  functionalities, testing gaps, competitive analysis, research improvements,
  code quality issues, priority roadmap, ClearML usage, external RAG usage,
  fixes, info.
- `obsidian-vault/` (20 notes across Architecture, Modules, Operations,
  Research, Testing, Decisions, Agent Notes).
- `Optimizations/` (14 analysis notes: audit report, IR metric suggestions,
  RAGAS parser fixes, local judge models, Langfuse, etc.).

---

## 3. Maturity verdict per area

| Area | State |
|---|---|
| Internal pipeline (chunk→retrieve→generate→eval) | Production-ready, actively producing runs |
| External HTTP adapter | Ready, documented, demo server included |
| Resumable worker + manifest | Ready, tested |
| ClearML remote execution | Ready, documented |
| Autonomous agent (LangGraph) | Functional, optional explorer |
| Evaluation (RAGAS + custom IR/NLG) | Working; run87 data shows all metrics populated |
| Reporting + paper figures | Working; `tools/ci_plots.py` + `paper_charts.py` emit final PDFs |
| Tests | 258/258 passing |
| Paper (NGEN-AI) | Draft assembled, PDF compiles, figures wired in |
| Poster | Compiles, current |
| EVAL_MATRIX planning doc | 502-row matrix populated with best configs |

---

## 4. Known gaps / watch-items

- `Paper/` is stale (2026-05-28) relative to `NGEN-AI/` (2026-06-14). Treat
  `NGEN-AI/` as authoritative for the paper.
- RAGAS imports emit deprecation warnings
  (`ragas.metrics.SemanticSimilarity`, `LangchainEmbeddingsWrapper`). Not
  breaking yet, but will need migration before RAGAS v1.0.
- `ast.Num` deprecation in `benchmark/generation.py:100` — remove before
  Python 3.14.
- `mlflow.db` is 569 MB — large but expected for the run history.
- Untracked changes in working tree: `NGEN-AI/`, `Poster/`,
  `benchmark/reporting/paper_charts.py`, `benchmark/reporting/run_tracker.py`,
  plus new untracked `tools/` and `NGEN-AI/sections/declarations.tex`. Commit
  pending.
- `EVAL_PHASE2_VARIANTS.md` present but older (2026-05-25); cross-check
  against current `EVAL_MATRIX.md` Phase-2 section before relying on it.

---

## 5. How to verify the "works" claim yourself

```bash
# Unit tests
.venv/bin/python -m pytest -q            # expect: 258 passed

# Import the entrypoint
.venv/bin/python -c "import main; print('ok')"

# Latest data sanity
.venv/bin/python -c "import json; \
  d=json.load(open('results/run87/benchmark_20260610_092401.json')); \
  print(d['num_configs'], 'configs,', len(d['results'][0]['per_sample']), 'samples')"

# Paper compiles
cd NGEN-AI && latexmk -pdf main.tex

# Poster compiles
cd Poster && latexmk -pdf main.tex
```
