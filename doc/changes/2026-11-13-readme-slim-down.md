# 2026-11-13 — README slim-down: detail sections moved into doc/

## What

The root `README.md` was reduced from ~600 to ~110 lines by moving detailed
sections into dedicated guides and keeping only quick starts with links:

- New `doc/env-vars.md` — the full environment-variable table (previously ~90
  rows in the README), grouped by topic, plus the `BENCHMARK_STAGE` notes.
- New `doc/llm-load-test.md` — the LLM performance/load-test section
  (`llm_performance_*` settings, sources, result locations).
- New `doc/clearml-agent.md` — the ClearML agent workflow (replaces the thin
  German notes in `doc/clearml_usage.md` as the English reference).
- New `doc/results-and-mlflow.md` — output layout, compacted reports, LLM
  answer judge re-scoring, MLflow run comparison.
- README now has: Install, Quick Start (built-in pipeline + worker), a compact
  integration-mode table linking `doc/Benchmark_Your_RAG.md` and the per-mode
  guides, parser pointer, configuration principles + env-vars link, results
  link, documentation index link, publication.
- `doc/00-README.md` index lists the four new guides.

## Why

The README had grown into a full manual (complete env-var table, MCP metric
reference, HTTP request/response contracts), making the entry point hard to
scan. Details now live next to their topic guides; the README links to them.

## Migration

Content only moved, nothing dropped or reworded except section intros. External
links that pointed at README anchors should now link to the new doc files.
Note for future changes: the env-var table to keep in sync with `config.py`
now lives in `doc/env-vars.md` (not README).

## Verification

- All README content preserved across `doc/env-vars.md`, `doc/llm-load-test.md`,
  `doc/clearml-agent.md`, `doc/results-and-mlflow.md`, and the existing
  integration guides.
- Every link in the README checked against existing files under `doc/` and
  `experiments/`.
- `doc/00-README.md` index updated with the new entries.
