# Testing Gaps

## Current State

13 test files, ~201 test functions. Most are mock-heavy unit tests. Coverage is uneven.

## 1. Zero Coverage Areas

### MLflow Integration (`benchmark/tracking.py`)
- No tests for `setup_mlflow()`, `log_benchmark_run()`, `log_genai_eval()`, `log_plots_to_mlflow()`
- MLflow server startup, connection failure, auth errors all untested
- Risk: silent data loss if MLflow is down

### Custom Metrics (`benchmark/custom_metrics.py`)
- Hit@k, nDCG@k, Recall@k — no tests found
- ROUGE-L, BLEU, METEOR — no tests found
- BERTScore — no tests found
- Context relevance — no tests found
- Vector distance metrics — no tests found
- Risk: incorrect metric computation silently corrupts results

### Reporting System (`benchmark/reporting/`)
- `analysis.py` — ranking, insights: untested
- `exports.py` — JSON, CSV, Markdown generation: untested
- `terminal.py` — console output: untested
- `visualization.py` — plot generation: untested
- Risk: broken reports, missing data in exports

### Agentic System (`agentic/`)
- `graph.py` — LangGraph state machine: untested
- `config_proposer.py` — config suggestion: untested
- `result_analyzer.py` — result analysis: untested
- `benchmark_runner.py` — pipeline execution: untested
- `cli.py` — CLI interface: untested
- Risk: agentic loop failures undetected

### GPU Monitoring (`benchmark/metrics.py`)
- `nvidia-smi` integration: untested
- GPU utilization parsing: untested
- Risk: misleading performance metrics

### End-to-End Pipeline
- `run_single_benchmark()` — 450-line orchestration function: untested
- `run_all_benchmarks()` — full pipeline: untested
- Risk: integration failures between stages

## 2. Weak Coverage Areas

### Dataset Adapters (`benchmark/dataset_adapters.py`)
- Only 5 adapters, tests use mocked loading
- No tests for edge cases: empty datasets, missing fields, large datasets
- No tests for adapter registry validation

### Vector Store Operations
- ChromaDB: collection creation, caching, cleanup partially tested
- LanceDB: untested
- No tests for concurrent access, corruption recovery

### Generation Edge Cases
- Thinking tag stripping: tested
- Refusal detection: partially tested
- Very long answers, empty answers, unicode: minimal coverage
- Percentage normalization: minimal coverage

## 3. Missing Test Types

### Integration Tests
No tests that exercise multiple modules together:
- Dataset -> Chunking -> Retrieval -> Generation
- Generation -> Evaluation -> Reporting
- Config -> Pipeline -> Result

### Performance Tests
No tests for:
- Large dataset handling (1000+ samples)
- Memory usage under load
- Concurrent operations
- Vector store scaling

### Property-Based Tests
No property-based testing (e.g., Hypothesis library):
- Config parsing: test that any valid env var combo works
- Metrics: test that scores are always in [0, 1]
- Chunking: test that chunks preserve total content

### Smoke Tests
No fast smoke tests that verify basic functionality without models:
- Config loading with defaults
- Report generation with fixture data
- Metric computation with known inputs

## 4. Test Infrastructure Gaps

### No CI/CD Configuration
No `.github/workflows/`, no `Makefile`, no `tox.ini`, no `pyproject.toml` test config.

### No Coverage Reporting
No `coverage.py` configuration. Actual coverage percentage unknown.

### No Test Fixtures
No shared test fixtures for:
- Sample benchmark results
- Mock vector stores
- Mock LLM responses
- Sample datasets

### No Test Isolation
Tests may share state via:
- `.chroma/` directory (vector store cache)
- `results/` directory
- `mlruns/` directory
- Environment variables

## 5. Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| P0 | Add tests for custom metrics (hit@k, nDCG, ROUGE, BLEU, METEOR, BERTScore) | Medium |
| P0 | Add integration test with mocked LLM/embeddings | Medium |
| P1 | Add `pyproject.toml` with test config and coverage | Low |
| P1 | Add CI workflow (GitHub Actions) | Low |
| P1 | Add test fixtures for common data shapes | Low |
| P2 | Add smoke tests that run without models | Low |
| P2 | Add property-based tests for config/metrics | Medium |
| P2 | Add reporting tests (exports, visualization) | Medium |
| P3 | Add agentic system tests | High |
| P3 | Add end-to-end pipeline test | High |
