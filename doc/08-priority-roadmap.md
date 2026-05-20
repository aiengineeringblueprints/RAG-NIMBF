# Priority Roadmap

Ordered by impact and feasibility. Quick wins first.

## Tier 1: Quick Wins (1-3 days each)

### 1.1 Enable All RAGAS Metrics
- **What:** Enable answer_relevancy, answer_correctness, context_precision
- **Why:** 3 free metrics already available, just disabled
- **Files:** `benchmark/evaluation.py`

### 1.2 Add `.env.example`
- **What:** Template env file with all 40+ vars and comments
- **Why:** Onboarding friction. Users read config.py source otherwise
- **Files:** New `.env.example`

### 1.3 Add `pyproject.toml`
- **What:** Project metadata, test config, build system, entry points
- **Why:** Standard Python project setup
- **Files:** New `pyproject.toml`

### 1.4 Pin Dependency Versions
- **What:** Pin exact versions in requirements.txt or add requirements.lock
- **Why:** Reproducibility
- **Files:** `requirements.txt`

### 1.5 Separate Runtime Artifacts
- **What:** Move results/mlruns/.chroma to configurable output dir
- **Why:** Clean repo, no accidental commits
- **Files:** `config.py`, `main.py`, `.gitignore`

## Tier 2: High Impact (3-7 days each)

### 2.1 Custom Metrics Tests
- **What:** Unit tests for hit@k, nDCG, ROUGE-L, BLEU, METEOR, BERTScore
- **Why:** Zero test coverage on metrics that drive paper claims
- **Files:** `tests/test_custom_metrics.py`

### 2.2 Statistical Significance Testing
- **What:** Bootstrap CIs, paired tests, effect sizes
- **Why:** Paper needs statistical rigor. Current claims are point estimates
- **Files:** New `benchmark/statistics.py`, update reporting

### 2.3 YAML Config Support
- **What:** Load config from YAML with env var overrides
- **Why:** 40+ env vars unmanageable
- **Files:** `config.py`, new `benchmark/config_loader.py`

### 2.4 Hybrid Search (BM25 + Dense)
- **What:** BM25 retrieval, fuse with dense via RRF
- **Why:** 3-10% recall gain
- **Files:** `benchmark/retrieval.py`, new `benchmark/sparse_retrieval.py`

### 2.5 Agentic Runner Deduplication
- **What:** Call main.run_single_benchmark() from agentic runner
- **Why:** Current duplication causes drift
- **Files:** `agentic/benchmark_runner.py`

### 2.6 More Datasets
- **What:** NaturalQuestions, HotpotQA, custom CSV loader
- **Why:** Single-dataset results don't generalize
- **Files:** `benchmark/dataset_adapters.py`

## Tier 3: Medium Impact (1-2 weeks each)

### 3.1 Pipeline Refactor
- Decompose run_single_benchmark into stage objects
- 450-line function -> composable stages

### 3.2 Interactive Dashboard
- Streamlit/Gradio dashboard for result exploration

### 3.3 Bayesian Optimization
- Replace grid search with Bayesian parameter tuning

### 3.4 LLM-as-Judge Evaluation
- G-Eval style judging alongside RAGAS

### 3.5 Baselines Implementation
- No-retrieval, random-retrieval, oracle-retrieval baselines

## Tier 4: Strategic (2+ weeks each)

### 4.1 Plugin System
- Entry-point based registration for metrics, datasets, providers

### 4.2 Distributed Execution
- Multi-GPU or multi-node benchmark execution

### 4.3 Multi-Modal RAG
- Image/table extraction, multi-modal embeddings

### 4.4 Continuous Benchmarking
- Scheduled runs, regression detection, model update monitoring

### 4.5 REST API
- FastAPI server for benchmark management

## Summary

| Tier | Items | Total Effort | Impact |
|------|-------|-------------|--------|
| 1 | 5 quick wins | 1-2 weeks | Foundation |
| 2 | 6 high impact | 3-6 weeks | Paper quality |
| 3 | 5 medium | 5-10 weeks | Research depth |
| 4 | 5 strategic | 10+ weeks | Production readiness |

**Start here:** Tier 1 (all), then 2.1 + 2.2 + 2.6 for paper quality.
