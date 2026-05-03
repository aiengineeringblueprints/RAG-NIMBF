# Benchmarking Framework - Comprehensive Audit Report

**Date:** 2026-04-17
**Scope:** Full codebase audit covering architecture, bugs, security, testing, code quality, and missing functionality

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Bugs & Implementation Gaps](#1-critical-bugs--implementation-gaps)
3. [Security Issues](#2-security-issues)
4. [Test Coverage Gaps](#3-test-coverage-gaps)
5. [Code Quality Issues](#4-code-quality-issues)
6. [Architectural Concerns](#5-architectural-concerns)
7. [Missing Functionality](#6-missing-functionality)
8. [Git & Repository Hygiene](#7-git--repository-hygiene)
9. [File-by-File Deep Analysis](#8-file-by-file-deep-analysis)
10. [Prioritized Recommendations](#9-prioritized-recommendations)

---

## Executive Summary

The RAG Benchmarking Framework is a well-architected project using clean design patterns (adapter, factory, strategy, registry). It supports multiple LLM providers, embedding models, chunking strategies, retrieval methods, and comprehensive reporting. However, this audit reveals **critical gaps in metric integration, significant test coverage holes, security concerns, and numerous code quality issues** that should be addressed.

**Test Matrix Status:** Only 13 of 128 planned benchmark configurations have been tested (see `TEST_MATRIX.md`).

---

## 1. Critical Bugs & Implementation Gaps

### 1.1 RAGAS Metrics Hardcoded to Only Faithfulness

**Files:** `benchmark/evaluation.py:123`, `benchmark/tracking.py`, `benchmark/reporting/exports.py`

Only `faithfulness` is actually evaluated. All other RAGAS metrics (`answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall`) are **commented out**. The entire data model, exports, tracking, and reporting layer expects all 5 metrics, but only 1 is ever computed. This means:

- Rankings and composite scores in `analysis.py` are based on incomplete data
- Exports show `None` for most metric columns
- Terminal reports show misleading gaps
- The 80/20 weighting in `analysis.py` (80% RAGAS, 20% performance) is meaningless with only 1 metric

### 1.2 Custom Metrics Module NOT Wired Into Pipeline

**File:** `benchmark/custom_metrics.py:11`

The module explicitly states: *"This module is **not** wired into the main benchmark pipeline yet."* A comprehensive IR/NLG metrics module exists (Hit@k, nDCG, Recall@k, ROUGE, BLEU, METEOR, BERTScore) but is **completely unused**. Current uncommitted changes in `main.py` appear to be attempting this integration but it's incomplete.

### 1.3 Duplicate Dataset Labels in Reports

**Files:** `benchmark/reporting/exports.py:210`, `benchmark/reporting/terminal.py:115`

Dataset label is printed twice in Markdown and terminal output:

```python
lines.append(f"**Dataset:** {label} ({dataset_sample_size} samples)")
lines.append(f"**Dataset:** {dataset_subset} ({dataset_sample_size} samples)")  # Duplicate!
```

### 1.4 Division by Zero Risks

**File:** `benchmark/reporting/analysis.py:184`

TTFT ratio calculation has no guard against `lo_t == 0`. Also in `analysis.py:95`, composite score calculation doesn't handle `None` values properly for weight distribution.

### 1.5 Cache Key Missing Embedding Provider

**File:** `benchmark/retrieval.py`

The cache key for vector stores doesn't include the embedding provider, which could cause cache collisions when switching between Ollama and HuggingFace embeddings for the same model name.

### 1.6 Dead Code

**File:** `benchmark/generation.py:364`

```python
pass  # usage might be in response_metadata
```

This is dead code that should be removed or properly implemented.

### 1.7 Unpopulated Fields

**File:** `benchmark/reporting/models.py:68`

`ragas_valid_sample_counts` field is defined in the dataclass but never populated or used anywhere in the codebase.

---

## 2. Security Issues

### 2.1 API Keys in .env File

**File:** `.env`

```
LLM_OLLAMA_API_KEY=<redacted>
EVAL_CRITIC_OPENAI_COMPAT_API_KEY=<redacted>
OPENAI_COMPAT_API_KEY=<redacted>
```

While `.env` is in `.gitignore`, this is still risky on shared machines. Consider using a proper secrets manager or system-level environment variables.

### 2.2 Internal IP Addresses Exposed

**File:** `.env:15`

```
LLM_OLLAMA_BASE_URL=<redacted-internal-url>
```

Internal network topology is exposed in configuration.

### 2.3 Binary Database in Git

**File:** `mlflow.db` (1.1MB SQLite database tracked in git)

Binary files should never be in version control because:

- They increase repository size with every commit
- Changes are not diffable
- They contain local state not suitable for version control
- Multiple commits in the history are just "Update MLflow database" with no functional value

### 2.4 Hardcoded API Key Placeholders

**File:** `benchmark/providers.py:103`

```python
api_key="not-needed"
```

Hardcoded placeholder API key could cause issues with providers that validate keys.

---

## 3. Test Coverage Gaps

### 3.1 Untested Modules

No test files exist for the following modules:

| Module | Risk Level | Description |
|--------|-----------|-------------|
| `benchmark/tracking.py` | **High** | MLflow integration - experiment tracking, run logging, metrics export |
| `benchmark/custom_metrics.py` | **High** | Complex math for IR/NLG metrics |
| `benchmark/reporting/analysis.py` | **High** | Ranking logic, composite scores, normalization |
| `benchmark/reporting/exports.py` | **Medium** | Data serialization to JSON/CSV/Markdown |
| `benchmark/reporting/visualization.py` | **Low** | Plot generation |
| `benchmark/reporting/terminal.py` | **Low** | Terminal display formatting |
| `benchmark/dataset.py` | **Medium** | Data loading pipeline |
| Prompt template implementations | **Low** | `concise.py`, `detailed.py`, `finqa.py` |

### 3.2 Existing Test Files (11 files)

Well-tested components:

| Test File | Lines | Quality |
|-----------|-------|---------|
| `test_config.py` | 377 | Excellent - comprehensive validation testing |
| `test_generation.py` | 377 | Excellent - thorough answer processing tests |
| `test_dataset.py` | 205 | Good - adapter pattern coverage |
| `test_evaluation.py` | 209 | Good - RAGAS integration testing |
| `test_retrieval.py` | 160 | Good - caching, HyDE, strategies |
| `test_providers.py` | 111 | Good - factory routing |
| `test_chunking.py` | 121 | Good - strategy coverage |
| `test_models.py` | 110 | Good - stats computation |
| `test_reranker.py` | 79 | Adequate |
| `test_embedding.py` | 61 | Adequate |
| `test_metrics.py` | 43 | Minimal |
| `test_prompt_templates.py` | 40 | Minimal - only registry, not implementations |

### 3.3 No CI/CD Pipeline

- No GitHub Actions, GitLab CI, or other automation
- No automated testing on pull requests
- No code quality checks
- No coverage reporting
- No security scanning

### 3.4 Missing Test Infrastructure

- `pytest` not listed in `requirements.txt`
- No `pytest.ini` or `pyproject.toml`
- No coverage configuration
- No test fixtures directory

### 3.5 No Integration Tests

- No end-to-end tests for the full benchmark pipeline
- No tests for `main.py`'s `run_single_benchmark` function
- All tests use mocks - no tests with real data

---

## 4. Code Quality Issues

### 4.1 Hardcoded Values Throughout

| Location | Hardcoded Value | Issue |
|----------|----------------|-------|
| `chunking.py:49` | `min_chunk_length=50` | No configuration option |
| `evaluation.py` | `max_workers=1`, `timeout=600`, `max_retries=2`, `max_wait=60` | Not configurable |
| `tracking.py` | Experiment name `"RAG-Benchmark"` | Hardcoded |
| `generation.py` | `SentenceTransformer("roberta-large")` | Model name hardcoded |
| `custom_metrics.py` | METEOR weights, fragmentation penalty (0.5), BERTScore max length (512) | Not configurable |
| `providers.py:93` | `think=False` | May not be supported by all models |
| `metrics.py` | `nvidia-smi` timeout, assumes single GPU | Not flexible |

### 4.2 Overly Large Functions

| Function | Lines | Issue |
|----------|-------|-------|
| `main.py:run_single_benchmark` | ~179-280 lines | Does everything: chunking, retrieval, generation, evaluation, metrics, results |
| `config.py:get_all_combinations` | ~167 lines | Deeply nested loops, hard to follow |
| `evaluation.py` | 47 parameters | Untestable, unmaintainable |

### 4.3 Inconsistent Error Handling

- **Evaluation**: Catches exceptions gracefully with error responses
- **Generation/HyDE**: Fails silently, falls back to original query without logging
- **Retrieval**: ChromaDB connection failures not handled
- **Tracking**: MLflow connection failures silently ignored
- **Chunking**: No handling for document conversion failures

### 4.4 Missing Input Validation

| File | Issue |
|------|-------|
| `chunking.py:54` | Assumes all list elements are strings |
| `evaluation.py:171` | Assumes `per_sample_ragas` length matches questions |
| `analysis.py:160` | Assumes all configs have all metrics |
| `config.py:142-144` | `_validate_positive_int` doesn't properly reject negative numbers |
| `config.py:257-260` | Chunk overlap validation only checks `overlap < chunk_size`, not `overlap > 0` |
| `config.py:214` | Boolean parsing for `LLM_ANSWER_VALUE_FALLBACK` could be more robust |

---

## 5. Architectural Concerns

### 5.1 No Async Support

All operations are synchronous. For a benchmarking framework that makes many LLM API calls, this is a significant performance limitation. No async APIs exist in any component.

### 5.2 Single GPU Assumption

**File:** `benchmark/metrics.py`

GPU monitoring assumes exactly one NVIDIA GPU via `nvidia-smi`. No graceful fallback for:
- CPU-only environments
- Multi-GPU setups
- Non-NVIDIA GPUs
- Systems without `nvidia-smi` installed

### 5.3 No Parallel Benchmark Execution

Benchmarks run strictly sequentially. With 128 planned configurations, this means extremely long run times.

### 5.4 In-Memory-Only Caching

Embedding and vector store caches are in-memory only. No persistence across runs means repeated computation every restart.

### 5.5 ChromaDB Tight Coupling

**File:** `benchmark/retrieval.py`

Tightly coupled to ChromaDB with no abstraction layer. Cannot swap vector databases without significant refactoring.

### 5.6 No Package Configuration

Missing `pyproject.toml` or `setup.py` - the project cannot be installed as a package.

### 5.7 Memory Management

No cleanup for large models in memory. The reranker's CrossEncoder loads a new model on every instantiation with no caching.

### 5.8 Global Lock Contention

**File:** `benchmark/retrieval.py`

A global `_chroma_lock` is used for thread safety. This could become a bottleneck with many concurrent operations.

---

## 6. Missing Functionality

### 6.1 Critical Missing Features

| Feature | Status | Impact |
|---------|--------|--------|
| Full RAGAS metrics evaluation | Commented out | **Critical** - evaluations are incomplete |
| Custom metrics in pipeline | Not wired | **High** - module exists but unused |
| Hybrid retrieval (keyword + semantic) | Not implemented | **Medium** |
| Metadata filtering during retrieval | Not implemented | **Medium** |
| Batch retrieval API | Not implemented | **Medium** |
| CPU/RAM/memory metrics | Not implemented | **Medium** |
| Local dataset support | Not implemented | **Medium** |

### 6.2 Nice-to-Have Missing Features

| Feature | Status | Impact |
|---------|--------|--------|
| Structured output (JSON mode) | Not implemented | Low |
| Async support | Not implemented | Medium |
| Checkpoint/resume for long runs | Not implemented | Medium |
| Distributed benchmarking | Not implemented | Low |
| Document preprocessing pipeline | Not implemented | Low |
| Few-shot example support in templates | Not implemented | Low |
| Custom distance metrics | Not implemented | Low |
| Similarity score threshold filtering | Not implemented | Low |

---

## 7. Git & Repository Hygiene

### 7.1 Uncommitted Changes Are Substantial

The current working tree has significant uncommitted changes representing a custom metrics integration effort:

| File | Nature of Change |
|------|-----------------|
| `benchmark/chunking.py` | Added `min_chunk_length` parameter |
| `benchmark/custom_metrics.py` | Enhanced `determine_relevance` with embedding support |
| `benchmark/evaluation.py` | Commented out most RAGAS metrics |
| `benchmark/reporting/exports.py` | Removed performance metrics, added custom metrics |
| `benchmark/reporting/models.py` | Added custom metrics fields |
| `benchmark/tracking.py` | Added custom metrics logging to MLflow |
| `main.py` | Integrated custom metrics computation |
| `mlflow.db` | Binary database modified |

These should be committed in logical increments, not as one large commit.

### 7.2 `mlflow.db` Commits Polluting History

Multiple commits with messages like "Update MLflow database to reflect recent changes..." add no functional value and make the history harder to read. This binary file should be gitignored.

### 7.3 Results Directory

`results/run19/` is untracked. The `results/` pattern should be consistently handled in `.gitignore`.

### 7.4 TEST_MATRIX.md Is Untracked

This is valuable project documentation that should be committed.

### 7.5 Commit Message Pattern

Recent commits show a pattern of vague "Update MLflow database" messages. Commit messages should describe the actual functional changes.

---

## 8. File-by-File Deep Analysis

### 8.1 Core Pipeline

#### `main.py`

- **Role:** Main orchestrator for benchmark runs
- **Issues:**
  - `run_single_benchmark` is 179-280 lines with too many responsibilities
  - `SentenceTransformer("roberta-large")` hardcoded
  - GPU metrics extraction assumes specific dict keys without validation
  - HyDE query expansion failure is silent
  - No progress reporting for long operations
  - No cleanup of temporary resources
  - No checkpoint/resume capability

#### `config.py`

- **Role:** Configuration management and benchmark combination generation
- **Issues:**
  - `get_all_combinations` is ~167 lines with nested loops
  - `_validate_positive_int` doesn't properly reject negative numbers
  - Chunk overlap doesn't validate `overlap > 0`
  - No configuration schema validation
  - Environment variable naming inconsistent

#### `benchmark/chunking.py`

- **Role:** Text chunking strategies
- **Issues:**
  - `min_chunk_length=50` hardcoded
  - No validation that `chunk_size > chunk_overlap`
  - Direct import inside function for `SemanticChunker`
  - No logging for chunking statistics
  - Character strategy hardcoded separator override

#### `benchmark/evaluation.py`

- **Role:** RAGAS evaluation integration
- **Issues:**
  - Only `faithfulness` metric active (line 123)
  - 47-parameter function signature
  - Effective critic model falls back to generator (potential bias)
  - `max_workers`, `timeout`, `max_retries` hardcoded
  - No metric selection configuration

#### `benchmark/custom_metrics.py`

- **Role:** IR and NLG metrics
- **Issues:**
  - Not wired into main pipeline (explicitly stated in docstring)
  - BLEU brevity penalty assumes non-zero reference length
  - BERTScore returns 0.0 for empty strings (should be `None`)
  - Stemming algorithm is O(n^2) for large texts
  - Hardcoded weights, thresholds, and max lengths
  - No caching for expensive operations like BERTScore

### 8.2 Retrieval & Embeddings

#### `benchmark/retrieval.py`

- **Role:** Vector store management and document retrieval
- **Issues:**
  - Cache key doesn't include embedding provider
  - Global lock could become bottleneck
  - ChromaDB connection failures not handled
  - MMR default `fetch_k = top_k * 4` may be suboptimal
  - Collection cleanup race condition possible
  - No metadata filtering support

#### `benchmark/embedding.py`

- **Role:** Embedding model factory
- **Issues:**
  - No validation that model_name is available
  - No timeout handling for model loading
  - HuggingFace provider ignores API keys (inconsistent)
  - No retry logic for failed embeddings
  - No batch embedding API

#### `benchmark/generation.py`

- **Role:** LLM answer generation and post-processing
- **Issues:**
  - Dead code at line 364 (`pass  # usage might be in response_metadata`)
  - HyDE expansion fails silently
  - `str(response.content)` could fail if content is `None`
  - Arithmetic evaluator doesn't handle scientific notation
  - `_looks_like_thinking()` is heuristic and could misclassify
  - No temperature control in generation API

#### `benchmark/reranker.py`

- **Role:** Document reranking with cross-encoders
- **Issues:**
  - CrossEncoder loads model on every instantiation (no caching)
  - Could fail if `documents` is `None`
  - Only supports HuggingFace provider
  - No batch reranking optimization

#### `benchmark/providers.py`

- **Role:** LLM/embedding provider abstraction
- **Issues:**
  - `think=False` may not be supported by all models
  - Hardcoded `"not-needed"` API key placeholder
  - Content wrapper only needed for RAGAS compatibility
  - Limited to Ollama and OpenAI-compatible providers

### 8.3 Data & Metrics

#### `benchmark/metrics.py`

- **Role:** GPU metrics collection
- **Issues:**
  - Assumes exactly one NVIDIA GPU
  - Could fail if `nvidia-smi` not installed
  - Subprocess call is expensive
  - No CPU, RAM, or network metrics

#### `benchmark/dataset.py` & `benchmark/dataset_adapters.py`

- **Role:** Dataset loading and normalization
- **Issues:**
  - `data.shuffle(seed=42)` modifies original dataset
  - SQuAD ground truth assumes specific list structure
  - No validation that required columns exist
  - No support for local datasets
  - Loading entire dataset into memory

### 8.4 Reporting & Tracking

#### `benchmark/tracking.py`

- **Role:** MLflow experiment tracking
- **Issues:**
  - Experiment name `"RAG-Benchmark"` hardcoded
  - `_mlflow_safe` only replaces `"@"`, but MLflow has many invalid characters
  - No error handling for MLflow connection failures
  - No logging for chunking strategy details beyond name
  - Uses `tempfile.mkdtemp()` without cleanup

#### `benchmark/reporting/models.py`

- **Role:** Data structures for results
- **Issues:**
  - `ragas_valid_sample_counts` defined but never populated
  - No validation for NaN values in most fields
  - `gpu_usage` is `dict[str, float] | None` - could be more structured

#### `benchmark/reporting/exports.py`

- **Role:** Export to JSON, CSV, Markdown
- **Issues:**
  - Duplicate dataset label (line 210)
  - Hardcoded RAGAS keys in CSV export don't match computed metrics
  - Custom metrics not correctly mapped to their names

#### `benchmark/reporting/analysis.py`

- **Role:** Ranking computation and insights
- **Issues:**
  - Division by zero risk at line 184
  - Best/worst metrics assumes all configs have all metrics
  - Composite score doesn't handle `None` values
  - Metric weights don't match actual usage (only faithfulness evaluated)

#### `benchmark/reporting/visualization.py`

- **Role:** Matplotlib plot generation
- **Issues:**
  - Statistics access assumes `count > 1` for std deviation
  - Radar chart doesn't handle `None` values (uses 0.0)
  - Only 8 colors defined - insufficient for many configs
  - Missing units in axis labels

#### `benchmark/reporting/terminal.py`

- **Role:** Rich terminal output
- **Issues:**
  - Duplicate dataset label (line 115)
  - Sparkline bar width calculation could overflow
  - Only shows metrics that are actually computed (faithfulness only)

### 8.5 Prompt Templates

#### `benchmark/prompt_templates/`

- **Role:** Prompt template registry and implementations
- **Issues:**
  - No validation that templates contain required placeholders
  - Complex formatting rules in `finqa.py` could be fragile
  - No template versioning
  - No support for dynamic templates or few-shot examples

---

## 9. Prioritized Recommendations

### Phase 1: Immediate (Week 1)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | **Re-enable all 5 RAGAS metrics** or make them configurable | Critical | Low |
| 2 | **Wire custom metrics** into the main pipeline | High | Medium |
| 3 | **Remove `mlflow.db` from git** (`git rm --cached`, add to `.gitignore`) | High | Low |
| 4 | **Fix duplicate dataset labels** in `exports.py` and `terminal.py` | Medium | Low |
| 5 | **Add division-by-zero guards** in `analysis.py` | Medium | Low |
| 6 | **Commit current work** - the uncommitted changes are substantial | Medium | Low |
| 7 | **Remove dead code** (`generation.py:364`) | Low | Trivial |

### Phase 2: Short-term (Week 2-3)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 8 | **Add test files** for `tracking.py`, `custom_metrics.py`, and `reporting/` | High | Medium |
| 9 | **Add `pyproject.toml`** with proper package configuration | Medium | Low |
| 10 | **Add pytest to requirements.txt** and create pytest configuration | Medium | Low |
| 11 | **Set up CI/CD** (GitHub Actions) with automated testing | Medium | Medium |
| 12 | **Extract hardcoded values** into configuration | Medium | Medium |
| 13 | **Fix `config.py` validation** (negative numbers, overlap > 0) | Medium | Low |
| 14 | **Fix cache key** to include embedding provider | Medium | Low |

### Phase 3: Medium-term (Month 1-2)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 15 | **Break down `run_single_benchmark`** into smaller focused functions | High | High |
| 16 | **Break down `get_all_combinations`** into smaller functions | Medium | Medium |
| 17 | **Add async support** for parallel LLM calls | High | High |
| 18 | **Implement vector DB abstraction** layer | Medium | High |
| 19 | **Add persistent caching** for embeddings/vector stores | Medium | Medium |
| 20 | **Add CPU/RAM monitoring** alongside GPU metrics | Medium | Low |
| 21 | **Implement batch retrieval** API | Medium | Medium |
| 22 | **Add checkpoint/resume** for long benchmark runs | Medium | High |

### Phase 4: Long-term

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 23 | **Hybrid retrieval** (keyword + semantic) | High | High |
| 24 | **Metadata filtering** during retrieval | Medium | Medium |
| 25 | **Distributed benchmarking** across multiple machines | High | Very High |
| 26 | **Model caching** for reranker and BERTScore | Medium | Low |
| 27 | **Local dataset support** | Low | Medium |
| 28 | **Structured output (JSON mode)** support | Low | Medium |

---

## Appendix: Project Structure

```
Benchmarking-Framework/
├── main.py                    # Main entry point and orchestrator
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── .env                       # Active configuration (gitignored)
├── compare_runs.py            # Compare benchmark results
├── clear_mlflow.py            # Clear MLflow data
├── view_mlflow_runs.py        # CLI to view MLflow experiments
├── helper_functions/
│   └── view_ground_truths.py
├── benchmark/                 # Core modules
│   ├── __init__.py
│   ├── dataset.py             # Dataset loading
│   ├── dataset_adapters.py    # Dataset-specific adapters
│   ├── chunking.py            # Text chunking strategies
│   ├── retrieval.py           # Vector retrieval with caching
│   ├── embedding.py           # Embedding model management
│   ├── generation.py          # LLM answer generation
│   ├── evaluation.py          # RAGAS evaluation
│   ├── metrics.py             # Performance metrics (GPU, TTFT, TPS)
│   ├── custom_metrics.py      # IR/NLG metrics (not wired)
│   ├── providers.py           # LLM/embedding provider abstraction
│   ├── reranker.py            # Document reranking
│   ├── tracking.py            # MLflow integration
│   ├── prompt_templates/      # Prompting strategies
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── concise.py
│   │   ├── detailed.py
│   │   └── finqa.py
│   └── reporting/             # Result analysis and reporting
│       ├── __init__.py
│       ├── models.py          # Data models
│       ├── analysis.py        # Ranking and statistics
│       ├── terminal.py        # Terminal output
│       ├── exports.py         # JSON/CSV/Markdown export
│       └── visualization.py   # Plot generation
├── tests/                     # Test suite (11 files)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_embedding.py
│   ├── test_generation.py
│   ├── test_evaluation.py
│   ├── test_metrics.py
│   ├── test_reranker.py
│   ├── test_providers.py
│   ├── test_prompt_templates.py
│   └── test_models.py
├── results/                   # Benchmark output directory
└── mlflow.db                  # MLflow SQLite database (should not be tracked)
```
