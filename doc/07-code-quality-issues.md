# Code Quality Issues

## 1. Hardcoded Values

| File | Line | Value | Issue |
|------|------|-------|-------|
| `benchmark/generation.py` | 16 | `_MAX_RETRIES = 5` | Not configurable |
| `benchmark/generation.py` | 17 | `_BASE_DELAY = 10` | Not configurable |
| `benchmark/generation.py` | 510 | `GPU_METRICS_INTERVAL_SECONDS = "30"` | Not configurable |
| `benchmark/custom_metrics.py` | ~140 | `8000` char truncation for embeddings | Not configurable |
| `benchmark/custom_metrics.py` | ~211 | `512` token BERT limit | Not configurable |
| `benchmark/tracking.py` | 29 | `2` second MLflow timeout | Too short for slow networks |
| `benchmark/evaluation.py` | 146-150 | 3 of 6 RAGAS metrics enabled | Should be all 6 by default |
| `benchmark/custom_metrics.py` | 65-90 | Refusal patterns hardcoded | Should be configurable |
| `main.py` | 66 | `lru_cache(maxsize=2)` | Arbitrary cache size |

## 2. Error Handling Gaps

### Silent Failures
- `benchmark/generation.py:370` — streaming fallback masks underlying errors
- `benchmark/custom_metrics.py` — metric errors logged as warning, not propagated
- `benchmark/evaluation.py` — RAGAS failure sets error string but continues with None metrics

### Missing Retries
- `benchmark/evaluation.py:164` — network errors to critic LLM caught but no retry
- `benchmark/retrieval.py` — vector store operations have no retry on transient failures
- `benchmark/dataset.py` — HuggingFace download failures not retried

### No Circuit Breaker
- If provider is down, every sample timeouts sequentially. No skip after N failures.
- If embedding model crashes, every subsequent sample fails. No fallback.

## 3. Type Safety Issues

### Missing Type Hints
- `benchmark/custom_metrics.py` — most functions lack return type annotations
- `benchmark/reporting/exports.py` — `_result_to_dict()` returns untyped dict
- `agentic/*.py` — minimal type hints throughout

### Runtime Type Violations
- `benchmark/generation.py:254` — `strip_mode` cast at config level but not enforced at usage
- `config.py:344` — `cast()` satisfies type checker but no runtime validation

## 4. Code Organization Issues

### God Function
- `main.py:run_single_benchmark()` — 450 lines, 10+ responsibilities
- Should be decomposed into stage objects

### God File
- `benchmark/custom_metrics.py` — all IR and NLG metrics in one file
- Should be split: `metrics/ir.py`, `metrics/nlg.py`, `metrics/relevance.py`

### Mixed Concerns
- `benchmark/generation.py` — generation, streaming, post-processing, GPU monitoring
- `benchmark/retrieval.py` — vector store construction, caching, retrieval, HyDE

## 5. Dependency Management

### No Version Pinning
`requirements.txt` uses `>=` ranges everywhere. No lower bounds tested.

**Fix:** Pin exact versions. Use `pip freeze > requirements.lock`.

### Missing pyproject.toml
No project metadata, build system, test config, or entry points.

### Heavy Dependencies
- `sentence-transformers` pulls PyTorch (~2GB) even if BERTScore disabled
- `mlflow` pulls extensive deps even for basic tracking
- Consider optional extras

## 6. Security Considerations

### API Key Handling
- Keys from env vars (good) but potentially logged in MLflow params
- No `.env.example` template

### Input Validation
- `config.py` validates types but not ranges (chunk_size could be 1)
- `benchmark/dataset.py` trusts HuggingFace data without validation
