# Benchmark Framework Fix Plan

Created: 2026-05-20

This note captures the main correctness and reproducibility issues found during the framework review, excluding intentionally deferred reranker configuration and expanded RAGAS metric options.

## Priority Fixes

### 1. Rework or Rename Custom IR Metrics

**Severity:** High

Current custom IR metrics are not true retrieval metrics. `compute_custom_metrics()` determines relevant contexts from the already retrieved contexts, then sets:

```python
retrieved = list(range(len(ctx)))
```

That means `hit@k`, `nDCG@k`, and `recall@k` cannot penalize retrieval for missing a relevant corpus document. They only describe overlap between the retrieved contexts and the ground truth answer.

**Best fix:**

- Add optional gold retrieval IDs to normalized samples, such as `metadata["gold_doc_ids"]` or `metadata["gold_context_ids"]`.
- Preserve document IDs in chunk metadata during chunking.
- Return retrieved document or chunk IDs from retrieval.
- Compute IR metrics from `gold_ids` versus `retrieved_ids`.

**Fallback fix:**

- Rename the current metrics to make their semantics explicit, for example:
  - `retrieved_context_ground_truth_overlap@k`
  - `context_answer_hit@k`

This avoids presenting proxy metrics as true retrieval quality.

## 2. Reject `BENCHMARK_STAGE=query` With Direct Mode

**Severity:** Medium

`BENCHMARK_STAGE=query` currently works with `RETRIEVAL_MODE=direct`, but direct mode skips vector store retrieval entirely. That makes “query” silently mean “direct evaluation” instead of “reuse persisted vector store.”

**Fix:**

Update config validation:

```python
if benchmark_stage in ("index", "query") and retrieval_mode == "direct":
    raise ValueError(
        f"BENCHMARK_STAGE={benchmark_stage} requires RETRIEVAL_MODE=retrieval"
    )
```

This keeps stage behavior explicit and prevents misleading runs.

## 3. Make LanceDB MMR Honest

**Severity:** Medium

The LanceDB adapter exposes `max_marginal_relevance_search()`, but currently falls back to plain similarity search. If `VECTOR_DB_BACKEND=lancedb` and `RETRIEVAL_STRATEGY=mmr`, the reported strategy does not match actual behavior.

**Short-term fix:**

Fail fast when the unsupported combination is configured:

```python
if vector_db_backend == "lancedb" and retrieval_strategy == "mmr":
    raise ValueError("LanceDB backend does not currently support MMR retrieval.")
```

**Long-term fix:**

- Store chunk vectors with stable row IDs.
- Fetch `fetch_k` LanceDB candidates.
- Apply local MMR using candidate vectors and the query vector.
- Return the selected top `k` documents.

## 4. Fix BLEU for Short Answers

**Severity:** Medium

The current BLEU implementation always evaluates up to 4-grams. For one-token or short exact answers, higher-order n-grams are empty, causing exact answers to score `0.0`.

**Fix:**

Use adaptive n-gram order:

```python
effective_max_n = min(max_n, len(pred_tokens), len(ref_tokens))
```

Then divide by `effective_max_n`, not `max_n`.

For stronger benchmark-grade behavior, consider adding smoothing or using a maintained implementation such as SacreBLEU or NLTK BLEU.

## 5. Improve Dependency Reproducibility

**Severity:** Low to Medium

The project currently uses lower-bound dependency constraints such as:

```txt
ragas>=0.2
mlflow>=3.4
chromadb>=1.0
```

That weakens benchmark reproducibility because future dependency releases can change evaluation behavior. The current test run already shows deprecation warnings from RAGAS wrappers/imports.

**Minimal fix:**

Pin exact versions for benchmark-critical packages.

**Better fix:**

- Split dependencies into `requirements.in` and a generated lockfile.
- Use `uv pip compile requirements.in -o requirements.lock`.
- Commit both files.
- Record installed package versions in each benchmark result JSON.

## Recommended Order

1. Rework or rename custom IR metrics.
2. Reject `BENCHMARK_STAGE=query` with direct mode.
3. Fix BLEU for short answers.
4. Make LanceDB MMR fail fast or implement true MMR.
5. Add dependency locking and version capture.

## Verification Notes

The project test suite passes when run in the virtual environment:

```bash
.venv/bin/python -m pytest
```

Latest observed result:

```text
209 passed
```

Bare `python -m unittest` is not a useful verification path in this checkout because it runs outside the project virtual environment and misses dependencies such as `mlflow`, `datasets`, and `langchain_ollama`.
