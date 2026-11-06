# Hybrid retrieval (dense + BM25 fusion)

Date: 2026-11-06

## Summary
Added `retrieval_strategy: "hybrid"` to the internal pipeline: dense similarity
search is fused with a BM25 (Okapi) keyword search over the same chunks using
weighted Reciprocal Rank Fusion (RRF). This covers exact-term queries
(IDs, error codes, names) that pure dense retrieval misses.

## Changes
- `benchmark/retrieval.py`: new `BM25Index` (pure-Python Okapi BM25, no new
  dependency) and `retrieve_hybrid(vector_store, query, top_k, *, chunks,
  alpha=0.5, fetch_k=None)`. Fusion: `alpha * 1/(60+rank_dense) +
  (1-alpha) * 1/(60+rank_bm25)`, deduped across both lists; empty corpus falls
  back to dense-only.
- `benchmark/adapters/internal.py`: `_search` routes `hybrid` to
  `retrieve_hybrid`, passing the chunks from `prepare()` (stored on the
  adapter, cleared in `cleanup()`). Component injection uses the same path.
- `config.py`: new field `retrieval_hybrid_alpha` (default `0.5`), env fallback
  `RETRIEVAL_HYBRID_ALPHA` (validated 0.0–1.0), `RETRIEVAL_STRATEGY` now also
  accepts `hybrid`, config name suffix `_hybrid-a<alpha>`.
- `benchmark/orchestration/matrix.py`: `retrieval_hybrid_alpha` accepted as
  float matrix dimension, so manifests can sweep the fusion weight.
- `README.md`: env-var rows for `RETRIEVAL_STRATEGY` / `RETRIEVAL_HYBRID_ALPHA`.

## Behaviour changes and migration
None by default (`similarity` unchanged). To use hybrid:

```yaml
retrieval_strategy: hybrid
retrieval_hybrid_alpha: 0.5   # 1.0 = pure dense, 0.0 = pure BM25
```

`retrieval_top_k` works as before; hybrid fetches `retrieval_fetch_k`
(default `top_k * 4`) candidates from each side before fusion. Multihop is not
combined with hybrid yet (hybrid takes precedence).

## Verification
- `tests/test_retrieval.py::TestHybridRetrieval`: RRF ordering at alpha 0/0.5/1,
  keyword-only docs merged, dedup across retrievers, top_k respected, empty
  corpus fallback.
- `tests/test_config.py`: default + `RETRIEVAL_HYBRID_ALPHA` env fallback.
- `tests/test_internal_adapter.py`: adapter routes `hybrid` through
  `retrieve_hybrid` with the prepared chunks; non-hybrid strategies don't.
- Full suite: 766 passed.

## Follow-ups
- Combine hybrid with multihop retrieval: currently hybrid takes precedence and
  multihop is skipped entirely when both are set.
- Optional: expose BM25 k1/b as config knobs if sweeps need them.
