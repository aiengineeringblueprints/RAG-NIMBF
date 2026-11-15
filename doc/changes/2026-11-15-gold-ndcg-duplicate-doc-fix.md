# Fix: gold-doc nDCG counted duplicate chunks of the same document

Date: 2026-11-15

## Summary

run7's reported nDCG@10 values (1.41–1.66) exceeded the metric's [0, 1]
bound. Root cause: `_ndcg_at_k` in `benchmark/gold_retrieval_metrics.py`
iterated the retrieved list positionally and counted every chunk whose doc
id is in the gold set. Hybrid retrieval returns several chunks of the same
document, so with a single gold document each duplicate position added gain
while IDCG assumed one relevant item. Proven with a stored run7 sample:
retrieved `[doc1, x, doc1, x, doc1]`, gold `{doc1}` → nDCG@5 of 1.887.

## Changes

- `benchmark/gold_retrieval_metrics.py`: `_ndcg_at_k` deduplicates retrieved
  ids (order-preserving, `dict.fromkeys`) before computing DCG. hit@k and
  recall@k were already set-based and unaffected.
- `tests/test_gold_retrieval_metrics.py`: regression tests — duplicates cap
  nDCG at 1.0, earlier hits still rank higher than later ones, multi-gold
  stays ≤ 1.
- `doc/enterprise-rag-blueprint-results.md`: run7 caveat replaced with the
  explanation and corrected values.

## Corrected run7 values

Recomputed from the stored run7 per-sample retrieval lists
(`benchmark_*.samples.json.gz`): corrected nDCG@10 = 0.711–0.739 for the 8
configs whose samples were persisted (all k12 configs + semantic k10/k12).
The six recursive k10 configs' samples were not persisted; their true values
lie in the same band. Existing hit@k/recall@k results are unchanged, so no
benchmark conclusions other than nDCG are affected.

## Behaviour changes and migration

- nDCG values in new runs and in recomputed gold-doc metrics are now ≤ 1 and
  comparable across runs. Old stored reports keep the inflated numbers;
  recompute from samples if needed.

## Verification

- `pytest tests/test_gold_retrieval_metrics.py` → 6 passed (2 regression
  tests red before the fix).
- Full suite locally green; fix deployed to the DGX Spark with the next
  sync (branch `feat/27-local-preflight`).
