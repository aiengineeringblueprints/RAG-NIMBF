# Hybrid retrieval for the Enterprise RAG Blueprint

Date: 2026-11-08

## Summary
`retrieval_strategy: hybrid` used to have no effect on the Blueprint adapter:
the Blueprint retrieves itself and only knew dense similarity search, while
runs were still labelled `_hybrid-a<alpha>`. The Blueprint now has hybrid
retrieval (branch `feat/hybrid-retrieval`), and the `enterprise_rag_mcp`
adapter forwards the strategy, alpha and fetch-k per request.

## Changes
- Blueprint (`Enterprise_RAG_Blueprint`, branch `feat/hybrid-retrieval`):
  `chain/hybrid_retriever.py` with BM25 + weighted RRF, the same algorithm as
  `benchmark/retrieval.py::retrieve_hybrid` (RRF k 60, BM25 k1 1.5 / b 0.75,
  `\w+` tokens, dense-first tie-breaking). `chain/retriever.py` reads
  `RETRIEVER_SEARCH_TYPE`, `RETRIEVER_HYBRID_ALPHA`, `RETRIEVER_HYBRID_FETCH_K`
  per request; role filter applies to the BM25 side too. Details in the
  Blueprint's `docs/hybrid-retrieval.md`.
- `examples/enterprise_rag_plugin/mcp_adapter.py`: `blueprint_settings_env`
  sets `RETRIEVER_SEARCH_TYPE` (and for hybrid `RETRIEVER_HYBRID_ALPHA`,
  `RETRIEVER_HYBRID_FETCH_K` from `retrieval_fetch_k`, empty = Blueprint
  default 4 * top-k). `retrieval_strategy: mmr` is rejected before the run.
  For hybrid, `retrieval_fetch_k` below `retrieval_top_k` is rejected: the
  Spark `.env` sets `RETRIEVAL_FETCH_K=3` (meant for MMR), which would have
  capped every hybrid result at 6 documents without an error.
  The in-process transport rejects `hybrid` on a Blueprint checkout without
  `retriever.retriever_search_type`, so no run is mislabelled again.
- `experiments/enterprise-rag-blueprint-rag.yaml`: `retrieval_strategy:
  hybrid` with alpha 0.5 and `retrieval_fetch_k: 50` pinned (fixed rather than
  4 * top-k so k10 and k12 fuse the same candidate pool; with RRF k 60 ranks
  past 50 add almost nothing), comments corrected (Blueprint top-k default is 5).
- `doc/enterprise-rag-blueprint-setup.md`: prerequisites and "what the
  manifest controls" updated.

## Behaviour changes and migration
- Every applied-settings request now also sets `RETRIEVER_SEARCH_TYPE`
  (default `similarity`, which is the previous Blueprint behaviour).
- `retrieval_strategy: hybrid` with the in-process transport needs the
  Blueprint on `feat/hybrid-retrieval` (or later, once merged).
- Runs `results/Enterprise-RAG/run1..run5` used dense similarity search; any
  config named `_hybrid-...` from before this change was dense only.

## Verification
- Blueprint: `chain/tests/test_hybrid_retrieval.py` (real on-disk Chroma):
  exact-term match missed by dense search, role filter on the BM25 side,
  index cache refresh after new chunks, alpha 0 / 1, invalid settings, RAG
  chain with the hybrid retriever. Role filter and cache tests checked by
  mutation. Chain suite 243 passed, coverage 96.6 %; hybrid, retriever,
  load-chain and handle-llms tests also pass under the framework venv.
- Framework: `tests/test_enterprise_rag_mcp_adapter.py` (env mapping, mmr
  rejection, fetch-k below top-k, old-Blueprint rejection); full suite 774
  passed. Manifest expands to 14 configs, all `hybrid` alpha 0.5, fetch-k 50,
  also on the Spark with its `.env`.

## Follow-ups
- The internal pipeline (hybrid and MMR) has the same `RETRIEVAL_FETCH_K` <
  top-k pitfall and no check yet; the Spark `.env` comment ("calculated as
  TOP_K * FETCH_K") is wrong, the value is absolute.
- No benchmark run with hybrid yet; compare against run5 (same matrix, dense).
- The older `enterprise_rag` adapter (`examples/enterprise_rag_plugin/adapter.py`)
  calls `create_retriever` without setting the search type, so it stays dense.
