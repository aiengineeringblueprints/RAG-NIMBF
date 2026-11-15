# PR #33 review fixes: async-safe embeddings, nDCG positions, retry scope

Date: 2026-11-15 (round 2 update below)

## Summary

Addresses the review of PR #33 ("Fix Ollama embedding crashes, gold-doc nDCG
bug"). Items 1–3 (critical) and 4–10 (medium/minor) are fixed; the run7
nDCG numbers should be recomputed (see migration).

## Changes

- `benchmark/embedding.py`:
  - **Async paths (review item 1, critical)**: `aembed_documents` /
    `aembed_query` now apply the same sanitize → batch → retry pipeline as
    the sync methods. RAGAS calls these via `LangchainEmbeddingsWrapper`
    (`benchmark/evaluation.py`), so the evaluation stage is protected too.
  - **nDCG-free metric file unchanged** (fix in
    `benchmark/gold_retrieval_metrics.py`, see below).
  - **Built-in `ConnectionError` retried (item 3, critical)**: the ollama
    client maps `httpx.ConnectError` to the built-in `ConnectionError`
    (`ollama/_client.py:144-145`); `_is_transient()` now retries it, so a
    refused connection during a server restart no longer aborts indexing.
  - **Retry per batch (item 4)**: retries wrap each `_EMBED_BATCH_SIZE`
    batch inside the loop instead of the whole list — one bad batch no
    longer re-embeds everything that already succeeded.
  - **Selective retry, no sleep after last attempt (item 5)**: only 5xx
    `ResponseError`, `httpx.HTTPError`, and `ConnectionError` are retried;
    400/404 (e.g. a typo in the model name) fail immediately.
  - **Non-whitespace placeholder (item 6)**: empty/whitespace-only input is
    replaced with `"empty"` instead of `" "` — whitespace-only was exactly
    the crashing input.
  - **Truncation cap configurable + logged (item 7)**: cap is now
    `_max_text_chars()`, overridable via the `EMBEDDING_MAX_TEXT_CHARS` env
    var (default 6000); a warning is logged once when input is truncated.
  - **Module-level class, cached (item 8)**: `_ollama_embeddings_class()`
    is wrapped in `functools.cache`, so the Pydantic class is built once
    and instances are picklable (no more `<locals>` qualname).
- `benchmark/gold_retrieval_metrics.py` (**item 2, critical**):
  - `_ndcg_at_k` no longer deduplicates before ranking (which renumbered
    gold documents to better ranks). Gold ids are credited once each, at
    their original position; duplicates neither move ranks nor add gain.
    Example: `gold={doc1}`, `retrieved=[x,x,x,x,doc1]`, k=5 now yields
    `1/log2(6) ≈ 0.39` instead of 0.63.
- `tests/test_embedding.py` (**item 10**): new tests for built-in
  `ConnectionError` retry, no-retry on 404, no sleep after the final
  attempt, per-batch retry, class reuse, env-override handling, and the
  async (`aembed_*`) paths. Removed the unused `httpx` import (it is now
  used, plus `ollama` imported at module level).
- `tests/test_gold_retrieval_metrics.py`: exact-value regression test for
  the duplicate-before-gold case (0.39, not 0.63).
- `.env.example`: documents `EMBEDDING_MAX_TEXT_CHARS`.

Note on item 9 (rank semantics across metrics): `hit@k`/`recall@k` are
set-based and unaffected by duplicates; `span_ndcg@k` already dedupes by
evidence index (`seen` set in `_append_span_metrics`) while keeping
positions. With `_ndcg_at_k` fixed, the metrics now share position-preserving
semantics, so no further change was needed.

## Behaviour changes and migration

- Placeholder change: empty inputs embed as the `"empty"` token vector
  instead of `" "` — both near-zero; retrieval results for healthy corpora
  are unchanged.
- Transient-only retry may raise earlier for persistent 4xx errors — that
  is intended (fail fast on configuration mistakes).
- **run7 results (Blueprint results matrix) may be too high for nDCG** and
  should be recomputed with the fixed `_ndcg_at_k` before being cited.
- `EMBEDDING_MAX_TEXT_CHARS` is currently read from the environment only
  (not a `BenchmarkConfig` field); a full config field can be added later
  if manifests need per-run caps.

## Verification

- `pytest` — 855 passed (full suite), including 37 tests in
  `tests/test_embedding.py` + `tests/test_gold_retrieval_metrics.py`.

## Follow-ups

- Recompute and re-publish the run7 gold-doc nDCG numbers in the Blueprint
  results matrix (values were inflated by the dedup-then-renumber bug; raw
  run7 data lives on the DGX Spark host, not in this checkout).

## Round 2 — follow-up review of 67fc90cb

- **Item 8 (picklability)**: the class is now registered as a module
  attribute with a cleaned-up qualname/module, and `__getstate__` drops the
  cached ollama httpx clients (`_client`/`_async_client`) that hold thread
  locks (they are lazily re-created after unpickling). Test
  `test_instance_is_picklable` round-trips a real instance.
- **Item 9 (missing doc ids shift ranks)**: hit/nDCG/recall are computed on
  `ranked_doc_ids` (with `None` placeholders kept in place) instead of the
  None-filtered list, so chunks without a provider doc id no longer move
  gold documents up or widen the top-k window. Two regression tests added
  (gold behind id-less chunks at k=1 → 0.0; gold at true rank 2 →
  1/log2(3)).
- **Item 7**: env override + warning stay; the required `BenchmarkConfig`
  field is now tracked in issue #34.
- **Nits**: removed the redundant inner list comprehension in
  `embed_documents`; the char cap is read once per `embed_documents` /
  `aembed_documents` call and passed into `_sanitize(text, limit)`.
- Note on the reviewer's run7 question: raw run7 metadata is stored on the
  DGX Spark (`results/Enterprise-RAG/run1..run7` there), so the gap check
  must run on that host.

Verification: full suite — 858 passed (40 in the two touched test files).

