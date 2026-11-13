# Gold document IDs for per-sample-context datasets (RAGBench)

Date: 2026-11-13

Implements GitHub issue #16.

## Summary

Datasets where every question brings its own documents (`has_shared_corpus = false`,
e.g. `ragbench`, `jsonl`, `csv`, `t2-ragbench`) now carry a gold document ID in
every sample's metadata, matching the source ID the adapters write when indexing.
Previously `ground_truth_doc_ids` was empty for these datasets and gold
`hit@k` / `ndcg@k` / `recall@k` fell back to the embedding-similarity heuristic,
which saturated on TechQA (hit@1 = 1.0, recall@k = k/10).

## Changes

- `benchmark/dataset.py` — new `assign_gold_doc_ids(samples)`: stamps
  `metadata["gold_doc_id"]` (and `metadata["doc_id"]`) onto each sample. The ID
  is resolved via the shared `resolve_dataset_doc_id()` helper
  (`benchmark/doc_ids.py`): an existing `source` (then `doc_id`, same
  precedence as the Blueprint adapter's indexing) canonicalised through
  `canonical_doc_id()`, else `dataset-doc-{index}` — the same fallback the
  Enterprise RAG Blueprint adapter
  (`examples/enterprise_rag_plugin/adapter.py`) uses as indexed `source`, and the
  ID the internal pipeline now copies onto chunks. Wired into all
  `load_benchmark_data` exits via the shared `_finalize_per_sample_dataset()`
  guard for adapters with `has_shared_corpus = false`
  (HF path, local jsonl/csv path, OCR path), so future loader paths cannot
  skip stamping. Shared-corpus datasets are untouched.
- `benchmark/doc_ids.py` (new) — canonical document ID resolution shared by
  dataset stamping, adapter indexing, and adapter retrieval metadata.
  `canonical_doc_id()` is **idempotent** (applying it twice is a no-op) and
  **collision-free**: values containing a path separator keep the full
  normalised path (`/a/manual.pdf` and `/b/manual.pdf` stay distinct), and
  suffix stripping only removes *known document extensions* from bare file
  names, so `report.1` / `report.2` / `release.v1` keep their identity while
  `manual.pdf` → `manual`. Explicit dotted IDs pass through unchanged.
  `resolve_dataset_doc_id()` applies the common `source` → `doc_id` →
  `dataset-doc-{index}` precedence for per-sample data;
  `resolve_raw_doc_id()` passes shared-corpus IDs through verbatim.
  All three producer sides now use the same function, so a retrieved chunk
  can no longer be scored as a miss because of an ID mismatch.
- `examples/enterprise_rag_plugin/adapter.py` — indexing writes the
  resolved ID into both `source` and `doc_id`: canonicalised for
  per-sample data, verbatim for shared-corpus entries (loaded via
  `corpus`, whose IDs come from the dataset loaders / user manifests).
  String `src->content` sources and Document metadata in `answer()` get
  the canonical `doc_id` **assigned** (`_retrieved_metadata()`), so a
  stale or contradictory `doc_id` returned by the Blueprint cannot shadow
  the canonical one.
- `benchmark/adapters/mcp.py` — retrieval metadata doc IDs now come from
  `canonical_doc_id()` instead of a local `Path(...).stem`, and are
  assigned (not defaulted), so contradictory tool-supplied `doc_id`s are
  replaced by the canonical ID.
- `benchmark/orchestration/runner.py` — `gold_doc_ids` is now recorded for every
  sample regardless of `retrieval_mode`, so `PerSampleResult.ground_truth_doc_ids`
  is populated in all stages, not just `retrieve`; both gold-metric call sites
  pass an explicit `k_values=[1, 3, 5, 10]` (the function default only
  computes k=1,3,5, so `hit@10` was previously never reported).
- `experiments/enterprise-rag-techqa-gold-smoke.yaml` — retrieval smoke manifest
  for the TechQA verification below.
- `experiments/enterprise-rag-gold-run7.yaml` — full verification manifest
  (500 TechQA samples, gold_doc retrieval metrics, RAGAS off). It replicates
  the matrix of `results/Enterprise-RAG/run6` but swaps the heuristic
  retrieval metrics for `custom_retrieval_metrics_mode: gold_doc`, so the
  gold hit@k values are directly comparable to the manual reconstruction in
  `doc/enterprise-rag-techqa-refusal-analysis.md` §2.4 (91–97% top-10).
  Status: not yet executed; run it with `RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.mcp_adapter`
  and `ENTERPRISE_RAG_BLUEPRINT_PATH` set (machine-local values in `.env`),
  output to `results/Enterprise-RAG/run7`.
- `results/run-gold-smoke/reproducibility/manifest.json` — machine-local
  paths and endpoint URLs redacted (per AGENTS.md, machine-specific values
  belong in `.env` only).

Downstream, no code changes were needed: `gold_retrieval_metrics.py` already
resolved `metadata["gold_doc_id"]` from sample metadata, the internal adapter
copies sample metadata onto chunks (`chunking.py`), and the Blueprint adapter
prefers `metadata["doc_id"]` as indexed `source`.

## Behaviour changes and migration

- Per-sample datasets get `gold_doc_id`/`doc_id` in sample metadata. Custom jsonl
  datasets that already set `doc_id` or `source` keep that ID as gold —
  canonicalised (path-like values reduce to their stem on both the gold and the
  retrieval side), so IDs match across adapter paths, but string forms in result
  files may change (e.g. `/docs/manual.pdf` → `manual`). Result files now
  contain the extra metadata keys and non-empty `ground_truth_doc_ids`.
- Full-pipeline (`benchmark_stage: all`) runs on shared-corpus datasets now also
  populate `ground_truth_doc_ids` (previously only `retrieval` mode did). The IDs
  themselves are unchanged, so MLflow classic retriever metrics may now run where
  they were previously skipped (`skipped_no_doc_ids`).
- Existing results and reports are not modified; re-running a config produces the
  richer fields.

## Verification

- Unit tests: `tests/test_dataset.py::TestPerSampleGoldDocIds` (jsonl and mocked
  HF ragbench path stamp `dataset-doc-{i}`; existing `doc_id`/`source` are kept;
  `jsonl-shared` is not stamped), updated assertions in `TestLocalDatasets`.
- Canonicalisation tests: `tests/test_doc_ids.py` — idempotency
  (`canonical_doc_id(canonical_doc_id(x)) == canonical_doc_id(x)`, including the
  multi-dot `release.v1.pdf` roundtrip), collision guards (`report.1` vs
  `report.2`, `/a/manual.pdf` vs `/b/manual.pdf` stay distinct), known-suffix
  stripping, and resolver precedence/fallback.
- End-to-end contract tests:
  `tests/test_enterprise_rag_adapter.py::TestGoldDocIdEndToEnd` covers the
  Blueprint path (string `src->content` sources get a metric-recognizable
  `doc_id`; `dataset-doc-0` scores as a hit), ID consistency between
  `assign_gold_doc_ids()` and indexing (path-like, empty, `None` and dotted
  sources), shared-corpus IDs passing through the actual corpus path
  verbatim, stale `doc_id`s being overwritten on the Blueprint and MCP
  retrieval side, and the internal pipeline path (chunk `doc_id` scores
  against gold, `hit@10` reported).
- Focused suites: `97 passed` (`tests/test_doc_ids.py`,
  `tests/test_dataset.py`, `tests/test_enterprise_rag_adapter.py`,
  `tests/test_gold_retrieval_metrics.py`, `tests/test_mcp_adapter.py`);
  full suite `826 passed`.
- Smoke run on TechQA (20 questions, hybrid α=0.5, k=10, RAGAS off,
  `custom_retrieval_metrics_mode: gold_doc`, run
  `results/run-gold-smoke`, re-executed after the canonical-ID fix):
  `ground_truth_doc_ids` populated 20/20; gold hit@1 = 0.85, hit@3 = 0.95,
  hit@5 = 1.0, **hit@10 = 1.0** — in the range of the manual reconstruction
  (91–97% top-10, `doc/enterprise-rag-techqa-refusal-analysis.md` §2.4), no
  longer trivially 1.0 / k/10. A sanitized evidence file with the metric
  means is checked in as `results/run-gold-smoke/gold_metrics_summary.json`;
  heavy raw outputs (`configs/`) stay untracked. The run used the local
  smoke vector store (`VECTORDB_DIR=.chroma-smoke`, machine-local, not
  checked in).

## Follow-ups

- Gold `ndcg@k` can exceed 1.0 when multiple retrieved chunks of the same gold
  document each count as relevant; consider deduplicating retrieved doc IDs
  before the DCG computation.
- `benchmark_stage: retrieve` is rejected for `enterprise_rag_mcp`; the smoke run
  had to use the `all` stage with generation enabled.
- Persist `ground_truth_doc_ids` in the samples of jsonl-shared runs (open TODO
  from `doc/changes/2026-11-12-llm-answer-judge.md`).
