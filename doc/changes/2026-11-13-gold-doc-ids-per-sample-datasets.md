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
  is an existing `source` (then `doc_id`, same precedence as the Blueprint
  adapter's indexing) if the sample carries one, else `dataset-doc-{index}` —
  the same fallback the Enterprise RAG Blueprint adapter
  (`examples/enterprise_rag_plugin/adapter.py`) uses as indexed `source`, and the
  ID the internal pipeline now copies onto chunks. Wired into all
  `load_benchmark_data` exits for adapters with `has_shared_corpus = false`
  (HF path, local jsonl/csv path, OCR path). Shared-corpus datasets are untouched.
- `benchmark/orchestration/runner.py` — `gold_doc_ids` is now recorded for every
  sample regardless of `retrieval_mode`, so `PerSampleResult.ground_truth_doc_ids`
  is populated in all stages, not just `retrieve`.
- `experiments/enterprise-rag-techqa-gold-smoke.yaml` — retrieval smoke manifest
  for the TechQA verification below.

Downstream, no code changes were needed: `gold_retrieval_metrics.py` already
resolved `metadata["gold_doc_id"]` from sample metadata, the internal adapter
copies sample metadata onto chunks (`chunking.py`), and the Blueprint adapter
prefers `metadata["doc_id"]` as indexed `source`.

## Behaviour changes and migration

- Per-sample datasets get `gold_doc_id`/`doc_id` in sample metadata. Custom jsonl
  datasets that already set `doc_id` or `source` keep that ID as gold — no
  mismatch, but result files now contain the extra metadata keys and non-empty
  `ground_truth_doc_ids`.
- Full-pipeline (`benchmark_stage: all`) runs on shared-corpus datasets now also
  populate `ground_truth_doc_ids` (previously only `retrieval` mode did). The IDs
  themselves are unchanged, so MLflow classic retriever metrics may now run where
  they were previously skipped (`skipped_no_doc_ids`).
- Existing results and reports are not modified; re-running a config produces the
  richer fields.

## Verification

- Unit tests: `tests/test_dataset.py::TestPerSampleGoldDocIds` (jsonl and mocked
  HF ragbench path stamp `dataset-doc-{i}`; existing `doc_id`/`source` are kept;
  `jsonl-shared` is not stamped), Blueprint source-ID contract test in
  `tests/test_enterprise_rag_adapter.py`, updated assertions in
  `TestLocalDatasets`. Full suite: 800 passed.
- Smoke run on TechQA (20 questions, hybrid α=0.5, k=10, RAGAS off,
  `custom_retrieval_metrics_mode: gold_doc`, run
  `results/run-gold-smoke`): `ground_truth_doc_ids` populated 20/20; gold
  hit@1 = 0.85, hit@5 = 1.0, gold hit@10 = 1.0 — in the range of the manual
  reconstruction (91–97% top-10, `doc/enterprise-rag-techqa-refusal-analysis.md`
  §2.4), no longer trivially 1.0 / k/10.

## Follow-ups

- Gold `ndcg@k` can exceed 1.0 when multiple retrieved chunks of the same gold
  document each count as relevant; consider deduplicating retrieved doc IDs
  before the DCG computation.
- `benchmark_stage: retrieve` is rejected for `enterprise_rag_mcp`; the smoke run
  had to use the `all` stage with generation enabled.
- Persist `ground_truth_doc_ids` in the samples of jsonl-shared runs (open TODO
  from `doc/changes/2026-11-12-llm-answer-judge.md`).
