# Dataset split selection and batched Blueprint indexing

Date: 2026-10-30

## Summary
A 500-question run on `ragbench/techqa` loaded only 314 questions (the size of
the hard-wired `test` split) and then failed every config while indexing the
Blueprint corpus, because 16 358 chunks went to Chroma in one call. Manifests
can now choose the HuggingFace split, and the Blueprint index is written in
batches.

## Changes
- **Dataset split** — new `BenchmarkConfig.dataset_split` (manifest
  `dataset: split:`, env fallback `DATASET_SPLIT`, `null` keeps the adapter's
  preferred split). `load_benchmark_data` / `load_corpus_and_questions` take
  `split`; an explicitly requested split that the dataset lacks raises instead
  of silently falling back. The split is part of the RAG-vs-MCP fairness
  controls. Files: `config.py`, `benchmark/dataset.py`,
  `benchmark/orchestration/matrix.py`, `benchmark/orchestration/worker.py`.
- **Chroma batching** — `replace_blueprint_index` adds chunks in batches of
  `min(5000, client.get_max_batch_size())`, matching `benchmark/retrieval.py`.
  Root cause of `ValueError: Batch size of 16358 is greater than max batch size
  of 5461`. File: `examples/enterprise_rag_plugin/adapter.py`.
- Docs: README env table, troubleshooting rows in
  `doc/enterprise-rag-blueprint-setup.md` (incl. the HuggingFace Xet download
  hang, workaround `HF_HUB_DISABLE_XET=1`).

## Behaviour changes and migration
`DATASET_SPLIT` was previously read only by env-driven adapters
(`hf-generic`, …); it now applies to every HuggingFace adapter. A stale
`DATASET_SPLIT` in `.env` therefore changes the split of other datasets —
pin `dataset: split:` in manifests. Without it, behaviour is unchanged.

## Verification
- `tests/test_enterprise_rag_adapter.py::test_replace_blueprint_index_batches_above_chroma_max_batch_size`
  — real Chroma, `max_batch + 1` documents; red with the original error, green
  after the fix.
- `tests/test_dataset.py` — explicit split overrides the adapter default;
  missing split raises with the available splits listed.
- `tests/test_config.py` — manifest field, default `None`, env fallback.
- Full suite: 751 passed.

## Follow-ups
- MLflow params still omit subset and split; the split is visible per sample
  in the `source` field (`ragbench:train[i]`).
