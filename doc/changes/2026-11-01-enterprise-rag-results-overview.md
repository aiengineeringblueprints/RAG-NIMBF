# Enterprise RAG Blueprint results overview

Date: 2026-11-01

## Summary
Added a results overview for the Enterprise RAG Blueprint plain-RAG runs on the
DGX Spark (`results/Enterprise-RAG/run1..run4`), so the tested combinations,
current numbers and known metric pitfalls are in one place.

## Changes
- `doc/enterprise-rag-blueprint-results.md`: new. Fixed settings, run list,
  combination matrix (dataset × embedding × chunking), result tables for SQuAD
  and TechQA, runtime, findings, caveats, next steps.
- `doc/00-README.md`: index entry.

## Behaviour changes and migration
None. Documentation only; result files on the Spark were read, not modified.

## Verification
Numbers extracted from the `benchmark_*.json` summaries, `progress.json` and
`reproducibility/manifest.json` of each run; sample files checked to confirm
that `ground_truth_doc_ids` is empty (heuristic retrieval metrics).

## Follow-ups
- run4 is incomplete (rec 1024/128 hung, semantic not started); run3 semantic
  failed on an Ollama tokenize crash.
- Heuristic hit@k saturates at 1.0 with nomic on TechQA; gold-doc metrics needed.
