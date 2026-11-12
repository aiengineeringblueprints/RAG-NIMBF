# Enterprise RAG run6 results: dense vs hybrid retrieval

Date: 2026-11-11

## Summary
Compared run5 (dense retrieval) with run6 (hybrid dense + BM25) of the Blueprint
plain-RAG benchmark on RAGBench TechQA and recorded the results. Also corrected
the re-run command for a failed config. Documentation only.

## Changes
- `doc/enterprise-rag-blueprint-results.md`: run6 in the runs table and fixed
  settings; new section "dense (run5) vs hybrid retrieval (run6)" with paired
  per-config deltas, Gold@k and relevant-sentence coverage from the RAGBench
  labels, context composition and answering time; finding 9.
- `doc/enterprise-rag-techqa-refusal-analysis.md`: re-run command now uses
  `scripts/run_blueprint_rag.sh`. Calling the worker directly in a fresh shell
  fails with `Unsupported RAG_SYSTEM_ADAPTER='enterprise_rag_mcp'` because
  `RAG_ADAPTER_MODULES` is unset.
- `doc/00-README.md`: results guide description mentions run6.

## Behaviour changes and migration
None.

## Verification
Read-only analysis on the DGX Spark. The run5 and run6 reproducibility manifests
differ only in `retrieval_strategy`, `retrieval_hybrid_alpha`,
`retrieval_fetch_k` and adapter defaults that do not apply with MCP off.
Questions matched per index across runs. Paired bootstrap (2,000 resamples) per
config. 13 of 14 configs compared; semantic p95 k10 is being re-run.

## Update 2026-11-12
The semantic p95 k10 re-run finished; its row and the semantic findings were
added to `doc/enterprise-rag-blueprint-results.md`. It is the only config where
answer correctness drops with hybrid (0.493 → 0.475, significant) although
retrieval improves most (Gold@k +0.106, relevant-sentence coverage +0.140).
The recommended default changed from semantic p95 k12 to rec 2048/64 k12
hybrid.

## Follow-ups
- Next steps from `doc/enterprise-rag-techqa-refusal-analysis.md` remain open
  (gold metrics, `declined_rate`, ablation, runtime).
