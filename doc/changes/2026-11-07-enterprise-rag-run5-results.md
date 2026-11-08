# Enterprise RAG Blueprint run5 results

Date: 2026-11-07

## Summary
Evaluated run5 of the Enterprise RAG Blueprint plain-RAG sweep on the DGX Spark
(`results/Enterprise-RAG/run5`: RAGBench TechQA, nomic, 14 configs: recursive
512/1024/2048 × overlap 64/128 + semantic p95, each at top-k 10 and 12) and added
the numbers and findings to the results overview.

## Changes
- `doc/enterprise-rag-blueprint-results.md`: run5 in the run list and
  combination matrix, run5 result table (incl. context length and refusal rate),
  paired-bootstrap comparison, runtime row, findings 5–8 (best TechQA config now
  rec 1024/128 k10; top-k lifts recall more than correctness; refusals as main
  bottleneck; semantic chunks too long for a fair comparison), new caveats
  (duplicate corpus documents, missing token / load-test metrics for the
  adapter), updated next steps.
- `doc/00-README.md`: index entry covers run5.

## Behaviour changes and migration
None. Documentation only; result files on the Spark were read, not modified.

## Verification
Numbers taken from `benchmark_20261106_202719.json` and `progress.json`
(14/14 completed). Refusal rates, answered-vs-refused splits, distinct-context
counts and the paired bootstrap (2 000 resamples, per-question differences
against rec 1024/128 k10) were computed from
`benchmark_20261106_202719.samples.json.gz`. run4 sample answers were used for
the refusal comparison (43–45 %).

## Follow-ups
- Deduplicate the TechQA corpus before indexing.
- Investigate over-cautious refusals with high context recall (Blueprint prompt /
  query refinement).
- Gold-doc retrieval metrics for TechQA.
- `enterprise_rag_mcp` adapter reports no generator tokens and skips the LLM
  load test.
