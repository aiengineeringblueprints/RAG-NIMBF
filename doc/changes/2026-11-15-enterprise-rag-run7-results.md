# Enterprise RAG run7 results: gold-doc retrieval metrics (issue 16)

Date: 2026-11-15

## Summary

run7 (`experiments/enterprise-rag-gold-run7.yaml`, 14 configs, TechQA n=500,
hybrid retrieval) re-ran the run6 setup with
`custom_retrieval_metrics_mode: gold_doc` and RAGAS off, verifying the
framework's built-in gold-doc retrieval metrics against the manual
reconstruction in `doc/enterprise-rag-techqa-refusal-analysis.md` §2.4.
Results are documented in `doc/enterprise-rag-blueprint-results.md` (new
run7 section, combination matrix column, finding 10, failure details).

## Changes

- `doc/enterprise-rag-blueprint-results.md`: status 2026-11-15 / run1..run7;
  fixed-settings row for metrics mode; run7 row in the runs table; new
  combination-matrix column for run6+run7; failure details for the Ollama
  embed-runner crashes; new section "gold-doc retrieval metrics (run7)" with
  the full per-config table (hit@1/5/10, recall@10, nDCG@10, context
  relevance, BERT F1, ROUGE-L); finding 10.
- `doc/00-README.md`: results-guide description mentions run1..run7.
- `doc/changes/2026-11-14-resilient-ollama-embeddings.md` (previous session):
  sanitization + retries; follow-up commit `04a171f6` added server-side
  64-text batching after the diagnosis showed the crash was batch-size
  driven (3,530 texts in one `/api/embed` POST reset the runner; single-text
  requests never failed).

## Key results

- Gold hit@10 = 0.898–0.948 across all 14 configs, matching the manual §2.4
  band (91–97 %) and run6's ad-hoc Gold@10 within ~1–2 pp → issue 16 verified.
- hit@k == recall@k (single gold doc per TechQA question); k10 → k12 adds
  ≤ 0.006.
- Chunking order mirrors run6: 512 worst, 2048/semantic best; best hit@10
  rec 2048/128 k10, semantic k10/k12, rec 2048/64 k10 (0.940–0.948).
- Open caveat: reported nDCG@10 (1.41–1.66) exceeds the [0,1] band although
  the per-question implementations are capped at 1 — aggregation needs
  investigation before nDCG is used cross-run.

## Behaviour changes and migration

None (documentation only; code changes were documented in the 2026-11-14
change log).

## Verification

- Worker status: 14/14 configs `completed` in
  `results/Enterprise-RAG/run7/progress.json`; final summary
  `benchmark_20261115_141853.json` (covers the last re-run config; per-config
  aggregates in `configs/*.json`).
- Docs render as Markdown tables; no code touched in this session.
