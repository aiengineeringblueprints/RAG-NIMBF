# TechQA refusal and runtime analysis (Enterprise RAG run6)

Date: 2026-11-10

## Summary
Analysed why the Blueprint plain-RAG run on RAGBench TechQA (`results/Enterprise-RAG/run6`)
takes ~50 h and why `declined_rate` is 0.584. Documentation only; no code or
config changed.

## Changes
- New guide `doc/enterprise-rag-techqa-refusal-analysis.md`: `declined_rate`
  false positives, refusal causes (dataset, passage-level retrieval, Blueprint),
  framework measurement gaps, runtime breakdown, re-run command for the failed
  config, next steps.
- `doc/00-README.md`: guide added to the index.
- `doc/enterprise-rag-blueprint-results.md`: link from "Open / next".

## Behaviour changes and migration
None.

## Verification
Read-only analysis on the DGX Spark of the run6 sample file
`recursive_cs512_co64 … k10` against `rungalileo/ragbench` `techqa` `train`
(`shuffle(seed=42)`, first 500, order verified). Key numbers: real refusals
165–188 of 500; gold document set in top 10 for 91 %; relevant-sentence coverage
19 % (refused) vs 48 % (answered); 71 questions without relevant sentences;
`ragas_eval` 10,692 s of 12,445 s per config.

## Follow-ups
See section 5 of the new guide: re-run semantic k10, gold metrics and
`relevant_sentence_recall` for RAGBench, fix `declined_rate`, ablation
(query refinement, guardrails, internal pipeline), eval runtime knobs.
