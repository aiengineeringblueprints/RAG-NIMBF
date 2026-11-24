# 2026-11-24 — Enterprise RAG run8 ablation and run9 RAGAS follow-up (#22)

## What
- New manifest `experiments/enterprise-rag-run8-ablation.yaml`: Enterprise RAG
  Blueprint on TechQA (500 questions, same as run7), fixed cell recursive
  2048/64, top-k 10, nomic-embed-text, hybrid retrieval; 2x2 matrix over
  `enterprise_rag_mcp_query_refinement` [true, false] and
  `retrieval_hybrid_alpha` [0.5, 0.3] (4 configs, 2000 answers).
- RAGAS off, custom metrics on (`gold_doc`), `mlflow_tracing_enabled: true`
  for per-sample refusal inspection.
- New manifest `experiments/enterprise-rag-run9-ragas.yaml`: RAGAS on for
  refinement on vs off at alpha 0.5 (2 configs) to add LLM-judged metrics to
  the main run8 effect.
- New `scripts/analyze_blueprint_ablation.py`: per-config refusal rate
  (`declines()` and `is_refusal_answer()`), refusals with gold doc in top 10,
  answer time (mean/median/p95) and exact paired McNemar tests from the
  `*.samples.json.gz` files.
- Results written up in `doc/enterprise-rag-blueprint-results.md` (run8
  section, finding 11): query refinement off is significantly better (hit@1
  +4–7 pp, refusals 32 % → 24 %, ~0.25 s faster); alpha is secondary.

## Why
Issue #22: isolate the effect of Blueprint query refinement and the hybrid
alpha on refusal rate, relevant_sentence_recall, gold hit@k and answer time.
500 questions keep the numbers comparable with run6/run7 for presentations.

## Migration
None. Run on the Spark with
`scripts/run_blueprint_rag.sh results/Enterprise-RAG/run8 experiments/enterprise-rag-run8-ablation.yaml`.

## Verification
- `python -m benchmark.worker plan` expands to 4 configs with unique names.
- Full suite on the Spark at dev d4744b2: 1021 passed, 5 failed only because
  the Spark `.env` leaks into tests (e.g. `RETRIEVAL_STRATEGY=mmr`); all 5 pass
  with `.env` moved aside. Test isolation from `.env` is a known gap.
- run8 finished 4/4 configs; analysis script run on the Spark against
  `results/Enterprise-RAG/run8`; config-hash to refinement mapping verified by
  rebuilding the configs from the manifest.
- Found (not fixed): `is_refusal_answer()` in `benchmark/custom_metrics.py`
  flags genuine answers ("insufficient", "provided context") and zeroes their
  NLG metrics (~37 of 500 answers per config).
- Known gap: the adapter does not record the refined query, so "refused with
  gold in context" cases cannot yet be inspected for query drift.
