# Poster Notes

This poster starts from the current codebase and project outputs, not from a
final conference camera-ready claim set.

## Central Story

The project is a modular framework for systematic benchmarking of
retrieval-augmented generation pipelines. It evaluates configuration choices
across chunking, retrieval, prompt templates, reranking, local/open-compatible
models, and evaluator metrics, then persists reproducible artifacts.

## Evidence Snapshot

Source snapshot inspected on 2026-05-24:

- `results/**/results_summary.csv`: 63 summary files, 151 aggregate rows.
- `EVAL_MATRIX.md`: manual matrix with 128 planned configurations.
- `obsidian-vault/04-Research/Research Notes.md`: warns that matrix state is
  historical/manual and should be reconciled before publication claims.
- Strongest observed `ragas_faithfulness`: 0.955 for Qwen3.5-397B-A17B with
  detailed prompt and recursive chunking.
- Strongest observed `custom_bert_score_f1`: 0.969 for Qwen/Qwen3-32B-AWQ with
  concise prompt, semantic chunking, and MMR.
- Strongest observed `custom_ndcg@5` and `custom_recall@5`: 1.000 in the same
  Qwen/Qwen3-32B-AWQ semantic/MMR run.

Treat duplicated or rerun result rows carefully. The poster text calls these
"current local results" and "observed" rather than final general findings.

## What Seems Important To Show

1. The framework is the contribution, not only one benchmark leaderboard.
2. Configuration interactions matter: prompt template, retrieval strategy,
   chunking, reranking, and generator choice shift different metrics.
3. RAG quality needs several lenses: faithfulness, retrieval ranking,
   context relevance, lexical/semantic answer similarity, latency, and runtime.
4. Artifacts are reproducible: JSON, CSV, Markdown, plots, MLflow, and traces.
5. Limitations must be visible: SQuAD/WikiQA domain, sample size, evaluator
   reliability, local hardware, stochastic generation, and manual matrix drift.

## TODO Before July

- Replace all `TODO` placeholders in `main.tex`.
- Decide whether the poster should emphasize framework engineering,
  experimental results, or a balanced methods/results contribution.
- Re-run result aggregation after final benchmarks.
- Verify that every plotted result exists in the final submission package.
- Add QR code to repository, paper PDF, or demo once URL is stable.
