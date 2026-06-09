# Poster Notes

This poster starts from the current codebase and project outputs, not from a
final conference camera-ready claim set.

## Central Story

The project is a modular framework for systematic benchmarking of
retrieval-augmented generation pipelines. It evaluates configuration choices
across chunking, retrieval, prompt templates, reranking, local/open-compatible
models, and evaluator metrics, then persists reproducible artifacts.

## Evidence Snapshot

Source snapshot inspected on 2026-06-09:

- `results/run*/benchmark_*.json`: 82 report JSON files with 234 aggregate
  configuration rows; 217 rows have `N >= 100`.
- `EVAL_MATRIX.md`: manual matrix now tracks configuration IDs up to 182; 172
  full `N=100` rows are marked tested, with 2 open rows in the parsed matrix.
- `obsidian-vault/04-Research/Research Notes.md`: still warns that matrix state
  is manual and should be reconciled with generated `results/` before final
  publication claims.
- Strongest observed `ragas_faithfulness`: 0.959 for `run68`,
  Qwen3.5-397B-A17B, detailed prompt, recursive chunking, 500/100,
  similarity retrieval, `N=300`.
- Strongest observed `custom_bert_score_f1`: 0.969 for `run1`,
  Qwen/Qwen3-32B-AWQ, concise prompt, semantic chunking, MMR, `N=100`.
- Strongest observed `custom_ndcg@5` and `custom_recall@5`: 1.000 in the same
  Qwen/Qwen3-32B-AWQ semantic/MMR run.
- Strongest observed `custom_meteor`: 0.665 for `run76`, deepseek-v4-pro,
  concise prompt, recursive 1000/100, similarity retrieval, `N=100`.

- Runtime block now uses `LLM_Performance_Tests-main/benchmark_checkpoints/`
  instead of the RAG pipeline stage timing plot. The poster table uses the
  common `N=1`, `max_tokens=1024`, sequential comparison point across all
  available models/checkpoints: latency and generation tokens/s. TTFT is
  described as partially unavailable instead of being shown with blank table
  cells; output tokens are treated as the fixed benchmark cap.

Treat duplicated or rerun result rows carefully. The poster text calls these
"current local results" and "observed" rather than final general findings.
Paper-chart regeneration on 2026-06-09 succeeded for the heatmap, grouped bars,
speed dashboard, faithfulness-by-chunking, and speed trend. The composite paper
ranking plot failed on `NaN` conversion, so the poster uses the regenerated
`ranking_top20.png` from `benchmark.reporting.run_tracker` instead.

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

- Decide whether the poster should emphasize framework engineering,
  experimental results, or a balanced methods/results contribution.
- Re-run result aggregation after any new final benchmarks.
- Verify that every plotted result exists in the final submission package.
- Add QR code to repository, paper PDF, or demo once URL is stable.
