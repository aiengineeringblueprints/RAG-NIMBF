# Research Notes

Source folders:

- [Optimizations/](../Optimizations)
- [EVAL_MATRIX.md](../EVAL_MATRIX.md)
- [Paper/](../Paper)
- [docs/superpowers/](../docs/superpowers)

Current documented research themes:

- Retrieval, chunking, and query improvement ideas.
- Additional benchmark metrics and evaluation alternatives.
- MLflow and RAG metric tracking.
- Local judge model considerations.
- RAGAS parser/output issues and Qwen3 parsing fixes.
- Agentic small language model exploration.
- Full project audit notes.

Benchmark matrix snapshot:

- `EVAL_MATRIX.md` contains German-language manual benchmark planning and results.
- Fixed parameters noted there include `nomic-embed-text:latest`, no HyDE, often no reranker, `squad`, and a Qwen3.5 critic model.
- The matrix compares chunking strategy, chunk size/overlap, retrieval method, prompt template, top-k, reranker use, embedding, dataset, and RAGAS/custom metrics.
- Treat this as historical/manual state unless reconciled with generated `results/` and MLflow.
- Recount on 2026-06-09 for the poster update: `results/run*/benchmark_*.json` contains 82 report JSON files with 234 aggregate configuration rows; 217 rows have `N >= 100`.
- Recount on 2026-06-10 for `EVAL_MATRIX.md`: the manual matrix reaches configuration ID 222 after adding `results/run84` through `results/run87`; it marks 212 full `N=100` rows as tested, with 2 older open rows still present in the parsed matrix. Treat this as manually maintained evidence and reconcile with generated `results/` before publication.
- Current observed pattern from generated results: detailed prompts improve mean faithfulness across full runs (0.901 vs. 0.833); similarity retrieval improves mean faithfulness (0.881 vs. 0.849), while MMR improves nDCG@5 (0.789 vs. 0.703) and recall@5 (0.839 vs. 0.544).
- Current strongest observed lenses: faithfulness 0.959 in `run68` (Qwen3.5-397B-A17B, detailed, recursive 500/100, similarity, `N=300`); BERT-F1 0.969 and nDCG@5/recall@5 1.000 in `run1` (Qwen/Qwen3-32B-AWQ, concise, semantic, MMR, `N=100`); METEOR 0.665 in `run76` (deepseek-v4-pro, concise, recursive 1000/100, similarity, `N=100`).
- Poster runtime baseline on 2026-06-09 now comes from `LLM_Performance_Tests-main/benchmark_checkpoints/`, using the common `N=1`, `max_tokens=1024`, sequential comparison point. The poster table now reports latency and generation throughput only, avoiding blank TTFT cells and the constant output-token cap. Observed generation throughput ranges from 56.5 tok/s for Spark gpt-oss-20B via Ollama to 130.9 tok/s for Sigurt Qwen3.5-397B-A17B Cloud; Qwen3.6-27B-NVFP4 on vLLM RTX5090 reaches 118.1 tok/s.
- Suggested score lenses in the notes: retrieval via `ndcg@5`, context via `context_relevance`, and generation via `bert_score_f1 + meteor`.

Paper:

- `Paper/` contains LaTeX manuscript material and references.
- `Poster/` contains a first scientific conference poster draft for early July
  2026, including a `beamerposter` source, build Makefile, starter
  bibliography, and notes about current evidence and remaining placeholders.
- Keep high-level experimental claims aligned with `results/`, `EVAL_MATRIX.md`, and [[Evaluation and Metrics]].
- 2026-06-09 NGEN-AI paper update: shortened from 19 to 15 pages by removing redundant result/runtime figures, moving runtime evidence to a compact table sourced from `LLM_Performance_Tests-main` N=6/T=1024 checkpoints, compressing related work, and shortening the conclusion.
- The results section currently appears placeholder-like and should be reconciled before publication claims.
- The discussion already flags validity threats: dataset size, evaluator reliability, domain mismatch, stochasticity, and external provider changes.

Roadmap themes from optimization notes:

- Hybrid dense plus BM25 retrieval.
- Retrieve more, rerank fewer.
- Stronger rerankers and contextual retrieval.
- RAGChecker-style evaluation.
- Prefix caching, corrective RAG, model cascading, parent-child retrieval.
- Longer-horizon ideas: adaptive/speculative retrieval, GraphRAG, ColBERT.

Related notes:

- [[Evaluation and Metrics]]
- [[Known Risks and Gaps]]
- [[Testing and Coverage]]
- [[Known Stale Docs]]
