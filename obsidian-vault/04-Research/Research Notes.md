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
- Subagent inspection counted 128 planned configurations, with roughly 62 full `N=100` tested rows, 2 `N=50` test rows, and 64 open rows. Recount before using this in reports because the matrix is manually edited.
- Observed pattern from the matrix: detailed prompts often improve Qwen3.5 faithfulness; MMR sometimes improves deterministic recall but can reduce faithfulness/precision depending on setup.
- Suggested score lenses in the notes: retrieval via `ndcg@5`, context via `context_relevance`, and generation via `bert_score_f1 + meteor`.

Paper:

- `Paper/` contains LaTeX manuscript material and references.
- Keep high-level experimental claims aligned with `results/`, `EVAL_MATRIX.md`, and [[Evaluation and Metrics]].
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
