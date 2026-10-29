# Plain RAG manifest for the Enterprise RAG Blueprint

Date: 2026-10-29

## Summary
New manifest `experiments/enterprise-rag-blueprint-rag.yaml` benchmarks the
Enterprise RAG Blueprint as a plain RAG system, without MCP. It reuses the
`enterprise_rag_mcp` adapter with MCP off, so the Blueprint's full `/call_llm`
path runs while the manifest controls its models and retrieval settings.

## Changes
- `experiments/enterprise-rag-blueprint-rag.yaml`: `enterprise_rag_mcp_enabled:
  false`, `enterprise_rag_mcp_apply_settings: true`, in-process transport,
  dataset `squad` (20 samples), `llm_models: [openai:qwen3.8-27b]`,
  `embedding_models: [nomic-embed-text:latest]`, `retrieval_top_k: [5]`,
  recursive 512/64 chunking. Retrieval fields (`retrieval_mode`,
  `retrieval_strategy`, `retrieval_multihop`, `retrieval_use_hyde`) are pinned
  against `.env` drift; without the `retrieval_strategy` pin a `.env` with
  `RETRIEVAL_STRATEGY=mmr` puts `mmr` into the config name although the
  Blueprint retriever ignores it.

Manifest fields that reach the Blueprint: `llm_models`, `embedding_models`,
`retrieval_top_k`, `generation_temperature`, the chunking axes (index build),
`enterprise_rag_mcp_query_refinement` and `enterprise_rag_mcp_guardrails`.
`prompt_templates` and `reranker_models` have no effect; the Blueprint uses its
own prompt and Chroma retriever. Setup and endpoints:
[enterprise-rag-blueprint-setup.md](../enterprise-rag-blueprint-setup.md)
(skip the MCP server config and `MCP_IDENTITY_SECRET`; neither is needed with
MCP off).

- `examples/enterprise_rag_plugin/mcp_adapter.py` (`_evidence`): document
  contexts handed to the evaluator now start with `Document: <source>`, the
  label the Blueprint puts before each document in its prompt
  (`chain/load_chain.py` `format_docs`). The Blueprint prompt (`rag_source`)
  makes the model name the documents it used; without the label RAGAS could
  not verify those sentences and counted them as unsupported. MCP evidence is
  unchanged.

## Behaviour changes and migration
`enterprise_rag_mcp` runs from now on report contexts with the
`Document: <source>` prefix. Faithfulness and context metrics are not directly
comparable with earlier `enterprise_rag_mcp` runs (e.g. `results/run98` on the
Spark); re-run the baseline when comparing.

## Verification
- `python -m benchmark.worker plan experiments/enterprise-rag-blueprint-rag.yaml`
  expands to one config, 20 questions.
- Smoke run on the DGX Spark (Blueprint `feat/mcp-integration` at c5780df,
  models from the Spark `.env`: `openai:qwen3.8-27b` via `OPENAI_COMPAT_*`,
  `nomic-embed-text` on local Ollama), 3 SQuAD questions,
  `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false --no-mlflow`: exit 0,
  `answer_match_rate` 1.0 (3/3), 5.0 contexts per question (top-k applied),
  6.2 s total. No Blueprint chain packages had to be installed in the Spark
  venv.
- Full run on the Spark (20 SQuAD questions, critic `openai:claude-opus-5-5`,
  before the context label): faithfulness 0.59, answer correctness 0.52,
  context recall 0.95, context precision 0.91, `answer_match_rate` 0.80.
- Re-scoring the same 20 answers with RAGAS: with `Document: <source>`
  contexts faithfulness rises to 0.87; additionally dropping the citation
  sentences from the answers gives 0.92. Answer correctness stays at 0.53
  with the label and reaches 0.60 without the citation sentences; it compares
  against the short SQuAD span, so the Blueprint's three-sentence answers
  keep it low. Context recall moved 0.95 → 0.85 between two scorings of
  identical contexts, so differences below about 0.1 on 20 questions are
  critic noise.
- Not reported for this adapter: token counts (0) and `num_chunks` (0); the
  Blueprint does not return token usage and the adapter builds the index.

## Follow-ups
- Blueprint `main` reads its model settings only at import and hard-codes
  `top_k=5` and `nomic-embed-text`; this manifest needs `feat/mcp-integration`
  (c5780df or later).
