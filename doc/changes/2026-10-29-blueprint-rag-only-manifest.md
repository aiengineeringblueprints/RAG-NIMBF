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
  `retrieval_multihop`, `retrieval_use_hyde`) are pinned against `.env` drift.

Manifest fields that reach the Blueprint: `llm_models`, `embedding_models`,
`retrieval_top_k`, `generation_temperature`, the chunking axes (index build),
`enterprise_rag_mcp_query_refinement` and `enterprise_rag_mcp_guardrails`.
`prompt_templates` and `reranker_models` have no effect; the Blueprint uses its
own prompt and Chroma retriever. Setup and endpoints:
[enterprise-rag-blueprint-setup.md](../enterprise-rag-blueprint-setup.md)
(skip the MCP server config and `MCP_IDENTITY_SECRET`; neither is needed with
MCP off).

## Behaviour changes and migration
None.

## Verification
- `python -m benchmark.worker plan experiments/enterprise-rag-blueprint-rag.yaml`
  expands to one config, 20 questions.
- Smoke run on the DGX Spark: see below.

## Follow-ups
- Blueprint `main` reads its model settings only at import and hard-codes
  `top_k=5` and `nomic-embed-text`; this manifest needs `feat/mcp-integration`
  (c5780df or later).
