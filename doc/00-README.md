# Deep Research: RAG Benchmarking Framework

Generated: 2026-05-18 | Sources: 40+ | Agents: 3 parallel

## Documents

| # | Document | Key Findings |
|---|----------|-------------|
| 01 | [Architecture Gaps](01-architecture-gaps.md) | 10 structural issues: no abstract interfaces, 450-line god function, unbounded cache, agentic dedup needed |
| 02 | [Feature Improvements](02-feature-improvements.md) | 6 categories: hybrid BM25+Dense, enable all 6 RAGAS metrics (3 disabled), LLM-as-judge, more datasets |
| 03 | [New Functionalities](03-new-functionalities.md) | 7 areas: Self-RAG, CRAG, multi-modal, Bayesian opt, REST API, Docker, continuous benchmarking |
| 04 | [Testing Gaps](04-testing-gaps.md) | Zero coverage: MLflow, custom metrics, reporting, agentic, GPU monitoring, E2E pipeline |
| 05 | [Competitive Analysis](05-competitive-analysis.md) | vs RAGAS/TruLens/DeepEval/RAGChecker. 3/6 RAGAS metrics enabled. 1 embedding model. Knowledge leakage undetected |
| 06 | [Research Improvements](06-research-improvements.md) | No CIs, no significance tests, no baselines, no ablation. Only 16% of benchmarks use stats |
| 07 | [Code Quality Issues](07-code-quality-issues.md) | 9 hardcoded values, silent failures, no circuit breaker, no version pinning |
| 08 | [Priority Roadmap](08-priority-roadmap.md) | 4 tiers. Start: enable all RAGAS metrics, .env.example, pyproject.toml. Then: stats + more datasets |

## Guides

| Document | Contents |
|----------|----------|
| [MCP Benchmarking](mcp-benchmarking.md) | Evaluating MCP servers: result/execution modes, YAML + `.env` configuration, full metric reference (reliability, latency, tool use, evidence budget), fairness validation, troubleshooting |
| [Enterprise RAG Blueprint Setup](enterprise-rag-blueprint-setup.md) | Step-by-step guide for evaluating the Enterprise RAG Blueprint with MCP: `.env` block, MCP server config, model endpoints with Bearer keys (OpenAI-compatible, Ollama, proxies), manifest with LLM / embedding / top-k matrix, smoke and full runs, `http` variant, troubleshooting |
| [MCP Host Benchmarking](mcp-host-benchmarking.md) | Benchmarking a RAG system that calls MCP servers itself (Enterprise RAG Blueprint `enterprise_rag_mcp` adapter): Blueprint LLM / embedding / top-k selected from the manifest matrix, call success rates, latency, budget exhaustion, tool selection vs. `expected_tools`, deterministic answer match per category, 1000-question / 240-document fixture with its own MCP server, reference result MCP off vs. on |
| [Change Log](changes/README.md) | One record per coding session: what changed, why, migration, verification. Format and template inside |
| [Multi-Hop Retrieval](multihop-retrieval.md) | Why "I cannot answer" refusals happen on HotpotQA; paragraph-level shared corpus, `paragraph` chunking, `RETRIEVAL_MULTIHOP` iterative retrieval, recommended .env |
