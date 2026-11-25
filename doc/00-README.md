# Documentation Index

All documents in `doc/` are listed below, grouped by purpose. Status legend:

- **current** — authoritative for its topic; kept up to date.
- **experimental** — working notes for a feature under development.
- **historical** — point-in-time analysis or results; kept for provenance,
  not maintained. Archived material lives in [archive/](archive/README.md).

## Start here

| Document | Status | Contents |
|----------|--------|----------|
| [Benchmark Your RAG](Benchmark_Your_RAG.md) | current | **Main integration guide**: choose an integration mode (HTTP / Python plugin / managed / MCP / internal), dataset prep, minimal manifest, smoke-test checklist |
| [Environment Variables Reference](env-vars.md) | current | Full reference for every env var (moved out of the root README): core, HTTP adapter, parsers, retrieval, datasets, MCP, Blueprint plugins, generation, evaluation, experiment lineage, cost monitoring |
| [Implementation Overview](implementation-overview.md) | current | How the codebase is organized: pipeline stages, adapters, orchestration, evaluation, reporting |

## Running benchmarks

| Document | Status | Contents |
|----------|--------|----------|
| [Results and MLflow](results-and-mlflow.md) | current | Output layout, storage policy (issue #66: MLflow is the canonical durable record, `results/runN/` is a temporary working area, `python -m benchmark.cleanup`), compacted sample reports, LLM answer judge re-scoring, MLflow UI comparison and aggregate summary runs, opt-in per-sample MLflow Tracing (`mlflow_tracing_enabled`) with ground-truth expectations on traces, opt-in post-run MLflow scorers judged by the critic LLM (`mlflow_scorers`, incl. the custom RAGAS-style `answer_correctness` judge), experiment lineage (experiment/system/dataset/scorer provenance) and paired run comparisons (`python -m benchmark.lineage compare`) |
| [RAGAS Runtime](ragas-runtime.md) | current | Shortening RAGAS critic runtime: `ragas_metrics` / `RAGAS_METRICS` metric selection, measured `EVAL_CRITIC_MAX_WORKERS` 8 vs 16 on TechQA, smaller `sample_size` for sweeps |
| [LLM Load Test](llm-load-test.md) | current | Integrated LLM latency/throughput testing inside a benchmark run: `llm_performance_*` settings, `generation` vs `load_test` sources, where results land |
| [Multi-Hop Retrieval](multihop-retrieval.md) | current | Why "I cannot answer" refusals happen on HotpotQA; paragraph-level shared corpus, `paragraph` chunking, `RETRIEVAL_MULTIHOP` iterative retrieval, recommended .env |
| [Experiment TODOs](experiment-todos.md) | experimental | What to test next with the existing manifests: smoke tests, OCR parser benchmarks, planned comparisons |
| [ClearML Agent Workflow](clearml-agent.md) | current | Clone/edit/enqueue benchmark tasks via ClearML: baseline task creation, agent daemon, hyperparameter cloning, what is (not) published |
| [ClearML Usage Notes](clearml_usage.md) | historical | Older, shorter ClearML notes; [clearml-agent.md](clearml-agent.md) is the current guide |

## Integrating external systems

| Document | Status | Contents |
|----------|--------|----------|
| [External RAG System Usage](External_RAG_System_Usage.md) | current | HTTP adapter in depth: request/response contract, dotted fields, auth headers, token/cost fields, env vars, plugin loading via `RAG_ADAPTER_MODULES` |
| [External RAG Python Adapter](External_RAG_Python_Adapter.md) | current | Python plugin adapter without HTTP: skeleton, `RagSystemOutput` fields, component sweep options, component injection (`supports_components` / `RAG_ADAPTER_ACCEPTS`) |
| [Managed RAG System Usage](Managed_RAG_System_Usage.md) | current | Managed/hosted adapters with their own index lifecycle (`rag_managed_*` config, `ingest()` protocol, OptimiseRAG reference, how to add another provider) |
| [Benchmark Your Parser](Benchmark_Your_Parser.md) | current | Parser adapter family: `benchmark/parsing/` protocol, HTTP parser adapter, `PARSER_*` config, corpus parsing |
| [MCP Benchmarking](mcp-benchmarking.md) | current | Evaluating MCP servers: result/execution modes, YAML + `.env` configuration, full metric reference (reliability, latency, tool use, evidence budget), fairness validation, troubleshooting |
| [MCP Host Benchmarking](mcp-host-benchmarking.md) | current | Benchmarking a RAG system that calls MCP servers itself (Enterprise RAG Blueprint `enterprise_rag_mcp` adapter): Blueprint LLM / embedding / top-k selected from the manifest matrix, call success rates, latency, budget exhaustion, tool selection vs. `expected_tools`, deterministic answer match per category, 1000-question / 240-document fixture with its own MCP server, RAG only vs. RAG + MCP vs. MCP only, 210 trust questions (freshness, permission, robustness, unanswerable) scored by expected behaviour and forbidden values, reference result MCP off vs. on |
| [Enterprise RAG Blueprint Setup](enterprise-rag-blueprint-setup.md) | current | Step-by-step guide for evaluating the Enterprise RAG Blueprint with MCP: `.env` block, MCP server config, model endpoints with Bearer keys (OpenAI-compatible, Ollama, proxies), manifest with LLM / embedding / top-k matrix, dense or hybrid (BM25 + RRF) Blueprint retrieval, smoke and full runs, `http` variant, troubleshooting |
| [Enterprise RAG Blueprint Benchmark](Enterprise_RAG_Blueprint_Benchmark.md) | current | Ready-made Blueprint comparison: manifests and workflow for benchmarking the Enterprise RAG Blueprint against the internal pipeline (see also the Setup and Results guides above) |
| [OptimiseRAG Benchmark](OptimiseRAG_Benchmark.md) | current | Concrete managed-adapter benchmark setup (manifests, smoke runs) |

## Results and analyses

| Document | Status | Contents |
|----------|--------|----------|
| [Enterprise RAG Blueprint Results](enterprise-rag-blueprint-results.md) | historical | Results overview of the Blueprint plain-RAG runs on the DGX Spark (`results/Enterprise-RAG/run1..run8`): SQuAD and RAGBench TechQA, nomic vs Qwen3-Embedding-4B, recursive 512/1024/2048 × overlap 64/128 and semantic chunking, top-k 5/10/12, dense vs hybrid retrieval (run5 vs run6), gold-doc retrieval metrics (run7), query refinement × hybrid alpha ablation with McNemar tests (run8), NLG metrics re-scored after the refusal-check fix (run7, run8, issue #57), refusal analysis, combination matrix, failures, runtime, findings and metric caveats |
| [Enterprise RAG TechQA Refusal Analysis](enterprise-rag-techqa-refusal-analysis.md) | historical | Why ~⅓ of Blueprint answers on RAGBench TechQA (run6) are refusals: `declined_rate` phrase heuristic false positives (fixed, see `changes/2026-11-16-declined-rate-false-positives.md`; NLG refusal check fixed in `changes/2026-11-24-refusal-nlg-false-positives.md`), unanswerable questions in the dataset, passage-level retrieval misses within the correct documents (now measured automatically as `relevant_sentence_recall`, see `changes/2026-11-16-relevant-sentence-recall.md`), gold-ID gaps (now fixed for per-sample datasets, see `changes/2026-11-13-gold-doc-ids-per-sample-datasets.md`); RAGAS runtime breakdown; re-running a failed config; next steps |
| [Blueprint Regression Benchmark](blueprint-regression-benchmark.md) | current | Fixed regression benchmark for iterative Blueprint changes (issue #64): versioned TechQA question set v1 (dev 500 = run7–9 questions, holdout 200, answerable/unanswerable categories, sha256 manifest), immutable reference baseline + current-best + one-change candidate manifests, metric/label/judge per dimension (correctness, faithfulness, retrieval, abstention, latency, cost), `python -m benchmark.regression freeze/verify/compare` with paired cluster-bootstrap CIs, per-question sign test, judge/dataset config check and accept/reject/inconclusive verdict from `criteria.yaml` (ADR-006), holdout protocol |
| [Calibration: direct RAGAS vs. MLflow scorers](14-ragas-vs-mlflow-calibration.md) | historical | Issue #46, `results/run100` (50 SQuAD, critic gemma3:12b): no MLflow scorer meets mean \|Δ\| < 0.05 + Spearman > 0.9; `ragas_factual_correctness` / `retrieval_relevance` are not substitutes; MLflow defaults flood Ollama (38–100 % failures), throttled 2.4× slower per sample. Decision: side by side, direct RAGAS stays default |

## Architecture decisions and agent guides

| Location | Status | Contents |
|----------|--------|----------|
| [adr/](adr/) | current | Architecture decision records: managed-adapter seam, YAML-first manifests, worker-only orchestration, deferred cleanup, parser adapter family, regression criteria |
| [agents/](agents/) | current | Notes for coding agents working on this repo: issue tracker usage, triage labels, domain-doc conventions |
| [changes/](changes/README.md) | current | Change log: one record per coding session — what changed, why, migration, verification. Format and template inside |
| [archive/](archive/README.md) | archived | Superseded documents kept for research provenance |
| [superpowers/](superpowers/) | experimental | Agent-workflow plans and specs (historical working material) |

## Gap analyses (May 2026 deep-research pass)

Historical review documents (generated 2026-05-18, agents: 3 parallel). Many
findings have since been addressed — check `doc/changes/` and the ADRs before
acting on them; they are kept as the project's research record.

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
| 09 | [Framework Status Assessment](09-framework-status-assessment.md) | Point-in-time overall status snapshot |
| 10 | [RAG Evaluation Best Practices](10-rag-evaluation-best-practices.md) | Evaluation methodology research |
| 11 | [OCR Benchmarking SOTA](11-ocr-benchmarking-sota.md) | State of the art for OCR/parser benchmarking |
| 12 | [MCP in RAG: State of the Art](12-mcp-in-rag-state-of-the-art.md) | MCP-for-RAG research background |

Further historical research notes live in `Optimizations/` (see its README).
