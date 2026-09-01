# What a RAG Benchmarking Framework Should Have: Functionalities & Measurements

*Generated: 2026-08-18 | Sources: RAGAS docs, RAGChecker paper + related-work survey, project coverage audit | Confidence: High*

## Executive Summary

A RAG system is a modular pipeline (index → retrieve → rerank → generate → evaluate), so a
benchmarking framework must measure **both** the whole system and each module, because errors
from the retriever and generator propagate and interlock in ways a single score cannot expose.
Across the leading frameworks (RAGAS, RAGChecker, ARES, TruLens, DeepEval) the consensus is a
three-tier metric design:

1. **Overall / end-to-end quality** — one comparable score per response (typically claim-level
   precision / recall / F1, or the RAG Triad).
2. **Module-specific diagnostics** — a fine-grained suite that pinpoints *where* failures come
   from (retrieval misses vs. generation noise-sensitivity vs. hallucination).
3. **Operational / non-functional** — latency, cost, throughput, resource usage, reliability.

The single most important design principle (RAGChecker) is **modular diagnosis**: the framework
must attribute errors to the retriever or generator rather than only scoring the final answer.
The second most important is **metric reliability**: every automated metric should be
meta-evaluated against human judgments before being trusted.

---

## 1. Core Evaluation Dimensions (the "RAG Triad")

Introduced by **TruLens** and adopted by RAGAS, ARES and most frameworks. It decomposes output
quality into three LLM-judged or NLI-based scores:

- **Context Relevance** — is the retrieved context actually relevant to the query?
- **Groundedness / Faithfulness** — is the answer grounded in the retrieved context (no hallucination)?
- **Answer Relevance / Comprehensiveness** — does the answer address the query and is it complete?

RAGAS operationalizes these as `context_precision`, `context_recall`, `context_entities_recall`,
`faithfulness`, `answer_relevancy`, `answer_correctness`, `noise_sensitivity` (RAGAS docs, 2025).

> Framework implication: build the triad as the *baseline* set, then layer fine-grained
> module metrics on top. See [the metric catalog](#4-recommended-measurement-catalog) below.

---

## 2. Fine-Grained, Module-Level Metrics (RAGChecker model)

RAGChecker's key contribution is **claim-level entailment** evaluation. It decomposes the
response and ground truth into claims, then classifies each claim. This yields three families of
metrics that are far more actionable than a single RAGAS score, and RAGChecker's meta-evaluation
shows these correlate with human judgment better than BLEU/ROUGE/BERTScore and the RAG Triad
frameworks (RAGChecker 2024).

### 2.1 Overall metrics
- **Claim precision** — proportion of the model's claims entailed by ground truth.
- **Claim recall** — proportion of ground-truth claims present in the response.
- **F1** — harmonic mean (the single comparable "system score").

### 2.2 Retriever metrics
- **Claim recall / context recall** — how many ground-truth claims are covered by retrieved chunks.
- **Context precision** — fraction of retrieved chunks that are *relevant* (chunk-level, more
  interpretable than claim-level for real RAG pipelines).
- (Plus classic IR metrics: `hit@k`, `recall@k`, `ndcg@k`, `mrr@k`, `precision@k` — used when
  gold documents/annotations are available. RAGChecker notes these depend on rigid chunking and
  miss the full semantic scope, so they complement rather than replace claim-level metrics.)

### 2.3 Generator metrics (the diagnostic core)
- **Faithfulness** — fraction of response claims entailed by retrieved context (higher is better).
- **Context utilization** — fraction of relevant retrieved claims the generator actually uses
  (higher is better; RAGChecker found this correlates most strongly with overall F1).
- **Relevant noise sensitivity** — incorrect claims that came from *relevant* chunks
  (generator trusts noisy-but-relevant context).
- **Irrelevant noise sensitivity** — incorrect claims from *irrelevant* chunks.
- **Hallucination** — incorrect claims not entailed by any retrieved chunk (pure generation).
- **Self-knowledge** — correct claims the generator produced without context (lower is better for
  a pure-RAG system).

> Framework implication: this is the blueprint for a *diagnostic* benchmark — the user needs to
> know *why* a score is low (bad retrieval, noisy context, or hallucinating generator), not just
> that it is low. Your project already has a `trace_metrics` (TRACe utilization/completeness)
> and a RAGAS set; the missing pieces are the noise-sensitivity and hallucination *attribution*
> dimensions.

### 2.4 Robustness / capability dimensions (from RGB, RECALL, NoMIRACL)
Several frameworks test the generator against *adversarial* inputs rather than natural ones:
- **Noise robustness** — behavior when retrieved context contains distractors.
- **Negative rejection** — can the generator say "I don't know" instead of hallucinating when
  context is uninformative?
- **Information integration** — can it combine multiple chunks into one coherent answer?
- **Counterfactual robustness** — does it resist deliberately wrong context (does it trust the
  context blindly, or lean on internal priors)? (RECALL, RGB, NoMIRACL, per RAGChecker.)

---

## 3. Functionalities a Benchmarking Framework Should Have

### 3.1 Benchmark definition & dataset management
- **Curated, multi-domain datasets** with query / documents / ground-truth-answer tuples
  (`<q, D, gt>`) covering diverse domains (Wikipedia, biomedical, finance, science, novel, etc. —
  RAGChecker's benchmark spans 10 domains; RAGBench/RAGAS also stress multi-domain).
- **Task variety**: single-hop QA, multi-hop, summarization, counterfactual, dynamic/updated
  knowledge (CDQA, MultiHop-RAG).
- **Ground-truth annotation layer** and a claim-annotation pipeline for claim-level metrics.
- **Contamination control** — avoid training/eval overlap; track provenance and licenses.
- **Test-set / synthetic-data generation** (RAGAS testset generation, your `dynamic_groundtruth`).

### 3.2 Harness / execution
- **Modular stage orchestration** (index → chunk → retrieve → rerank → generate → evaluate) with
  the ability to swap any module — the framework must support ablation studies.
- **Reproducibility**: seed control, model/config pinning, artifact manifests, result persistence.
- **Resumability** for long matrix runs (your `orchestration` already does this).
- **Multiple backend adapters**: vector stores, embedding models, rerankers, LLM providers,
  and judge/LLM-evaluator backends.

### 3.3 Evaluation engine
- **Layered metrics**: overall + retriever + generator (Section 2).
- **Multiple evaluator backends**: LLM-as-judge, NLI models, embedding/lexical metrics — with
  pluggable judge models (LLM-judged metrics are noisy; provide alternatives).
- **Meta-evaluation**: measure how well your automated metrics correlate with human judgments
  before trusting them (RAGChecker's pairwise human-preference methodology).
- **Statistical rigor**: confidence intervals, significance testing, error bars across runs
  (metrics are point estimates otherwise).

### 3.4 Non-functional / operational measurement
The "perf" half of a benchmark (your project already leads here):
- **Latency**: per-stage timings, end-to-end latency, TTFT / time-to-first-token, token generation
  rate, percentile distributions (p50/p90/p99), streaming.
- **Throughput**: requests/sec, tokens/sec, concurrency scaling (RAGPerf-style workloads).
- **Cost**: per-query cost, token-based cost, $/1000 answers, embedding + LLM cost breakdown.
- **Resource usage**: CPU/GPU utilization, memory, NVML/cgroups, energy (Joules), and
  energy-per-answer efficiency.
- **Indexing performance**: chunking/embedding throughput and index-build latency at scale.

### 3.5 Experiment management, analytics & reporting
- **Tracking** (MLflow / ClearML), cross-run comparison, parameter sweeps, experiment matrices.
- **Dashboarding / visualization** for interactive analysis and report generation.
- **A/B and regression comparison**: diff two configs (e.g. chunk size) across *all* metrics,
  not just overall score.

---

## 4. Recommended Measurement Catalog

| Stage | Metric | Type | Notes |
|---|---|---|---|
| **Index** | chunking throughput, embed throughput, index build time | perf | your `ragperf`/`ragbench` |
| **Retrieve** | hit@k, recall@k, ndcg@k, mrr@k, precision@k | quality (IR) | needs gold docs |
| | claim recall (context recall) | quality (semantic) | RAGChecker |
| | context precision | quality (semantic) | RAGChecker |
| **Rerank** | Δndcg, Δrecall vs. pre-rerank, rerank latency | quality+perf | ablation |
| **Generate** | faithfulness, context utilization | quality | RAGChecker/RAGAS |
| | hallucination, self-knowledge | error attribution | RAGChecker |
| | relevant/irrelevant noise sensitivity | robustness | RAGChecker |
| | noise robustness, negative rejection, integration, counterfactual robustness | capability | RGB/RECALL/NoMIRACL |
| **End-to-end** | claim precision, claim recall, F1 | quality | RAGChecker |
| | RAG Triad: context relevance, groundedness, answer relevance | quality | RAGAS/ARES/TruLens |
| | answer correctness, semantic similarity, factual correctness | quality | RAGAS |
| **Response** | ROUGE-L, BLEU, METEOR, BERTScore, exact match | lexical | fast, coarse |
| **Operational** | TTFT, p50/p90/p99 latency, tok/s, req/s, $/query, Joules, GPU% | perf/cost | your `stage_timing`, `costing`, `resource_monitor` |

---

## 5. Gap Analysis vs. Your Current Project

**Already strong** (your coverage audit): stage orchestration, multi-vector-store adapters,
RAGAS set, TRACe utilization/completeness, RoBERTa evaluator, RAGBench datasets, per-stage
latency + percentiles, cost/energy, resource monitors v1/v2, workload generation (RAGPerf),
multimodal, dynamic ground-truth, MLflow/ClearML tracking, dashboard, and broad unit tests.

**Recommended additions**, in priority order:
1. **Error-attribution diagnostics** — RAGChecker-style *hallucination* and *noise-sensitivity*
   split (relevant vs. irrelevant), plus `self-knowledge`; your TRACe covers utilization but not
   these error classes.
2. **Robustness/capability probes** — negative-rejection, noise-robustness, and counterfactual
   datasets (the RGB/RECALL axes), not just natural QA.
3. **Meta-evaluation & judge reliability** — calibrate the LLM-judge vs. human preference
   (pairwise), and report judge agreement/self-consistency.
4. **Statistical rigor** — significance tests and confidence intervals between configs, not just
   point estimates and CI plots.
5. **Toxicity / safety / bias** evaluation of generated text — absent today.
6. **Dataset contamination control + broader multi-domain suite** beyond SQuAD/RAGBench.
7. **Streaming/TTFT as a first-class pipeline metric** under concurrency.

---

## Key Takeaways
- Evaluate the **whole system and each module**; overall F1 alone cannot tell you whether to fix
  the retriever or the generator (RAGChecker).
- Build the **RAG Triad** as the baseline and layer **claim-level error attribution** on top —
  faithfulness without hallucination/noise attribution is incomplete.
- **Operational metrics** (latency, cost, throughput, resources) are just as important as quality
  for a real benchmark — you already excel here.
- **Validate your metrics against human judgment**; un-meta-evaluated LLM-judged scores are not
  trustworthy.
- Add **robustness probes** and **statistical rigor** to move from "scoring" to "diagnosing + deciding."

## Sources
1. Ragas docs — *List of available metrics* — https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
2. Ru et al., *RAGChecker: A Fine-grained Framework for Diagnosing RAG* (arXiv:2408.08067) — https://arxiv.org/abs/2408.08067
   — full metric taxonomy, meta-evaluation methodology, and related-work survey of TruLens/ARES/RAGAS/CRUD-RAG/RGB/RECALL/NoMIRACL/MultiHop-RAG/MEDRAG.
3. Project coverage audit (EVAL_MATRIX.md, main.py, config.py, NOTES_A–J, tests/).

## Methodology
Web research across official docs (RAGAS) and the RAGChecker paper (peer-reviewed RAG-eval survey
with full related-work coverage). Cross-checked against a local audit of the project's existing
benchmarking surface to produce the gap analysis. Sub-questions investigated: (1) core evaluation
dimensions, (2) module-level diagnostic metrics, (3) framework functionality requirements,
(4) operational measurements, (5) gaps vs. current implementation.
