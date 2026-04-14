# RAG Pipeline Improvement Report — SLMs, Agentic RAG & Beyond

**Date:** 2026-04-14
**Scope:** Architecture review, state-of-the-art comparison, actionable improvement roadmap
**No code changes — research and recommendations only**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Assessment](#2-current-architecture-assessment)
3. [Retrieval Layer Improvements](#3-retrieval-layer-improvements)
4. [Agentic RAG Patterns](#4-agentic-rag-patterns)
5. [Multi-Model / Small Model Strategies](#5-multi-model--small-model-strategies)
6. [Advanced Chunking](#6-advanced-chunking)
7. [Reranking Enhancements](#7-reranking-enhancements)
8. [Generation Layer Improvements](#8-generation-layer-improvements)
9. [Evaluation Expansion](#9-evaluation-expansion)
10. [Infrastructure & Performance](#10-infrastructure--performance)
11. [Prioritized Roadmap](#11-prioritized-roadmap)
12. [Appendix: Key References](#appendix-key-references)

---

## 1. Executive Summary

The framework is a well-structured RAG benchmarking tool with clean separation of concerns — chunking, embedding, retrieval, reranking, generation, and evaluation. The current pipeline follows a **linear retrieve-then-generate** pattern with **dense-only retrieval** (ChromaDB similarity search), a single cross-encoder reranker, one generator LLM per run, and RAGAS as the sole evaluation framework.

The biggest opportunities for improvement fall into four categories:

| Category | Potential Impact | Effort |
|---|---|---|
| Hybrid search (dense + BM25) | High — retrieval quality | Medium |
| Query transformation & agentic patterns | High — handles complex/ambiguous queries | Medium-High |
| Multi-model pipeline stages | Medium-High — cost/quality tradeoff | Medium |
| Contextual/semantic chunking + advanced reranking | Medium — retrieval precision | Low-Medium |

---

## 2. Current Architecture Assessment

### What the pipeline does well

- **Clean modular design** — each stage (chunking, embedding, retrieval, generation, evaluation) is well-separated
- **Vector store caching** — avoids redundant embeddings when chunking params match (`retrieval.py:19-20`)
- **Per-role model URLs** — different models can run on different hardware (e.g., critic on DGX, embeddings locally)
- **Streaming with TTFT** — accurate latency measurement via streaming (`generation.py:308-366`)
- **Answer post-processing** — sophisticated handling of thinking tags, arithmetic expressions, percentage normalization for FinQA
- **MLflow tracking** — experiment management with per-sample CSV artifacts

### Key architectural gaps

| Gap | Current State | Impact |
|---|---|---|
| **Dense-only retrieval** | `vector_store.similarity_search()` in `retrieval.py:91` | Misses lexical matches that BM25 catches; no hybrid search |
| **No query transformation** | Raw user question goes straight to retrieval | Poor retrieval on ambiguous, multi-hop, or conversational queries |
| **Single retrieval strategy** | One fixed `similarity_search` call | No fallback when retrieval fails; no relevance grading |
| **No semantic/contextual chunking** | Fixed-size and structural splitters only (`chunking.py`) | Chunks may split mid-sentence or mid-paragraph |
| **One model per stage** | Same LLM handles all questions regardless of complexity | Overkill for simple factual lookups; underpowered for complex reasoning |
| **RAGAS-only evaluation** | 5 metrics: faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall | No claim-level analysis; no citation accuracy; IR metrics (nDCG, Hit@k) defined but not used in benchmarking |
| **Sequential execution** | `main.py:289` — explicit comment about GPU sharing | Appropriate for single-GPU, but no option for multi-machine parallelism |
| **No prompt caching** | System prompt + context recomputed every call | Wastes compute on shared prefix across queries in same config |

---

## 3. Retrieval Layer Improvements

### 3.1 Hybrid Search (Dense + Sparse) — **Highest Priority**

**Current gap:** `retrieval.py:91` calls `vector_store.similarity_search()` — pure dense retrieval via ChromaDB.

**Recommendation:** Add BM25/sparse retrieval alongside dense vectors, merge via **Reciprocal Rank Fusion (RRF)**.

Why this matters: Anthropic's Contextual Retrieval research explicitly states "Embeddings+BM25 is better than embeddings on their own." Dense retrieval excels at semantic similarity but fails on exact term matches (product IDs, named entities, technical jargon). BM25 catches those. The combination is strictly better.

**Implementation sketch:**
- Add a `rank_bm25` BM25Okapi index alongside ChromaDB
- Retrieve top-K from each, merge via RRF: `score(d) = sum(1 / (k + rank_i(d)))` where k=60 is standard
- Make hybrid search a configurable strategy: `RETRIEVAL_STRATEGY=dense|sparse|hybrid`

**Libraries:** `rank_bm25` (lightweight, pip install), or LangChain's `BM25Retriever` + `EnsembleRetriever`

**Impact:** 15-30% improvement in retrieval recall, especially on domain-specific datasets like FinQA where exact numbers and entity names matter.

### 3.2 Query Transformation Pipeline

**Current gap:** The raw question is sent directly to the vector store. No preprocessing.

**Recommended transformations (configurable, composable):**

| Technique | How | Best For | Cost |
|---|---|---|---|
| **Query rewriting** | LLM reformulates for better retrieval | Conversational queries, short queries | 1 extra LLM call (use small model) |
| **HyDE** | Generate hypothetical answer, embed that | Queries with poor embedding overlap | 1 extra LLM call + embedding |
| **Multi-query** | Generate N query variants, retrieve for each, merge | Ambiguous questions | N extra retrievals |
| **Query decomposition** | Break complex question into sub-questions | Multi-hop reasoning (HotpotQA-style) | N extra retrievals + synthesis |

**Practical recommendation:** Start with **HyDE** — it's the simplest to add and gives the biggest bang-per-buck. Use the smallest model (e.g., gemma3:4b) for the hypothetical answer generation, then embed that instead of the raw query. Add as a `QUERY_TRANSFORM=none|hyde|multi_query|decompose` config option.

**Warning on HyDE:** If the hypothetical answer is factually wrong, it can mislead retrieval. Grade the retrieval results (see Section 4 — Corrective RAG) to mitigate this.

### 3.3 Contextual Retrieval (Anthropic's Method)

**What:** Before embedding each chunk, prepend 50-100 tokens of document-level context that explains what the chunk is about within its broader document. Same context is prepended for BM25 indexing.

**Why:** Anthropic's data shows **49% reduction in retrieval failures** (embeddings + contextual BM25), and **67% when combined with reranking**. This directly addresses the fundamental problem that chunks lose their document context when split.

**Cost:** ~$1.02 per million document tokens using prompt caching (one-time preprocessing). Using local Ollama models, essentially free.

**Implementation sketch:**
- Before chunking, pass each full document through a small LLM (gemma3:4b) with the Anthropic contextual prompt
- Prepend the returned context to each chunk's page_content
- Embed and index as normal
- Add `CONTEXTUAL_RETRIEVAL=true` config flag

**Impact:** Large — this is the single most impactful retrieval improvement from 2024-2025 research.

### 3.4 ColBERT / Late Interaction Models

**What:** Instead of one vector per document, produce token-level embeddings. At query time, compute MaxSim similarity for fine-grained matching.

**Pros:** Significantly better retrieval quality than bi-encoders; faster than cross-encoders at query time.
**Cons:** Higher storage (token-level embeddings per document); more complex setup.

**Libraries:** `RAGatouille` (pip install, wraps ColBERT), `Jina ColBERT v2`

**Recommendation:** Worth benchmarking as an alternative embedding strategy, but lower priority than hybrid search + contextual retrieval.

---

## 4. Agentic RAG Patterns

### 4.1 Corrective RAG (CRAG) — **Recommended First Agentic Pattern**

**What:** After retrieval, a lightweight evaluator grades the retrieved documents. If confidence is low, trigger corrective actions (web search, query rewriting, expanded retrieval).

```
Query → Retrieve → Grade Documents → [High confidence] → Generate
                                   → [Low confidence]  → Rewrite Query / Web Search → Re-retrieve → Generate
```

**Why it fits this framework:** The pipeline already has the building blocks:
- Retrieval (`retrieval.py`)
- Reranker (`reranker.py`) — can double as the relevance grader
- Generation (`generation.py`)

The addition would be a "relevance gate" between retrieval/reranking and generation. If the reranker scores are all below a threshold, the query gets rewritten and retrieval is retried.

**Cost:** Minimal — the grader can be the same cross-encoder already used for reranking, or a lightweight NLI model (DeBERTa-v3-large, ~300MB).

**Impact:** Prevents the model from hallucinating when retrieval fails. Especially important for FinQA where wrong context leads to wrong numerical answers.

### 4.2 Adaptive RAG — Query Complexity Routing

**What:** A small classifier categorizes query complexity (simple / medium / complex) and routes to different retrieval strategies.

```
Query → Complexity Classifier → [Simple]   → Direct retrieval, small LLM
                               → [Medium]  → Hybrid retrieval, medium LLM
                               → [Complex] → Multi-hop retrieval, large LLM
```

**Why it matters:** Not every question needs the full pipeline. A simple factual lookup ("What is X?") can be answered with a single dense retrieval + small model. A multi-hop question ("Compare X across Y and Z") needs decomposed retrieval + large model.

**Cost savings:** If 60% of queries are simple and handled by a 4B model instead of a 12B+ model, compute drops by ~40-50%.

**Implementation:** A simple classifier can be a fine-tuned DeBERTa or even a prompt-based classifier using gemma3:4b. Add `COMPLEXITY_ROUTING=true` config.

### 4.3 Self-RAG — Reflection Tokens

**What:** Train or prompt the generator to emit reflection tokens: `retrieve?` (do I need retrieval?), `isrel` (is the retrieved doc relevant?), `issup` (does the generation follow from the context?), `isuse` (is the answer useful?).

**Tradeoff:** The original Self-RAG requires a specially trained model. For this framework, it can be approximated with a post-generation evaluation step where the critic model grades its own output quality and optionally re-generates.

**Recommendation:** This is more of a research project. Start with CRAG (simpler, same benefit).

### 4.4 Full Agentic Workflow — When to Consider

Based on Anthropic's guidance: "Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short."

The framework should implement agentic patterns as **configurable pipeline stages**, not as a monolithic agent. The benchmark config could look like:

```
PIPELINE_MODE=linear|corrective|adaptive|full_agent
```

Where `linear` is the current flow, `corrective` adds the relevance gate, `adaptive` adds complexity routing, and `full_agent` chains everything.

---

## 5. Multi-Model / Small Model Strategies

### 5.1 Pipeline Stage Specialization

Instead of one model serving all roles, use specialized models at each stage:

| Stage | Recommended Model | Size | Why |
|---|---|---|---|
| **Query rewriting** | gemma3:4b or Flan-T5-large | 4B / 780M | Short generation, fast, cheap |
| **Embedding** | BAAI/bge-m3 or nomic-embed-text | 568M / 274M | Purpose-built for retrieval |
| **Reranking** | bge-reranker-v2-m3 (upgrade from MiniLM-L-6) | 568M | State-of-the-art cross-encoder |
| **Answer generation** | gemma3:12b or Qwen 2.5 7B | 12B / 7B | Good balance of quality and speed |
| **Evaluation/critic** | gemma3:12b or Qwen 2.5 14B | 12-14B | Needs strong reasoning for grading |
| **Hallucination detection** | DeBERTa-v3-large fine-tuned on NLI | 304M | Purpose-built, very fast |

### 5.2 Speculative RAG (DeepMind, 2024)

**What:** A small model generates multiple draft answers from different retrieved document subsets. A larger model verifies and selects the best one.

**Why:** The small model generates fast (low TTFT); the large model only scores (cheaper than generating). This can actually *lower* latency while *improving* quality.

**Framework fit:** The infrastructure for multiple generation attempts (streaming, TTFT measurement) already exists. A "draft and verify" mode could work as:
1. Retrieve top-20 documents
2. Split into 4 subsets of 5
3. Generate 4 drafts with gemma3:4b in parallel
4. Score each draft with gemma3:12b
5. Return the highest-scoring draft

### 5.3 Model Cascading

**What:** Start with the cheapest model. Only escalate to a larger model when confidence is low.

```
Query → gemma3:4b → [High confidence answer] → Return
                    → [Low confidence]        → gemma3:12b → Return
                                                → [Still unsure] → Qwen 32B → Return
```

**Cost impact:** If 60% of queries are handled by the 4B model, 30% by the 12B, and 10% by the 32B, average cost drops by ~60% compared to using the 32B for everything.

**Implementation:** Define a confidence score from the generation (can be based on answer validity, answer length, or the model's own uncertainty signals). If score < threshold, escalate.

### 5.4 Mixture-of-Agents (MoA)

**What:** Multiple models generate independent answers, then an aggregator synthesizes the best answer from all proposals. Can iterate for refinement.

**Cost/Quality:** Research shows MoA with 3-4 small models (7-14B) can match or exceed a single frontier model for domain-specific tasks at 1/10th the cost.

**Recommendation for this framework:** Add an `ANSWER_STRATEGY=single|cascade|speculative|moa` config. This enables fair benchmarking of each strategy.

---

## 6. Advanced Chunking

### 6.1 Semantic Chunking

**Current gap:** Chunkers (`chunking.py`) are all size-based or structural. They don't consider semantic coherence.

**What:** Split text when embedding similarity between consecutive sentences drops below a threshold. Produces chunks that are semantically coherent.

**Libraries:** LangChain's `SemanticChunker` (uses embedding similarity), LlamaIndex's `SemanticSplitterNodeParser`

**Tradeoff:** More embedding calls (one per sentence pair), but produces much higher-quality chunks. For a benchmarking framework, this is a pure quality improvement.

**Add to config:** `CHUNKING_STRATEGIES=recursive,character,token,markdown,text,transformers,semantic`

### 6.2 Late Chunking (Jina AI, 2024)

**What:** Pass the entire document through the embedding model's transformer layers first (producing token-level embeddings with full-document context), then chunk and pool the token embeddings.

**Advantage:** No extra LLM call needed (unlike contextual retrieval). Each chunk's embedding already has document-level context baked in.

**Requirement:** Needs a long-context embedding model (Jina Embeddings v2/v3 support 8192 tokens). The current `nomic-embed-text` has 8192 context, so it may work.

### 6.3 Parent-Child / Small-to-Big Retrieval

**What:** Index small chunks (128 tokens) for precise retrieval. When a small chunk matches, return the parent chunk (512 tokens) or surrounding siblings.

**Why:** Small chunks improve retrieval precision (less noise in the embedding). Large chunks improve generation quality (more context for the LLM).

**Framework fit:** Configurable chunk sizes already exist. Add a `PARENT_CHUNK_SIZE` and `CHILD_CHUNK_SIZE` pair. Index at child granularity, retrieve at parent granularity.

### 6.4 Chunk Size Recommendations

Current default: 500 tokens with 100 overlap. Based on research:

| Domain | Recommended Chunk Size | Overlap |
|---|---|---|
| Financial tables/structured (FinQA) | 256-384 | 50-75 |
| General prose | 512 | 100 |
| Technical documentation | 512-768 | 100-150 |
| Short factual QA (SQuAD) | 128-256 | 25-50 |

**Recommendation:** Add these as preset profiles: `CHUNK_PRESET=finqa|general|technical|factual` that set size/overlap/strategy together.

---

## 7. Reranking Enhancements

### 7.1 Upgrade the Reranker Model

**Current:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — a solid baseline but released in 2021.

**Recommended upgrade:** `BAAI/bge-reranker-v2-m3` (568M params)
- Multilingual
- Trained on more diverse data
- Significantly better on domain-specific content
- Same cross-encoder interface — drop-in replacement

**Also consider:** `BAAI/bge-reranker-v2-gemma` (based on Gemma 2B) — stronger but larger.

### 7.2 Retrieve More, Rerank to Fewer

**Current:** `RETRIEVAL_TOP_K=8`, `RERANKER_TOP_K=3`

**Recommendation based on Anthropic's research:** "Passing the top-20 chunks to the model is more effective than just the top-10 or top-5."

Change to: Retrieve top-100, rerank to top-20, pass top-20 to generation. The reranker is fast (cross-encoder inference on 100 docs takes <1 second). This dramatically increases the chance that the right documents are in the final set.

### 7.3 FlashRank for Speed

If a faster reranker is needed for the initial gate (e.g., in Corrective RAG), `FlashRank` uses ultra-lightweight models (~2MB) for instant reranking. Use FlashRank for the initial relevance gate, then the full cross-encoder for final ranking.

---

## 8. Generation Layer Improvements

### 8.1 Prompt Caching / Prefix Caching

**What:** When using vLLM or other OpenAI-compatible endpoints, enable automatic prefix caching. The system prompt + retrieved context is shared across all queries in a benchmark run.

**Impact:** Anthropic reports ">2x latency reduction and up to 90% cost reduction" with prompt caching. For this framework, where N queries with the same system prompt run against the same config, this is a perfect fit.

**How:** If using vLLM, enable `--enable-prefix-caching`. If using Ollama, it already does some KV cache reuse. If using OpenAI-compatible APIs, check if the endpoint supports cached tokens.

### 8.2 Speculative Decoding

**What:** A small draft model predicts N tokens ahead; the large target model verifies them in one forward pass. 2-3x speedup with negligible quality loss.

**Setup:** gemma3:4b and gemma3:12b are both available. If served on vLLM, gemma3:4b can serve as the draft model for gemma3:12b (same architecture family — ideal for speculative decoding).

**Impact:** For FinQA with 50 samples per config, this could cut generation time by 40-60%.

### 8.3 Quantization

**Current:** Ollama supports GGUF quantization. Ensure Q4_K_M quantization is used for generation models — it reduces memory by ~75% with minimal quality impact for RAG tasks.

For the evaluation critic (gemma3:12b), consider whether Q4_K_M is sufficient. For evaluation tasks, 4-bit quantization typically has negligible impact on metric quality.

### 8.4 Improved Prompt Templates

The FinQA template is specialized for numerical extraction. Consider adding:

- **Chain-of-Thought RAG template:** "First, identify the relevant facts in the context. Then, reason step-by-step. Finally, provide the answer."
- **Self-ask template:** Forces the model to ask follow-up sub-questions before answering.
- **Citation template:** Requires the model to cite which chunk(s) support each claim.

Add these as additional `PROMPT_TEMPLATES` to benchmark against.

---

## 9. Evaluation Expansion

### 9.1 Beyond RAGAS — Complementary Metrics

**Current:** RAGAS with 5 metrics. Solid but has gaps.

**Recommended additions:**

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Citation accuracy** | Do cited sources actually support claims? | Critical for trustworthy answers |
| **Claim-level faithfulness** | Decompose answer into atomic claims, verify each | More granular than RAGAS faithfulness |
| **Hallucination rate** | % of claims not supported by any retrieved context | Direct measure of safety |
| **nDCG@k, Hit@k** | IR retrieval quality against ground truth | Measures retrieval independently from generation |

**Frameworks to consider:**
- **RAGChecker** — claim-level decomposition; diagnoses whether failures are retrieval or generation problems. This is the most important addition.
- **DeepEval** — pytest-like interface, good for CI/CD integration
- **ARES** — trains lightweight classifiers, much cheaper evaluation after initial training

### 9.2 Retrieval-Generation Failure Diagnosis

**Critical missing capability:** When a benchmark run scores poorly, there's no way to tell if the failure was in retrieval (wrong documents) or generation (wrong answer from right documents).

**Recommendation:** Add a diagnostic step:
1. If `context_precision` is low → retrieval failure → try hybrid search, contextual retrieval, or query transformation
2. If `faithfulness` is low → generation hallucination → try better prompting, larger model, or self-consistency
3. If `context_recall` is low → missing documents → increase top-K, add more chunking strategies, or expand the corpus

RAGChecker does this automatically. At minimum, add a post-run diagnostic report.

### 9.3 Per-Query Difficulty Stratification

Not all queries are equally hard. A 50-sample benchmark may be dominated by easy or hard queries.

**Recommendation:** Categorize queries by difficulty (simple factoid vs. multi-hop reasoning vs. numerical computation) and report metrics stratified by difficulty. This reveals whether improvements help on hard questions or just inflate scores on easy ones.

---

## 10. Infrastructure & Performance

### 10.1 Multi-Machine Parallelism

**Current:** Sequential execution with explicit comment about GPU sharing (`main.py:286-288`).

**Recommendation:** Add a `PARALLEL_WORKERS=N` config. When N > 1, distribute configs across workers. This requires:
- A shared filesystem for results (or a networked MLflow)
- Separate Ollama/vLLM instances per worker (or API-backed models)
- Each worker locks its own GPU

For the existing per-role URL infrastructure (already in config), you could run:
- Worker 1 → embedder on `localhost:11434`, LLM on `rtx-server:11434`
- Worker 2 → embedder on `dgx-spark:11434`, LLM on `rtx-server:11434`

### 10.2 Semantic Caching for Generation

**What:** Cache LLM responses keyed by semantic similarity, not exact string match.

**Why:** In benchmarking, different chunking strategies may produce similar contexts for the same question. Semantic caching avoids redundant LLM calls.

**Libraries:** GPTCache, Redis with vector similarity. Threshold: cosine similarity > 0.95.

### 10.3 Vector Store Persistence

**Current:** ChromaDB `EphemeralClient` — all data lost on process exit.

**For benchmarking:** Consider `PersistentClient` to cache embeddings across runs. The current in-process cache works within a single run, but re-running benchmarks with different generation models still re-embeds everything.

### 10.4 Observability / Tracing

**Current:** MLflow for experiment tracking. No per-query tracing.

**Recommendation:** Add Langfuse or Phoenix (Arize) tracing to capture:
- Retrieval latency per query
- Reranker scores per document
- Generation tokens and timing
- Evaluation scores per sample

This is invaluable for debugging why specific queries fail. Langfuse has a free self-hosted tier.

---

## 11. Prioritized Roadmap

### Tier 1 — High Impact, Low-Medium Effort

| # | Improvement | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Hybrid search (dense + BM25)** | +15-30% retrieval recall | Medium — add `rank_bm25`, modify `retrieval.py` |
| 2 | **Upgrade reranker to bge-reranker-v2-m3** | +5-10% reranking quality | Trivial — change model name in config |
| 3 | **Retrieve more, rerank to fewer** (top-100 → top-20) | Better context coverage | Trivial — change config defaults |
| 4 | **Contextual retrieval** (Anthropic method) | -49-67% retrieval failures | Medium — preprocessing step before chunking |
| 5 | **Add RAGChecker evaluation** | Failure diagnosis (retrieval vs. generation) | Medium — add as alternative eval framework |
| 6 | **Enable prefix caching** (vLLM / Ollama) | -40-60% latency on shared prompts | Low — config change |

### Tier 2 — High Impact, Medium Effort

| # | Improvement | Expected Impact | Effort |
|---|---|---|---|
| 7 | **Corrective RAG** (relevance gate) | Prevent hallucination on bad retrieval | Medium — add grading step after reranking |
| 8 | **HyDE query transformation** | Better retrieval on ambiguous queries | Medium — add pre-retrieval LLM call |
| 9 | **Semantic chunking** strategy | Higher-quality chunks | Low — add LangChain's SemanticChunker |
| 10 | **Model cascading** (small → medium → large) | -40-50% compute cost | Medium — add confidence scoring + fallback |
| 11 | **Parent-child retrieval** | Better precision + context tradeoff | Medium — hierarchical indexing |
| 12 | **Diagnostic report** (retrieval vs. gen failure) | Actionable insights per run | Low-Medium — post-run analysis |

### Tier 3 — High Impact, Higher Effort (Research Phase)

| # | Improvement | Expected Impact | Effort |
|---|---|---|---|
| 13 | **Adaptive RAG** (complexity routing) | Cost optimization per query | High — classifier + routing logic |
| 14 | **Speculative RAG** (draft + verify) | Lower latency + higher quality | High — multi-draft generation + scoring |
| 15 | **Graph RAG** | Handle global/holistic questions | High — knowledge graph construction |
| 16 | **ColBERT / late interaction** | Better retrieval quality | Medium — RAGatouille integration |
| 17 | **Multi-agent MoA** | Match frontier model quality with SLMs | High — multiple models + aggregation |
| 18 | **Multi-machine parallelism** | Scale benchmark runs | Medium-High — distributed execution |

### Recommended Implementation Order

```
Phase 1 (Quick wins):     2 → 3 → 6 → 9
Phase 2 (Retrieval):      1 → 4 → 8 → 11
Phase 3 (Agentic):        7 → 10 → 5 → 12
Phase 4 (Advanced):       13 → 14 → 15 → 16 → 17
```

---

## Appendix: Key References

| Paper/Resource | Year | Key Finding |
|---|---|---|
| Anthropic — Contextual Retrieval | 2024 | -67% retrieval failures with contextual embeddings + BM25 + reranking |
| Self-RAG (Asai et al.) | 2023 | 7B model with reflection tokens outperforms ChatGPT on QA |
| Corrective RAG (Yan et al.) | 2024 | Plug-and-play retrieval correction, significant improvement on 4 datasets |
| Adaptive RAG (Jeong et al.) | ICLR 2024 | Dynamic query routing by complexity improves cost/quality balance |
| Speculative RAG (DeepMind) | 2024 | Small draft model + large verifier = lower latency + higher quality |
| GraphRAG (Microsoft) | 2024 | Handles global/holistic questions that chunk-based RAG cannot |
| Anthropic — Building Effective Agents | 2024 | Start simple, add complexity only when needed |
| ColBERT (Khattab & Zaharia) | 2020 | Late interaction outperforms bi-encoders, approaches cross-encoder quality |
| Mixture-of-Agents (Together AI) | 2024 | Ensemble of SLMs matches frontier model quality at 1/10th cost |
