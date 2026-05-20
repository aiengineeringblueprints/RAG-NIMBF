# Competitive Analysis

## 1. RAG Evaluation Framework Landscape (2025-2026)

### RAGAS (Used by this project)
- **Version:** 0.2+
- **Metrics:** Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall, Semantic Similarity
- **Strengths:** Well-established, LLM-based evaluation, framework-agnostic
- **Weaknesses:** Requires strong critic LLM, noisy with small models, no retrieval-specific IR metrics
- **This project:** Uses RAGAS as core evaluator but only enables 3 of 6 metrics

### TruLens
- **Focus:** RAG triad (groundedness, answer relevance, context relevance)
- **Strengths:** Real-time evaluation, nice dashboard, failure mode analysis
- **Weaknesses:** Less comprehensive than RAGAS for offline eval
- **Gap:** This project has no real-time evaluation or dashboard

### DeepEval
- **Metrics:** Answer relevance, faithfulness, contextual recall, hallucination, bias, toxicity
- **Strengths:** Built-in test cases, pytest integration, dataset management
- **Weaknesses:** Newer, smaller community
- **Gap:** This project has no pytest-style eval integration or toxicity/bias detection

### ARES
- **Focus:** Automated RAG evaluation using fine-tuned LLM judges
- **Strengths:** No human annotation needed, fine-tuned evaluation
- **Weaknesses:** Requires training data, complex setup
- **Gap:** This project uses generic critic models, no fine-tuned judges

### RAGChecker
- **Focus:** Fine-grained claim-level evaluation
- **Metrics:** Claim recall, claim precision, correctness decomposition, noise robustness
- **Strengths:** Most granular evaluation available
- **Weaknesses:** Computationally expensive, complex output
- **Gap:** This project evaluates at answer level, not claim level

### Prometheus 2 / Auto-J
- **Focus:** LLM-as-judge with fine-tuned evaluator models
- **Strengths:** More consistent than prompting general LLMs
- **Weaknesses:** Requires specific model, limited to supported languages
- **Gap:** This project uses general-purpose critic models

## 2. RAG Benchmarking Frameworks

### RAGAS Bench (ragas benchmark suite)
- Standardized benchmark for comparing RAG systems
- This project builds similar functionality from scratch

### LangBench
- Multi-language RAG benchmarking
- Supports 10+ languages
- Gap: This project is English/German only

### CRUD-RAG
- Tests Create, Read, Update, Delete operations in RAG
- Beyond simple QA
- Gap: This project only tests Read (retrieval + generation)

## 3. SOTA Retrieval Methods Not Implemented

| Method | Description | Impact |
|--------|-------------|--------|
| **ColBERT/ColBERTv2** | Late interaction, token-level matching | +5-15% recall vs dense |
| **Hybrid BM25+Dense** | Sparse + dense fusion via RRF | +3-10% recall |
| **Splade** | Sparse learned representations | Competitive with dense, faster |
| **BGE-M3** | Multi-lingual, multi-function (dense+sparse+colbert) | SOTA on MTEB |
| **Contextual Retrieval** | Anthropic chunk context prepending | +5-10% on ambiguous queries |
| **RAPTOR** | Recursive summarization tree | Better for broad questions |
| **GraphRAG** | Knowledge graph + vector retrieval | Better for multi-hop reasoning |

## 4. SOTA Embedding Models Not Tested

| Model | Size | MTEB Rank | Notes |
|-------|------|-----------|-------|
| **bge-m3** | 568M | Top 5 | Multi-lingual, multi-functional |
| **GTE-Qwen2-7B** | 7B | Top 3 | Best open embedding model |
| **E5-mistral-7B** | 7B | Top 5 | Strong general purpose |
| **nomic-embed-text-v2** | 137M | Top 20 | Currently used, lightweight |
| **jina-embeddings-v3** | 572M | Top 10 | Multi-task, LoRA adapter |

This project only tests with `nomic-embed-text`. No embedding model comparison.

## 5. SOTA Reranking Approaches

| Method | Description | Used Here? |
|--------|-------------|------------|
| **Cross-encoder (MiniLM-L-6)** | Pairwise scoring | Yes (only option) |
| **ColBERT reranking** | Token-level late interaction | No |
| **LLM-based reranking** | Use LLM to score relevance | No |
| **RankGPT** | Permutation-based LLM reranking | No |
| **BGE-reranker-v2-m3** | Multi-lingual cross-encoder | No |

## 6. Key Datasets This Project Should Support

| Dataset | Domain | Size | Why |
|---------|--------|------|-----|
| **NaturalQuestions** | General | 300K | Real user queries |
| **HotpotQA** | Multi-hop | 113K | Tests reasoning chains |
| **MS MARCO** | Passage ranking | 8.8M | Industry standard |
| **CRAG** (Meta) | Factual QA, 5 domains | 4.4K | Mock web + KG APIs |
| **RAGBench** | 5 industry domains | 100K | Comprehensive RAG benchmark |
| **T2-RAGBench** | Text + table | — | Multi-format retrieval |
| **BEIR** | 18 datasets, 9 IR tasks | — | Retrieval benchmark suite |

## 7. Critical Benchmark Design Pitfalls (from 2025-2026 research)

### Knowledge Leakage
LLMs answer 31-78% of benchmark questions without any retrieval (SeedRG paper, arXiv 2605.08838). SQuAD is especially vulnerable. **This project has no leakage detection.**

**Fix:** Add no-retrieval baseline. If LLM scores >20% without context, benchmark is contaminated.

### Statistical Rigor Crisis
Only 16% of LLM benchmarks use statistical tests (arXiv 2511.04703). This project is in the 84% majority.

### Construct Validity
81.3% of benchmarks rely on exact matching, which fails for semantically correct but lexically divergent responses. This project uses BERTScore (good) but also ROUGE-L/BLEU (exact-match biased).

## 8. New Tools Worth Evaluating

| Tool | What | Why Relevant |
|------|------|-------------|
| **RAG BenchKit** | Auto grid search + heatmaps | Does what this project does, with better viz |
| **InspectorRAGet** (IBM) | Aggregate + instance analysis | Human + algorithmic metrics, multi-turn |
| **RAGVue** | 18 reference-free metrics | Side-by-side comparison, calibration metrics |
| **RAGLint** | Drift detection, UMAP viz | CI/CD native |
| **EvalScope** (Alibaba) | Web dashboard, RAGEval backend | W&B/SwanLab export |

## 7. What This Project Does Well (vs Competitors)

1. **Local-first:** Works with Ollama, no cloud dependency
2. **Multi-metric:** Combines RAGAS + custom IR/NLG metrics
3. **Comprehensive pipeline:** Full RAG pipeline benchmarking
4. **Agentic mode:** LangGraph-based autonomous exploration (unique)
5. **Multiple providers:** Ollama + OpenAI-compatible
6. **Per-sample detail:** QA logs, timing, GPU metrics per sample
7. **Grid search:** Combinatorial parameter exploration built-in
