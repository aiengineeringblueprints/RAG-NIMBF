# Feature Improvements

## 1. Retrieval Enhancements

### 1a. Hybrid Search (BM25 + Dense)
Currently only dense retrieval (similarity/MMR). BM25 sparse retrieval + dense fusion is standard in production RAG.

**Implementation:** Use `rank_bm25` library or LangChain's `BM25Retriever`. Fuse with Reciprocal Rank Fusion (RRF).

### 1b. Parent-Child / Small-to-Big Retrieval
Retrieve small chunks for precision, return parent chunks for context. LangChain supports this via `ParentDocumentRetriever`.

**Implementation:** Add `retrieval_mode: "parent_child"` config option. Store parent-child chunk mapping in vector store metadata.

### 1c. Contextual Retrieval
Anthropic's contextual retrieval prepends chunk-specific context to each chunk before embedding. Improves retrieval accuracy significantly.

**Implementation:** Add `retrieval_contextualize: true` option. Use a small LLM to generate chunk context before embedding.

### 1d. Query Transformation Beyond HyDE
Currently only HyDE. Add:
- **Multi-query:** Generate multiple query variations, retrieve for each, merge results
- **Step-back prompting:** Generate abstract version of query for broader retrieval
- **Query decomposition:** Break complex queries into sub-queries

### 1e. More Vector Store Backends
Only ChromaDB and LanceDB. Add:
- **Qdrant** — production-grade, filtering support
- **Weaviate** — hybrid search built-in
- **FAISS** — fastest for pure similarity search
- **Milvus** — distributed scale

## 2. Evaluation Improvements

### 2a. Enable All RAGAS Metrics
`benchmark/evaluation.py` — only 3 of 6 RAGAS metrics enabled by default (faithfulness, context_recall, semantic_similarity). Missing:
- `answer_relevancy` — how relevant the answer is to the question
- `answer_correctness` — factual correctness vs ground truth
- `context_precision` — whether relevant chunks rank higher

**Fix:** Enable all 6 by default. Add config flag to selectively disable.

### 2b. LLM-as-Judge Evaluation
Beyond RAGAS metrics, add direct LLM-based evaluation:
- G-Eval style scoring
- Pairwise comparison (which answer is better?)
- Criteria-based scoring (accuracy, completeness, conciseness)

### 2c. RAGChecker Metrics
RAGChecker (2024) provides fine-grained claim-level evaluation:
- **Claim recall/precision** — per-claim attribution
- **Correctness vs faithfulness** decomposition
- **Noise robustness** — how well system handles irrelevant context

### 2d. Human Evaluation Interface
No mechanism for human-in-the-loop evaluation. Add:
- Side-by-side answer comparison UI
- Human preference logging
- Inter-annotator agreement tracking

## 3. Generation Improvements

### 3a. Temperature / Sampling Control
`benchmark/generation.py` — no temperature, top_p, or repetition_penalty config. All generation is deterministic (temperature=0 implied).

**Fix:** Add `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_REPETITION_PENALTY` env vars.

### 3b. Streaming Metrics
TTFT (time-to-first-token) is tracked but not leveraged. Add:
- Per-token latency distribution
- Time-per-output-token (TPOT) separate from TPS
- Generation cancel/timeout per sample

### 3c. Batch Generation
Currently generates one answer at a time. For API-based providers, batch generation reduces cost and latency.

## 4. Dataset Improvements

### 4a. More Datasets
Only 5 datasets supported (t2-ragbench, ragbench, squad, ragas-wikiqa, ragperf-wikipedia-nq). Add:
- **NaturalQuestions (NQ)** — Google's real-user questions
- **HotpotQA** — multi-hop reasoning
- **MS MARCO** — passage ranking
- **TriviaQA** — trivia with distant supervision
- **Climate-FEVER** — climate fact verification
- **FinanceBench** — financial domain
- **Custom dataset loader** — user-provided CSV/JSON

### 4b. Dataset Versioning
No dataset version tracking. Results may not be reproducible if dataset changes upstream.

**Fix:** Pin dataset versions. Hash dataset contents. Warn on version mismatch.

### 4c. Domain-Specific Subsets
Squad is general-purpose. Add domain-specific evaluation:
- Legal, medical, financial, technical domains
- Multi-lingual evaluation
- Long-document evaluation

## 5. Reporting Improvements

### 5a. Interactive Dashboard
Current output is static (JSON, CSV, Markdown, PNG plots). Add:
- Streamlit or Gradio dashboard
- Interactive plotly charts (already imported but not used interactively)
- Run comparison view

### 5b. Statistical Significance Testing
No significance testing between runs. Add:
- Paired t-test or bootstrap confidence intervals
- Effect size reporting (Cohen's d)
- Multiple comparison correction (Bonferroni)

### 5c. A/B Test Framework
No framework for structured A/B testing. Add:
- Baseline vs treatment comparison
- Automatic significance detection
- Metric regression detection

## 6. Experiment Tracking Improvements

### 6a. Beyond MLflow
MLflow is the only tracker. Add:
- **Weights and Biases** integration
- **Neptune.ai** integration
- **TensorBoard** for training-like tracking
- Abstract `Tracker` interface for any backend

### 6b. Experiment Comparison
`compare_runs.py` exists but basic. Add:
- Metric diff with significance
- Config diff highlighting
- Regression/improvement flagging
