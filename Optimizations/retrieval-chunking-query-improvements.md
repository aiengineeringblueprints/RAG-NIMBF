# Pipeline Score Improvements: Retrieval, Chunking & Query Expansion

**Date:** 2026-04-14
**Baseline run:** `run19_sigurd_qwen3.5-397B-cs500_co100_concise` (WikiQA, 100 samples)
**Status:** Implemented

---

## Baseline Scores

| Metric | Score |
|---|---|
| Faithfulness | 0.888 |
| Answer Relevancy | 0.525 |
| Answer Correctness | 0.374 |
| Context Precision | 0.441 |
| Context Recall | 0.430 |

**Core problem:** Retrieval quality (precision + recall ~44%) cascades into poor answer correctness (37%). Context recall median is 0.0 — half the samples receive zero relevant context.

---

## Root Cause Analysis

The pipeline uses plain `similarity_search` with `nomic-embed-text` on heterogeneous chunks. Many retrieved documents are irrelevant to the question. The QA logs show frequent "I cannot answer from the provided context" responses where the retrieved chunks discuss unrelated topics (e.g., retrieving T-SQL and Saint Patrick's Day for a question about elephants).

---

## Implemented Improvements

### 1. MMR Retrieval (`RETRIEVAL_STRATEGY=mmr`)

Maximal Marginal Relevance diversifies retrieved documents by balancing relevance with novelty. Instead of returning the k most similar documents (which are often redundant), MMR fetches a larger candidate set then selects results that maximize both relevance to the query and dissimilarity to already-selected results.

**Config options:**
```env
RETRIEVAL_STRATEGY=mmr          # similarity | mmr
RETRIEVAL_FETCH_K=20            # candidates to fetch before filtering
RETRIEVAL_MMR_LAMBDA=0.5        # 0 = max diversity, 1 = max relevance
```

**Why it helps:** When the top-8 results are all fragments of the same passage, MMR forces the retriever to include diverse passages, increasing the chance that at least one contains the answer.

**Implementation:** `benchmark/retrieval.py` — `retrieve()` routes to Chroma's built-in `max_marginal_relevance_search()` when strategy is `"mmr"`.

---

### 2. Semantic Chunking (`CHUNKING_STRATEGIES=semantic`)

Standard fixed-size chunking (recursive/character) splits at arbitrary character boundaries, often breaking mid-sentence. Semantic chunking uses embeddings to detect natural topic shifts and splits there instead, producing chunks that contain coherent, topically unified passages.

**Config options:**
```env
CHUNKING_STRATEGIES=semantic
SEMANTIC_BREAKPOINT_TYPE=percentile       # percentile | standard_deviation | interquartile
SEMANTIC_BREAKPOINT_AMOUNT=95             # threshold (higher = fewer/larger chunks)
```

**Why it helps:** A chunk that contains a complete thought is more likely to match a question about that topic than a fragment that cuts off mid-sentence. The `breakpoint_threshold_amount` controls sensitivity — lower values create more/smaller chunks (higher granularity), higher values create fewer/larger chunks.

**Implementation:** `benchmark/chunking.py` — `get_chunker()` lazy-imports `SemanticChunker` from `langchain_experimental` when strategy is `"semantic"`. Requires `langchain_experimental>=0.3`.

**Note:** Semantic chunking ignores `CHUNK_SIZE` and `CHUNK_OVERLAP` (boundaries are determined by embeddings, not character count). These values are still required in the config for the Cartesian product but have no effect.

---

### 3. HyDE Query Expansion (`RETRIEVAL_USE_HYDE=true`)

Hypothetical Document Embedding (HyDE) uses the generator LLM to produce a hypothetical answer to the question, then embeds that answer instead of the raw question for retrieval. Since the hypothetical answer is semantically closer to actual document content than a short question, retrieval quality improves.

**Config option:**
```env
RETRIEVAL_USE_HYDE=true
```

**Why it helps:** A query like "what circuit court is maryland" is terse and may not embed close to relevant chunks. A hypothetical answer like "Maryland's Circuit Courts are the state trial courts of general jurisdiction, organized into eight judicial circuits..." is much more semantically similar to the actual Wikipedia passage, improving retrieval.

**Implementation:** `benchmark/retrieval.py` — `expand_query_with_hyde()` invokes the generator LLM (same model used for answering) with a simple prompt to generate a factual answer. Falls back to the original question if the LLM fails or returns empty.

**Tradeoff:** Adds one LLM call per question (increases total runtime). The hypothetical answer is only used for retrieval — the original question is still used for the final answer generation.

---

## Combining Features

All three features can be combined. A recommended configuration for improving retrieval:

```env
CHUNKING_STRATEGIES=semantic
SEMANTIC_BREAKPOINT_TYPE=percentile
SEMANTIC_BREAKPOINT_AMOUNT=85

RETRIEVAL_STRATEGY=mmr
RETRIEVAL_FETCH_K=20
RETRIEVAL_MMR_LAMBDA=0.5

RETRIEVAL_USE_HYDE=true

RERANKER_MODELS=huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_K=3
RETRIEVAL_TOP_K=8
```

This gives: semantic chunks → HyDE-expanded query → MMR retrieval of 8 from 20 candidates → reranker selects top 3.

---

## Expected Impact

| Change | Metric Affected | Expected Improvement |
|---|---|---|
| MMR retrieval | Context precision | +10-15% (less redundant retrieval) |
| Semantic chunking | Context recall | +5-10% (coherent chunks match queries better) |
| HyDE expansion | Context recall | +5-15% (better query-document alignment) |
| All combined | Answer correctness | +10-20% (better context → better answers) |

---

## Other Improvements Identified (Not Implemented)

These were identified in the analysis but not implemented as part of this change:

1. **Upgrade embedding model** — `nomic-embed-text` is small; `bge-large-en-v1.5` or `e5-large-v2` are significantly stronger
2. **Upgrade reranker** — `ms-marco-MiniLM-L-6-v2` is the smallest cross-encoder; L-12 or `mxbai-rerank-large-v1` perform better
3. **Reduce `_looks_like_thinking` aggressiveness** — drops valid answers starting with "So,", "Well,", etc.
4. **Wire in custom metrics** — `custom_metrics.py` has ROUGE-L, BLEU, METEOR, BERTScore but is not connected to the pipeline
5. **Use stronger critic model** — `gemma3:12b` is decent but larger models give more accurate RAGAS scores
6. **Template-dataset alignment** — WikiQA ground truth is short/factual → use `concise` template, not `detailed`
7. **Context compression** — With 8 retrieved chunks, the context window may overwhelm the model; compress or summarize before generation
