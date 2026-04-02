# RAG Evaluation Alternatives

Alternatives to RAGAS for evaluating RAG (Retrieval-Augmented Generation) systems.

## LLM-as-Judge Frameworks

| Framework | Key Idea | Metrics |
|-----------|----------|---------|
| **DeepEval** | End-to-end eval with built-in + custom metrics | Faithfulness, Answer Relevancy, Hallucination, GEval (custom criteria) |
| **TruLens** | "Triad" of groundedness, answer relevance, context relevance | Also tracks latency, cost, token usage out of the box |
| **LLamaIndex Evaluation** | Native to LLamaIndex RAG pipelines | Faithfulness, Relevancy, Correctness, Guideline evaluation |
| **UpTrain** | Open-source + managed, checks for 20+ pre-built issues | Factual accuracy, response completeness, context relevance, tone |

## Traditional / Reference-Based (No LLM judge)

| Approach | Key Idea |
|----------|----------|
| **BLEU / ROUGE / METEOR** | N-gram overlap between generated answer and ground truth |
| **BERTScore** | Contextual embedding similarity (cosine) instead of n-grams |
| **Semantic Similarity (e.g., Cross-Encoder)** | Trained reranker scores answer vs. reference |

## Retrieval-Only Evaluation

| Approach | Key Idea |
|----------|----------|
| **MRR / Hit Rate / nDCG** | Classic IR metrics on retrieved chunks vs. ground truth |
| **BEIR Benchmark** | Standardized benchmark suite for retrieval |

## Trade-offs for Local Models (Ollama)

1. **DeepEval** — Most direct alternative to RAGAS. Similar metrics (faithfulness, relevancy), also uses an LLM-as-judge. Has a nicer dashboard. Works with any LangChain model, so it would plug into the existing `get_chat_model` setup.

2. **TruLens** — Best if you want **latency and cost tracking** baked in (already tracked manually in this project). Its "triad" maps closely to RAGAS's faithfulness/relevancy/precision.

3. **BLEU/ROUGE/BERTScore** — **No LLM judge needed**, just string/embedding comparison. Much faster, fully deterministic, works great as a baseline. But can't catch semantic errors like hallucination.

4. **Hybrid approach** — Use cheap n-gram/BERTScore metrics for every run, and RAGAS/DeepEval LLM-judge metrics only for spot-checks or final comparisons. Most practical for local models where LLM-based eval is slow.
