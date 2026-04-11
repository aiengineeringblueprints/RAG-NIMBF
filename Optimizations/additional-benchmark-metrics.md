# Additional Benchmark Metrics

## Current Metrics

The framework currently tracks three categories of metrics:

### RAGAS Quality Metrics

| Metric               | What it measures                                  |
| -------------------- | ------------------------------------------------- |
| **Faithfulness**     | How well the answer is grounded in retrieved context |
| **Answer Relevancy** | Relevance of the answer to the question           |
| **Answer Correctness** | Factual correctness against ground truth         |
| **Context Precision** | Precision of retrieved chunks (relevant vs. irrelevant) |
| **Context Recall**   | Recall of retrieved chunks (coverage of ground truth) |

### Performance Metrics

| Metric          | What it measures            |
| --------------- | --------------------------- |
| **TTFT**        | Time to first token         |
| **Total Time**  | End-to-end generation time  |
| **Tokens/sec**  | Throughput                  |
| **Token Count** | Output length               |

### Hardware Metrics

| Metric                | What it measures     |
| --------------------- | -------------------- |
| **GPU Utilization**   | GPU usage percentage |
| **GPU Memory Used/Total** | VRAM consumption |

---

## Missing Metrics Worth Considering

### Retrieval Quality

| Metric | Why it matters |
| ------ | -------------- |
| **Mean Reciprocal Rank (MRR)** | Captures how high the first relevant chunk ranks — complementary to context precision which only measures the ordered relevance of all retrieved chunks |
| **Retrieval Latency** | Time for the vector search + optional reranking step — generation time is tracked but retrieval time is not measured separately |
| **Embedding Throughput** | Time to embed the full document corpus — important when comparing embedding models |

### Generation Quality (beyond RAGAS)

| Metric | Why it matters |
| ------ | -------------- |
| **Hallucination Rate** | Percentage of answers containing claims not supported by context — a stricter, binary version of faithfulness |
| **Refusal Rate** | How often the model refuses to answer — relevant if some configs produce overly conservative behavior |
| **Answer Completeness** | Whether the answer covers all aspects of the ground truth — distinct from correctness (a partially correct answer scores differently) |
| **Conciseness / Verbosity Ratio** | Answer length relative to ground truth — helps identify configs that produce unnecessarily long responses |

### Robustness & Consistency

| Metric | Why it matters |
| ------ | -------------- |
| **Consistency (variance across runs)** | Running the same config multiple times with the same data — if faithfulness swings from 0.6 to 0.9 across runs, the metric is unreliable |
| **Error/Failure Rate** | Percentage of samples where generation or retrieval failed entirely — currently not tracked as a metric |

### Cost & Efficiency

| Metric | Why it matters |
| ------ | -------------- |
| **Cost per Answer** | Estimated API cost based on token counts x model pricing — critical for production decisions |
| **Input vs. Output Token Split** | Only output tokens are counted — input tokens (prompt + context) often dominate cost |
| **Context Efficiency** | Ratio of relevant context to total context retrieved — how much of the retrieved context actually contributed to the answer |

### Reranker-Specific (since reranking is supported)

| Metric | Why it matters |
| ------ | -------------- |
| **Reranker Latency** | Time spent in the reranking step alone |
| **Reranker Lift** | Delta in context precision/recall before vs. after reranking — quantifies whether the reranker actually helps |

---

## Priority Recommendations

### High-value, low-effort additions

1. **Retrieval Latency** — already have timing infrastructure, just not instrumenting the retrieval step separately
2. **Input Token Count** — trivially available from the LLM response metadata
3. **Error/Failure Rate** — just counting exceptions that likely already occur
4. **Cost per Answer** — a lookup table against model pricing

### High-value, moderate-effort

5. **Consistency/Variance across runs** — requires running each config multiple times
6. **Reranker Lift** — compare metrics with/without reranking (the data pipeline already exists)

### Worth considering if expanding beyond FinQA

7. **Hallucination Rate** — especially for open-domain datasets
8. **Conciseness** — useful when comparing prompt templates
