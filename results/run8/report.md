# RAG Benchmark Report

**Date:** 20260411_153217
**Dataset:** FinQA (150 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 15.543 | 61.6 | 0.0 | 12347.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.514 | 0.332 | 0.367 | 0.399 | 0.353 |

## Insights

- Single configuration: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise
-   Processed 150 questions across 928 chunks
-   TTFT: 15.543s (std 10.517s)
-   Throughput: 61.6 tok/s (std 1.9)
-   Faithfulness: 0.514
-   Answer Relevancy: 0.332
-   Context Precision: 0.399
-   Context Recall: 0.353
-   Total time: 12347.3s

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 15.543s +/- 10.517s (range: 2.742-32.609)
- Throughput: 61.6 tok/s +/- 1.9
- Faithfulness: 0.514 +/- 0.468 (range: 0.000-1.000)
- Answer Relevancy: 0.332 +/- 0.286 (range: 0.000-0.997)
- Answer Correctness: 0.367 +/- 0.362 (range: 0.075-1.000)
- Context Precision: 0.399 +/- 0.427 (range: 0.000-1.000)
- Context Recall: 0.353 +/- 0.480 (range: 0.000-1.000)
