# RAG Benchmark Report

**Date:** 20260411_170542
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.529 | 11.5 | 1.7 | 2840.0 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.200 | 0.507 | 0.355 | 0.421 | 0.380 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise
-   Processed 50 questions across 312 chunks
-   TTFT: 0.529s (std 0.135s)
-   Throughput: 11.5 tok/s (std 5.4)
-   Faithfulness: 0.200
-   Answer Relevancy: 0.507
-   Context Precision: 0.421
-   Context Recall: 0.380
-   Total time: 2840.0s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.529s +/- 0.135s (range: 0.186-0.980)
- Throughput: 11.5 tok/s +/- 5.4
- Faithfulness: 0.200 +/- 0.391 (range: 0.000-1.000)
- Answer Relevancy: 0.507 +/- 0.088 (range: 0.000-0.672)
- Answer Correctness: 0.355 +/- 0.354 (range: 0.095-1.000)
- Context Precision: 0.421 +/- 0.411 (range: 0.000-1.000)
- Context Recall: 0.380 +/- 0.490 (range: 0.000-1.000)
