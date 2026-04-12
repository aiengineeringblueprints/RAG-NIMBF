# RAG Benchmark Report

**Date:** 20260411_160741
**Dataset:** FinQA (1 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 5.023 | 42.8 | 0.0 | 47.1 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 1.000 | 0.431 | 1.000 | 1.000 | 1.000 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise
-   Processed 1 questions across 6 chunks
-   TTFT: 5.023s (std 0.000s)
-   Throughput: 42.8 tok/s (std 0.0)
-   Faithfulness: 1.000
-   Answer Relevancy: 0.431
-   Context Precision: 1.000
-   Context Recall: 1.000
-   Total time: 47.1s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 5.023s +/- 0.000s (range: 5.023-5.023)
- Throughput: 42.8 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.431 +/- 0.000 (range: 0.431-0.431)
- Answer Correctness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)
