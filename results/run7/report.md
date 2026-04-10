# RAG Benchmark Report

**Date:** 20260410_095924
**Dataset:** FinQA (1 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 3.769 | 57.0 | 0.0 | 70.9 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 1.000 | 0.432 | 1.000 | 1.000 | 1.000 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise
-   Processed 1 questions across 6 chunks
-   TTFT: 3.769s (std 0.000s)
-   Throughput: 57.0 tok/s (std 0.0)
-   Faithfulness: 1.000
-   Answer Relevancy: 0.432
-   Context Precision: 1.000
-   Context Recall: 1.000
-   Total time: 70.9s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 3.769s +/- 0.000s (range: 3.769-3.769)
- Throughput: 57.0 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.432 +/- 0.000 (range: 0.432-0.432)
- Answer Correctness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)
