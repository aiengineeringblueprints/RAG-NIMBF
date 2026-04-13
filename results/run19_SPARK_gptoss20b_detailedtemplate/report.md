# RAG Benchmark Report

**Date:** 20260413_110725
**Dataset:** t2-ragbench/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.854 | 51.7 | 0.0 | 4193.9 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.568 | 0.446 | 0.277 | 0.432 | 0.400 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_gpt-oss:20b_detailed
-   Processed 50 questions across 312 chunks
-   TTFT: 0.854s (std 0.777s)
-   Throughput: 51.7 tok/s (std 5.1)
-   Faithfulness: 0.568
-   Answer Relevancy: 0.446
-   Context Precision: 0.432
-   Context Recall: 0.400
-   Total time: 4193.9s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.854s +/- 0.777s (range: 0.523-6.199)
- Throughput: 51.7 tok/s +/- 5.1
- Faithfulness: 0.568 +/- 0.445 (range: 0.000-1.000)
- Answer Relevancy: 0.446 +/- 0.470 (range: 0.000-0.987)
- Answer Correctness: 0.277 +/- 0.203 (range: 0.073-0.874)
- Context Precision: 0.432 +/- 0.385 (range: 0.000-1.000)
- Context Recall: 0.400 +/- 0.495 (range: 0.000-1.000)
