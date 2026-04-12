# RAG Benchmark Report

**Date:** 20260412_090516
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 9.471 | 51.2 | 0.0 | 3416.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.265 | 0.501 | 0.441 | 0.429 | 0.400 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_gpt-oss:20b_concise
-   Processed 50 questions across 312 chunks
-   TTFT: 9.471s (std 8.250s)
-   Throughput: 51.2 tok/s (std 7.9)
-   Faithfulness: 0.265
-   Answer Relevancy: 0.501
-   Context Precision: 0.429
-   Context Recall: 0.400
-   Total time: 3416.3s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 9.471s +/- 8.250s (range: 1.583-35.783)
- Throughput: 51.2 tok/s +/- 7.9
- Faithfulness: 0.265 +/- 0.446 (range: 0.000-1.000)
- Answer Relevancy: 0.501 +/- 0.084 (range: 0.000-0.628)
- Answer Correctness: 0.441 +/- 0.388 (range: 0.100-1.000)
- Context Precision: 0.429 +/- 0.407 (range: 0.000-1.000)
- Context Recall: 0.400 +/- 0.495 (range: 0.000-1.000)
