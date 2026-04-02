# RAG Benchmark Report

**Date:** 20260402_104806
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 2.139 | 15.2 | 46.4 | 5785.5 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.598 | 0.708 | 0.426 | 0.543 |

## Insights

- Single configuration: recursive_cs1000_co200_nomic-embed-text:latest_gemma3:4b
-   Processed 50 questions across 320 chunks
-   TTFT: 2.139s (std 0.920s)
-   Throughput: 15.2 tok/s (std 4.9)
-   Faithfulness: 0.598
-   Answer Relevancy: 0.708
-   Context Precision: 0.426
-   Context Recall: 0.543
-   Total time: 5785.5s

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 2.139s +/- 0.920s (range: 0.957-5.112)
- Throughput: 15.2 tok/s +/- 4.9
- Faithfulness: 0.598 +/- 0.453 (range: 0.000-1.000)
- Answer Relevancy: 0.708 +/- 0.227 (range: 0.000-0.996)
- Context Precision: 0.426 +/- 0.382 (range: 0.000-1.000)
- Context Recall: 0.543 +/- 0.504 (range: 0.000-1.000)
