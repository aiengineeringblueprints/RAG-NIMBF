# RAG Benchmark Report

**Date:** 20260409_104548
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 1.954 | 16.4 | 46.8 | 2326.2 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.490 | 0.468 | 0.410 | 0.420 |

## Insights

- Single configuration: recursive_cs1000_co200_nomic-embed-text:latest_gemma3:4b
-   Processed 50 questions across 320 chunks
-   TTFT: 1.954s (std 0.779s)
-   Throughput: 16.4 tok/s (std 5.1)
-   Faithfulness: 0.490
-   Answer Relevancy: 0.468
-   Context Precision: 0.410
-   Context Recall: 0.420
-   Total time: 2326.2s

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 1.954s +/- 0.779s (range: 0.928-4.639)
- Throughput: 16.4 tok/s +/- 5.1
- Faithfulness: 0.490 +/- 0.467 (range: 0.000-1.000)
- Answer Relevancy: 0.468 +/- 0.328 (range: 0.000-0.931)
- Context Precision: 0.410 +/- 0.387 (range: 0.000-1.000)
- Context Recall: 0.420 +/- 0.499 (range: 0.000-1.000)
