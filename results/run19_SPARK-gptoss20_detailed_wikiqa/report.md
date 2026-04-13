# RAG Benchmark Report

**Date:** 20260413_144916
**Dataset:** ragas-wikiqa/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.851 | 48.0 | 0.0 | 4979.1 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.722 | 0.595 | 0.397 | 0.405 | 0.341 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_gpt-oss:20b_detailed
-   Processed 50 questions across 580 chunks
-   TTFT: 0.851s (std 1.011s)
-   Throughput: 48.0 tok/s (std 6.4)
-   Faithfulness: 0.722
-   Answer Relevancy: 0.595
-   Context Precision: 0.405
-   Context Recall: 0.341
-   Total time: 4979.1s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.851s +/- 1.011s (range: 0.545-7.842)
- Throughput: 48.0 tok/s +/- 6.4
- Faithfulness: 0.722 +/- 0.340 (range: 0.000-1.000)
- Answer Relevancy: 0.595 +/- 0.295 (range: 0.000-0.957)
- Answer Correctness: 0.397 +/- 0.204 (range: 0.124-0.967)
- Context Precision: 0.405 +/- 0.423 (range: 0.000-1.000)
- Context Recall: 0.341 +/- 0.480 (range: 0.000-1.000)
