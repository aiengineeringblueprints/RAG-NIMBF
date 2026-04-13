# RAG Benchmark Report

**Date:** 20260413_214454
**Dataset:** ragas-wikiqa/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 0.747 | 0.0 | 0.0 | 4746.7 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs500_co100_nomic... | 0.931 | 0.564 | 0.338 | 0.402 | 0.360 |

## Insights

- Single configuration: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed
-   Processed 50 questions across 1162 chunks
-   TTFT: 0.747s (std 0.648s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.931
-   Answer Relevancy: 0.564
-   Context Precision: 0.402
-   Context Recall: 0.360
-   Total time: 4746.7s

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 0.747s +/- 0.648s (range: 0.358-2.848)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.931 +/- 0.163 (range: 0.000-1.000)
- Answer Relevancy: 0.564 +/- 0.319 (range: 0.000-0.975)
- Answer Correctness: 0.338 +/- 0.170 (range: 0.130-0.834)
- Context Precision: 0.402 +/- 0.401 (range: 0.000-1.000)
- Context Recall: 0.360 +/- 0.485 (range: 0.000-1.000)
