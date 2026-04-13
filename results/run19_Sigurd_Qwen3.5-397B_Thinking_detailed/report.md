# RAG Benchmark Report

**Date:** 20260413_161758
**Dataset:** ragas-wikiqa/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.752 | 0.0 | 0.0 | 5161.5 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.955 | 0.564 | 0.345 | 0.431 | 0.340 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed
-   Processed 50 questions across 580 chunks
-   TTFT: 0.752s (std 0.541s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.955
-   Answer Relevancy: 0.564
-   Context Precision: 0.431
-   Context Recall: 0.340
-   Total time: 5161.5s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.752s +/- 0.541s (range: 0.486-3.303)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.955 +/- 0.119 (range: 0.286-1.000)
- Answer Relevancy: 0.564 +/- 0.318 (range: 0.000-0.975)
- Answer Correctness: 0.345 +/- 0.177 (range: 0.131-0.954)
- Context Precision: 0.431 +/- 0.431 (range: 0.000-1.000)
- Context Recall: 0.340 +/- 0.479 (range: 0.000-1.000)
