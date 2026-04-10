# RAG Benchmark Report

**Date:** 20260410_094120
**Dataset:** FinQA (1 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 4.168 | 51.6 | 0.0 | 36.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 1.000 | 0.431 | 1.000 | 1.000 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise
-   Processed 1 questions across 6 chunks
-   TTFT: 4.168s (std 0.000s)
-   Throughput: 51.6 tok/s (std 0.0)
-   Faithfulness: 1.000
-   Answer Relevancy: 0.431
-   Context Precision: 1.000
-   Context Recall: 1.000
-   Total time: 36.3s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 4.168s +/- 0.000s (range: 4.168-4.168)
- Throughput: 51.6 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.431 +/- 0.000 (range: 0.431-0.431)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)
