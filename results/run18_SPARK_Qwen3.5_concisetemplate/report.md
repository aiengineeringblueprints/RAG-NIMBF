# RAG Benchmark Report

**Date:** 20260412_140224
**Dataset:** t2-ragbench/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 1.316 | 5.1 | 0.1 | 2890.5 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.160 | 0.521 | 0.404 | 0.432 | 0.360 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_qwen3.5:35b_concise
-   Processed 50 questions across 312 chunks
-   TTFT: 1.316s (std 1.078s)
-   Throughput: 5.1 tok/s (std 2.6)
-   Faithfulness: 0.160
-   Answer Relevancy: 0.521
-   Context Precision: 0.432
-   Context Recall: 0.360
-   Total time: 2890.5s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 1.316s +/- 1.078s (range: 0.841-8.706)
- Throughput: 5.1 tok/s +/- 2.6
- Faithfulness: 0.160 +/- 0.370 (range: 0.000-1.000)
- Answer Relevancy: 0.521 +/- 0.041 (range: 0.432-0.606)
- Answer Correctness: 0.404 +/- 0.372 (range: 0.117-1.000)
- Context Precision: 0.432 +/- 0.403 (range: 0.000-1.000)
- Context Recall: 0.360 +/- 0.485 (range: 0.000-1.000)
