# RAG Benchmark Report

**Date:** 20260413_095320
**Dataset:** t2-ragbench/FinQA (50 samples)
**Dataset:** FinQA (50 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 4.444 | 42.7 | 0.0 | 4745.8 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.810 | 0.513 | 0.229 | 0.407 | 0.360 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_qwen3.5:35b_detailed
-   Processed 50 questions across 312 chunks
-   TTFT: 4.444s (std 1.600s)
-   Throughput: 42.7 tok/s (std 6.0)
-   Faithfulness: 0.810
-   Answer Relevancy: 0.513
-   Context Precision: 0.407
-   Context Recall: 0.360
-   Total time: 4745.8s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 4.444s +/- 1.600s (range: 2.509-11.018)
- Throughput: 42.7 tok/s +/- 6.0
- Faithfulness: 0.810 +/- 0.314 (range: 0.000-1.000)
- Answer Relevancy: 0.513 +/- 0.479 (range: 0.000-1.000)
- Answer Correctness: 0.229 +/- 0.156 (range: 0.071-0.843)
- Context Precision: 0.407 +/- 0.403 (range: 0.000-1.000)
- Context Recall: 0.360 +/- 0.485 (range: 0.000-1.000)
