# RAG Benchmark Report

**Date:** 20260409_224650
**Dataset:** FinQA (50 samples)
**Configurations:** 4

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 12.656 | 62.1 | 0.0 | 3156.6 |
| recursive_cs1000_co50_nomic... | 13.962 | 62.0 | 0.0 | 3228.6 |
| recursive_cs500_co100_nomic... | 12.180 | 62.6 | 0.0 | 3218.4 |
| recursive_cs500_co50_nomic-... | 14.161 | 62.6 | 0.0 | 3315.0 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.582 | 0.619 | 0.422 | 0.400 |
| recursive_cs1000_co50_nomic... | 0.582 | 0.669 | 0.378 | 0.400 |
| recursive_cs500_co100_nomic... | 0.519 | 0.692 | 0.337 | 0.320 |
| recursive_cs500_co50_nomic-... | 0.543 | 0.658 | 0.321 | 0.320 |

## Insights

- Best overall: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (composite score: 0.682)
-   Best at: context_precision, context_recall
- Highest faithfulness: recursive_cs1000_co50_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (0.582)
- Highest answer relevancy: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (0.692)
- Fastest TTFT: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (12.18s) — range: 12.18s to 14.16s (1.2x spread)
- Best throughput: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (62.6 tok/s)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 12.656s +/- 11.377s (range: 5.353-71.245)
- Throughput: 62.1 tok/s +/- 1.1
- Faithfulness: 0.582 +/- 0.421 (range: 0.000-1.000)
- Answer Relevancy: 0.619 +/- 0.450 (range: 0.000-1.000)
- Context Precision: 0.422 +/- 0.412 (range: 0.000-1.000)
- Context Recall: 0.400 +/- 0.495 (range: 0.000-1.000)

### recursive_cs1000_co50_nomic...

- TTFT: 13.962s +/- 11.738s (range: 6.299-58.619)
- Throughput: 62.0 tok/s +/- 1.3
- Faithfulness: 0.582 +/- 0.409 (range: 0.000-1.000)
- Answer Relevancy: 0.669 +/- 0.425 (range: 0.000-1.000)
- Context Precision: 0.378 +/- 0.402 (range: 0.000-1.000)
- Context Recall: 0.400 +/- 0.495 (range: 0.000-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 12.180s +/- 8.356s (range: 5.719-43.877)
- Throughput: 62.6 tok/s +/- 1.1
- Faithfulness: 0.519 +/- 0.428 (range: 0.000-1.000)
- Answer Relevancy: 0.692 +/- 0.417 (range: 0.000-1.000)
- Context Precision: 0.337 +/- 0.400 (range: 0.000-1.000)
- Context Recall: 0.320 +/- 0.471 (range: 0.000-1.000)

### recursive_cs500_co50_nomic-...

- TTFT: 14.161s +/- 12.455s (range: 5.067-59.232)
- Throughput: 62.6 tok/s +/- 1.3
- Faithfulness: 0.543 +/- 0.380 (range: 0.000-1.000)
- Answer Relevancy: 0.658 +/- 0.436 (range: 0.000-1.000)
- Context Precision: 0.321 +/- 0.404 (range: 0.000-1.000)
- Context Recall: 0.320 +/- 0.471 (range: 0.000-1.000)
