# RAG Benchmark Report

**Date:** 20260409_143652
**Dataset:** FinQA (1 samples)
**Configurations:** 9

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 6.719 | 58.9 | 0.0 | 83.5 |
| recursive_cs1000_co50_nomic... | 8.430 | 60.9 | 0.0 | 95.2 |
| recursive_cs1000_co25_nomic... | 7.347 | 60.8 | 0.0 | 74.0 |
| recursive_cs500_co100_nomic... | 6.051 | 58.7 | 0.0 | 87.1 |
| recursive_cs500_co50_nomic-... | 5.954 | 59.6 | 0.0 | 95.3 |
| recursive_cs500_co25_nomic-... | 5.913 | 60.0 | 0.0 | 55.0 |
| recursive_cs200_co100_nomic... | 5.354 | 60.0 | 0.0 | 90.5 |
| recursive_cs200_co50_nomic-... | 4.567 | 58.9 | 0.0 | 82.0 |
| recursive_cs200_co25_nomic-... | 4.410 | 57.6 | 0.0 | 52.7 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.500 | 0.969 | 1.000 | 1.000 |
| recursive_cs1000_co50_nomic... | 1.000 | 0.969 | 1.000 | 1.000 |
| recursive_cs1000_co25_nomic... | 1.000 | 0.969 | 1.000 | 1.000 |
| recursive_cs500_co100_nomic... | 1.000 | 0.969 | 1.000 | 1.000 |
| recursive_cs500_co50_nomic-... | 1.000 | 0.969 | 1.000 | 1.000 |
| recursive_cs500_co25_nomic-... | 1.000 | 0.969 | 1.000 | 1.000 |
| recursive_cs200_co100_nomic... | 0.000 | 0.969 | 0.500 | 1.000 |
| recursive_cs200_co50_nomic-... | 0.000 | 0.969 | 0.500 | 1.000 |
| recursive_cs200_co25_nomic-... | 0.000 | 0.969 | 0.500 | 1.000 |

## Insights

- Best overall: recursive_cs500_co25_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (composite score: 0.538)
- Highest faithfulness: recursive_cs1000_co50_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (1.000)
- Highest answer relevancy: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (0.969)
- Fastest TTFT: recursive_cs200_co25_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (4.41s) — range: 4.41s to 8.43s (1.9x spread)
- Best throughput: recursive_cs1000_co50_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (60.9 tok/s)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 6.719s +/- 0.000s (range: 6.719-6.719)
- Throughput: 58.9 tok/s +/- 0.0
- Faithfulness: 0.500 +/- 0.000 (range: 0.500-0.500)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs1000_co50_nomic...

- TTFT: 8.430s +/- 0.000s (range: 8.430-8.430)
- Throughput: 60.9 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs1000_co25_nomic...

- TTFT: 7.347s +/- 0.000s (range: 7.347-7.347)
- Throughput: 60.8 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 6.051s +/- 0.000s (range: 6.051-6.051)
- Throughput: 58.7 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs500_co50_nomic-...

- TTFT: 5.954s +/- 0.000s (range: 5.954-5.954)
- Throughput: 59.6 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs500_co25_nomic-...

- TTFT: 5.913s +/- 0.000s (range: 5.913-5.913)
- Throughput: 60.0 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs200_co100_nomic...

- TTFT: 5.354s +/- 0.000s (range: 5.354-5.354)
- Throughput: 60.0 tok/s +/- 0.0
- Faithfulness: 0.000 +/- 0.000 (range: 0.000-0.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 0.500 +/- 0.000 (range: 0.500-0.500)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs200_co50_nomic-...

- TTFT: 4.567s +/- 0.000s (range: 4.567-4.567)
- Throughput: 58.9 tok/s +/- 0.0
- Faithfulness: 0.000 +/- 0.000 (range: 0.000-0.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 0.500 +/- 0.000 (range: 0.500-0.500)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs200_co25_nomic-...

- TTFT: 4.410s +/- 0.000s (range: 4.410-4.410)
- Throughput: 57.6 tok/s +/- 0.0
- Faithfulness: 0.000 +/- 0.000 (range: 0.000-0.000)
- Answer Relevancy: 0.969 +/- 0.000 (range: 0.969-0.969)
- Context Precision: 0.500 +/- 0.000 (range: 0.500-0.500)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)
