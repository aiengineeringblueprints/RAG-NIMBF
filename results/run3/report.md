# RAG Benchmark Report

**Date:** 20260409_140652
**Dataset:** FinQA (1 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 5.598 | 2.1 | 49.0 | 83.7 |
| recursive_cs1000_co200_nomi... | 4.733 | 54.1 | 0.0 | 112.8 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.333 | 0.476 | 1.000 | 1.000 |
| recursive_cs1000_co200_nomi... | 1.000 | 0.000 | 1.000 | 1.000 |

## Insights

- Best overall: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (composite score: 0.400)
-   Best at: faithfulness, ttft, tok_per_s
- Highest faithfulness: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (1.000)
- Highest answer relevancy: recursive_cs1000_co200_nomic-embed-text:latest_gemma3:4b (0.476)
- Fastest TTFT: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (4.73s) — range: 4.73s to 5.60s (1.2x spread)
- Best throughput: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ (54.1 tok/s)

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 5.598s +/- 0.000s (range: 5.598-5.598)
- Throughput: 2.1 tok/s +/- 0.0
- Faithfulness: 0.333 +/- 0.000 (range: 0.333-0.333)
- Answer Relevancy: 0.476 +/- 0.000 (range: 0.476-0.476)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 4.733s +/- 0.000s (range: 4.733-4.733)
- Throughput: 54.1 tok/s +/- 0.0
- Faithfulness: 1.000 +/- 0.000 (range: 1.000-1.000)
- Answer Relevancy: 0.000 +/- 0.000 (range: 0.000-0.000)
- Context Precision: 1.000 +/- 0.000 (range: 1.000-1.000)
- Context Recall: 1.000 +/- 0.000 (range: 1.000-1.000)
