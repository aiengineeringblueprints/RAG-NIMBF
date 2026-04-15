# RAG Benchmark Report

**Date:** 20260415_124250
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 0.811 | 0.0 | 0.2 | 4926.1 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs500_co100_nomic... | 0.821 | 0.490 | 0.323 | 0.462 | 0.310 |

## Insights

- Single configuration: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5
-   Processed 100 questions across 2393 chunks
-   TTFT: 0.811s (std 0.275s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.821
-   Answer Relevancy: 0.490
-   Context Precision: 0.462
-   Context Recall: 0.310
-   Total time: 4926.1s

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 0.811s +/- 0.275s (range: 0.433-1.638)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.821 +/- 0.362 (range: 0.000-1.000)
- Answer Relevancy: 0.490 +/- 0.370 (range: 0.000-0.977)
- Answer Correctness: 0.323 +/- 0.244 (range: 0.096-0.998)
- Context Precision: 0.462 +/- 0.451 (range: 0.000-1.000)
- Context Recall: 0.310 +/- 0.465 (range: 0.000-1.000)
