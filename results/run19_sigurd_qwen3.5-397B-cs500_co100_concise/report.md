# RAG Benchmark Report

**Date:** 20260414_093559
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 0.706 | 0.0 | 0.3 | 6842.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs500_co100_nomic... | 0.888 | 0.525 | 0.374 | 0.441 | 0.430 |

## Insights

- Single configuration: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise
-   Processed 100 questions across 2393 chunks
-   TTFT: 0.706s (std 0.595s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.888
-   Answer Relevancy: 0.525
-   Context Precision: 0.441
-   Context Recall: 0.430
-   Total time: 6842.3s

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 0.706s +/- 0.595s (range: 0.275-3.395)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.888 +/- 0.306 (range: 0.000-1.000)
- Answer Relevancy: 0.525 +/- 0.344 (range: 0.000-0.981)
- Answer Correctness: 0.374 +/- 0.272 (range: 0.096-1.000)
- Context Precision: 0.441 +/- 0.399 (range: 0.000-1.000)
- Context Recall: 0.430 +/- 0.498 (range: 0.000-1.000)
