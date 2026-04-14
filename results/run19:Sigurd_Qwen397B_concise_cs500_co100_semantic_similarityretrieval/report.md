# RAG Benchmark Report

**Date:** 20260414_215123
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs500_co100_nomic-... | 2.979 | 0.0 | 0.0 | 12355.7 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs500_co100_nomic-... | 0.921 | 0.559 | 0.420 | 0.599 | 0.590 |

## Insights

- Single configuration: semantic_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise
-   Processed 100 questions across 402 chunks
-   TTFT: 2.979s (std 1.200s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.921
-   Answer Relevancy: 0.559
-   Context Precision: 0.599
-   Context Recall: 0.590
-   Total time: 12355.7s

## Detailed Statistics

### semantic_cs500_co100_nomic-...

- TTFT: 2.979s +/- 1.200s (range: 0.456-5.982)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.921 +/- 0.260 (range: 0.000-1.000)
- Answer Relevancy: 0.559 +/- 0.304 (range: 0.000-0.981)
- Answer Correctness: 0.420 +/- 0.283 (range: 0.096-1.000)
- Context Precision: 0.599 +/- 0.419 (range: 0.000-1.000)
- Context Recall: 0.590 +/- 0.494 (range: 0.000-1.000)
