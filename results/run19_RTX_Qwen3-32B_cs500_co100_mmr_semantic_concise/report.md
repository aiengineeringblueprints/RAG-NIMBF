# RAG Benchmark Report

**Date:** 20260415_161733
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs500_co100_nomic-... | 1.213 | 0.0 | 0.2 | 5205.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs500_co100_nomic-... | 0.877 | 0.560 | 0.386 | 0.625 | 0.470 |

## Insights

- Single configuration: semantic_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5
-   Processed 100 questions across 402 chunks
-   TTFT: 1.213s (std 0.426s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.877
-   Answer Relevancy: 0.560
-   Context Precision: 0.625
-   Context Recall: 0.470
-   Total time: 5205.3s

## Detailed Statistics

### semantic_cs500_co100_nomic-...

- TTFT: 1.213s +/- 0.426s (range: 0.430-2.692)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.877 +/- 0.307 (range: 0.000-1.000)
- Answer Relevancy: 0.560 +/- 0.324 (range: 0.000-0.980)
- Answer Correctness: 0.386 +/- 0.252 (range: 0.096-1.000)
- Context Precision: 0.625 +/- 0.439 (range: 0.000-1.000)
- Context Recall: 0.470 +/- 0.502 (range: 0.000-1.000)
