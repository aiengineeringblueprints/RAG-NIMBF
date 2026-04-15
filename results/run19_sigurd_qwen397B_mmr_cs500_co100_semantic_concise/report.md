# RAG Benchmark Report

**Date:** 20260415_095845
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs500_co100_nomic-... | 1.158 | 0.0 | 0.1 | 5555.9 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs500_co100_nomic-... | 0.887 | 0.521 | 0.386 | 0.607 | 0.470 |

## Insights

- Single configuration: semantic_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_mmr-l0.5
-   Processed 100 questions across 402 chunks
-   TTFT: 1.158s (std 1.130s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.887
-   Answer Relevancy: 0.521
-   Context Precision: 0.607
-   Context Recall: 0.470
-   Total time: 5555.9s

## Detailed Statistics

### semantic_cs500_co100_nomic-...

- TTFT: 1.158s +/- 1.130s (range: 0.279-7.712)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.887 +/- 0.301 (range: 0.000-1.000)
- Answer Relevancy: 0.521 +/- 0.336 (range: 0.000-0.981)
- Answer Correctness: 0.386 +/- 0.272 (range: 0.096-1.000)
- Context Precision: 0.607 +/- 0.445 (range: 0.000-1.000)
- Context Recall: 0.470 +/- 0.502 (range: 0.000-1.000)
