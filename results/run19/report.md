# RAG Benchmark Report

**Date:** 20260417_083155
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Dataset:** FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs500_co100_nomic-... | 1.267 | 0.0 | 0.2 | 1350.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs500_co100_nomic-... | 0.871 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| semantic_cs500_co100_nomic-... | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.625 | 0.241 | 0.065 | 0.245 | 0.974 | 0.963 | 0.969 |

## Insights

- Single configuration: semantic_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5
-   Processed 100 questions across 379 chunks
-   TTFT: 1.267s (std 0.411s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.871
-   Total time: 1350.3s

## Detailed Statistics

### semantic_cs500_co100_nomic-...

- TTFT: 1.267s +/- 0.411s (range: 0.562-2.816)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.871 +/- 0.318 (range: 0.000-1.000)
