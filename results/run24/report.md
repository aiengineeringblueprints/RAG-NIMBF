# RAG Benchmark Report

**Date:** 20260417_223046
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co200_nomic... | 0.979 | 0.0 | 0.4 | 970.5 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs500_co200_nomic... | 0.910 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs500_co200_nomic... | 0.720 | 0.720 | 0.202 | 0.850 | 0.723 | 0.413 | 0.890 | 0.725 | 0.553 | 0.605 | 0.320 | 0.053 | 0.484 | 0.851 | 0.866 | 0.859 |

## Insights

- Single configuration: recursive_cs500_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise
-   Processed 100 questions across 238 chunks
-   TTFT: 0.979s (std 0.809s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.910
-   Total time: 970.5s

## Detailed Statistics

### recursive_cs500_co200_nomic...

- TTFT: 0.979s +/- 0.809s (range: 0.408-3.427)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.910 +/- 0.288 (range: 0.000-1.000)
