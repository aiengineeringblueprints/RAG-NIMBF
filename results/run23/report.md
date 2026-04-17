# RAG Benchmark Report

**Date:** 20260417_221022
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 1.021 | 0.0 | 0.2 | 993.5 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.907 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.222 | 0.820 | 0.680 | 0.397 | 0.850 | 0.678 | 0.508 | 0.574 | 0.343 | 0.050 | 0.516 | 0.881 | 0.896 | 0.889 |

## Insights

- Single configuration: recursive_cs1000_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise
-   Processed 100 questions across 113 chunks
-   TTFT: 1.021s (std 0.650s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.907
-   Total time: 993.5s

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 1.021s +/- 0.650s (range: 0.548-3.538)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.907 +/- 0.270 (range: 0.000-1.000)
