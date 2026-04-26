# RAG Benchmark Report

**Date:** 20260426_193805
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 1.174 | 0.0 | 0.0 | 1300.5 |

## RAGAS Scores

| Config | Faithfulness | Semantic Sim. | Ctx Recall | Answer Rel. | Answer Corr. | Ctx Precision |
|--------|-------------|---------------|------------|-------------|--------------|---------------|
| recursive_cs1000_co200_nomi... | 0.944 | 0.660 | 0.960 | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.222 | 0.820 | 0.680 | 0.397 | 0.850 | 0.678 | 0.508 | 0.574 | 0.474 | 0.165 | 0.240 | 0.040 | 0.432 | 0.876 | 0.895 | 0.885 |

## Insights

- Single configuration: recursive_cs1000_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed
-   Processed 100 questions across 113 chunks
-   TTFT: 1.174s (std 0.543s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.944
-   Semantic Similarity: 0.660
-   Context Recall: 0.960
-   Total time: 1300.5s

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 1.174s +/- 0.543s (range: 0.819-4.393)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.944 +/- 0.206 (range: 0.000-1.000)
- Context Recall: 0.960 +/- 0.197 (range: 0.000-1.000)
- Semantic Similarity: 0.660 +/- 0.092 (range: 0.414-0.889)
