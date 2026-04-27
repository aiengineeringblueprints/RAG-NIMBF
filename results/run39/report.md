# RAG Benchmark Report

**Date:** 20260427_143806
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 0.463 | 0.0 | 0.4 | 918.6 |
| recursive_cs500_co100_nomic... | 0.648 | 0.0 | 0.2 | 978.8 |

## RAGAS Scores

| Config | Faithfulness | Semantic Sim. | Ctx Recall | Answer Rel. | Answer Corr. | Ctx Precision |
|--------|-------------|---------------|------------|-------------|--------------|---------------|
| recursive_cs500_co100_nomic... | 0.767 | 0.708 | 0.900 | N/A | N/A | N/A |
| recursive_cs500_co100_nomic... | 0.936 | 0.654 | 0.900 | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.368 | 0.860 | 0.805 | 0.860 | 0.860 | 0.805 | 0.860 | 0.672 | 0.474 | 0.312 | 0.412 | 0.077 | 0.509 | 0.789 | 0.800 | 0.794 |
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.368 | 0.860 | 0.805 | 0.860 | 0.860 | 0.805 | 0.860 | 0.672 | 0.474 | 0.164 | 0.232 | 0.037 | 0.413 | 0.837 | 0.855 | 0.846 |

## Insights

- Best overall: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_mmr-l0.5 (composite score: 0.375)
-   Best at: semantic_similarity, context_recall, ttft, tok_per_s
- Highest faithfulness: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (0.936)
- Highest semantic similarity: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_mmr-l0.5 (0.708)
- Fastest TTFT: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_mmr-l0.5 (0.46s) — range: 0.46s to 0.65s (1.4x spread)

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 0.463s +/- 0.148s (range: 0.286-1.537)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.767 +/- 0.416 (range: 0.000-1.000)
- Context Recall: 0.900 +/- 0.302 (range: 0.000-1.000)
- Semantic Similarity: 0.708 +/- 0.185 (range: 0.343-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 0.648s +/- 0.592s (range: 0.293-3.986)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.936 +/- 0.204 (range: 0.000-1.000)
- Context Recall: 0.900 +/- 0.302 (range: 0.000-1.000)
- Semantic Similarity: 0.654 +/- 0.103 (range: 0.414-0.889)
