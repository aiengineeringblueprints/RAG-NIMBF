# RAG Benchmark Report

**Date:** 20260426_201834
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 1.181 | 0.0 | 0.3 | 1125.5 |
| recursive_cs500_co100_nomic... | 1.038 | 0.0 | 0.6 | 1249.8 |

## RAGAS Scores

| Config | Faithfulness | Semantic Sim. | Ctx Recall | Answer Rel. | Answer Corr. | Ctx Precision |
|--------|-------------|---------------|------------|-------------|--------------|---------------|
| recursive_cs500_co100_nomic... | 0.825 | 0.720 | 0.930 | N/A | N/A | N/A |
| recursive_cs500_co100_nomic... | 0.955 | 0.656 | 0.930 | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.197 | 0.860 | 0.718 | 0.409 | 0.900 | 0.728 | 0.553 | 0.600 | 0.474 | 0.298 | 0.442 | 0.084 | 0.536 | 0.838 | 0.849 | 0.843 |
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.197 | 0.860 | 0.718 | 0.409 | 0.900 | 0.728 | 0.553 | 0.600 | 0.474 | 0.166 | 0.235 | 0.039 | 0.421 | 0.856 | 0.875 | 0.865 |

## Insights

- Best overall: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed (composite score: 0.375)
-   Best at: faithfulness, ttft
- Highest faithfulness: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed (0.955)
- Highest semantic similarity: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise (0.720)
- Fastest TTFT: recursive_cs500_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed (1.04s) — range: 1.04s to 1.18s (1.1x spread)

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 1.181s +/- 0.972s (range: 0.612-4.718)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.825 +/- 0.365 (range: 0.000-1.000)
- Context Recall: 0.930 +/- 0.256 (range: 0.000-1.000)
- Semantic Similarity: 0.720 +/- 0.190 (range: 0.334-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 1.038s +/- 0.683s (range: 0.581-4.228)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.955 +/- 0.182 (range: 0.000-1.000)
- Context Recall: 0.930 +/- 0.256 (range: 0.000-1.000)
- Semantic Similarity: 0.656 +/- 0.099 (range: 0.392-0.889)
