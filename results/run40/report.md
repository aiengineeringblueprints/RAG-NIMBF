# RAG Benchmark Report

**Date:** 20260427_154919
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co200_nomic... | 0.894 | 0.0 | 1.1 | 935.8 |
| recursive_cs500_co200_nomic... | 0.757 | 0.0 | 1.1 | 945.1 |

## RAGAS Scores

| Config | Faithfulness | Semantic Sim. | Ctx Recall | Answer Rel. | Answer Corr. | Ctx Precision |
|--------|-------------|---------------|------------|-------------|--------------|---------------|
| recursive_cs500_co200_nomic... | 0.812 | 0.738 | 0.890 | N/A | N/A | N/A |
| recursive_cs500_co200_nomic... | 0.949 | 0.650 | 0.890 | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs500_co200_nomic... | 0.770 | 0.770 | 0.397 | 0.850 | 0.814 | 0.850 | 0.850 | 0.814 | 0.850 | 0.678 | 0.474 | 0.318 | 0.466 | 0.090 | 0.552 | 0.829 | 0.839 | 0.834 |
| recursive_cs500_co200_nomic... | 0.770 | 0.770 | 0.397 | 0.850 | 0.814 | 0.850 | 0.850 | 0.814 | 0.850 | 0.678 | 0.474 | 0.163 | 0.239 | 0.041 | 0.424 | 0.847 | 0.865 | 0.856 |

## Insights

- Best overall: recursive_cs500_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_rerank-huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2_mmr-l0.5 (composite score: 0.375)
-   Best at: faithfulness, ttft
- Highest faithfulness: recursive_cs500_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_rerank-huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2_mmr-l0.5 (0.949)
- Highest semantic similarity: recursive_cs500_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_rerank-huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2_mmr-l0.5 (0.738)
- Fastest TTFT: recursive_cs500_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_rerank-huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2_mmr-l0.5 (0.76s) — range: 0.76s to 0.89s (1.2x spread)

## Detailed Statistics

### recursive_cs500_co200_nomic...

- TTFT: 0.894s +/- 1.098s (range: 0.273-4.013)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.812 +/- 0.380 (range: 0.000-1.000)
- Context Recall: 0.890 +/- 0.314 (range: 0.000-1.000)
- Semantic Similarity: 0.738 +/- 0.185 (range: 0.335-1.000)

### recursive_cs500_co200_nomic...

- TTFT: 0.757s +/- 0.782s (range: 0.240-4.228)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.949 +/- 0.185 (range: 0.000-1.000)
- Context Recall: 0.890 +/- 0.314 (range: 0.000-1.000)
- Semantic Similarity: 0.650 +/- 0.106 (range: 0.403-0.889)
